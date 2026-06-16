"""
run_op2mac_3way_comparison.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compare three op2mac checkpoints against the main (Old-SD1-10x05) model using the
paper-faithful sampling protocol (100 independent stochastic trajectories, report min
makespan — i.e. inference_type='aug_sample').

Checkpoints evaluated:
  main  : ckpt/REINFORCE/FJSP/Old-SD1-10x05.pth          (use_op2mac_attn=False)
  new   : ckpt/REINFORCE/FJSP/SD1-10x05-op2mac-new.pth   (epoch 2000, val 106.36)
  old   : ckpt/REINFORCE/FJSP/SD1-10x05-op2mac-old.pth   (epoch 1310, val 106.47)
  best  : ckpt/REINFORCE/FJSP/SD1-10x05-op2mac-best.pth  (epoch  130, val 105.63)

Usage:
    cd REINFORCE
    python run_op2mac_3way_comparison.py

Output (under result/):
    op2mac_3way_raw/main.csv  new.csv  old.csv  best.csv
    new_vs_main/new_vs_main.csv
    old_vs_main/old_vs_main.csv
    best_vs_main/best_vs_main.csv
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd

from utils import load_dataset, load_baseline
from SchedulingEnvironment import FJSPEnv
from SchedulingModel import Model
from SchedulingEvaluator import validate_model
from configs.fjsp import model_params

# ---------------------------------------------------------------------------
DATA_DIR   = '../data/FJSP/TNNLS'
CKPT_DIR   = '../ckpt/REINFORCE/FJSP'
RESULT_DIR = '../result'

GREEDY_BATCH = 200   # instances per batch for greedy
AUG_BATCH    = 5     # instances per batch for aug_sample (effective = 5×100 = 500)
SAMPLE_TIMES = 100   # paper: 100 independent trajectories, report min makespan

DATASETS = [
    ('SD1 10x5',                f'{DATA_DIR}/SD1/10x5'),
    ('OOD SD1 30x10',           f'{DATA_DIR}/SD1/30x10'),
    ('OOD SD1 40x10',           f'{DATA_DIR}/SD1/40x10'),
    ('Bench Brandimarte (10x5)',  f'{DATA_DIR}/BenchData/Brandimarte'),
    ('Bench Hurink-edata (10x5)', f'{DATA_DIR}/BenchData/Hurink_edata'),
    ('Bench Hurink-rdata (10x5)', f'{DATA_DIR}/BenchData/Hurink_rdata'),
    ('Bench Hurink-vdata (10x5)', f'{DATA_DIR}/BenchData/Hurink_vdata'),
]

CHECKPOINTS = [
    # (label, ckpt_filename, use_op2mac_attn)
    ('main', 'Old-SD1-10x05.pth',         False),
    ('new',  'SD1-10x05-op2mac-new.pth',  True),
    ('old',  'SD1-10x05-op2mac-old.pth',  True),
    ('best', 'SD1-10x05-op2mac-best.pth', True),
]
# ---------------------------------------------------------------------------


def load_model(ckpt_file, use_op2mac, device):
    mp = {**model_params, 'use_op2mac_attn': use_op2mac}
    model = Model(**mp).to(device)
    path = os.path.join(CKPT_DIR, ckpt_file)
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(label, ckpt_file, use_op2mac, env, device):
    print(f'\n[{label}]  {ckpt_file}')
    model = load_model(ckpt_file, use_op2mac, device)
    rows = []
    for ds_label, data_path in DATASETS:
        dataset  = load_dataset(data_path)
        baseline = load_baseline(data_path)

        # greedy
        score, gap = validate_model(
            env, model, dataset, GREEDY_BATCH,
            inference_type='greedy', baseline=baseline)
        rows.append({'checkpoint': label, 'dataset': ds_label,
                     'inference_type': 'greedy',
                     'avg_makespan': round(score, 4),
                     'gap(%)': round(gap, 4) if gap is not None else None})
        print(f'  {ds_label:35s} | greedy       -> makespan={score:.4f}'
              + (f'  gap={gap:.4f}%' if gap is not None else ''))

        # paper sampling: 100 trajectories, report min (aug_sample)
        score, gap = validate_model(
            env, model, dataset, AUG_BATCH,
            inference_type='aug_sample', sampling_times=SAMPLE_TIMES,
            baseline=baseline)
        rows.append({'checkpoint': label, 'dataset': ds_label,
                     'inference_type': 'sampling_x100',
                     'avg_makespan': round(score, 4),
                     'gap(%)': round(gap, 4) if gap is not None else None})
        print(f'  {ds_label:35s} | sampling_x100-> makespan={score:.4f}'
              + (f'  gap={gap:.4f}%' if gap is not None else ''))

    return pd.DataFrame(rows)


def save_comparison(df_variant, label, df_main):
    out_dir = os.path.join(RESULT_DIR, f'{label}_vs_main')
    os.makedirs(out_dir, exist_ok=True)

    main_cols = df_main[['dataset', 'inference_type', 'gap(%)']].rename(
        columns={'gap(%)': 'gap_main(%)'})
    var_cols  = df_variant[['dataset', 'inference_type', 'gap(%)']].rename(
        columns={'gap(%)': f'gap_{label}(%)'})

    merged = var_cols.merge(main_cols, on=['dataset', 'inference_type'], how='left')
    merged['delta(%)'] = (merged[f'gap_{label}(%)'] - merged['gap_main(%)']).round(4)
    merged['winner']   = merged['delta(%)'].apply(
        lambda d: label if d < 0 else 'main')
    merged = merged[['dataset', 'inference_type', 'gap_main(%)',
                      f'gap_{label}(%)', 'delta(%)', 'winner']]

    out_csv = os.path.join(out_dir, f'{label}_vs_main.csv')
    merged.to_csv(out_csv, index=False)
    print(f'\nSaved: {os.path.abspath(out_csv)}')
    print(merged.to_string(index=False))
    wins = (merged['winner'] == label).sum()
    print(f'{label} wins: {wins}/{len(merged)}   '
          f'mean gap -> main={merged["gap_main(%)"].mean():.3f}%  '
          f'{label}={merged[f"gap_{label}(%)"].mean():.3f}%')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    raw_dir = os.path.join(RESULT_DIR, 'op2mac_3way_raw')
    os.makedirs(raw_dir, exist_ok=True)

    env = FJSPEnv(None, sd1=True)
    results = {}

    for label, ckpt_file, use_op2mac in CHECKPOINTS:
        df = evaluate_model(label, ckpt_file, use_op2mac, env, device)
        results[label] = df
        raw_csv = os.path.join(raw_dir, f'{label}.csv')
        df.to_csv(raw_csv, index=False)
        print(f'Raw saved: {os.path.abspath(raw_csv)}')

    print('\n' + '='*70)
    print('COMPARISONS vs MAIN')
    print('='*70)
    for label in ('new', 'old', 'best'):
        save_comparison(results[label], label, results['main'])

    print('\nDone.')


if __name__ == '__main__':
    main()
