"""
run_op2mac_v2_comparison.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluate the new op2mac-v2 checkpoint (June-23 run, epoch 220, val 104.27)
against the main model using the paper-faithful sampling protocol:
  - greedy decoding
  - sampling_x100: 100 independent stochastic trajectories, report min makespan
    (inference_type='aug_sample', sampling_times=100)

The four previously-evaluated models (main / old / new / best) are NOT re-run;
their gaps are embedded as constants and used to produce the combined 5-model
summary sheet.

Checkpoint evaluated:
  v2 : ckpt/REINFORCE/FJSP/SD1-10x05-op2mac-v2.pth  (epoch 220, val 104.27)
       use_op2mac_attn=True

Usage:
    cd REINFORCE
    python run_op2mac_v2_comparison.py

Output (under result/):
    op2mac_v2_raw/v2.csv
    v2_vs_main/v2_vs_main.csv
    summary_all5_vs_main.csv
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
AUG_BATCH    = 1     # instances per batch for aug_sample  (effective = 1×100 = 100)
SAMPLE_TIMES = 100   # paper: 100 independent trajectories, report min makespan

V2_CKPT = 'SD1-10x05-op2mac-v2.pth'   # epoch 220, val 104.27

DATASETS = [
    ('SD1 10x5',                f'{DATA_DIR}/SD1/10x5'),
    ('OOD SD1 30x10',           f'{DATA_DIR}/SD1/30x10'),
    ('OOD SD1 40x10',           f'{DATA_DIR}/SD1/40x10'),
    ('Bench Brandimarte (10x5)',  f'{DATA_DIR}/BenchData/Brandimarte'),
    ('Bench Hurink-edata (10x5)', f'{DATA_DIR}/BenchData/Hurink_edata'),
    ('Bench Hurink-rdata (10x5)', f'{DATA_DIR}/BenchData/Hurink_rdata'),
    ('Bench Hurink-vdata (10x5)', f'{DATA_DIR}/BenchData/Hurink_vdata'),
]

# Reference gaps from the 3-way comparison (result/op2mac_3way_raw/*.csv)
# Keys: (dataset_label, inference_type)  Values: gap(%)
REFERENCE_GAPS = {
    # main  (Old-SD1-10x05.pth, use_op2mac_attn=False)
    ('SD1 10x5',                'greedy'):        12.1536,
    ('SD1 10x5',                'sampling_x100'):  6.1659,
    ('OOD SD1 30x10',           'greedy'):         3.6129,
    ('OOD SD1 30x10',           'sampling_x100'):  3.6094,
    ('OOD SD1 40x10',           'greedy'):         2.6481,
    ('OOD SD1 40x10',           'sampling_x100'):  3.6338,
    ('Bench Brandimarte (10x5)', 'greedy'):         9.5928,
    ('Bench Brandimarte (10x5)', 'sampling_x100'):  6.0937,
    ('Bench Hurink-edata (10x5)','greedy'):        15.7422,
    ('Bench Hurink-edata (10x5)','sampling_x100'):  8.5028,
    ('Bench Hurink-rdata (10x5)','greedy'):        10.6432,
    ('Bench Hurink-rdata (10x5)','sampling_x100'):  4.9577,
    ('Bench Hurink-vdata (10x5)','greedy'):         3.2919,
    ('Bench Hurink-vdata (10x5)','sampling_x100'):  0.9896,
}

# Gaps for old/new/best — used only for the 5-model summary sheet
REF_OLD = {
    ('SD1 10x5',                'greedy'):        14.616,
    ('SD1 10x5',                'sampling_x100'):  6.5841,
    ('OOD SD1 30x10',           'greedy'):         5.9847,
    ('OOD SD1 30x10',           'sampling_x100'):  3.8344,
    ('OOD SD1 40x10',           'greedy'):         4.504,
    ('OOD SD1 40x10',           'sampling_x100'):  3.3569,
    ('Bench Brandimarte (10x5)', 'greedy'):        12.9617,
    ('Bench Brandimarte (10x5)', 'sampling_x100'):  9.2957,
    ('Bench Hurink-edata (10x5)','greedy'):        14.0185,
    ('Bench Hurink-edata (10x5)','sampling_x100'):  9.3557,
    ('Bench Hurink-rdata (10x5)','greedy'):         9.6865,
    ('Bench Hurink-rdata (10x5)','sampling_x100'):  4.6082,
    ('Bench Hurink-vdata (10x5)','greedy'):         4.6866,
    ('Bench Hurink-vdata (10x5)','sampling_x100'):  1.6766,
}

REF_NEW = {
    ('SD1 10x5',                'greedy'):        15.1449,
    ('SD1 10x5',                'sampling_x100'):  6.7666,
    ('OOD SD1 30x10',           'greedy'):         5.6465,
    ('OOD SD1 30x10',           'sampling_x100'):  3.5424,
    ('OOD SD1 40x10',           'greedy'):         4.297,
    ('OOD SD1 40x10',           'sampling_x100'):  2.8514,
    ('Bench Brandimarte (10x5)', 'greedy'):        14.7186,
    ('Bench Brandimarte (10x5)', 'sampling_x100'):  9.4578,
    ('Bench Hurink-edata (10x5)','greedy'):        15.1355,
    ('Bench Hurink-edata (10x5)','sampling_x100'):  9.774,
    ('Bench Hurink-rdata (10x5)','greedy'):        10.379,
    ('Bench Hurink-rdata (10x5)','sampling_x100'):  5.31,
    ('Bench Hurink-vdata (10x5)','greedy'):         5.8567,
    ('Bench Hurink-vdata (10x5)','sampling_x100'):  1.9985,
}

REF_BEST = {
    ('SD1 10x5',                'greedy'):        13.14,
    ('SD1 10x5',                'sampling_x100'):  7.4614,
    ('OOD SD1 30x10',           'greedy'):         7.6398,
    ('OOD SD1 30x10',           'sampling_x100'):  8.4075,
    ('OOD SD1 40x10',           'greedy'):         6.8935,
    ('OOD SD1 40x10',           'sampling_x100'):  8.8924,
    ('Bench Brandimarte (10x5)', 'greedy'):        11.0345,
    ('Bench Brandimarte (10x5)', 'sampling_x100'):  9.5167,
    ('Bench Hurink-edata (10x5)','greedy'):        13.7272,
    ('Bench Hurink-edata (10x5)','sampling_x100'): 10.4619,
    ('Bench Hurink-rdata (10x5)','greedy'):         9.4206,
    ('Bench Hurink-rdata (10x5)','sampling_x100'):  5.8896,
    ('Bench Hurink-vdata (10x5)','greedy'):         2.7395,
    ('Bench Hurink-vdata (10x5)','sampling_x100'):  0.9449,
}
# ---------------------------------------------------------------------------


def load_v2_model(device):
    mp = {**model_params, 'use_op2mac_attn': True}
    model = Model(**mp).to(device)
    path = os.path.join(CKPT_DIR, V2_CKPT)
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded v2 checkpoint: {os.path.abspath(path)}')
    return model


def evaluate_v2(env, device):
    model = load_v2_model(device)
    rows = []
    for ds_label, data_path in DATASETS:
        dataset  = load_dataset(data_path)
        baseline = load_baseline(data_path)

        # greedy
        score, gap = validate_model(
            env, model, dataset, GREEDY_BATCH,
            inference_type='greedy', baseline=baseline)
        rows.append({'checkpoint': 'v2', 'dataset': ds_label,
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
        rows.append({'checkpoint': 'v2', 'dataset': ds_label,
                     'inference_type': 'sampling_x100',
                     'avg_makespan': round(score, 4),
                     'gap(%)': round(gap, 4) if gap is not None else None})
        print(f'  {ds_label:35s} | sampling_x100-> makespan={score:.4f}'
              + (f'  gap={gap:.4f}%' if gap is not None else ''))

    return pd.DataFrame(rows)


def save_v2_vs_main(df_v2):
    out_dir = os.path.join(RESULT_DIR, 'v2_vs_main')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for _, row in df_v2.iterrows():
        key = (row['dataset'], row['inference_type'])
        gap_main = REFERENCE_GAPS.get(key)
        gap_v2   = row['gap(%)']
        delta    = round(gap_v2 - gap_main, 4) if gap_main is not None else None
        winner   = 'v2' if (delta is not None and delta < 0) else 'main'
        rows.append({
            'dataset':         row['dataset'],
            'inference_type':  row['inference_type'],
            'gap_main(%)':     gap_main,
            'gap_v2(%)':       gap_v2,
            'delta(%)':        delta,
            'winner':          winner,
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, 'v2_vs_main.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nSaved: {os.path.abspath(out_csv)}')
    print(df.to_string(index=False))
    wins = (df['winner'] == 'v2').sum()
    print(f'v2 wins: {wins}/{len(df)}   '
          f'mean gap -> main={df["gap_main(%)"].mean():.3f}%  '
          f'v2={df["gap_v2(%)"].mean():.3f}%')
    return df


def save_summary_all5(df_v2):
    """Build a combined 5-model summary sheet and save to result/summary_all5_vs_main.csv."""
    rows = []
    order = [
        ('SD1 10x5',                 'greedy'),
        ('SD1 10x5',                 'sampling_x100'),
        ('OOD SD1 30x10',            'greedy'),
        ('OOD SD1 30x10',            'sampling_x100'),
        ('OOD SD1 40x10',            'greedy'),
        ('OOD SD1 40x10',            'sampling_x100'),
        ('Bench Brandimarte (10x5)', 'greedy'),
        ('Bench Brandimarte (10x5)', 'sampling_x100'),
        ('Bench Hurink-edata (10x5)','greedy'),
        ('Bench Hurink-edata (10x5)','sampling_x100'),
        ('Bench Hurink-rdata (10x5)','greedy'),
        ('Bench Hurink-rdata (10x5)','sampling_x100'),
        ('Bench Hurink-vdata (10x5)','greedy'),
        ('Bench Hurink-vdata (10x5)','sampling_x100'),
    ]

    # build a lookup from the freshly computed v2 DataFrame
    v2_lookup = {
        (r['dataset'], r['inference_type']): r['gap(%)']
        for _, r in df_v2.iterrows()
    }

    for ds, it in order:
        key = (ds, it)
        rows.append({
            'dataset':              ds,
            'inference_type':       it,
            'gap_main(%)':          REFERENCE_GAPS.get(key),
            'gap_op2mac_old(%)':    REF_OLD.get(key),
            'gap_op2mac_new(%)':    REF_NEW.get(key),
            'gap_op2mac_best(%)':   REF_BEST.get(key),
            'gap_op2mac_v2(%)':     v2_lookup.get(key),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULT_DIR, 'summary_all5_vs_main.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nSaved: {os.path.abspath(out_csv)}')
    print(df.to_string(index=False))
    return df


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Evaluating op2mac-v2 (epoch 220, val 104.27) on {len(DATASETS)} datasets ...\n')

    env = FJSPEnv(None, sd1=True)

    # --- evaluate v2 ---
    df_v2 = evaluate_v2(env, device)

    # --- save raw ---
    raw_dir = os.path.join(RESULT_DIR, 'op2mac_v2_raw')
    os.makedirs(raw_dir, exist_ok=True)
    raw_csv = os.path.join(raw_dir, 'v2.csv')
    df_v2.to_csv(raw_csv, index=False)
    print(f'\nRaw saved: {os.path.abspath(raw_csv)}')

    # --- v2 vs main ---
    print('\n' + '='*70)
    print('v2 vs main')
    print('='*70)
    save_v2_vs_main(df_v2)

    # --- 5-model summary ---
    print('\n' + '='*70)
    print('5-MODEL SUMMARY (main / old / new / best / v2)')
    print('='*70)
    save_summary_all5(df_v2)

    print('\nDone.')


if __name__ == '__main__':
    main()
