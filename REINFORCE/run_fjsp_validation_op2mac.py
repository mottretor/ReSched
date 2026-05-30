"""
run_fjsp_validation_op2mac.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluate the new op2mac checkpoint (use_op2mac_attn=True) and write results
to result/fjsp_results_op2mac.csv.

After training finishes on Sydney, copy the best model:
    cp result/FJSP/<timestamp>_SD1-10x05-RL_<id>/best_model.pth \
       ckpt/REINFORCE/FJSP/SD1-10x05-op2mac.pth

Then run this script:
    cd REINFORCE
    python run_fjsp_validation_op2mac.py

Compare against the baseline:
    python compare_results.py          # see bottom of this file for the snippet
"""

import os
import sys

# Ensure working directory is REINFORCE/ so all relative paths in utils/configs work
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd

from utils import load_dataset, load_baseline
from SchedulingEnvironment import FJSPEnv
from SchedulingModel import Model
from SchedulingEvaluator import validate_model
from configs.fjsp import model_params

# ---- Checkpoint to evaluate ------------------------------------------------
CKPT_DIR = '../ckpt/REINFORCE/FJSP'
CKPT_STEM = 'SD1-10x05-op2mac'      # copy best_model.pth here after training
# ----------------------------------------------------------------------------

DATA_DIR = '../data/FJSP/TNNLS'
RESULT_PATH = '../result/fjsp_results_op2mac.csv'
BASELINE_PATH = '../result/fjsp_results.csv'   # existing reference

BATCH_SIZE = 200
SAMPLE_TIMES = 100
INFERENCE_TYPES = ['greedy', 'sampling']

# Mirror the 10x5 rows from run_fjsp_validation.py for apples-to-apples comparison.
RUNS = [
    # In-distribution
    (CKPT_STEM, 'SD1 10x5',   f'{DATA_DIR}/SD1/10x5'),
    # OOD generalisation (small model on larger instances)
    (CKPT_STEM, 'OOD SD1 30x10', f'{DATA_DIR}/SD1/30x10'),
    (CKPT_STEM, 'OOD SD1 40x10', f'{DATA_DIR}/SD1/40x10'),
    # Open benchmarks
    (CKPT_STEM, 'Bench Brandimarte (10x5)',  f'{DATA_DIR}/BenchData/Brandimarte'),
    (CKPT_STEM, 'Bench Hurink-edata (10x5)', f'{DATA_DIR}/BenchData/Hurink_edata'),
    (CKPT_STEM, 'Bench Hurink-rdata (10x5)', f'{DATA_DIR}/BenchData/Hurink_rdata'),
    (CKPT_STEM, 'Bench Hurink-vdata (10x5)', f'{DATA_DIR}/BenchData/Hurink_vdata'),
]


def load_model(ckpt_stem, device):
    # Build model with the new attention flag enabled.
    mp = {**model_params, 'use_op2mac_attn': True}
    model = Model(**mp).to(device)
    ckpt_path = f'{CKPT_DIR}/{ckpt_stem}.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}\n'
            'Copy your trained best_model.pth there first:\n'
            f'  cp result/FJSP/<run>/best_model.pth {ckpt_path}'
        )
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    env = FJSPEnv(None, sd1=True)
    results = []
    loaded_ckpt = None
    model = None

    for ckpt_stem, label, data_path in RUNS:
        if ckpt_stem != loaded_ckpt:
            model = load_model(ckpt_stem, device)
            loaded_ckpt = ckpt_stem
            print(f'\n[{ckpt_stem}]')

        dataset = load_dataset(data_path)
        baseline = load_baseline(data_path)

        for inf_type in INFERENCE_TYPES:
            score, gap = validate_model(
                env, model, dataset, BATCH_SIZE,
                inference_type=inf_type,
                sampling_times=SAMPLE_TIMES,
                baseline=baseline,
            )
            results.append({
                'checkpoint': ckpt_stem,
                'dataset': label,
                'inference_type': inf_type,
                'avg_makespan': round(score, 4),
                'gap(%)': round(gap, 4) if gap is not None else None,
            })
            gap_str = f'{gap:.4f}%' if gap is not None else 'N/A'
            print(f'  {label} | {inf_type:8s} -> makespan={score:.4f}, gap={gap_str}')

    df = pd.DataFrame(results)
    df.to_csv(RESULT_PATH, index=False)
    print(f'\nSaved: {os.path.abspath(RESULT_PATH)}')

    # --- Quick side-by-side comparison against the existing baseline CSV ----
    if os.path.exists(BASELINE_PATH):
        print('\n=== Gap comparison (op2mac vs Old-SD1-10x05 baseline) ===')
        base = pd.read_csv(BASELINE_PATH)
        # Keep only the Old-SD1-10x05 rows that match our datasets
        base = base[base['checkpoint'] == 'Old-SD1-10x05'][['dataset', 'inference_type', 'gap(%)']]
        base = base.rename(columns={'gap(%)': 'gap_baseline(%)'})

        merged = df[['dataset', 'inference_type', 'gap(%)']].rename(
            columns={'gap(%)': 'gap_op2mac(%)'})
        merged = merged.merge(base, on=['dataset', 'inference_type'], how='left')
        merged['delta(%)'] = merged['gap_op2mac(%)'] - merged['gap_baseline(%)']
        print(merged.to_string(index=False))
        print('\nNegative delta = op2mac model wins (lower optimality gap).')
    else:
        print(f'\nBaseline CSV not found at {BASELINE_PATH}; skipping comparison.')


if __name__ == '__main__':
    main()
