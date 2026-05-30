"""
run_experiment.py
~~~~~~~~~~~~~~~~~
End-to-end controlled A/B experiment: trains BOTH variants from scratch
with the paper-exact SD1 10×5 setup, then evaluates both and prints a
side-by-side gap comparison.

Variants
--------
  base   — paper-exact ReSched architecture (use_op2mac_attn=False)
  op2mac — adds operation←machine edge-aware cross-attention (use_op2mac_attn=True)

Paper-exact configuration (Appendix B.2, Xiao et al. ICLR 2026)
-----------------------------------------------------------------
  Adam lr=5e-5, weight_decay=1e-6
  MultiStepLR milestones=[500,1000,1500], gamma=0.5
  2000 epochs × 1000 instances/epoch × batch_size=50
  discount_factor=0.99
  Feature extractor: 2 Transformer blocks, 8 heads, dim=128, ff_dim=512
  Decision MLP: 3 layers, hidden=64

Fidelity check: the 'base' greedy gap on SD1 10×5 should reproduce
~12.25% (ReSched row, Table 1 / Table 5 of the paper).

Launch on Sydney (inside a tmux session)
-----------------------------------------
  cd /share/pasindu/projects/ReSched/REINFORCE
  nohup python -u run_experiment.py \\
      > ../result/logs/experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  tail -f ../result/logs/experiment_*.log
"""

import os
import sys
import copy
import shutil
import logging

# Make all relative imports work regardless of where the script is called from
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd

from configs.fjsp import (
    env_params,
    model_params,
    optimizer_params,
    runner_params,
    logger_params,
)
from SchedulingRunner import Runner
from SchedulingEnvironment import FJSPEnv
from SchedulingModel import Model
from SchedulingEvaluator import validate_model
from utils import load_dataset, load_baseline

# ---------------------------------------------------------------------------
# Experiment controls  (edit these; leave the rest of the file unchanged)
# ---------------------------------------------------------------------------

TRAIN = True          # False → skip training, evaluate existing checkpoints only

VARIANTS = [
    ('base',   False),   # (variant_name, use_op2mac_attn)
    ('op2mac', True),
]

# None → use the paper-exact 2000 epochs from configs/fjsp.py.
# Set e.g. 400 ONLY for a one-off pipeline smoke-test; not valid for reporting.
EPOCHS = None

CKPT_DIR = '../ckpt/REINFORCE/FJSP'
DATA_DIR = '../data/FJSP/TNNLS'

# Datasets evaluated for both variants after training.
# Mirrors the rows from run_fjsp_validation.py for the 10×5 model.
EVAL_DATASETS = [
    ('SD1 10x5',             f'{DATA_DIR}/SD1/10x5'),
    ('OOD SD1 30x10',        f'{DATA_DIR}/SD1/30x10'),
    ('OOD SD1 40x10',        f'{DATA_DIR}/SD1/40x10'),
    ('Bench Brandimarte',    f'{DATA_DIR}/BenchData/Brandimarte'),
    ('Bench Hurink-edata',   f'{DATA_DIR}/BenchData/Hurink_edata'),
    ('Bench Hurink-rdata',   f'{DATA_DIR}/BenchData/Hurink_rdata'),
    ('Bench Hurink-vdata',   f'{DATA_DIR}/BenchData/Hurink_vdata'),
]

INFERENCE_TYPES = ['greedy', 'sampling']
BATCH_SIZE   = 200
SAMPLE_TIMES = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_runner_logger():
    """Remove all handlers from the 'runner' logger.

    Runner registers file/console handlers on every __init__.  Without
    clearing them between variants the second variant's log lines appear
    in both variant log files.
    """
    logger = logging.getLogger('runner')
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)


def train_variant(name: str, use_op2mac_attn: bool) -> str:
    """Train one variant from scratch and return its save directory path."""
    print(f'\n{"=" * 70}')
    print(f'  TRAINING  variant={name!r}  use_op2mac_attn={use_op2mac_attn}')
    print(f'{"=" * 70}\n')

    # Build per-variant params — only the flag and run_name differ.
    mp = {**model_params, 'use_op2mac_attn': use_op2mac_attn}

    rp = copy.deepcopy(runner_params)
    rp['test_only']  = False   # train mode
    rp['checkpoint'] = None    # always start from scratch
    rp['model_path'] = None    # no warm-start
    if EPOCHS is not None:
        rp['training']['epochs'] = EPOCHS

    lp = {**logger_params, 'run_name': f'SD1-10x05-{name}'}

    _clear_runner_logger()
    runner = Runner(
        logger_params=lp,
        env_params=env_params,
        model_params=mp,
        optimizer_params=optimizer_params,
        runner_params=rp,
    )
    runner.run()          # train + built-in quick test at the end
    save_dir = runner.save_dir

    # Free GPU memory before the next variant
    del runner
    torch.cuda.empty_cache()
    _clear_runner_logger()

    return save_dir


def publish_checkpoint(save_dir: str, name: str) -> str:
    """Copy best_model.pth to the canonical ckpt location."""
    src = os.path.join(save_dir, 'best_model.pth')
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'best_model.pth not found in {save_dir}.\n'
            'Training may not have completed a validation step yet.'
        )
    os.makedirs(CKPT_DIR, exist_ok=True)
    dst = os.path.join(CKPT_DIR, f'SD1-10x05-{name}.pth')
    shutil.copy(src, dst)
    print(f'  Checkpoint → {dst}')
    return dst


def evaluate_variant(name: str, use_op2mac_attn: bool, device: torch.device) -> list:
    """Load checkpoint, run validate_model on all eval datasets, return rows."""
    ckpt_path = os.path.join(CKPT_DIR, f'SD1-10x05-{name}.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}\n'
            'Run with TRAIN=True first, or copy the weights manually:\n'
            f'  cp result/FJSP/<run>/best_model.pth {ckpt_path}'
        )

    mp = {**model_params, 'use_op2mac_attn': use_op2mac_attn}
    model = Model(**mp).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # sd1=True is required for the SD1 generator; for eval we load data
    # from disk so it only affects the env's internal generator (unused).
    env = FJSPEnv(None, sd1=True)

    rows = []
    print(f'\n[Evaluating variant: {name}]')
    for label, data_path in EVAL_DATASETS:
        dataset  = load_dataset(data_path)
        baseline = load_baseline(data_path)
        for inf_type in INFERENCE_TYPES:
            score, gap = validate_model(
                env, model, dataset, BATCH_SIZE,
                inference_type=inf_type,
                sampling_times=SAMPLE_TIMES,
                baseline=baseline,
            )
            rows.append({
                'variant':        name,
                'dataset':        label,
                'inference_type': inf_type,
                'avg_makespan':   round(score, 4),
                'gap(%)':         round(gap, 4) if gap is not None else None,
            })
            gap_str = f'{gap:.4f}%' if gap is not None else 'N/A'
            print(f'  {label:30s} | {inf_type:8s} → makespan={score:.4f}  gap={gap_str}')

    del model
    torch.cuda.empty_cache()
    return rows


def print_comparison(df: pd.DataFrame):
    """Pivot base vs op2mac and print a gap comparison table."""
    gap_df = df[df['gap(%)'].notna()].copy()

    base   = (gap_df[gap_df['variant'] == 'base']
              [['dataset', 'inference_type', 'gap(%)']]
              .rename(columns={'gap(%)': 'gap_base(%)'}))
    op2mac = (gap_df[gap_df['variant'] == 'op2mac']
              [['dataset', 'inference_type', 'gap(%)']]
              .rename(columns={'gap(%)': 'gap_op2mac(%)'}))

    merged = base.merge(op2mac, on=['dataset', 'inference_type'], how='outer')
    merged['delta(%)'] = merged['gap_op2mac(%)'] - merged['gap_base(%)']

    print('\n' + '=' * 80)
    print('  GAP COMPARISON   (negative delta = op2mac WINS = lower optimality gap)')
    print('=' * 80)
    print(merged.to_string(index=False))

    valid = merged['delta(%)'].dropna()
    n_better = (valid < 0).sum()
    avg_delta = valid.mean()
    print(f'\n  op2mac better in {n_better}/{len(valid)} dataset×strategy settings')
    print(f'  Average delta: {avg_delta:+.4f}%  (negative = op2mac better overall)')

    # Also merge in the shipped Old-SD1-10x05 numbers if available.
    ref_csv = '../result/fjsp_results.csv'
    if os.path.exists(ref_csv):
        ref = pd.read_csv(ref_csv)
        ref = (ref[ref['checkpoint'] == 'Old-SD1-10x05']
               [['dataset', 'inference_type', 'gap(%)']]
               .rename(columns={'gap(%)': 'gap_shipped(%)'}))
        # Align dataset labels (the ref CSV uses slightly different names)
        merged_ref = merged.merge(ref, on=['dataset', 'inference_type'], how='left')
        has_ref = merged_ref['gap_shipped(%)'].notna().any()
        if has_ref:
            print('\n  (Reference column gap_shipped(%) = Old-SD1-10x05 from fjsp_results.csv)')
            print(merged_ref.to_string(index=False))


def main():
    os.makedirs('../result/logs', exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1 — Training
    # ------------------------------------------------------------------
    if TRAIN:
        for name, use_attn in VARIANTS:
            save_dir = train_variant(name, use_attn)
            publish_checkpoint(save_dir, name)
    else:
        print('TRAIN=False — skipping training, using existing checkpoints.\n')

    # ------------------------------------------------------------------
    # Phase 2 — Evaluation
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nEvaluation device: {device}')

    all_rows = []
    for name, use_attn in VARIANTS:
        all_rows.extend(evaluate_variant(name, use_attn, device))

    df = pd.DataFrame(all_rows)

    result_path = '../result/fjsp_experiment_results.csv'
    df.to_csv(result_path, index=False)
    print(f'\nFull results saved → {os.path.abspath(result_path)}')

    print_comparison(df)


if __name__ == '__main__':
    main()
