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

CKPT_DIR = '../ckpt/REINFORCE/FJSP'
DATA_DIR = '../data/FJSP/TNNLS'
RESULT_PATH = '../result/fjsp_results.csv'
BATCH_SIZE = 200
SAMPLE_TIMES = 100
INFERENCE_TYPES = ['greedy', 'sampling']

# (checkpoint_stem, result_label, data_path)
# SD1 flag only affects data generation; the same env handles all loaded datasets.
RUNS = [
    # SD1 in-distribution
    ('Old-SD1-10x05', 'SD1 10x5',   f'{DATA_DIR}/SD1/10x5'),
    ('Old-SD1-20x05', 'SD1 20x5',   f'{DATA_DIR}/SD1/20x5'),
    ('Old-SD1-15x10', 'SD1 15x10',  f'{DATA_DIR}/SD1/15x10'),
    ('Old-SD1-20x10', 'SD1 20x10',  f'{DATA_DIR}/SD1/20x10'),
    # SD2 in-distribution
    ('Old-SD2-10x05', 'SD2 10x5',   f'{DATA_DIR}/SD2/10x5+mix'),
    ('Old-SD2-20x05', 'SD2 20x5',   f'{DATA_DIR}/SD2/20x5+mix'),
    ('Old-SD2-15x10', 'SD2 15x10',  f'{DATA_DIR}/SD2/15x10+mix'),
    ('Old-SD2-20x10', 'SD2 20x10',  f'{DATA_DIR}/SD2/20x10+mix'),
    # OOD: small checkpoint evaluated on larger instances
    ('Old-SD1-10x05', 'OOD SD1 30x10', f'{DATA_DIR}/SD1/30x10'),
    ('Old-SD1-10x05', 'OOD SD1 40x10', f'{DATA_DIR}/SD1/40x10'),
    ('Old-SD2-10x05', 'OOD SD2 30x10', f'{DATA_DIR}/SD2/30x10+mix'),
    ('Old-SD2-10x05', 'OOD SD2 40x10', f'{DATA_DIR}/SD2/40x10+mix'),
    # Open benchmarks with 10x5 model
    ('Old-SD1-10x05', 'Bench Brandimarte (10x5)',  f'{DATA_DIR}/BenchData/Brandimarte'),
    ('Old-SD1-10x05', 'Bench Hurink-edata (10x5)', f'{DATA_DIR}/BenchData/Hurink_edata'),
    ('Old-SD1-10x05', 'Bench Hurink-rdata (10x5)', f'{DATA_DIR}/BenchData/Hurink_rdata'),
    ('Old-SD1-10x05', 'Bench Hurink-vdata (10x5)', f'{DATA_DIR}/BenchData/Hurink_vdata'),
    # Open benchmarks with 20x10 model
    ('Old-SD1-20x10', 'Bench Brandimarte (20x10)',  f'{DATA_DIR}/BenchData/Brandimarte'),
    ('Old-SD1-20x10', 'Bench Hurink-edata (20x10)', f'{DATA_DIR}/BenchData/Hurink_edata'),
    ('Old-SD1-20x10', 'Bench Hurink-rdata (20x10)', f'{DATA_DIR}/BenchData/Hurink_rdata'),
    ('Old-SD1-20x10', 'Bench Hurink-vdata (20x10)', f'{DATA_DIR}/BenchData/Hurink_vdata'),
]


def load_model(ckpt_stem, device):
    model = Model(**model_params).to(device)
    ckpt_path = f'{CKPT_DIR}/{ckpt_stem}.pth'
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


if __name__ == '__main__':
    main()
