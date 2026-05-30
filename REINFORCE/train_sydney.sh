#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_sydney.sh — launch a REINFORCE training run on the Sydney GPU server.
#
# Usage (inside a tmux session so the process survives SSH disconnects):
#
#   ssh sydney
#   cd <repo>/REINFORCE
#   tmux new -s op2mac          # or: tmux attach -t op2mac
#   bash train_sydney.sh
#   # Ctrl-b d  to detach from the session
#   # tmux attach -t op2mac     to reconnect later
#
# Prerequisites:
#   - conda (or mamba) available on PATH
#   - environment created at ../venv (python=3.11, torch==2.3.1+cu121, ortools, numpy, pandas)
#     If the env is at a different path, edit CONDA_ENV below.
#   - Verify the correct GPU is visible: nvidia-smi
#     If needed, edit cuda_device_num in configs/fjsp.py (default 0).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Edit this if the conda env lives elsewhere on Sydney ----------------
# On Sydney the layout is: /share/pasindu/projects/{venv,ReSched/}
# SCRIPT_DIR = .../ReSched/REINFORCE  =>  ../../venv = /share/pasindu/projects/venv
CONDA_ENV="$SCRIPT_DIR/../../venv"
# --------------------------------------------------------------------------

# Activate environment
# Works with both `conda activate <path>` and `source activate <path>`
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "ERROR: conda not found on PATH. Activate the environment manually." >&2
    exit 1
fi

# Verify GPU visibility
echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(nvidia-smi not available)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())"

mkdir -p "$SCRIPT_DIR/../result/logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$SCRIPT_DIR/../result/logs/train_op2mac_${TS}.log"

echo ""
echo "=== Starting training ==="
echo "Config: REINFORCE/configs/fjsp.py (use_op2mac_attn=True, test_only=False)"
echo "Log:    $LOG"
echo ""

# -u: unbuffered stdout/stderr so tail -f shows live progress
nohup python -u SchedulingMain.py > "$LOG" 2>&1 &
PID=$!

echo "PID $PID started."
echo "Monitor with:  tail -f $LOG"
echo "Stop with:     kill $PID"
echo ""
echo "When training finishes, copy best_model.pth to ckpt/REINFORCE/FJSP/SD1-10x05-op2mac.pth"
echo "then run:  python run_fjsp_validation_op2mac.py"
