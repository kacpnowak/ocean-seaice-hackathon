#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train `base` from scratch on ONE JURECA node (4 x A100-40GB) in under 12 h.
#
#   sbatch scripts/pretrain_base.slurm
#   sbatch --export=ALL,NAME=base_v2,MAX_STEPS=72000 scripts/pretrain_base.slurm
#
# This is the kit's reference pre-training run.  `base` at 84.6M parameters is
# the largest preset that fits a dc-gpu A100, at batch_size 1 per GPU
# (21.33 GiB peak).
#
# WHY 84000 STEPS.  Measured on one dc-gpu node, real dataloader, real metrics,
# four DDP ranks: 2.18 it/s, i.e. 0.459 s per optimiser step.  With ~35
# validation passes and 14 checkpoints that is 10.9 h, which leaves an hour of
# margin inside a 12 h allocation.  `base`'s own preset budget is 110000 steps
# and that measures 14.2 h, so it does NOT fit one window -- hence the override
# here rather than the preset's default.  The cosine schedule follows
# `num_training_steps: ${max_steps}`, so this is a complete, properly annealed
# 84000-step run and not a run that was cut off.
#
# `max_steps` counts OPTIMISER steps and more GPUs do not divide it; what four
# ranks buy is 4 samples per step instead of 1 (see docs/04 4.3).
#
# ONE TASK, FOUR VISIBLE GPUS -- deliberately not `--ntasks-per-node=4`.
# Lightning spawns its own four DDP workers from a single process, and that is
# the configuration the 2.18 it/s above was measured in, so the budget and the
# launch mode match.  Under `--ntasks-per-node=4` Lightning would instead attach
# to SLURM's four tasks; that also works, and the timing would have to be
# re-measured before it could be trusted against this wall clock.
# ---------------------------------------------------------------------------
#SBATCH --account=training2635
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --job-name=pretrain_base
#SBATCH --output=logs/pretrain_base-%j.out

set -euo pipefail

# SLURM copies the batch script into its spool directory, so BASH_SOURCE points
# there and not at the repository.  SLURM_SUBMIT_DIR is where `sbatch` ran.
# Getting this wrong is not subtle: the first run of this script derived the root
# from BASH_SOURCE, landed in SLURM's spool, and died in 19 s on
# `mkdir: cannot create directory 'logs': Permission denied`.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source config.env

NAME="${NAME:-base_pretrained}"
MODULE="${MODULE:-base}"
DATALOADER="${DATALOADER:-glorys}"
MAX_STEPS="${MAX_STEPS:-84000}"
# Must divide MAX_STEPS, or the finished model is the one thing that never
# reaches disk.  A startup guard refuses the run if it does not.
SAVE_EVERY="${SAVE_EVERY:-6000}"

echo "run         : ${NAME}"
echo "preset      : module=${MODULE} dataloader=${DATALOADER}"
echo "budget      : ${MAX_STEPS} steps, checkpoint every ${SAVE_EVERY}"
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader || echo unknown)"
echo "node        : $(hostname), ${gpu_names%%$'\n'*} x$(printf '%s\n' "$gpu_names" | grep -c .)"
echo "projected   : $(awk -v s="$MAX_STEPS" 'BEGIN{printf "%.1f h", s*0.459/3600}') at the measured 0.459 s/step"
echo

.venv/bin/python -m geoarches.main_hydra --config-path "$REPO_ROOT/configs" \
    cluster=jureca_4gpu \
    module="${MODULE}" \
    dataloader="${DATALOADER}" \
    "++name=${NAME}" \
    "++max_steps=${MAX_STEPS}" \
    "++save_step_frequency=${SAVE_EVERY}" \
    "++module.module.num_training_steps=${MAX_STEPS}"

echo
echo "Done. Checkpoints in modelstore/${NAME}/checkpoints/"
ls -la "modelstore/${NAME}/checkpoints/" || true
