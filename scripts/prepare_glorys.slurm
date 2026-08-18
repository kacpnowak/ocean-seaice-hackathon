#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Prepare the whole GLORYS archive (1993-2025) in one short job.
#
#   sbatch scripts/prepare_glorys.slurm
#
# One process per year on a single node.  Each year is an independent
# read/write, so this is limited by the filesystem, not the CPU.
# Expect roughly 10 minutes and ~100 GB of output.
#
# The account and partition below are the defaults from config.env.  SBATCH
# directives cannot read that file, so override on the command line if needed:
#   sbatch -A training2635 -p dc-gpu scripts/prepare_glorys.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=glorys-prep
#SBATCH --account=training2635
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/glorys-prep-%j.out

set -euo pipefail

# SLURM copies the batch script into its spool directory, so BASH_SOURCE points
# there and not at the repository. SLURM_SUBMIT_DIR is where `sbatch` was run.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source config.env

# Override the account/partition from config.env if SLURM did not get them from
# the sbatch command line.
: "${SLURM_ACCOUNT:=training2635}"
: "${SLURM_PARTITION:=dc-gpu}"

echo "host      : $(hostname)"
echo "raw       : ${GLORYS_RAW}"
echo "prepared  : ${GLORYS_PREPPED}"
echo "started   : $(date -Is)"
echo

.venv/bin/python scripts/prepare_glorys.py --years 1993-2025 --jobs 16

echo
echo "finished  : $(date -Is)"
.venv/bin/python -m oceanarches.doctor || true
