#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fine-tune a pre-trained checkpoint on one GPU.  This is the path participants
# actually use: the organisers pre-train `large` with
# scripts/pretrain_large.slurm, you start from those weights.
#
#   sbatch --export=ALL,FROM=task6_tiny,NAME=my_finetune \
#          scripts/finetune.slurm
#
# Everything is an environment variable with a default, so the common case is
# just `sbatch scripts/finetune.slurm`.
#
#   FROM        run under modelstore/ to take the WEIGHTS from   [task6_tiny]
#   NAME        run under modelstore/ to WRITE                   [<FROM>_finetuned]
#   MODULE      size preset -- must match FROM exactly           [tiny]
#   DATALOADER  which variables / which years                    [glorys]
#   MAX_STEPS   fine-tuning budget                               [2000]
#   SAVE_EVERY  checkpoint interval; must divide MAX_STEPS       [500]
#   LR          learning rate                                    [1e-4]
#   CLUSTER     configs/cluster/*.yaml -- batch, precision, workers  [jureca_1gpu]
#   EXTRA_HYDRA_ARGS  further hydra overrides, space separated     [none]
#
# The wall clock is a real #SBATCH directive and not a variable, because SBATCH
# directives cannot read the environment: override it on the command line with
# `sbatch --time=04:00:00 ... scripts/finetune.slurm`.
#
# ---------------------------------------------------------------------------
# What `load_ckpt` does, and what it does not do
# ---------------------------------------------------------------------------
# geoarches has two different mechanisms and only one of them is fine-tuning:
#
#   * `resume` (automatic, whenever modelstore/<name>/checkpoints exists)
#     continues a run: same optimiser state, same global step, same schedule.
#   * `load_ckpt=<dir>` loads the WEIGHTS from another run's newest checkpoint
#     and starts a fresh run around them: fresh optimiser, step 0, fresh LR
#     schedule.  That is what this script uses.
#
# Because the two are different, this script writes to a NEW exp_dir.  The
# pre-trained checkpoint under modelstore/${FROM}/ is opened read-only and is
# never written to -- which is the whole point, since on hackathon day one
# `large` checkpoint is shared by everybody.  The script refuses to run if
# NAME == FROM.
#
# `load_ckpt` calls `load_state_dict` in STRICT mode, so MODULE, the component
# (i.e. DATALOADER) and the depth count must match the run FROM came from.  You
# cannot load `tiny` into `small`, or a `full` model into a `seaice` specialist.
#
# Use a lower learning rate than pre-training used.  At the pre-training rate
# the first few hundred steps throw the pre-trained weights away.
#
# How to tell it worked: the FIRST logged `train_loss`.  An untrained model
# starts near 3800 on this loss; a `tiny` model that has finished pre-training
# is near 0.74.  If step 0 reads ~3800, the weights did not land.
# ---------------------------------------------------------------------------
#SBATCH --job-name=ocean-finetune
#SBATCH --account=training2635
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.out

set -euo pipefail

# SLURM copies the batch script into its spool directory, so BASH_SOURCE points
# there and not at the repository.  SLURM_SUBMIT_DIR is where `sbatch` ran.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source config.env

FROM="${FROM:-task6_tiny}"
NAME="${NAME:-${FROM}_finetuned}"
MODULE="${MODULE:-tiny}"
DATALOADER="${DATALOADER:-glorys}"
CLUSTER="${CLUSTER:-jureca_1gpu}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LR="${LR:-1e-4}"
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-}"

# --- refuse the two mistakes that destroy the pre-trained model --------------
if [[ "${NAME}" == "${FROM}" ]]; then
    echo "FATAL: NAME and FROM are both '${NAME}'." >&2
    echo "       That would resume the pre-trained run and overwrite its checkpoints" >&2
    echo "       instead of fine-tuning from them. Pick a different NAME." >&2
    exit 1
fi
if [[ ! -d "${MODELSTORE}/${FROM}/checkpoints" ]]; then
    echo "FATAL: ${MODELSTORE}/${FROM}/checkpoints does not exist, so there are no" >&2
    echo "       weights to fine-tune from. Available runs:" >&2
    ls -1 "${MODELSTORE}" 2>/dev/null | sed 's/^/         /' >&2
    exit 1
fi
# The header above says it in prose ("MODULE must be the preset FROM was trained
# with"), and MODULE defaults to `tiny`, so fine-tuning a `small` or `base`
# checkpoint with the default is a very easy mistake -- and it costs 343 lines of
# `size mismatch for backbone...` after the queue clears, not before.
FROM_PRESET="$(sed -n 's/^  *emb_dim: *\([0-9]*\) *$/\1/p' "${MODELSTORE}/${FROM}/config.yaml" | head -1)"
THIS_PRESET="$(sed -n 's/^ *emb_dim: *\([0-9]*\) *$/\1/p' "configs/module/${MODULE}.yaml" | head -1)"
if [[ -n "${FROM_PRESET}" && -n "${THIS_PRESET}" && "${FROM_PRESET}" != "${THIS_PRESET}" ]]; then
    echo "FATAL: MODULE=${MODULE} has emb_dim=${THIS_PRESET}, but ${FROM} was trained" >&2
    echo "       with emb_dim=${FROM_PRESET}. load_ckpt copies weights tensor by tensor," >&2
    echo "       so every layer would fail with 'size mismatch' -- after the job starts." >&2
    echo "       Set MODULE to the preset ${FROM} was trained with. Its config is:" >&2
    echo "         ${MODELSTORE}/${FROM}/config.yaml" >&2
    exit 1
fi
if (( MAX_STEPS % SAVE_EVERY != 0 )); then
    echo "FATAL: SAVE_EVERY=${SAVE_EVERY} does not divide MAX_STEPS=${MAX_STEPS};" >&2
    echo "       geoarches checkpoints on 'global_step % save_step_frequency == 0'" >&2
    echo "       and nothing checkpoints at the end of fit, so the finished model" >&2
    echo "       would be the one checkpoint never written." >&2
    exit 1
fi

# Pin this process to exactly one card, whatever the allocation handed back.
#
# On JUPITER this was mandatory: the booster allocates WHOLE NODES, so --gres=gpu:1
# came back with all four cards and Lightning's devices="auto" would shard the
# data four ways inside this single process -- a quarter of the dataset, with a
# loss curve that looks completely normal.  JURECA's dc-gpu is not whole-node, so
# --gres=gpu:1 should really give one card and this line should be a no-op.
# It is kept because it is a no-op when the allocation is already one GPU, and it
# is the difference between a correct run and a silently quartered one when it
# is not.
export CUDA_VISIBLE_DEVICES=0

echo "=============================================================="
echo "job         : ${SLURM_JOB_ID:-<none>}"
echo "host        : $(hostname)"
echo "repo        : ${REPO_ROOT}"
echo "weights from: ${MODELSTORE}/${FROM}   (read-only)"
echo "writing to  : ${MODELSTORE}/${NAME}"
echo "preset      : module=${MODULE} dataloader=${DATALOADER} lr=${LR}"
echo "schedule    : ${MAX_STEPS} steps, checkpoint every ${SAVE_EVERY}"
echo "started     : $(date -Is)"
echo "=============================================================="
ls -la "${MODELSTORE}/${FROM}/checkpoints/"
BEFORE=$(find "${MODELSTORE}/${FROM}/checkpoints" -type f -printf '%p %s %T@\n' | sort)
echo

CONFIG_PATH="${REPO_ROOT}/configs"
HYDRA_ARGS=(
    --config-path "${CONFIG_PATH}"
    cluster="${CLUSTER}"
    module="${MODULE}"
    dataloader="${DATALOADER}"
    "++name=${NAME}"
    "+load_ckpt=${MODELSTORE}/${FROM}"
    "++max_steps=${MAX_STEPS}"
    "++save_step_frequency=${SAVE_EVERY}"
    "++module.module.lr=${LR}"
    # The cosine schedule is built for THIS run's step budget, not the
    # pre-training one, so warm-up has to shrink with it or the run spends a
    # large fraction of its life below the learning rate it was given.
    "++module.module.num_warmup_steps=$(( MAX_STEPS / 20 > 10 ? MAX_STEPS / 20 : 10 ))"
)
# shellcheck disable=SC2206
[[ -n "${EXTRA_HYDRA_ARGS}" ]] && HYDRA_ARGS+=(${EXTRA_HYDRA_ARGS})

echo "--- resolved config ------------------------------------------"
.venv/bin/python -m geoarches.main_hydra "${HYDRA_ARGS[@]}" --cfg job --resolve || {
    echo "FATAL: the config does not compose. Nothing was launched." >&2
    exit 1
}
echo "--------------------------------------------------------------"
echo

# No `srun` here on purpose.  This is a single process, and `srun` with
# --cpus-per-task=N but without --ntasks=1 launches N concurrent copies of the
# script; that has already corrupted a measurement in this project.  sbatch
# already gives us exactly one shell.
.venv/bin/python -m geoarches.main_hydra "${HYDRA_ARGS[@]}"

echo
echo "finished    : $(date -Is)"
echo "new run     : ${MODELSTORE}/${NAME}"
ls -la "${MODELSTORE}/${NAME}/checkpoints/" 2>/dev/null || true

AFTER=$(find "${MODELSTORE}/${FROM}/checkpoints" -type f -printf '%p %s %T@\n' | sort)
if [[ "${BEFORE}" == "${AFTER}" ]]; then
    echo "pre-trained checkpoints under ${MODELSTORE}/${FROM} are untouched -- OK"
else
    echo "WARNING: ${MODELSTORE}/${FROM} CHANGED during fine-tuning. It should not have." >&2
    diff <(echo "${BEFORE}") <(echo "${AFTER}") >&2 || true
fi
