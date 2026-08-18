#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pre-train the `large` preset on four JUPITER booster nodes (16x GH200, DDP).
#
#   sbatch scripts/pretrain_large.slurm
#
# This is the ORGANISERS' job, not a participant's.  It produces the checkpoint
# that scripts/finetune.slurm starts from.  Do not launch it during the
# hackathon.
#
# It runs `python -m oceanarches.main_multinode`, NOT `geoarches.main_hydra`.
# geoarches builds `L.Trainer(...)` without `num_nodes`, so Lightning assumes
# one node and refuses to start on four:
#
#   ValueError: You set `num_nodes=1` in Lightning, but the number of nodes
#   configured in SLURM `--nodes=4` does not match. HINT: Set `num_nodes=4`.
#
# `oceanarches.main_multinode` supplies `num_nodes` and hands the composed
# config to geoarches' own `main`; nothing is forked, and at one node it is a
# no-op.  Read its header before changing the launch line.
#
# ---------------------------------------------------------------------------
# What it costs, and what four nodes actually buy
# ---------------------------------------------------------------------------
# `large` is 459.6 M parameters.  MEASURED on `dataloader=glorys` with metrics
# on, each arm timed at 60 and at 260 steps so the slope cancels start-up
# (jobs 1285595 and 1285596; the 1-GPU column is task 10's):
#
#      GPUs  nodes   s/optimiser step   samples/s   start-up   batches/epoch
#         1      1        0.5909           1.69         --         9488
#         4      1        0.7041           5.68        38 s        2372
#        16      4        0.7694          20.80        41 s         593
#
# Four nodes cost **9.3% more per optimiser step** than one and return
# **3.66x the throughput** -- 91.5% of perfect scaling from 4 GPUs to 16.
# Against a single GPU it is 1.30x per step for 12.29x the samples per second.
# One allocation each, and the other one was SLOWER: job 1285787, on different
# nodes, took 158.29 s for 200 extra steps plus one epoch boundary, i.e. at most
# 0.7914 s/step -- about 3% above the figure above, which is therefore the
# optimistic end of a +-5% band and not its middle.
#
# `max_steps` counts OPTIMISER steps, and one step is one sample per GPU here
# (batch_size: 1), so 16 GPUs see 16 samples per step where 4 see 4.  That
# leaves a choice, and this script makes it explicitly:
#
#   * keep max_steps=300000 -> 4.8 M samples (506 epochs of the 9488-sample
#     train split) in 300000 x 0.7694 = 64.1 h.  More wall clock than the
#     one-node run it replaces, for 500-odd epochs over a small archive.
#   * hold SAMPLES SEEN constant at the 1.2 M the one-node plan was sized for
#     -> max_steps = 1200000 / 16 = 75000, i.e. 75000 x 0.7694 = **16.0 h**
#     against 300000 x 0.7041 = 58.7 h on one node.  Same data, 3.66x sooner.
#
# It ships the second: MAX_STEPS below is derived from the sample budget and
# the size of the allocation, so `sbatch --nodes=1 scripts/pretrain_large.slurm`
# still runs the original 300000 steps and four nodes run 75000.  The cosine
# schedule follows (`num_training_steps: ${max_steps}`), so the shorter run is
# a complete schedule and not a truncated one.
#
# Those hours are a MEASUREMENT-BASED PROJECTION: the rate is measured, the
# total is arithmetic, and nobody has run this preset to completion.  The epoch
# boundaries barely move it: 75000 steps is 126 of them, and one is ~8 s --
# measured directly as the validation loop's 46 batches (job 1285787), and
# consistent with its 500-step and 700-step runs (507.3 s and 665.6 s, which
# leave 0.751 s/step once 8 s of boundary is taken out).  That is ~17 minutes on
# top, so call the whole thing **~16.5 h** -- two 12-hour jobs, with the +-5%
# band above on the rate.  If you are the first to finish it, write down what it
# took.
#
# One thing this does NOT adjust: `lr`.  The effective batch goes from 4 to 16,
# and configs/module/large.yaml explains what that means and gives the override.
#
# NOTE: the cap below is JUPITER's and has NOT been checked on JURECA. Confirm
# with `sacctmgr show qos <qos> format=Name,MaxWall` for dc-gpu before planning
# the number of launches; 12 h is safe under any larger cap, just not optimal.
# The booster QOS caps a job at **12 hours**:
#
#   $ sacctmgr show qos part_booster format=Name,MaxWall
#         part_booster    12:00:00
#
# 75000 steps is therefore two of these jobs back to back, not the five or six
# the one-node plan needed.  That is expected and is handled two ways, below.
# ---------------------------------------------------------------------------
#SBATCH --job-name=ocean-pretrain-large
#SBATCH --account=training2635
#SBATCH --partition=dc-gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pretrain-large-%j.out
#SBATCH --error=logs/pretrain-large-%j.out

set -euo pipefail

# SLURM copies the batch script into its spool directory, so BASH_SOURCE points
# there and not at the repository.  SLURM_SUBMIT_DIR is where `sbatch` ran.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source config.env

# --- knobs ------------------------------------------------------------------
# Override on the command line, e.g.
#   sbatch --export=ALL,NAME=large_v2,SAVE_EVERY=2500 scripts/pretrain_large.slurm
NAME="${NAME:-large_pretrained}"
MODULE="${MODULE:-large}"
DATALOADER="${DATALOADER:-glorys}"

# The cluster config follows the allocation, because it is the one thing that
# has to agree with `--nodes`: jureca_4nodes declares `num_nodes: 4` and
# `oceanarches.main_multinode` refuses to run it in a job of any other size.
# jureca_4gpu declares nothing, so the shim reads the node count off the
# allocation and it is correct at 1 node (and at 2, for an experiment).
if [[ -z "${CLUSTER:-}" ]]; then
    if [[ "${SLURM_NNODES:-1}" == "4" ]]; then
        CLUSTER=jureca_4nodes
    else
        CLUSTER=jureca_4gpu
    fi
fi

# The shipped plan is a SAMPLE budget, not a step budget: 1.2 M samples, which
# is what 300000 steps on one node came to and about 126 epochs of the
# 9488-sample train split.  Deriving `max_steps` from the world size keeps the
# science fixed while the resource decision changes -- 75000 steps on 16 GPUs,
# 300000 on 4, and no silent quartering of the data if someone submits this at
# `--nodes=1`.  It assumes `large`'s batch_size of 1 per GPU, which is what
# this script is for; pass MAX_STEPS yourself for anything else.
#
# A budget chosen here is a decision for the WHOLE run: raising it partway
# through means relaunching with a different `max_steps`, which is exactly the
# case the resume section below warns about.  Decide it before the first launch,
# or start a fresh NAME.
SAMPLE_BUDGET="${SAMPLE_BUDGET:-1200000}"
WORLD_SIZE="${SLURM_NTASKS:-4}"
if [[ -z "${MAX_STEPS:-}" ]]; then
    MAX_STEPS=$(( SAMPLE_BUDGET / WORLD_SIZE ))
    STEPS_NOTE="(${SAMPLE_BUDGET} samples / ${WORLD_SIZE} ranks)"
else
    # Do not claim the sample budget produced a number it did not: the log of a
    # job launched with MAX_STEPS=800 used to read `800 (1200000 samples / 16
    # ranks)`, which is false and would be believed six months from now.
    STEPS_NOTE="(MAX_STEPS was given; the ${SAMPLE_BUDGET}-sample budget is unused)"
fi

# How often a checkpoint is written.  It MUST divide max_steps, or the final
# step is never written to disk (geoarches checkpoints on
# `global_step % save_step_frequency == 0` and nothing checkpoints at the end of
# `fit`).  It is also the amount of work a timeout throws away: at 0.7694 s/step
# on 16 GPUs, 5000 steps is 1.07 h, i.e. under a tenth of a 12-hour job.  5000
# divides 75000, 150000 and 300000, so it survives a change of node count.
# Each one is ~5.5 GB (459.6 M parameters plus two AdamW moments), and nothing
# deletes them: budget ~83 GB for the 15 a 75000-step run writes.
SAVE_EVERY="${SAVE_EVERY:-5000}"
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-}"

if (( MAX_STEPS % SAVE_EVERY != 0 )); then
    echo "FATAL: save_step_frequency=${SAVE_EVERY} does not divide max_steps=${MAX_STEPS};" >&2
    echo "       the finished model would be the one checkpoint never written." >&2
    exit 1
fi

# --- resume / requeue -------------------------------------------------------
# configs/config.yaml ships `resume: True`, and geoarches resumes from
# `exp_dir` whenever `modelstore/<name>/checkpoints/` already holds anything:
# it reloads the newest checkpoint with the optimiser state and the global step
# intact.  So there are two ways to run the full budget:
#
#   (a) MANUAL, and on JUPITER this is the only one that works: run
#       `sbatch scripts/pretrain_large.slurm` again with the same NAME.  The
#       trap below prints the exact command, with `--nodes` and MAX_STEPS
#       pinned, and both matter.
#
#       **THE STEP BUDGET MUST NOT CHANGE ACROSS A RELAUNCH.**  It is derived
#       from the world size, so relaunching at a different `--nodes` moves it,
#       and so does raising SAMPLE_BUDGET partway through.  Two things then go
#       wrong: the finish line moves, and -- until this script started passing
#       `++module.module.num_training_steps` (see HYDRA_ARGS below) -- the
#       cosine schedule stayed behind at the old budget and the learning rate
#       climbed back toward its peak instead of decaying to zero.  The override
#       fixes the schedule; nothing can fix the fact that a changed budget means
#       the run is no longer the run you started.  Want more samples?  Choose
#       SAMPLE_BUDGET before the first launch, or start a fresh NAME.
#
#       Both halves measured on this script.  Job 1286618 ran 100 steps at
#       `--nodes=2` (800 samples / 8 ranks); job 1286623 relaunched the same
#       NAME at `--nodes=1`, derived 200 steps (800 / 4), restored from
#       `checkpoint_global_step=100.ckpt` and finished at 200.  geoarches' own
#       dumps show the merge doing its work: the pre-merge `hydra config` line
#       still carried `'num_training_steps': 100` from the first launch, and the
#       post-merge `updated cfg` line carried 200 on all four ranks.
#
#       JUPITER's submit filter rejects requeueable jobs outright:
#
#         $ sbatch --requeue ... scripts/pretrain_large.slurm
#         sbatch: error: job_submit_filter: --requeue option is not supported
#         sbatch: error: Batch job submission failed: Requested operation not
#                        supported on this system
#
#       so `#SBATCH --requeue` is deliberately absent -- with it, the job does
#       not submit at all.  Found by actually submitting this script.
#
#   (b) automatic, where the site allows it -- `--signal=B:USR1@600` asks SLURM
#       to signal THIS shell (that is what the `B:` prefix means; without it the
#       signal goes to the job steps and the trap never fires) 600 s before the
#       wall clock runs out, and the trap tries `scontrol requeue`.  On a site
#       that permits it the requeued job keeps its JOBID, re-runs this script
#       from the top, finds the checkpoints and continues.  On JUPITER the
#       `scontrol requeue` fails and the trap prints the relaunch command
#       instead, which is why (a) is listed first.
#
# Lightning prints "SLURM auto-requeueing enabled" and registers its own SIGUSR1
# handler.  That is deliberately NOT the mechanism here.  Its handler writes
# `.pl_auto_save.ckpt` into `trainer.default_root_dir` (the repository root),
# and geoarches only ever looks in `modelstore/<name>/checkpoints/`, so that
# checkpoint would be orphaned and the resume would come off the last geoarches
# checkpoint anyway.  The `B:` in `--signal=B:USR1@600` keeps the signal on the
# batch shell so Lightning's handler never fires and no stray file is left.
#
# Either way you lose at most SAVE_EVERY steps.  Verified at 16 GPUs on this
# script, with the real preset: job 1285799 (MAX_STEPS=800 SAVE_EVERY=400) was
# cancelled at global step 400, and job 1285909, submitted with the same NAME,
# reported `Restored all states from the checkpoint at
# .../checkpoint_global_step=400.ckpt` and carried on from 400 through the epoch
# boundary at 593 to 800.  The same was proved at 4 GPUs in task 10.
requeue_on_timeout() {
    echo
    echo "=== USR1 at $(date -Is): wall clock nearly up on job ${SLURM_JOB_ID} ==="
    echo "    Training resumes from the newest checkpoint in ${MODELSTORE}/${NAME}/checkpoints/,"
    echo "    so at most ${SAVE_EVERY} steps are lost."
    if scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null; then
        echo "    Requeued. Nothing more to do."
    else
        echo "    This site does not allow requeueing (JUPITER does not). Relaunch with:"
        echo
        echo "      sbatch --nodes=${SLURM_NNODES} --export=ALL,NAME=${NAME},MODULE=${MODULE},DATALOADER=${DATALOADER},CLUSTER=${CLUSTER},SAVE_EVERY=${SAVE_EVERY},MAX_STEPS=${MAX_STEPS} scripts/pretrain_large.slurm"
        echo
    fi
}
trap requeue_on_timeout USR1

# --- what we are about to run ----------------------------------------------
echo "=============================================================="
echo "job         : ${SLURM_JOB_ID:-<none>}  (restart #${SLURM_RESTART_COUNT:-0})"
echo "host        : $(hostname)"
echo "nodes/tasks : ${SLURM_NNODES:-?} node(s), ${SLURM_NTASKS:-?} task(s)"
echo "cluster cfg : ${CLUSTER}"
echo "steps       : ${MAX_STEPS} ${STEPS_NOTE}, checkpoint every ${SAVE_EVERY}"
echo "repo        : ${REPO_ROOT}"
echo "data        : ${GLORYS_PREPPED}"
echo "exp_dir     : ${MODELSTORE}/${NAME}"
echo "started     : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo

CONFIG_PATH="${REPO_ROOT}/configs"
HYDRA_ARGS=(
    --config-path "${CONFIG_PATH}"
    cluster="${CLUSTER}"
    module="${MODULE}"
    dataloader="${DATALOADER}"
    "++name=${NAME}"
    "++max_steps=${MAX_STEPS}"
    "++save_step_frequency=${SAVE_EVERY}"
    # The learning-rate schedule, pinned to the same number as `max_steps`, and
    # this is NOT redundant with the `${max_steps}` interpolation in
    # configs/module/large.yaml.  On a resume, geoarches replaces `cfg.module`
    # wholesale with the RESOLVED module config written at the first launch and
    # then re-applies only the `+`-prefixed command-line overrides:
    #
    #     cfg.module = exp_cfg.module            # ${max_steps} already baked in
    #     cli_overrides = [x.removeprefix("++") for x in ... if x.startswith("+")]
    #     cfg.merge_with_dotlist(cli_overrides)
    #
    # so `++max_steps=` moves the trainer's budget and leaves
    # `module.module.num_training_steps` frozen at the first launch's value.
    # The schedule then runs off its end, and diffusers' cosine lambda,
    # `max(0, 0.5 * (1 + cos(pi * progress)))`, does not stay at zero past
    # progress 1 -- it CLIMBS BACK toward the peak learning rate.  Passing it
    # here starts with `+`, so it survives that merge.
    "++module.module.num_training_steps=${MAX_STEPS}"
)
# shellcheck disable=SC2206
[[ -n "${EXTRA_HYDRA_ARGS}" ]] && HYDRA_ARGS+=(${EXTRA_HYDRA_ARGS})

# Print the config hydra will actually compose, before spending four node-days
# on it.  `--cfg job --resolve` composes and exits; it costs a couple of
# seconds.  Same entry point as the run, so it also proves the shim imports.
echo "--- resolved config ------------------------------------------"
.venv/bin/python -m oceanarches.main_multinode "${HYDRA_ARGS[@]}" --cfg job --resolve || {
    echo "FATAL: the config does not compose. Nothing was launched." >&2
    exit 1
}
echo "--------------------------------------------------------------"
echo

# One process per GPU, on every node.  `--cpus-per-task=72` is safe here BECAUSE
# ntasks-per-node is 4 and that is exactly how many processes we want per node;
# on a one-GPU job the same flag without `--ntasks=1` silently launches N copies
# of the script.
# Do NOT pin CUDA_VISIBLE_DEVICES here: each rank must see its own card, and
# SLURM_LOCALID already selects it.
#
# `--kill-on-bad-exit=1` is not a nicety.  The startup guards refuse on rank 0
# only (every rank composes the config, so a guard that fired on all 16 would cry
# wolf at the very configuration this script exists for).  Rank 0 is also the
# rendezvous master.  So when a guard refuses -- job 1349666, a stale lock -- rank
# 0 prints the reason and exits, and the remaining 15 ranks sit in
# `init_process_group` waiting for a store that will never be created, for the
# full 30-minute default timeout.  What comes out at the end is 15 copies of
# `DistNetworkError: The client socket has timed out`, which reads as a network
# fault and buries the one line that says what actually happened two thousand
# lines up.  Two of those cost 4 nodes x 1 h 15 each.  With this flag srun tears
# the step down the moment any task exits non-zero, and the refusal is the last
# thing in the log.
srun --cpu-bind=none --kill-on-bad-exit=1 .venv/bin/python -m oceanarches.main_multinode "${HYDRA_ARGS[@]}" &
TRAIN_PID=$!

# `wait` is interruptible, which is what lets the USR1 trap run.  A trapped
# signal makes `wait` return >128 without the child having exited, so loop until
# it has really gone and keep the status it exited with.
#
# The status matters.  `wait ... || true` here would make the script exit 0
# whatever training did, SLURM would report COMPLETED for a job that died on
# step 1, and across the requeues of a multi-day run nobody would notice
# until the checkpoint was missing at the end.
TRAIN_STATUS=0
while true; do
    # `|| TRAIN_STATUS=$?` and not a bare `wait`: under `set -e` a bare failing
    # `wait` kills the script on the spot, so the status is never captured, the
    # message below never prints and the checkpoint listing never runs.
    TRAIN_STATUS=0
    wait "${TRAIN_PID}" || TRAIN_STATUS=$?
    # >128 means `wait` itself was interrupted by a signal we trapped; the child
    # is still running, so go round again.
    if (( TRAIN_STATUS > 128 )) && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        continue
    fi
    break
done

echo
echo "finished    : $(date -Is)  (training exited ${TRAIN_STATUS})"
ls -la "${MODELSTORE}/${NAME}/checkpoints/" 2>/dev/null || true

if (( TRAIN_STATUS != 0 )); then
    echo "FAILED: training exited ${TRAIN_STATUS}. Not treating this as a completed run." >&2
fi
exit "${TRAIN_STATUS}"
