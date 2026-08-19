# For the tutors

Kacper and Nils. Participants do not read this file -- they start at
[docs/00_start_here.md](00_start_here.md). This is the operational side: what has
to be true before the room fills up, what only a tutor can do, and the one-line
answer to each failure we have already seen somebody hit.

---

## 0. This kit was moved from JUPITER

Everything below was built, measured and run on **JUPITER**. It now lives on
**JURECA**, under `/p/scratch/training2635/4_ocean_ai/nowak2/hackathon-ocean-sea-ice`,
because JUPITER's `/e/scratch` and `/e/data1` are that machine's Exascale
filesystems and are not mounted here.

**What came across**

| | |
|---|---|
| prepared GLORYS, 1993--2025 | 92 GB, `data/glorys_1deg_prepped/` |
| IFS forcing (optional, ~2024) | 833 MB, `data/ifs_1deg/` |
| masks, normalisation, climatology | 94 MB, `oceanarches/stats/` |
| the five shipped runs | 16 GB, `modelstore/` (of which `base_pretrained` is 14 GB) |
| the `base` scorecard | 724 MB, `evalstore/base_pretrained/` |
| the repository, with history and the `public` release branch | `git log` |

**What deliberately did not, and why**

* **`.venv`** -- JUPITER is aarch64, JURECA is x86_64. Those wheels cannot run
  here. Build it with `make setup`.
* **The raw GLORYS archive** (640 GB) -- only `make prep-data` reads it, and the
  92 GB it produces came across ready to use. `make doctor` reports this as a
  WARN, not a FAIL, for exactly that reason.
* **Most of the 0.25-degree archive** (~10 TB, 292 GB/year). **Three years are
  staged** for [docs/07 question 3](07_challenge_ideas.md), raw and unprepared,
  at `/p/scratch/training2635/4_ocean_ai/nowak2/glorys_025/{2017,2018,2019}` --
  876 GB. Consecutive, and chosen against `SPLIT_YEARS`: 2017--2018 are training
  years and 2019 is the validation year, which is also exactly `tiny_val`, so
  `dataloader=glorys_tiny` works on them unedited. There is **no test year** in a
  run of three (`test` starts at 2021), so that question scores with
  `--domain val`. Another year is 292 GB and about eight minutes:

  ```bash
  rsync -a /e/data1/climateai/hclimrep/data/glorys_025/2020 \
      /p/scratch/training2635/4_ocean_ai/nowak2/glorys_025/     # from JUPITER
  ```
* **Fourteen of the fifteen `large` checkpoints** (72 GB) -- `load_module` takes
  the newest, so only `checkpoint_global_step=75000.ckpt` shipped.
* **The diverged `lr 2e-4` run** (47 GB) -- failure evidence, it stays on JUPITER.

**What was changed for JURECA:** the account (`training2635`), the partition
(`dc-gpu`), `--cpus-per-task` (12), every path, three new
`configs/cluster/jureca_*.yaml`, the `CLUSTER` default, and the raw-data check in
`make doctor`. The account, partition and core count are not guesses -- they are
what the other training2635 challenges already run.

### What has been verified on JURECA, and what has not

The move itself was made from a JUPITER login node with no access to JURECA, so
this section once said that nothing here had ever run. **That is no longer true:
the kit has since been walked end to end on JURECA as a participant would.**
What that found is in the commit `Make the quickstart work on JURECA`, and the
two things it fixed had both stopped a fresh clone before it trained anything --
`make setup` hitting the **inode** quota on `$HOME` (not a space quota; `df`
shows terabytes free), and `make train-tiny` dying of CUDA OOM because every
preset's `batch_size` was measured on a 96 GB GH200.

**Verified here, on a dc-gpu A100-SXM4-40GB (39.49 GiB usable):**

| | |
|---|---|
| `make setup` | 58 s cold, with `UV_*` pointed at `.uv/` on scratch |
| `make doctor` | passes, and now checks `$HOME` inode headroom |
| `make test` | the suite, on x86_64 |
| `make train-tiny` | 21.44 GiB peak at `batch_size 4`, 1.98 it/s, ~34 min |
| `make eval` on a `tiny`-class run | fits and scores |
| `base` on 4 GPUs, real path | 2.18 it/s = 0.459 s/step, four DDP ranks -- the budget behind `scripts/pretrain_base.slurm` |

**The one thing that does not fit, at all: `large` on this card.** It needs
50.81 GiB at batch 1 with gradient checkpointing already on, against 39.5 GiB,
whether you are training it or fine-tuning it. More GPUs do not help -- measured:
DDP replicates the model on every rank and all four died at 39.4 GiB. So the
`large` checkpoint has been **withdrawn** ([section 3](#3-the-withdrawn-large-model-and-what-replaced-it))
and `base` on four GPUs is the reference run instead. README and
[docs/04](04_scaling_finetuning.md) both say this.

**Still not re-measured:** `small`'s wall clock, `base` on ONE GPU (only the
four-GPU figure above exists), and every other wall-clock number in the
documents. Peak memory per preset IS measured -- `make benchmark` on this card
gives `tiny` 21.44 GiB at batch 4, `small` 21.43 at 2, `base` 21.33 at 1, and
`large` does not fit at any batch -- and those are the `batch_size_40gib` numbers
in `configs/module/*.yaml`. The remaining GH200 figures were labelled rather than
rewritten, because inventing numbers is worse than dating old ones -- re-measure
what you intend to quote.

If you rebuild from scratch on another machine, the order that works:

```bash
cd /p/scratch/training2635/4_ocean_ai/nowak2/hackathon-ocean-sea-ice
make setup                     # 1: builds the x86_64 .venv
make doctor                    # 2: says what else is missing
make test                      # 3: the suite, on this architecture
srun --account=training2635 --partition=dc-gpu --gres=gpu:1 --ntasks=1 \
     --cpus-per-task=12 --time=01:00:00 --pty bash
make benchmark                 # 4: peak memory per preset -- the batch-size answer
make eval NAME=task6_tiny EVAL_ARGS="--n-inits 2 --lead-days 2 --skip-animations"
                               # 5: proves a shipped checkpoint loads and scores here
```

## 1. Before day one

| | | |
|---|---|---|
| ☐ | **Participants are in the `training2635` group** | **Only you can do this, and nothing works without it.** |
| ☑ | Prepared data, 92 GB, 1993--2025 | `data/glorys_1deg_prepped/` |
| ☑ | Statistics, masks, climatology | `oceanarches/stats/` |
| ☑ | Four small trained runs | `modelstore/{task6_tiny,ocean_tiny,seaice_tiny,seaice_isolated_tiny}` |
| ☑ | The pre-trained `base` checkpoint | `modelstore/base_pretrained`, 84 000 steps, fits one A100 for scoring AND fine-tuning -- [section 3](#3-the-withdrawn-large-model-and-what-replaced-it) |
| ☑ | Its scorecard | `evalstore/base_pretrained/report.md` |
| ☒ | ~~The pre-trained `large` checkpoint~~ | **WITHDRAWN** -- neither trainable nor fine-tunable on a 40 GiB A100. |
| ☑ | Public repository | `git@github.com:kacpnowak/ocean-seaice-hackathon.git` |

### The group is the whole gate

```
/p/scratch/training2635              drwxrws---   root:training2635
/p/scratch/training2635/4_ocean_ai   drwxrwsr-x   patnala1:training2635
```

Everything below it is world-readable, so **any member of `training2635` can read the
kit, the data and the checkpoints, and a non-member can read none of it** -- not
the clone, not the 92 GB of prepared data, not the statistics. There is no
partial state and no useful error: a non-member gets `Permission denied` on the
`git clone` in [docs/01 §1.1](01_setup.md) and stops there.

Check the room against the group before the session, not during it:

```bash
getent group training2635
```

---

## 2. The shared store

There is no separate share. **This checkout is the shared store**, and
[docs/01 §1.1](01_setup.md) sends participants straight at it:

```
/p/scratch/training2635/4_ocean_ai/nowak2/hackathon-ocean-sea-ice
```

They `git clone` it (about 7 MB -- everything large is git-ignored), run
`make setup` and `make stats` in their own copy, and **symlink the shipped runs
in one by one**:

```bash
SHARED=/p/scratch/training2635/4_ocean_ai/nowak2/hackathon-ocean-sea-ice/modelstore
for run in task6_tiny ocean_tiny seaice_tiny seaice_isolated_tiny; do
    ln -s $SHARED/$run modelstore/$run
done
```

One by one, **not** `ln -s $SHARED modelstore`. The rehearsal did the latter
anyway, which makes every run name subsequently trained resolve into this
directory -- shared with the whole room. That is why
`oceanarches/paths.py` refuses to write to or delete anything reached through a
symlink, and why the four names above are in `SHIPPED_RUNS` and cannot be
overwritten even where they are a directory of somebody's own.

Your `modelstore/` is mode `755` and owned by you, so the room reads it and
cannot write it. Nothing enforces that from inside the kit -- it is the
filesystem. **Do not loosen it.**

### Adding a run to the shipped set

Four places, and all four matter:

1. put the run in `modelstore/<name>/` (checkpoint + `config.yaml`),
2. add `"<name>"` to `SHIPPED_RUNS` in [oceanarches/paths.py](../oceanarches/paths.py)
   -- otherwise the kit will happily let a participant delete it,
3. add it to the `for run in ...` loop in [docs/01 §1.1](01_setup.md),
4. `bash scripts/publish.sh "..."` ([section 4](#4-publishing)).

`make test` catches steps 2 and 3 if you forget: `test_run_store.py` asserts that
`SHIPPED_RUNS` and the `for run in ...` loop in docs/01 are the same list, and
that every name in it is really in the store. Steps 1 and 4 are yours.

---

## 3. The withdrawn `large` model, and what replaced it

**`large_pretrained` has been deleted from the shared store, along with its
scorecard.** It cannot be trained *or* fine-tuned on a JURECA dc-gpu A100: it
needs 50.81 GiB at batch 1 with `gradient_checkpointing: True` already on,
against 39.5 GiB available. Four GPUs do not help -- measured, all four DDP ranks
OOM at 39.4 GiB, because Lightning replicates the model per rank rather than
sharding it. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does not help
either, also measured. Shipping a 5.5 GB checkpoint that participants could only
ever run inference on, in a kit whose whole point is that they train something,
was the wrong trade.

**It is probably recoverable if you want it back.** What was deleted here was the
JURECA copy: `modelstore/large_pretrained/` (5.2 GB, the step-75 000 checkpoint
only) and `evalstore/large_pretrained/` (736 MB). Per
[section 3.4](#34-housekeeping), JUPITER's `modelstore/` held the full training
directory -- 78 GB, fifteen checkpoints -- and no JUPITER filesystem is visible
from JURECA, so that copy was not touched. Check there before assuming the 18
GPU-hours are gone.

**What it was, for the record.** 75 000 steps of `module=large` on 4 nodes x 4
GH200 (16 ranks), `batch_size 1` per rank, `dataloader=glorys`, `lr 1e-4` with a
10 000-step warm-up. About 1.3 it/s per rank -- roughly 18 hours of wall clock
across two launches, because that is more than the 12-hour QOS limit, which is
the whole reason [section 3.2](#32-relaunching-after-the-wall-clock) exists.
Final `val_loss` 0.236, still falling. On the held-out test years it beat
persistence on every variable at every lead time out to 10 days -- 36% on day-1
SST, 53% on sea surface height, 35% on sea-ice concentration -- and stayed better
than climatology for 23 days on SST. **Those numbers are no longer reproducible
in this kit**; they are kept here as the shape of a good result, not as a claim.

**The replacement is trained and shipped: `base_pretrained`.**
[`scripts/pretrain_base.slurm`](../scripts/pretrain_base.slurm) produced it in
**9 h 23 m** on one node (4 x A100-40GB), 84 000 steps, final `val_loss` 0.263
against the withdrawn `large`'s 0.236 -- from a fifth of the parameters and a
sixteenth of the GPUs. Measured 2.52 it/s against the 2.18 the budget assumed,
which is where the 2.5 h of margin inside the 12 h window came from.

| | |
|---|---|
| beats 1-day persistence | **all 17 scored variables, every lead time to day 10** |
| day-1 gains | SSH -50%, SST -31%, `siconc` -36%, `so` at 1684 m -50% |
| against `tiny`, identical samples | loss **0.978** vs 2.114 (persistence 2.261) |
| 90-day free run | SST RMSE **2.4 degC** (clim 0.68); **0.07%** of cells outside [-5, 40] degC, min -10.9 |
| `tiny` on the same test | 15-20 degC, 19-22% of cells outside, minima -514 |
| fits one dc-gpu A100 | scoring **and** fine-tuning, `batch_size 1`, 21.33 GiB -- both tested |

**It did not fix everything, and the documents say so.** Sea-ice *extent bias* is
still worse than persistence in both hemispheres at every lead time -- NH +0.032
at day 1 against persistence's -0.018, +0.414 by day 10, i.e. systematically too
much ice. `tiny` had the same failure and `base` inherited it. Every RMSE and
ice-edge number improved; that one did not, and it is a real thing for a team to
go after.

**Housekeeping you may want to do.** `modelstore/base_pretrained/` is **14 GB**
across fourteen checkpoints and `load_module` reads only the newest (step 84 000,
1.0 GB). Participants symlink the directory, so the other thirteen cost them
nothing and cost you 13 GB. Thin it if you want the space; nothing depends on them.

### 3.1 The learning rate is not a free knob

The first attempt ran at `lr 2e-4` and **diverged between step 15 000 and
20 000**: healthy at 15 000 (loss 1.96), destroyed by 20 000 (loss 1.05e8,
output std 652 against a target of 0.69). Gradient clipping at 1.0 was on the
whole time and did not save it, and the run then trained for eight more hours
producing nothing. `1e-4` with a 10 000-step warm-up is the setting that holds.

Two things came out of that and are now permanent:

* `DivergenceGuard` aborts the run instead of burning the rest of the
  allocation (`++module.module.divergence_guard=False` turns it off).
* Do not raise the learning rate for the `large` preset without watching the
  first 20 000 steps.

### 3.2 Relaunching after the wall clock

The QOS limit is 12:00:00, so a full `large` run needs at least two launches.
The script is built for it -- `sbatch scripts/pretrain_large.slurm` resumes from
the newest checkpoint on its own, and carries `num_training_steps` with it so the
cosine schedule does not climb back toward peak on the second launch.

**A relaunch used to be refused by the previous launch's lock.** SLURM enforces
the wall clock with SIGKILL, so nothing cleans `modelstore/<name>/.training.lock`
up, and the guard believed a lock from another host for a day. Two 4-node
relaunches died an hour in with fifteen copies of
`DistNetworkError: The client socket has timed out`, which is what it looks like
when the guard refuses on rank 0 -- the rendezvous master -- and the other
fifteen ranks wait for a store that is never created.

Both halves are fixed: the lock now asks `squeue` whether the job that wrote it
still exists, and `srun --kill-on-bad-exit=1` tears the step down the moment any
rank exits. **If you see a rendezvous timeout again, read the top of the log
before blaming the network** -- a rank-0 refusal is now the last thing in it.

### 3.3 Staging the finished checkpoint

This is the recipe for staging any finished run, and the one waiting for it is
the `base` run from [section 3](#3-the-withdrawn-large-model-and-what-replaced-it).
Follow [section 2](#adding-a-run-to-the-shipped-set): the run directory, then
`SHIPPED_RUNS`, then the loop in docs/01, then a release. `load_module` takes the
newest checkpoint in `checkpoints/`, so the run directory is what ships and the
intermediate checkpoints are dead weight rather than a problem
([section 3.4](#34-housekeeping)).

**Fine-tuning on JURECA means `base` or smaller.** `large` does not fit at all
(above). The fine-tune path against a preset that does:

```bash
sbatch --export=ALL,FROM=task6_tiny,NAME=my_finetune_tiny scripts/finetune.slurm
```

**`MODULE=large` is the part they will forget.** It defaults to `tiny`, and
`load_ckpt` copies weights tensor by tensor, so the default against a `large`
checkpoint is 343 lines of `size mismatch` -- after the queue clears. The script
compares `emb_dim` and refuses before spending the GPU time, which is the whole
reason that guard is in it. See [docs/04](04_scaling_finetuning.md).

### 3.4 Housekeeping

**On JURECA** `modelstore/` now holds ~2.0 GB: the four `tiny`-class runs and
nothing else. The `large` directories below were never copied here, and the one
5.2 GB checkpoint that was has been deleted
([section 3](#3-the-withdrawn-large-model-and-what-replaced-it)).

**On JUPITER**, where the kit was built, `modelstore/` held ~102 GB, of which two
directories were almost all of it. This is the copy to go to if you want the
withdrawn model back:

* `large_pretrained_diverged_lr2e-4/` -- 47 GB, the failed `2e-4` run. Keep one
  checkpoint if you want the evidence; the other nine are worth nothing.
* `large_pretrained/` -- 78 GB, fifteen checkpoints, including step 75 000.

Plus a handful of rehearsal runs (`probe`, `t10_*`, `t12_*`, `docs_tiny`) that
are not in `SHIPPED_RUNS` and are nobody's dependency.

---

## 4. Publishing

`main` here is the development history and stays private. The public repository
sees **one squashed commit per release**, built from `main^{tree}`:

```bash
bash scripts/publish.sh --dry-run "Add the large pre-trained checkpoint"
bash scripts/publish.sh          "Add the large pre-trained checkpoint"
git push origin public:main
```

It refuses a dirty tree, the wrong branch and a tainted `public` branch, and
shows you the file list before doing anything. `git diff main public` is empty by
construction, so the public tree is exactly `main`'s tracked tree -- nothing
git-ignored can leak into it.

---

## 5. When somebody is stuck

The kit was rehearsed cold, end to end, before it shipped. Every expensive
failure that turned up was a *silence*, and each one now refuses out loud -- so the fastest move is
almost always **read the message**, which names the override that fixes it. The
ones worth recognising on sight:

| What they say | What happened | Answer |
|---|---|---|
| "it trained but there's no checkpoint" | `max_steps` below `save_step_frequency` | the startup guard now refuses this outright |
| "my override did nothing" | a bare `++max_steps=10` on a `make` command line is parsed as a make variable | `HYDRA_ARGS="++max_steps=10"` |
| "it says it's already training" | a lock, and now only ever a live one | `squeue -j <id>`, then delete it |
| "`make doctor` fails" | `make stats` has not been run, 8 minutes, once per clone | [docs/01 §1.1](01_setup.md) |
| "resume crashes on the optimizer state" | fixed -- torch 2.6 `weights_only` versus OmegaConf | none, but check they are on a current clone |
| "the loss went to NaN" | see [3.1](#31-the-learning-rate-is-not-a-free-knob) | lower the learning rate; the guard already stopped the run |
| "it's training on the login node" | no allocation | the startup guard pauses and prints the `srun` line |
| "shapes don't match after I changed `n_depths`" | the backbone needs exactly 8 latent levels, so every preset uses 13 depths | [docs/02](02_data_and_masking.md) |

`make doctor` answers most environment questions before you have to.
