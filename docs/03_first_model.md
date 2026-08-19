# 3. Your first model

[< the data](02_data_and_masking.md) | [next: bigger models >](04_scaling_finetuning.md)

**Time: about 31 minutes of GPU, plus 5 minutes of reading.**

By the end of this page you will have a trained checkpoint of your own, and you
will be able to tell from the logs whether it worked.

---

## 3.1 Get a GPU and run one command

```bash
srun --account=training2635 --partition=dc-gpu --gres=gpu:1 --ntasks=1 \
     --cpus-per-task=12 --time=01:00:00 --pty bash
export CUDA_VISIBLE_DEVICES=0
cd /p/scratch/training2635/4_ocean_ai/$USER/hackathon-ocean-sea-ice

make train-tiny NAME=my_first_run
```

That is it. **Do not skip `--ntasks=1` or `CUDA_VISIBLE_DEVICES=0`** -- see
[the three cluster traps](01_setup.md#three-cluster-traps-that-have-already-bitten-this-project).
And do not skip the `srun` itself: the login node has a real GPU that is not
yours, so the run would start, and every training command now warns loudly when
`SLURM_JOB_ID` is unset rather than letting you find out at the OOM.

`make train-tiny` expands to:

```
.venv/bin/python -m geoarches.main_hydra --config-path /path/to/repo/configs \
	cluster=jureca_1gpu module=tiny dataloader=glorys_tiny ++name=my_first_run
```

Check it yourself with `make -n train-tiny`, which prints the recipe without
running it.

Three parts of that line:

* **`--config-path` with an absolute path.** Never `--config-dir`. See
  [the cheatsheet](cheatsheet.md#the-training-command-in-full) for what goes
  wrong; it cost a 40-minute GPU run here.
* **`module=tiny`** -- the size preset: 13.5 M parameters, 4000 steps.
* **`dataloader=glorys_tiny`** -- five years of training data (2014-2018, 1825
  samples) and one of validation (2019, 364 samples), so the first batch arrives
  in seconds.

## 3.2 What you should see, in order

### The run plan, before anything is read

Every route into training -- `make train-tiny`, `make train`, the raw
`geoarches.main_hydra` command line, `oceanarches.main_multinode`, the notebooks
-- prints this first, before the data, the model or the GPU are touched:

```
==============================================================================
[oceanarches] run plan -- notebook_probe   (mode=train)
  config       module=tiny  dataloader=glorys_tiny  cluster=jureca_1gpu  forcing=none
  budget       max_steps 200 (command line), batch_size 8 (cluster=jureca_1gpu)
  checkpoints  every 100 steps -> 100, 200
               in modelstore/notebook_probe/checkpoints
  start        FRESH START -- modelstore/notebook_probe holds no checkpoint
  allocation   SLURM job 123456, 1 node(s), on jrc0001
==============================================================================
```

(That one is notebook 02's 200-step probe; `make train-tiny NAME=my_first_run`
says `max_steps 4000 (module=tiny)` and `every 1000 steps -> 1000, 2000, 3000,
4000`.)

Four lines worth reading every time:

* **`budget`** names where each number came from, so a preset swap that also
  moved `batch_size` cannot hide.
* **`checkpoints`** lists the steps that will actually be written. If the
  arithmetic gives none, the run is refused here rather than finishing and
  leaving nothing (see [3.8](#38-when-it-goes-wrong)).
* **`start`** is the answer to "did it resume or start over?", which
  `resume: True` never gave -- it printed either way. A re-run reads
  `RESUMING from checkpoint_global_step=1000.ckpt -- step 1000 of 4000, 3000 to
  go`. Note that **`resume: False` does not prevent a resume**: geoarches loads
  the newest checkpoint whenever `modelstore/<name>/checkpoints/` exists. The
  banner tells you which it is doing.
* **`allocation`** is `none (SLURM_JOB_ID unset) on jpbl-...` when you forgot the
  `srun`, and a warning with the `srun` line follows it.

Two more things the same startup check refuses, both measured on participants
rather than imagined: **reusing a run name under a different preset** (the stored
architecture would silently win -- [docs/04
4.4(a)](04_scaling_finetuning.md#a-continue-a-run-that-stopped)), and **launching
the same `NAME=` twice at once** (two terminals, one forgotten name; the second
one is told which host and pid holds `modelstore/<name>/.training.lock`).

### Start-up, about 60 seconds

```
wandb: W&B syncing is set to `offline` in this directory. ...
Working dir /p/scratch/.../hackathon-ocean-sea-ice
is main node True
registering exp on main node
```

Those three wandb lines are the entire cost of logging. `wandb_mode: offline` is
set in every cluster config, so nothing needs an account, an API key or the
network -- it writes to `wandblogs/` and that is all. **Do not turn logging off.**
geoarches writes `modelstore/<name>/config.yaml` only when `log` is on, and
without that file the checkpoint cannot be loaded again.

Then the model:

```
  | Name           | Type                      | Params | Mode
-----------------------------------------------------------------
0 | backbone       | ArchesWeatherCondBackbone | 13.0 M | train
1 | embedder       | OceanEncodeDecodeLayer    | 171 K  | train
...
13.5 M    Trainable params
```

**13.5 M parameters.** If you see a different number, you are not running `tiny`.

(Two counts for one model appear in this kit and both are right. Lightning prints
the **whole module**: 13.465 M, which includes the two timestep embedders that
carry the month-of-year and hour-of-day conditioning. `make benchmark` and
[docs/04](04_scaling_finetuning.md#41-the-four-presets) print **backbone +
embedder** only: 13.2 M. Compare like with like.)

Then Lightning checks that a validation step works before spending an hour on
training:

```
Sanity Checking DataLoader 0: 100%|##########| 2/2 [00:04<00:00,  0.47it/s]
```

### Training

```
Epoch 0:  44%|####3     | 100/229 [00:44<00:57,  2.23it/s, v_num=tiny, train_loss=94.20, ...]
```

**229 batches per epoch**, and that number is the single best check you have that
your allocation is what you think it is.

A **batch** is the group of samples the model sees in one step before its weights
are updated; `tiny` uses 8. An **epoch** is one pass over the whole training
split; `tiny_train` holds 1825 samples, and `ceil(1825 / 8) = 229`. A **step**
(what `max_steps` counts) is one batch, so 4000 steps is 4000 / 229 = 17.5
epochs.

**If you see 58 instead of 229**, SLURM gave you four GPUs and Lightning is
sharding your data four ways inside one process -- you are training on a quarter
of the dataset, and the loss curve looks completely normal.
[Fix: `export CUDA_VISIBLE_DEVICES=0`.](01_setup.md#three-cluster-traps-that-have-already-bitten-this-project)

(58 is *also* the right number for a deliberate four-rank DDP job with
`cluster=jureca_4gpu`, where the effective batch really is 32 and nothing is
lost -- `ceil(ceil(1825 / 4) / 8) = 58`. Same number, opposite meanings. The
thing that tells them apart is `SLURM_NTASKS`: 1 means you have the trap, 4 means
you have DDP. See
[docs/04](04_scaling_finetuning.md#what-four-gpus-actually-buy-measured).)

The rest of that line: 2.23 iterations (batches) per second, so 4000 steps is
about 30 minutes.

A checkpoint is written every 1000 steps:

```
saving checkpoint to modelstore/my_first_run/checkpoints/checkpoint_global_step=1000.ckpt
```

Four in total, the last of them the finished model. **`save_step_frequency` must
divide `max_steps`**, because nothing checkpoints at the end of `fit`. A budget
where it does not is refused before the first batch, with the arithmetic and a
value that does divide -- it used to train for real and leave nothing behind.

## 3.3 The measured wall clock

Timed on one dc-gpu A100, `bf16-mixed`, at the preset's batch size and 8
dataloader workers, `CUDA_VISIBLE_DEVICES=0`:

```
$ S=$(date +%s); make train-tiny NAME=docs_tiny; echo $(( $(date +%s) - S ))
1851
```

**1851 seconds = 30 minutes 51 seconds** for 4000 steps, including ~60 s of
start-up and 18 validation passes. Peak GPU memory is about 48.6 GiB of the 96 GB
card.

Timed twice more since, on other nodes: **1768 s** and **1753 s**. So the honest
range is **29 to 31 minutes**, and that spread -- 6% -- is what you should expect
between nodes.

That is the whole basis of the "30-minute model" claim, and it is a measurement,
not a target. It decomposes, if you want to know where it goes:

```
4000 steps x 0.2814 s  = 1126 s   the training steps themselves (measured)
                          +61 s   start-up
17.5 epochs x ~38 s    = +664 s   validation + the dataloader respawning
                         ------
                          1851 s
```

The last line is why `tiny` looks so much slower than `make benchmark` predicts:
`tiny_train` is only 229 batches, so 4000 steps is 17.5 epochs and it pays that
per-epoch cost seventeen times. A preset on the full `glorys` split has an epoch
5x longer and barely notices it -- see
[docs/04](04_scaling_finetuning.md#the-wall-clocks-what-is-measured-and-what-is-projected).

## 3.4 Did it work?

Three things to look at, in increasing order of usefulness.

### The training loss

The loss is normalised so that **a one-day persistence forecast scores about 1**
-- each error is divided by `delta_std`, the standard deviation of that
variable's one-day change. So the number is directly readable.

Measured, from the run above:

| step | train loss |
|---|---|
| untrained | ~3800 |
| 99 | 80.6 |
| 299 | 9.8 |
| 999 | 1.57 |
| 1999 | 0.935 |
| 2999 | 0.749 |
| 3999 | **0.724** |

Three orders of magnitude in the first 300 steps -- that is the model learning
that land is land and that the ocean is not random. The interesting part is the
last row.

**The line to beat is 0.82-0.87, and it moves with the split**, so always say
which split you measured on:

| split | 1-day persistence loss |
|---|---|
| `tiny_val` (2019), first 128 samples in order | 0.8141 |
| `tiny_val`, whole split (364) | 0.8718 |
| `tiny_train` (2014-2018), whole split (1825) | 0.8439 |
| `train` (1993-2018), whole split (9488) | 0.8204 |

So a final training loss of 0.724 against a `tiny_train` persistence of 0.8439 is
a model that has beaten persistence. Modestly.

### The validation metrics

These are in physical units, on data the model never trained on, and they are the
ones that matter. From the same run, at the end (epoch 17):

```
val_loss=0.785
val_rmse_thetao0m_24h=0.130    val_mae_thetao0m_24h=0.0834   val_acc_thetao0m_24h=0.985
val_rmse_siconc_24h=0.0107     val_bias_siconc_24h=-2.85e-5  val_acc_siconc_24h=0.982
val_rmse_zos_24h=0.0189        val_rmse_sithick_24h=0.0128   val_rmse_so0m_24h=0.0756
val_iiee_seaiceNH_24h=0.199    val_extentbias_seaiceNH_24h=0.0333
val_iiee_seaiceSH_24h=0.243    val_extentbias_seaiceSH_24h=0.0491
```

How the numbers moved over the run:

| after epoch | `val_loss` | `val_rmse_thetao0m` [degC] | `val_rmse_siconc` | `iiee` NH [10^6 km^2] |
|---|---|---|---|---|
| 0 | 23.02 | 0.647 | 0.0183 | 28.3 |
| 4 | 1.41 | 0.155 | 0.0114 | 0.226 |
| 9 | 0.933 | 0.145 | 0.0108 | 0.203 |
| 17 | **0.785** | **0.130** | **0.0107** | **0.199** |

The sea-ice numbers are the clearest sanity check in the list. An **untrained**
model puts ice over roughly 28 million km^2 of open water in each hemisphere. A
trained one has an ice-edge error of about 0.2 million km^2 at one day. For scale,
the whole Arctic September ice pack is 4-5 million km^2, so a day-1 edge error of
0.2 is about 4% of it: plausible for a forecast that starts from the truth, and
not so small that it looks like the answer leaked in.

**Do not read the 0.647 -> 0.130 degC drop as skill.** Most of it is the untrained
model disappearing. 1-day persistence on the same kind of samples is 0.126 degC.

### The honest comparison

Like for like, on the **first 128 samples of `tiny_val` in dataset order** (a
deterministic set, so it reproduces), both columns through the same metric code.
These come from the shipped `modelstore/task6_tiny`, which is the same preset
trained for 4500 steps rather than 4000, so treat them as the neighbourhood your
own run will land in rather than as numbers to reproduce exactly:

| metric | trained `tiny` | 1-day persistence | better by |
|---|---|---|---|
| loss | 0.7095 | 0.8141 | **12.8%** |
| `rmse_thetao0m` (SST) | 0.1214 degC | 0.1260 degC | 3.7% |
| `rmse_siconc` | 0.01116 | 0.01281 | 12.9% |
| `rmse_zos` | 0.01851 m | 0.02228 m | 16.9% |
| `rmse_sithick` | 0.01149 m | 0.01235 m | 7.0% |
| `rmse_so0m` | 0.07109 | 0.07166 | 0.8% |
| `iiee` NH | 0.2437 | 0.2777 | 12.2% |
| `iiee` SH | 0.1803 | 0.2297 | 21.5% |
| **`extentbias` NH** | **+0.0485** | **+0.0106** | **worse** |
| **`rmse_so` at 1684 m** | **0.001321** | **0.001181** | **worse** |

**A 31-minute model beats one-day persistence by about 4% on sea surface
temperature, 13% on sea-ice concentration and 17% on sea surface height, and is
worse than persistence on Northern-Hemisphere extent bias and on deep salinity.**

That is a respectable day-1 result for half an hour on one GPU. It is also
exactly the "bad, but improving it is worth doing" the preset is for, and the two
losses are the most useful things on the page -- see
[docs/06](06_evaluation.md#the-two-places-it-loses-which-you-should-go-after).

Note the `val_rmse_thetao0m` of 0.130 in the log and the 0.1214 in this table are
the same model on different samples: geoarches builds the validation loader with
`shuffle=True` and reloads it every epoch, so "the 128 validation samples" is not
a reproducible set. **Quote the deterministic numbers**, which is what
`make eval` produces.

## 3.5 What you now have on disk

```
modelstore/my_first_run/
  config.yaml                                  <- load_module needs this
  checkpoints/checkpoint_global_step=1000.ckpt
  checkpoints/checkpoint_global_step=2000.ckpt
  checkpoints/checkpoint_global_step=3000.ckpt
  checkpoints/checkpoint_global_step=4000.ckpt <- the finished model, 164 MB
```

Load it back:

```bash
.venv/bin/python -c "
from geoarches.lightning_modules.base_module import load_module
m, cfg = load_module('modelstore/my_first_run')
print(type(m).__name__, m.component.name, sum(p.numel() for p in m.parameters()) / 1e6, 'M params')
"
```

```
Restored from modelstore/my_first_run/checkpoints/checkpoint_global_step=4000.ckpt
OceanForecastModule full 13.465088 M params
```

You will also find `outputs/<date>/<time>/` (hydra's record of the run, including
the fully resolved config -- occasionally very useful) and `wandblogs/` (the
offline metric log). Both accumulate; delete old ones freely.

## 3.6 Now score it properly

```bash
make eval NAME=my_first_run LEAD_DAYS=10
```

**231.9 s and 246.3 s on two cold runs** (`task6_tiny`, `--lead-days 10`), two
thirds of it rendering the six animations -- and **about 5 s** on a warm one,
because the rollout, the figures and the animations are all cached. It produces
10 figures, 6 animations, `report.md` and a self-contained `report.html`, with
every number sitting next to persistence and climatology, and it leaves about
750 MB in `evalstore/<run>/`.
[What all of it means.](06_evaluation.md)

Here is what the run timed in [3.3](#33-the-measured-wall-clock) -- 4000 steps,
the shipped preset, nothing tuned -- actually scored on the `test` split
(2021-2023), 16 initialisations, straight out of its own `report.md`:

| variable | unit | day 1 model | day 1 persistence | model vs persistence |
|---|---|---|---|---|
| sea surface temperature @ 0 m | degC | 0.12731 | 0.13093 | -2.8% (better) |
| sea ice concentration | 1 | 0.010701 | 0.012685 | -15.6% (better) |
| sea surface height | m | 0.018384 | 0.021653 | -15.1% (better) |
| sea ice thickness | m | 0.012504 | 0.013592 | -8.0% (better) |
| sea water salinity @ 0 m | 1e-3 | 0.074953 | 0.075464 | -0.7% (better) |
| mixed layer depth | m | 10.413 | 11.009 | -5.4% (better) |

and the module loss over the whole 10-day rollout: **model 2.1886, persistence
2.2605, climatology 48.7485**.

**Your numbers will differ from these**, and that is normal: `bf16` arithmetic
and dataloader shuffling are not bit-reproducible run to run, so two runs of the
identical command give two different models.

Quote the tolerance on the **RMSEs**, not on the margins. Measured across three
independently trained `tiny` checkpoints, day-1 SST RMSE came out at 0.1249,
0.1273 and 0.1290 degC -- a spread of about 3% -- while the *margin* over
persistence moved from -4.6% to -2.8%, which is a 60% change in a number that
looks like the headline. A second run from a fresh clone got -9.4% on
`zos` against the -15.1% recorded above.

So: an RMSE more than about 5% away from these, or a margin with the **wrong
sign**, means something really did change. A margin that is half or double is
just a different run.

## 3.7 What is actually happening inside

Worth knowing before you start changing things.

**The model predicts the change, not the state.** `add_input_state: True` means
the network's output is a *tendency* which is added to the input state. Sea
surface temperature moves about 0.1 degC per day against an 11.7 degC spatial
spread, so predicting the state directly would be an enormous scale for a tiny
signal. It also means **the model gets persistence for free**, which is why a
variable it makes *worse* than persistence is capacity spent badly.

**The forward pass** is `encode -> backbone -> decode -> add input state -> clamp
-> mask`, in that order:

* *clamp* forces `siconc` into [0, 1] and `sithick`, `mlotst` and `so` to be
  `>= 0`, in **normalised** space so an out-of-range prediction gets zero
  gradient rather than one pushing it further out. `thetao` deliberately has no
  bound: clamping it would hide the instability in
  [docs/06 6.5](06_evaluation.md#65-the-90-day-free-run-and-what-it-tells-you)
  rather than fix it;
* *mask* comes last, because a clamped land cell is no longer 0 and the whole
  framework treats land as exactly 0.

**The loss** is a weighted mean over variables, each averaged over its depths and
over ocean cells with `cos(lat)` weighting, with each error divided by that
channel's `delta_std`. Land contributes exactly nothing -- verified by
overwriting land in both the prediction and the target and confirming the loss is
bit-for-bit identical.

**The model also sees the previous day** -- two states concatenated, which is
`embedder.n_concatenated_states: 1` in the preset and `dataset.load_prev: True`
in the dataloader config (both, or neither) -- and the month of the year and the
hour of the day as conditioning.

The code is
[`oceanarches/lightning_modules/ocean_forecast.py`](../oceanarches/lightning_modules/ocean_forecast.py),
and it is written to be read.

## 3.8 When it goes wrong

| symptom | cause |
|---|---|
| 58 batches per epoch, not 229 | you got four GPUs; `export CUDA_VISIBLE_DEVICES=0` |
| trains far past 4000 steps | `--config-dir` instead of `--config-path`; use the `make` target |
| no checkpoint at the end | `save_step_frequency` does not divide `max_steps` -- refused up front now, so only an interrupted run can still do this |
| `FileNotFoundError: config.yaml` when loading | the run used `log=False` |
| loss is `nan` | you changed the masking or the fill -- [docs/02](02_data_and_masking.md#212-the-masking-order-and-why-it-is-not-negotiable) |
| loss stuck near 1 | the model has learned persistence and nothing more |
| `ValueError: Path does not exist` | `make doctor` |
| your hydra override did nothing | a bare `++x=y` on a `make` line is a make variable -- `make` now refuses that command line and names `HYDRA_ARGS="++x=y"`. A `++` typo that creates a top-level key (`++max_step=10`) or shadows a real one (`++module.lr=`) is refused at startup too; prefer a bare `max_steps=10`, which errors when the key is absent |
| `refusing to start (nothing to train)` | you reused the name of a run already at or past `max_steps`; it would have exited 0 having trained nothing. Pick a fresh `NAME=`, a bigger `++max_steps`, or `make eval` |
| **SLURM killed the job before it finished** | **re-run the identical command.** geoarches resumes: "Experiment already exists. Trying to resume it." and restarts from the newest checkpoint, keeping the optimiser and the schedule. Verified after a `scancel` at step 800. [docs/04 4.4(a)](04_scaling_finetuning.md#a-continue-a-run-that-stopped) |
| `torch.OutOfMemoryError: CUDA out of memory` | you raised `emb_dim`, `batch_size` or `rollout_iterations`; `tiny` already peaks at 42.6 GiB of 96. The kit now prints its own note under PyTorch's traceback naming `batch_size` -- PyTorch's `expandable_segments` suggestion is about fragmentation and will not fit a batch that does not fit. `HYDRA_ARGS="++batch_size=4"` first, then `++module.backbone.gradient_checkpointing=True`; `make benchmark` prints every preset's peak |
| `--exp <run>: no such run` from `make eval` | `NAME` defaults to `tiny` on **both** `train-tiny` and `eval`, so a run you named something else needs `make eval NAME=my_first_run`. The error lists every run in `modelstore/`. |

---

[< the data](02_data_and_masking.md) | [next: bigger models >](04_scaling_finetuning.md)
