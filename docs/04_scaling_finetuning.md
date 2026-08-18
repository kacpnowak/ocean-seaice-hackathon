# 4. Bigger models, and fine-tuning

[< your first model](03_first_model.md) | [next: coupling >](05_coupling.md)

You have a `tiny` model. It is bad on purpose. This page is about what to do
next: make it bigger, train it longer, or start from someone else's weights.

Read time: 10 minutes.

---

## 4.1 The four presets

All four are the **same architecture** -- an ArchesWeather encoder, backbone and
decoder from geoarches. Two numbers say how big one is:

* `emb_dim` -- how wide a token is;
* `depth_multiplier` -- how many attention blocks.

**But `emb_dim` does not move alone.** It appears in four places that have to
agree, and overriding only `module.backbone.emb_dim` used to instantiate happily
and then die in the first forward pass with `RuntimeError: mat1 and mat2 shapes
cannot be multiplied (7200x768 and 1536x1536)` -- after the dataloader had spun
up, on a GPU you had queued for. `OceanForecastModule` now refuses to build and
says which numbers disagree, but you still have to move them yourself:

| preset | `backbone.emb_dim` | `backbone.num_heads` | `embedder.emb_dim` | `embedder.out_emb_dim` |
|---|---|---|---|---|
| `tiny` | 96 | `[3, 6, 6, 3]` | 96 | 192 |
| `small` | 192 | `[6, 12, 12, 6]` | 192 | 384 |
| `base` | 192 | `[6, 12, 12, 6]` | 192 | 384 |
| `large` | 384 | `[12, 24, 24, 12]` | 384 | 768 |

The rules, which `tests/test_configs.py::test_backbone_and_embedder_agree` pins
for the shipped presets and the module re-checks on every override:
`embedder.emb_dim == backbone.emb_dim`; `embedder.out_emb_dim == 2 *
backbone.emb_dim` (twice, because `use_skip: True` concatenates the skip
connection); and each entry of `num_heads` must divide the features that stage
attends over, which are `emb_dim, 2*emb_dim, 2*emb_dim, out_emb_dim`. So scaling
by hand is four overrides, not one:

```bash
make train-tiny NAME=wider HYDRA_ARGS="++module.backbone.emb_dim=192 \
    ++module.backbone.num_heads=[6,12,12,6] \
    ++module.embedder.emb_dim=192 ++module.embedder.out_emb_dim=384"
```

or just `make train MODULE=small`, which is that plus a step budget.

| preset | `emb_dim` | `depth_mult` | parameters | ms per sample | batch | peak GPU | `max_steps` | wall clock |
|---|---|---|---|---|---|---|---|---|
| `tiny` | 96 | 1 | **13.2 M** | 40.2 | 8 | 42.6 GiB | 4000 | **31 min** (measured) |
| `small` | 192 | 1 | **45.0 M** | 75.5 | 4 | 42.2 GiB | 40000 | ~3.6 h (projected) |
| `base` | 192 | 2 | **84.6 M** | 163.8 | 2 | 41.4 GiB | 110000 | ~10.8 h (projected) |
| `large` | 384 | 3 | **459.6 M** | 542.3 | 1 | 50.8 GiB | 300000 | ~50 h (projected) |

Measured on one GH200 in `bf16-mixed`, median of 12 steps after 5 warm-up steps,
by `make benchmark`.

**The parameter column is backbone + embedder.** Lightning's own summary during
training prints the whole module and is therefore larger -- 13.465 M for `tiny`
against the 13.2 M here -- because it also counts the two timestep embedders that
carry the month and hour conditioning. Neither is wrong; just do not compare one
against the other.

### The wall clocks: what is measured and what is projected

**Only `tiny`'s wall clock is a stopwatch measurement** (1851 s for the whole
`make train-tiny`, [see docs/03](03_first_model.md#33-the-measured-wall-clock)).
Nobody has ever run `small`, `base` or `large` to completion, so those three
numbers are projections and this section says exactly what they are projected
from.

What *has* been measured for all four is the **real training-step time**: a real
run on the real archive with the metrics on, timed at 60 and at 260 steps so the
slope cancels start-up, all four on `dataloader=glorys` so that none of them
crosses an epoch boundary inside the window. One GH200, `bf16-mixed`:

| preset | measured s/step | measured ms/sample | `make benchmark` ms/sample | ratio | `max_steps` x s/step |
|---|---|---|---|---|---|
| `tiny` | 0.2814 | 35.2 | 40.2 | 0.87 | 0.31 h |
| `small` | 0.3105 | 77.6 | 75.5 | 1.03 | **3.45 h** |
| `base` | 0.3455 | 172.7 | 163.8 | 1.05 | **10.6 h** |
| `large` | 0.5909 | 590.9 | 542.3 | 1.09 | **49.2 h** |

An earlier version of this page told you to multiply the projections by ~1.4,
because `make benchmark` projects `tiny` at 22 minutes and the stopwatch says 31.
**That advice was wrong and is withdrawn.** The step time itself needs a factor
of 1.03-1.09, not 1.4. `tiny`'s missing nine minutes are *epoch boundaries*:

```
4000 steps x 0.2814 s      = 1126 s   training
                             +61 s    start-up (measured intercept)
17.5 epochs x ~38 s        = +664 s   validation + dataloader respawn
                            ------
                             1851 s   which is the stopwatch number
```

`tiny` runs on `glorys_tiny`, whose epoch is only 229 batches, so 4000 steps is
17.5 epochs of it. The other three run on `glorys`, whose epoch is 5x longer
(2372, 4744 and 9488 batches at their batch sizes), so they pay that overhead
fewer times relative to a much longer run:

| preset | epochs | epoch overhead at ~38 s | steps x s/step | total | overhead as a share |
|---|---|---|---|---|---|
| `small` | 16.9 | 0.18 h | 3.45 h | **~3.6 h** | 5.2% |
| `base` | 23.2 | 0.25 h | 10.56 h | **~10.8 h** | 2.3% |
| `large` | 31.6 | 0.33 h | 49.24 h | **~50 h** | 0.7% |

(An earlier draft of this section called that overhead "under 1%" for all three.
It is only under 1% for `large`.)

Two caveats that keep these honest. The per-epoch cost above (~38 s) is `tiny`'s
and has not been measured for the others -- their validation loops are longer
(`large` uses `limit_val_batches: 64`) and their batches are smaller. And a
projection is not a measurement: if you are the first person to run `base` to
the end, please write down what it actually took.

Re-running `make benchmark` on a different node reproduced the table to within
3% (41.0 / 78.0 / 160.9 / 551.9 ms per sample, parameters and peak memory
identical), which is about the noise you should expect.

### What four GPUs actually buy, measured

`base` and `large` want more than one GPU, and `cluster=jupiter_4gpu` gives you a
whole node. **It does not divide the wall clock by four.** An earlier version of
this page said it did; that was wrong, and here is the measurement that settles
it -- `tiny` on `glorys_tiny`, one booster node, the same commit, 100 and 500
steps so that the slope cancels start-up:

| | batches per epoch | s per optimiser step | samples/s | start-up |
|---|---|---|---|---|
| 1 GPU (`jupiter_1gpu`) | 229 | 0.4300 | 18.6 | 17 s |
| 4 GPU (`jupiter_4gpu`, DDP) | **58** | 0.5717 | 56.0 | 78 s |

`max_steps` counts **optimiser** steps, and on four GPUs one optimiser step is
still one forward and backward plus an all-reduce -- so a step got **1.33x
slower**, not four times faster. What the other three cards buy is a four times
larger effective batch (4 x `batch_size` = 32 samples instead of 8), i.e.
**3.01x the samples per second**, which is 75% of perfect scaling. The missing
25% is the gradient all-reduce and four times as many epoch boundaries.

So: **four GPUs let you train on four times as much data in the same wall clock,
or reach the same number of samples in a third of the time. They do not make a
fixed `max_steps` finish sooner -- they make it finish slightly later.** Halve
`max_steps` when you move a recipe from one GPU to four if you want the same
number of samples seen.

**58 batches per epoch is the correct number for four ranks**, and it is worth
knowing why, because 58 is also the symptom of the cluster trap in
[docs/01](01_setup.md#three-cluster-traps-that-have-already-bitten-this-project).
`tiny_train` holds 1825 samples; Lightning shards them across ranks
(`ceil(1825 / 4) = 457` each) and each rank makes batches of 8, so
`ceil(457 / 8) = 58`. On a genuine four-rank job that is right and the effective
batch is 32. On a **one**-process job that SLURM handed four GPUs to, you see the
same 58 -- and there the effective batch is still 8, so you are training on a
quarter of the data. Same number, opposite meanings: check `SLURM_NTASKS`, not
just the batch count.

### And what sixteen GPUs buy, measured on `large`

The table above is `tiny` on one node. The preset that actually needs the
hardware is `large`, and it has been measured directly rather than extrapolated
-- `dataloader=glorys`, metrics on, 60 and 260 steps so the slope cancels
start-up, jobs 1285596 (one node) and 1285595 (four):

| | batches per epoch | s per optimiser step | samples/s | start-up |
|---|---|---|---|---|
| 1 GPU (`jupiter_1gpu`) | 9488 | 0.5909 | 1.69 | not recorded |
| 4 GPU, 1 node (`jupiter_4gpu`) | 2372 | 0.7041 | 5.68 | 38 s |
| 16 GPU, 4 nodes (`jupiter_4nodes`) | **593** | 0.7694 | 20.80 | 41 s |

**`large` scales far better than `tiny` did**, and the reason is the thing that
makes it expensive: one step is 0.59 s of compute on a 459.6 M-parameter model,
which is enough to hide most of a 1.8 GB gradient all-reduce behind it. Four
GPUs cost 1.19x per step for 3.36x the throughput (84% of perfect); sixteen cost
1.30x for 12.29x (77%). **Crossing the network -- four GPUs to sixteen -- costs
9.3% per step and returns 3.66x the samples, which is 91.5% of perfect.** The
1.33x measured on `tiny` was the pessimistic end of the range, not the rule.

Two caveats on those figures. Each is one allocation, and **the second four-node
allocation was slower, not faster**: job 1285787, on different nodes, spent
158.29 s on 200 extra steps plus one epoch boundary, so at most 0.7914 s/step.
Read 0.7694 as the optimistic end of a +-5% band rather than as four significant
figures. And start-up depends on the page cache -- the first `large` invocation
in a fresh allocation spent 73 s inside the epoch bar before its first optimiser
step, against 4 s once the archive was warm.

### Sixteen GPUs are worth it only if `max_steps` moves too

`max_steps` counts optimiser steps and one step is one sample per GPU for
`large`, so keeping it fixed means four times the data *and* more wall clock:

| the 16-GPU plan | samples | epochs of the 9488-sample split | wall clock |
|---|---|---|---|
| keep `max_steps: 300000` | 4.8 M | 506 | 64.1 h |
| hold samples seen constant, `max_steps: 75000` | 1.2 M | 126 | **16.0 h** |
| *(for comparison: 4 GPUs, `max_steps: 300000`)* | 1.2 M | 126 | 58.7 h |

[`scripts/pretrain_large.slurm`](../scripts/pretrain_large.slurm) ships the
middle row, and does it by deriving `max_steps` from a **sample** budget and the
size of the allocation rather than hardcoding a step count, so submitting it at
`--nodes=1` still runs the original 300000 steps. Five hundred epochs of a
9488-sample archive is not four times the model for four times the wait; the
same data 3.66x sooner is what the extra nodes are for. `num_training_steps:
${max_steps}` in the preset means the cosine schedule follows `max_steps`, so
the shorter run is a complete schedule and not a truncated one.

**The learning rate is deliberately left at `2e-4`.** The effective batch goes
from 4 to 16, and the usual rules would put it at 4e-4 (square root) or 8e-4
(linear); configs/module/large.yaml explains why it was not raised without a run
to justify it, and gives the override.

```bash
make train MODULE=small NAME=my_small_run
make train MODULE=base  NAME=my_base_run  CLUSTER=jupiter_4gpu
```

Reproduce the table yourself:

```bash
make benchmark
```

That runs a real forward, backward and optimiser step for every preset on the
real data, and prints parameters, step time, peak memory and the projected wall
clock. It takes a few minutes on a GPU.

### Reading the table without being misled

`s/step` and `peak GiB` are **not comparable across rows**, because each preset
is benchmarked at the batch size that fits it. That is why the table above
quotes **ms per sample**, which is comparable, and it is monotone in model size
as it must be: 40.2 < 75.5 < 163.8 < 542.3.

Peak memory is very nearly linear in `batch x emb_dim x depth_multiplier`. The
batch sizes were chosen to sit near 45 GiB -- under half of the 96 GB card -- so
that a long run survives fragmentation, the validation loop and DDP buffers.
`large` additionally turns on gradient checkpointing, which is why it breaks the
linear rule.

## 4.2 The vertical is not a scaling axis

This surprises everyone, so it is worth a section.

Every preset loads the **same 13 depth levels**. You cannot make a model "deeper
in the ocean" by giving it more levels, and here is why.

geoarches' backbone hardcodes **8 latent vertical positions** in three places:
`LinVert`, the axial attention's positional embedding, and the final reshape in
`ArchesWeatherCondBackbone.forward`. The embedder patches depth with a patch size
of 2 and adds one surface token, so

```
1 surface token + ceil_to_even(n_depths) / 2  ==  8
```

which admits **12 or 13 depth levels and nothing else**. Note that 14 does *not*
work: geoarches pads by a whole patch when the depth count is already even, so 14
levels give a latent depth of 9, not 8.

Why it matters: `LinVert` and the axial attention are the **only** places the
model mixes information between depth levels -- the attention window is
`[1, 6, 10]`, so there is no attention across depth, and the down- and
up-sampling stages act only on latitude and longitude. Turn those two off and
the water column cannot communicate at all, which for an ocean model throws away
most of the physics.

An earlier version of this kit shipped 6/10/14-level presets with those two
layers disabled, and a probe found the damage: perturbing the input at depth 0
moved depths 4-12 by 3.17 with the mixing on, and by **exactly 0.0** with it off.

So the guard is loud. Change `DEPTH_PRESETS` in `variables.py` to something else
and you get:

```
10 depth levels give a latent depth of 7, but geoarches' ArchesWeather backbone
only works at 8. ...
Fix: pick a depth count from [12, 13] ...
```

If you genuinely want another latent depth -- for instance because you have
swapped the backbone -- the escape hatch is
`OceanEncodeDecodeLayer(..., allow_any_z_dim=True)` together with
`first_interaction_layer: null` and `axis_attn: false`. The one shipped
configuration that needs it is `seaice_isolated`, which has no 3-D variables at
all.

Scale by `emb_dim` and `depth_multiplier` instead. Because the latent depth is
pinned, the backbone's sequence length is identical for every preset, so
carrying 13 levels in `tiny` costs almost nothing.

## 4.3 Just train `tiny` for longer

The cheapest improvement available, and the first one to try.

```bash
.venv/bin/python -m geoarches.main_hydra --config-path $PWD/configs \
    cluster=jupiter_1gpu module=tiny dataloader=glorys_tiny \
    ++name=tiny_long ++max_steps=16000 ++save_step_frequency=4000
```

`tiny` at 4000 steps is only about 17.5 passes over the 1825-sample `tiny_train`
split, and its validation loss had flattened but not obviously overfitted. Four
times the steps is about two hours.

**`save_step_frequency` must divide `max_steps`.** geoarches checkpoints on
`global_step % save_step_frequency == 0` and nothing checkpoints at the end of
`fit`, so if the two do not divide, **the finished model is the one thing that
never reaches disk**. This bit the shipped `base` preset (110000 steps, every
25000) and there is now a test for it -- and a startup guard, which refuses the
run before the GPU time is spent and prints a `++save_step_frequency=` that does
divide. A participant lost five short runs to this in its silent form.

The other obvious lever is more data: `dataloader=glorys` gives you 1993-2018
(9488 samples) instead of 2014-2018 (1825).

```bash
make train MODULE=tiny DATALOADER=glorys NAME=tiny_alldata
```

## 4.4 Fine-tuning from an existing checkpoint

Two different things get called "fine-tuning". Be clear which you want.

### (a) Continue a run that stopped

geoarches does this automatically. If `modelstore/<name>/checkpoints/` already
exists, launching the same name **resumes** from the newest checkpoint, with the
optimiser state and the step counter intact.

```bash
make train-tiny NAME=my_run          # crashes at step 2500
make train-tiny NAME=my_run          # picks up at step 2500
```

Measured, by killing a run for real rather than assuming: a `tiny` run stopped
263 s in had written `checkpoint_global_step=250` and `=500`, and the identical
command relaunched printed

```
Experiment already exists. Trying to resume it.
Found checkpoints [.../checkpoint_global_step=250.ckpt, .../checkpoint_global_step=500.ckpt]
Using checkpoint modelstore/my_run/checkpoints/checkpoint_global_step=500.ckpt
Restored all states from the checkpoint at .../checkpoint_global_step=500.ckpt
```

and carried on from step 500, not from 0.

> **This was broken until Task 10 and nothing had noticed**, because no test and
> no earlier task had ever killed a run and relaunched it. Since torch 2.6
> `torch.load` defaults to `weights_only=True`, Lightning's resume path takes
> that default, and every checkpoint here carries the hydra config -- so the
> relaunch died before the first batch with
> `UnpicklingError: Unsupported global: GLOBAL omegaconf.listconfig.ListConfig`.
> Fresh runs and `load_ckpt` were unaffected, which is why it stayed hidden.
> `oceanarches.lightning_modules` allowlists the OmegaConf container classes at
> import, and `plain_betas` stops new checkpoints containing any OmegaConf at
> all -- the real source was `optimizer_states[...]["betas"]`, a `ListConfig` from
> `betas: [0.9, 0.98]` in the preset. The regression tests are
> `tests/test_module.py::test_the_optimiser_state_carries_no_omegaconf`,
> `::test_a_checkpoint_holding_omegaconf_still_resumes` and
> `::test_the_allowlist_still_refuses_code_execution`.
>
> **If you hit that `UnpicklingError` in your own script, the import you need is
> `import oceanarches.lightning_modules`, not `import oceanarches`.** The
> allowlist lives in the subpackage, not in the top-level `__init__`, so the
> shorter one does not do it -- checked both ways.

Two things to know about resuming:

* geoarches replaces `cfg.module` and `cfg.dataloader` with the ones stored in
  `modelstore/<name>/config.yaml`, and only command-line overrides written with
  a leading `+` or `++` survive. So `++max_steps=8000` works and
  `max_steps=8000` is silently dropped.
* **A `module=` or `dataloader=` swap onto a name that already has checkpoints is
  therefore undone**, and used to be undone in silence: `NAME=baseline0
  MODULE=small` kept training the `tiny` network -- 13.5 M parameters where
  `small` is 45 M -- and overwrote the old checkpoints in place. That is now
  **refused at startup**, with the keys that differ printed and two ways out: a
  name of its own, or `mv modelstore/baseline0 modelstore/baseline0.old`.
  `++resume=False` is not one of them: geoarches loads the newest checkpoint in
  that directory either way.
* **One name, one running job.** The first process to get past the startup checks
  writes `modelstore/<name>/.training.lock`; a second launch of the same name is
  refused and prints the host, pid and SLURM job id holding it -- and how it
  decided the holder is still alive. A lock is taken over automatically when its
  SLURM job has left the queue (which is what a job killed at its wall clock
  leaves behind, so an ordinary relaunch is never blocked by one), or when its pid
  on this host is gone. Only where neither is knowable -- another host, no SLURM
  -- does it fall back to believing the lock for a day, and there you delete it
  (`rm modelstore/<name>/.training.lock`) once you are sure the job is gone.
* The learning-rate schedule was built for the *original* `max_steps`. Extend a
  finished run and you continue at the cosine schedule's floor.

### (b) Start from someone else's weights and train on something new

This is the interesting one, and it is the mechanism behind most of the
[challenge ideas](07_challenge_ideas.md). geoarches supports it with `load_ckpt`,
which loads the weights and **does not** resume the run -- fresh optimiser, fresh
step counter, fresh schedule:

```bash
.venv/bin/python -m geoarches.main_hydra --config-path $PWD/configs \
    cluster=jupiter_1gpu module=tiny dataloader=glorys \
    ++name=ft_probe +load_ckpt=modelstore/task6_tiny \
    ++max_steps=2000 ++save_step_frequency=500 ++module.module.lr=1e-4
```

Measured, with `++max_steps=20 ++save_step_frequency=10` so it finishes quickly
-- `tiny` ships `save_step_frequency: 1000`, and a 20-step run against that saves
nothing and is now refused. The proof that the weights really landed is the
**first logged training loss**:

```
train_loss=0.744
```

An untrained model starts at about **3800** on this loss. 0.744 is where
`task6_tiny` finished, so the checkpoint was loaded and the run continued from
it. Afterwards `modelstore/ft_probe/` holds a fresh `config.yaml` and
`checkpoint_global_step=20.ckpt` -- a new run, not a continuation of the old one.

Note the lower learning rate. Fine-tuning at the pre-training rate throws the
pre-trained weights away in the first few hundred steps.

**The state dict must match exactly.** `load_ckpt` calls `load_state_dict` in
strict mode, so you can only load a checkpoint into a model with the same
preset, the same component and the same depth count. You cannot load `tiny` into
`small`, and you cannot load a `full` model into a `seaice` specialist. If you
want that, you have to copy the weights layer by layer yourself.

### The two SLURM scripts

Everything above is a command line you type inside an `srun`. For the two long
jobs there are batch scripts, so you can submit them and go away.

**On JURECA, the `large` line below does not run.** `large` was measured at
50.81 GiB at batch 1 with `gradient_checkpointing: True` already on, and a dc-gpu
A100 has 39.5 GiB. It dies on its first optimiser step with an out-of-memory
error, and the guard now says so in as many words rather than advising you to
halve a batch that is already 1. Four GPUs do not fix it either: Lightning runs
DDP, which replicates the whole model on every rank, so per-rank memory is
unchanged. Sharding (FSDP/ZeRO-3) would recover only a few GiB, because what does
not fit is the activations and those are per-rank whatever you shard.

**`large_pretrained` is still fully usable on one A100 -- for inference.**
`make eval NAME=large_pretrained` has no optimiser state and no stored
activations and fits comfortably; that is tested. Fine-tune `base` or smaller, or
find a bigger card.

```bash
# THE MAIN PATH ON A 96 GB GH200 -- see the note above, it OOMs on a 40 GiB A100.
# MODULE=large is not optional -- it must be the preset FROM was trained with.
sbatch --export=ALL,FROM=large_pretrained,MODULE=large,NAME=my_finetune \
    scripts/finetune.slurm

# the same thing against the 31-minute model, if you want a quick loop first
sbatch --export=ALL,FROM=task6_tiny,NAME=my_finetune_tiny scripts/finetune.slurm

# what the ORGANISERS ran before the hackathon: `large` on 4 nodes x 4 GH200
sbatch scripts/pretrain_large.slurm
```

**`large_pretrained` is the checkpoint this challenge is built around.** It is
75 000 steps of `module=large` at `lr 1e-4`, trained on 1993--2018 across 16
GH200s, and on the held-out test years it beats persistence on *every* variable
at *every* lead time out to 10 days -- 36% better on day-1 SST, 53% on sea
surface height, 35% on sea-ice concentration -- and stays better than
climatology for 23 days on SST and 29 on sea-ice concentration.
[docs/06](06_evaluation.md) explains what those numbers mean and how to produce
them for your own model; `evalstore/large_pretrained/report.md` in the shared
store is its full scorecard.

Fine-tune it rather than training from scratch: you have a day, and the
pre-training was 18 hours on 16 GPUs.

[`scripts/finetune.slurm`](../scripts/finetune.slurm) is (b) above with the
mistakes made impossible. Every knob is an environment variable with a default
(`FROM`, `NAME`, `MODULE`, `DATALOADER`, `MAX_STEPS`, `SAVE_EVERY`, `LR`), it
exports `CUDA_VISIBLE_DEVICES=0`, it prints the resolved config before spending
any GPU time on it, and it refuses to start if

* `NAME == FROM` -- which would *resume* the pre-trained run and overwrite the
  checkpoint everybody else is starting from, rather than fine-tune from it;
* `SAVE_EVERY` does not divide `MAX_STEPS`, so the finished model would be the
  one checkpoint never written;
* `modelstore/$FROM/checkpoints/` does not exist.

Afterwards it re-reads the source checkpoints and says whether they changed.
Measured, `FROM=task6_tiny MAX_STEPS=200 DATALOADER=glorys LR=1e-4`:

```
$ sacct -j 1284009 --format=JobName,State,Elapsed
ocean-finetune  COMPLETED  00:02:05

Epoch 0:  50%|#####  | 100/200 [00:43<00:43, 2.31it/s, train_loss=0.722, ...]
saving checkpoint to modelstore/t10_ft_sbatch/checkpoints/checkpoint_global_step=100.ckpt
saving checkpoint to modelstore/t10_ft_sbatch/checkpoints/checkpoint_global_step=200.ckpt
pre-trained checkpoints under modelstore/task6_tiny are untouched -- OK
```

**0.722 is the proof.** The same preset started from scratch, on the same
dataloader, logs `train_loss=2.35e+3` after twenty steps. Three orders of
magnitude is not a tuning difference; the weights landed.

[`scripts/pretrain_large.slurm`](../scripts/pretrain_large.slurm) is the other
end: `--nodes=4 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=72`,
`cluster=jupiter_4nodes`, sixteen GH200 in one DDP job. Three things about it are
worth knowing even if you never run it.

**It does not launch `geoarches.main_hydra`, and it cannot.** geoarches builds
its trainer as `L.Trainer(devices="auto", accelerator="auto", strategy=...)` with
no `num_nodes`; Lightning defaults that to 1, and on more than one node every
rank stops before the first batch:

```
ValueError: You set `num_nodes=1` in Lightning, but the number of nodes
configured in SLURM `--nodes=4` does not match. HINT: Set `num_nodes=4`.
```

[`oceanarches/main_multinode.py`](../oceanarches/main_multinode.py) is the
answer, and it is about fifty lines of code: it composes the same hydra config,
hands `L.Trainer` the `num_nodes` from `cluster.num_nodes` (or, when the cluster
config does not say, from the allocation), and then calls geoarches' own `main`
with the composed config. Nothing is forked and nothing is copied -- geoarches is
installed non-editable from a pinned commit and this project has never vendored
any of it -- so resume, the checkpoint-every-N-steps callback and the
`modelstore/<name>/config.yaml` dump are all still geoarches' code. At one node
it changes nothing. Use it wherever you would have used `geoarches.main_hydra`
on more than one node:

```bash
srun --nodes=4 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=72 \
    .venv/bin/python -m oceanarches.main_multinode --config-path $PWD/configs \
    cluster=jupiter_4nodes module=large dataloader=glorys ++name=my_run
```

Submitting `cluster=jupiter_4nodes` into an allocation of some other size is
refused before any GPU work, on every rank:

```
FATAL: cluster=jupiter_4nodes is a 4-node configuration (cluster.num_nodes: 4),
but this allocation has 2 node(s). Either submit with --nodes=4, or pick the
cluster config that matches the allocation ...
```

**The job is much shorter than the run.** On JUPITER, where this was run, the
booster QOS capped a single job at twelve hours -- **check the equivalent for
`dc-gpu` on JURECA before planning launches**, because none of the numbers in
this section were re-measured after the move:

```
$ sacctmgr show qos part_booster format=Name,MaxWall
part_booster    12:00:00
```

against a **~16.5 h** projection for the shipped four-node plan -- 75000 steps at
the measured `0.7694 s/step` is 16.0 h, and the 126 epoch boundaries add ~17
minutes (one is ~8 s, measured directly as the validation loop's 46 batches) --
so the run is two of
these jobs back to back rather than the five or six the one-node plan needed.
(The one-node numbers, for comparison: 300000 steps at `0.7041 s/step` is 58.7 h,
and a single GPU would be ~50 h for a quarter of the data.) The script handles
the split two ways:
you `sbatch` it again with the same `NAME`, and `--signal=B:USR1@600` fires a
trap ten minutes before the wall clock ends that prints exactly that command
(and tries `scontrol requeue` first, for sites that allow it). Either way it is
the same path -- `resume: True` in `configs/config.yaml`, and geoarches
reloading the newest checkpoint in `exp_dir` with the optimiser state and the
global step intact -- so at most `SAVE_EVERY` steps are lost.

**JUPITER does not allow requeueing at all**, which is why the manual relaunch
is the primary mechanism and why `#SBATCH --requeue` is deliberately absent.
With it the job does not even submit:

```
$ sbatch --requeue ... scripts/pretrain_large.slurm
sbatch: error: job_submit_filter: --requeue option is not supported
sbatch: error: Batch job submission failed: Requested operation not supported on this system
```

**The step budget must not change across a relaunch**, and the reason is sharper
than "the finish line moves". geoarches' resume does not re-compose `cfg.module`:
it replaces it with the *resolved* module config written at the first launch and
then re-applies only the `+`-prefixed command-line overrides. So `++max_steps=8`
moves the trainer's budget while `module.module.num_training_steps` stays at the
first launch's 4 -- reproduced exactly that way:

```
updated cfg ... 'max_steps': 8 ... 'num_training_steps': 4
```

and diffusers' cosine lambda, `max(0, 0.5 * (1 + cos(pi * 2 * num_cycles *
progress)))`, does not clamp past the end of its schedule. At twice the stale
budget the cosine is back at `cos(2 pi) = 1`, i.e. **the learning rate climbs
back to its peak** on a run that was supposed to be annealing to zero. The script
now passes `++module.module.num_training_steps=${MAX_STEPS}`, which starts with
`+` and therefore survives that merge, so the schedule follows the budget. That
fixes the mechanism; it does not make a changed budget a good idea. This is
reachable through the script's own knobs -- the budget is derived from the world
size, so a relaunch at a different `--nodes` changes it -- so relaunch at the
same node count, and choose `SAMPLE_BUDGET` before the first launch or start a
fresh `NAME`.

Measured on the script itself: job 1286618 ran 100 steps at `--nodes=2`
(800 samples / 8 ranks), and job 1286623 relaunched the same `NAME` at
`--nodes=1`, derived 200 steps (800 / 4), restored from
`checkpoint_global_step=100.ckpt` and finished at 200. geoarches' pre-merge
`hydra config` line still held `'num_training_steps': 100`; its post-merge
`updated cfg` line held 200, on all four ranks.

Both halves were verified on four nodes with the real script and the real preset
(`MODULE=large MAX_STEPS=800 SAVE_EVERY=400`). Job 1285799 brought up all sixteen
ranks (`GLOBAL_RANK: 0..15, MEMBER: 1/16..16/16`), reported 593 batches per epoch
-- `ceil(9488 / 16)`, which is what sixteen ranks should shard the train split
into -- and wrote `checkpoint_global_step=400.ckpt` (5.5 GB) before it was
cancelled at step 400. Job 1285909, submitted with the same `NAME`, printed

```
Found checkpoints [.../checkpoint_global_step=400.ckpt]
Restored all states from the checkpoint at .../checkpoint_global_step=400.ckpt
```

and carried on from 400 through the epoch boundary at 593 to 800, `COMPLETED` in
7 min 47 s. The same was proved at four GPUs in task 10.

(You will see `SLURM auto-requeueing enabled` in the log. That is Lightning's own
handler, and it is *not* what requeues this job: the `B:` in `--signal=B:USR1@600`
sends the signal to the batch shell only, never to the training tasks. Left to
Lightning it would write `.pl_auto_save.ckpt` into the repository root, which
geoarches does not look at -- it only reads `modelstore/<name>/checkpoints/`.)

**More GPUs do not make a fixed `max_steps` finish sooner.** `max_steps` counts
*optimiser* steps, and one optimiser step is still one forward and backward plus
an all-reduce whatever the world size. What the extra cards buy is a larger
effective batch -- more data per step -- which is why this script derives
`max_steps` from a sample budget instead of hardcoding it. See
[4.1](#41-the-four-presets).

### Which checkpoints exist for you to start from

| run | what it is | steps |
|---|---|---|
| `modelstore/task6_tiny` | a `full` model, 13.5 M parameters, `tiny_train` | 4500 |
| `modelstore/ocean_tiny` | the ocean specialist, `train` | 2000 |
| `modelstore/seaice_tiny` | the sea-ice specialist, `train` | 2000 |
| `modelstore/seaice_isolated_tiny` | sea ice with no ocean input at all, `train` | 2000 |

**A fresh clone has none of these.** `modelstore/` is git-ignored, so link or
copy it from the shared checkout first --
[docs/01 section 1.1](01_setup.md#the-normal-route) step 3.

The three specialists were trained to prove the coupling plumbing works. **They
are not skill models** -- 2000 steps is under two epochs of the full split, and
`ocean_tiny` actually loses to persistence. Use them as starting points and as
worked examples, not as baselines to beat.

## 4.5 What else is worth changing

In rough order of how much a day's work is likely to buy you:

| lever | where | note |
|---|---|---|
| more steps / more data | `++max_steps`, `dataloader=glorys` | the cheapest real gain |
| a bigger preset | `MODULE=small` or `base` | 45 M or 85 M parameters |
| the loss weights | `loss_weight` in `variables.py` | deep salinity is currently *worse* than persistence -- see [docs/06](06_evaluation.md) |
| multi-step training | `module.train.rollout_iterations` | train on a 2-day rollout instead of 1; directly targets the divergence in [docs/06](06_evaluation.md#65-the-90-day-free-run-and-what-it-tells-you). **Read [4.6](#46-training-on-a-multi-step-rollout) before you queue for it.** |
| the learning rate and schedule | `module.module.lr`, `num_warmup_steps` | 3e-4 with 150 warm-up steps was never tuned |
| `add_input_state` | `configs/module/*.yaml` | on by default: the model predicts the one-day *change*. Turning it off makes the problem much harder and is not recommended, but it is instructive to see why |
| the vertical | see [4.2](#42-the-vertical-is-not-a-scaling-axis) | you would have to replace the backbone |

Everything above is a hydra override, so nothing needs editing:

```bash
.venv/bin/python -m geoarches.main_hydra --config-path $PWD/configs \
    cluster=jupiter_1gpu module=tiny dataloader=glorys_tiny ++name=experiment_7 \
    ++module.module.lr=1e-3 ++module.train.rollout_iterations=2 ++max_steps=4000
```

The [cheatsheet](cheatsheet.md) lists the overrides worth knowing.

**A preset is a whole recipe, not only a network size.** `module=base` brings its
own `batch_size` (2, against `tiny`'s 8) and its own `max_steps` (110000, against
4000): swap the preset and you have changed the experiment in three ways, not
one. Two cases now warn -- naming the same group twice on one command line
(`make train-tiny HYDRA_ARGS="module=base"` expands to `module=tiny ...
module=base`, and hydra silently keeps the last), and re-using a run name whose
previous `modelstore/<name>/config.yaml` recorded a different budget. A brand-new
name with a different preset is not a "swap" and cannot be detected as one, but
the run plan banner states the budget and the batch size outright before
anything runs.

**One trap in that command line.** `++key=value` means *add or override*, so a
mistyped key is created rather than rejected: `++max_step=10` would leave
`max_steps` at 4000 and add a stray `max_step`, and `++module.lr=1e-5` would
leave the real `module.module.lr` untouched. Both used to give you a 30-minute
run at settings you did not ask for, with no warning; both are now **refused at
startup**, the second with the real path named. A genuinely new *nested* key --
`++module.module.multistep_curriculum=True` is the documented example -- only
warns, because creating one is sometimes exactly right. **Prefer bare
`key=value` for keys that already exist** -- hydra errors with
`Could not override 'max_step'` when they do not --
and keep `++` for the two cases that need it: a key the config genuinely does not
have yet, and an override on a resumed run (see [4.4](#a-continue-a-run-that-stopped),
where only `+`/`++` survive). When in doubt, read back
`outputs/<date>/<time>/.hydra/config.yaml`, which is the resolved truth.

## 4.6 Training on a multi-step rollout

This is the lever that targets the 90-day divergence directly: instead of
scoring one step against tomorrow's truth, roll the model forward *n* steps from
one initial condition and score all of them, so it sees its own errors during
training.

```bash
make train-tiny NAME=multistep HYDRA_ARGS="++module.train.rollout_iterations=2"
```

**What it costs.** Measured on one booster GH200, `tiny`, batch 8, 40 training
steps each:

| `rollout_iterations` | s / training step | relative |
|---|---|---|
| 1 (the default) | 0.475 | 1.0x |
| 2 | 1.05 | **2.2x** |
| 5 | 2.53 | 5.3x |
| 10 | 4.98 | 10.5x |

So a 31-minute `make train-tiny` becomes about **68 minutes** at
`rollout_iterations=2`. Budget for it, and ask for the wall clock before you
start -- an `srun --time=01:00:00` will not finish it.

**A trap that was live in this kit until the final review.** geoarches'
`ForecastModule.on_train_epoch_start` is:

```python
if dataset.multistep > 1:
    dataset.multistep = 2 + self.current_epoch // self.increase_multistep_period
```

-- unconditional, and it overwrites whatever the dataloader was built with. With
it, `rollout_iterations=2` did not train a 2-step model: the rollout climbed
2 -> 3 -> ... -> 10 over `tiny`'s 18 epochs, a mean of **5.88** steps per
training step, so the run cost about **5.9x**, not 2.2x -- 31 minutes became
roughly three hours. And `=3` was silently *reduced* to 2 at epoch 0.

`OceanForecastModule` now holds the rollout at the length you asked for. If you
want geoarches' curriculum, ask for it and budget for it:

```bash
make train-tiny NAME=curriculum HYDRA_ARGS="++module.train.rollout_iterations=2 \
    ++module.module.multistep_curriculum=True"
```

With `increase_multistep_period=2` (geoarches' default) the mean rollout length
over `E` epochs is about `2 + E/4`, and the cost follows the table above roughly
linearly.

**Which is the better recipe?** Unmeasured here -- nobody in this project has
trained a converged multi-step model, and the honest answer is that this is
exactly the experiment worth doing in a day. What is measured is the cost, and
that the divergence in [docs/06 section 6.5](06_evaluation.md#65-the-90-day-free-run-and-what-it-tells-you)
reproduces on three single-step checkpoints. Start at 2, keep the wall clock
honest, and score the result the same way as the baseline (`make eval`, same
`--domain`, `--n-inits` and `--init-selection`) so the comparison means
something.

Two mechanical notes if you go down this road:

* `module.val.rollout_iterations` is separate and stays at 1 unless you raise it,
  so your validation loss stays comparable with a single-step run's.
* the dataset loads `multistep` future states per sample, so each batch is bigger
  in host memory as well as slower.

---

[< your first model](03_first_model.md) | [next: coupling >](05_coupling.md)
