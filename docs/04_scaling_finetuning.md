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

| preset | `emb_dim` | `depth_mult` | parameters | batch | `max_steps` | wall clock |
|---|---|---|---|---|---|---|
| `tiny` | 96 | 1 | **13.2 M** | 4 | 4000 | **~34 min** (measured) |
| `small` | 192 | 1 | **45.0 M** | 2 | 40000 | run `make benchmark` |
| `base` | 192 | 2 | **84.6 M** | 1 | 84000 | **9 h 23 m** on four GPUs (measured) |

`batch` is per GPU, and every one of them is measured to fit a dc-gpu
A100-SXM4-40GB: `tiny` peaks at 21.44 GiB, `base` at 21.33 GiB.

**The parameter column is backbone + embedder.** Lightning's own summary during
training prints the whole module and is therefore larger -- 13.465 M for `tiny`
against the 13.2 M here -- because it also counts the two timestep embedders that
carry the month and hour conditioning. Neither is wrong; just do not compare one
against the other.

### What the wall clocks mean

Two of the three are measured. `tiny` runs at 1.98 it/s end to end -- dataloader
and metrics included -- so 4000 steps is about 34 minutes. `base_pretrained` took
9 h 23 m for 84 000 steps on one node of four GPUs, at 2.52 it/s. Nobody has run
`small` to completion; `make benchmark` will give you the step time on the card
you are actually on, which is the only number worth planning a job against.

`tiny` also pays a noticeable overhead at epoch boundaries: it runs on
`glorys_tiny`, whose epoch is only 229 batches, so 4000 steps is 17.5 epochs of
validation and dataloader respawn. The other presets run on `glorys`, whose epoch
is several times longer, so they pay it far less often.

### What more GPUs buy

**Data, not wall clock.** `max_steps` counts optimiser steps, and DDP does not
divide them: four ranks make each step slightly more expensive while showing the
model four times as many samples. A four-GPU run of the same `max_steps` takes
about the same wall clock -- or a little more -- and sees four times the data,
with an effective batch of `4 x batch_size`.

That is worth having, but it is a different experiment, not a faster one. If you
want the same run to finish sooner, shorten `max_steps` -- and move
`num_training_steps` with it, or the cosine schedule anneals to a budget the run
never reaches.

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
    cluster=jureca_1gpu module=tiny dataloader=glorys_tiny \
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
    cluster=jureca_1gpu module=tiny dataloader=glorys \
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

**`base_pretrained` is the checkpoint this challenge is built around.** 84 000
steps of `module=base` on one JURECA node (4 x A100-40GB) in 9 h 23 m, final
`val_loss` 0.263. On the held-out test years it beats 1-day persistence on every
one of the 17 scored variables at every lead time out to 10 days -- 50% better on
day-1 sea surface height, 31% on SST, 36% on sea-ice concentration, 50% on deep
salinity -- and its 90-day free run keeps 99.93% of ocean cells inside
[-5, 40] degC where `tiny` puts 19-22% outside. [docs/06](06_evaluation.md)
explains what those numbers mean and how to produce them for your own model;
`evalstore/base_pretrained/report.md` in the shared store is its full scorecard.

**Start from it rather than training from scratch: you have a day, and this was
9 h 23 m on four GPUs.** Both scoring and fine-tuning fit a single dc-gpu A100 at
`batch_size 1` (21.33 GiB peak), and both are tested -- a fine-tune resumes from a
training loss of 0.20 rather than the 35 a fresh `base` starts at, which is how you
can tell the weights really loaded.

```bash
# THE MAIN PATH: fine-tune the pre-trained `base` model on one GPU.
# MODULE=base is not optional -- it must be the preset FROM was trained with.
sbatch --export=ALL,FROM=base_pretrained,MODULE=base,NAME=my_finetune \
    scripts/finetune.slurm

# fine-tune the 31-minute model instead, if you want a quicker loop first
sbatch --export=ALL,FROM=task6_tiny,NAME=my_finetune_tiny scripts/finetune.slurm

# train a `base` from scratch, one node, one 12-hour window -- how the shipped
# model above was made
sbatch scripts/pretrain_base.slurm
```

[`scripts/pretrain_base.slurm`](../scripts/pretrain_base.slurm) runs **84 000
steps**, not the preset's 110 000. Measured on one dc-gpu node with the real
dataloader, the real metrics and four DDP ranks: **2.18 it/s** budgeted and 2.52
achieved, so 84 000 steps came in at 9 h 23 m against the preset's 110 000 at a
projected 14.2 h, which does not fit a window. `num_training_steps` follows
`max_steps`, so the shorter budget is a complete annealed run rather than one that
was cut off. Remember that `max_steps` counts optimiser steps and four ranks do
not divide it -- what they buy is 4 samples per step instead of 1.

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

[`scripts/pretrain_base.slurm`](../scripts/pretrain_base.slurm) is the other end:
one node, `--ntasks=1`, four visible GPUs, `cluster=jureca_4gpu`. Two things about
it are worth knowing even if you never run it.

**One task, four GPUs -- not one task per GPU.** Lightning picks its launch mode
from the environment. With `--ntasks=1` it forks one process per visible device
itself, which is what the trainer is configured for; under `--ntasks-per-node=4`
it would attach to SLURM's ranks instead, and the two disagree about who spawns
whom.

**It runs 84 000 steps, not the preset's 110 000.** At the measured 2.52 it/s
that is 9 h 23 m, which fits a single wall-clock window; 110 000 would not.
`num_training_steps` follows `max_steps`, so the shorter budget is a complete,
properly annealed run rather than one that was cut off -- and `SAVE_EVERY` divides
`MAX_STEPS`, so the finished model actually reaches disk. Both knobs are
environment variables:

```bash
sbatch --export=ALL,NAME=base_v2,MAX_STEPS=72000 scripts/pretrain_base.slurm
```

If a job is killed at the wall clock, `sbatch` it again with the same `NAME`:
`resume: True` in `configs/config.yaml` makes geoarches reload the newest
checkpoint in `exp_dir` with the optimiser state and the global step intact, so at
most `SAVE_EVERY` steps are lost. Move `MAX_STEPS` between launches and the
learning-rate schedule follows it, because the script passes
`++module.module.num_training_steps` explicitly -- without that the cosine would
keep annealing toward the first launch's budget.

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
    cluster=jureca_1gpu module=tiny dataloader=glorys_tiny ++name=experiment_7 \
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

**What it costs.** Measured on `tiny`, 40 training
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
