# Cheatsheet

[< challenge ideas](07_challenge_ideas.md) | [back to the start](00_start_here.md)

Everything on one page. Nothing here is explained; the links go to the
explanation.

---

## Make targets

```bash
make help          # list them all
make setup         # build .venv (~30 s warm, ~5 min cold)
make doctor        # check python, allocation, GPU, data, stats  (~6 s)
make test          # the test suite (604 passed, 1 skipped, 82 s) -- after `make stats`
make lint          # ruff check + ruff format --check
make format        # fix formatting

make prep-data YEARS="2015 2016"    # prepare a couple of years
make prep-data-slurm                # the whole archive as a batch job (~7 min)

sbatch --export=ALL,FROM=task6_tiny,NAME=my_ft scripts/finetune.slurm   # 1 GPU
sbatch scripts/pretrain_large.slurm  # organisers only: `large` on 4 nodes x 4 GH200, DDP
make stats                          # masks + statistics + climatology (5-8 min)
make stats-quick                    # same, fewer dates (~1 min, smoke tests only)
#                                     recorded in both artefacts; doctor WARNs and
#                                     every report built on them says so on its face
make forcing-stats                  # normalisation for `forcing=file` only (~10 s)

make train-tiny NAME=my_run                      # the 30-minute model
make train MODULE=base DATALOADER=glorys NAME=x  # any preset
make benchmark                                   # params / step time / memory, all presets

make eval NAME=my_run LEAD_DAYS=10               # score + figures + animations (232-246 s cold, ~5 s warm)
make couple OCEAN=ocean_tiny SEAICE=seaice_tiny  # coupled rollout (~270 s)
```

`make eval` and `make couple` need an allocation exactly as `make train-*` does.
Without one they print the same warning and pause ten seconds on a tty.
One `LEAD_DAYS=10` evaluation leaves about **750 MB** in `evalstore/<run>/`;
`run_eval` prints that and the total when it finishes.

Overridable variables: `NAME`, `MODULE`, `DATALOADER`, `CLUSTER`, `YEARS`,
`LEAD_DAYS`, `EVAL_ARGS`, `OCEAN`, `SEAICE`, `MODE`.

> **Hydra overrides go in `HYDRA_ARGS`, quoted.** A bare
> `make train MODULE=tiny ++max_steps=10` is parsed by make as *a variable
> assignment* `++max_steps = 10`, so nothing would read it. **make now refuses
> that command line** rather than training the preset's full budget in silence:
> it prints what it parsed and names `HYDRA_ARGS`. Write
> `make train-tiny NAME=my_run HYDRA_ARGS="++max_steps=10"` instead, or call the
> python entry point directly. The same guard catches `module=base`,
> `++module.module.lr=1e-4` and a lower-case `name=my_run`; upper-case
> assignments such as `CUDA_VISIBLE_DEVICES=0` are untouched and still reach the
> recipe's environment.

## The training command in full

```bash
.venv/bin/python -m geoarches.main_hydra --config-path $PWD/configs \
    cluster=jupiter_1gpu module=tiny dataloader=glorys_tiny ++name=my_run
```

**It must be `--config-path` with an absolute path, never `--config-dir`.**
`--config-dir` is searched *after* the directory in geoarches' own
`@hydra.main(config_path="configs")`, both hold a `config.yaml`, so geoarches'
root config wins: you get its `max_steps: 300000` and `save_step_frequency:
50000`, your `configs/config.yaml` is ignored, and **nothing warns you**. Groups
(`module=`, `dataloader=`) still come from your directory, so the run looks
completely normal. This cost a 40-minute GPU run during development. The `make`
targets get it right; a test now runs the real recipe and checks.

## Hydra overrides worth knowing

| override | effect |
|---|---|
| `module=tiny\|small\|base\|large` | the size preset |
| `dataloader=glorys\|glorys_tiny\|glorys_ocean\|glorys_seaice\|glorys_seaice_isolated\|glorys_forced` | which variables, which years |
| `cluster=local\|jupiter_1gpu\|jupiter_4gpu` | batch size, precision, workers |
| `cluster=jupiter_4nodes` | the organisers' 16-GPU pre-training job. Needs `python -m oceanarches.main_multinode` instead of `geoarches.main_hydra`, which cannot do more than one node ([docs/04](04_scaling_finetuning.md#the-two-slurm-scripts)) |
| `forcing=none\|file` | prescribed atmosphere. `file` needs `make forcing-stats` and `dataloader=glorys_forced`, and only covers 2024 -- which is holdout ([docs/05 5.6](05_coupling.md#56-external-forcing-the-three-routes-in-and-the-file-one)) |
| `++name=my_run` | writes to `modelstore/my_run` |
| `++max_steps=8000` | run length. `save_step_frequency` must **divide** it. |
| `++save_step_frequency=1000` | how often to checkpoint |
| `++batch_size=4` | per-GPU batch |
| `++limit_val_batches=16` | validation batches per epoch |
| `++module.module.lr=1e-4` | learning rate |
| `++module.train.rollout_iterations=2` | train on a 2-day rollout. **Costs 2.2x the wall clock** (measured); [docs/04 4.6](04_scaling_finetuning.md#46-training-on-a-multi-step-rollout) |
| `++module.module.multistep_curriculum=True` | let geoarches lengthen the rollout as the run goes on (`2 + epoch//2`). Off by default; ~5.9x the wall clock on `tiny`. |
| `++seed=1` | a different run of the same configuration |
| `+load_ckpt=modelstore/other_run` | load weights **without** resuming the run |
| `log=False` | no wandb -- and **no `config.yaml`**, so `load_module` will fail |
| `--cfg job --resolve` | print the composed config and exit |

Through `make`, wrap them: `make train-tiny NAME=my_run HYDRA_ARGS="++max_steps=1000 ++module.module.lr=1e-4"`.

**`++` means add-or-override, so it will happily create a key nothing reads.**
One rule: **a `++` key that nothing would read is refused at startup**, before
the GPU is touched. "Would read it" is answered from the config itself --

* a **top-level** key that does not exist (`++lr=1e-4`, `++max_step=10`) is
  refused; `configs/config.yaml` enumerates every top-level key there is;
* a nested key whose **leaf name exists elsewhere** (`++module.lr=1e-4` against
  the real `module.module.lr`) is refused, and the message names the path you
  meant;
* any other nested key is refused **unless the `_target_` of the node it sits in
  takes it as a constructor keyword**. `++module.module.multistep_curriculum=True`
  and `++module.backbone.gradient_checkpointing=True` pass that test -- both are
  real keywords with defaults, which is why no yaml mentions them -- and
  `++module.embed_dim=512`, `++module.module.embed_dim=512` and
  `++dataloader.n_levels=20` do not. Those three were measured as the natural
  guesses for "make the model wider"; the keys that were meant are
  `++module.backbone.emb_dim=` and `++module.embedder.emb_dim=` (and they move
  together -- see [docs/04](04_scaling_finetuning.md)), and `dataloader.n_level_in`.
  The refusal names the nearest existing keys.

A single `+` is hydra's "I mean to add this" spelling and is never refused:
`+load_ckpt=modelstore/other_run` is unaffected. **Prefer a bare `key=value` for
a key that already exists** -- hydra then errors with
`Could not override 'max_step'` -- and keep `++` for a genuinely new key and for
overrides on a resumed run, where only `+`/`++` survive. Verify with
`outputs/<date>/<time>/.hydra/config.yaml`, or `--cfg job --resolve`.

**Every run prints a plan before it reads any data** -- presets, step budget,
which steps get checkpointed and where, `FRESH START` or `RESUMING from ...`, and
whether you hold an allocation. Read it instead of guessing.

`OCEANARCHES_SKIP_GUARDS=1` turns every one of these refusals into a printed
warning. It is deliberately not mentioned in the messages themselves: a bypass
printed next to the fix is the one people copy.

Check what a command will actually do without running it:

```bash
.venv/bin/python -m geoarches.main_hydra --config-path $PWD/configs \
    cluster=jupiter_1gpu module=tiny dataloader=glorys_tiny --cfg job --resolve | head -40
```

## SLURM one-liners

```bash
# interactive shell on one GPU for an hour
srun --account=hclimrep --partition=booster --gres=gpu:1 --ntasks=1 \
     --cpus-per-task=16 --time=01:00:00 --pty bash

# one command on one GPU
srun --account=hclimrep --partition=booster --gres=gpu:1 --ntasks=1 \
     --cpus-per-task=16 --time=01:00:00 make train-tiny NAME=my_run

sbatch run.slurm            # submit a batch job
squeue -u $USER             # what is queued or running
scancel <jobid>             # kill it
sacct -j <jobid>            # what happened to it
tail -f logs/*.out          # watch the output
```

**Always set `export CUDA_VISIBLE_DEVICES=0` in a single-GPU job.**

## Evaluation one-liners

```bash
P=".venv/bin/python -m oceanarches.evaluation.run_eval"

$P --exp my_run --lead-days 10                     # the default
$P --exp my_run --domain val --n-inits 64          # a steadier number
$P --exp my_run --skip-animations                  # 3.2 s on a warm cache
$P --exp my_run --skip-fields                      # metrics only, nothing large on disk
$P --exp my_run --force                            # ignore the cache
$P --exp my_run --free-days 30                     # shorter drift check
$P --exp a --compare-with "baseline=b" "variant=c" # overlay scored runs

$P --coupled --components ocean=o seaice=s --mode sequential
$P --coupled --components seaice=s --unpredicted-forcing ground_truth
```

## Poking at the data

```bash
# what is in a state (14 depths: a bare dataset loads every PREPARED level;
# a model preset asks for 13 of them with depth_indices=DEPTH_PRESETS['tiny'])
.venv/bin/python -c "
from oceanarches.dataloaders.glorys import GlorysForecast
ds = GlorysForecast(domain='tiny_val')
s = ds[0]
print(sorted(s)); print(tuple(s['state']['surface'].shape), tuple(s['state']['level'].shape))
"

# how big is each split
.venv/bin/python -c "
from oceanarches.dataloaders.glorys import GlorysForecast
for d in ('train','val','test','holdout','tiny_train','tiny_val'):
    print(d, len(GlorysForecast(domain=d)))
"

# reload a trained model
.venv/bin/python -c "
from geoarches.lightning_modules.base_module import load_module
m, cfg = load_module('modelstore/task6_tiny')
print(type(m).__name__, m.component.name, sum(p.numel() for p in m.parameters()) / 1e6, 'M params')
"
```

## Errors we expect you to hit

| what you see | what it means | what to do |
|---|---|---|
| `ValueError: Path does not exist` | `GLORYS_PREPPED` wrong, or data not prepared | `make doctor`; fix `config.env` |
| `FileNotFoundError: ...glorys_1deg_stats.pt` | statistics never built | `make stats` |
| `FileNotFoundError: modelstore/<name>/config.yaml` | the run was launched with `log=False` | re-run with `log=True` (the default), or regenerate it with `--cfg job --resolve > modelstore/<name>/config.yaml` |
| `WARN allocation none -- SLURM_JOB_ID is unset` | you are on a login node. The GPU it can see is real and is **not** yours | `srun ... --pty bash`, then `export CUDA_VISIBLE_DEVICES=0` |
| `refusing to run: that command line does not do what it says` | a bare hydra override on a `make` line | quote it into `HYDRA_ARGS="..."` |
| `FATAL: this run would train N steps and leave no checkpoint at all` | `save_step_frequency` does not divide `max_steps` | the message prints a value that does divide |
| `FATAL: modelstore/<run> already holds a checkpoint at step N` | you reused a finished run's name; it would train nothing and exit 0 | a fresh `NAME=`, a bigger `++max_steps`, or `make eval` |
| ``FATAL: `++lr=1e-4` creates a new top-level key`` | the override lands where nothing reads it | the message names the real path (`++module.module.lr=`) |
| 58 batches per epoch instead of 229 | SLURM gave you four GPUs | `export CUDA_VISIBLE_DEVICES=0` -- unless you meant it: 58 is also correct for a real 4-rank DDP job, so check `SLURM_NTASKS` |
| the run trains far past `max_steps` | `--config-dir` instead of `--config-path` | use the `make` targets |
| your override had no effect | you passed it to `make` as a bare word | put it in `HYDRA_ARGS="..."` |
| no checkpoint at the end of training | `save_step_frequency` does not divide `max_steps` | it cannot happen to a run that started after this commit: it is refused up front |
| `RuntimeError: size of tensor a (180) must match b (121)` | an ERA5 assumption reached our data | you are using geoarches' module or metrics, not ours |
| `... give a latent depth of 7, but ... only works at 8` | you changed the depth preset | 12 or 13 levels, nothing else -- [docs/04](04_scaling_finetuning.md#42-the-vertical-is-not-a-scaling-axis) |
| `Reusing cached rollout` and your scores did not change | that is the cache doing its job | if you really did retrain, the content hash catches it; otherwise `--force` |
| every metric is `NaN` | the metric was never updated | your loop scored nothing; NaN is deliberate, it used to report a perfect 0.0 |
| `wandb` wants a login | `wandb_mode: offline` was lost from the cluster config | put it back |
| `UnpicklingError: Unsupported global: GLOBAL omegaconf...` when resuming | torch 2.6 defaults `torch.load` to `weights_only=True`, and older checkpoints carry a `ListConfig` in `optimizer_states[...]["betas"]` | `import oceanarches.lightning_modules` -- the allowlist is in the subpackage, and plain `import oceanarches` does **not** do it (checked both ways). New checkpoints contain no OmegaConf at all. |
| `--exp <run>: no such run` | a typo, or `make eval` with no `NAME=` (both train and eval default it to `tiny`) | the message lists every run under `modelstore/`; pass the one you trained |
| `--exp <run>: checkpoints/ holds no *.ckpt file` | the run was **interrupted** before `save_step_frequency` steps -- a run that completes is now guaranteed a checkpoint | re-run the identical command to resume, or lower `++save_step_frequency` |
| `InstantiationException` wrapping `FileNotFoundError` from `make couple` | a mistyped component run | fixed: `--components ocean=X seaice=Y` now names the component and lists the runs that exist |
| `No prepared file covers the 'test' split (2021-2023)` | you prepared a subset of years; `make eval` scores on `test` | `make prep-data YEARS="2021-2023"`, or `make prep-data-slurm`. `make doctor` WARNs about this. |
| `[2026] REFUSED: 282 of 365 days are absent` | `make prep-data` with no `YEARS` met a year the archive has not finished | that is the guard working. `--allow-incomplete` if a partial year is really what you want. |
| `torch.OutOfMemoryError: CUDA out of memory` | a bigger `emb_dim`, `batch_size` or `rollout_iterations`; `tiny` already peaks at 42.6 GiB of 96 | `HYDRA_ARGS="++batch_size=4"`, then `++module.backbone.gradient_checkpointing=True`; `make benchmark` prints every preset's peak before you queue |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | you raised `emb_dim` without `num_heads` and `out_emb_dim` | all four move together -- the table in [docs/04 4.1](04_scaling_finetuning.md#41-the-four-presets). The module now refuses to build and says which disagree. |
| ``FATAL: run `x` already exists and was trained with a DIFFERENT architecture`` | you reused a run name under another preset (`NAME=baseline0 MODULE=small`) | geoarches' resume path copies the STORED `module`/`dataloader` config over yours, so the old network would be trained under the new name. Use a name of its own, or `mv modelstore/x modelstore/x.old` |
| ``FATAL: run `x` is already being trained by another process`` | two terminals, one forgotten `NAME=` | give the second one a name of its own. A lock whose SLURM job has ended is cleared automatically, so this means the other job is genuinely alive -- the message prints its host, pid and job id. Confirm with `squeue -j <id>`, and only then `rm modelstore/x/.training.lock` |
| ``FATAL: the GPU ran out of memory ...`` printed under PyTorch's own traceback | the batch does not fit | halve `++batch_size=`, then `++module.backbone.gradient_checkpointing=True`. PyTorch's `expandable_segments` suggestion is about fragmentation and will not fit a batch that does not fit |
| a killed job, and you want the steps back | geoarches resumes automatically | re-run the **identical** command: it prints "Experiment already exists. Trying to resume it." and restarts from the newest checkpoint. [docs/04 4.4(a)](04_scaling_finetuning.md#a-continue-a-run-that-stopped) |

## Numbers to keep in your head

| | |
|---|---|
| grid | 180 x 360, 1 degree, 45115 ocean cells at the surface |
| land | 30.4% at the surface, 42.5% at 1684 m |
| depths | 13 levels, 0.49 m to 1684 m |
| splits | train 9488 / val 730 / test 1094 / holdout 730 samples |
| the line to beat | 1-day persistence loss **0.82-0.87**, moves with the split |
| `tiny` | 13.2 M parameters, 29-31 min on one GPU, 12.8% better than persistence |
| day-1 SST | model 0.125 degC, persistence 0.131 degC |
| a full `make eval` | 232-246 s cold, ~5 s warm, ~4 s warm with `--skip-animations`; 10 figures, 6 animations, 2 reports, ~750 MB |
| the test suite | 604 passed, 1 skipped, 82 s (602/3 without `make forcing-stats`; 495/110 with no statistics at all) |
| a fresh clone needs | `make setup` (~30 s warm, ~5 min cold), `make stats` (~8 min), and its own `modelstore/` with the shipped runs symlinked in ([docs/01 1.1](01_setup.md#the-normal-route)) |

---

[< challenge ideas](07_challenge_ideas.md) | [back to the start](00_start_here.md)
