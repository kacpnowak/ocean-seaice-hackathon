# 1. Setting up

[< back to the start](00_start_here.md) | [next: the data >](02_data_and_masking.md)

Read time: 10 minutes. Doing it: 5 minutes if the environment already exists,
about 10 if you build it yourself.

Everything here happens on **JUPITER**. If you have never used a compute cluster
before, [1.4](#14-slurm-in-five-minutes) is written for you and nothing in it is
assumed knowledge.

---

## 1.1 Get the repository

**There is a shared checkout on the cluster** with the environment already built,
the 92 GB of prepared data, the generated statistics and four trained
checkpoints:

```
/e/scratch/hclimrep/nowak2/hackathon-ocean-sea-ice
```

Take your own copy of it. It is an ordinary git repository, so cloning it locally
works and gives you your own branch to commit on, without duplicating any of the
large generated directories (they are all git-ignored):

```bash
mkdir -p /e/scratch/hclimrep/$USER
cd /e/scratch/hclimrep/$USER
git clone /e/scratch/hclimrep/nowak2/hackathon-ocean-sea-ice
cd hackathon-ocean-sea-ice
```

That takes a few seconds and about 7 MB. **The rest of this kit assumes you are
in `/e/scratch/hclimrep/$USER/hackathon-ocean-sea-ice`.**

### The normal route

Your clone has the **code and nothing else**. Everything large is git-ignored, so
a fresh clone has no environment, no statistics and no checkpoints, and it has
to get them:

```bash
cd /e/scratch/hclimrep/$USER/hackathon-ocean-sea-ice

# 1. the environment                             (~30 s warm, ~5 min cold)
make setup

# 2. the generated statistics: masks, normalisation, climatology  (~8 min)
make stats

# 3. the shipped checkpoints, if you want the no-training experiments:
#    YOUR OWN modelstore/, with the shipped runs linked in one by one
mkdir -p modelstore
SHARED=/e/scratch/hclimrep/nowak2/hackathon-ocean-sea-ice/modelstore
for run in large_pretrained task6_tiny ocean_tiny seaice_tiny seaice_isolated_tiny; do
    ln -s $SHARED/$run modelstore/$run
done

# 4. now this passes
make doctor
```

**Step 2 is not optional and `make doctor` FAILs without it.**
`oceanarches/stats/*.nc` and `*.pt` are generated artefacts, not committed ones:
the mask file, the normalisation statistics and the 95 MB climatology are all
rebuilt from the prepared data. Eight minutes, once per clone. Do **not**
substitute `make stats-quick` -- see [1.6](#16-if-you-have-to-prepare-the-data-yourself).

**Step 3 links the four shipped runs individually, into a `modelstore/` of your
own.** That is deliberate, and it is not the same as linking the whole directory.
`modelstore/` is where *your* runs are written -- geoarches creates
`modelstore/<name>/` per run -- so it has to be a real directory you own. The
shipped runs are 164 MB each and you only ever read them, so they come in as
symlinks.

Linking the whole directory instead (`ln -s $SHARED modelstore`) breaks one way
or the other and there is no third option: if the shared copy is read-only to
you, the very next documented command dies with
`PermissionError: [Errno 13] Permission denied: 'modelstore/my_first_run'`; if it
is writable, your runs land in the tutor's directory. `cp -r` does not rescue it
either -- it copies the source's mode bits verbatim, so a read-only source gives
a read-only copy. Both routes were tried by two participants; both failed.

Do *not* try to redirect this with the `MODELSTORE` setting in `config.env`.
It is honoured when the evaluation pipeline looks a run up, but geoarches'
own `load_module` hardcodes `modelstore/`, so a redirected run is found and
then fails to load. A directory of your own with symlinks in it is the route
that works.

**Do not reuse a shipped name for a run of your own.** `make train-tiny
NAME=task6_tiny` used to exit 0 having trained and saved nothing; it is now
refused at startup, and the message offers a fresh name, a longer `++max_steps`
or `make eval`.

The runs the documents use by name are `large_pretrained` -- **the pre-trained
model you fine-tune** (docs/04) -- `task6_tiny` (docs/04, docs/06, docs/07, the
cheatsheet), and `ocean_tiny` + `seaice_tiny` + `seaice_isolated_tiny` (docs/05,
docs/07). Without step 3, every `[GOOD FIRST]` experiment that needs no training
will stop at "no such run".

`large_pretrained` is 78 GB and the other four are under 1 GB each. **Linking
costs you none of it** -- a symlink is a symlink, the bytes stay in the shared
store, and you only ever read them.

The data itself is the one thing you do not have to rebuild: `config.env` already
points `GLORYS_PREPPED` at `nowak2`'s prepared 92 GB, which is read-only to you,
which is what you want.

**Or work directly in the shared checkout** -- fastest, and fine for a first
look, but you will be treading on other people's `modelstore/` and `evalstore/`
directories. If you do, always pass your own `NAME=`.

The tutors will give you a git URL on the day if the kit has been published
somewhere by then; there is no public remote configured in this copy.

Either way, `make doctor` in [1.3](#13-check-that-it-works) tells you what is
missing.

## 1.2 Build the environment

One command:

```bash
make setup
```

It creates `.venv/` and installs `oceanarches` plus
[geoarches](https://github.com/INRIA/geoarches), the framework this kit is built
on, pinned to one commit so everybody in the room has the same version. It takes
~30 s warm, ~5 min cold: measured at 20 s for a brand-new `.venv` and 8 s to
re-check an existing one, with uv's wheel cache already populated. The first
build on a machine downloads ~2 GB of PyTorch and takes about five minutes.

**Never run `pip install` by hand in this project.** Use `.venv/bin/python`
directly, or `make` targets, which already do. You do not need to `activate`
anything.

Four things had to be pinned or overridden to make this work on JUPITER's
Grace-Hopper (aarch64) nodes. They are all in
[`overrides.txt`](../overrides.txt), with the reason next to each, and they are
recorded here so they do not surprise you:

* `torch` comes from `download.pytorch.org/whl/cu126`, not PyPI -- the plain PyPI
  aarch64 wheel is **CPU-only** and would silently give you no GPU;
* geoarches pins `tensordict <0.7`, but tensordict only ships aarch64 wheels from
  0.8.0, so the pin is relaxed to `>=0.9`;
* geoarches pins `torchvision <0.21`, which would drag torch back to 2.5.
  geoarches never imports torchvision anywhere, so that pin is overridden too;
* torch 2.9.1 asks for `nvidia-nccl-cu12==2.27.5`; JUPITER needs a newer NCCL, so
  it is moved up.

## 1.3 Check that it works

```bash
make doctor
```

This is the command to run first, and again whenever something breaks. Real
output from a working checkout:

```
OceanArches doctor -- repo at /e/scratch/hclimrep/nowak2/hackathon-ocean-sea-ice

  PASS  python           3.12.13 (/e/scratch/.../.venv/bin/python)
  PASS  imports          all core packages import
  PASS  power spectrum   pyshtools available
  WARN  allocation       none -- SLURM_JOB_ID is unset, on jpbl-s02-02 (a login node)
                         -> anything that trains or evaluates needs a node of your own: srun --account=hclimrep --partition=booster --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --time=01:00:00 --pty bash  (then: export CUDA_VISIBLE_DEVICES=0)
  WARN  gpu              1x NVIDIA GH200 480GB (torch 2.9.1+cu126), 94 of 95 GiB free on device 0 -- visible, but NOT allocated to you
                         -> a visible card is not an idle one; see the `allocation` line above
  PASS  ffmpeg           ffmpeg-linux-aarch64-v7.0.2
  PASS  raw GLORYS       /e/data1/climateai/hclimrep/data/glorys_1deg  1993-2026 (34 years)
  PASS  prepared data    /e/scratch/.../data/glorys_1deg_prepped  33 yearly files (1993-2025)
  PASS  masks            glorys_1deg_masks.nc (2.9 MB)
  PASS  norm stats       glorys_1deg_stats.pt (4.1 KB)
  PASS  climatology      glorys_1deg_climatology.nc (95.4 MB)
  PASS  stats depth      statistics: normalisation from 400 dates from 1993-2025, climatology from 33 of 33 years (1993-2025)
  PASS  IFS forcing      /e/data1/climateai/hclimrep/data/glorys_forcings/ifs_1deg  52 files
  PASS  forcing stats    ifs_1deg_forcing_stats.pt (2.5 KB)

  No failures, 2 warning(s) -- you can start working.
```

That is the **login node**, and those two warnings are the point of the check.
It takes about 6 seconds (5.7 s measured). Every row that is not `PASS` prints
the command that fixes it on the line below.

**A visible GPU is not an allocated one.** The JUPITER login nodes carry a real
card, so `torch.cuda.is_available()` is `True` there and tells you nothing about
whether that card's memory, cores or process slots are yours -- a participant
measured the login GPU at 99% utilisation with 49 GiB in use by somebody else's
job while their own training ran on it. `doctor` therefore keys the `allocation`
row on `SLURM_JOB_ID`, never on CUDA visibility. Inside an allocation it reads
`PASS  allocation  SLURM job <id>, 1 node(s), on jpbo-...` and the `gpu` row goes
back to `PASS`; off SLURM entirely (a laptop with no `srun`) the row is not
printed at all.

On a fresh clone the artefact rows read `FAIL masks`, `FAIL norm stats` and
`WARN climatology`, and doctor exits 1. That is [1.1](#11-get-the-repository)
step 2 missing, not a broken machine: run `make stats`.

After `make stats-quick` the `stats depth` row drops to `WARN` and names what was
sampled. It is the same fact the evaluation report prints on its own face --
see [1.6](#16-if-you-have-to-prepare-the-data-yourself).

`WARN prepared data` with a list of splits means the prepared years do not cover
a split you will need -- `make eval` scores on `test` (2021-2023), so two years
of data is not enough even though the files are there.

## 1.4 SLURM in five minutes

JUPITER is a shared machine. You log in to a **login node**, which is for
editing files, running `make doctor` and reading logs. You must **not** train
there. Real work goes to a **compute node**, and you ask for one through SLURM.

A booster node has **4x NVIDIA GH200** (96 GB of GPU memory each), 288 CPU cores
and 878 GB of RAM. You normally want one quarter of that.

### Interactive: run one command on a GPU and watch it

```bash
srun --account=hclimrep --partition=booster --gres=gpu:1 --ntasks=1 \
     --cpus-per-task=16 --time=01:00:00 --pty bash
```

That drops you into a shell **on** the compute node with one GPU. Then:

```bash
export CUDA_VISIBLE_DEVICES=0
cd /e/scratch/hclimrep/$USER/hackathon-ocean-sea-ice
make train-tiny NAME=my_first_run
```

`--time=01:00:00` asks for one hour. When it runs out, your job is killed
without warning, so ask for more than you think you need. Type `exit` to give
the node back.

### Batch: submit it and go away

Put your commands in a file:

```bash
cat > run.slurm <<'EOF'
#!/usr/bin/env bash
#SBATCH --account=hclimrep
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export CUDA_VISIBLE_DEVICES=0
make train-tiny NAME=my_first_run
EOF

mkdir -p logs
sbatch run.slurm
squeue -u $USER          # is it running yet?
tail -f logs/*.out       # watch it
```

`scripts/prepare_glorys.slurm` is a working example you can copy.

### Three cluster traps that have already bitten this project

These are not hypothetical. Each of them cost real time here.

**1. `--cpus-per-task=N` without `--ntasks=1` runs your script N times.**

```bash
srun --cpus-per-task=16 my_script.py         # WRONG: several concurrent copies
srun --ntasks=1 --cpus-per-task=16 my_script.py   # right
```

During this project that launched four concurrent copies of the test suite on
one node -- and, on another occasion, four copies of a script that edited a
source file, which half-mutated it. **Always pass `--ntasks=1`.**

**2. `--gres=gpu:1` gives you all four GPUs anyway.**

The booster partition allocates whole nodes. Ask for one GPU and sixteen cores
and SLURM hands you the lot. Measured, on the exact `srun` line printed above:

```
$ sacct -j 1282871 --format=JobID,AllocTRES%80,ReqTRES%60
JobID          AllocTRES                                                ReqTRES
1282871        billing=288,cpu=288,gres/gpu:gh200=4,gres/gpu=4,node=1   billing=16,cpu=16,gres/gpu=1,...
```

`gres/gpu=4`. `nvidia-smi --list-gpus` lists four cards. PyTorch Lightning then
sees four devices, decides to shard your data four ways -- inside a **single**
process, because you asked for one task -- and you silently train on a
**quarter** of your dataset: 58 batches per epoch instead of 229. The loss curve
looks completely normal.

> **Two words you need for that sentence**, if you have not trained a network
> before. A **batch** is the group of samples the model looks at in one step
> before updating its weights -- 8 days of ocean, for the `tiny` preset. An
> **epoch** is one pass over the whole training split. `tiny_train` holds 1825
> samples, so at a batch of 8 one epoch is `ceil(1825 / 8) = 229` batches. If
> Lightning only shows you a quarter of the data, that count falls to 58.
>
> Note this is a different "batch" from a SLURM **batch job**, which just means
> "submitted with `sbatch` and run without you watching". Same word, unrelated
> meanings.

```bash
export CUDA_VISIBLE_DEVICES=0     # do this in EVERY single-GPU job
```

There is no core count that avoids this, so the export is not optional. Check it
worked by looking at the batch count in the first epoch: **229** for
`make train-tiny`, not 58.

**3. Hydra leaves a directory behind after every run.**

Each launch writes `outputs/<date>/<time>/` with the resolved configuration and
the logs. It is harmless and occasionally very useful -- if you want to know what
a run actually ran with, `outputs/.../.hydra/config.yaml` is the honest record --
but it accumulates. It is small: 28 run directories measured 243 KB in total, so
about **9 KB per run**, against the 164 MB of a single `tiny` checkpoint. Delete
old ones freely; nothing reads them. `rm -rf outputs/` is always safe.

## 1.5 Where everything lives

| path | what it is | writable? |
|---|---|---|
| `/e/data1/climateai/hclimrep/data/glorys_1deg` | the raw GLORYS archive, 608 GB | no, read-only |
| `data/glorys_1deg_prepped/` | 33 prepared yearly files, 92 GB | yes, but do not |
| `oceanarches/stats/` | masks, normalisation statistics, climatology | rebuilt by `make stats` |
| `oceanarches/stats/ifs_1deg_forcing_stats.pt` | normalisation for `forcing=file` only | rebuilt by `make forcing-stats` |
| `/e/data1/climateai/hclimrep/data/glorys_forcings/ifs_1deg` | the optional IFS atmosphere, 1.1 GB | no, read-only |
| `modelstore/<name>/` | checkpoints and the config of a training run | yes |
| `evalstore/<name>/` | figures, animations, reports, cached rollouts | yes |
| `logs/`, `outputs/`, `wandblogs/`, `lightning_logs/` | job output and run metadata | yes |

All of those except the first two are in `.gitignore`. Nothing large is ever
committed.

Paths are configured in **one** file: [`config.env`](../config.env). Edit that
and everything follows. `oceanarches/paths.py` reads it, and the resolution
order is *environment variable > `config.env` > built-in default*, so
`GLORYS_PREPPED=/somewhere/else make train` works for a one-off.

## 1.6 If you have to prepare the data yourself

You almost certainly do not -- `make doctor` should already show 33 prepared
files. But if you are on a fresh checkout:

```bash
make prep-data YEARS="2015 2016"    # a couple of years, locally
make prep-data-slurm                # the whole 1993-2025 archive, as a batch job
make stats                          # masks + statistics + climatology, ~8 minutes
```

The full preparation took **7 minutes** as a SLURM job and produced 33 files,
12051 days, 92 GB. `make stats` took **4 minutes 55 seconds** on the machine it
was first timed on and **8 minutes** in a cold-start rehearsal; plan for eight.
It peaked at 5.6 GB of memory.

**A subset is a real choice, not a smaller version of the same thing.** Two years
of data pass `make doctor`'s file check but do not cover `test` (2021-2023), and
`make eval` -- whose default domain is `test` -- then stops with a message naming
the years it needs. `make doctor` WARNs about exactly this.

**`make prep-data` with no `YEARS` prepares every year the raw archive has**, and
the raw archive grows. It now holds 83 days of 2026, so the bare command would
add a quarter-year file that looks exactly like a whole year everywhere it is
listed. `prepare_glorys.py` refuses any year missing more than
`--max-missing-days` (10 by default -- 2003 really is missing two days and is
still prepared) and says so:

```
[2026] REFUSED: 282 of 365 days are absent from the raw archive, which is more
       than --max-missing-days (10). ... Pass --allow-incomplete if you really
       do want a partial 2026.
```

`--allow-incomplete` is the escape hatch if a partial year is what you want.

`make stats-quick` is about a minute and samples fewer dates. It is fine for a
smoke test and **not** fine for a model whose numbers you intend to report --
the normalisation statistics would be built from a handful of days.

That is now enforced rather than only stated. Both artefacts record how deeply
they were sampled, `make doctor` shows it in its `stats depth` row, and every
evaluation report built on them prints it in bold above its first table. You
cannot end up presenting a scorecard built on a one-minute climatology without
the page saying so.

`make stats` uses **every** prepared year, test and holdout included, and that
also builds the climatology, which is one of the two scored baselines. See
[docs/06 section 6.9](06_evaluation.md#69-what-this-evaluation-does-not-tell-you);
`.venv/bin/python scripts/compute_stats.py --years 1993-2018` builds the
strictly-train version.

## 1.7 Sanity: run the tests

```bash
make test
```

605 tests, CPU only, no GPU. **Run `make stats` first**: the suite reads the
generated mask file, and 45 of its tests cannot run without it.

```
604 passed, 1 skipped, 32 warnings in 81.62s (0:01:21)
```

That is the full route -- `make setup`, `make stats`, `make forcing-stats`. Two
other counts are correct rather than broken, and both say so when you look:

| what you ran | result |
|---|---|
| `make setup` + `make stats` + `make forcing-stats` | **604 passed, 1 skipped** |
| `make setup` + `make stats` (the usual route) | **602 passed, 3 skipped** -- the two extra skips are the `forcing=file` normalisation, which needs `make forcing-stats` (~10 s) |
| `make setup` only, no statistics | **495 passed, 110 skipped**, each skip naming `make stats` |

All three measured on this commit. The last one is the one to know: a clone that
has not run `make stats` used to give **45 failures** here, all of them the same
missing mask file wearing a stack trace that looked like a broken install.

The permanent skip is an opt-in check that needs a trained checkpoint;
[docs/06](06_evaluation.md#66-the-check-that-matters-most) shows how to switch it
on. If tests *fail* on a fresh checkout, something is wrong with the environment,
not with your idea.

How long it takes is mostly the login node's load, not the suite: 82 s twice
while this was written, and between 73 and 131 s across earlier runs, while the
node's load average moved between 20 and 50 on its 72 cores.

`.venv/bin/python -m pytest -q` does the same thing a few seconds faster, and
takes the usual pytest arguments if you want to run one file or one test.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ValueError: Path does not exist` | `GLORYS_PREPPED` is wrong or the data is not prepared | `make doctor`, then fix `config.env` |
| `FileNotFoundError: ...glorys_1deg_stats.pt` | statistics never built | `make stats` |
| `WARN allocation none -- SLURM_JOB_ID is unset` | you are on a login node; the GPU you can see is not yours | request one with `srun` ([1.4](#14-slurm-in-five-minutes)) |
| `make` refuses: "that command line does not do what it says" | a bare hydra override such as `++max_steps=10` on a `make` line | quote it: `HYDRA_ARGS="++max_steps=10"` |
| `refusing to start (no checkpoint would be written)` | `save_step_frequency` does not divide `max_steps` | the message prints a value that does divide |
| 45 test failures about `glorys_1deg_masks.nc` | you are on a commit older than this one | on this commit they are skips; run `make stats` |
| training runs but each epoch is 58 batches, not 229 | you got 4 GPUs | `export CUDA_VISIBLE_DEVICES=0` |
| the run trains far longer than `max_steps` says | `--config-dir` instead of `--config-path` | use the `make` targets; see [the cheatsheet](cheatsheet.md) |
| `wandb` asks for a login | your cluster config lost `wandb_mode: offline` | put it back; nothing here needs an account |
| `torch.OutOfMemoryError: CUDA out of memory` | you raised `emb_dim`, `batch_size` or `rollout_iterations`; `tiny` already sits at 42.6 GiB of 96 | `HYDRA_ARGS="++batch_size=4"` first, then `++module.backbone.gradient_checkpointing=True`; `make benchmark` prints the peak memory of every preset before you queue for a GPU |
| `make doctor` FAILs on masks and norm stats in a fresh clone | `oceanarches/stats/*` is generated, not committed | `make stats` (~8 min) |
| `--exp <run>: no such run` | a typo, or `make eval` with no `NAME=` (it defaults to `tiny`) | the message lists every run under `modelstore/` |

---

[< back to the start](00_start_here.md) | [next: the data >](02_data_and_masking.md)
