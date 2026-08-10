# 0. Start here

**Challenge: AI-based Ocean and Sea-Ice Modelling.**
Tutors: Kacper Nowak and Nils Hutter.

You are going to train a neural network to forecast the global ocean and the sea
ice on top of it, one day at a time, and then find out whether it is any good.

**You do not need to know any oceanography.** Everything you need is explained as
it comes up. You do not need to have trained a weather model before, and you do
not need to have used a compute cluster before.

---

## The path, in order

Follow it top to bottom the first time. Each row says what you will have when you
finish it.

| # | do this | time | you end up with |
|---|---|---|---|
| 1 | [**Set up**](01_setup.md) -- environment, the cluster, `make doctor` | 15 min | a checkout that passes every check |
| 2 | [**Look at the data**](02_data_and_masking.md) + [notebook 01](../notebooks/01_explore_glorys.ipynb) | 45 min | you know what you are forecasting, and every trap in it |
| 3 | [**Train the tiny model**](03_first_model.md) | 31 min of GPU | a trained checkpoint of your own |
| 4 | [**Score it**](06_evaluation.md) + [notebook 03](../notebooks/03_evaluate_and_animate.ipynb) | 5 min | 10 figures, 6 animations and an honest scorecard |
| 5 | [**Pick an experiment**](07_challenge_ideas.md) | the rest of the time | a result |

Steps 4 and 5 are where the challenge actually is. Steps 1 to 3 exist so that
everybody in the room starts from the same place.

The middle two documents are optional on a first pass and become relevant as soon
as you pick an experiment:

* [**4. Bigger models and fine-tuning**](04_scaling_finetuning.md) -- the four
  size presets, and how to start from someone else's weights.
* [**5. Coupling**](05_coupling.md) -- running two specialised models together as
  one forecast system. This is one of the three open questions.

And [**the cheatsheet**](cheatsheet.md) is the page to keep open in a second tab.

## The notebooks

| notebook | needs | runtime | what it is |
|---|---|---|---|
| [01 explore GLORYS](../notebooks/01_explore_glorys.ipynb) | a built clone | 18 s | the data and its traps, as pictures. **Works with no trained model.** |
| [02 train and roll out](../notebooks/02_train_and_rollout.ipynb) | a GPU | ~3.5 min | train, forecast 10 days, then watch a 90-day run fall apart |
| [03 evaluate and animate](../notebooks/03_evaluate_and_animate.ipynb) | a GPU | 3.7 min | metrics, figures, animations, the report |
| [04 couple two models](../notebooks/04_couple_two_models.ipynb) | a GPU | 1.5 min | two specialists run as one system |

Notebook 01 needs no GPU, but like everything else here it needs a clone that
has been built: `make setup` (~30 s warm, ~5 min cold) and `make stats` (~8 min), once, from
[docs/01](01_setup.md#the-normal-route). Without them it stops at
`ModuleNotFoundError: torch` and then at a missing mask file.

Notebooks 02 to 04 need a GPU, which means the Jupyter server has to run on a
**compute node** while your browser is on your **laptop**. That takes three
steps and an SSH tunnel.

**1. Get a compute node and start the server there.**

```bash
srun --account=hclimrep --partition=booster --gres=gpu:1 --ntasks=1 \
     --cpus-per-task=16 --time=02:00:00 --pty bash

export CUDA_VISIBLE_DEVICES=0
cd /e/scratch/hclimrep/$USER/hackathon-ocean-sea-ice
hostname                       # <- write this down, e.g. jpbo-053-01.jupiter.internal
.venv/bin/jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

It prints the address and the token you will need:

```
[I ServerApp] Jupyter Server 2.20.0 is running at:
[I ServerApp] http://0.0.0.0:8888/lab?token=90a102aa1359a502f16563c1bc0b3a18a...
    Or copy and paste one of these URLs:
        http://jpbo-053-01.jupiter.internal:8888/lab?token=90a102aa1359a502f16563c1bc0b3a18a...
```

Leave it running. If port 8888 is taken -- you are sharing this machine -- pick
another one and use it in both places, for instance `--port=8899`.

**2. Open a tunnel, in a second terminal on your laptop.** Use the same host you
normally SSH into JUPITER with, and the compute node name from step 1:

```bash
ssh -N -L 8888:jpbo-053-01.jupiter.internal:8888 <your-user>@<the JUPITER login host>
```

`-N` means "do not run a command, just forward the port". It looks like it has
hung; that is correct. Leave it running too.

**3. Open the notebook** at `http://localhost:8888/lab?token=<the token from
step 1>` in your browser.

If your laptop cannot reach the compute node directly, add the login node as a
jump host: `ssh -N -J <your-user>@<login host> -L 8888:jpbo-053-01...:8888 ...`.
If any of this fights you, ask a tutor -- it is not the interesting part of the
day.

## What the challenge is asking

Three open questions. [Document 7](07_challenge_ideas.md) turns each of them into
experiments a team can finish in a day.

**1. Coupling dynamics.** The ocean and the sea ice are two physical systems that
constantly affect each other. Real climate models keep them as separate
components that exchange fields. Is that a good idea for a *learned* model? One
network that predicts everything can learn any relationship it likes, but spends
its capacity on everything at once. Two specialists each get a whole network for
their own problem, but can only see each other through the fields they exchange
-- and each inherits the other's errors. Nobody knows which wins here.

**2. Implicit learning.** How much does a model learn about a system it is never
asked to predict? A sea-ice model that reads the ocean is never scored on the
ocean. Does it build an internal representation of it anyway?

**3. Higher resolution.** Everything here is 1 degree. A 0.25-degree archive is
on the same filesystem. What breaks, what has to change, and what does it buy?

## The one thing to take away

**The ocean is slow, and that makes it a deceptive thing to forecast.**

Sea surface temperature moves about **0.13 degC in a day**, against a global
spread of 11.7 degC. So "tomorrow looks like today" -- the **persistence**
baseline -- is already a very good forecast, and a model that sounds impressive
can easily be worse than doing nothing.

The trained `tiny` model in this kit beats 1-day persistence by 12.8% on the
loss and by **3.7%** on sea surface temperature -- both measured on the first 128
samples of `tiny_val`, which the model never trained on. That is a real result,
and it is a small one.

**So: never report a number without a baseline beside it, and always say which
split you measured on.** Everything in this kit is built to make that automatic.

## What is where

```
README.md                the front door
docs/                    you are here
notebooks/               01 to 04
configs/                 hydra: cluster / dataloader / module / forcing
oceanarches/             the code
  dataloaders/           variables.py, masks.py, glorys.py   <- read variables.py first
  backbones/             the encoder/decoder that wraps geoarches' backbone
  lightning_modules/     ocean_forecast.py, coupled.py
  metrics/               masked (ocean-only) metrics and the sea-ice diagnostics
  evaluation/            rollout, baselines, figures, animations, report
  stats/                 masks, normalisation statistics, climatology (generated)
scripts/                 data preparation, statistics, the benchmark, two SLURM jobs
tests/                   605 tests, CPU only
data/                    33 prepared yearly files, 92 GB (generated)
modelstore/              training runs: checkpoints + config    (generated)
evalstore/               figures, animations, reports, caches   (generated)
```

If you read one source file, make it
[`oceanarches/dataloaders/variables.py`](../oceanarches/dataloaders/variables.py).
It is the single source of truth for what is in the state, and everything else
derives from it.

## Getting unstuck

1. `make doctor` -- it prints the fix on the line below anything that is not
   `PASS`.
2. [The cheatsheet's error table](cheatsheet.md#errors-we-expect-you-to-hit) --
   the failures we already know about, including three cluster traps that have
   each cost this project real time.
3. Ask a tutor. That is what we are here for.

---

[**Next: setting up >**](01_setup.md)
