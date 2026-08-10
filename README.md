# AI-based Ocean and Sea-Ice Modelling

A hackathon starter kit for forecasting the global ocean and sea ice with a
neural network.

**Tutors: Kacper Nowak and Nils Hutter.**

Built on [geoarches](https://github.com/INRIA/geoarches) and the GLORYS ocean
reanalysis at 1 degree: 33 years of daily fields, 11 variables, 13 depth levels.

You do not need any oceanography, and you do not need to have trained a weather
model or used a cluster before. Everything is explained where it comes up.

---

## Zero to a trained model in about 45 minutes

```bash
# 1. build the environment                            (~30 s warm, ~5 min cold)
make setup

# 2. build the masks, normalisation statistics and climatology       (~8 min)
make stats

# 3. check the machine can do this                                     (~6 s)
make doctor

# 4. get a GPU -- and you do need one: the login node's card is real,
#    is shared with everyone logged in, and is not allocated to you
srun --account=hclimrep --partition=booster --gres=gpu:1 --ntasks=1 \
     --cpus-per-task=16 --time=01:00:00 --pty bash
export CUDA_VISIBLE_DEVICES=0

# 5. train                                                     (30 min 51 s)
make train-tiny NAME=my_first_run

# 6. score it against persistence and climatology, with figures  (4-6 min)
make eval NAME=my_first_run LEAD_DAYS=10
```

Steps 1 and 2 are once per clone, and `make doctor` FAILs until step 2 has run:
`oceanarches/stats/*` is generated, not committed. So does `make test`, in its
own way -- without step 2 it skips 45 tests and tells you to run it. If you want
the no-training experiments in [docs/07](docs/07_challenge_ideas.md) as well,
link the shipped checkpoints into a `modelstore/` of your own --
[docs/01 section 1.1](docs/01_setup.md#the-normal-route) gives the four lines.

Step 4 is not optional either. `make doctor` and every training or evaluation
command now WARN when `SLURM_JOB_ID` is unset, because a *visible* GPU is not an
*allocated* one: the login nodes carry a real card, `torch.cuda.is_available()`
is `True` on them, and it says nothing about who else is using it.

Then read `evalstore/my_first_run/report.html`. There is no browser on JUPITER,
so either open it in the Jupyter server [docs/00](docs/00_start_here.md) sets up,
or copy that one file to your laptop -- it is self-contained. From **your own
machine**, not from the cluster:

```bash
scp <you>@<the JUPITER login host>:/e/scratch/hclimrep/<you>/hackathon-ocean-sea-ice/evalstore/my_first_run/report.html .
```

The host is whichever one you already `ssh` into; the path is absolute, and
`ls $PWD/evalstore/<run>/report.html` on the cluster prints it for you to paste.

Those are stopwatch measurements on one JUPITER booster GH200, not targets.
`make train-tiny` took **1851 s = 31 minutes** for 4000 steps and a cold-start
rehearsal from a fresh clone measured **1767 s = 29 min 27 s** for the same
command. [The full walk-through is here.](docs/03_first_model.md)

If you have no GPU yet, start with
[`notebooks/01_explore_glorys.ipynb`](notebooks/01_explore_glorys.ipynb) --
it needs no GPU and no trained model, and it takes about 20 seconds once steps 1
and 2 above have been run.

## Read these, in this order

| | |
|---|---|
| [**docs/00_start_here.md**](docs/00_start_here.md) | **the guided path.** Start here. |
| [docs/01_setup.md](docs/01_setup.md) | environment, SLURM, the data, `make doctor` |
| [docs/02_data_and_masking.md](docs/02_data_and_masking.md) | **the important one.** What GLORYS is, and every trap in it. |
| [docs/03_first_model.md](docs/03_first_model.md) | train the tiny model and read the logs |
| [docs/04_scaling_finetuning.md](docs/04_scaling_finetuning.md) | the four size presets, and fine-tuning |
| [docs/05_coupling.md](docs/05_coupling.md) | two specialists run as one forecast system |
| [docs/06_evaluation.md](docs/06_evaluation.md) | how to tell whether a model is good |
| [docs/07_challenge_ideas.md](docs/07_challenge_ideas.md) | nine experiments you can finish in a day |
| [docs/cheatsheet.md](docs/cheatsheet.md) | every command, every override, every error |

And four notebooks:

| notebook | needs | runtime |
|---|---|---|
| [01 explore GLORYS](notebooks/01_explore_glorys.ipynb) | steps 1 and 2 above | 18 s |
| [02 train and roll out](notebooks/02_train_and_rollout.ipynb) | a GPU | ~3.5 min |
| [03 evaluate and animate](notebooks/03_evaluate_and_animate.ipynb) | a GPU | 3.7 min |
| [04 couple two models](notebooks/04_couple_two_models.ipynb) | a GPU | 1.5 min |

## What you are forecasting

Eleven variables on a 180 x 360 grid: sea surface height, mixed layer depth, sea
floor temperature, sea ice concentration, thickness and velocity, and
temperature, salinity and velocity at 13 depths from 0.49 m to 1684 m.

| | |
|---|---|
| data | GLORYS12V1 daily means, 1993-2025, 12051 days, 92 GB prepared from a 608 GB archive |
| splits | train 1993-2018 (9488) / val 2019-2020 (730) / test 2021-2023 (1094) / holdout 2024-2025 (730) |
| grid | 1 degree, 45115 ocean cells at the surface; **30.4% of the grid is land, and 42.5% at 1684 m** |
| presets | `tiny` 13.2M params, `small` 45.0M, `base` 84.6M, `large` 459.6M |

## The one thing to understand before you start

**The ocean is slow.** Sea surface temperature moves about 0.13 degC in a day
against a global spread of 11.7 degC. So "tomorrow looks like today" is already a
very good forecast, and a model that sounds impressive can easily be worse than
doing nothing.

The shipped `tiny` model beats 1-day persistence by 12.8% on the loss, and per
variable:

| | model | 1-day persistence | better by |
|---|---|---|---|
| sea surface temperature | 0.1214 degC | 0.1260 degC | 3.7% |
| sea ice concentration | 0.01116 | 0.01281 | 12.9% |
| sea surface height | 0.01851 m | 0.02228 m | 16.9% |
| **NH sea-ice extent bias** | **+0.0485** | **+0.0106** | **worse** |
| **salinity at 1684 m** | **0.001321** | **0.001181** | **worse** |

(First 128 samples of `tiny_val`, in dataset order.)

Two of those are worse than doing nothing, and they are on the list on purpose.
An honest scorecard is the point of this kit: every number it prints sits next to
the baseline it has to beat, computed through exactly the same code on exactly
the same samples.

And the same model, rolled out freely for 90 days, **leaves the attractor**. Not
gently: the sea surface temperature *field* stops being a temperature. Measured
on three independently trained `tiny` checkpoints, by day 90 **19.1% / 21.1% /
22.0% of all ocean cells are outside [-5, 40] degC**, with minima of -514, -275
and -315 degC -- an adjacent-cell checkerboard that the global mean averages
away. The free-running RMSE table says the same thing on all three: 15 to 20 degC
on SST at day 90, against a climatology's 0.68.

**How that shows up in the summary curves is run-specific, so expect your figure
07 to differ from this one and from your neighbour's.** Across those same three
runs the day-90 Southern-Hemisphere ice extent came out at 6.1, 17.3 and 65.8 x
10^6 km^2 against a truth of 6.2 -- one of them looks perfectly healthy in the
ice panel while its temperature field is exploding. Trust the free-run RMSE
table, not the extent curve.

That is not a bug. It is what autoregressive rollout does to a small model
trained on a single step, and fixing it is one of the best things you can do in a
day.
[Figure 07, the measurements and the levers.](docs/06_evaluation.md#65-the-90-day-free-run-and-what-it-tells-you)

## The three open questions

1. **Coupling dynamics** -- one model that predicts everything, or two
   specialists that exchange fields? [docs/05](docs/05_coupling.md)
2. **Implicit learning** -- how much does a sea-ice model learn about the ocean
   without being told? [docs/07](docs/07_challenge_ideas.md#question-2-implicit-learning)
3. **Higher resolution** -- what breaks at 0.25 degrees?
   [docs/07](docs/07_challenge_ideas.md#question-3-higher-resolution)

## Every command

```bash
make help
```

or [the cheatsheet](docs/cheatsheet.md).

## Licence and data

This kit is **BSD 3-Clause** -- [the full text is in `LICENSE`](LICENSE).
Share it, adapt it, build on it for the hackathon and afterwards, commercially or
not, as long as you keep the copyright notice and do not use the tutors' names to
endorse what you build.

That matches [geoarches](https://github.com/INRIA/geoarches), which nothing here
runs without. Note that geoarches contradicts itself on this: its `pyproject.toml`
metadata says CC BY-NC-SA 4.0, but the `LICENSE` file it actually ships -- and the
one upstream at the commit pinned in our `pyproject.toml` -- is 3-clause BSD, with
no non-commercial clause. We followed the licence text rather than the metadata.
If you plan to build something commercial on this, confirm that with INRIA first.

The licence covers this repository. It does **not** cover the data. GLORYS12V1 is
produced by Mercator Ocean International and distributed by the Copernicus Marine
Service under their terms. The copy on this filesystem is read-only, it is not
part of this repository, and you may not redistribute it -- if you need it
somewhere else, get your own from the
[Copernicus Marine Service](https://marine.copernicus.eu/). Nor is anything
derived from it in here: the prepared years, the generated statistics, the
checkpoints and everything under `evalstore/` are all gitignored and built on your
own machine, so a clone of this repository carries no GLORYS with it.
