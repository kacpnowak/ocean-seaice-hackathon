# 7. What to actually do: nine experiments

[< evaluation](06_evaluation.md) | [the cheatsheet >](cheatsheet.md)

The challenge asks three open questions. This page turns them into experiments a
team can finish in a day, each with the commands and with what a result would
look like.

Each idea carries a difficulty and a time estimate. **The ones marked
[GOOD FIRST] are for people who have never trained a weather model.** They are
not the boring ones -- they are the ones where the measurement is unambiguous.

Before you start: pick **one** and finish it. A finished experiment with an
honest negative result beats three half-run ideas, and "we tried X and it did not
help, here is the evidence" is a genuine contribution.

---

## Ground rules for every experiment

1. **Change one thing.** If you change the preset and the loss weights together
   you have learned nothing about either.
2. **Same samples across arms.** `--domain`, `--lead-days`, `--n-inits` and
   `--init-selection` must match, or the numbers are not comparable. The
   manifests record all four so a mismatch is visible.
3. **Quote the baseline.** Never report an RMSE without persistence beside it,
   and always say which split.
4. **Check your components individually first.** A coupled system inherits its
   weakest component's error -- see [docs/05](05_coupling.md#55-what-the-measurements-actually-said),
   where coupling the ice to a bad ocean model was worse than telling the ice
   nothing changed.
5. **Two runs of the same configuration differ.** Before you believe a 0.5%
   improvement, train the same thing twice with `++seed=1` and see how much it
   moves on its own.

---

# Question 1: coupling dynamics

*Is one model that predicts everything better than two specialists that talk to
each other?*

## 1a. Joint model against two specialists [GOOD FIRST]

**Difficulty:** easy. **Time:** ~1 hour of GPU, ~2 hours in total.

The central question of the challenge, and the shipped configs answer it
directly.

```bash
# NOTE: a bare `++max_steps=4000` on a `make` line is parsed by make as a
# variable assignment, so nothing reads it -- make now refuses such a command
# line. Either quote it into HYDRA_ARGS, or use the python entry point directly
# as below.
G=".venv/bin/python -m geoarches.main_hydra --config-path $PWD/configs cluster=jupiter_1gpu"

# the joint model: one network, all 11 variables
$G module=tiny dataloader=glorys ++name=joint_run ++max_steps=4000

# two specialists, same architecture, same total steps
$G module=ocean_component  dataloader=glorys_ocean  ++name=ocean_run  ++max_steps=2000
$G module=seaice_component dataloader=glorys_seaice ++name=seaice_run ++max_steps=2000

# score them on identical samples
.venv/bin/python -m oceanarches.evaluation.run_eval --exp joint_run --lead-days 10 --n-inits 16
make couple OCEAN=ocean_run SEAICE=seaice_run LEAD_DAYS=10

# and overlay
.venv/bin/python -m oceanarches.evaluation.run_eval --coupled \
    --components ocean=ocean_run seaice=seaice_run --lead-days 10 \
    --compare-with "joint model=joint_run"
```

**Give the two arms the same total compute.** 4000 steps for one joint model
against 2000 + 2000 for the pair is the fair comparison; 4000 + 4000 is not.

**What a result looks like.** `figures/comparison_rmse.png` with two curves per
panel, plus the table in `report.md`. A clean answer is one arm beating the other
on `siconc` **and** `thetao` at day 1 and still at day 10. A mixed answer -- the
pair better on ice, the joint model better on the ocean -- is also a result, and
probably the more interesting one, because it says the two problems want
different capacity.

**Extensions if that goes quickly:** three specialists (split the ocean into
surface and interior); give the joint model 4000 steps against a pair with 3000
each and see whether the pair's advantage survives the unfairness.

## 1b. Coupled against uncoupled [GOOD FIRST]

**Difficulty:** easy. **Time:** ~20 minutes, and it needs no training at all --
the checkpoints ship with the kit. **A fresh clone has none of them**
(`modelstore/` is git-ignored); link them in first, per
[docs/01 section 1.1](01_setup.md#the-normal-route) step 3.

Take **one** set of sea-ice weights and change only what it is told about the
ocean. Because the weights are identical across arms, any difference is
information, not optimisation luck.

```bash
P=".venv/bin/python -m oceanarches.evaluation.run_eval"
COMMON="--lead-days 10 --n-inits 16 --init-selection spread"

$P --coupled --components seaice=seaice_tiny --unpredicted-forcing ground_truth $COMMON \
   --out evalstore/1b_true_ocean
$P --coupled --components seaice=seaice_tiny                                    $COMMON \
   --out evalstore/1b_frozen_ocean
$P --coupled --components ocean=ocean_tiny seaice=seaice_tiny                   $COMMON \
   --out evalstore/1b_predicted_ocean
```

**Give every arm its own `--out`.** A coupled run's directory is named from the
component *slots*, the mode and (when it is not the default) the forcing policy
-- `coupled_seaice_sequential`, `coupled_seaice_sequential_ground_truth`,
`coupled_ocean_seaice_sequential`. **The checkpoint names are not in it.** So the
moment two arms fill the same slot with different weights -- `seaice=seaice_tiny`
here and `seaice=ice_sees_ocean` in [2a](#2a-seaice-against-seaice_isolated-good-first) --
they share a directory and the second overwrites the first's `report.md`,
`summary.json` and figures.

The **numbers** are safe either way: the cache is keyed on a content hash of each
checkpoint, so nothing is ever served from the wrong arm. The **reports** are
not, and reports are what you will be reading side by side. `--out` costs
nothing; use it whenever you run more than one arm.

`--out` names the output **root**, not the run directory, so
`--out evalstore/1b_true_ocean` writes to
`evalstore/1b_true_ocean/coupled_seaice_sequential_ground_truth/`. That is one
level deeper than it looks, and it is what keeps the arms apart -- which is all
that matters here.

Three arms: a perfectly forced ice model, an ice model with the ocean frozen at
its initial state, and an ice model coupled to a predicted ocean.

**The expected ordering is `true ocean` < `frozen ocean` < `predicted ocean`,**
and it held on `siconc`, `sithick`, `usi`, `vsi` and Arctic ice-edge error with
the shipped 2000-step models. The gap between arm 1 and arm 3 is the error the
ice model *inherits* from the ocean model.

Re-measured on JURECA with the shipped checkpoints, day-10 `siconc` RMSE:

| arm | day-10 siconc RMSE | useful horizon |
|---|---|---|
| true ocean | 0.044483 | 23 days |
| frozen ocean | 0.044735 | 22 days |
| predicted ocean | 0.045429 | 18 days |

Day **1** is identical for arms 1 and 2 (0.010445), and that is the arithmetic
working: at *t=0* an ocean frozen at its initial state *is* the true ocean, so the
first step cannot tell them apart. If your arm 1 and arm 2 differ at day 1,
something is wrong with the forcing policy rather than with the ice model.

**Read the ordering off the per-variable RMSE tables, not off the loss.** Arms 1
and 2 fill only the `seaice` slot, so their `losses` in `summary.json` are the
sea-ice loss; arm 3 also fills `ocean`, so its loss covers the ocean variables
too. The two numbers are not on the same scale and the persistence baseline moves
with them -- 0.8855 for the ice-only arms against 2.2605 for arm 3. Compared
naively, arm 3 looks catastrophically worse (2.7151 against 0.6712) when it is
simply being scored on more variables. `report.md`'s RMSE tables are per variable
and are directly comparable; the loss is not.

**What to do with it.** Train a better ocean model and watch the gap close. If it
does not close, the coupling is not where your error is.

## 1c. `parallel` against `sequential` [GOOD FIRST]

**Difficulty:** easy to run, hard to get a big effect. **Time:** ~20 minutes.

```bash
make couple OCEAN=ocean_tiny SEAICE=seaice_tiny MODE=sequential
make couple OCEAN=ocean_tiny SEAICE=seaice_tiny MODE=parallel
```

With the shipped models the two modes differ in **all 59** deterministic labels
and by only **0.04%** on the loss. At a 24-hour step and this model's skill,
whether the ice sees today's ocean or yesterday's is a second-order effect.

**That is the interesting part, not a disappointment.** The experiment is to make
it matter:

* reverse the order (`--order seaice ocean`) so the ocean sees today's ice
  instead, and see whether the difference is symmetric;
* couple less often rather than more: `++module.inference.rollout_iterations`
  stays at one day, but you can compare a 10-day rollout against one where you
  deliberately hold the ice fixed for two days at a time. If a *coarser*
  exchange costs nothing either, the two systems really are that loosely coupled
  at this skill level, and that is the result;
* **you cannot shorten the coupling interval.** GLORYS ships **daily means**,
  every dataloader config sets `timedelta_hours: 24`, and `GlorysForecast`
  refuses anything that is not a whole multiple of it:

  ```
  ValueError: lead_time_hours=12 is not a whole multiple of timedelta_hours=24
  ```

  A sub-daily experiment needs sub-daily data, which this archive does not have;
* look at where the difference lives, not just its size. `zos` at day 1 is
  **bit-identical** between the modes because the ocean runs first, and diverges
  from day 2. Which variable diverges first, and how fast, tells you which
  coupling direction is actually carrying information.

## 1d. Forcing ablations: how much of the error is inherited?

**Difficulty:** medium. **Time:** ~1 hour.

The three-arm experiment from 1b, run on *your* models, and reported as a
decomposition:

| the ice model is given | reports |
|---|---|
| the true ocean at every step | the ice model's own error |
| the ocean frozen at *t=0* | + the cost of not knowing the ocean evolves |
| a predicted ocean | + the ocean model's error |

Two differences, each attributable to one thing. This is the cleanest experimental
design available in the kit and it costs no training.

**A real result** would be a sentence like: *"at day 10, 60% of our coupled
sea-ice error is inherited from the ocean model."* Nobody has computed that
number properly yet.

## 1e. A forced ocean: does telling the model the atmosphere help?

**Difficulty:** medium. **Time:** ~25 minutes for the pair on one GPU.
**Read [docs/05 section 5.6](05_coupling.md#56-external-forcing-the-three-routes-in-and-the-file-one)
first** -- the shipped atmosphere covers 2024 only, which is the **holdout**
split, so this experiment can demonstrate a mechanism and cannot produce a
reportable score.

This was written off as impossible in an earlier version of the kit ("520 time
steps but only 104 distinct times"). It is not: `time_counter` in those files is
the *initialisation* time and the archive is daily on its valid times. The
plumbing now works end to end.

```bash
make forcing-stats
COMMON="MODULE=tiny DATALOADER=glorys_forced"
# ++module.module.num_warmup_steps=100 matters: the preset warms up over 150 of
# its 4000 steps, and left alone a 1000-step run spends 15% of itself warming up.
ARGS="++max_steps=1000 ++seed=0 ++module.module.num_warmup_steps=100"
make train $COMMON NAME=unforced HYDRA_ARGS="forcing=none $ARGS"
make train $COMMON NAME=forced   HYDRA_ARGS="forcing=file $ARGS"

EV="--domain ifs_forced_val --lead-days 10 --n-inits 20 --init-selection first --skip-free-rollout"
make eval NAME=unforced EVAL_ARGS="$EV"
make eval NAME=forced   EVAL_ARGS="$EV --compare-with unforced=unforced"
```

These are the commands the numbers below came from, flag for flag.

**What we measured, so you do not repeat it:** at 1000 steps on 302 samples the
forced arm is **not better** -- 5.4% worse on the one-step loss, 1.1% better on
the ten-day rollout loss, and both arms lose to persistence on almost everything.
The full tables are in
[docs/05 section 5.6](05_coupling.md#what-a-short-forced-run-actually-measured).
That is a null result at a tiny budget with one run per arm, not evidence that
forcing cannot help, and it is exactly the sort of thing worth pushing on.

**Re-run on JURECA, and the null result holds -- but it is not uniform, which is
the useful part.** Same commands, `batch_size` 4 rather than the GH200 run's 8
(so these 1000 steps saw half the samples that one did; the two arms are
comparable to each other and not to the numbers above):

| | day 1 | day 10 |
|---|---|---|
| variables where forcing *helped* | 32 of 59 | 29 of 59 |
| variables where the forced arm beats persistence | 24 of 59 | **9 of 59** |

32 of 59 is a coin flip. But at day 10 the sign is not random across variables --
it splits by *which* variable:

| day-10 RMSE | unforced | forced | | persistence |
|---|---|---|---|---|
| `siconc` | 0.06691 | **0.06377** | **-4.7%** | 0.05621 |
| `vsi` | | | **-3.9%** | |
| `mlotst` | 26.849 | **26.526** | **-1.2%** | 21.457 |
| `thetao0m` | 1.30007 | 1.39593 | +7.4% | 0.67451 |
| `zos` | 0.13850 | 0.15200 | +9.7% | 0.06438 |

**Forcing helped the sea ice and the mixed layer, and hurt the surface height and
the SST.** That is the shape you would expect if the atmosphere is carrying real
information to the ice and the mixed layer while also destabilising a
sea-surface-height field that a 1000-step model has not learned to hold steady --
and it is a much better lead than the single whole-field number, which averages
the two into "no effect". Point 4 below is therefore not a nice-to-have: score
where the atmosphere should matter, because on the average it does not show.

Where to push, in rough order of expected value:

1. **More data, not more steps.** 302 samples is one year of one variable set.
   Bring your own atmosphere -- ERA5 on this grid -- for 1993-2018 and the
   experiment becomes a real one, on a real split, with a real baseline.
2. **Give the forcing somewhere useful to go.** It is concatenated onto the
   surface input channels and projected once. A separate encoder, or feeding it
   at every rollout step of a multi-step loss, are both cheap to try.
3. **Ablate it the way 1d ablates the ocean.** `PersistenceForcing` in
   `forcing.py` freezes the atmosphere at the initial time; a forced model
   rolled out with a frozen atmosphere against the same model with the true one
   separates "the model uses the atmosphere" from "the model was helped by
   knowing it".
4. **Score where the atmosphere should matter most** -- the mixed-layer depth,
   SST in the summer hemisphere, the sea-ice edge -- rather than on a
   whole-field average that a 10-metre wind cannot be expected to move.

---

# Question 2: implicit learning

*How much does a model learn about a system it is never asked to predict?*

## 2a. `seaice` against `seaice_isolated` [GOOD FIRST]

**Difficulty:** easy. **Time:** ~40 minutes including training.

Two sea-ice models. One reads the whole ocean state as input; the other has no
ocean channel at all. Same architecture, same steps, same targets.

```bash
make train MODULE=seaice_component          DATALOADER=glorys_seaice          NAME=ice_sees_ocean
make train MODULE=seaice_isolated_component DATALOADER=glorys_seaice_isolated NAME=ice_blind

P=".venv/bin/python -m oceanarches.evaluation.run_eval"
COMMON="--lead-days 10 --n-inits 16 --init-selection spread"
$P --coupled --components seaice=ice_sees_ocean     $COMMON --out evalstore/2a_sees_ocean
$P --coupled --components seaice_isolated=ice_blind $COMMON --out evalstore/2a_blind
```

(Separate `--out` again, for the reason given in 1b: without it these two arms
would share a directory with 1b's and overwrite each other's reports.)

**Read this one carefully; it is the easiest to over-claim.** It is *not* a clean
information ablation -- the two models have different input widths (11 channels
against 4) and different latent geometry (a vertical of 8 against 1), so at a
fixed step budget the smaller problem is simply easier to optimise. With the
shipped 2000-step pair, the blind model actually **won** on the ice drift
components for exactly that reason.

Where the comparison is least confounded is the **Southern Hemisphere ice edge**,
which the Southern Ocean drives, and there the blind model was clearly worst
(day-10 Antarctic IIEE 1.4092 against 1.2536-1.2718 for every arm that could see
an ocean).

**Make it cleaner.** Train the blind model for longer until its *training* loss
matches the other's, then compare validation. Or keep the architecture identical
and feed the ocean channels as pure noise -- then the only difference is the
information, not the shape.

## 2b. Linear probes on a frozen encoder

**Difficulty:** hard. **Time:** most of a day. **The most publishable idea here.**

Does a sea-ice-only model build an internal representation of the ocean without
ever being told about it?

Method:

1. Train `seaice_isolated` (no ocean input at all).
2. Freeze it. Take the encoder's output -- the token grid -- for a few thousand
   states. **Check its shape rather than assuming it**: the latent vertical
   depends on the component. `seaice_isolated` has no 3-D variables at all, so
   its grid is `(batch, emb_dim, 1, 60, 120)`; `seaice`, `ocean` and `full` carry
   13 depth levels and give `(batch, emb_dim, 8, 60, 120)`. Measured on the
   shipped checkpoints at `emb_dim=96`:

   ```
   seaice_isolated_tiny   tensor_size [1, 60, 120]   tokens (4, 96, 1, 60, 120)
   seaice_tiny            tensor_size [8, 60, 120]   tokens (4, 96, 8, 60, 120)
   ```

   That factor of 8 is the feature dimension of your probe, so getting it wrong
   is not a detail.
3. Fit a **linear** map from those tokens to an ocean field the model never saw:
   sea surface temperature, or mixed layer depth, at the same time.
4. Score it against two controls: the same probe on an untrained encoder
   (random weights), and the same probe on the `seaice` encoder, which *did* see
   the ocean.

If the trained-but-blind encoder beats the random one substantially, the model
has inferred ocean structure from ice alone -- which it can only have done
through the ice's own dynamics.

The hook you need, verbatim and runnable:

```bash
.venv/bin/python -c "
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset
from geoarches.lightning_modules.base_module import load_module
from oceanarches.evaluation import rollout

module, cfg = load_module('modelstore/seaice_isolated_tiny', device='cpu')
module = module.eval().requires_grad_(False)          # frozen

dataset = instantiate(cfg.dataloader.dataset, domain='tiny_val', multistep=1)
batch = next(iter(DataLoader(Subset(dataset, range(4)), batch_size=4,
                             collate_fn=rollout.collate_fn)))

with torch.no_grad():
    tokens = module.embedder.encode(batch['state'], batch['prev_state'])
print('tokens', tuple(tokens.shape))
print('latent vertical:', tokens.shape[2], 'for component', module.component.name)
"
```

```
tokens (4, 96, 1, 60, 120)
latent vertical: 1 for component seaice_isolated
```

**`encode` takes two states, not one.** Every shipped preset sets
`n_concatenated_states: 1`, meaning the model also reads the state one lead time
earlier, and the embedder says so rather than guessing:

```
ValueError: This embedder was built with n_concatenated_states=1 and expects the
previous state, but encode() got cond_state=None. Either build the dataloader
with load_prev=True or set n_concatenated_states=0.
```

**Keep the probe linear.** A deep probe tells you the information could be
recovered by some network, not that this one represents it.

**Controls are the whole experiment.** A probe with no random-init baseline
proves nothing: the input fields alone carry a lot of information, and a random
projection of them still does.

## 2c. What does the joint model do with its capacity?

**Difficulty:** medium. **Time:** half a day.

A `full` model predicts ocean and ice from one backbone. Does it share machinery
between them or partition it?

Cheap probes:

* **Gradient attribution.** Backpropagate the `siconc` loss alone and look at
  which encoder channels get gradient; repeat for `thetao`. Overlapping or
  disjoint?
* **Perturbation.** Change `siconc` in the input by a small amount and measure
  how much the *ocean* output moves, and vice versa. This is exactly the probe
  used to test the vertical mixing in `tests/test_configs.py::depth_coupling` --
  copy it. One warning from that experience: on an untrained model geoarches
  zero-initialises the adaLN gates, so every block is exactly the identity and
  the probe reports **0.0** for reasons that have nothing to do with your
  question. Open the gates (set every `adaLN_modulation[-1].bias` to something
  non-zero) or use a trained model.
* **Ablate a channel.** Zero `siconc` in the input at inference and see how much
  the SST forecast degrades.

---

# Question 3: higher resolution

## 3a. Take the same configs to 0.25 degrees

**Difficulty:** hard, and mostly engineering. **Time:** a full day, at least.

**Three years are staged and waiting**, raw and unprepared, at

```
/p/scratch/training2635/4_ocean_ai/nowak2/glorys_025/{2017,2018,2019}
```

876 GB of it. Three *consecutive* years, chosen against `SPLIT_YEARS` in
`oceanarches/dataloaders/glorys.py` so the shipped splits work without editing
them: **2017 and 2018 are training years and 2019 is the validation year** --
and 2019 is exactly `tiny_val`, while 2017-2018 sit inside `tiny_train`
(2014-2018). So `dataloader=glorys_tiny` works on this data as it stands.

**There is no test year**, because a run of three cannot reach one: `test` starts
at 2021 and `val` is two years wide. Score with `--domain val`:

```bash
make eval NAME=my_025_run EVAL_ARGS="--domain val"
```

The full archive is 1993 onwards and roughly 10 TB, so if you need more years,
**ask a tutor** -- another year is 292 GB and about eight minutes to copy, and
there is room for it.

Verified from a real file:

```
{'time': 1, 'lat': 720, 'lon': 1440, 'depth': 50}
['mlotst', 'zos', 'bottomT', 'sithick', 'siconc', 'usi', 'vsi', 'thetao', 'so', 'uo', 'vo']
858584012 bytes per day
```

Same eleven variables, same 50 levels, **16 times the horizontal cells** and
**859 MB per day**. One year is about 310 GB raw.

Here is what will actually need to change, honestly:

**Preparation, and this is the part that bites first.**
`scripts/prepare_glorys.py` takes `--raw-root` and `--out-root` (there is no
`--data-root`), so pointing it at the 0.25-degree tree is easy:

```bash
.venv/bin/python scripts/prepare_glorys.py --years 2017 \
    --raw-root /p/scratch/training2635/4_ocean_ai/nowak2/glorys_025 \
    --out-root data/glorys_025_prepped
```

**But it will not work as it stands**, and here is exactly how it fails:

```
ValueError: operands could not be broadcast together with remapped shapes
[original->remapped]: (720,1440)  and requested shape (1,180,360)
```

The script does not regrid -- the shipped 1-degree archive was regridded upstream
with `cdo remap,r360x180`, which is what the prepared files record as their
`source` attribute. `prepare_glorys.py` writes its dimensions from
`N_LAT, N_LON = 180, 360` in `variables.py` and is then handed 720x1440 arrays.
So you have a choice, and it is the same choice §3b faces:

* **change the grid constants** `N_LAT`, `N_LON`, `LAT` and `LON` in
  `variables.py` and rebuild everything downstream of them (masks, statistics,
  climatology); or
* **regrid the 0.25-degree data down** to something the existing constants
  describe, which defeats the purpose.

Take the first. It is a small edit and a large blast radius: `variables.py` is
the single source of truth, so every mask, statistic, metric and plot follows it,
and all of them have to be regenerated.

**The edit itself, read off a real staged file** (`2017/01/...`, checked against
`variables.py` rather than guessed):

```python
# oceanarches/dataloaders/variables.py, replacing lines 29-31
N_LAT, N_LON = 720, 1440
LAT = [-89.875 + i * 0.25 for i in range(N_LAT)]   # -89.875 ... 89.875
LON = [0.0 + i * 0.25 for i in range(N_LON)]       #   0.000 ... 359.750
```

**The vertical and the variable set really are unchanged**, and this has now been
run rather than reasoned about. The 0.25-degree files carry the same eleven
variables and the same 50 depth levels, and `PREPPED_DEPTH_INDICES` resolves to
the identical depths -- confirmed by reading them back off a prepared
0.25-degree file: 0.5, 5.1, 15.8, 29.4, 55.8, 92.3, 155.9, 222.5, 318.1, 453.9,
643.6, 902.3, 1245.3, 1684.3 m. Nothing about the vertical, the variable set or
the masking logic changes.

**It was not, however, the only edit.** `variables.py` is the single source of
truth for the grid, and two scripts quietly did not read it: both
`prepare_glorys.py` and `compute_stats.py` took `N_LAT`/`N_LON` from it for their
array *shapes* but wrote the coordinate *values* from a hardcoded
`np.arange(-89.5, 90.0, 1.0)`. Changing the constants therefore got you

```
ValueError: shape mismatch ... arg 0 with shape (720,) and arg 1 with shape (180,)
ValueError: operands could not be broadcast together ... (180,1) and requested shape (720,1440)
```

from inside netCDF4 and numpy, several layers below anything that mentions a
grid. **Both now read `LAT` and `LON` from `variables.py`, so this is fixed** and
the grid-constants edit above is genuinely all the preparation needs. The
prepared file's `title` and `source` are derived too, so a 0.25-degree file no
longer claims to be "regridded to 1 degree via cdo remap,r360x180".

**Measured, preparing a full year (2019) on one dc-gpu node:**

| | |
|---|---|
| one year | **34.9 min**, steady at 0.2 day/s |
| output | **41.74 GB** for 365 days at 14 levels, i.e. 114 MB/day |
| whole archive | ~1.4 TB, so the 1.5 TB estimate above is right |
| `compute_stats.py` | its climatology accumulators scale with the **grid**, not with the number of years, so more years cost time and not memory. Measured at **14.8 GiB** in the parent process. But `--jobs N` holds a copy of them per worker -- the docstring in the script puts one accumulator at 350 MB on 1 degree, so 5.6 GB each here -- so divide your memory budget by `N` before choosing it. |

Three years is therefore about **1.7 hours and 125 GB**, which is a coffee break
rather than the day the rest of this section needs.

**One thing that is NOT fixed, and cannot be locally: the prepared files are
still named `glorys_1deg_<year>.nc`, and the statistics are still
`glorys_1deg_masks.nc`, `glorys_1deg_stats.pt` and
`glorys_1deg_climatology.nc`.** That prefix is baked into the discovery globs in
`oceanarches/doctor.py` and `oceanarches/dataloaders/glorys.py` and into
`oceanarches/paths.py`, so renaming it is an eight-file change and not part of
this experiment. The consequence matters: **a 0.25-degree run and a 1-degree run
cannot share a checkout.** The masks and statistics have the same filenames and
different shapes, so building one silently invalidates the other. Do the whole
experiment in a **separate copy of the repo with its own `config.env`**
(`DATA_ROOT`, `GLORYS_PREPPED`) and its own `oceanarches/stats/`; that is how the
numbers on this page were produced, and it keeps your working 1-degree setup
alive while you break things.

**Patch size and the latent grid.** The embedder patches `(2, 3, 3)`, so 1 degree
gives a latent grid of `8 x 60 x 120`. At 0.25 degrees the same patch gives
`8 x 240 x 480`, which is **16x the tokens** and therefore roughly 16x the
attention cost and memory. Two options:

* increase the horizontal patch to `(2, 12, 12)`, giving the identical
  `8 x 60 x 120` latent grid and therefore the identical cost, at the price of a
  much coarser token;
* keep the patch and pay for it.

**The arithmetic checks out for both**, so this is a cost decision and not a
correctness one -- worked through rather than left to you:

| patch | latent grid | `window_size [1,6,10]` divides? | downsampled (halved) divides? |
|---|---|---|---|
| `(2, 3, 3)` (shipped) | 240 x 480 | yes (240/6, 480/10) | yes (120/6, 240/10) |
| `(2, 12, 12)` | 60 x 120 | yes | yes |

The first is 16x the tokens of 1 degree, and therefore roughly 16x the attention
cost and memory. The second is free, and blunter.

**`patch_size` is not the only thing to change if you take the first option.**
`module.backbone.tensor_size` is pinned to `[8, 60, 120]` in every preset
(`configs/module/tiny.yaml`), and it is the latent grid the backbone asserts
against. Keep the shipped patch at 0.25 degrees and leave it alone and you get

```
AssertionError: input feature has wrong size
```

from inside geoarches, naming nothing. Set it to the latent grid your patch
actually produces -- `++module.backbone.tensor_size=[8,240,480]` for the shipped
`(2, 3, 3)`. With `(2, 12, 12)` the latent grid stays `8 x 60 x 120` and the
shipped value is already right, which is why that route works untouched.

**The depth axis is untouchable.** 12 or 13 levels, latent depth 8, whatever the
horizontal resolution -- see
[docs/04](04_scaling_finetuning.md#42-the-vertical-is-not-a-scaling-axis).

**Memory, measured on a JURECA dc-gpu A100 (39.5 GiB usable), `tiny` at
0.25 degrees.** These are `make benchmark` runs, forward + backward + optimiser,
compute only:

| patch | `tensor_size` | latent grid | batch | peak | s/step | 4000 steps |
|---|---|---|---|---|---|---|
| `(2, 3, 3)` | `[8,60,120]` (shipped) | — | 1 | **AssertionError** | — | — |
| `(2, 3, 3)` | `[8,240,480]` | 240 x 480 | 1 | **OOM** | — | — |
| `(2, 3, 3)` | `[8,240,480]` + grad ckpt | 240 x 480 | 1 | **OOM** | — | — |
| `(2, 12, 12)` | `[8,60,120]` | 60 x 120 | 1 | 6.78 GiB | 0.127 s | 8 min |
| `(2, 12, 12)` | `[8,60,120]` | 60 x 120 | 2 | 13.32 GiB | 0.182 s | 12 min |
| `(2, 12, 12)` | `[8,60,120]` | 60 x 120 | 4 | **OOM** | — | — |

**So on JURECA, "keep the patch and pay for it" is not a cost decision -- it is
not available.** Not at batch 1, and not with `gradient_checkpointing: True`
either. An earlier version of this page projected "batch 1, gradient
checkpointing, probably four GPUs" from the 96 GB GH200 the kit was built on; on
a 40 GiB card batch 1 does not fit, and **four GPUs do not fix it**, because
Lightning's DDP replicates the whole model on every rank rather than sharding it.
Sharding the parameters, gradients and optimiser state (FSDP, ZeRO-3) would buy
back only a few GiB of the total -- 459.6M params is the `large` preset, and
`tiny` here is 15.8M -- because what does not fit is the **activations**, and
those are per-rank whatever you shard.

The route that works on this machine is the blunt one: **patch `(2, 12, 12)`, at
batch 1 or 2.** Note the embedder grows from 0.17M to 2.74M parameters (a 12x12
patch projection instead of 3x3), so the preset is 15.8M rather than 13.2M.
Note also that peak memory is *not* linear in the batch here -- 6.78 GiB at
batch 1, 13.32 at batch 2, and batch 4 does not fit in 39.5 -- because the
decoder reconstructs the full 720 x 1440 field, so measure rather than
extrapolate.

**The dataloader will become the bottleneck**, which it is not at 1 degree
(3% data tax with 16 workers). Each sample is 16x larger. Measure it before
blaming the GPU.

**The masks change qualitatively.** More coastline, more small seas, more
straits -- and the sea-ice extent problem from
[docs/02](02_data_and_masking.md#214-one-more-thing-sea-ice-extent-on-this-grid-runs-high)
should shrink substantially, because the 15% smear is a regrid artefact. That is
itself a nice measurement: **compute the Arctic March maximum at 0.25 degrees and
see how far it moves toward the published 14-16.**

**A realistic day's target** is not a trained 0.25-degree model. It is: prepare
two years, rebuild the masks and statistics, get one forward and backward step to
run inside memory, and report the measured step time and the projected wall
clock. That is a genuinely useful result and it is what the next team would need.

**That target has now been reached**, end to end: 2019 prepared in full (34.9 min,
41.74 GB), 24 days of 2017 as a training year, masks and statistics and
climatology rebuilt at 0.25 degrees, `make doctor` green, and `make train` running
real steps through the real dataloader with a checkpoint on disk. The two tables
above are the memory and compute. Three results from it:

**1. The dataloader is indeed the bottleneck, and it is not close.** Measured with
`make benchmark BENCH_ARGS="--data"` at patch `(2, 12, 12)`, batch 2:

| | 1 degree | 0.25 degrees |
|---|---|---|
| compute only | 0.307 s/step | 0.182 s/step |
| fed by the dataloader | +3% | **0.766 s/step** |
| the data tax | 3% | **4.2x the compute -- 76% of the step** |

So at 0.25 degrees you are not waiting for the GPU. Every hour of optimisation
spent on the model is an hour spent on the wrong 24%.

**2. A real `make train` step is much slower than either of those, and we could
not cleanly attribute it.** The observed rate was well below the benchmark's
0.766 s/step, but on a 22-sample training split with validation every 11 steps a
12-to-20-step run is dominated by start-up and by epoch boundaries, and turning
`compute_acc` off made the *measured* rate worse rather than better -- which is
the signature of a measurement that is not measuring steps at all. **So this is an
open question, not a result.**

It is worth chasing, because the mechanism is plausible: geoarches recomputes every
training metric on every step, ACC interpolates the monthly climatology per
sample, and that climatology is **1366 MB** at 0.25 degrees against 95 MB at
1 degree. At 1 degree the metrics cost 0.09 s/step
([docs/04](04_scaling_finetuning.md)). **To settle it, prepare a full training
year first**, then time 200+ steps with `compute_acc` on and off. Do not trust a
short run on a 22-sample split -- we tried, and it told us nothing.

**3. The masks did move.** The surface ocean fraction comes out at **67.0%** at
0.25 degrees against **69.6%** at 1 degree -- land goes from 30.4% to 33.0% of the
grid as the finer coastline resolves. The sea-ice extent measurement below is the
interesting follow-on, and now costs nothing to try.

What is left for a team that wants to go further: prepare 2017 and 2018 in full
(~35 min each), and train properly. Budget from a *measured* step time on that
data rather than from the benchmark's compute-only figure -- the 4.2x data tax
above is the floor, not the estimate.

## 3b. Regional at high resolution

**Difficulty:** hard. **Time:** a full day, and it is **not** a beginner task --
it needs everything 3a needs (new grid constants, re-prepared data, rebuilt masks
and statistics, a patch size that divides) *plus* a decision about what enters
through the open boundaries. It is cheaper than 3a only in GPU hours, not in
work.

Instead of the globe at 0.25 degrees, take one region -- the Arctic, or the
Southern Ocean, or the North Atlantic -- at full resolution. The cost drops back
to something a single GPU handles, and sea ice is a regional problem anyway.

You will need to change the grid constants in `variables.py` (`N_LAT`, `N_LON`,
`LAT`, `LON`), rebuild the masks and statistics for the crop, and check that the
new grid divides by the patch size and the window size. Beware the boundaries:
a regional model has to be told what comes in from outside, and the honest cheap
answer is to forecast only the interior.

---

## A summary table

| # | experiment | difficulty | GPU time | good first? |
|---|---|---|---|---|
| 1a | joint model vs two specialists | easy | ~1 h | **yes** |
| 1b | coupled vs uncoupled | easy | ~20 min, no training | **yes** |
| 1c | `parallel` vs `sequential` | easy | ~20 min | yes |
| 1d | forcing ablation / error decomposition | medium | ~1 h | |
| 1e | a forced ocean (prescribed atmosphere) | medium | ~25 min | |
| 2a | `seaice` vs `seaice_isolated` | easy | ~40 min | **yes** |
| 2b | linear probes on a frozen encoder | hard | a day | |
| 2c | what the joint model does with capacity | medium | half a day | |
| 3a | 0.25 degrees, globally | hard | a day+ | |
| 3b | 0.25 degrees, one region | hard | a day | |

And two that are not on the list above but are the highest-value things in the
kit, because the shipped model is measurably bad at both:

| # | experiment | difficulty | GPU time |
|---|---|---|---|
| 4a | **stop the 90-day rollout diverging** ([docs/06](06_evaluation.md#65-the-90-day-free-run-and-what-it-tells-you)). Multi-step training, harder clamping, a drift penalty, noise on the inputs. Measure it with `--free-days 90` before and after. | medium | ~1 h per attempt |
| 4b | **make deep salinity better than persistence** ([docs/06](06_evaluation.md#the-two-places-it-loses-which-you-should-go-after)). It is currently 11% worse at day 1 and 19% worse at day 10. Look at the loss weights and at what `add_input_state` gives you for free. | easy | ~40 min |

---

[< evaluation](06_evaluation.md) | [the cheatsheet >](cheatsheet.md)
