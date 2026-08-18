# 5. Coupling: two models, one forecast

[< scaling and fine-tuning](04_scaling_finetuning.md) | [next: evaluation >](06_evaluation.md)

Read time: 15 minutes. Running the whole worked example: about 60 minutes of GPU
time.

---

## 5.1 What "coupled" means, and why anyone cares

The ocean and the sea ice are two different physical systems that talk to each
other constantly. Ice insulates the ocean from the atmosphere and reflects
sunlight; when it melts it dumps fresh water into the surface layer; when it
forms it dumps salt. In the other direction, ocean currents drag the ice around
and warm water melts it from below.

Real climate models are built as **separate components** -- an ocean model, a
sea-ice model, an atmosphere model -- that exchange fields every timestep through
a coupler. This is not only a physics choice, it is a software one: different
groups build different components, and a component can be swapped without
rewriting the rest.

The interesting question for machine learning is whether that separation is a
good idea for *learned* models too:

* One model that predicts everything can learn any relationship it likes between
  ocean and ice, at the cost of spending its capacity on everything at once.
* Two specialists each get the whole network for their own problem, but can only
  see each other through the fields they exchange -- and each inherits the
  other's errors.

Nobody knows which wins here. That is one of the three open questions of this
challenge.

## 5.2 How it works in this kit

The single property everything rests on:

> `OceanForecastModule.forward_multistep` is a **grid-space** rollout. Every step
> produces a full physical state on the 180x360 grid. Nothing is carried between
> steps in a latent space.

Because of that, one model's prediction is an ordinary field, and an ordinary
field can be written into a shared state that another model reads next step. Two
independently trained models can therefore be coupled **with no retraining and no
shared encoder**. They only have to agree about the variable list in
`variables.py`.

### A "component" is which variables you predict and which you only read

`ComponentSpec` in
[`variables.py`](../oceanarches/dataloaders/variables.py) is two lists:

| component | predicts (prognostic) | reads only (forcing) |
|---|---|---|
| `full` | all 11 variables | -- |
| `ocean` | `zos mlotst bottomT thetao so uo vo` | `siconc sithick usi vsi` |
| `seaice` | `siconc sithick usi vsi` | `zos mlotst bottomT thetao so uo vo` |
| `seaice_isolated` | `siconc sithick usi vsi` | -- |

The dataloader config picks the component, and the module, the embedder's channel
counts, the loss and the metrics all read `${dataloader.component}`. So
`dataloader=glorys_seaice` rewires all of them at once and they cannot disagree.

### The shared state and the router

The coupled system holds **one** state in **one** channel order -- the *union
component*, derived automatically. `ocean` + `seaice` comes back as exactly
`full`, in canonical order.

`StateRouter` does the index bookkeeping and nothing else:

```python
router  = StateRouter.for_layout(COMPONENTS["full"])
ice_in  = router.gather(shared, COMPONENTS["seaice"])   # its own channels, its own order
shared  = router.scatter(shared, COMPONENTS["seaice"], ice_out)  # write back what it predicts
```

`gather` selects a component's inputs in the order its embedder was wired for;
`scatter` writes back only its prognostic channels and passes everything else
through untouched.

That the coupled system's state is channel-for-channel the state
`configs/dataloader/glorys.yaml` already loads is the reason the entire
evaluation pipeline scores it with **no special case anywhere**.

### Two ways to step

```python
visible = merged = shared_state
for key in order:                                       # e.g. ocean, then seaice
    prediction = components[key](gather(visible, spec))
    merged = scatter(merged, spec, prediction)
    if mode == "sequential":
        visible = merged
```

* **`sequential`** (the default): the components run in order, and later ones see
  the fields earlier ones have already updated. The ice responds to *today's*
  ocean.
* **`parallel`**: everybody reads the state at time *t*. This is what running two
  models side by side gives you.

### Channels nobody predicts

Run the sea-ice model on its own and the seven ocean channels it reads are
nobody's output. They are **never zeroed** -- zero in normalised space asserts
"the climatological mean everywhere", which is a much stronger claim than "I do
not know". Instead there is an explicit policy:

* `--unpredicted-forcing persistence` (default): hold them at their initial
  value. A genuinely free-running forecast -- no information from outside the
  system enters the rollout.
* `--unpredicted-forcing ground_truth`: read them from the dataset at each valid
  time. A **perfect-forcing** experiment, which separates a component's own error
  from the error it inherited.

Training a specialist needs neither: training is single-step, so a specialist's
forcing channels come straight off the dataloader and are ground truth by
construction.

## 5.3 The worked example, in five commands

This is the thing to copy.

```bash
# 1. two specialists, about 16 minutes each on one GPU
make train MODULE=ocean_component            DATALOADER=glorys_ocean            NAME=ocean_tiny
make train MODULE=seaice_component           DATALOADER=glorys_seaice           NAME=seaice_tiny
make train MODULE=seaice_isolated_component  DATALOADER=glorys_seaice_isolated  NAME=seaice_isolated_tiny

# 2. run the pair together: figures, animations and a report, about 270 s
make couple OCEAN=ocean_tiny SEAICE=seaice_tiny
make couple OCEAN=ocean_tiny SEAICE=seaice_tiny MODE=parallel

# 3. the three-way sea-ice question, on identical samples
P=".venv/bin/python -m oceanarches.evaluation.run_eval"
COMMON="--lead-days 10 --n-inits 16 --init-selection spread"
$P --coupled --components seaice=seaice_tiny --unpredicted-forcing ground_truth $COMMON
$P --coupled --components seaice=seaice_tiny                                    $COMMON
$P --coupled --components seaice_isolated=seaice_isolated_tiny                  $COMMON

# 4. put them all on one axes
$P --coupled --components ocean=ocean_tiny seaice=seaice_tiny --lead-days 10 \
   --compare-with "ice, true ocean=coupled_seaice_sequential_ground_truth" \
                  "ice, no ocean=coupled_seaice_isolated_sequential" \
                  "ice, frozen ocean=coupled_seaice_sequential"
```

**`--lead-days`, `--n-inits` and `--init-selection` must match across every arm**
or the numbers are not comparable. Every run records all three in its manifest,
so a mismatch is visible rather than silent.

Add `--skip-fields` to `COMMON` if you only want the numbers; it skips writing
`predictions.zarr` and `targets.zarr` and changes no metric.

The three specialist checkpoints ship with the kit, so you can go straight to
step 2 -- **but a fresh clone has none of them**, because `modelstore/` is
git-ignored. Symlink them into a `modelstore/` of your own first
([docs/01 section 1.1](01_setup.md#the-normal-route) step 3); without that, step 2
stops with `--components ocean=ocean_tiny: no such run`.

Everything in this section needs a GPU allocation. `make couple` warns and pauses
for ten seconds without one, in the same words `make train-*` uses.

Step 1 is 3 x ~16 min of training plus 2 x ~270 s of evaluation, about 57
minutes; step 2 alone is a few minutes.

## 5.4 The control that proves the plumbing

Before believing any coupled number, check that coupling one model to nothing
changes nothing. Build a "coupled system" out of a single `full` component and
compare it with that component alone:

```bash
.venv/bin/python -m oceanarches.evaluation.run_eval --exp task6_tiny --lead-days 10
.venv/bin/python -m oceanarches.evaluation.run_eval --coupled --components full=task6_tiny --lead-days 10
```

| | `--exp task6_tiny` | `--coupled --components full=task6_tiny` |
|---|---|---|
| loss, model | 2.11447811126709 | 2.11447811126709 |
| loss, persistence | 2.2604580521583557 | 2.2604580521583557 |
| loss, climatology | 48.74854373931885 | 48.74854373931885 |
| all 7620 (label x lead x forecaster) entries | -- | maximum difference **0.000e+00** |

Bit-identical. And it is not vacuous -- instrumenting the router during a real
rollout shows `gather` firing twice per step and `scatter` once, through real
`index_select` and `index_copy` calls.

The same check on the sea-ice specialist, where the router is a genuine
permutation (it reads 11 channels and predicts 4), also gives `0.000e+00` on
every label.

## 5.5 What the measurements actually said

All of the numbers below come from the four 2000-step throwaway specialists.
**Their job was to prove the plumbing. Draw no skill conclusions from them.**

### `sequential` against `parallel`

Same two checkpoints, same 16 initialisations, 10 days, `test` split. Only the
mode differs.

| | sequential (ocean, then ice) | parallel |
|---|---|---|
| module loss | 2.71541029 | 2.71438372 |
| all 59 deterministic model labels | | **all 59 differ**, max 4.894e-03 |
| persistence and climatology labels | | **0.000e+00** -- the control |

Three things worth reading off that:

1. The modes genuinely differ -- at every lead time, in every model label -- but
   by only ~0.04% on the loss and ~0.4% on day-10 `siconc`. **At a 24-hour step
   and this model's skill, whether the ice sees today's ocean or yesterday's is a
   second-order effect.** Halve the step, or couple a variable with a shorter
   memory, and it should grow. That is a real experiment for a team to run.
2. `zos` at day 1 is **bit-identical** in both modes and only diverges from day 2
   (1.9e-06, then 3.7e-06, then 6.2e-06...). That is the ordering doing exactly
   what it says: the ocean runs first, so at the first step it reads the same
   state either way, and the difference reaches it only through the ice fields it
   reads on the *next* step.
3. The baselines are identical between the two runs. The difference is in the
   forecast, not in the scoring.

Here `parallel` is very slightly better. With 2000-step models that is well
inside what retraining would move. It is a measurement, not a claim.

### How much does the ice get from the ocean?

Four arms, same initialisations, same days, same split. Arms (b), (c) and (d)
are the **same weights**; only what they are told about the ocean changes. Arm
(e) is a **different model** that never had an ocean channel to read.

| arm | how the ocean reaches the ice | day-10 `siconc` RMSE | day-10 Arctic IIEE | day-10 Antarctic IIEE |
|---|---|---|---|---|
| (a) coupled to `ocean_tiny`, sequential | predicted, every step | 0.0454395 | 1.2531 | 1.2668 |
| (b) ground-truth ocean | read from the dataset | **0.0444834** | **1.2391** | **1.2536** |
| (c) frozen ocean | held at the initial state | 0.0447349 | 1.2481 | 1.2718 |
| (e) no ocean at all | there is no ocean channel | 0.0447550 | 1.2496 | 1.4092 |
| persistence | | 0.0515832 | 1.3749 | 1.5835 |

**The ordering (b) < (c) < (a) holds** on `siconc`, `sithick`, `usi`, `vsi` and
on Arctic ice-edge error. Read that carefully, because it is the coupling result
and it is not the comfortable one:

> The same sea-ice weights do **best** when handed the true ocean, **worse** when
> the ocean is frozen at its initial state, and **worst** when the ocean comes
> from a model.

**A coupled system inherits its weakest component's error.** Here that is stark:
`ocean_tiny` at 2000 steps *loses to persistence* (loss 4.0496 against 3.1571 on
the same 16 initialisations), so coupling the ice to it is worse than telling the
ice that nothing changed. **Check each component against persistence before you
believe a coupled score.**

Arm (e) is **not** a clean information ablation, and must not be read as one. It
is different weights: 4 input channels instead of 11, a latent vertical of 1
instead of 8, the same 2000 steps. At this budget the smaller problem is simply
easier to optimise, which is why (e) wins on the ice-drift components. Where the
comparison is least confounded -- the Southern Hemisphere ice edge, which the
Southern Ocean drives -- (e) is clearly **worst**: day-10 Antarctic IIEE 1.4092
against 1.2536-1.2718 for every arm that can see an ocean.

The honest summary: *ocean information helps the ice edge, most visibly in the
Antarctic; on the interior pack at 10 days it is within the noise of how these
two models were trained.*

## 5.6 External forcing: the three routes in, and the file one

Component-to-component exchange goes through the shared state. A **prescribed
atmosphere** is a different thing, and it stays per component. There are exactly
three routes a component can get forcing, and they are not interchangeable:

| route | what it is | where it lives | when to use it |
|---|---|---|---|
| **nothing** | the model predicts the ocean from the ocean | `forcing=none`, the default | almost always; it is the best-tested path and every shipped baseline uses it |
| **another model** | a coupled component writes into the shared state between steps | `oceanarches/lightning_modules/coupled.py` (5.2-5.5 above) | ocean <-> sea ice, and anything else *inside* the system you are modelling |
| **a file** | fields read off disk and concatenated onto the model's input channels | `forcing=file` -> `XarrayForcing` | a prescribed atmosphere, i.e. something *outside* the system, whose future you are handed rather than predicting |

The first two are the ones you will use. The third is documented here because it
now works end to end on the shipped archive, and because the archive has one
property that will bite you if nobody tells you about it.

### What the shipped archive actually is

`IFS_FORCING` is `/p/scratch/training2635/4_ocean_ai/nowak2/hackathon-ocean-sea-ice/data/ifs_1deg`:
52 weekly files, 1.1 GB, already on our exact 180x360 grid (lat -89.5..89.5
ascending, lon 0..359), so no regridding.

Read naively it looks useless. All 52 files together hold **520 records at only
104 distinct `time_counter` values**, and an earlier version of this document
concluded from that the archive was weekly and could not be trained on. That was
wrong, and it is worth knowing why, because the same mistake is one `ncdump -h`
away in any forecast archive:

> **`time_counter` is the *initialisation* time, not the valid time.** These are
> IFS forecasts. Every record also carries `leadtime`
> (`standard_name: forecast_period`) of 13, 37, 61, 85, 108, 132, 156, 180 or
> 204 hours, and the time a field describes is `time_counter + leadtime`.

On valid times the same 520 records are **478 distinct times covering all 366
calendar days from 2024-01-03T01:00 to 2025-01-02T00:00** -- one or two per day,
with no gaps. The archive is daily.

Two consequences, both handled and both worth stating:

* **Several forecasts are valid on the same day at different leads.** Counting
  records: 234 days carry one, **110 carry two and 22 carry three** -- 132 days
  with more than one. `XarrayForcing` keeps one record per distinct valid time
  and the **shortest lead wins**, because it is the most accurate forecast of
  that instant. 42 of the 520 records are dropped that way, which leaves 254
  days with a single valid time and 112 with two.
* **Valid times fall at 00:00 and 01:00; GLORYS daily means are stamped 12:00.**
  A request is matched to the nearest valid time, a tie goes to the *earlier*
  one, and anything further away than `tolerance_hours` (12 by default) is
  refused. Measured over the whole archive, all 366 covered days resolve to a
  field valid on that same calendar day, and the two days before the first
  forecast -- 1 and 2 January 2024 -- are refused rather than served stale.

It is still a *forecast* atmosphere, not an analysis: the field for a given day
comes from an initialisation up to 8.5 days earlier. For a plumbing
demonstration that does not matter. For a scientific result it would, and you
would want ERA5.

### One channel of the eight is empty

The archive has eight atmospheric variables. `configs/forcing/file.yaml` uses
**seven**:

```
sowinu10  sowinv10   10 m wind, eastward and northward
sotemair  sod2m      2 m air temperature and dewpoint
sowaprec             total precipitation
sosudosw  sosudolw   downward short- and long-wave radiation
```

`somslpre` (mean sea-level pressure) is **NaN over the whole grid in all 520
records of all 52 files**. Fed to the model it would survive `nan_to_num` as a
constant zero: a channel that looks like forcing, costs an embedder channel and
carries nothing. `XarrayForcing` now refuses an all-NaN channel at construction,
so this fails loudly instead of quietly.

The seven that are there are NaN on the nine southernmost latitude rows
(-89.5..-81.5, i.e. the Antarctic interior), and `sosudosw` has scattered extra
gaps in 24 of the 520 records. Those become 0 after normalisation, which is "the
channel mean" -- the same convention the state uses over land.

### Running it

```bash
make forcing-stats          # once: oceanarches/stats/ifs_1deg_forcing_stats.pt

make train MODULE=tiny DATALOADER=glorys_forced NAME=my_forced \
    HYDRA_ARGS="forcing=file ++max_steps=1000"
```

`forcing` is a hydra **group**, so a bare `forcing=file` on a `make` line would
be parsed by make as a variable assignment and dropped. Quote it into
`HYDRA_ARGS`, or call the entry point directly:

```bash
.venv/bin/python -m geoarches.main_hydra --config-path $PWD/configs \
    cluster=jupiter_1gpu module=tiny dataloader=glorys_forced forcing=file \
    ++name=my_forced ++max_steps=1000
```

Three things are wired together and all three have to agree:

* `configs/forcing/file.yaml` sets `n_channels`, and the embedder reserves
  exactly that many through `embedder.forcing_ch: ${forcing.n_channels}`;
* the module gets the source through `module.forcing: ${forcing.source}` and
  fetches one field per batch element per step in
  `OceanForecastModule.external_forcing`;
* `configs/dataloader/glorys_forced.yaml` restricts the sample window to the days
  the archive covers.

`make forcing-stats` writes the normalisation. It is separate from `make stats`
because the atmosphere is not part of the model state, is never scored, and is
averaged over **land as well as ocean** -- the ocean-only rule that
`scripts/compute_stats.py` uses for GLORYS would give a 2 m temperature several
kelvin wrong over the continents. `configs/forcing/file.yaml` pins the file, so
two runs normalise the atmosphere identically; without it `XarrayForcing` derives
statistics from a 64-time sample of whatever it is pointed at, which is fine in a
notebook and not fine in a comparison.

### What a short forced run actually measured

Two `tiny` models, trained back to back on one GH200, **identically
configured except for the forcing**: `dataloader=glorys_forced`, 1000 steps, batch 8,
`seed=0`, 302 training samples (2024-01-03..2024-10-30), the same 29 validation
samples (2024-11-01..2024-11-29). **Both splits are inside 2024, the holdout
split. These are not skill results.**

Wall clock: 617 s unforced, 595 s forced. The forcing costs less than the
run-to-run variation at this batch size.

**One step, the module's own loss, on the same 29 validation samples through the
same code:**

| | unforced | forced | change |
|---|---|---|---|
| one-step val loss | **1.698157** | **1.789886** | +5.40% (worse) |
| SST RMSE (K) | 0.176419 | 0.205662 | +16.58% (worse) |
| `zos` RMSE (m) | 0.031858 | 0.031933 | +0.24% (worse) |
| `siconc` RMSE | 0.019848 | 0.019781 | -0.34% (better) |

**Ten-day rollout, 20 initialisations, `run_eval` with the same `--domain`,
`--lead-days`, `--n-inits` and `--init-selection` for both, and the same
baselines through the same metric code:**

| RMSE at day 1 | unforced | forced | persistence |
|---|---|---|---|
| SST (degC) | 0.17480 | 0.18781 | **0.12896** |
| `zos` (m) | 0.026486 | 0.027458 | **0.022808** |
| `siconc` | 0.012878 | 0.012901 | **0.012910** |
| `sithick` (m) | 0.015794 | 0.015928 | **0.013254** |
| surface salinity (1e-3) | 0.087149 | 0.088685 | **0.073234** |
| mixed-layer depth (m) | **9.902** | 10.003 | 9.9849 |

| RMSE at day 10 | unforced | forced | persistence |
|---|---|---|---|
| SST (degC) | 1.3207 | 1.3624 | **0.67451** |
| `zos` (m) | 0.14829 | **0.13695** | 0.064378 |
| `siconc` | 0.064528 | 0.064693 | **0.056211** |
| `sithick` (m) | 0.08877 | 0.093925 | **0.054607** |
| surface salinity (1e-3) | 0.5374 | 0.55589 | **0.28488** |
| mixed-layer depth (m) | 25.714 | 26.984 | **21.457** |

| module loss on the 10-day rollout | value |
|---|---|
| unforced | 6.6878 |
| forced | 6.6137 |
| persistence | **2.5595** |
| climatology | 65.060 |

**Read it like this.** The first thing to say is not about forcing at all: at
1000 steps **both** models lose to persistence on almost everything, at every
lead. They are 1000-step models on 302 samples; that is what such a model looks
like, and it is the same lesson section 5.5 draws about `ocean_tiny` at 2000
steps. Nothing here is a forecast anyone should use.

Against that, the forced arm is **not better**. It is 5.4% worse on the one-step
loss and 16.6% worse on day-1 SST; it is 1.1% better on the ten-day rollout loss
and 7.6% better on day-10 `zos`. The differences point both ways and are small
next to how far both arms are from persistence.

Two reasons not to read more into it than that:

* **One run per arm.** `seed=0` for both, but the forced model has seven more
  input channels, so its parameter tensors are a different shape and the random
  draw is not the same one. Arm-to-arm differences of a few per cent at this
  budget are not attributable to the forcing without a seed sweep, which was not
  run.
* **The atmosphere is a 4% perturbation.** Section 4.2 of the Task 12 report
  measures it: replacing the whole atmosphere with its mean moves this model's
  prediction by 4.1% of the field, against 156% for a one-sigma change in the
  ocean state. A residual one-day model is dominated by the state it is handed.
  Forcing would be expected to matter over long rollouts and for the surface
  fields the atmosphere drives, neither of which a 1000-step model reaches.

The honest summary: **the plumbing works, the atmosphere demonstrably reaches
the model, and at the only scale this kit can afford to test it does not help.**
Whether it helps at a real scale, on a real split, with a real (analysed rather
than forecast) atmosphere, is an open question and a good one --
[docs/07 idea 1e](07_challenge_ideas.md#1e-a-forced-ocean-does-telling-the-model-the-atmosphere-help)
says where to push.

### The caveat you may not skip

**2024 is the holdout split.** The shipped forcing covers 2024 and nothing else,
so a forced run trains on holdout data. `configs/dataloader/glorys_forced.yaml`
exists to make that window explicit -- train 2024-01-03..2024-10-30 (302
samples), validate 2024-11-01..2024-11-29 (29) -- and its header says the same
thing this paragraph does:

> A forced run in this kit is a **plumbing demonstration**. It is not a skill
> result and must never be reported as one.

A forced run on train, val or test fails on the first batch with a message
saying so. That is deliberate. If you want a forced model you can actually
report, bring your own atmosphere for 1993-2023 -- ERA5 on this grid is the
obvious choice, and `XarrayForcing` will read it as long as the grid conforms.

## 5.7 Things the coupled path refuses to do, and why

`CoupledForecastModule.__init__` raises, naming the parts, when:

* two components predict the same variable -- coupling is an exchange, not a
  vote;
* a checkpoint's own `component` is not the name you gave it;
* the components use **different depth presets** -- there is no honest way to
  reconcile two vertical grids, and a silent misalignment gives you a model that
  looks like it works;
* the components use different `lead_time_hours` -- a coupled rollout advances
  one clock.

Not verified, and worth knowing if you go past what is shipped: coupled rollouts
under `bf16-mixed` or on multiple GPUs, a genuine depth-preset mismatch (all four
shipped presets are the same 13 levels, so one cannot be built), and systems of
more than two components.

## 5.8 Where to look in the code

| file | what it is |
|---|---|
| [`oceanarches/lightning_modules/coupled.py`](../oceanarches/lightning_modules/coupled.py) | `StateRouter`, `CoupledForecastModule`, `union_component`. Written to be read. |
| [`configs/module/coupled.yaml`](../configs/module/coupled.yaml) | components, mode, order, forcing policy |
| [`configs/module/ocean_component.yaml`](../configs/module/ocean_component.yaml) | the ocean specialist preset |
| [`configs/module/seaice_component.yaml`](../configs/module/seaice_component.yaml) | the sea-ice specialist preset |
| [`configs/dataloader/glorys_seaice_isolated.yaml`](../configs/dataloader/glorys_seaice_isolated.yaml) | the state a sea-ice model with no ocean reads |
| [`oceanarches/dataloaders/forcing.py`](../oceanarches/dataloaders/forcing.py) | `NoForcing`, `XarrayForcing`, `PersistenceForcing`; the valid-time index |
| [`configs/forcing/none.yaml`](../configs/forcing/none.yaml) / [`file.yaml`](../configs/forcing/file.yaml) | the two forcing groups |
| [`configs/dataloader/glorys_forced.yaml`](../configs/dataloader/glorys_forced.yaml) | the days the shipped atmosphere covers, and the holdout warning |
| [`scripts/compute_forcing_stats.py`](../scripts/compute_forcing_stats.py) | `make forcing-stats` |
| [`notebooks/04_couple_two_models.ipynb`](../notebooks/04_couple_two_models.ipynb) | all of this, interactively, in about ninety seconds |

---

[< scaling and fine-tuning](04_scaling_finetuning.md) | [next: evaluation >](06_evaluation.md)
