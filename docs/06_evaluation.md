# 6. Is your model any good?

[< coupling](05_coupling.md) | [next: challenge ideas >](07_challenge_ideas.md)

Read time: 15 minutes. Running `make eval`: about 5 minutes on one GPU.

---

## 6.1 One command

```bash
make eval NAME=task6_tiny LEAD_DAYS=10
```

`task6_tiny` is one of the shipped checkpoints. **A fresh clone has none of
them** -- `modelstore/` is git-ignored -- so either link them in first
([docs/01 section 1.1](01_setup.md#the-normal-route)) or put your own run's
name here.

**This needs an allocation, exactly as training does.** `make eval` and
`make couple` warn and pause for ten seconds on a tty when `SLURM_JOB_ID` is
unset -- the same words `make train-*` uses, because it is the same function.
`--device cpu` is the exemption if you really do mean to score on the CPU.

That rolls the model out, scores it against two baselines through exactly the
same code, and writes:

```
evalstore/task6_tiny/
  figures/      10 PNGs
  animations/   6 mp4s
  lead10d/      the cached rollout + the metrics as netCDF
  free90d/      the 90-day free-running rollout
  report.md     everything in the figures, as tables
  report.html   the same, self-contained -- copy it anywhere, it still works
  summary.json
```

**Opening it.** There is no browser on the cluster, so `report.html` has to come to
you or be served to you. Either:

```bash
# from YOUR machine, not from the cluster. <host> is whichever one you already
# ssh into JURECA through; the path is absolute and starts /p/scratch.
scp <you>@<host>:/p/scratch/training2635/4_ocean_ai/<you>/hackathon-ocean-sea-ice/evalstore/task6_tiny/report.html .
open report.html            # or xdg-open, or drag it into a browser tab
```

On the cluster, `ls $PWD/evalstore/task6_tiny/report.html` prints that remote
path in full, ready to paste after the colon.

Or open it in the Jupyter server [docs/00](00_start_here.md) already sets up:
`evalstore/<run>/report.html` appears in the file browser, and Jupyter serves it
straight through. The file is self-contained -- every figure and animation is
embedded as a data URI, nothing is fetched from the network -- so a single `scp`
of that one file is enough, and it still works on a laptop with no repository on
it.

**One honest caveat about `report.html`.** It has never been opened in a
browser. There is no browser engine on the cluster and the environment is pinned,
so nobody in this project has actually looked at the rendered page -- only at
the figures it embeds, which are the same PNGs `report.md` links. What *is*
checked, mechanically and on every test run
(`tests/test_evaluation.py::test_report_html_is_well_formed_and_every_asset_decodes`),
is that the markup is well formed, that every embedded asset decodes and carries
the magic bytes of the type it claims, and that the page fetches nothing from the
network. (The test feeds it a PNG, a GIF and an mp4, one per embedding branch.
A real report has 10 PNGs and 6 mp4s and no GIF -- the GIF branch is the
fallback `animate.write_animation` takes when ffmpeg cannot be found.) If it looks wrong when you open it, that is a
bug we could not have found here -- please tell a tutor.

Real output, from a cold cache. Every stage announces itself with the elapsed
wall clock, so a long silence is now visible as the stage it belongs to:

```
$ make eval NAME=task6_tiny LEAD_DAYS=10 EVAL_ARGS="--force"
[   1.4s] statistics: normalisation from 400 dates from 1993-2025, climatology from 33 of 33 years (1993-2025)
Loading task6_tiny on cuda ...
Restored from modelstore/task6_tiny/checkpoints/checkpoint_global_step=4500.ckpt
  checkpoint checkpoint_global_step=4500.ckpt, component full
[  10.0s] Rolling out 16 initialisations x 10 days on domain 'test' (1085 samples available)
  batch 1/4  (4/16 initialisations, 12.4s)
  batch 2/4  (8/16 initialisations, 14.8s)
  batch 3/4  (12/16 initialisations, 16.9s)
  batch 4/4  (16/16 initialisations, 19.2s)
  loss[model] = 2.1144
  loss[persistence] = 2.2605
  loss[climatology] = 48.7485
[  37.2s] Free-running rollout: 90 days
  batch 1/1  (1/1 initialisations, 15.9s)
[  61.1s] Rendering figures
  figure 01_rmse_vs_lead.png (0.6s)
  ...
  figure 08_power_spectra.png (0.8s)
[  88.8s] Rendering animations -- the most expensive stage of the run
  animation sst_rollout.mp4 (8.4s)
  ...
  animation free_90d_seaice_arctic.mp4 (70.3s)
[ 245.7s] Writing the report
Wrote evalstore/task6_tiny/report.md
Wrote evalstore/task6_tiny/report.html

Done in 246.3s: load 7.9s, rollout 27.1s, free rollout 23.9s, figures 27.7s, animations 156.9s, report 0.5s
This run's output in evalstore/task6_tiny is 740.7 MB.
evalstore/ now holds 4.8 GB across 14 run(s).
```

Where the time goes, from a clean `evalstore/`, on two cold runs:

| stage | run A | run B |
|---|---|---|
| load the checkpoint | 7.6 | 7.9 |
| scoring rollout, 16 inits x 10 days, 3 forecasters | 20.0 | 27.1 |
| free-running rollout, 1 init x 90 days | 16.3 | 23.9 |
| 10 figures | 27.5 | 27.7 |
| 6 animations, 210 frames | **155.8** | **156.9** |
| the report | ~3 | 0.5 |
| **total** | **231.9 s** | **246.3 s** |

Two of the animations are most of it: `free_90d_sst.mp4` took 58.9 s and
`free_90d_seaice_arctic.mp4` 70.3 s, both 90 frames. Older cold runs on other
nodes came in at **285 s** and **290 s**.

Those totals are the pipeline's own clock. **Time the whole command and you get
more**, because the interpreter has to import torch, cartopy and the rest first:
run B's `Done in 246.3s` sat inside 5 min 30 s of `real`. Plan for **5 to 6
minutes** the first time on a node, less afterwards.

**A re-run is now cheap, and that is new.** The rollout, the ten figures and the
six animations are all cached, on the same key. Measured on `task6_tiny` at
`--lead-days 10`, the identical command each time:

| run | pipeline clock |
|---|---|
| cold | **231.9 s** and **246.3 s** |
| warm, identical command | **5.1 s** (4.4 s here) |
| warm, `--skip-animations` | **3.2 s** (4.1 s here) |

Again, add the interpreter start-up for the wall clock of the whole command: a
warm `make eval ... EVAL_ARGS="--skip-animations"` measured 10.3 s of `real` for
a `Done in 4.1s`.

Change anything that matters -- `--lead-days`, `--n-inits`, a retrained
checkpoint, rebuilt statistics -- and it re-renders; `--force` ignores the cache
outright. While you iterate, `--skip-animations` still skips the most expensive
stage before it starts.

**What it leaves on disk.** One `LEAD_DAYS=10` evaluation is about **750 MB** in
`evalstore/<run>/` (740.2 MB measured for `task6_tiny`), nearly all of it
`predictions.zarr` and `targets.zarr` in the two rollout caches. A dozen runs is
therefore ~9 GB of a shared quota. `run_eval` prints both this run's size and the
total across `evalstore/` when it finishes. Deleting `evalstore/<run>/` reclaims
it all, and `--skip-fields` scores without writing the maps at all -- at the cost
of every map, polar and spectrum figure.

## 6.2 The two baselines, and why they are not optional

A forecast error on its own is meaningless. Every number this pipeline prints
sits next to two lines it has to clear.

**Persistence** -- "tomorrow looks like today", i.e. repeat the initial state at
every lead time. On a slow system like the ocean this is a genuinely strong
1-day forecast, and it is the line a short forecast must beat. Anything that does
not beat persistence at day 1 has learned nothing useful.

**Climatology** -- the 1993-2025 monthly mean, interpolated to each valid day.
This is the line a *long* forecast falls back to. Once your error reaches the
climatology's, your forecast contains no information the calendar did not already
have, and that lead time is where your model's usefulness ends.

**They go through the same code as the model.** `score_batch` builds the target
once, denormalises it once, and loops over forecasters without ever branching on
which one it is holding:

```python
targets = targets_for(module, batch, iters)
targets_physical = module.denormalize_state(targets)
for forecaster in forecasters:                # model, persistence, climatology
    predictions = forecaster.predict(batch, iters)
    ...
    metric.update(targets_physical, module.denormalize_state(predictions), ...)
```

There is therefore no way for the model and its baselines to be masked
differently, denormalised differently or scored on different samples. That
matters more than it sounds: an early version of this project published a
persistence number of 0.9253 that did not reproduce -- the real value is
0.814-0.872 depending on the split, and the model's margin was 12.8%, not the
~20% first claimed. **A baseline computed through a second, parallel code path is
how a project ends up publishing a difference between two pieces of its own
code.**

## 6.3 The metrics

Everything is computed **over ocean cells only**, with `cos(lat)` weighting at
the true cell centres.

| metric | what it is | read it as |
|---|---|---|
| `rmse` | root mean squared error, in physical units | the headline. Compare against persistence. |
| `mae` | mean absolute error | less sensitive to a few big misses than RMSE |
| `bias` | mean signed error | systematic drift. A model can have a small RMSE and a bad bias. |
| `acc` | anomaly correlation coefficient | correlation between predicted and true *departure from climatology*. 1 is perfect, 0 is no better than climatology. |
| `extent` | area of every ocean cell with `siconc > 0.15`, in 10^6 km^2 | the standard sea-ice number |
| `extentbias` | `extent(forecast) - extent(truth)` | **positive means too much ice** |
| `iiee` | Integrated Ice-Edge Error: total area where model and truth disagree about whether there is ice | the honest sea-ice score. Split into `iieeover` and `iieeunder`. |

A model that puts too much ice in one place and too little in another is wrong in
a different way from one that is simply too icy, which is why IIEE is split.

Two naming rules you will trip over if you add a metric: no metric name and no
variable name may contain an underscore (geoarches splits labels on `_`), which
is why they are `iieeover` and `seaiceNH`; and labels come out as
`rmse_thetao0m_24h`.

## 6.4 What the shipped `tiny` model actually scores

The honest scorecard, on the `test` split (2021-2023), 16 initialisations spread
over the three years. Bold is the winner.

| variable | unit | day 1 model | day 1 persistence | day 10 model | day 10 persistence |
|---|---|---|---|---|---|
| `thetao` @ 0.49 m (SST) | degC | **0.12490** | 0.13093 | **0.56841** | 0.61548 |
| `siconc` | 1 | **0.010642** | 0.012685 | **0.044014** | 0.051583 |
| `zos` | m | **0.017937** | 0.021653 | 0.061441 | **0.059344** |
| `sithick` | m | **0.012326** | 0.013592 | **0.046006** | 0.049183 |
| `so` @ 0.49 m | 1e-3 | **0.074995** | 0.075464 | 0.29355 | **0.28908** |
| `mlotst` | m | **10.278** | 11.009 | **21.979** | 23.600 |

At day 1 it beats persistence on every headline variable: 4.6% on SST, 16.1% on
sea-ice concentration, 17.2% on sea surface height, 9.3% on ice thickness, 6.6%
on mixed-layer depth and 0.6% on surface salinity. Against climatology it is
better by a factor of 2.2 to 9.5. At day 10 it still wins four of six, by 6.9% on
mixed-layer depth.

These are the numbers **after** `mlotst` and `so` were given `bounds=(0.0,
None)`. Without them the same checkpoint produced negative mixed-layer depths and
negative salinities over ocean -- measured on `t10_tiny` over 16 spread
initialisations, 79 cells at day 1 growing to 15333 at day 10, minimum -53.9 m --
and `rmse_mlotst` was computed over them. The clamp moves `mlotst` at day 10 from
22.083 to 21.979 and everything else in the fifth significant figure. If you add
a variable to `variables.py`, ask whether it has a floor.

### The two places it loses, which you should go after

| channel | day 1 model | day 1 persistence | day 10 model | day 10 persistence |
|---|---|---|---|---|
| `so` @ 1684 m | 0.00139194 | **0.00125301** (+11.1% worse) | 0.0112482 | **0.00948559** (+18.6% worse) |
| NH extent bias | +0.03459 | **-0.01805** | +0.44257 | **-0.17091** |

**Deep salinity.** At 1684 m the one-day change is essentially zero, so
persistence is very nearly a perfect forecast and any noise the model adds makes
it worse. Remember `add_input_state=True`: the model predicts the *tendency* and
adds the input state, so it gets persistence for free. A variable it makes worse
is capacity spent badly. The penalty grows monotonically with lead (+11% to
+19%), which is what an accumulating noise term looks like -- not what a masking
bug looks like, which would be lead-independent.

**Northern-Hemisphere extent bias.** The model is systematically **too icy** in
both hemispheres (+0.035 NH and +0.034 SH at day 1, growing to +0.44 and +0.37 at
day 10), while persistence is unbiased by construction. And yet the same model
beats persistence on IIEE by 19% (NH) and 29% (SH) at day 1 -- so it is putting
the ice *edge* in a better place while putting slightly too much ice inside it.

These are the two most interesting things on the scorecard, and the generated
report says so.

### Useful forecast horizon

The lead time at which the model's error reaches the climatology's. Taken from
the 90-day free run, and therefore from **one** initialisation -- an estimate,
not a statistic.

| variable | no better than climatology from |
|---|---|
| `mlotst` | 10 days |
| `thetao` @ 0.49 m | 19 days |
| `zos` | 19 days |
| `so` @ 0.49 m | 23 days |
| `siconc` | 30 days |
| `sithick` | 41 days |

## 6.5 The 90-day free run, and what it tells you

This is the most informative single output of the pipeline: **the model does not
merely degrade, it leaves the attractor.**

**Read this section before you compare your figure 07 with anyone else's.** Part
of what follows reproduces on every `tiny` model we have trained and part of it
does not, and the difference matters more than either half.

### What reproduces: the field stops being a temperature

Measured on three independently trained `tiny` checkpoints -- `task6_tiny`
(4500 steps), `docs_tiny` and `t10_tiny` (4000 steps each, identical configs,
two separate runs of the same command) -- scored the same way, one initialisation
of the `test` split, 90 days free-running:

| SST field, ocean cells only | day 10 | day 30 | day 60 | day 90 |
|---|---|---|---|---|
| % outside [-5, 40] degC, `task6_tiny` | 0.00 | 0.55 | 9.04 | **22.00** |
| % outside [-5, 40] degC, `docs_tiny` | 0.02 | 0.44 | 6.21 | **21.05** |
| % outside [-5, 40] degC, `t10_tiny` | 0.02 | 0.35 | 7.05 | **19.09** |

Day-90 minimum sea surface temperature: **-315.0, -275.4 and -514.1 degC**;
day-90 maximum **+322, +246 and +410 degC**. This is not a drift, it is a
grid-scale instability: adjacent cells alternate sign, so a cell at -745 degC can
sit next to one at +739 degC. It is invisible in the global mean, which averages
the checkerboard away.

The **free-running RMSE table** in `report.md` is the reliable diagnostic, and it
says the same thing on all three:

| day-90 free-run RMSE | `task6_tiny` | `docs_tiny` | `t10_tiny` | climatology |
|---|---|---|---|---|
| `thetao` @ 0.49 m [degC] | 18.70 | 15.45 | 19.72 | **0.675** |
| `mlotst` [m] | 164.9 | 80.6 | 67.2 | **21.7** |
| `so` @ 0.49 m [1e-3] | 4.55 | 4.51 | 5.50 | **0.377** |
| `zos` [m] | 1.13 | 1.25 | 0.63 | **0.081** |

A forecast that has simply run out of information settles *onto* the climatology.
An RMSE 23 to 29 times the climatology's is a model that has left it behind.

Also reproducible: by day 90 all three models put some sea ice within 30 degrees
of the equator (3633, 161 and 224 grid cells). At day 30 none of them do.

### What does not reproduce: how big the numbers get

| quantity at day 90 | truth | climatology | `task6_tiny` | `docs_tiny` | `t10_tiny` |
|---|---|---|---|---|---|
| global-mean SST [degC] | 18.526 | 18.426 | 15.539 | 17.800 | 15.882 |
| NH sea-ice extent [10^6 km^2] | 17.464 | 19.159 | 54.240 | 20.309 | 21.498 |
| SH sea-ice extent [10^6 km^2] | 6.2061 | 8.1522 | **65.783** | **6.0557** | **17.252** |

Those three numbers come from the same recipe, the same data and the same initial
condition -- only the training run differs. `task6_tiny`'s Southern-Hemisphere
extent is larger than the entire Southern Ocean; `docs_tiny`'s sits essentially
on the truth *while its temperature field is exploding*. The global-mean SST
drift ranges from -0.73 to -3.0 degC.

**So: if your ice-extent panel looks healthy, that is not evidence your model is
stable, and if it explodes, that is not evidence you did anything wrong.** Read
the free-run RMSE table.

Look at `figures/07_free_timeseries.png` and `animations/free_90d_sst.mp4`.

**This is not a bug. It is what autoregressive rollout does to a small model
trained on a single step.** The model was trained to predict one day ahead from
*truth*. At step 2 it is fed its own output, which is slightly wrong in a way no
real ocean state ever is. It has never seen such a state, so its next error is a
little larger and a little more structured, and the process compounds. After a
few dozen steps the input is nothing like the data distribution and the model's
output is arbitrary.

Three things about the figure are worth pointing out to a beginner:

* The **climatology curve sits next to the truth.** That is the control. It tells
  you the divergence is the model, not the diagnostic.
* Divergence is not the same as loss of skill. A forecast that has simply run out
  of information settles *onto* the climatology. This one does not settle on
  anything.
* The 10-day scores in [6.4](#64-what-the-shipped-tiny-model-actually-scores)
  look respectable, and at day 10 the field is still almost entirely inside
  [-5, 40] degC. Ten days is not long enough to see this. **A model can score
  well at the lead time you tested and be unusable at three times that.**

### Fixing it

**This is one of the best things a team can do in a day.** The usual levers:

* **Train on a multi-step rollout.** `++module.train.rollout_iterations=2` and
  up, so the model sees its own errors during training. Read
  [docs/04 section 4.6](04_scaling_finetuning.md#46-training-on-a-multi-step-rollout)
  first: it costs about what the number says (measured: 2.2x the wall clock at
  2 steps, 5.3x at 5, 10.5x at 10), and there is a curriculum flag that used to
  make it cost six times more than anyone expected.
* **Push the clamp harder**, or add bounds to more variables in
  `oceanarches/dataloaders/variables.py`. `siconc`, `sithick`, `mlotst` and `so`
  already carry one; `thetao` -- the field that explodes -- does not, on purpose,
  because a clamp there would hide the instability rather than fix it.
* **Penalise drift** in a global integral, or in the spatial spectrum: this
  failure is grid-scale, so a penalty on high-wavenumber power targets it
  directly.
* **Add noise to the training inputs** so the model learns to be stable off the
  data distribution.

## 6.6 The check that matters most

Every number this pipeline prints must agree with what the Lightning module logs
during its own validation. Otherwise you are comparing two implementations, not
two models.

```bash
CUDA_VISIBLE_DEVICES=0 OCEANARCHES_EVAL_CHECKPOINT=modelstore/task6_tiny \
  OCEANARCHES_EVAL_DOMAIN=tiny_val OCEANARCHES_EVAL_SAMPLES=16 \
  .venv/bin/python -m pytest tests/test_evaluation.py -k lightning_module -q
```

```
1 passed, 53 deselected in 21.20s
```

That test scores the same samples twice -- once through the evaluation pipeline,
once through the module's own `_predict`, which is what `validation_step` calls
-- and asserts every labelled metric agrees to `rtol=1e-5`. It was proved able to
fail: perturbing the pipeline's predictions by 0.05% breaks it.

This is the one test in the suite that is skipped by default, because it needs a
checkpoint.

## 6.7 The figures

| figure | what it shows |
|---|---|
| `01_rmse_vs_lead.png` | RMSE against lead time, six panels, model and both baselines. **The headline.** |
| `02_scorecard.png` | 17 rows x 10 leads, error relative to persistence, every cell numbered. Warm = worse than persistence. |
| `03_seaice_vs_lead.png` | ice-edge error and extent bias per hemisphere. The bias panels say "positive = too much ice" on the axis. |
| `04_map_{thetao,siconc,zos}.png` | truth / prediction / error, at days 1, 6 and 10 |
| `05_depth_hovmoller.png` | depth against lead time: where in the water column the error lives |
| `06_seaice_polar.png` | Arctic and Antarctic, both 15% ice edges over the concentration field |
| `07_free_timeseries.png` | the 90-day drift check. **Look at this one.** |
| `08_power_spectra.png` | is the model blurring? Spectra at day 10 and the model/truth power ratio |

One convention holds everywhere: **warm means more of the bad thing.** A warm
error cell is a prediction that is too high; a warm scorecard cell is a forecast
with *more* error than the baseline. Land is painted neutral grey, never left as
zero -- a grey Sahara is honest, a Sahara with a sea surface height of 0 m is not.

Every figure is also a table in `report.md` and `report.html`, so nothing depends
on being able to distinguish two colours.

`08_power_spectra.png` needs `pyshtools`. If it is missing the figure is skipped
with a warning and the rest of the run is unaffected.

## 6.8 Useful flags

```bash
P=".venv/bin/python -m oceanarches.evaluation.run_eval"

$P --exp my_run --lead-days 10                       # the default: test split, 16 inits
$P --exp my_run --domain val                         # score on validation instead
$P --exp my_run --n-inits 64                         # a steadier number, a longer run
$P --exp my_run --skip-animations                    # skips ~156 s cold; 3.2 s on a warm cache
$P --exp my_run --skip-fields                        # metrics only, no zarr on disk
$P --exp my_run --free-days 30                       # a shorter drift check
$P --exp my_run --force                              # ignore the cache
$P --exp a --compare-with "baseline=b" "variant=c"   # overlay previously scored runs
```

`--compare-with` requires the other runs to have been scored on the **same**
`--domain`, `--lead-days`, `--n-inits` and `--init-selection`.

### The cache

Results live in `evalstore/<run>/lead<N>d/` with a `manifest.json` recording the
experiment, domain, lead time, initialisation indices, selection mode, whether
fields were saved, a **content hash of the checkpoint**, a hash of its
`config.yaml`, and a digest of `oceanarches/stats/`. A cache whose manifest does
not match the request is recomputed, not reused. The figures and animations are
cached beside it on the same key.

**The statistics are part of the key on purpose.** Rebuild them -- `make stats`,
or `make stats-quick` -- and the next `make eval` re-scores instead of reusing,
because the climatology is one of the two baselines: different climatology,
different comparison. You do not need `--force` for that case; it happens by
itself. The cost is one full ~4-minute run per `evalstore/<run>/`, once. Any cache
written before this key existed is in the same position and is recomputed the
first time you touch it -- so do not count on a warm cache carried over from an
earlier session.

`--compare-with` will not lay such a cache over a new one, because it cannot tell
whether the two were scored against the same statistics; the warning it prints
names the command that re-scores the older run.

The checkpoint hash is not paranoia. geoarches names its checkpoint files after
the global step, so retraining `my_run` with the same `max_steps` writes
*different weights under an identical file name*. Keyed on the name, `make eval`
would print "Reusing cached rollout" and report the **previous** model's scores
-- telling you your change did nothing. Demonstrated during development:
1.0034, then "Reusing cached", then 1.0034, when the true answer was 0.7053.
Hashing the 164 MB checkpoint costs 0.26 s against a ~240 s evaluation.

## 6.9 What this evaluation does not tell you

Say these out loud before you report anything.

* **16 initialisations is not a climatology.** They are spread over three years,
  which is respectable, but it is 16 samples.
* **The 90-day numbers come from one initialisation.** So does the forecast
  horizon table.
* **Sea-ice extent on this grid runs high** against published satellite figures
  (Arctic March maximum ~18.1 against a published 14-16). See
  [docs/02](02_data_and_masking.md#214-one-more-thing-sea-ice-extent-on-this-grid-runs-high).
  Compare your model to the truth *on this grid*.
* **The power spectrum makes two stated approximations**: `pyshtools` puts its
  first row at the north pole while ours is south-first and cell-centred, and
  land is filled with each field's own ocean mean. Both are applied identically
  to truth and prediction.
* **A report built on sampled statistics says so on its own face.** `make
  stats-quick` records the sampling in both artefacts, `make doctor` shows it in
  its `stats depth` row, and every report built on them carries a bold caution
  above its first table plus two rows in "How this was produced". If you see it,
  run `make stats` before quoting anything.
* **The normalisation statistics and the climatology are built from all 33
  prepared years, test and holdout included.** For the statistics that is
  negligible: recomputed train-only, the means move by at most 0.03 sigma, the
  level standard deviations by 1.6% and `delta_std` by 3.2%. For the
  **climatology it is not**, because the climatology is also one of the two
  baselines: scored on 2021-2023, the all-years version beats a train-only one by
  about 7% on SST, 9% on surface salinity, 9% on `zos` and 6% on `siconc`, almost
  all of it bias. It works *against* the model -- the line it has to clear is
  higher, not lower -- so nothing here is flattered by it, but say it out loud if
  you report a margin over climatology. `make stats` takes `--years 1993-2018`
  for the strictly-train version, and `glorys_1deg_stats.pt` records the years it
  was built from.
* **GLORYS is not observations.** It is a reanalysis -- itself a model, with its
  own errors. Beating it is not the same as being right about the ocean.
* **The `holdout` split (2024-2025) exists so that there is one set of years
  nobody has looked at.** Keep it that way until the end.

---

[< coupling](05_coupling.md) | [next: challenge ideas >](07_challenge_ideas.md)
