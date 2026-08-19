# 2. The data, and every trap in it

[< back to the start](00_start_here.md) | [next: your first model >](03_first_model.md)

This is the longest document in the kit and the one worth reading properly.
Everything in it was found the hard way while building this challenge. If your
model behaves strangely, the answer is very often on this page.

Read time: about 20 minutes. There is a notebook that shows all of it as
pictures: [`notebooks/01_explore_glorys.ipynb`](../notebooks/01_explore_glorys.ipynb).
It needs no trained model, so you can run it on day one.

---

## 2.1 What GLORYS is

GLORYS is an **ocean reanalysis**. A reanalysis is a physics model of the ocean
run forward in time with observations (satellites, floats, ships, moorings) fed
into it, so that it stays close to what really happened. The result is a
gap-free, physically consistent estimate of the ocean state everywhere, every
day.

For this challenge we treat GLORYS as **the truth**. It is not the truth -- it
is a model with errors of its own -- but it is the best gridded estimate
available, and every model and every baseline in this kit is scored against the
same fields, so the comparison is fair.

What we ship:

| | |
|---|---|
| product | GLORYS12V1, daily means |
| period | 1993-01-01 to 2025-12-31 |
| grid | 1 degree, 180 latitudes x 360 longitudes |
| depths | 14 levels selected from GLORYS' 50, from 0.49 m to 1684 m |
| variables | 7 two-dimensional, 4 three-dimensional (below) |
| size | 92 GB in 33 yearly files, prepared from a 608 GB raw archive |
| days | 12051 -- two are missing, see [2.11](#211-the-archive-has-a-hole-in-february-2003) |

The raw archive is **not on this machine**: it is 640 GB, and the 92 GB of
prepared files it produces are already here, so nothing you do reads it. For
the record, it is
read-only. `scripts/prepare_glorys.py` turns it into one file per year with only
the depths we use, which is what everything else reads. That has already been
done for you; `make doctor` confirms it.

## 2.2 The eleven variables, in plain English

You do not need any oceanography to do this challenge, but you do need to know
what you are predicting. Here is the whole list.

**Two-dimensional fields** -- one number per grid cell.

| name | what it is | units |
|---|---|---|
| `zos` | **Sea surface height.** How high the sea surface stands relative to a resting reference level. Ocean currents show up in it: water piles up on one side of a current and dips on the other, so a map of `zos` is essentially a map of the large-scale circulation. Range is roughly -2 m to +1.4 m. | m |
| `mlotst` | **Mixed layer depth.** The ocean is stirred by wind and by cooling at the surface, and the top part of the water column ends up well mixed -- almost the same temperature and salinity from the surface down to some depth. That depth is the mixed layer depth. It is a few tens of metres in a calm summer and can reach hundreds of metres in a stormy winter or where deep water forms. It is the single most jumpy field in this dataset: it is a diagnostic quantity, computed from the temperature profile, so it can move a long way overnight. That is why it carries a reduced weight in the loss. | m |
| `bottomT` | **Temperature at the sea floor.** Cold and almost constant nearly everywhere. | degC |
| `siconc` | **Sea ice concentration.** The fraction of a grid cell covered by ice. **It is a fraction, so it lives in [0, 1] and nothing else is physical.** The models here clamp their prediction into that range. A cell of 0.4 does not mean "ice 40 cm thick"; it means 40% of that 111 km x 111 km box has ice on it and 60% is open water. | 1 (a fraction) |
| `sithick` | **Sea ice thickness.** How thick that ice is, averaged over the ice-covered part of the cell. Cannot be negative. | m |
| `usi`, `vsi` | **Sea ice velocity**, eastward and northward. Ice drifts: it is pushed by wind and dragged by ocean currents. | m/s |

**Three-dimensional fields** -- one number per grid cell *per depth level*.

| name | what it is | units |
|---|---|---|
| `thetao` | **Potential temperature** of the sea water. "Potential" means the temperature the water would have if you brought it to the surface without letting it exchange heat -- it removes the warming that pressure alone causes, so you can compare deep and shallow water fairly. `thetao` at the shallowest level (0.49 m) is what everyone else calls **sea surface temperature**, and it is the headline variable of this challenge. | degC |
| `so` | **Salinity**: dissolved salt, in grams per kilogram. Open ocean is about 35. The Baltic is nearly fresh; the Red Sea reaches 41. | 1e-3 |
| `uo`, `vo` | **Sea water velocity**, eastward and northward, at each depth. | m/s |

The full table, with the loss weight and the plotting metadata for each, is
[`oceanarches/dataloaders/variables.py`](../oceanarches/dataloaders/variables.py).
That file is the single source of truth: if you want to add a variable, change
it there and everything downstream follows.

## 2.3 Why the ocean is slow, and what that means for you

The atmosphere reorganises itself in days. A weather forecast is useful for
about a week and then it is not, because small errors grow fast.

The ocean is different. Water is about 800 times denser than air and holds about
4000 times more heat per kilogram, so it takes an enormous amount of energy to
change it. The sea surface temperature at a point moves by roughly **0.1 degC in
a day**, against a spatial spread of about **11.7 degC** between the tropics and
the poles. Deep salinity at 1684 m barely moves at all.

Two consequences, and they shape the whole challenge:

**1. "Tomorrow looks like today" is an excellent forecast.** This is called the
**persistence** baseline, and on the ocean it is hard to beat. Simply copying
today's state gives a 1-day sea surface temperature error of 0.126 degC. Our
trained `tiny` model gets 0.121 degC. That is a 3.7% improvement -- real, but
small. **Any number you quote without a baseline beside it is meaningless.**

Here is the shape of the problem in one table (measured, `test` split, 16
initialisations, days 1 and 10):

| variable | day 1 model | day 1 persistence | day 10 model | day 10 persistence |
|---|---|---|---|---|
| sea surface temperature [degC] | **0.1249** | 0.1309 | **0.5684** | 0.6155 |
| sea ice concentration | **0.01064** | 0.01269 | **0.04401** | 0.05158 |
| sea surface height [m] | **0.01794** | 0.02165 | 0.06147 | **0.05934** |

Bold is the winner. Notice that at day 10 the model has already *lost* on sea
surface height.

**2. The loss is scaled by how much each field changes in a day.** If you simply
averaged squared errors across variables, deep salinity -- which spans 0.002 to
41.7 across the globe but changes by 0.001 overnight -- would dominate, and the
model would spend its capacity learning geography instead of dynamics. The
training loss therefore divides each error by `delta_std`, the standard
deviation of the one-day change, taken from
`oceanarches/stats/glorys_1deg_stats.pt`. The useful consequence is that **the
loss is calibrated: a persistence forecast scores about 1.**

Measured, through the shipped loss:

| split | 1-day persistence loss |
|---|---|
| `tiny_val` (2019), first 128 samples in order | 0.8141 |
| `tiny_val`, whole split (364) | 0.8718 |
| `tiny_train` (2014-2018), whole split (1825) | 0.8439 |
| `train` (1993-2018), whole split (9488) | 0.8204 |

So: **a training loss below about 0.82-0.87 means you are beating persistence.**
It is not exactly 1.0 because `delta_std` is one unweighted number per variable
and depth while the loss is a latitude-weighted mean over ocean area. And the
line moves with the split, so always say which split you measured on.

## 2.4 The shape of the data in memory

Everything in this project uses two tensors, borrowed from geoarches:

```
surface   (variable, 1, latitude, longitude)     = (7, 1, 180, 360)
level     (variable, depth, latitude, longitude) = (4, 14, 180, 360)
```

The length-1 depth axis on `surface` is not a mistake. It means surface and
level tensors have the same rank, so the same code can weight, mask and average
both.

**Orientation.** GLORYS runs latitude from south to north (-89.5 to 89.5) and
longitude from 0 to 359. We keep that exactly. `tensor[..., 0, :]` is the
**South** Pole. geoarches' own ERA5 loader flips latitude and rolls longitude to
centre Europe; ours deliberately does neither, and there is a comment in
`glorys.py` at the two places where those lines are missing.

Two classes read the data:

* `GlorysDataset` -- one day at a time, physical units, land still NaN. Use it to
  **look** at the data.
* `GlorysForecast` -- what training and evaluation use: `(previous, state, next)`
  tuples, normalised, with no NaN anywhere.

```bash
.venv/bin/python -c "
from oceanarches.dataloaders.glorys import GlorysForecast
ds = GlorysForecast(domain='tiny_val')
s = ds[0]
print(sorted(s))
print('state.surface', tuple(s['state']['surface'].shape))
print('state.level  ', tuple(s['state']['level'].shape))
"
```

```
['lead_time_hours', 'next_state', 'prev_state', 'state', 'timestamp']
state.surface (7, 1, 180, 360)
state.level   (4, 14, 180, 360)
```

**Note the 14.** A bare `GlorysForecast()` loads all 14 *prepared* levels. Every
shipped model preset asks for **13** of them by passing `depth_indices`, for a
reason that is about the neural network rather than the ocean --
[docs/04](04_scaling_finetuning.md#42-the-vertical-is-not-a-scaling-axis)
explains it. So a model's tensors are `(4, 13, 180, 360)`:

```bash
.venv/bin/python -c "
from oceanarches.dataloaders.glorys import GlorysForecast
from oceanarches.dataloaders.variables import DEPTH_PRESETS
ds = GlorysForecast(domain='tiny_val', depth_indices=DEPTH_PRESETS['tiny'])
print('state.level', tuple(ds[0]['state']['level'].shape))
"
```

```
state.level (4, 13, 180, 360)
```

Opening a dataset prints a progress bar while it scans the yearly files. That is
normal and takes a second or two.

## 2.5 The splits

| split | years | samples | what it is for |
|---|---|---|---|
| `train` | 1993-2018 | 9488 | training |
| `val` | 2019-2020 | 730 | choosing between runs |
| `test` | 2021-2023 | 1094 | the number you report |
| `holdout` | 2024-2025 | 730 | leave it alone until the end |
| `tiny_train` | 2014-2018 | 1825 | the 30-minute model |
| `tiny_val` | 2019 | 364 | the 30-minute model |
| `ifs_forced_train` | 2024-01-03..2024-10-30 | 302 | **inside `holdout`** -- forcing demo only |
| `ifs_forced_val` | 2024-11-01..2024-11-29 | 29 | **inside `holdout`** -- forcing demo only |

The last two are the odd ones out and are deliberately named so you cannot use
them by accident. They are not calendar years but explicit date windows (see
`SPLIT_DATES` in `oceanarches/dataloaders/glorys.py`), and they sit **inside
`holdout`**, because the shipped IFS atmosphere covers 2024 and nothing else. A
model trained on them has seen holdout data; they exist so that `forcing=file`
can be exercised end to end, never so that anything can be scored. See
[docs/05 section 5.6](05_coupling.md#56-external-forcing-the-three-routes-in-and-the-file-one).

No target ever leaves its own split. An *input* may come from the split before
(the state on 1 January reads its previous state from 31 December), which is
fine; a *target* in the following split would be leakage, and the dataloader
prevents it.

**That is the whole story for the dataloader, and it is not the whole story for
the kit.** `make stats` builds the normalisation statistics *and the climatology*
from every prepared year, 1993-2025, test and holdout included -- 85 of the 400
sampled dates (21%) are outside `train`. For the statistics the effect is
negligible: recomputed train-only, the means move by at most 0.03 sigma, the
level standard deviations by 1.6% and `delta_std` by 3.2%. For the **climatology**
it is not negligible, because the climatology is also one of the two scored
baselines: on 2021-2023 the all-years version beats a train-only one by about 7%
on SST, 9% on surface salinity, 9% on `zos` and 6% on `siconc`, almost all of it
bias. It makes the baseline *stronger*, so no model here is flattered by it -- but
say so if you report a margin over climatology.
`scripts/compute_stats.py --years 1993-2018` builds the strictly-train version,
and `glorys_1deg_stats.pt` records the years it was built from under the `years`
key.

```bash
.venv/bin/python -c "
from oceanarches.dataloaders.glorys import GlorysForecast
for d in ('train', 'val', 'test', 'holdout', 'tiny_train', 'tiny_val',
          'ifs_forced_train', 'ifs_forced_val'):
    f = GlorysForecast(domain=d)
    lo, hi = f.state_timestamp_range()
    print(f'{d:11s} {len(f):5d} samples   dropped: {f.n_dropped_at_edges} at the edges, '
          f'{f.n_dropped_by_time_check} by the timestamp check   states {lo} .. {hi}')
"
```

```
train        9488 samples   dropped: 2 at the edges, 4 by the timestamp check   states 1993-01-02T12:00:00 .. 2018-12-30T12:00:00
val           730 samples   dropped: 2 at the edges, 0 by the timestamp check   states 2019-01-01T12:00:00 .. 2020-12-30T12:00:00
test         1094 samples   dropped: 2 at the edges, 0 by the timestamp check   states 2021-01-01T12:00:00 .. 2023-12-30T12:00:00
holdout       730 samples   dropped: 2 at the edges, 0 by the timestamp check   states 2024-01-01T12:00:00 .. 2025-12-30T12:00:00
tiny_train   1825 samples   dropped: 2 at the edges, 0 by the timestamp check   states 2014-01-01T12:00:00 .. 2018-12-30T12:00:00
tiny_val      364 samples   dropped: 2 at the edges, 0 by the timestamp check   states 2019-01-01T12:00:00 .. 2019-12-30T12:00:00
ifs_forced_train  302 samples   dropped: 2 at the edges, 0 by the timestamp check   states 2024-01-03T12:00:00 .. 2024-10-30T12:00:00
ifs_forced_val     29 samples   dropped: 2 at the edges, 0 by the timestamp check   states 2024-11-01T12:00:00 .. 2024-11-29T12:00:00
```

Hold on to those two "dropped" columns. They come back in [2.11](#211-the-archive-has-a-hole-in-february-2003).

---

# The traps

Six of them. Every one cost real time to find.

## 2.6 The land mask is three-dimensional

The ocean is not a rectangle. On this grid **30.4% of cells are land at the
surface**, so nearly a third of what a convolution or an attention window sees
carries no information at all.

And it gets worse with depth. A cell on the continental shelf is ocean at 5 m
and rock at 300 m. **Every depth level has its own coastline.**

```bash
.venv/bin/python -c "
from oceanarches.dataloaders.masks import load_masks
m = load_masks()
print(f'grid          {tuple(m.wet_surface.shape)}')
print(f'ocean cells   {m.n_ocean_surface} of {m.wet_surface.numel()}')
for d, w in zip(m.depths, m.wet_level):
    print(f'  {d:8.2f} m   ocean {100 * w.float().mean():5.2f} %   land {100 - 100 * w.float().mean():5.2f} %')
"
```

```
grid          (180, 360)
ocean cells   45115 of 64800
      0.49 m   ocean 69.62 %   land 30.38 %
      5.08 m   ocean 69.62 %   land 30.38 %
     15.81 m   ocean 69.02 %   land 30.98 %
     29.44 m   ocean 68.35 %   land 31.65 %
     55.76 m   ocean 67.12 %   land 32.88 %
     92.33 m   ocean 65.95 %   land 34.05 %
    155.85 m   ocean 64.67 %   land 35.33 %
    222.48 m   ocean 63.58 %   land 36.42 %
    318.13 m   ocean 62.63 %   land 37.37 %
    453.94 m   ocean 61.44 %   land 38.56 %
    643.57 m   ocean 60.42 %   land 39.58 %
    902.34 m   ocean 59.50 %   land 40.50 %
   1245.29 m   ocean 58.57 %   land 41.43 %
   1684.28 m   ocean 57.49 %   land 42.51 %
```

At 1684 m, the deepest level we keep, **42.5%** of the grid is rock. If you go
deeper still in the raw archive it reaches **58.9% at 3993 m**.

The first two rows are identical (69.62% both). That is correct, not a bug: no
ocean column on Earth ends between 0.5 m and 5 m.

**What the code does.** `oceanarches/dataloaders/masks.py` loads three masks:
`wet_surface` (2-D fields), `wet_level` (one per depth, for 3-D fields) and
`wet_seaice`. The loss and every metric multiply by the *per-depth* mask, so a
1684 m field is never scored on the continental shelf.

## 2.7 The southernmost 13 rows are always empty, for two different reasons

The **southernmost 13 rows of every field are entirely empty**, forever, for
every variable and every depth. It is worth knowing *why*, because it is two
things and not one:

* rows 0 to 9 (latitudes -89.5 to -80.5) are **outside GLORYS' domain** -- the
  product covers latitudes from -80 upward, and our grid starts at -89.5;
* rows 10 to 12 (-79.5 to -77.5) are inside the domain and are **Antarctica**.
  There is no ocean surface there to report.

The first row with any water in it is -76.5, in both the raw archive and our
mask, and it has 79 ocean cells out of 360.

```bash
.venv/bin/python -c "
from oceanarches.dataloaders.masks import load_masks
m = load_masks()
rows = m.wet_surface.sum(1)
first = int((rows > 0).nonzero()[0])
print(f'rows with no ocean at all, counting from the south: {first}')
print(f'they are latitudes -89.5 to {-89.5 + first - 1}')
print(f'first row with ocean: index {first}, latitude {-89.5 + first}, {int(rows[first])} cells')
"
```

```
rows with no ocean at all, counting from the south: 13
they are latitudes -89.5 to -77.5
first row with ocean: index 13, latitude -76.5, 79 cells
```

We keep the empty rows rather than cropping the grid, because 180 divides
cleanly by the model's patch size of 3 and 167 does not. The mask deals with
them. But it is worth knowing, because those rows are 7% of your grid and they
are pure padding -- and because a Southern-Ocean map that looks blank at the
bottom is showing you the data, not a bug.

Note that the Antarctic coast is a *moving* target further north too: the ocean
fraction climbs from 79 cells at -76.5 to 208 at -70.5, so almost every row in
the Southern Ocean is mostly land. Any metric you restrict to the Antarctic is
averaging over far fewer cells than the row count suggests.

## 2.8 One whole depth level is nothing at all

GLORYS has 50 native depth levels. The last one, index 49 at **5728 m, is 100%
NaN everywhere on every date**. Not "mostly land" -- entirely empty.

```bash
.venv/bin/python -c "
import numpy as np, xarray as xr
from oceanarches import paths
f = paths.glorys_raw() / '2015/07/mercatorglorys12v1_gl12_mean_20150715_R20150722.nc'
ds = xr.open_dataset(f)
t = ds['thetao'].isel(time=0)
for i in (0, 38, 45, 49):
    v = np.isfinite(t.isel(depth=i).to_numpy())
    print(f'native level {i:2d}  {float(ds.depth[i]):8.1f} m   ocean {100 * v.mean():6.2f} %')
"
```

```
native level  0       0.5 m   ocean  69.62 %
native level 38    1684.3 m   ocean  57.49 %
native level 45    3992.5 m   ocean  41.11 %
native level 49    5727.9 m   ocean   0.00 %
```

This is why the preparation step selects levels rather than keeping all 50.
`PREPPED_DEPTH_INDICES` in `variables.py` keeps 14 of them, spanning 0.49 m to
1684 m, sampling the mixed layer, the thermocline and the upper deep ocean.
Every shipped model preset then loads **13** of those 14 -- it drops 1245 m --
for a reason that is about the neural network, not the ocean, and is explained
in [docs/04](04_scaling_finetuning.md#42-the-vertical-is-not-a-scaling-axis).

## 2.9 NaN that means zero, and a dataset that changes its mind

This is the subtlest trap in the kit, and the most damaging.

The four sea-ice variables -- `siconc`, `sithick`, `usi`, `vsi` -- carry this
attribute in the raw files:

```
siconc:cell_methods = "area: mean where sea_ice"
```

Read it carefully. It means the value is an average **over the part of the cell
that has sea ice on it**. If there is no ice, there is nothing to average, so
the field is *undefined* -- and GLORYS writes NaN.

**Those NaNs are over open ocean, not over land, and they mean zero.** There is
no ice there. A model that receives them as NaN produces NaN gradients and dies;
a model that receives them as "missing" learns nothing.

It gets worse. **GLORYS changed its convention halfway through the archive.**
Until 2015-12-29 it wrote an exact `0.0` over ice-free ocean for `usi` and
`vsi`. From 2015-12-30 it writes NaN there instead. Overnight:

```bash
.venv/bin/python -c "
import numpy as np
from oceanarches.dataloaders.glorys import GlorysDataset
from oceanarches.dataloaders.masks import load_masks
ocean = load_masks().wet_surface.numpy()
raw = GlorysDataset(domain='all', fill_seaice=False)
filled = GlorysDataset(domain='all')
t = np.array([x[2] for x in raw.timestamps], dtype='datetime64[s]')
for day in ('2015-12-29', '2015-12-30'):
    i = int(np.where(t == np.datetime64(day + 'T12:00:00'))[0][0])
    print(day)
    for v in ('siconc', 'usi'):
        c = raw.surface_variables.index(v)
        a = raw[i]['surface'][c, 0].numpy()
        b = filled[i]['surface'][c, 0].numpy()
        print(f'  {v:7s} on disk: NaN over ocean {int((np.isnan(a) & ocean).sum()):6d}   exactly 0 over ocean {int(((a == 0) & ocean).sum()):6d}')
        print(f'  {v:7s} filled : NaN over ocean {int((np.isnan(b) & ocean).sum()):6d}   exactly 0 over ocean {int(((b == 0) & ocean).sum()):6d}')
"
```

```
2015-12-29
  siconc  on disk: NaN over ocean  35066   exactly 0 over ocean     52
  siconc  filled : NaN over ocean      0   exactly 0 over ocean  35118
  usi     on disk: NaN over ocean     10   exactly 0 over ocean  35762
  usi     filled : NaN over ocean      0   exactly 0 over ocean  35772
2015-12-30
  siconc  on disk: NaN over ocean  35124   exactly 0 over ocean     54
  siconc  filled : NaN over ocean      0   exactly 0 over ocean  35178
  usi     on disk: NaN over ocean  35724   exactly 0 over ocean     73
  usi     filled : NaN over ocean      0   exactly 0 over ocean  35797
```

Look at the `usi` row. **10 NaN cells one day, 35 724 the next.** Nothing
happened to the ice; the file format changed.

Why this matters to you: every year from 2016 onward is validation, test or
holdout. A model trained without the fill sees one data convention in training
and a different one at every evaluation. `siconc` and `sithick` are NaN over
ice-free ocean for the *whole* archive, so they are unaffected by the switch,
but they need the same fill for the same reason.

**What the code does.** `Variable.nan_means_zero` in `variables.py` flags the
four fields, and `masks.fill_seaice_nans` sets them to 0 wherever the cell is
ocean, before anything else happens. Land is left NaN, to be handled later.
The fill is on by default; `GlorysDataset(..., fill_seaice=False)` turns it off
so that you can see the raw bytes, which is exactly what the snippet above does.

## 2.10 The velocity fields have 630 holes the mask calls ocean

`uo` and `vo` are NaN at 630 cells (summed over all 14 depth levels, 10 of them
at the surface) that `wet_level` says are ocean. `thetao` and `so` have none.
The pattern is identical on every date in the archive -- they are isolated
coastal points where the velocity grid has no valid neighbour.

```bash
.venv/bin/python -c "
import numpy as np
from oceanarches.dataloaders.glorys import GlorysDataset
from oceanarches.dataloaders.masks import load_masks
wet = load_masks().wet_level.numpy()
ds = GlorysDataset(domain='holdout', fill_seaice=False)
x = ds[100]['level'].numpy()
for i, v in enumerate(ds.level_variables):
    print(f'{v:7s} NaN at cells the mask calls ocean: {int((np.isnan(x[i]) & wet).sum()):4d}')
"
```

```
thetao  NaN at cells the mask calls ocean:    0
so      NaN at cells the mask calls ocean:    0
uo      NaN at cells the mask calls ocean:  630
vo      NaN at cells the mask calls ocean:  630
```

They are filled like land -- 0 after normalisation -- and the mask is **not**
shrunk to match, because those cells are genuine ocean for every other variable.
The consequence is that the velocity loss is computed at 630 cells where the
target is a fabricated 0. It is 1.4% of one variable's cells and nobody has
found it to matter, but it is there. If you want them out, intersect the
velocity mask with `~uo.isnan()`; there is a comment in `glorys.py.__getitem__`
saying so.

## 2.11 The archive has a hole in February 2003

**2003-02-07 and 2003-02-11 are missing from the raw archive.** Not corrupted --
absent. The 2003 file holds 363 days.

```bash
.venv/bin/python -c "
import numpy as np
from oceanarches.dataloaders.glorys import GlorysDataset
ds = GlorysDataset(domain='all')
t = np.array([x[2] for x in ds.timestamps], dtype='datetime64[s]')
gaps = np.diff(t).astype('timedelta64[h]').astype(int)
print(f'{len(t)} daily fields, {t[0]} to {t[-1]}')
for i in np.where(gaps != 24)[0]:
    print(f'  gap: {t[i]} -> {t[i+1]}  ({gaps[i]} hours)')
"
```

```
12051 daily fields, 1993-01-01T12:00:00 to 2025-12-31T12:00:00
  gap: 2003-02-06T12:00:00 -> 2003-02-08T12:00:00  (48 hours)
  gap: 2003-02-10T12:00:00 -> 2003-02-12T12:00:00  (48 hours)
```

**Why this is dangerous.** The obvious way to build a forecasting dataset is
index arithmetic: sample `i` is the input, sample `i + 1` is the target one day
later. Across that gap `i + 1` is **two** days later, and nothing tells you.
Four samples in the training split would be silently labelled 1-day forecasts
when they are really 2-day forecasts. Your model would be trained to make a
mistake, and the loss would look fine.

geoarches' own ERA5 loader does exactly this index arithmetic. It is correct for
ERA5, which has no gaps.

**What the code does.** `GlorysForecast` validates the *real timestamps* of
every `(previous, state, next, future...)` tuple against the requested lead time
and drops the ones that do not line up. It drops exactly four in `train` -- the
neighbours of the two missing days, 2003-02-06, -08, -10 and -12 -- which is the
`4` in the "dropped by the timestamp check" column above. Two more are dropped
in every split at the edges: the very first state has no previous state and the
very last has no next state.

If you write your own dataset class, this is the one thing you must not
reimplement naively.

## 2.12 The masking order, and why it is not negotiable

Four steps, in this order, in `GlorysForecast.__getitem__`:

```
1. read the raw field                  land = NaN, ice-free ocean = NaN
2. fill_seaice_nans                    ice-free ocean -> 0     (physical units)
3. normalise: (x - mean) / std         everything is now in standard deviations
4. nan_to_num(0.0)                     whatever is still NaN is land -> 0
```

Steps 2 and 3 must stay in that order. Here is what each cell ends up as:

```bash
.venv/bin/python -c "
import numpy as np
from oceanarches.dataloaders.glorys import GlorysForecast
from oceanarches.dataloaders.masks import load_masks
ocean = load_masks().wet_surface.numpy()
onDisk = GlorysForecast(domain='tiny_val', fill_seaice=False, norm_scheme=None, nan_to_num=False)
model  = GlorysForecast(domain='tiny_val')
c = model.surface_variables.index('siconc')
mean = float(model.data_mean['surface'][c]); std = float(model.data_std['surface'][c])
a = onDisk[200]['state']['surface'][c, 0].numpy()
b = model[200]['state']['surface'][c, 0].numpy()
icefree = np.isnan(a) & ocean
print(f'siconc mean {mean:.4f}  std {std:.4f}  ->  -mean/std = {-mean/std:.4f}')
print(f'ice-free ocean ({int(icefree.sum())} cells) reaches the model as {np.unique(b[icefree].round(4))}')
print(f'land ({int((~ocean).sum())} cells) reaches the model as {np.unique(b[~ocean].round(4))}')
"
```

```
siconc mean 0.1700  std 0.3528  ->  -mean/std = -0.4817
ice-free ocean (35429 cells) reaches the model as [-0.4817]
land (19685 cells) reaches the model as [0.]
```

* **Ice-free ocean becomes `-mean/std`.** That is the normalised representation
  of a real, physical zero. Correct: there genuinely is no ice there.
* **Land becomes exactly `0` in normalised space**, which is the climatological
  mean. It is the least disruptive value to hand a network, and it is what
  every downstream mask assumes.

Swap steps 2 and 3 and ice-free ocean would come out as `0` in normalised space
-- i.e. you would be telling the model that open water has the *global mean* sea
ice concentration of 0.17. Fill land before normalising and every land cell
would come out at `-mean/std`, indistinguishable from ice-free ocean. Neither
crashes. Both quietly poison the training.

There is a test that pins this
(`tests/test_dataset.py::test_the_seaice_fill_happens_before_normalisation`), and
it was checked against a deliberately reversed implementation to make sure it
can actually fail.

## 2.13 Land must be out of the loss and out of the metrics

This is the last one and it is the easiest to get wrong, because nothing breaks.

Land reaches the model as exactly 0 in normalised space. A model quickly learns
to output 0 there -- it is the easiest thing in the world to predict. If land is
in your average, **30.4% of your score is a constant that every model gets
right.** Your numbers look good and mean nothing.

Concretely: score a "model" that predicts exactly zero everywhere -- that is, the
climatological mean and nothing else -- with and without land:

```bash
.venv/bin/python -c "
import torch
from oceanarches.dataloaders.glorys import GlorysForecast
from oceanarches.metrics.masked_metrics import compute_lat_weights_glorys, ocean_area_weights

ds = GlorysForecast(domain='tiny_val')
mask = ds.state_mask()['surface']           # (7, 1, 180, 360), 1 ocean 0 land
truth = ds[200]['state']['surface']         # normalised, land is exactly 0
zeros = torch.zeros_like(truth)             # a model that always predicts the mean
sq = (zeros - truth) ** 2

lat = compute_lat_weights_glorys(180)
whole_grid = float((sq * lat).mean())
ocean_only = float((sq * ocean_area_weights(mask)).sum() / mask.shape[0])
print(f'mean squared error over the whole grid : {whole_grid:.4f}')
print(f'mean squared error over the ocean only : {ocean_only:.4f}')
print(f'the model looks {100 * (1 - whole_grid / ocean_only):.0f}% better than it is')
"
```

```
mean squared error over the whole grid : 0.5037
mean squared error over the ocean only : 0.6770
the model looks 26% better than it is
```

A free 26%, for nothing, from cells with no water in them. And it gets worse
with depth, because there is more land down there.

**What the code does.** `ocean_area_weights` in
[`oceanarches/metrics/masked_metrics.py`](../oceanarches/metrics/masked_metrics.py)
returns, per `(variable, depth)` channel, weights that are zero over land and
**sum to 1 over the ocean**. One function, used by both the loss and the
metrics, so the two cannot disagree about what "the ocean" is. Because the
denominator is the ocean area rather than the grid area, a constant error of size
`e` scores exactly `e` no matter how much land there is.

This was verified rather than assumed: with the real model and a real batch,
overwriting land in *both* the prediction (with `+1e6`) and the target (with
noise) leaves the loss **bit-for-bit identical**.

**If you replace the loss, keep this property.** It is the single easiest way to
produce a model that looks better than the shipped one and is not.

## 2.14 One more thing: sea ice extent on this grid runs high

Not a bug, and not something you should try to fix -- but you need to know it
before you compare a number to something you read in a paper.

**Sea ice extent** is the standard sea-ice diagnostic: the total area of every
cell whose concentration exceeds 15%. The 15% threshold comes from what
passive-microwave satellites can reliably see.

On this 1-degree grid, the Arctic March maximum computes to about **18.1 x 10^6
km^2**. Published satellite figures are 14-16. The area integral is not wrong.
The regrid is the reason: GLORYS is natively 1/12 degree, and averaging it onto
1-degree cells smears the ice edge across whole cells. A cell that was 90% open
water at 1/12 degree comes out at a concentration of 0.1-0.3, clears the 15%
threshold, and then contributes its **whole** 12000 km^2 to the total.

Measured on March 2019, Northern Hemisphere, daily fields:

| threshold | monthly mean | daily maximum |
|---|---|---|
| `siconc > 0.15` (the standard) | 17.82 | **18.13** |
| `siconc > 0.30` | 17.07 | 17.35 |
| `siconc > 0.50` | 16.23 | 16.49 |
| `siconc > 0.80` | 14.18 | 14.61 |

**3.63 x 10^6 km^2 of the March total sits in cells whose concentration is
between 0.15 and 0.8** -- that is the smear, and at a 0.8 threshold the number
lands right where the published one is.

The summer number is fine: September 2019 gives 4.53 as a monthly mean and 4.30
on its lowest day, against a published Arctic September minimum of about 4-5.

**What to do about it:** nothing, except compare your model against *the truth on
this grid*, which is what every metric in this kit does. Do not compare your
`siconc` extent against a satellite product on a different grid and conclude your
model is broken.

---

## 2.15 Where all of this lives

| file | what it holds |
|---|---|
| [`oceanarches/dataloaders/variables.py`](../oceanarches/dataloaders/variables.py) | the variable table, the depth levels, the components. Change data here and nowhere else. |
| [`oceanarches/dataloaders/masks.py`](../oceanarches/dataloaders/masks.py) | the three masks, `fill_seaice_nans`, `state_mask`, and the masking order in the module docstring |
| [`oceanarches/dataloaders/glorys.py`](../oceanarches/dataloaders/glorys.py) | `GlorysDataset`, `GlorysForecast`, the timestamp validation |
| [`oceanarches/metrics/masked_metrics.py`](../oceanarches/metrics/masked_metrics.py) | `ocean_area_weights` -- the one definition of "average over the ocean" |
| [`scripts/compute_stats.py`](../scripts/compute_stats.py) | builds the masks, the normalisation statistics and the monthly climatology |
| `oceanarches/stats/glorys_1deg_masks.nc` | 2.9 MB, the masks |
| `oceanarches/stats/glorys_1deg_stats.pt` | 12 kB, mean / std / delta_std per variable and depth |
| `oceanarches/stats/glorys_1deg_climatology.nc` | 95 MB, the 1993-2025 monthly mean |

Rebuild all three artefacts with `make stats` (5 to 8 minutes, 5.6 GB of
memory) or `make stats-quick` (about a minute, fewer dates, good enough for
smoke tests but **not** for a model you intend to report). A quick build is not
untraceable: both artefacts record how deeply they were sampled, `make doctor`
drops its `stats depth` row to WARN, and every evaluation report built on them
prints that in bold above its first table.

---

[< back to the start](00_start_here.md) | [next: your first model >](03_first_model.md)
