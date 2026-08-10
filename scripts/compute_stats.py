#!/usr/bin/env python
"""Derive the three artefacts the model needs from the prepared GLORYS files.

    oceanarches/stats/glorys_1deg_masks.nc         which cells are ocean, and the
                                                   static fields fed to the network
    oceanarches/stats/glorys_1deg_stats.pt         normalisation + tendency statistics
    oceanarches/stats/glorys_1deg_climatology.nc   monthly climatology (ACC, baselines)

All statistics are computed over **ocean points only**.  Averaging over land, where
the data is NaN or zero, would give meaningless means and far too small standard
deviations, and the model would then be trained on badly scaled inputs.

    python scripts/compute_stats.py                    # the real thing (~8 min)
    python scripts/compute_stats.py --quick            # a handful of dates (~1 min)
    python scripts/compute_stats.py --years 1993-2018  # train years only

**Which years these are built from is a real choice, and the default is every
prepared year (1993-2025), test and holdout included.**  For the normalisation
statistics that is negligible -- recomputed train-only, the means move by at most
0.03 sigma, the level standard deviations by 1.6% and ``delta_std`` by 3.2%.  For
the *climatology*, which is also a scored baseline, it is not negligible: an
all-years climatology beats a train-only one on the 2021-2023 test split by about
7% on SST, 9% on surface salinity, 9% on ``zos`` and 6% on ``siconc``, almost all
of it bias.  That works **against** the model -- the baseline it has to beat is
stronger, not weaker -- so no published result here is flattered by it, but it is
not the leakage-free story ``docs/02`` section 2.5 reads like on its own.  Use
``--years 1993-2018`` if you want the strictly-train version; the years actually
used are recorded in the ``years`` key of ``glorys_1deg_stats.pt`` and in the
climatology file's ``years`` attribute.

Both artefacts also record **how deeply they were sampled** -- ``n_dates``,
``n_years`` and a ``sampling`` key that reads ``full`` or ``sampled``.  That is
what ``--quick`` used to leave no trace of: an evaluation report quoted
five-figure scores against a climatology built from three years with nothing
anywhere saying so.  ``oceanarches.evaluation.provenance`` reads it back and
``report.py`` prints it on the face of every report.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

import netCDF4
import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from oceanarches import paths  # noqa: E402
from oceanarches.dataloaders.variables import (  # noqa: E402
    LEVEL_VARIABLES,
    N_LAT,
    N_LON,
    NAN_MEANS_ZERO,
    PREPPED_DEPTHS,
    SURFACE_VARIABLES,
)


def _shrink_netcdf_chunk_cache() -> None:
    """Cut netCDF4's per-variable HDF5 chunk cache from 64 MB to 4 MB.

    Here the default is pure waste: the prepared files are chunked one
    (time, depth) slice at a time and every chunk is read exactly once.  Left
    alone it costs ~350 MB in every process.

    Called from :func:`main` and from each pool worker, *not* at import.
    ``netCDF4.set_chunk_cache`` is a process-wide library default, so setting it
    on import would silently change the behaviour of anything that merely
    imported this module for one of its helpers -- the test suite included.
    """
    netCDF4.set_chunk_cache(4 * 1024 * 1024, 127, 0.75)


N_DEPTH = len(PREPPED_DEPTHS)

#: Dates a full build of the normalisation statistics uses (the ``--n-dates``
#: default).  Recorded in the artefacts so that a *reader* -- the evaluation
#: report, not only a person -- can tell a full build from a sampled one.
#: ``oceanarches.evaluation.provenance`` holds the same number and reads it back.
DEFAULT_N_DATES = 400

#: The two values the ``sampling`` key/attribute takes.
FULL, SAMPLED = "full", "sampled"


def sampling_label(used: int, available: int) -> str:
    """``"full"`` when everything available was used, ``"sampled"`` otherwise.

    Writing this down rather than leaving it to be inferred is the point:
    ``make stats-quick`` used to leave *no* trace downstream.  The climatology's
    ``years`` attribute read ``1993-2025`` whether it was built from 33 years or
    from 3 spread across the same span, and an evaluation report could quote a
    five-figure score against a baseline built from a handful of dates with
    nothing anywhere saying so.
    """
    return FULL if used >= available else SAMPLED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_years(tokens: list[str]) -> list[int]:
    """``["2015", "2017-2019"]`` -> ``[2015, 2017, 2018, 2019]``.

    Same spelling as ``scripts/prepare_glorys.py --years``, deliberately.
    """
    years: list[int] = []
    for token in tokens:
        if "-" in token:
            start, end = token.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(token))
    return sorted(set(years))


def year_of(file: Path) -> int:
    return int(file.stem[-4:])


def prepared_files(root: Path, years: list[int] | None = None) -> list[Path]:
    files = sorted(root.glob("glorys_1deg_[0-9][0-9][0-9][0-9].nc"))
    if not files:
        raise SystemExit(
            f"No prepared files in {root}.\n"
            'Run:  make prep-data YEARS="2015 2016"   (or: make prep-data-slurm)'
        )
    if years is None:
        return files
    wanted = set(years)
    kept = [file for file in files if year_of(file) in wanted]
    if not kept:
        raise SystemExit(
            f"--years selected {sorted(wanted)[0]}-{sorted(wanted)[-1]} but "
            f"{root} holds {year_of(files[0])}-{year_of(files[-1])}."
        )
    absent = sorted(wanted - {year_of(file) for file in kept})
    if absent:
        print(f"[years] not prepared, so not used: {absent}")
    return kept


def years_label(files: list[Path]) -> str:
    """``"1993-2025"``, or the explicit list when the years are not contiguous."""
    years = sorted(year_of(file) for file in files)
    if years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}-{years[-1]}"
    return ",".join(str(year) for year in years)


def date_index(files: list[Path]) -> list[tuple[Path, int]]:
    """Every ``(file, time-index)`` pair the selected years hold, in order.

    Separate from :func:`subsample` so that a caller can record how deeply it
    sampled: "20 dates" means nothing without the number it was drawn from.
    """
    index: list[tuple[Path, int]] = []
    for file in files:
        with xr.open_dataset(file) as ds:
            index.extend((file, i) for i in range(ds.sizes["time"]))
    return index


def subsample(index: list[tuple[Path, int]], n_dates: int) -> list[tuple[Path, int]]:
    """``n_dates`` entries spread evenly over ``index``, or all of it."""
    if n_dates >= len(index):
        return index
    picks = np.linspace(0, len(index) - 1, n_dates).round().astype(int)
    return [index[i] for i in dict.fromkeys(picks)]


def sample_dates(files: list[Path], n_dates: int) -> list[tuple[Path, int]]:
    """Pick ``n_dates`` (file, time-index) pairs spread evenly over the archive.

    Spreading them matters: sampling one year would bake that year's ENSO state
    and sea-ice extent into the normalisation.
    """
    return subsample(date_index(files), n_dates)


def read_state(ds: xr.Dataset, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one timestep as ``(surface[var, lat, lon], level[var, depth, lat, lon])``."""
    snap = ds.isel(time=time_index)
    surface = np.stack([snap[name].values for name in SURFACE_VARIABLES]).astype("f4")
    level = np.stack([snap[name].values for name in LEVEL_VARIABLES]).astype("f4")
    return surface, level


# ---------------------------------------------------------------------------
# 1. Masks
# ---------------------------------------------------------------------------
def compute_masks(files: list[Path], n_check: int) -> xr.Dataset:
    """Build the ocean masks and verify they really are constant in time."""
    print(f"[masks] deriving from {n_check} dates spread over the archive")
    picks = sample_dates(files, n_check)

    reference: dict[str, np.ndarray] | None = None
    disagreements: dict[str, int] = {}
    usi_coverage = 0.0

    for file, time_index in picks:
        with xr.open_dataset(file) as ds:
            snap = ds.isel(time=time_index)
            current = {
                # A 3-D wet mask: the ocean gets shallower as you go deeper, so
                # every depth level has its own coastline.
                "wet_level": np.isfinite(snap["thetao"].values),
                "wet_surface": np.isfinite(snap["zos"].values),
            }
            # Diagnostic only, never a mask: usi carries `cell_methods: area:
            # mean where sea_ice` and the archive is inconsistent about what that
            # means.  Until 2015-12-29 GLORYS writes an exact 0 over ice-free
            # ocean, from 2015-12-30 it writes NaN there, so this number drops
            # from ~100 % to ~25 % partway through the archive.
            usi_coverage = max(usi_coverage, float(np.isfinite(snap["usi"].values).mean()))
        if reference is None:
            reference = current
            continue
        for key, mask in current.items():
            if not np.array_equal(mask, reference[key]):
                disagreements[key] = disagreements.get(key, 0) + 1

    if reference is None:
        raise SystemExit(
            "No date could be read while checking the masks, so nothing was verified. "
            f"Are the prepared files readable? (--n-mask-checks was {n_check})"
        )
    if disagreements:
        raise SystemExit(
            "The ocean mask is NOT constant in time: "
            f"{disagreements}. Something is wrong with the prepared data."
        )
    print("[masks] wet_level and wet_surface identical across every sampled date")

    wet_level = reference["wet_level"]
    wet_surface = reference["wet_surface"]

    # The sea-ice domain is the ocean surface, full stop.  siconc/sithick/usi/vsi
    # are all `nan_means_zero`, so once the "no ice today" NaNs are filled they
    # carry a value on every wet surface cell.  Deriving this from isfinite(usi)
    # instead would make the mask depend on which dates happened to be sampled:
    # over 2016 alone it collapses to a maximum-ice-extent mask (16.7 % of the
    # globe) and nothing downstream would notice.  It also keeps the sea-ice
    # domain identical to the one the statistics below accumulate over.
    wet_seaice = wet_surface.copy()

    # Guard the invariant rather than trusting it: if the sea-ice fields ever
    # stop covering the ocean surface -- a future convention change, or someone
    # re-deriving this mask from the data -- fail here, not three tasks later.
    coverage = wet_seaice.sum() / max(wet_surface.sum(), 1)
    if coverage < 0.95:
        raise SystemExit(
            f"The sea-ice mask covers only {coverage:.1%} of the ocean surface. "
            "The sea-ice fields are filled with 0 over ice-free ocean, so it should "
            "be the whole ocean surface."
        )

    # Depth of the deepest wet level in each column, 0 over land.
    depths = np.asarray(PREPPED_DEPTHS, dtype="f4")
    n_wet_levels = wet_level.sum(axis=0)
    bathymetry = np.where(n_wet_levels > 0, depths[np.clip(n_wet_levels - 1, 0, None)], 0.0)

    for name, mask in [
        ("wet_surface", wet_surface),
        ("wet_seaice", wet_seaice),
    ]:
        print(f"[masks] {name:12s} ocean fraction {mask.mean() * 100:5.1f} %")
    print(
        f"[masks] usi was written on {usi_coverage * 100:5.1f} % of the globe on the "
        "best sampled date (informational: the sea-ice NaNs are filled with 0)"
    )
    for i, depth in enumerate(PREPPED_DEPTHS):
        if i in (0, N_DEPTH // 2, N_DEPTH - 1):
            print(
                f"[masks] wet_level  depth {depth:7.1f} m  ocean fraction "
                f"{wet_level[i].mean() * 100:5.1f} %"
            )

    # --- static fields handed to the network as extra input channels ---------
    lat = np.arange(-89.5, 90.0, 1.0, dtype="f4")
    lon = np.arange(0.0, 360.0, 1.0, dtype="f4")
    lat2d = np.broadcast_to(lat[:, None], (N_LAT, N_LON))
    lon2d = np.broadcast_to(lon[None, :], (N_LAT, N_LON))

    # log-depth, scaled to roughly [0, 1]; 0 on land.
    log_bathy = np.log1p(bathymetry)
    log_bathy = np.where(bathymetry > 0, log_bathy / max(log_bathy.max(), 1e-6), 0.0)

    constant_names = [
        "land_sea_mask",
        "log_bathymetry",
        "sin_lat",
        "cos_lat",
        "sin_lon",
        "cos_lon",
    ]
    constants = np.stack(
        [
            wet_surface.astype("f4"),
            log_bathy.astype("f4"),
            np.sin(np.deg2rad(lat2d)).astype("f4"),
            np.cos(np.deg2rad(lat2d)).astype("f4"),
            np.sin(np.deg2rad(lon2d)).astype("f4"),
            np.cos(np.deg2rad(lon2d)).astype("f4"),
        ]
    )
    # Shape (channel, 1, lat, lon): geoarches' encoder indexes constants as
    # `constant_masks[None, :, 0]`, so it expects that singleton axis.
    constants = constants[:, None]

    ds = xr.Dataset(
        data_vars=dict(
            wet_level=(("depth", "lat", "lon"), wet_level),
            wet_surface=(("lat", "lon"), wet_surface),
            wet_seaice=(("lat", "lon"), wet_seaice),
            bathymetry=(("lat", "lon"), bathymetry.astype("f4")),
            constants=(("channel", "singleton", "lat", "lon"), constants),
        ),
        coords=dict(depth=PREPPED_DEPTHS, lat=lat, lon=lon, channel=constant_names),
    )
    ds.attrs["n_dates_checked"] = n_check
    ds.attrs["description"] = (
        "Ocean masks and static input fields for the 1-degree GLORYS grid. "
        "wet_* are True over ocean. The mask is depth dependent."
    )
    return ds


# ---------------------------------------------------------------------------
# 2. Normalisation and tendency statistics
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Streaming mean/variance over ocean points, one value per (variable, level).

    Welford's algorithm keeps this numerically stable over hundreds of dates
    without ever holding the whole archive in memory.
    """

    def __init__(self, shape: tuple[int, ...], name: str = "unnamed") -> None:
        self.name = name
        self.count = np.zeros(shape, dtype="f8")
        self.mean = np.zeros(shape, dtype="f8")
        self.m2 = np.zeros(shape, dtype="f8")

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        """``values`` and ``valid`` have shape ``(*self.shape, lat, lon)``."""
        n = valid.sum(axis=(-2, -1))
        if not n.any():
            return
        masked = np.where(valid, values.astype("f8"), 0.0)
        batch_mean = np.divide(
            masked.sum(axis=(-2, -1)), n, out=np.zeros_like(n, dtype="f8"), where=n > 0
        )
        deviation = np.where(valid, values.astype("f8") - batch_mean[..., None, None], 0.0)
        batch_m2 = (deviation**2).sum(axis=(-2, -1))

        total = self.count + n
        delta = batch_mean - self.mean
        # No `np.errstate` guard: every divisor below is `np.maximum(total, 1)`,
        # so there is nothing here that can divide by zero or produce a NaN, and
        # a suppression that cannot fire only hides the next one that can.
        self.mean = np.where(total > 0, self.mean + delta * n / np.maximum(total, 1), self.mean)
        self.m2 = self.m2 + batch_m2 + delta**2 * self.count * n / np.maximum(total, 1)
        self.count = total

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        std = np.sqrt(np.divide(self.m2, np.maximum(self.count - 1, 1)))
        # A level that is entirely land has no statistics; use a harmless unit
        # scale so that normalising it cannot produce NaN or a division by zero.
        substituted = ~((self.count > 1) & (std > 1e-8))
        std = np.where(substituted, 1.0, std)
        mean = np.where(self.count > 0, self.mean, 0.0)
        # Say so.  A substituted 1.0 is indistinguishable in the output file from
        # a channel whose real standard deviation happens to be 1, and a silent
        # one would make a genuinely broken channel look normalised.
        if substituted.any():
            print(
                f"[stats] WARNING: {int(substituted.sum())} of {substituted.size} "
                f"({self.name}) channel/depth slots had no usable data "
                "(entirely land, or a constant field); their std is a substituted 1.0 "
                "and their mean a substituted 0.0, so those channels are NOT normalised."
            )
        return mean, std


def apply_seaice_fill(surface: np.ndarray, wet_surface: np.ndarray) -> np.ndarray:
    """Turn "NaN because there is no ice" into 0, leaving land NaN.

    GLORYS reports every sea-ice field with ``cell_methods: area: mean where
    sea_ice``, so an ice-free ocean cell is NaN rather than 0.  Treating those as
    missing would tell the model that most of the ocean has unknown ice cover.
    """
    surface = surface.copy()
    for name in NAN_MEANS_ZERO:
        i = SURFACE_VARIABLES.index(name)
        surface[i] = np.where(wet_surface & np.isnan(surface[i]), 0.0, surface[i])
    return surface


def compute_statistics(
    files: list[Path], masks: xr.Dataset, n_dates: int
) -> dict[str, torch.Tensor]:
    """Mean/std of each field, and std of its one-day tendency."""
    print(f"[stats] accumulating over {n_dates} dates")
    wet_surface = masks["wet_surface"].values
    wet_level = masks["wet_level"].values

    # Sea-ice fields live on the ocean mask once their "no ice" NaNs are filled,
    # which is why wet_seaice is wet_surface: normalisation and any masked loss
    # have to be computed over the same set of cells or they disagree.
    surface_valid = np.broadcast_to(wet_surface, (len(SURFACE_VARIABLES), N_LAT, N_LON))
    level_valid = np.broadcast_to(wet_level, (len(LEVEL_VARIABLES), N_DEPTH, N_LAT, N_LON))

    value_surface = WelfordAccumulator((len(SURFACE_VARIABLES),), "surface value")
    value_level = WelfordAccumulator((len(LEVEL_VARIABLES), N_DEPTH), "level value")
    delta_surface = WelfordAccumulator((len(SURFACE_VARIABLES),), "surface delta")
    delta_level = WelfordAccumulator((len(LEVEL_VARIABLES), N_DEPTH), "level delta")

    index = date_index(files)
    picks = subsample(index, n_dates)
    open_files: dict[Path, xr.Dataset] = {}
    try:
        for counter, (file, time_index) in enumerate(picks):
            if file not in open_files:
                open_files[file] = xr.open_dataset(file)
            ds = open_files[file]

            surface, level = read_state(ds, time_index)
            surface = apply_seaice_fill(surface, wet_surface)
            value_surface.update(surface, surface_valid & np.isfinite(surface))
            value_level.update(level, level_valid & np.isfinite(level))

            # One-day tendency, which is what the model actually has to predict.
            if time_index + 1 < ds.sizes["time"]:
                next_surface, next_level = read_state(ds, time_index + 1)
                next_surface = apply_seaice_fill(next_surface, wet_surface)
                d_surface = next_surface - surface
                d_level = next_level - level
                delta_surface.update(d_surface, surface_valid & np.isfinite(d_surface))
                delta_level.update(d_level, level_valid & np.isfinite(d_level))

            if (counter + 1) % 50 == 0 or counter == len(picks) - 1:
                print(f"[stats] {counter + 1}/{len(picks)} dates", flush=True)
    finally:
        for ds in open_files.values():
            ds.close()

    surface_mean, surface_std = value_surface.result()
    level_mean, level_std = value_level.result()
    _, surface_delta_std = delta_surface.result()
    _, level_delta_std = delta_level.result()

    print("\n[stats] per-variable summary (ocean points only)")
    print(f"    {'variable':<12}{'mean':>12}{'std':>12}{'1-day std':>12}")
    for i, name in enumerate(SURFACE_VARIABLES):
        print(
            f"    {name:<12}{surface_mean[i]:12.4f}{surface_std[i]:12.4f}"
            f"{surface_delta_std[i]:12.4f}"
        )
    for i, name in enumerate(LEVEL_VARIABLES):
        print(
            f"    {name + ' (surf)':<12}{level_mean[i, 0]:12.4f}{level_std[i, 0]:12.4f}"
            f"{level_delta_std[i, 0]:12.4f}"
        )

    # The statistics are broadcast against tensors of shape
    # (batch, var, depth, lat, lon), so they need a depth axis (1 for surface
    # fields) and two trailing singleton axes standing in for lat/lon.
    surface_shape = (len(SURFACE_VARIABLES), 1, 1, 1)
    level_shape = (len(LEVEL_VARIABLES), N_DEPTH, 1, 1)

    def as_tensor(array: np.ndarray, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.from_numpy(array.astype("f4")).reshape(shape)

    return {
        "surface_mean": as_tensor(surface_mean, surface_shape),
        "surface_std": as_tensor(surface_std, surface_shape),
        "level_mean": as_tensor(level_mean, level_shape),
        "level_std": as_tensor(level_std, level_shape),
        "surface_delta_std": as_tensor(surface_delta_std, surface_shape),
        "level_delta_std": as_tensor(level_delta_std, level_shape),
        "surface_variables": SURFACE_VARIABLES,
        "level_variables": LEVEL_VARIABLES,
        "depths": PREPPED_DEPTHS,
        "n_dates": len(picks),
        # Provenance, so that a checkpoint's statistics can be traced to the
        # years they came from without guessing. See the module docstring: the
        # default is every prepared year, not the training split.
        "years": years_label(files),
        "n_years": len(files),
        # ... and to the *depth* they were sampled to, which is the part that
        # used to travel nowhere: `--quick` and a full build produced files that
        # a reader could not tell apart, and the evaluation report quoted both to
        # five figures. `full` means every date a full build would have used.
        "n_dates_available": len(index),
        "sampling": sampling_label(len(picks), min(DEFAULT_N_DATES, len(index))),
    }


# ---------------------------------------------------------------------------
# 3. Monthly climatology
# ---------------------------------------------------------------------------
def _monthly_sums_for_chunk(
    args: tuple[Path, int, np.ndarray],
) -> tuple[int, np.ndarray, np.ndarray, int]:
    """Sums for one (year file, month): ``(month, surface_sum, level_sum, count)``.

    One month at a time, not a whole year: a whole-year accumulator is
    ``(12, 4, 14, 180, 360)`` float64 = 350 MB in *every* worker, so ``--jobs 8``
    alone cost ~3 GB before any of the results reached the parent.  Per month
    that drops to 29 MB, and it balances the work better across the pool.
    """
    _shrink_netcdf_chunk_cache()  # a *spawned* worker never runs main()
    file, month, wet_surface = args
    surface_sum = np.zeros((len(SURFACE_VARIABLES), N_LAT, N_LON), dtype="f8")
    level_sum = np.zeros((len(LEVEL_VARIABLES), N_DEPTH, N_LAT, N_LON), dtype="f8")
    count = 0

    with xr.open_dataset(file) as ds:
        months = ds["time"].dt.month.values
        # Count the days actually present rather than assuming a full month: 2003
        # is missing two days of February, and dividing by a nominal length would
        # bias that month low.
        for time_index in np.flatnonzero(months == month):
            surface, level = read_state(ds, int(time_index))
            surface = apply_seaice_fill(surface, wet_surface)
            surface_sum += np.nan_to_num(surface, nan=0.0)
            level_sum += np.nan_to_num(level, nan=0.0)
            count += 1
    # Accumulate in f8 but hand back f4: the result is pickled through a pipe and
    # the parent holds a few of them at once. One month of f4 sums still resolves
    # the f4 daily values to ~1e-7 relative, far below anything that matters here.
    return month, surface_sum.astype("f4"), level_sum.astype("f4"), count


def compute_climatology(
    files: list[Path], masks: xr.Dataset, jobs: int, n_years_available: int | None = None
) -> xr.Dataset:
    """Monthly means over every prepared year.

    Monthly rather than daily: the ocean's seasonal cycle is smooth, a monthly
    field is 30x smaller, and the evaluation interpolates it to the day it needs.
    """
    print(f"[clim] reading {len(files)} year(s) x 12 months with {jobs} process(es)")
    wet_surface = masks["wet_surface"].values

    surface_sum = np.zeros((12, len(SURFACE_VARIABLES), N_LAT, N_LON), dtype="f8")
    level_sum = np.zeros((12, len(LEVEL_VARIABLES), N_DEPTH, N_LAT, N_LON), dtype="f8")
    # One count per month, not one per cell.  That is exact only if every cell
    # of a field is finite on every day the month contributes -- and it is, for
    # this archive: land is NaN on *every* date and is masked back to NaN below,
    # and no field has a transient over-ocean NaN (checked by compute_masks,
    # which verifies the wet masks are constant across sampled dates; the
    # sea-ice fill removes the one convention that varies).  A per-cell valid
    # count would be the general answer and costs a second float64 array of the
    # same shape as the sums -- 12 x 4 x 14 x 180 x 360 x 8 B = 348 MB for the
    # level fields, 392 MB with the surface ones -- so it is deliberately not
    # paid here.  If you add a variable that can be missing over ocean on some
    # days, this becomes a low bias and you have to change it.
    counts = np.zeros(12, dtype="f8")
    work = [(file, month, wet_surface) for file in files for month in range(1, 13)]
    done = 0

    def accumulate(result: tuple[int, np.ndarray, np.ndarray, int]) -> None:
        nonlocal done
        month, surface, level, count = result
        surface_sum[month - 1] += surface
        level_sum[month - 1] += level
        counts[month - 1] += count
        done += 1
        if done % 24 == 0 or done == len(work):
            print(f"[clim] {done}/{len(work)} year-months", flush=True)

    if jobs > 1 and len(work) > 1:
        # Keep only a small window of tasks in flight.  ProcessPoolExecutor.map
        # submits everything at once and every finished task then parks a 15 MB
        # array in the parent until it is consumed; a window keeps that bounded.
        remaining = iter(work)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending: set[Future] = {
                pool.submit(_monthly_sums_for_chunk, item)
                for item in itertools.islice(remaining, jobs + 2)
            }
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    accumulate(future.result())
                    item = next(remaining, None)
                    if item is not None:
                        pending.add(pool.submit(_monthly_sums_for_chunk, item))
    else:
        for item in work:
            accumulate(_monthly_sums_for_chunk(item))

    if not counts.all():
        empty = [int(m) + 1 for m in np.flatnonzero(counts == 0)]
        print(f"[clim] WARNING: no data at all for month(s) {empty} -- left as NaN")
    # A month with no data must stay NaN.  A silent zero would look like a valid
    # climatology and poison every anomaly computed against it.
    # No `np.errstate` guard: `divisor` is either a positive count or NaN, never
    # zero, and dividing a finite number by NaN raises no floating-point warning.
    # The suppression that used to be here could not fire.
    divisor = np.where(counts > 0, counts, np.nan)
    surface_mean = (surface_sum / divisor[:, None, None, None]).astype("f4")
    level_mean = (level_sum / divisor[:, None, None, None, None]).astype("f4")
    print(f"[clim] days per month: min {counts.min():.0f}, max {counts.max():.0f}")

    # Put land back to NaN so plots and metrics keep ignoring it.  Each field has
    # its own coastline: uo/vo are NaN at a few hundred cells that thetao fills,
    # and the sea-ice fields are valid everywhere the surface is wet once filled.
    # Always intersect with the masks that compute_masks verified constant over
    # many dates, so that one bad first day cannot punch holes in the result.
    wet_level = masks["wet_level"].values
    with xr.open_dataset(files[0]) as ds:
        reference_surface, reference_level = read_state(ds, 0)
    for i, name in enumerate(SURFACE_VARIABLES):
        wet = wet_surface
        if name not in NAN_MEANS_ZERO:
            wet = wet & np.isfinite(reference_surface[i])
        surface_mean[:, i] = np.where(wet, surface_mean[:, i], np.nan)
    for i in range(len(LEVEL_VARIABLES)):
        wet = wet_level & np.isfinite(reference_level[i])
        level_mean[:, i] = np.where(wet, level_mean[:, i], np.nan)

    data_vars = {
        name: (("month", "lat", "lon"), surface_mean[:, i])
        for i, name in enumerate(SURFACE_VARIABLES)
    }
    data_vars.update(
        {
            name: (("month", "depth", "lat", "lon"), level_mean[:, i])
            for i, name in enumerate(LEVEL_VARIABLES)
        }
    )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            month=np.arange(1, 13),
            depth=PREPPED_DEPTHS,
            lat=masks["lat"].values,
            lon=masks["lon"].values,
        ),
    )
    # `years_label`, not first-to-last: `--quick` builds this from three years
    # spread across the archive, and a span would have labelled that "1993-2025"
    # -- indistinguishable from the full build, in the one baseline every score
    # in the report is quoted against.
    available = len(files) if n_years_available is None else int(n_years_available)
    ds.attrs["years"] = years_label(files)
    ds.attrs["n_years"] = len(files)
    ds.attrs["n_years_available"] = available
    ds.attrs["sampling"] = sampling_label(len(files), available)
    ds.attrs["description"] = (
        "Monthly climatology over the prepared years. Interpolate to day-of-year "
        "for anomaly correlation and for the climatology forecast baseline."
    )
    return ds


# ---------------------------------------------------------------------------
def main() -> int:
    _shrink_netcdf_chunk_cache()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override GLORYS_PREPPED.")
    parser.add_argument(
        "--years",
        nargs="*",
        default=[],
        help="Years to build the statistics and climatology from: '2015', '2015 2016' or "
        "'1993-2018'. Default: every prepared year, which includes test and holdout -- "
        "see the note at the top of this file.",
    )
    parser.add_argument(
        "--n-dates",
        type=int,
        default=DEFAULT_N_DATES,
        help="Dates used for the statistics. Fewer than the default is recorded as a "
        "sampled build, and says so on the face of every evaluation report.",
    )
    parser.add_argument(
        "--n-mask-checks", type=int, default=25, help="Dates used to verify masks."
    )
    parser.add_argument(
        "--jobs", type=int, default=8, help="Parallel year-months for the climatology."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Few dates and a climatology from sampled years only (~1 min).",
    )
    parser.add_argument("--skip-climatology", action="store_true")
    args = parser.parse_args()

    root = args.data_root or paths.glorys_prepped()
    files = prepared_files(root, parse_years(args.years) if args.years else None)
    print(f"prepared data: {root}  ({len(files)} year(s): {years_label(files)})")
    if not args.years:
        print(
            "years        : every prepared year, test and holdout included. The "
            "climatology is also a scored baseline, so say so when you report it; "
            "--years 1993-2018 builds the train-only version."
        )

    n_dates, n_mask_checks = args.n_dates, args.n_mask_checks
    climatology_files = files
    if args.quick:
        n_dates, n_mask_checks = 20, 5
        climatology_files = files[:: max(1, len(files) // 3)]
        print(
            f"quick mode: {n_dates} dates, climatology from {len(climatology_files)} of "
            f"{len(files)} years. Both artefacts record that they were sampled, and every "
            "evaluation report built on them says so on its face."
        )
    print()

    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)

    masks = compute_masks(files, n_mask_checks)
    masks.to_netcdf(paths.masks_file())
    print(f"[masks] wrote {paths.masks_file()}\n")

    stats = compute_statistics(files, masks, n_dates)
    torch.save(stats, paths.stats_file())
    print(f"\n[stats] wrote {paths.stats_file()}\n")

    if not args.skip_climatology:
        climatology = compute_climatology(
            climatology_files, masks, args.jobs, n_years_available=len(files)
        )
        encoding = {name: {"zlib": True, "complevel": 1} for name in climatology.data_vars}
        climatology.to_netcdf(paths.climatology_file(), encoding=encoding)
        size = paths.climatology_file().stat().st_size / 1e6
        print(f"[clim] wrote {paths.climatology_file()} ({size:.0f} MB)")

    sampled = [
        name
        for name, label in (
            ("normalisation statistics", stats["sampling"]),
            ("climatology", sampling_label(len(climatology_files), len(files))),
        )
        if label == SAMPLED and (name != "climatology" or not args.skip_climatology)
    ]
    if sampled:
        print(
            f"\nSampled, not full: {', '.join(sampled)}. Every evaluation report built on "
            "these carries that warning on its face; `make stats` builds the full set "
            '(~8 min) and `make eval ... EVAL_ARGS="--force"` rescores against it.'
        )

    print("\nDone. Run `make doctor` to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
