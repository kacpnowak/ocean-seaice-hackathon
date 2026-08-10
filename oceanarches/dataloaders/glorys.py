"""GLORYS datasets: the bridge between the prepared netCDF files and the model.

Two classes, mirroring ``geoarches.dataloaders.era5``:

``GlorysDataset``   one timestamp at a time, raw physical units, land still NaN.
                    Use it when you want to *look* at the data.
``GlorysForecast``  what training and evaluation actually use: a state, the
                    state before it, one or more states after it, normalised,
                    with no NaN anywhere.  It is a drop-in replacement for
                    ``Era5Forecast``, so geoarches' Lightning modules, metrics
                    and evaluation scripts work on our data unchanged.

``GlorysForecast.__getitem__`` returns a plain dict:

    state            TensorDict(surface=(var, 1, 180, 360), level=(var, depth, 180, 360))
    prev_state       same, at t - lead_time_hours          (only when load_prev)
    next_state       same, at t + lead_time_hours          (only when multistep > 0)
    future_states    stacked (multistep, ...)              (only when multistep > 1)
    timestamp        int32 tensor, seconds since the epoch
    lead_time_hours  int tensor

The masking order
-----------------
Applied in exactly this order; :mod:`oceanarches.dataloaders.masks` explains why.

1. read the raw fields -- land is NaN, and the four ``NAN_MEANS_ZERO`` variables
   are also NaN over ice-free ocean from 2015-12-30 onwards,
2. ``fill_seaice_nans`` -- those variables become 0 where the cell is ocean,
3. normalise ``(x - mean) / std`` with ``oceanarches/stats/glorys_1deg_stats.pt``,
4. ``nan_to_num(0.0)`` -- what is still NaN is land, and 0 after normalisation is
   the climatological mean.

Two traps this module exists to handle
--------------------------------------
**The archive has a gap.**  2003-02-07 and 2003-02-11 are missing upstream, so
the 2003 file holds 363 days.  ``Era5Forecast`` finds neighbouring states by
*index* arithmetic (``i + lead_time_hours // timedelta``), which across that gap
quietly hands you a state two days later while labelling it a one-day forecast.
We therefore validate the real timestamps of every (prev, state, next, future)
tuple and drop the samples whose spacing is wrong -- see :meth:`_rebuild_valid_index`.

**Orientation.**  GLORYS latitude runs south to north (-89.5 ... 89.5) and
longitude 0 ... 359.  ``Era5Dataset`` flips latitude (to north-first) and rolls
longitude by half a turn to centre the map on Europe.  **We do neither**: the
tensors keep GLORYS' native orientation, so ``tensor[..., 0, :]`` is the South
Pole, not the North.  Anyone comparing this file against geoarches' ERA5 loader
will notice the two missing lines; they are missing on purpose.
"""

from __future__ import annotations

import re
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from geoarches.dataloaders.netcdf import XarrayDataset
from tensordict.tensordict import TensorDict

from .. import paths
from .masks import fill_seaice_nans, load_masks
from .masks import state_mask as build_state_mask
from .variables import (
    LEVEL_VARIABLES,
    N_LAT,
    N_LON,
    PREPPED_DEPTHS,
    SURFACE_VARIABLES,
    VARIABLES,
)

__all__ = [
    "GlorysDataset",
    "GlorysForecast",
    "filename_filters",
    "SPLIT_YEARS",
    "SPLIT_DATES",
]

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
#: First and last calendar year (inclusive) of each split.  These are the years
#: a sample's *state and all its targets* must lie in.
SPLIT_YEARS: dict[str, tuple[int, int]] = {
    "train": (1993, 2018),
    "val": (2019, 2020),
    "test": (2021, 2023),
    "holdout": (2024, 2025),
    # The 30-minute model: five years of training, one of validation.
    "tiny_train": (2014, 2018),
    "tiny_val": (2019, 2019),
    # The forced-run demonstration.  Both sit INSIDE `holdout` -- see SPLIT_DATES.
    "ifs_forced_train": (2024, 2024),
    "ifs_forced_val": (2024, 2024),
    # Everything we have.  Useful for climatology and for plots, never for scoring.
    "all": (1993, 2025),
}

#: Splits whose bounds are not whole calendar years, as ``(low, high)`` with
#: ``high`` exclusive.  Only the two forced-run splits need this.
#:
#: **They are inside 2024, which is the holdout split.**  They exist because the
#: shipped IFS forcing covers valid times 2024-01-03 to 2025-01-02 and nothing
#: else, so a forced run has to be confined to the days the atmosphere is
#: actually available for -- otherwise `XarrayForcing` refuses the first batch,
#: correctly and unhelpfully.  A number measured on them is a plumbing
#: demonstration, **never a skill result**: `ifs_forced_train` is holdout data
#: and the model has been trained on the same year it is validated in.
#: See docs/05_coupling.md section 5.6.
#:
#: `ifs_forced_val` stops on 1 December so that a 10-day rollout from its last
#: initialisation still lands inside the forcing's coverage.
SPLIT_DATES: dict[str, tuple[np.datetime64, np.datetime64]] = {
    "ifs_forced_train": (
        np.datetime64("2024-01-03T00:00:00"),
        np.datetime64("2024-11-01T00:00:00"),
    ),
    "ifs_forced_val": (
        np.datetime64("2024-11-01T00:00:00"),
        np.datetime64("2024-12-01T00:00:00"),
    ),
}

_YEAR_IN_FILENAME = re.compile(r"(\d{4})")


def _year_of(filename: str) -> int | None:
    """Year encoded in ``glorys_1deg_YYYY.nc``, or None for anything else."""
    match = _YEAR_IN_FILENAME.search(filename)
    return int(match.group(1)) if match else None


def _year_filter(first: int, last: int) -> Callable[[str], bool]:
    """Keep the split's own years **and the two adjacent ones**.

    The adjacent years are needed because the state at 1 January has its previous
    state on 31 December of the year before, which lives in another file.  This
    is exactly what ``era5.filename_filters`` does (``val`` reads 2018, 2019 and
    2020 to serve a 2019 split).

    Admitting extra years is *not* the same as training on them: the constructor
    immediately calls :meth:`GlorysForecast.set_timestamp_bounds` to narrow the
    timestamps back to the split, and the sample validation then drops any state
    whose target would fall outside it.  Both halves are needed -- the filter
    alone would leak, the bounds alone would make the boundary samples unusable.
    """

    def keep(filename: str) -> bool:
        year = _year_of(filename)
        return year is not None and first - 1 <= year <= last + 1

    return keep


#: Filename filters by domain, in the style of ``era5.filename_filters``.
filename_filters: dict[str, Callable[[str], bool]] = {
    name: _year_filter(*years) for name, years in SPLIT_YEARS.items()
}
filename_filters["all"] = lambda name: _year_of(name) is not None


def _check_split_is_present(path: Path, domain: str, filename_filter: Callable) -> None:
    """Fail with the domain, the years it needs and the years on disk.

    geoarches' own message for this is ``ValueError: ('filename_filter filtered
    all files under path:', '/.../glorys_1deg_prepped')`` -- it names neither the
    split nor a single year, and it is exactly what a participant who prepared
    ``YEARS="2015 2016"`` (which is what ``make doctor`` suggests) gets from
    ``make eval``, whose default domain is ``test`` = 2021-2023.
    """
    if not path.is_dir():
        return  # a single file, or a missing path: geoarches says so clearly
    files = sorted(p.name for p in path.glob("glorys_1deg_*.nc"))
    if any(filename_filter(name) for name in files):
        return
    years = sorted({year for name in files if (year := _year_of(name)) is not None})
    have = f"{years[0]}-{years[-1]}" if years else "none"
    wanted = SPLIT_YEARS.get(domain)
    needs = f"{wanted[0]}-{wanted[1]}" if wanted else "an unknown range"
    raise ValueError(
        f"No prepared file covers the {domain!r} split ({needs}) in {path}. "
        f"Prepared years there: {have} ({len(files)} files). "
        f'Fix it with: make prep-data YEARS="{needs}"  (or make prep-data-slurm for '
        "all 33 years). `make doctor` lists which splits are covered."
    )


def split_bounds(domain: str, load_prev: bool, lead_time_hours: int) -> tuple:
    """``(low, high)`` timestamps to hand to ``set_timestamp_bounds`` for a split.

    ``high`` is the first instant *after* the split (exclusive), so no target can
    land in the next split.  ``low`` reaches one lead time back so that the first
    state of the split still has a previous state to read -- an *input* from the
    preceding split is fine, a *target* in the following one is leakage.

    Whole calendar years unless the split is in :data:`SPLIT_DATES`, which is
    where the two forced-run windows inside 2024 come from.
    """
    if domain in SPLIT_DATES:
        low, high = SPLIT_DATES[domain]
    else:
        first, last = SPLIT_YEARS[domain]
        low = np.datetime64(f"{first}-01-01T00:00:00")
        high = np.datetime64(f"{last + 1}-01-01T00:00:00")
    if load_prev:
        low = low - np.timedelta64(lead_time_hours, "h")
    return low, high


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GlorysDataset(XarrayDataset):
    """One GLORYS timestamp at a time, as a TensorDict. No normalisation.

    Shapes follow the geoarches convention:
    ``surface (var, 1, lat, lon)``, ``level (var, depth, lat, lon)``.
    Land is still NaN -- that is deliberate, so that you can see the mask.
    The sea-ice NaN fill (step 2 of the masking order) *is* applied, because
    without it the meaning of the field changes on 2015-12-30.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        domain: str = "train",
        filename_filter: Callable | None = None,
        variables: Dict[str, List[str]] | None = None,
        depth_indices: Sequence[int] | None = None,
        depth_select: str = "tensor",
        dimension_indexers: Dict[str, list] | None = None,
        return_timestamp: bool = False,
        fill_seaice: bool = True,
        masks_path: str | Path | None = None,
        limit_examples: int | None = None,
    ):
        """
        Args:
            path: Directory of prepared yearly files, or a single file.
                Defaults to ``GLORYS_PREPPED`` from ``config.env``.
            domain: One of :data:`SPLIT_YEARS`.  Picks the filename filter.
            filename_filter: Overrides ``domain``'s filter if given.
            variables: ``{"surface": [...], "level": [...]}``.  Defaults to every
                prepared variable, in the canonical order of ``variables.py``.
            depth_indices: Which of the 14 prepared levels to load, as indices
                (a model preset -- see ``DEPTH_PRESETS``).  None loads all 14.
            depth_select: How ``depth_indices`` is applied.  ``"tensor"`` (the
                default) reads every level and slices the tensor; ``"xarray"``
                pushes the selection down into ``Dataset.sel`` via
                ``dimension_indexers``, so less data comes off disk.  The second
                sounds better and measures worse: on the prepared files, picking
                6 of 14 levels with ``sel`` runs at 4.3 samples/s against 15.2
                for reading all of them, because a non-contiguous index list
                turns one chunked read into many small ones.  Kept as an option
                because that trade-off depends on the storage.
            dimension_indexers: Passed to ``Dataset.sel``.  Set automatically
                from ``depth_indices`` when ``depth_select="xarray"``; give it
                directly for anything else you want to select on.
            return_timestamp: Return ``(tensordict, timestamp)`` from ``__getitem__``.
            fill_seaice: Apply :func:`~oceanarches.dataloaders.masks.fill_seaice_nans`.
            masks_path: Mask file; defaults to ``oceanarches/stats/glorys_1deg_masks.nc``.
            limit_examples: Stop after this many timestamps (fast smoke tests).
        """
        path = Path(path) if path is not None else paths.glorys_prepped()
        if filename_filter is None:
            if domain not in filename_filters:
                raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(filename_filters)}")
            filename_filter = filename_filters[domain]

        if variables is None:
            variables = dict(surface=list(SURFACE_VARIABLES), level=list(LEVEL_VARIABLES))
        # Drop empty groups.  A surface-only component (`seaice_isolated`) is a
        # real configuration, and an empty "level" list would otherwise reach
        # xarray as a request for zero variables and fail somewhere unhelpful.
        variables = {key: list(names) for key, names in variables.items() if names}
        if not variables:
            raise ValueError(
                "No variables requested. Pass e.g. variables=dict(surface=['siconc', 'sithick'])."
            )

        self.domain = domain
        self.fill_seaice = fill_seaice
        self.depth_indices = list(depth_indices) if depth_indices is not None else None
        self.masks = load_masks(path=masks_path, depth_indices=self.depth_indices)
        self.depths = list(self.masks.depths)

        if depth_select not in ("tensor", "xarray"):
            raise ValueError(f"depth_select must be 'tensor' or 'xarray', not {depth_select!r}")
        if depth_select == "xarray" and dimension_indexers is None and self.depth_indices:
            # Select by value, not by position, because ``XarrayDataset`` uses
            # ``Dataset.sel``.  Take the values from PREPPED_DEPTHS so that the
            # selection is reproducible without opening a file first.
            dimension_indexers = {"depth": [PREPPED_DEPTHS[i] for i in self.depth_indices]}
        # Slice the tensor unless the depth selection already happened in xarray.
        self._slice_depth_in_tensor = self.depth_indices is not None and "depth" not in (
            dimension_indexers or {}
        )
        if depth_select == "xarray" and self._slice_depth_in_tensor:
            # The caller asked for the depths to be selected on the way off disk
            # but supplied their own indexers without a "depth" key, so they were
            # not.  The result is still correct -- the tensor is sliced instead --
            # but none of the I/O saving they asked for happens, and silently
            # doing the opposite of what a flag says is how a benchmark ends up
            # measuring the wrong thing.
            warnings.warn(
                'depth_select="xarray" was requested, but the dimension_indexers you '
                'passed have no "depth" key, so the depth levels are being sliced out of '
                "the tensor after the whole file has been read. Add "
                '{"depth": [...]} to dimension_indexers, or use depth_select="tensor".',
                stacklevel=2,
            )

        _check_split_is_present(path, domain, filename_filter)
        super().__init__(
            str(path),
            filename_filter=filename_filter,
            variables=variables,
            dimension_indexers=dimension_indexers,
            return_timestamp=return_timestamp,
            # Land is NaN by construction; warning on every sample would be noise.
            warning_on_nan=False,
            limit_examples=limit_examples,
        )

        # Plain name lists, in tensor-channel order.  variables.py's index
        # helpers and the metric labels expect exactly these.
        self.surface_variables = list(self.variables.get("surface", []))
        self.level_variables = list(self.variables.get("level", []))

        # Take the horizontal coordinates from the data itself rather than from
        # variables.py, so that a mismatch shows up here as a clear error instead
        # of as a silently mislabelled map three tasks later.
        with xr.open_dataset(self.files[0]) as first:
            self.lat = first["lat"].to_numpy().astype("float32")
            self.lon = first["lon"].to_numpy().astype("float32")
        grid = (len(self.lat), len(self.lon))
        if grid != tuple(self.masks.wet_surface.shape):
            raise ValueError(
                f"Grid mismatch: the data is {grid[0]}x{grid[1]} but the masks in "
                f"{paths.masks_file() if masks_path is None else masks_path} are "
                f"{tuple(self.masks.wet_surface.shape)}. Re-run: make stats"
            )
        if (grid != (N_LAT, N_LON)) and masks_path is None:
            warnings.warn(
                f"Data is on a {grid[0]}x{grid[1]} grid, not the {N_LAT}x{N_LON} "
                "declared in variables.py.",
                stacklevel=2,
            )

    def convert_to_tensordict(self, xr_dataset: xr.Dataset) -> TensorDict:
        """One time slice of the xarray dataset -> a state TensorDict."""
        if self.dimension_indexers:
            xr_dataset = xr_dataset.sel(self.dimension_indexers)
            # Tell the parent not to select again after the transpose (its own
            # comment says doing so blows up memory).
            self.already_ran_index_selection = True

        xr_dataset = xr_dataset.transpose(..., "depth", "lat", "lon")
        tdict = super().convert_to_tensordict(xr_dataset)

        if self._slice_depth_in_tensor and "level" in tdict:
            # depth_select="tensor": everything came off disk, keep the preset's
            # levels.  See the constructor for why this is the default.
            tdict["level"] = tdict["level"][:, self.depth_indices]

        # Give surface fields a length-1 depth axis so that surface and level
        # tensors have the same rank -- the whole framework assumes it.
        if "surface" in tdict:
            tdict["surface"] = tdict["surface"].unsqueeze(-3)
            if self.fill_seaice:
                # Step 2 of the masking order.  Must happen before normalisation.
                tdict["surface"] = fill_seaice_nans(
                    tdict["surface"], self.masks.wet_surface, self.surface_variables
                )

        # No latitude flip and no longitude roll, unlike Era5Dataset: we keep
        # GLORYS' native south-to-north, 0-360 orientation (see module docstring).
        return tdict

    def state_mask(self) -> TensorDict:
        """Ocean mask matching this dataset's variables and depths (1 ocean, 0 land).

        Cached, because the loss and every metric ask for it on every step.
        """
        if getattr(self, "_state_mask", None) is None:
            self._state_mask = build_state_mask(
                self.masks, self.surface_variables, self.level_variables
            )
        return self._state_mask


# ---------------------------------------------------------------------------
# Forecast dataset
# ---------------------------------------------------------------------------
class GlorysForecast(GlorysDataset):
    """The forecasting view of GLORYS: (prev, state, next...) tuples, normalised.

    Drop-in replacement for ``geoarches.dataloaders.era5.Era5Forecast``.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        domain: str = "train",
        filename_filter: Callable | None = None,
        variables: Dict[str, List[str]] | None = None,
        depth_indices: Sequence[int] | None = None,
        depth_select: str = "tensor",
        dimension_indexers: Dict[str, list] | None = None,
        lead_time_hours: int = 24,
        timedelta_hours: int = 24,
        multistep: int = 1,
        load_prev: bool = True,
        norm_scheme: str | None = "glorys",
        stats_path: str | Path | None = None,
        masks_path: str | Path | None = None,
        nan_to_num: bool = True,
        fill_seaice: bool = True,
        limit_examples: int | None = None,
    ):
        """
        Args:
            lead_time_hours: Gap between the state and its previous/next states.
                Must be a whole multiple of ``timedelta_hours``.
            timedelta_hours: Gap between two consecutive timestamps in the files.
                GLORYS daily means: 24.
            multistep: How many future states to load.  1 gives ``next_state``;
                >1 additionally gives ``future_states`` of that length.
                geoarches' forecast module raises this during training, so it is
                a settable property that rebuilds the sample index.
            load_prev: Also return the state one lead time *before*.
            norm_scheme: ``"glorys"`` or None (no normalisation).
            stats_path: Normalisation statistics.  Defaults to
                ``oceanarches/stats/glorys_1deg_stats.pt``.
            nan_to_num: Replace any remaining NaN (i.e. land) with 0 after
                normalisation.  Leave it on: NaN in, NaN gradients out.

        ``__getitem__(i, normalize=False)`` is a debugging escape hatch, not a
        training path: it skips step 3 (normalisation) and step 4 (the land
        fill), so you get physical units with land still NaN.  It does **not**
        skip step 2: ``fill_seaice_nans`` has already run inside
        ``convert_to_tensordict``, so the four ``NAN_MEANS_ZERO`` variables come
        back as 0 over ice-free ocean rather than as whatever is on disk.  To see
        the bytes as GLORYS wrote them, construct the dataset with
        ``fill_seaice=False`` as well.
        """
        if lead_time_hours % timedelta_hours:
            raise ValueError(
                f"lead_time_hours={lead_time_hours} is not a whole multiple of "
                f"timedelta_hours={timedelta_hours}"
            )

        self.lead_time_hours = lead_time_hours
        #: ``timedelta`` is geoarches' name for this and ``timedelta_hours`` is
        #: ours; they are one value under two names, kept in step here.  The
        #: parent's ``__getitem__`` and its index arithmetic read ``timedelta``,
        #: so it cannot simply be dropped.
        self.timedelta = timedelta_hours
        self.timedelta_hours = timedelta_hours
        self.load_prev = bool(load_prev)
        # Validated, not merely stored, because everything downstream only asks
        # whether it is truthy: `norm_scheme: none` in a YAML file is the *string*
        # "none", which is truthy, so the data comes back normalised -- the exact
        # opposite of what was asked, and silently. Same guard as `depth_select`.
        if norm_scheme not in (None, "glorys"):
            raise ValueError(
                f"norm_scheme must be 'glorys' or null (YAML) / None (Python), not "
                f"{norm_scheme!r}. In a config file write `norm_scheme: null`; the "
                "string 'none' is truthy and would normalise anyway."
            )
        self.norm_scheme = norm_scheme
        self.do_nan_to_num = nan_to_num
        #: Index distance between two states one lead time apart.
        self.index_step = lead_time_hours // timedelta_hours
        self._multistep = int(multistep)
        # Filled in by _rebuild_valid_index(); declared here so that the
        # multistep setter can tell "not constructed yet" from "no samples".
        self.valid_ids: np.ndarray | None = None

        super().__init__(
            path,
            domain=domain,
            filename_filter=filename_filter,
            variables=variables,
            depth_indices=depth_indices,
            depth_select=depth_select,
            dimension_indexers=dimension_indexers,
            return_timestamp=False,
            fill_seaice=fill_seaice,
            masks_path=masks_path,
            limit_examples=limit_examples,
        )

        self._load_stats(stats_path)

        # Narrow the timestamps from "the split plus its neighbouring years"
        # (what the filename filter gave us) to the split itself.
        if domain in SPLIT_YEARS:
            low, high = split_bounds(domain, self.load_prev, self.lead_time_hours)
            self.set_timestamp_bounds(low, high)
        else:
            self._rebuild_valid_index()

    # -- normalisation statistics -------------------------------------------
    def _load_stats(self, stats_path: str | Path | None) -> None:
        """Load the normalisation statistics and cut them down to our channels."""
        stats_path = Path(stats_path) if stats_path is not None else paths.stats_file()
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalisation statistics not found: {stats_path}\n"
                "run: make stats            (5-8 minutes)\n"
                "or:  make stats-quick      (about 1 minute, fewer dates -- smoke tests only)"
            )
        stats = torch.load(stats_path, weights_only=True)

        def select(kind: str, wanted: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            available = list(stats[f"{kind}_variables"])
            missing = [v for v in wanted if v not in available]
            if missing:
                raise KeyError(
                    f"{stats_path.name} has no statistics for {missing}. "
                    f"It knows about {available}. Re-run: make stats"
                )
            rows = [available.index(v) for v in wanted]
            mean, std = stats[f"{kind}_mean"][rows], stats[f"{kind}_std"][rows]
            if kind == "level" and self.depth_indices is not None:
                mean, std = mean[:, self.depth_indices], std[:, self.depth_indices]
            return mean.float(), std.float()

        entries_mean, entries_std = {}, {}
        if self.surface_variables:
            entries_mean["surface"], entries_std["surface"] = select(
                "surface", self.surface_variables
            )
        if self.level_variables:
            entries_mean["level"], entries_std["level"] = select("level", self.level_variables)

        for key, std in entries_std.items():
            if not bool((std > 0).all()):
                raise ValueError(
                    f"Non-positive standard deviation in the {key} statistics "
                    f"({stats_path}). Dividing by it would produce inf. Re-run: make stats"
                )

        self.data_mean = TensorDict(entries_mean)
        self.data_std = TensorDict(entries_std)
        #: Standard deviation of the one-lead-time tendency; Task 6 uses it to
        #: weight the loss so that slow, deep variables are not drowned out.
        self.delta_std = TensorDict(
            {
                key: (
                    stats[f"{key}_delta_std"][
                        [list(stats[f"{key}_variables"]).index(v) for v in names]
                    ]
                )
                for key, names in (
                    ("surface", self.surface_variables),
                    ("level", self.level_variables),
                )
                if names
            }
        )
        if self.depth_indices is not None and "level" in self.delta_std.keys():
            self.delta_std["level"] = self.delta_std["level"][:, self.depth_indices]

    # -- sample index --------------------------------------------------------
    @property
    def multistep(self) -> int:
        return self._multistep

    @multistep.setter
    def multistep(self, value: int) -> None:
        """geoarches raises this between epochs; the sample index has to follow.

        A longer rollout needs more future states, so more samples run off the
        end of the split and have to be dropped.

        This runs in whichever process holds the object, which for a
        ``DataLoader`` is the *parent*.  Workers hold their own copy, made when
        they were forked.  With the default ``persistent_workers=False`` they are
        re-forked every epoch and pick the change up; with
        ``persistent_workers=True`` they would keep the stale index for the rest
        of the run.  Nothing in this kit sets it -- geoarches' ``main_hydra``
        builds its loaders without it -- but if you turn it on, this is what
        breaks.
        """
        value = int(value)
        changed = value != self._multistep
        self._multistep = value
        if changed and self.valid_ids is not None:
            self._rebuild_valid_index()

    def _required_offsets(self) -> list[int]:
        """Neighbours a sample needs, in multiples of ``lead_time_hours``."""
        offsets = [-1] if self.load_prev else []
        offsets += list(range(1, self._multistep + 1))
        return offsets

    def _rebuild_valid_index(self) -> None:
        """Keep only the states whose neighbours are exactly one lead time apart.

        This is the fix for the 2003 gap.  ``Era5Forecast`` assumes that moving
        ``lead_time_hours // timedelta`` places along the timestamp list moves
        exactly ``lead_time_hours`` in time.  That is false wherever a day is
        missing, and there the model would be trained to make a 48-hour forecast
        while being told it is a 24-hour one.  Rather than paper over it, we
        check the real timestamps and drop the affected samples -- a handful out
        of twelve thousand.

        Sets ``self.valid_ids`` (positions into ``self.timestamps``) and records
        how many samples were dropped, and why, for the report.
        """
        times = np.array([t for (_, _, t) in self.timestamps], dtype="datetime64[s]")
        n = len(times)
        positions = np.arange(n)
        one_lead = np.timedelta64(self.lead_time_hours, "h").astype("timedelta64[s]")

        in_range = np.ones(n, dtype=bool)
        spacing_ok = np.ones(n, dtype=bool)
        for k in self._required_offsets():
            neighbour = positions + k * self.index_step
            inside = (neighbour >= 0) & (neighbour < n)
            in_range &= inside
            # Compare only where the neighbour exists; clip keeps the fancy
            # indexing legal, `inside` throws the nonsense away afterwards.
            clipped = np.clip(neighbour, 0, max(n - 1, 0))
            correct = (times[clipped] - times) == k * one_lead
            spacing_ok &= ~inside | correct

        valid = in_range & spacing_ok
        self.valid_ids = np.flatnonzero(valid)
        #: Samples lost because a neighbour would fall outside the split.
        self.n_dropped_at_edges = int((~in_range).sum())
        #: Samples lost because the archive skips a day near them -- the 2003 gap.
        self.n_dropped_by_time_check = int((in_range & ~spacing_ok).sum())

    def set_timestamp_bounds(self, low, high, debug: bool = False) -> None:
        """Narrow the timestamps, then rebuild the sample index over what is left."""
        super().set_timestamp_bounds(low, high, debug=debug)
        self._rebuild_valid_index()

    def state_timestamp_range(self) -> tuple[np.datetime64, np.datetime64]:
        """First and last *state* time actually reachable through ``__getitem__``."""
        if self.valid_ids is None or len(self.valid_ids) == 0:
            raise RuntimeError(f"No usable samples in domain {self.domain!r}.")
        times = [self.timestamps[i][2] for i in (self.valid_ids[0], self.valid_ids[-1])]
        return times[0].astype("datetime64[s]"), times[1].astype("datetime64[s]")

    def target_timestamp_range(self) -> tuple[np.datetime64, np.datetime64]:
        """First and last *target* time, i.e. what the model is scored against.

        This is the number that matters for leakage: it must stay inside the split.
        """
        first, last = self.state_timestamp_range()
        horizon = np.timedelta64(self.lead_time_hours * max(self._multistep, 1), "h")
        return first + np.timedelta64(self.lead_time_hours, "h"), last + horizon

    def __len__(self) -> int:
        return 0 if self.valid_ids is None else len(self.valid_ids)

    # -- reading -------------------------------------------------------------
    def _state_at(self, position: int) -> TensorDict:
        return GlorysDataset.__getitem__(self, position)

    def __getitem__(self, i: int, normalize: bool = True) -> dict:
        position = int(self.valid_ids[i])
        step = self.index_step

        out: dict = {}
        out["timestamp"] = torch.tensor(
            self.timestamps[position][2].astype("datetime64[s]").astype(np.int64),
            dtype=torch.int32,
        )
        out["state"] = self._state_at(position)
        out["lead_time_hours"] = torch.tensor(
            self.lead_time_hours * int(max(self._multistep, 1))
        ).int()

        if self._multistep > 0:
            out["next_state"] = self._state_at(position + step)
        if self._multistep > 1:
            out["future_states"] = torch.stack(
                [self._state_at(position + k * step) for k in range(1, self._multistep + 1)],
                dim=0,
            )
        if self.load_prev:
            out["prev_state"] = self._state_at(position - step)

        if normalize and self.norm_scheme:
            out = self.normalize(out)
        if normalize and self.do_nan_to_num:
            # Step 4 of the masking order.  Whatever is still NaN is land, and 0
            # after normalisation is the climatological mean.
            #
            # It also catches 630 cells per level where uo and vo are NaN even
            # though wet_level calls them ocean -- a GLORYS quirk, identical on
            # every date in the archive.  We fill them like land rather than
            # shrink the mask to match, because they are genuine ocean for every
            # other variable; a later task may want to exclude them from the
            # velocity loss, and can do so by intersecting with `uo.isnan()`.
            out = {
                key: (value.apply(lambda t: t.nan_to_num(0.0)) if "state" in key else value)
                for key, value in out.items()
            }
        return out

    # -- normalisation -------------------------------------------------------
    def normalize(self, batch):
        """``(x - mean) / std``.  Accepts a state TensorDict or a whole sample dict."""
        if self.norm_scheme is None:
            return batch
        device = next(iter(batch.values())).device
        means, stds = self.data_mean.to(device), self.data_std.to(device)
        if "surface" in batch or "level" in batch:
            return (batch - means) / stds
        return {k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()}

    def denormalize(self, batch):
        """``x * std + mean``.  geoarches calls this before every metric."""
        if self.norm_scheme is None:
            return batch
        device = next(iter(batch.values())).device
        means, stds = self.data_mean.to(device), self.data_std.to(device)
        if "surface" in batch or "level" in batch:
            return batch * stds + means
        return {k: (v * stds + means if "state" in k else v) for k, v in batch.items()}

    def iteration_hook(self, model) -> None:
        """Called by geoarches at the end of every training epoch.

        ``Era5Forecast`` uses it to switch to more recent years after a while.
        We train on the whole record from the start, so there is nothing to do --
        but the method has to exist, because geoarches calls it unconditionally.
        """
        return None

    # -- xarray conversion ---------------------------------------------------
    def convert_to_xarray(
        self,
        tdict: TensorDict,
        timestamp,
        levels: Sequence[float] | None = None,
    ) -> xr.Dataset:
        """State TensorDict -> a CF-ish xarray Dataset with land restored to NaN.

        Values are taken as they come: geoarches denormalises before calling
        this, so pass physical units unless you want normalised output on disk.

        Args:
            tdict: ``(var, depth, lat, lon)`` or ``(batch, var, depth, lat, lon)``.
            timestamp: seconds since the epoch, scalar or one per batch element.
            levels: Depths in metres to keep.  Note the unit: geoarches'
                ``Era5Dataset`` uses pressure levels here, and its forecast
                module hardcodes ``[300, 500, 700, 850]``.  Values that do not
                exist in our depth coordinate are ignored with a warning rather
                than raising, so that geoarches' own test loop still runs.
        """
        tdict = tdict.cpu()
        mask = self.state_mask()
        nan = torch.tensor(float("nan"))

        # Only the groups this dataset actually loads: a surface-only component
        # (`seaice_isolated`) has no "level" key at all.
        fields: dict[str, torch.Tensor] = {}
        for group in ("surface", "level"):
            if group not in tdict.keys():
                continue
            values = tdict[group]
            if values.ndim == 4:  # no batch axis -- add one
                values = values[None]
            # Land back to NaN.  On disk, "no data" should look like no data,
            # not like a suspiciously flat continent at the climatological mean.
            fields[group] = torch.where(mask[group].bool(), values, nan)
        if not fields:
            raise KeyError(
                f"Nothing to convert: expected a 'surface' and/or 'level' key, "
                f"got {sorted(tdict.keys())}."
            )
        n_batch = next(iter(fields.values())).shape[0]

        stamps = np.atleast_1d(np.asarray(torch.as_tensor(timestamp).cpu().numpy()).ravel())
        if len(stamps) == n_batch:
            times = pd.to_datetime(stamps, unit="s")
        elif len(stamps) == 1:
            # One timestamp for a whole batch, which is how the trajectory
            # writer calls this for a single initialisation time.
            times = pd.to_datetime(np.repeat(stamps, n_batch), unit="s")
        else:
            # Never guess here.  Task 7 writes these predictions to disk keyed on
            # the time coordinate, so a silent mismatch produces a file that
            # looks fine and is dated wrong.
            raise ValueError(
                f"convert_to_xarray got {len(stamps)} timestamps for a batch of "
                f"{n_batch}. Pass one timestamp per batch element, or a single "
                "one to apply to all of them."
            )

        data_vars = {}
        if "level" in fields:
            data_vars.update(
                {
                    name: (["time", "depth", "lat", "lon"], fields["level"][:, i].numpy())
                    for i, name in enumerate(self.level_variables)
                }
            )
        if "surface" in fields:
            # Drop the length-1 depth axis that keeps surface and level the same rank.
            surface = fields["surface"][:, :, 0]
            data_vars.update(
                {
                    name: (["time", "lat", "lon"], surface[:, i].numpy())
                    for i, name in enumerate(self.surface_variables)
                }
            )
        coords = dict(time=times, lat=self.lat, lon=self.lon)
        if "level" in fields:
            coords["depth"] = np.array(self.depths, dtype="float32")
        # Native GLORYS orientation throughout: south to north, 0 to 359 east.
        xr_dataset = xr.Dataset(data_vars=data_vars, coords=coords)
        for name, variable in xr_dataset.data_vars.items():
            meta = VARIABLES.get(name)
            if meta is not None:
                variable.attrs.update(long_name=meta.long_name, units=meta.units)
        xr_dataset["lat"].attrs.update(units="degrees_north", long_name="latitude")
        xr_dataset["lon"].attrs.update(units="degrees_east", long_name="longitude")
        if "depth" in xr_dataset.coords:
            xr_dataset["depth"].attrs.update(units="m", long_name="depth below sea surface")

        if levels is not None and "depth" in xr_dataset.coords:
            keep = [d for d in levels if d in set(self.depths)]
            if not keep:
                warnings.warn(
                    f"convert_to_xarray(levels={list(levels)}) matched none of our depth "
                    f"levels {self.depths}; our vertical coordinate is depth in metres, "
                    "not pressure. Keeping all levels.",
                    stacklevel=2,
                )
            else:
                xr_dataset = xr_dataset.sel(depth=keep)

        return xr_dataset.chunk(time=1)

    def convert_trajectory_to_xarray(
        self,
        preds_future,
        timestamp=None,
        denormalize: bool = True,
        levels: Sequence[float] | None = None,
    ) -> xr.Dataset:
        """``(batch, step, var, depth, lat, lon)`` -> xarray with a ``prediction_timedelta`` axis."""
        if denormalize:
            preds_future = self.denormalize(preds_future)
        step_iterations = preds_future.shape[1]

        per_step = [
            self.convert_to_xarray(preds_future[:, i], timestamp=timestamp, levels=levels)
            for i in range(step_iterations)
        ]
        prediction_timedeltas = [
            timedelta(hours=self.lead_time_hours * (i + 1)) for i in range(step_iterations)
        ]
        return xr.concat(per_step, pd.Index(prediction_timedeltas, name="prediction_timedelta"))
