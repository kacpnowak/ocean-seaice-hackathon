"""External forcing: fields the model *reads* but never predicts.

"External" means outside the GLORYS state -- in practice the prescribed
atmosphere: 10 m winds, air temperature, radiation, precipitation.  A forced
ocean model is told what the atmosphere does and works out what the ocean does
in response.

What this file is **not** about: the exchange between two coupled components.
When an ocean model and a sea-ice model are run together, they hand each other
fields that are part of the shared state (``siconc`` into the ocean, ``thetao``
into the ice), and that plumbing lives in the coupling task, not here.  The
distinction is worth keeping straight: forcing comes from outside the system you
are modelling, coupling happens inside it.

The default is :class:`NoForcing`, and it must stay the well-tested path.  A
model that predicts the ocean from the ocean alone is a perfectly good baseline,
and every later task has to work with ``n_channels == 0``:

    forcing = NoForcing()
    extra = forcing.get(timestamp)     # -> None
    if extra is not None:
        inputs = torch.cat([inputs, extra], dim=0)

The shipped archive is a set of *forecasts*, not an analysis
-----------------------------------------------------------
``IFS_FORCING`` in ``config.env`` is 52 weekly IFS forecast files.  Each holds
ten records stamped with the *initialisation* time in ``time_counter`` and a
``leadtime`` variable (``standard_name: forecast_period``) of 13, 37, 61, 85,
108, 132, 156, 180 or 204 hours.  Reading ``time_counter`` alone therefore sees
520 records at only 104 distinct times and concludes -- wrongly -- that the
archive is a weekly one.

:class:`XarrayForcing` indexes on the **valid time**, ``time_counter +
leadtime``.  That is 478 distinct times covering all 366 calendar days from
2024-01-03 to 2025-01-02, one or two per day.  Where several forecasts are valid
at the same instant the **shortest lead time wins**, because it is the most
accurate forecast of that instant.

Valid times fall at 00:00 and 01:00 while GLORYS daily means are stamped 12:00,
so a request is matched to the *nearest* valid time, ties going to the earlier
one, and refused when the nearest is further away than ``tolerance_hours``.
With the shipped default of 12 h that rule always returns a field valid on the
requested calendar day -- see
``tests/test_forcing.py::test_noon_requests_land_on_the_same_calendar_day``.

A time outside the archive is refused rather than served from the nearest
available field: an atmosphere six months stale looks plausible on a plot and
quietly ruins a rollout.  **2024 is the holdout split**, so a forced run trains
on holdout data; that is fine for a plumbing demonstration and must never be
reported as skill.  See ``docs/05_coupling.md`` section 5.6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .variables import N_LAT, N_LON

__all__ = [
    "ForcingSource",
    "NoForcing",
    "XarrayForcing",
    "PersistenceForcing",
    "to_datetime64",
]

#: Names a netCDF file might give the time axis, most likely first.
_TIME_DIM_CANDIDATES = ("time", "time_counter", "valid_time", "t")

#: Names a netCDF file might give the forecast lead time.  Anything carrying
#: ``standard_name: forecast_period`` is also accepted, whatever it is called.
_LEAD_CANDIDATES = ("leadtime", "lead_time", "forecast_period", "step")

#: ``units`` strings a lead-time variable may use, in seconds.
_LEAD_UNITS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}

#: The grid this project works on, as coordinate *values* rather than sizes.
_EXPECTED_LAT = -89.5 + np.arange(N_LAT, dtype="float64")
_EXPECTED_LON = np.arange(N_LON, dtype="float64")


def to_datetime64(timestamp) -> np.datetime64:
    """Accept whatever the caller has and return a ``datetime64[s]``.

    Rollout code carries timestamps as int32 seconds inside tensors, evaluation
    code carries them as numpy datetimes, and notebooks use strings.  All three
    end up here.
    """
    if isinstance(timestamp, torch.Tensor):
        if timestamp.numel() != 1:
            raise ValueError(f"Expected a single timestamp, got shape {tuple(timestamp.shape)}")
        timestamp = int(timestamp.item())
    if isinstance(timestamp, (int, np.integer)):
        return np.datetime64(int(timestamp), "s")
    if isinstance(timestamp, float):
        return np.datetime64(int(timestamp), "s")
    return np.datetime64(timestamp).astype("datetime64[s]")


class ForcingSource(ABC):
    """Where a model's external inputs come from.

    Implementations only have to answer one question -- "what is the atmosphere
    doing at this time?" -- and answer it with a tensor shaped
    ``(n_channels, 1, lat, lon)``: the same rank as a surface state, so it can be
    concatenated onto one along the channel axis.
    """

    #: Names of the channels, in order.
    variables: list[str] = []

    @property
    def n_channels(self) -> int:
        return len(self.variables)

    @abstractmethod
    def get(self, timestamp) -> torch.Tensor | None:
        """Forcing at ``timestamp``, or None when there is no forcing at all."""

    def __len__(self) -> int:
        return self.n_channels

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_channels={self.n_channels}, variables={self.variables})"


class NoForcing(ForcingSource):
    """No external forcing -- the default, and the path everything must support.

    ``get()`` returns None rather than a zero tensor on purpose: a caller that
    forgets to check gets an immediate, obvious error instead of silently
    training on a channel of zeros.
    """

    variables: list[str] = []

    def get(self, timestamp) -> None:  # noqa: D102 - inherited docstring
        return None


class XarrayForcing(ForcingSource):
    """Forcing read from netCDF files on the 180x360 GLORYS grid.

    Forecast-aware: when the files carry a lead-time variable the index is built
    on the **valid time** (initialisation + lead), and where two forecasts are
    valid at the same instant the shorter lead wins.  See the module docstring.

    Args:
        path: A file or a directory of files.
        variables: Data variables to read, in channel order.  This list *is* the
            channel order the embedder sees, so changing it changes the model.
        tolerance_hours: How far the nearest valid time may be from the requested
            one before the request is refused.  12 h is right for daily means
            stamped at noon against an archive valid at 00:00/01:00.
        mean, std: Normalisation statistics, ``(n_channels, 1, 1, 1)``.  Take
            precedence over ``stats_path``.
        stats_path: A ``.pt`` file written by ``scripts/compute_forcing_stats.py``
            (``make forcing-stats``).  Forcing has its own statistics because it
            is not part of the GLORYS state and so is not in
            ``glorys_1deg_stats.pt``.  When neither this nor ``mean``/``std`` is
            given, statistics are derived from a sample of the source itself --
            fine for a notebook, but pin them with a file for anything you train.
        normalize: Set False to get raw physical units.
        stats_max_times: How many time steps to read when deriving statistics.
        cache: Keep fields in memory after their first read.  The archive is
            random-access during training and every file is opened cold, which
            costs about 50 ms per field; caching removes that after the first
            epoch.  Set False if memory is tighter than time.
        cache_max_gb: Refuse to cache (silently, falling back to reading every
            time) when the whole archive would need more than this.
    """

    def __init__(
        self,
        path: str | Path,
        variables: list[str],
        tolerance_hours: float = 12.0,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        stats_path: str | Path | None = None,
        normalize: bool = True,
        stats_max_times: int = 64,
        cache: bool = True,
        cache_max_gb: float = 2.0,
    ):
        self.path = Path(path)
        self.variables = list(variables)
        self.tolerance = np.timedelta64(int(tolerance_hours * 3600), "s")
        self.normalize = normalize

        if not self.variables:
            raise ValueError(
                "XarrayForcing needs at least one variable. For 'no forcing at all' "
                "use NoForcing(), which is what `forcing=none` selects."
            )
        if not self.path.exists():
            raise FileNotFoundError(
                f"Forcing path does not exist: {self.path}\n"
                "Check IFS_FORCING in config.env, or use NoForcing()."
            )
        if self.path.is_file():
            self.files = [self.path]
        else:
            self.files = sorted(p for p in self.path.glob("*.nc"))
            if not self.files:
                raise FileNotFoundError(f"No .nc files under {self.path}")

        self._cached_file_id: int | None = None
        self._cached_dataset: xr.Dataset | None = None
        self._build_index()

        # A lazily filled field cache: allocated but untouched, so it only costs
        # resident memory for the times actually requested.
        n_bytes = len(self._times) * self.n_channels * N_LAT * N_LON * 4
        self._cache: torch.Tensor | None = None
        self._cached: np.ndarray | None = None
        if cache and n_bytes <= cache_max_gb * 1e9:
            self._cache = torch.empty(len(self._times), self.n_channels, 1, N_LAT, N_LON)
            self._cached = np.zeros(len(self._times), dtype=bool)

        self._check_probed_fields_have_data()

        if mean is None or std is None:
            if stats_path is not None:
                mean, std = self._read_stats_file(stats_path)
            elif not normalize:
                # Nothing will divide by them, so do not spend a pass over the
                # archive deriving statistics that are never used -- which is
                # also what lets scripts/compute_forcing_stats.py build a source
                # in order to compute the very statistics it would otherwise need.
                mean = torch.zeros(self.n_channels)
                std = torch.ones(self.n_channels)
            else:
                mean, std = self.compute_statistics(max_times=stats_max_times)
        self.mean = mean.reshape(self.n_channels, 1, 1, 1).float()
        self.std = std.reshape(self.n_channels, 1, 1, 1).float()

    # -- the index -----------------------------------------------------------
    def _build_index(self) -> None:
        """One entry per distinct valid time, shortest lead first.

        Records the raw counts too, because "520 records, 478 valid times" is
        exactly the sentence that stops the next reader concluding the archive is
        weekly.
        """
        rows: list[tuple[np.datetime64, int, int, int]] = []  # valid, lead_s, file, pos
        for file_id, file in enumerate(self.files):
            with xr.open_dataset(file, decode_timedelta=False) as ds:
                time_name = self._time_name(ds)
                self._check_grid(ds, file)
                self._check_variables(ds, file)
                stamps = ds[time_name].to_numpy().astype("datetime64[s]")
                leads = self._lead_seconds(ds, file, len(stamps))
            for position, (stamp, lead) in enumerate(zip(stamps, leads)):
                rows.append((stamp + np.timedelta64(int(lead), "s"), int(lead), file_id, position))

        #: Records read from disk, before duplicate valid times were dropped.
        self.n_records = len(rows)
        # Sorting by (valid time, lead) and keeping the first of each valid time
        # *is* the tie-break rule: shortest lead wins.
        rows.sort()
        kept: list[tuple[np.datetime64, int, int, int]] = []
        for row in rows:
            if not kept or row[0] != kept[-1][0]:
                kept.append(row)
        #: Records dropped because another forecast was valid at the same instant.
        self.n_duplicate_valid_times = len(rows) - len(kept)

        self._times = np.array([row[0] for row in kept], dtype="datetime64[s]")
        self._entries = [(row[2], row[3]) for row in kept]
        self._leads = np.array([row[1] for row in kept], dtype="int64")

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _time_name(ds: xr.Dataset) -> str:
        for candidate in _TIME_DIM_CANDIDATES:
            if candidate in ds.dims or candidate in ds.coords:
                return candidate
        raise KeyError(
            f"No time coordinate found (looked for {_TIME_DIM_CANDIDATES}); "
            f"the dataset has {sorted(ds.dims)}."
        )

    @staticmethod
    def _lead_variable(ds: xr.Dataset) -> str | None:
        """The forecast lead time, by name or by ``standard_name``."""
        for candidate in _LEAD_CANDIDATES:
            if candidate in ds.variables:
                return candidate
        for name, array in ds.variables.items():
            if array.attrs.get("standard_name") == "forecast_period":
                return str(name)
        return None

    @classmethod
    def _lead_seconds(cls, ds: xr.Dataset, file: Path, n_times: int) -> np.ndarray:
        """Lead time per record, in seconds; zeros when the file is an analysis.

        Opened with ``decode_timedelta=False``, so a lead time is a plain number
        plus a ``units`` attribute -- which is the robust thing to read anyway,
        since xarray's decoding of these varies with version.
        """
        name = cls._lead_variable(ds)
        if name is None:
            return np.zeros(n_times, dtype="int64")
        array = ds[name]
        if array.ndim != 1 or array.shape[0] != n_times:
            raise ValueError(
                f"{file.name}: {name!r} is shaped {tuple(array.shape)} but there are "
                f"{n_times} times. A lead time must be one value per record."
            )
        values = array.to_numpy()
        if np.issubdtype(values.dtype, np.timedelta64):
            return values.astype("timedelta64[s]").astype("int64")
        units = str(array.attrs.get("units", "")).strip().lower()
        scale = _LEAD_UNITS.get(units)
        if scale is None:
            raise ValueError(
                f"{file.name}: {name!r} has units {units!r}, which is not one of "
                f"{sorted(set(_LEAD_UNITS))}. The valid time cannot be computed, and "
                "guessing would silently mis-date every forcing field."
            )
        return np.rint(values.astype("float64") * scale).astype("int64")

    @staticmethod
    def _check_grid(ds: xr.Dataset, file: Path) -> None:
        """Refuse anything that is not our grid, loudly and early.

        Regridding is a real job with real choices in it (conservative? bilinear?
        what about the coast?), not something a dataloader should do behind your
        back at 3 a.m.  Both the size *and* the coordinate values are checked: a
        180x360 file with latitudes running north to south has the right shape
        and an upside-down atmosphere.
        """
        shape = (ds.sizes.get("lat"), ds.sizes.get("lon"))
        if shape != (N_LAT, N_LON):
            raise ValueError(
                f"{file.name} is on a {shape[0]}x{shape[1]} grid, but this project "
                f"works on {N_LAT}x{N_LON} (1 degree, lat -89.5..89.5, lon 0..359). "
                "Regrid it first, e.g. `cdo remapbil,r360x180 in.nc out.nc`."
            )
        for name, expected in (("lat", _EXPECTED_LAT), ("lon", _EXPECTED_LON)):
            if name not in ds.variables:
                raise ValueError(
                    f"{file.name} has a {name} dimension of the right size but no {name} "
                    "coordinate, so there is no way to check which way up it is. Add one."
                )
            found = ds[name].to_numpy().astype("float64")
            if np.allclose(found, expected, atol=1e-3):
                continue
            hint = ""
            if np.allclose(found, expected[::-1], atol=1e-3):
                hint = (
                    f" It is the same axis reversed: flip it with "
                    f"`ds.isel({name}=slice(None, None, -1))` (or `cdo invertlat`). Do flip "
                    "it -- the size check cannot see an upside-down atmosphere."
                )
            elif name == "lon" and np.allclose(np.sort(found % 360), expected, atol=1e-3):
                hint = " It looks like -180..179; roll it to 0..359 first."
            raise ValueError(
                f"{file.name} has {name} running {found[0]:g}..{found[-1]:g} "
                f"(step {found[1] - found[0]:g}), but this project works on "
                f"{name} {expected[0]:g}..{expected[-1]:g} (step 1).{hint}"
            )

    def _check_variables(self, ds: xr.Dataset, file: Path) -> None:
        missing = [v for v in self.variables if v not in ds.data_vars]
        if missing:
            raise KeyError(
                f"{file.name} has no variables {missing}. It contains {sorted(ds.data_vars)}."
            )
        flat = [v for v in self.variables if not {"lat", "lon"} <= set(ds[v].dims)]
        if flat:
            raise ValueError(
                f"{file.name}: {flat} have no lat/lon dimensions, so they are not fields. "
                f"({self._lead_variable(ds)!r} is the lead time, not a forcing channel.)"
            )

    def _check_probed_fields_have_data(self) -> None:
        """Refuse a channel that is NaN everywhere at the first or last valid time.

        MEASURED, and the reason this check exists: ``somslpre`` in the shipped
        IFS archive is NaN in **all 520 records of all 52 files**.  Requested as
        a channel it survives ``nan_to_num`` as a constant zero -- the exact
        failure ``NoForcing`` is written to make impossible, only silent, and
        one eighth of the forcing the model was told it was getting.

        Two fields are probed, from the two ends of the archive, which on the
        shipped one means two different files.  Measured: 13 ms on top of a
        430 ms construction, so the second probe is free.  It is still a probe,
        not a scan -- a channel that goes empty only in the middle of an archive
        would pass here and be caught by ``compute_statistics``, which reads
        every time and raises on a channel with no finite values at all.
        """
        probes = sorted({0, len(self._times) - 1})
        for index in probes:
            field = self._read(index)
            empty = [
                name
                for name, all_nan in zip(self.variables, field.isnan().all(dim=(1, 2, 3)).tolist())
                if all_nan
            ]
            if empty:
                raise ValueError(
                    f"{empty} are NaN over the whole grid at {self._times[index]} in "
                    f"{self.path}. A channel with no data in it normalises to a constant "
                    "and teaches the model nothing; drop it from the variables list. (In "
                    "the shipped IFS archive `somslpre` is empty in every record -- see "
                    "docs/05_coupling.md.)"
                )

    def _open(self, file_id: int) -> xr.Dataset:
        if self._cached_file_id != file_id:
            if self._cached_dataset is not None:
                self._cached_dataset.close()
            self._cached_dataset = xr.open_dataset(self.files[file_id], decode_timedelta=False)
            self._cached_file_id = file_id
        return self._cached_dataset

    def _read(self, index: int) -> torch.Tensor:
        """Raw field at index ``index`` of the deduplicated valid times."""
        if self._cache is not None and self._cached[index]:
            return self._cache[index]
        file_id, position = self._entries[index]
        ds = self._open(file_id)
        slice_ = ds[self.variables].isel({self._time_name(ds): position})
        array = slice_.to_array().to_numpy()  # (var, lat, lon)
        tensor = torch.from_numpy(np.asarray(array)).float().unsqueeze(-3)  # (var, 1, lat, lon)
        if self._cache is not None:
            self._cache[index] = tensor
            self._cached[index] = True
            return self._cache[index]
        return tensor

    def compute_statistics(
        self, max_times: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel mean and standard deviation over the valid times.

        Streamed in float64 rather than stacked, because the whole archive is
        1.2 GB of float32 and mean sea-level pressure squares to 1e10.
        Land is *included*: the atmosphere is defined everywhere, unlike the
        ocean state, whose statistics are ocean-only (see
        ``scripts/compute_stats.py``).

        Args:
            max_times: Use an evenly spaced sample of this many times.  None uses
                every one.
        """
        n_times = len(self._times)
        if max_times is None or max_times >= n_times:
            picks = range(n_times)
        else:
            step = max(1, n_times // max_times)
            picks = range(0, n_times, step)
        total = torch.zeros(self.n_channels, dtype=torch.float64)
        total_sq = torch.zeros(self.n_channels, dtype=torch.float64)
        counts = torch.zeros(self.n_channels, dtype=torch.float64)
        for index in picks:
            field = self._read(index).double()
            finite = ~field.isnan()
            values = field.nan_to_num(0.0)
            total += values.sum(dim=(1, 2, 3))
            total_sq += (values**2).sum(dim=(1, 2, 3))
            counts += finite.sum(dim=(1, 2, 3))
        if bool((counts == 0).any()):
            empty = [v for v, c in zip(self.variables, counts.tolist()) if c == 0]
            raise ValueError(f"{empty} are entirely missing in {self.path}; no statistics.")
        mean = total / counts
        variance = (total_sq / counts - mean**2).clamp(min=0.0)
        return mean.float(), variance.sqrt().float().clamp(min=1e-6)

    def _read_stats_file(self, stats_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Forcing statistics not found: {stats_path}\n"
                "run: make forcing-stats            (about a minute)"
            )
        stats = torch.load(stats_path, weights_only=True)
        available = list(stats["variables"])
        missing = [v for v in self.variables if v not in available]
        if missing:
            raise KeyError(
                f"{stats_path.name} has no statistics for {missing}. It knows about "
                f"{available}. Re-run: make forcing-stats"
            )
        rows = [available.index(v) for v in self.variables]
        mean = stats["mean"].reshape(len(available), -1)[rows]
        std = stats["std"].reshape(len(available), -1)[rows]
        if not bool((std > 0).all()):
            raise ValueError(
                f"Non-positive standard deviation in {stats_path}; dividing by it would "
                "produce inf. Re-run: make forcing-stats"
            )
        #: Where the normalisation came from, so a run can say so.
        self.stats_source = str(stats_path)
        return mean.float(), std.float()

    @property
    def time_range(self) -> tuple[np.datetime64, np.datetime64]:
        """First and last **valid** time in the archive."""
        return self._times[0], self._times[-1]

    @property
    def n_times(self) -> int:
        """Distinct valid times, i.e. how many different fields can be served."""
        return len(self._times)

    @property
    def lead_time_range(self) -> tuple[float, float]:
        """Shortest and longest forecast lead actually used, in hours."""
        return float(self._leads.min()) / 3600, float(self._leads.max()) / 3600

    # -- the interface -------------------------------------------------------
    def _nearest(self, wanted: np.datetime64) -> int:
        """Index of the valid time nearest ``wanted``; a tie goes to the earlier.

        ``argmin`` over the whole array would also work, but its tie-break is
        "whichever numpy happens to reach first", and ties are not exotic here:
        a noon request sits exactly 12 h from midnight on both sides.
        """
        insert = int(np.searchsorted(self._times, wanted))
        best, best_offset = insert, None
        for candidate in (insert - 1, insert):
            if 0 <= candidate < len(self._times):
                offset = abs(self._times[candidate] - wanted)
                if best_offset is None or offset < best_offset:
                    best, best_offset = candidate, offset
        return best

    def get(self, timestamp) -> torch.Tensor:
        """Forcing nearest to ``timestamp``, shaped ``(n_channels, 1, lat, lon)``."""
        wanted = to_datetime64(timestamp)
        first, last = self.time_range
        if not (first - self.tolerance <= wanted <= last + self.tolerance):
            raise ValueError(
                f"No forcing for {wanted}: {self.path} covers valid times {first} to "
                f"{last} ({self.n_times} of them). The shipped IFS forcing is one year "
                "(2024, the holdout split), so a run on train/val/test has to use "
                "NoForcing() -- `forcing=none`, the default -- or bring its own "
                "atmosphere. See docs/05_coupling.md section 5.6."
            )
        index = self._nearest(wanted)
        offset = abs(self._times[index] - wanted)
        if offset > self.tolerance:
            raise ValueError(
                f"Nearest forcing time to {wanted} is {self._times[index]}, "
                f"{offset / np.timedelta64(1, 'h'):.1f} h away, more than the "
                f"{self.tolerance / np.timedelta64(1, 'h'):.1f} h tolerance. "
                "Raise tolerance_hours if that is genuinely acceptable -- but a stale "
                "atmosphere is worse than no atmosphere."
            )
        tensor = self._read(index)
        if self.normalize:
            tensor = (tensor - self.mean) / self.std
        return tensor.nan_to_num(0.0)

    def __repr__(self) -> str:
        first, last = self.time_range
        return (
            f"XarrayForcing(n_channels={self.n_channels}, variables={self.variables}, "
            f"n_times={self.n_times}, valid {first}..{last})"
        )


class PersistenceForcing(ForcingSource):
    """Freeze the forcing at the first time asked for.

    An ablation: run the identical model with the atmosphere held still and see
    how much of the forecast skill came from the forcing rather than from the
    ocean's own dynamics.
    """

    def __init__(self, source: ForcingSource):
        self.source = source
        self.variables = list(source.variables)
        self._frozen: torch.Tensor | None = None
        self._frozen_at = None

    @property
    def frozen_at(self):
        """The timestamp the forcing is stuck at, or None before the first call."""
        return self._frozen_at

    def reset(self) -> None:
        """Forget the frozen field, so the next call re-freezes at a new time."""
        self._frozen = None
        self._frozen_at = None

    def get(self, timestamp) -> torch.Tensor | None:
        if self._frozen_at is None:
            self._frozen = self.source.get(timestamp)
            self._frozen_at = to_datetime64(timestamp)
        return self._frozen
