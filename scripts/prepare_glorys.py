#!/usr/bin/env python
"""Turn raw daily GLORYS files into one compact netCDF per year.

The raw archive is one file per day (~53 MB each, uncompressed, 50 depth levels).
That is 640 GB and 12 000 files -- workable, but slow to open and mostly levels we
do not train on.  This script writes one file per year containing only the depth
levels we use, compressed, and chunked so that reading a single (day, level) slab
is cheap.

    <GLORYS_RAW>/YYYY/MM/mercatorglorys12v1_gl12_mean_YYYYMMDD_R*.nc   (input)
    <GLORYS_PREPPED>/glorys_1deg_YYYY.nc                               (output)

Missing values are deliberately *kept as NaN*.  Handling them is part of the
challenge, and the dataloader does it explicitly and visibly -- see
docs/02_data_and_masking.md.

Examples
--------
    python scripts/prepare_glorys.py --years 2015 2016     # two years
    python scripts/prepare_glorys.py --years 1993-2025     # a range
    python scripts/prepare_glorys.py                       # everything available
    python scripts/prepare_glorys.py --years 2015 --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num

# Make `oceanarches` importable when this script is run directly from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oceanarches import paths  # noqa: E402
from oceanarches.dataloaders.variables import (  # noqa: E402
    LEVEL_VARIABLES,
    N_LAT,
    N_LON,
    PREPPED_DEPTH_INDICES,
    PREPPED_DEPTHS,
    SURFACE_VARIABLES,
    VARIABLES,
)

FILENAME_DATE = re.compile(r"_(\d{8})_R\d{8}\.nc$")
TIME_UNITS = "days since 1993-01-01 12:00:00"
TIME_CALENDAR = "proleptic_gregorian"


# ---------------------------------------------------------------------------
# Discovering the input
# ---------------------------------------------------------------------------
def parse_years(tokens: list[str]) -> list[int]:
    """Accept ``2015``, ``2015 2016`` and ``1993-2025`` in any combination."""
    years: list[int] = []
    for token in tokens:
        if "-" in token:
            start, end = token.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(token))
    return sorted(set(years))


def available_years(raw_root: Path) -> list[int]:
    return sorted(
        int(p.name) for p in raw_root.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}", p.name)
    )


def daily_files(raw_root: Path, year: int) -> dict[date, Path]:
    """Map date -> raw file for one year, keyed by the date in the filename."""
    found: dict[date, Path] = {}
    for path in sorted((raw_root / str(year)).glob("*/*.nc")):
        match = FILENAME_DATE.search(path.name)
        if not match:
            continue
        stamp = datetime.strptime(match.group(1), "%Y%m%d").date()
        if stamp.year == year:
            found[stamp] = path
    return found


def expected_dates(year: int) -> list[date]:
    day, out = date(year, 1, 1), []
    while day.year == year:
        out.append(day)
        day += timedelta(days=1)
    return out


#: A year may be missing this many days and still be prepared by default.
#: 2003 is missing two days in the real archive and is a perfectly good training
#: year; a year that is missing far more than that is not a gap, it is a year the
#: archive has not finished.
DEFAULT_MAX_MISSING_DAYS = 10


def refusal_reason(year: int, missing: list[date], max_missing: int) -> str | None:
    """Why this year should not be prepared, or ``None`` to go ahead.

    The archive grows.  ``python scripts/prepare_glorys.py`` with no ``--years``
    prepares *everything it finds*, so the day the raw tree gains the first 83
    days of a new year, the bare command silently adds a quarter-year file to a
    dataset that otherwise holds whole years -- and it looks exactly like the
    others in ``make doctor``, in the file listing and in the split filters.
    That happened to this project during review.

    A handful of absent days is a different thing and is allowed: the dataloader
    drops the samples that need them.
    """
    if len(missing) <= max_missing:
        return None
    total = len(expected_dates(year))
    return (
        f"{len(missing)} of {total} days are absent from the raw archive, which is "
        f"more than --max-missing-days ({max_missing}). This is what a year the "
        f"archive has not finished looks like. Prepared, it would sit in "
        f"GLORYS_PREPPED looking like a whole year. Pass --allow-incomplete if you "
        f"really do want a partial {year}."
    )


# ---------------------------------------------------------------------------
# Writing the output
# ---------------------------------------------------------------------------
def create_output(path: Path, dates: list[date], complevel: int) -> Dataset:
    """Create the yearly file with all dimensions, coordinates and empty variables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = Dataset(path, "w", format="NETCDF4")

    out.createDimension("time", len(dates))
    out.createDimension("depth", len(PREPPED_DEPTHS))
    out.createDimension("lat", N_LAT)
    out.createDimension("lon", N_LON)

    time_var = out.createVariable("time", "f8", ("time",))
    time_var.units = TIME_UNITS
    time_var.calendar = TIME_CALENDAR
    time_var.standard_name = "time"
    time_var[:] = date2num(
        [datetime(d.year, d.month, d.day, 12) for d in dates],
        units=TIME_UNITS,
        calendar=TIME_CALENDAR,
    )

    depth_var = out.createVariable("depth", "f4", ("depth",))
    depth_var.units = "m"
    depth_var.positive = "down"
    depth_var.standard_name = "depth"
    depth_var[:] = np.array(PREPPED_DEPTHS, dtype="f4")

    lat_var = out.createVariable("lat", "f4", ("lat",))
    lat_var.units = "degrees_north"
    lat_var.standard_name = "latitude"
    lat_var[:] = np.arange(-89.5, 90.0, 1.0, dtype="f4")

    lon_var = out.createVariable("lon", "f4", ("lon",))
    lon_var.units = "degrees_east"
    lon_var.standard_name = "longitude"
    lon_var[:] = np.arange(0.0, 360.0, 1.0, dtype="f4")

    # Chunk one (day) or one (day, level) slab at a time: that is exactly the
    # access pattern of the dataloader, so no read ever decompresses more than
    # it needs.
    common = dict(zlib=complevel > 0, complevel=complevel, fill_value=np.nan)
    for name in SURFACE_VARIABLES:
        var = out.createVariable(
            name, "f4", ("time", "lat", "lon"), chunksizes=(1, N_LAT, N_LON), **common
        )
        var.long_name = VARIABLES[name].long_name
        var.units = VARIABLES[name].units
    for name in LEVEL_VARIABLES:
        var = out.createVariable(
            name,
            "f4",
            ("time", "depth", "lat", "lon"),
            chunksizes=(1, 1, N_LAT, N_LON),
            **common,
        )
        var.long_name = VARIABLES[name].long_name
        var.units = VARIABLES[name].units

    out.title = "GLORYS12V1 daily means, regridded to 1 degree, depth-subset"
    out.source = "MERCATOR GLORYS12V1 via cdo remap,r360x180"
    out.comment = (
        "Prepared for the AI Ocean & Sea-Ice hackathon challenge. "
        "Missing values are kept as NaN on purpose -- masking is part of the task."
    )
    out.depth_indices_in_native_grid = str(PREPPED_DEPTH_INDICES)
    return out


def _to_nan_filled(raw) -> np.ndarray:
    """netCDF4 hands back a *masked* array; turn the mask into NaN.

    This matters: ``np.asarray(masked_array)`` silently drops the mask and gives
    you the raw fill values (for GLORYS, about -9944 after scaling), which then
    look like real ocean temperatures.  Always go through ``np.ma.filled``.
    """
    data = np.ma.filled(np.ma.asarray(raw).astype("f4"), np.nan)
    # Belt and braces: any remaining +/-inf also becomes NaN.
    return np.where(np.isfinite(data), data, np.nan)


def copy_day(src_path: Path, out: Dataset, index: int) -> dict[str, int]:
    """Copy one day's fields into slot ``index`` of the output file."""
    nan_counts: dict[str, int] = {}
    with Dataset(src_path, "r") as src:
        for name in SURFACE_VARIABLES:
            data = _to_nan_filled(src[name][0])
            out[name][index, :, :] = data
            nan_counts[name] = int(np.isnan(data).sum())
        for name in LEVEL_VARIABLES:
            # netCDF4 fancy-indexes the depth axis for us and only reads those slabs.
            data = _to_nan_filled(src[name][0, PREPPED_DEPTH_INDICES])
            out[name][index, :, :, :] = data
            nan_counts[name] = int(np.isnan(data).sum())
    return nan_counts


def prepare_year(
    year: int,
    raw_root: Path,
    out_root: Path,
    complevel: int,
    overwrite: bool,
    max_missing: int = DEFAULT_MAX_MISSING_DAYS,
    allow_incomplete: bool = False,
) -> dict | None:
    """Write ``glorys_1deg_<year>.nc``.  Returns a manifest entry, or None if skipped."""
    out_path = out_root / f"glorys_1deg_{year}.nc"
    if out_path.exists() and not overwrite:
        print(f"[{year}] already exists, skipping (use --overwrite to rebuild)")
        return None

    found = daily_files(raw_root, year)
    if not found:
        print(f"[{year}] no raw files found under {raw_root / str(year)}, skipping")
        return None

    wanted = expected_dates(year)
    missing = [d for d in wanted if d not in found]
    dates = [d for d in wanted if d in found]
    if not allow_incomplete:
        reason = refusal_reason(year, missing, max_missing)
        if reason is not None:
            print(f"[{year}] REFUSED: {reason}")
            return None
    if missing:
        print(
            f"[{year}] WARNING: {len(missing)} day(s) missing from the raw archive, "
            f"e.g. {missing[:3]}. They are dropped, so this year is NOT continuous."
        )

    started = time.time()
    tmp_path = out_path.with_suffix(".nc.tmp")
    out = create_output(tmp_path, dates, complevel)
    try:
        last_nan_counts: dict[str, int] = {}
        for index, day in enumerate(dates):
            last_nan_counts = copy_day(found[day], out, index)
            if index % 60 == 0 or index == len(dates) - 1:
                elapsed = time.time() - started
                print(
                    f"[{year}] {index + 1:>3}/{len(dates)} days  "
                    f"({elapsed:5.1f}s, {(index + 1) / max(elapsed, 1e-9):.1f} day/s)",
                    flush=True,
                )
    finally:
        out.close()

    tmp_path.replace(out_path)
    elapsed = time.time() - started
    size_gb = out_path.stat().st_size / 1e9
    print(f"[{year}] wrote {out_path.name}  {size_gb:.2f} GB in {elapsed / 60:.1f} min")

    return {
        "year": year,
        "file": out_path.name,
        "n_days": len(dates),
        "missing_days": [d.isoformat() for d in missing],
        "continuous": not missing,
        "size_bytes": out_path.stat().st_size,
        "seconds": round(elapsed, 1),
        "nan_counts_last_day": last_nan_counts,
    }


def _prepare_year_star(item: tuple) -> dict | None:
    """``prepare_year`` with its arguments packed into a tuple, for ProcessPoolExecutor."""
    return prepare_year(*item)


def update_manifest(out_root: Path, entries: list[dict]) -> None:
    """Merge new entries into ``prep_manifest.json`` so partial runs accumulate."""
    manifest_path = out_root / "prep_manifest.json"
    manifest = {"depths": PREPPED_DEPTHS, "depth_indices": PREPPED_DEPTH_INDICES, "years": {}}
    if manifest_path.exists():
        manifest.update(json.loads(manifest_path.read_text()))
    for entry in entries:
        manifest["years"][str(entry["year"])] = entry
    manifest["surface_variables"] = SURFACE_VARIABLES
    manifest["level_variables"] = LEVEL_VARIABLES
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=[],
        help="Years to prepare: '2015', '2015 2016' or '1993-2025'. Default: all available.",
    )
    parser.add_argument("--raw-root", type=Path, default=None, help="Override GLORYS_RAW.")
    parser.add_argument("--out-root", type=Path, default=None, help="Override GLORYS_PREPPED.")
    parser.add_argument(
        "--complevel", type=int, default=1, help="netCDF zlib compression level (0 disables)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild years that already exist."
    )
    parser.add_argument(
        "--max-missing-days",
        type=int,
        default=DEFAULT_MAX_MISSING_DAYS,
        help="Refuse a year missing more than this many days from the raw archive.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Prepare a year even when most of it is absent from the raw archive. "
        "Only for a deliberately partial dataset -- see docs/01_setup.md.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Prepare this many years in parallel (one process per year). "
        "The full 1993-2025 archive takes ~5 min with --jobs 16.",
    )
    args = parser.parse_args()

    raw_root = args.raw_root or paths.glorys_raw()
    out_root = args.out_root or paths.glorys_prepped()
    if not raw_root.is_dir():
        parser.error(f"raw GLORYS root does not exist: {raw_root}")

    years = parse_years(args.years) if args.years else available_years(raw_root)
    print(f"raw     : {raw_root}")
    print(f"output  : {out_root}")
    print(f"years   : {years[0]}-{years[-1]} ({len(years)})" if years else "years: none")
    print(
        f"depths  : {len(PREPPED_DEPTHS)} levels, {PREPPED_DEPTHS[0]:.1f} m -> {PREPPED_DEPTHS[-1]:.0f} m"
    )
    print()

    work = [
        (
            year,
            raw_root,
            out_root,
            args.complevel,
            args.overwrite,
            args.max_missing_days,
            args.allow_incomplete,
        )
        for year in years
    ]
    if args.jobs > 1 and len(work) > 1:
        # One process per year. Each year is an independent read/write, so this
        # scales until the filesystem, not the CPU, becomes the bottleneck.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(_prepare_year_star, work))
    else:
        results = [_prepare_year_star(item) for item in work]

    entries = [entry for entry in results if entry is not None]
    if entries:
        update_manifest(out_root, entries)

    not_continuous = [e["year"] for e in entries if not e["continuous"]]
    if not_continuous:
        print(f"\nWARNING: these years have missing days: {not_continuous}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
