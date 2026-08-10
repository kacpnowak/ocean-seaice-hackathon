"""Shared fixtures: a miniature GLORYS archive that behaves like the real one.

The real archive is 92 GB, so the tests build their own: the same variables, the
same 14 depth levels and the same file layout, on a 6x8 grid instead of 180x360.
It also reproduces the two traps the dataloader exists to handle -- a two-day
gap in one year, and sea-ice fields that switch from "0 over ice-free ocean" to
"NaN over ice-free ocean" partway through -- so that the tests exercise the real
failure modes rather than an idealised version of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from oceanarches.dataloaders.variables import (
    LEVEL_VARIABLES,
    PREPPED_DEPTHS,
    SURFACE_VARIABLES,
)

# A small grid: big enough to have land, coasts and a depth-dependent coastline,
# small enough that a decade of daily data is a few megabytes.
N_LAT, N_LON = 6, 8
N_DEPTH = len(PREPPED_DEPTHS)

#: Years written by the `tiny_archive` fixture.
ARCHIVE_YEARS = (2017, 2018, 2019, 2020, 2021, 2022)

#: Days deleted from 2019, mirroring the real 2003-02-07 / 2003-02-11 gap.
MISSING_DAYS = (np.datetime64("2019-02-07"), np.datetime64("2019-02-11"))

#: Date on which the synthetic sea-ice fields switch convention, mirroring the
#: real 2015-12-30.
SEAICE_CONVENTION_SWITCH = np.datetime64("2019-07-01")


def _grid() -> tuple[np.ndarray, np.ndarray]:
    lat = (-89.5 + np.arange(N_LAT) * (180.0 / N_LAT)).astype("float32")
    lon = (np.arange(N_LON) * (360.0 / N_LON)).astype("float32")
    return lat, lon


def _wet_masks() -> tuple[np.ndarray, np.ndarray]:
    """A surface mask and a depth-dependent one, both deterministic.

    The deep levels are progressively drier, like a real ocean: the shelf runs
    out before the abyss does.
    """
    rng = np.random.default_rng(0)
    wet_surface = rng.random((N_LAT, N_LON)) > 0.3
    wet_surface[0, :] = False  # a polar row of "land", like GLORYS' missing south
    wet_surface[2, 3] = True  # guarantee at least one always-ocean cell
    wet_level = np.empty((N_DEPTH, N_LAT, N_LON), dtype=bool)
    for k in range(N_DEPTH):
        deeper = rng.random((N_LAT, N_LON)) > (k / (2 * N_DEPTH))
        wet_level[k] = wet_surface & deeper
    wet_level[:, 2, 3] = True
    return wet_surface, wet_level


@pytest.fixture(scope="session")
def tiny_masks_file(tmp_path_factory) -> Path:
    """A mask file with the same variables and shapes as the real one."""
    lat, lon = _grid()
    wet_surface, wet_level = _wet_masks()
    # 0.0 over land, exactly as `scripts/compute_stats.py` writes it -- NOT NaN.
    # The fixture used to build it with NaN, which is why nothing noticed that
    # `masks.py` documented `bathymetry` as "NaN over land" while the shipped
    # file has zero NaNs. A fixture that is kinder than the real artefact hides
    # the bug it was meant to expose.
    bathymetry = np.where(wet_surface, 4000.0, 0.0).astype("float32")

    constants = np.stack(
        [
            wet_surface.astype("float32"),
            np.nan_to_num(np.log1p(bathymetry), nan=0.0),
            np.broadcast_to(np.sin(np.deg2rad(lat))[:, None], (N_LAT, N_LON)),
            np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (N_LAT, N_LON)),
            np.broadcast_to(np.sin(np.deg2rad(lon))[None, :], (N_LAT, N_LON)),
            np.broadcast_to(np.cos(np.deg2rad(lon))[None, :], (N_LAT, N_LON)),
        ]
    ).astype("float32")[:, None]

    ds = xr.Dataset(
        data_vars=dict(
            wet_level=(["depth", "lat", "lon"], wet_level),
            wet_surface=(["lat", "lon"], wet_surface),
            wet_seaice=(["lat", "lon"], wet_surface),
            bathymetry=(["lat", "lon"], bathymetry),
            constants=(["channel", "singleton", "lat", "lon"], constants),
        ),
        coords=dict(
            depth=np.array(PREPPED_DEPTHS),
            lat=lat,
            lon=lon,
            channel=[
                "land_sea_mask",
                "log_bathymetry",
                "sin_lat",
                "cos_lat",
                "sin_lon",
                "cos_lon",
            ],
        ),
    )
    path = tmp_path_factory.mktemp("stats") / "masks.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture(scope="session")
def tiny_stats_file(tmp_path_factory) -> Path:
    """Normalisation statistics in the layout `make stats` writes."""
    n_surface, n_level = len(SURFACE_VARIABLES), len(LEVEL_VARIABLES)
    rng = np.random.default_rng(1)
    stats = {
        "surface_mean": torch.from_numpy(rng.normal(size=(n_surface, 1, 1, 1))).float(),
        "surface_std": torch.from_numpy(rng.uniform(0.5, 2.0, size=(n_surface, 1, 1, 1))).float(),
        "level_mean": torch.from_numpy(rng.normal(size=(n_level, N_DEPTH, 1, 1))).float(),
        "level_std": torch.from_numpy(
            rng.uniform(0.5, 2.0, size=(n_level, N_DEPTH, 1, 1))
        ).float(),
        "surface_delta_std": torch.from_numpy(
            rng.uniform(0.1, 0.5, size=(n_surface, 1, 1, 1))
        ).float(),
        "level_delta_std": torch.from_numpy(
            rng.uniform(0.1, 0.5, size=(n_level, N_DEPTH, 1, 1))
        ).float(),
        "surface_variables": list(SURFACE_VARIABLES),
        "level_variables": list(LEVEL_VARIABLES),
        "depths": list(PREPPED_DEPTHS),
        "n_dates": 10,
    }
    path = tmp_path_factory.mktemp("stats") / "stats.pt"
    torch.save(stats, path)
    return path


def _year_dataset(year: int, wet_surface: np.ndarray, wet_level: np.ndarray) -> xr.Dataset:
    lat, lon = _grid()
    days = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")).astype(
        "datetime64[ns]"
    )
    if year == 2019:
        days = np.array([d for d in days if d.astype("datetime64[D]") not in MISSING_DAYS])
    # GLORYS daily means are stamped at noon.
    times = days + np.timedelta64(12, "h")
    n_time = len(times)
    rng = np.random.default_rng(year)

    data_vars = {}
    for name in SURFACE_VARIABLES:
        field = rng.normal(size=(n_time, N_LAT, N_LON)).astype("float32")
        field[:, ~wet_surface] = np.nan  # land
        data_vars[name] = (["time", "lat", "lon"], field)

    # Sea ice: defined only where there is ice.  Ice-free ocean is an exact 0
    # before the switch date and NaN after it -- exactly what GLORYS does.
    ice_free = np.zeros((n_time, N_LAT, N_LON), dtype=bool)
    ice_free[:, wet_surface] = rng.random((n_time, int(wet_surface.sum()))) > 0.5
    after_switch = times >= SEAICE_CONVENTION_SWITCH
    for name in ("siconc", "sithick", "usi", "vsi"):
        field = data_vars[name][1]
        field[ice_free] = 0.0
        field[ice_free & after_switch[:, None, None]] = np.nan

    for name in LEVEL_VARIABLES:
        field = rng.normal(size=(n_time, N_DEPTH, N_LAT, N_LON)).astype("float32")
        field[:, ~wet_level] = np.nan
        data_vars[name] = (["time", "depth", "lat", "lon"], field)

    # uo and vo are NaN at a cell the wet mask calls ocean, on every date --
    # the small inconsistency the real archive has at 630 cells per level.
    for name in ("uo", "vo"):
        data_vars[name][1][:, :, 2, 3] = np.nan

    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            time=times,
            lat=lat,
            lon=lon,
            depth=np.array(PREPPED_DEPTHS, dtype="float32"),
        ),
    )


@pytest.fixture(scope="session")
def tiny_archive(tmp_path_factory) -> Path:
    """A directory of ``glorys_1deg_YYYY.nc`` files, same layout as the real one."""
    root = tmp_path_factory.mktemp("glorys_tiny")
    wet_surface, wet_level = _wet_masks()
    for year in ARCHIVE_YEARS:
        _year_dataset(year, wet_surface, wet_level).to_netcdf(root / f"glorys_1deg_{year}.nc")
    # A file the filters must ignore, like the real prep_manifest.json.
    (root / "prep_manifest.json").write_text("{}")
    return root


@pytest.fixture(scope="session")
def tiny_dataset_kwargs(tiny_archive, tiny_masks_file) -> dict:
    """Keyword arguments that point a ``GlorysDataset`` at the miniature archive."""
    return dict(path=tiny_archive, masks_path=tiny_masks_file)


@pytest.fixture(scope="session")
def tiny_forecast_kwargs(tiny_dataset_kwargs, tiny_stats_file) -> dict:
    """Same, plus the statistics a ``GlorysForecast`` needs."""
    return dict(tiny_dataset_kwargs, stats_path=tiny_stats_file)


@pytest.fixture(scope="session")
def real_masks_path() -> Path:
    """The generated mask file, or skip -- it is not in git (`make stats` builds it)."""
    from oceanarches import paths

    if not paths.masks_file().exists():
        pytest.skip(f"{paths.masks_file()} not found; run: make stats")
    return paths.masks_file()


#: Latitude/longitude of the :func:`cropped_masks` fixture.  Both divide the
#: shipped ``patch_size`` of 3, and the latent grid they give (24 x 40) is
#: divisible twice over by the backbone's ``window_size`` of ``[1, 6, 10]``:
#: 24/6 = 4 and 40/10 = 4 at the top stage, 12/6 = 2 and 20/10 = 2 after the
#: downsample.
CROPPED_GRID = (72, 120)


@pytest.fixture(scope="session")
def cropped_masks(real_masks_path, tmp_path_factory) -> tuple[Path, int, int]:
    """The real mask file cropped to a 72x120 corner of the globe.

    Returns ``(path, n_lat, n_lon)``.

    For tests that have to run the *real* backbone but are not about the
    horizontal grid -- the vertical-mixing probe above all.  Cropping rather
    than synthesising keeps the real channel names, dtypes and land/sea
    structure, so the embedder is built from the same artefact it is built from
    in production; only the map is smaller.  The latent *depth*, which is what
    those tests are about, is untouched.
    """
    lat, lon = CROPPED_GRID
    with xr.open_dataset(real_masks_path) as masks:
        cropped = masks.isel(lat=slice(0, lat), lon=slice(0, lon)).load()
    path = tmp_path_factory.mktemp("cropped") / f"masks_{lat}x{lon}.nc"
    cropped.to_netcdf(path)
    return path, lat, lon


@pytest.fixture
def fake_modelstore(tmp_path, monkeypatch):
    """A MODELSTORE holding runs `a` and `b`, each with a config and a checkpoint.

    `run_eval.resolve_run` -- which `--exp` and every `--components NAME=RUN`
    now go through -- checks that a run exists before anything is loaded, so the
    command-line tests need runs that exist.
    """

    def make(*names: str) -> Path:
        root = tmp_path / "modelstore"
        for name in names:
            (root / name / "checkpoints").mkdir(parents=True, exist_ok=True)
            (root / name / "config.yaml").write_text("name: " + name + "\n")
            (root / name / "checkpoints" / "checkpoint_global_step=1.ckpt").write_bytes(b"x")
        monkeypatch.setenv("MODELSTORE", str(root))
        return root

    return make


# ---------------------------------------------------------------------------
# A fresh clone has no generated statistics, and 45 tests used to say so badly.
#
# `oceanarches/stats/*.nc` and `*.pt` are built by `make stats`, not committed
# (.gitignore lines 21-24), so a clone that has run `make setup` but not
# `make stats` has none of them. Measured on that clone: `make test` gave **45
# failed**, every one of them a `FileNotFoundError` for the mask file, raised
# from inside an embedder, a metric or a hydra instantiation and so wearing a
# stack trace that looks like a broken checkout. The documents suggest `make
# test` as a check that the environment is sound, so somebody will run it in
# that order, and 45 red lines is the wrong answer to "did the install work?".
#
# The right answer is a skip that names the command. The tests themselves are
# not weakened to get it: this hook only fires when the artefact is *genuinely
# absent from disk*, checked at the moment of the failure. An artefact that is
# present but wrong -- truncated, stale, built from the wrong years -- raises
# something else, or raises `FileNotFoundError` for a path that does exist, and
# in both cases the failure stands.
#
# `real_masks_path` above (and the `IFS_FORCING` tests) already skip by asking
# first. This is the same policy for the tests that do not ask, applied without
# editing 45 tests across four files.
# ---------------------------------------------------------------------------


def _generated_artefacts() -> tuple[tuple[Path, str], ...]:
    """Every artefact `oceanarches/stats/` holds, and the target that builds it."""
    from oceanarches import paths

    return (
        (paths.masks_file(), "make stats"),
        (paths.stats_file(), "make stats"),
        (paths.climatology_file(), "make stats"),
        (paths.forcing_stats_file(), "make forcing-stats"),
    )


def _missing_generated_artefact(error: BaseException) -> tuple[Path, str] | None:
    """``(path, command)`` if ``error`` is "a generated artefact is not there yet".

    Returns ``None`` for everything else, including a `FileNotFoundError` whose
    file exists by the time we look -- that is a different bug and must stay a
    failure.

    The whole ``__cause__``/``__context__`` chain is searched because the
    interesting ones arrive wrapped: hydra raises `InstantiationException` with
    the real `FileNotFoundError` as its cause.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__

    artefacts = _generated_artefacts()
    for exception in chain:
        if not isinstance(exception, FileNotFoundError):
            continue
        text = f"{exception}\n{getattr(exception, 'filename', '') or ''}"
        for path, command in artefacts:
            # The full path, not the file name: a test that writes its own
            # `glorys_1deg_stats.pt` into `tmp_path` and then deletes it is
            # testing something real, and must not be excused by this.
            if str(path) in text and not path.exists():
                return path, command
    return None


def _shown(path: Path) -> str:
    """The artefact's path relative to the repository, when it is inside it."""
    from oceanarches import paths

    try:
        return str(path.relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(path)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Turn "you never ran `make stats`" into a skip that says so."""
    try:
        return (yield)
    except BaseException as error:
        missing = _missing_generated_artefact(error)
        if missing is None:
            raise
        path, command = missing
        pytest.skip(
            f"{_shown(path)} has not been built yet -- run `{command}` first "
            "(the generated statistics are not in git; `make doctor` says the same)."
        )
