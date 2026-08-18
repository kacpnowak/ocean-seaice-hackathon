"""``make doctor`` must fail on a broken artefact, not only on a missing one.

Until Task 10 the three generated artefacts were checked by
``Path.exists()`` alone, so a mask file on the wrong grid, a statistics file
without ``delta_std`` and an all-NaN climatology all reported PASS -- and then
failed minutes into a training run with a shape error.  These tests exist to
keep the stronger check honest: each one hands ``_inspect_artefact`` a file that
is exactly one thing wrong and asserts it says so.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr

from oceanarches import doctor
from oceanarches.dataloaders.variables import (
    LEVEL_VARIABLES,
    N_LAT,
    N_LON,
    PREPPED_DEPTHS,
    SURFACE_VARIABLES,
)
from oceanarches.doctor import _inspect_artefact, _inspect_forcing_stats


# ---------------------------------------------------------------------------
# It passes the real thing
# ---------------------------------------------------------------------------
def test_the_real_masks_pass(real_masks_path):
    assert _inspect_artefact("masks", real_masks_path) is None


def test_the_real_statistics_pass():
    from oceanarches import paths

    if not paths.stats_file().exists():
        pytest.skip("statistics not generated; run: make stats")
    assert _inspect_artefact("norm stats", paths.stats_file()) is None


# ---------------------------------------------------------------------------
# ... and fails everything that is wrong with one
# ---------------------------------------------------------------------------
def test_a_mask_file_on_the_wrong_grid_is_caught(tiny_masks_file):
    """The `tiny_masks_file` fixture is a 6x8 grid: right structure, wrong size."""
    problem = _inspect_artefact("masks", tiny_masks_file)
    assert problem is not None
    assert "6x8" in problem and f"{N_LAT}x{N_LON}" in problem


def test_a_mask_file_missing_a_variable_is_caught(real_masks_path, tmp_path):
    with xr.open_dataset(real_masks_path) as masks:
        masks.drop_vars("wet_seaice").to_netcdf(tmp_path / "masks.nc")
    problem = _inspect_artefact("masks", tmp_path / "masks.nc")
    assert problem is not None and "wet_seaice" in problem


def test_an_implausible_ocean_fraction_is_caught(real_masks_path, tmp_path):
    with xr.open_dataset(real_masks_path) as masks:
        broken = masks.load()
    broken["wet_surface"][:] = False
    broken.to_netcdf(tmp_path / "masks.nc")
    problem = _inspect_artefact("masks", tmp_path / "masks.nc")
    assert problem is not None and "plausible" in problem


def _stats(n_depth: int = len(PREPPED_DEPTHS)) -> dict:
    n_surface, n_level = len(SURFACE_VARIABLES), len(LEVEL_VARIABLES)
    return {
        "surface_mean": torch.zeros(n_surface, 1, 1, 1),
        "surface_std": torch.ones(n_surface, 1, 1, 1),
        "surface_delta_std": torch.ones(n_surface, 1, 1, 1),
        "level_mean": torch.zeros(n_level, n_depth, 1, 1),
        "level_std": torch.ones(n_level, n_depth, 1, 1),
        "level_delta_std": torch.ones(n_level, n_depth, 1, 1),
        "surface_variables": list(SURFACE_VARIABLES),
        "level_variables": list(LEVEL_VARIABLES),
        "depths": list(PREPPED_DEPTHS)[:n_depth],
    }


def test_statistics_without_delta_std_are_caught(tmp_path):
    stats = _stats()
    del stats["level_delta_std"]
    torch.save(stats, tmp_path / "stats.pt")
    problem = _inspect_artefact("norm stats", tmp_path / "stats.pt")
    assert problem is not None and "level_delta_std" in problem


def test_statistics_for_the_wrong_variable_list_are_caught(tmp_path):
    stats = _stats()
    stats["surface_variables"] = list(SURFACE_VARIABLES)[:-1] + ["not_a_variable"]
    torch.save(stats, tmp_path / "stats.pt")
    problem = _inspect_artefact("norm stats", tmp_path / "stats.pt")
    assert problem is not None and "surface_variables" in problem


def test_statistics_with_the_wrong_depth_count_are_caught(tmp_path):
    stats = _stats()
    stats["level_std"] = stats["level_std"][:, :3]
    torch.save(stats, tmp_path / "stats.pt")
    problem = _inspect_artefact("norm stats", tmp_path / "stats.pt")
    assert problem is not None and "level_std" in problem


def test_a_zero_standard_deviation_is_caught(tmp_path):
    """It would divide by zero at the first batch, and nothing else notices."""
    stats = _stats()
    stats["surface_std"][0] = 0.0
    torch.save(stats, tmp_path / "stats.pt")
    problem = _inspect_artefact("norm stats", tmp_path / "stats.pt")
    assert problem is not None and "standard deviation" in problem


def test_an_unreadable_file_is_reported_not_raised(tmp_path):
    (tmp_path / "masks.nc").write_bytes(b"this is not a netCDF file")
    problem = _inspect_artefact("masks", tmp_path / "masks.nc")
    assert problem is not None and "could not be read" in problem


def _climatology(months: int = 12, all_nan: bool = False) -> xr.Dataset:
    values = np.full((months, 4, 6), np.nan if all_nan else 1.0, dtype="float32")
    return xr.Dataset(
        {name: (["month", "lat", "lon"], values.copy()) for name in SURFACE_VARIABLES},
        coords=dict(month=np.arange(1, months + 1), lat=np.arange(4.0), lon=np.arange(6.0)),
    )


def test_a_climatology_with_the_wrong_number_of_months_is_caught(tmp_path):
    _climatology(months=11).to_netcdf(tmp_path / "clim.nc")
    problem = _inspect_artefact("climatology", tmp_path / "clim.nc")
    assert problem is not None and "month axis" in problem


def test_an_all_nan_climatology_is_caught(tmp_path):
    _climatology(all_nan=True).to_netcdf(tmp_path / "clim.nc")
    problem = _inspect_artefact("climatology", tmp_path / "clim.nc")
    assert problem is not None and "all-NaN" in problem


def test_a_well_formed_climatology_passes(tmp_path):
    _climatology().to_netcdf(tmp_path / "clim.nc")
    assert _inspect_artefact("climatology", tmp_path / "clim.nc") is None


# ---------------------------------------------------------------------------
# What `_check_data` says about a partially prepared archive
# ---------------------------------------------------------------------------
def test_doctor_warns_when_the_prepared_years_do_not_cover_a_split(tmp_path, monkeypatch):
    """Counting files is not the same as covering a split.

    doctor's own fix line suggests `make prep-data YEARS="2015 2016"`. Two years
    passed the old file-count check as PASS, and `make eval` -- which scores on
    `test` = 2021-2023 -- then died inside geoarches.

    MUTANT: making `_check_data` add PASS unconditionally when `files` is
    non-empty (the behaviour this replaced) fails both assertions below.
    """
    from oceanarches import doctor

    prepped = tmp_path / "prepped"
    prepped.mkdir()
    for year in (2015, 2016):
        (prepped / f"glorys_1deg_{year}.nc").write_bytes(b"x")
    monkeypatch.setattr(doctor.paths, "glorys_prepped", lambda: prepped)
    monkeypatch.setattr(doctor.paths, "glorys_raw", lambda: tmp_path / "raw")

    report = doctor.Report()
    doctor._check_data(report)
    rows = {name: (status, detail) for status, name, detail, _ in report.rows}
    status, detail = rows["prepared data"]
    assert status == doctor.WARN
    for split in ("train", "val", "test", "holdout"):
        assert split in detail
    assert "2021-2023" in detail


def test_doctor_passes_a_fully_prepared_archive(tmp_path, monkeypatch):
    """The complement: 1993-2025 covers every split, so it must not warn."""
    from oceanarches import doctor

    prepped = tmp_path / "prepped"
    prepped.mkdir()
    for year in range(1993, 2026):
        (prepped / f"glorys_1deg_{year}.nc").write_bytes(b"x")
    monkeypatch.setattr(doctor.paths, "glorys_prepped", lambda: prepped)
    monkeypatch.setattr(doctor.paths, "glorys_raw", lambda: tmp_path / "raw")

    report = doctor.Report()
    doctor._check_data(report)
    rows = {name: status for status, name, _, _ in report.rows}
    assert rows["prepared data"] == doctor.PASS


def test_a_small_artefact_is_reported_in_kilobytes(tmp_path):
    """`glorys_1deg_stats.pt` is a healthy 3981 bytes and read as `0.0 MB`.

    The one tool whose job is to build confidence in the generated artefacts
    reported the good one as empty.

    MUTANT: going back to `f"{n_bytes / 1e6:.1f} MB"` makes the first assertion
    read "0.0 MB".
    """
    from oceanarches.doctor import _human_size

    assert _human_size(3981) == "4.0 KB"
    assert _human_size(2_866_736) == "2.9 MB"
    assert _human_size(95_401_764) == "95.4 MB"


# ---------------------------------------------------------------------------
# The forcing statistics -- optional, so a WARN and never a FAIL
# ---------------------------------------------------------------------------
def test_the_real_forcing_statistics_pass():
    from oceanarches import paths

    if not paths.forcing_stats_file().exists():
        pytest.skip("forcing statistics not generated; run: make forcing-stats")
    assert _inspect_forcing_stats(paths.forcing_stats_file()) is None


def test_forcing_statistics_missing_a_key_are_caught(tmp_path):
    path = tmp_path / "forcing.pt"
    torch.save({"variables": ["sowinu10"], "mean": torch.zeros(1, 1, 1, 1)}, path)
    assert "missing ['std']" in _inspect_forcing_stats(path)


def test_forcing_statistics_with_the_wrong_row_count_are_caught(tmp_path):
    path = tmp_path / "forcing.pt"
    torch.save(
        {
            "variables": ["sowinu10", "sowinv10"],
            "mean": torch.zeros(1, 1, 1, 1),
            "std": torch.ones(1, 1, 1, 1),
        },
        path,
    )
    assert "1 rows for 2 variables" in _inspect_forcing_stats(path)


def test_a_zero_forcing_standard_deviation_is_caught(tmp_path):
    path = tmp_path / "forcing.pt"
    torch.save(
        {
            "variables": ["sowinu10"],
            "mean": torch.zeros(1, 1, 1, 1),
            "std": torch.zeros(1, 1, 1, 1),
        },
        path,
    )
    assert "not positive" in _inspect_forcing_stats(path)


def test_an_unreadable_forcing_stats_file_is_reported_not_raised(tmp_path):
    path = tmp_path / "forcing.pt"
    path.write_bytes(b"not a torch file")
    assert "could not be read" in _inspect_forcing_stats(path)


def test_a_missing_forcing_archive_warns_rather_than_fails(tmp_path, monkeypatch):
    """`forcing=none` is the default; a clone without the IFS archive is fine."""
    from oceanarches import doctor, paths

    monkeypatch.setenv("IFS_FORCING", str(tmp_path / "absent"))
    paths._config_env.cache_clear()
    report = doctor.Report()
    doctor._check_forcing(report)
    paths._config_env.cache_clear()

    statuses = {name: status for status, name, *_ in report.rows}
    assert statuses["IFS forcing"] == doctor.WARN
    assert not any(status == doctor.FAIL for status, *_ in report.rows)


# ---------------------------------------------------------------------------
# config.env is written for bash, and every line in it carries a comment
# ---------------------------------------------------------------------------
def test_a_trailing_comment_is_not_part_of_the_value(tmp_path, monkeypatch):
    """Measured on the shipped file itself: `SLURM_ACCOUNT="training2635"  # ...`
    read back as `training2635"    # JUPITER was: hclimrep`.

    Nothing reads those two keys through this module, so it stayed invisible --
    but the same annotation on a path (`GLORYS_PREPPED="/my/copy"  # mine`) hands
    back a directory nobody typed, and the error names that directory rather than
    the comment that created it.

    MUTANT: dropping the `_value()` call from `_config_env` fails every case here.
    """
    from oceanarches import paths

    env = tmp_path / "config.env"
    env.write_text(
        'GLORYS_PREPPED="/data/prepped"   # the copied 92 GB\n'
        "SLURM_ACCOUNT=training2635       # unquoted, with a comment\n"
        'SLURM_PARTITION="dc-gpu"\n'
        "EVALSTORE=evalstore\n"
        "# a whole-line comment\n"
        'ODD_PATH="/data/a#b"             # a hash inside quotes is literal\n'
        "BARE_HASH=/data/c#d              # and outside them, with no space, too\n"
    )
    monkeypatch.setattr(paths, "CONFIG_ENV", env)
    paths._config_env.cache_clear()
    try:
        assert paths.setting("GLORYS_PREPPED") == "/data/prepped"
        assert paths.setting("SLURM_ACCOUNT") == "training2635"
        assert paths.setting("SLURM_PARTITION") == "dc-gpu"
        assert paths.setting("EVALSTORE") == "evalstore"
        assert paths.setting("ODD_PATH") == "/data/a#b"
        assert paths.setting("BARE_HASH") == "/data/c#d"
    finally:
        paths._config_env.cache_clear()


# ---------------------------------------------------------------------------
# $HOME's inode quota, which is what actually stops `make setup` on JURECA
# ---------------------------------------------------------------------------
def test_the_inode_probe_counts_what_it_could_create(tmp_path, monkeypatch):
    """No portable way to READ an inode quota exists here -- `quota`, `lfs` and
    `jutil ... quota` are all absent on the JURECA login nodes and `df` reports the
    filesystem's terabytes rather than the user's file count -- so doctor measures
    it by creating files until it cannot.

    On a directory with no quota it must return the ceiling and clean up after
    itself.
    """
    probe = tmp_path / "probe"
    assert doctor._home_inodes_left(probe, ceiling=25) == 25
    assert not probe.exists(), "the probe must not leave its files behind"


def test_a_nearly_full_home_fails_the_doctor_and_says_inodes(monkeypatch):
    """Measured on JURECA: ~2050 files total, ~480 already used, and `make setup`
    died unpacking CPython with `Disk quota exceeded (os error 122)` -- a message
    that names neither $HOME nor inodes.

    MUTANT: reporting PASS regardless of the headroom fails this; so does dropping
    the word "INODE" from the message, which is the only part that tells a
    participant why `df` disagrees.
    """
    monkeypatch.setattr(doctor, "_home_inodes_left", lambda *a, **k: 3)
    report = doctor.Report()
    doctor._check_home_quota(report)
    (status, name, detail, fix) = report.rows[0][:4]
    assert status == doctor.FAIL
    assert "INODE" in detail and "df" in detail
    assert "$HOME" in fix or "cache" in fix.lower()


def test_a_roomy_home_passes(monkeypatch):
    monkeypatch.setattr(doctor, "_home_inodes_left", lambda *a, **k: 600)
    report = doctor.Report()
    doctor._check_home_quota(report)
    assert report.rows[0][0] == doctor.PASS


def test_an_unprobeable_home_warns_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(doctor, "_home_inodes_left", lambda *a, **k: None)
    report = doctor.Report()
    doctor._check_home_quota(report)
    assert report.rows[0][0] == doctor.WARN
