"""Tests for the external (prescribed atmosphere) forcing sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from oceanarches import paths
from oceanarches.dataloaders.forcing import (
    NoForcing,
    PersistenceForcing,
    XarrayForcing,
    to_datetime64,
)
from oceanarches.dataloaders.variables import N_LAT, N_LON

FORCING_VARIABLES = ["sowinu10", "sowinv10", "sotemair"]
FIRST_TIME = np.datetime64("2024-01-02T12:00:00")


def _write_forcing(directory: Path, start: np.datetime64, n_times: int, n_lat=N_LAT, n_lon=N_LON):
    """A file shaped like the shipped IFS forcing (its time axis is `time_counter`)."""
    times = start + np.arange(n_times) * np.timedelta64(1, "D")
    rng = np.random.default_rng(int(start.astype("datetime64[D]").astype(int)))
    ds = xr.Dataset(
        data_vars={
            name: (
                ["time_counter", "lat", "lon"],
                (rng.normal(size=(n_times, n_lat, n_lon)) + i).astype("float32"),
            )
            for i, name in enumerate(FORCING_VARIABLES)
        },
        coords=dict(
            time_counter=times,
            lat=np.linspace(-89.5, 89.5, n_lat),
            lon=np.linspace(0, 359, n_lon),
        ),
    )
    path = directory / f"IFS_{str(start.astype('datetime64[D]')).replace('-', '')}.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def forcing_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("forcing")
    _write_forcing(directory, FIRST_TIME, 5)
    _write_forcing(directory, FIRST_TIME + np.timedelta64(7, "D"), 5)
    return directory


# ---------------------------------------------------------------------------
# NoForcing -- the default, and the path everything must support
# ---------------------------------------------------------------------------
def test_no_forcing_is_empty_and_returns_none():
    forcing = NoForcing()

    assert forcing.n_channels == 0
    assert forcing.variables == []
    # None, not a zero tensor: a caller that forgets to check should fail loudly
    # rather than train on a channel of zeros.
    assert forcing.get(0) is None
    assert forcing.get(np.datetime64("2024-06-01")) is None


def test_no_forcing_concatenation_pattern():
    """The shape every model's input assembly takes."""
    forcing = NoForcing()
    inputs = torch.zeros(7, 1, 4, 5)

    extra = forcing.get(0)
    if extra is not None:  # pragma: no cover - documents the pattern
        inputs = torch.cat([inputs, extra], dim=0)

    assert inputs.shape[0] == 7


# ---------------------------------------------------------------------------
# to_datetime64
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        1704196800,
        torch.tensor(1704196800, dtype=torch.int32),
        np.datetime64("2024-01-02T12:00:00"),
        "2024-01-02T12:00:00",
    ],
)
def test_to_datetime64_accepts_every_form_a_rollout_uses(value):
    assert to_datetime64(value) == FIRST_TIME


def test_to_datetime64_rejects_a_batch_of_timestamps():
    with pytest.raises(ValueError, match="single timestamp"):
        to_datetime64(torch.tensor([1, 2, 3]))


# ---------------------------------------------------------------------------
# XarrayForcing
# ---------------------------------------------------------------------------
def test_xarray_forcing_shape_and_channels(forcing_dir):
    forcing = XarrayForcing(forcing_dir, FORCING_VARIABLES)

    assert forcing.n_channels == 3
    field = forcing.get(FIRST_TIME)
    # Same rank as a surface state, so it concatenates onto one.
    assert field.shape == (3, 1, N_LAT, N_LON)
    assert torch.isfinite(field).all()


def test_xarray_forcing_normalises_with_its_own_statistics(forcing_dir):
    """Forcing is not part of the GLORYS state, so it is not in glorys_1deg_stats.pt."""
    normalised = XarrayForcing(forcing_dir, FORCING_VARIABLES).get(FIRST_TIME)
    raw = XarrayForcing(forcing_dir, FORCING_VARIABLES, normalize=False).get(FIRST_TIME)

    assert abs(float(normalised.mean())) < abs(float(raw.mean()))
    assert float(normalised.std()) == pytest.approx(1.0, abs=0.3)


def test_xarray_forcing_takes_the_nearest_time(forcing_dir):
    forcing = XarrayForcing(forcing_dir, FORCING_VARIABLES, tolerance_hours=12)
    exact = forcing.get(FIRST_TIME + np.timedelta64(1, "D"))
    nearby = forcing.get(FIRST_TIME + np.timedelta64(1, "D") + np.timedelta64(3, "h"))

    assert torch.equal(exact, nearby)


def test_xarray_forcing_refuses_a_time_outside_its_range(forcing_dir):
    """The shipped IFS forcing covers about one year.  A field six months stale
    looks plausible on a plot and quietly ruins a rollout."""
    forcing = XarrayForcing(forcing_dir, FORCING_VARIABLES)

    with pytest.raises(ValueError, match="covers valid times"):
        forcing.get(np.datetime64("1993-06-15T12:00:00"))
    with pytest.raises(ValueError, match="covers valid times"):
        forcing.get(np.datetime64("2030-06-15T12:00:00"))


def test_xarray_forcing_refuses_a_time_inside_its_gaps(forcing_dir):
    """The two files are a week apart; the middle of that week is not covered."""
    forcing = XarrayForcing(forcing_dir, FORCING_VARIABLES, tolerance_hours=12)
    with pytest.raises(ValueError, match="tolerance"):
        forcing.get(FIRST_TIME + np.timedelta64(6, "D"))


def test_xarray_forcing_checks_the_grid(tmp_path):
    _write_forcing(tmp_path, FIRST_TIME, 2, n_lat=90, n_lon=180)
    with pytest.raises(ValueError, match="Regrid"):
        XarrayForcing(tmp_path, FORCING_VARIABLES)


def test_xarray_forcing_reports_a_missing_variable(forcing_dir):
    with pytest.raises(KeyError, match="sofakingwrong"):
        XarrayForcing(forcing_dir, ["sofakingwrong"])


def test_xarray_forcing_reports_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="IFS_FORCING"):
        XarrayForcing(tmp_path / "nowhere", FORCING_VARIABLES)


def test_xarray_forcing_accepts_a_single_file(forcing_dir):
    one = sorted(forcing_dir.glob("*.nc"))[0]
    forcing = XarrayForcing(one, FORCING_VARIABLES)
    assert forcing.get(FIRST_TIME).shape == (3, 1, N_LAT, N_LON)


# ---------------------------------------------------------------------------
# Forecast-shaped files -- what the shipped IFS archive actually is
# ---------------------------------------------------------------------------
#: A file that reproduces every awkward property of the real archive: a
#: `time_counter` that repeats the *initialisation* time, a `leadtime` variable
#: in hours, valid times at odd hours, and two forecasts valid at the same
#: instant at different leads.
#:
#: File A: initialised 2024-01-02T12, leads 13/37/61/85 h
#:         -> valid 01-03T01, 01-04T01, 01-05T01, 01-06T01
#: File B: initialised 2024-01-04T12, leads 13/37/61/85 h
#:         -> valid 01-05T01, 01-06T01, 01-07T01, 01-08T01
#: so 01-05T01 and 01-06T01 each have two forecasts, and the shorter lead (B's
#: 13 h and 37 h) must win.
FORECAST_INITS = {
    "A": (np.datetime64("2024-01-02T12:00:00"), [13, 37, 61, 85]),
    "B": (np.datetime64("2024-01-04T12:00:00"), [13, 37, 61, 85]),
}


def _write_forecast_file(
    directory: Path,
    init: np.datetime64,
    leads_hours: list[int],
    lead_units: str = "hours",
    all_nan: tuple[str, ...] = (),
    flip_lat: bool = False,
) -> Path:
    """One forecast file.  Every cell of a record holds that record's lead time.

    Encoding the lead time in the data is what lets a test say *which* forecast
    was served, rather than merely that something was.  ``lead_units`` is written
    verbatim, including a nonsense one, so the unit handling can be tested.
    """
    n = len(leads_hours)
    fields = {}
    for i, name in enumerate(FORCING_VARIABLES):
        values = np.repeat(np.array(leads_hours, dtype="float32"), N_LAT * N_LON)
        values = values.reshape(n, N_LAT, N_LON) + i
        if name in all_nan:
            values[:] = np.nan
        if flip_lat:
            values = values[:, ::-1]
        fields[name] = (["time_counter", "lat", "lon"], values)
    per_hour = {"hours": 1.0, "seconds": 3600.0, "minutes": 60.0}.get(lead_units, 1.0)
    lat = -89.5 + np.arange(N_LAT, dtype="float64")
    ds = xr.Dataset(
        data_vars=fields,
        coords=dict(
            time_counter=np.repeat(init, n),
            lat=lat[::-1] if flip_lat else lat,
            lon=np.arange(N_LON, dtype="float64"),
        ),
    )
    # Assigned after construction and written with `encoding`, so xarray's
    # timedelta coder never gets to rewrite the units attribute.
    ds["leadtime"] = ("time_counter", np.array(leads_hours, dtype="float64") * per_hour)
    ds["leadtime"].attrs = {"standard_name": "forecast_period", "units": lead_units}
    path = directory / f"IFS_{str(init.astype('datetime64[D]')).replace('-', '')}.nc"
    ds.to_netcdf(path, encoding={"leadtime": {"dtype": "float64"}})
    return path


@pytest.fixture(scope="module")
def forecast_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("ifs_forecasts")
    for init, leads in FORECAST_INITS.values():
        _write_forecast_file(directory, init, leads)
    return directory


def test_the_index_is_built_on_valid_time_not_initialisation_time(forecast_dir):
    """The misdiagnosis this whole path exists to fix.

    Read on `time_counter` alone these two files hold 8 records at 2 distinct
    times, and the archive looks hopelessly sparse.  Read on
    `time_counter + leadtime` they hold 6 distinct valid times, one per day.
    """
    forcing = XarrayForcing(forecast_dir, FORCING_VARIABLES, normalize=False)

    assert forcing.n_records == 8
    assert forcing.n_times == 6
    assert forcing.n_duplicate_valid_times == 2
    assert forcing.time_range == (
        np.datetime64("2024-01-03T01:00:00"),
        np.datetime64("2024-01-08T01:00:00"),
    )
    assert forcing.lead_time_range == (13.0, 85.0)


def test_the_shortest_lead_wins_when_two_forecasts_are_valid_at_the_same_time(forecast_dir):
    """Both files are valid at 2024-01-05T01: A at 61 h, B at 13 h.  B wins."""
    forcing = XarrayForcing(forecast_dir, FORCING_VARIABLES, normalize=False)

    # The data of each record is its own lead time in hours, so the value *is*
    # the answer to "which forecast did you serve?".
    served = forcing.get(np.datetime64("2024-01-05T01:00:00"))
    assert float(served[0].min()) == float(served[0].max()) == 13.0
    served = forcing.get(np.datetime64("2024-01-06T01:00:00"))
    assert float(served[0].min()) == float(served[0].max()) == 37.0
    # ... and a valid time only one forecast reaches still comes from that one.
    served = forcing.get(np.datetime64("2024-01-03T01:00:00"))
    assert float(served[0].min()) == 13.0


def test_noon_requests_land_on_the_same_calendar_day(forecast_dir):
    """GLORYS daily means are stamped 12:00; the archive is valid at 00:00/01:00.

    The pair has to resolve to a field valid on the *requested* calendar day, or
    a forced model is quietly trained on tomorrow's atmosphere.  With the shipped
    12 h tolerance it does -- verified here on the fixture and, over all 366 days
    of the real archive, in the Task 12 report.
    """
    forcing = XarrayForcing(forecast_dir, FORCING_VARIABLES, normalize=False, tolerance_hours=12)

    for day in ("2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"):
        noon = np.datetime64(f"{day}T12:00:00")
        chosen = forcing._times[forcing._nearest(noon)]
        assert chosen.astype("datetime64[D]") == np.datetime64(day)
        assert torch.isfinite(forcing.get(noon)).all()


def test_a_tie_goes_to_the_earlier_valid_time(tmp_path):
    """A noon request is exactly 12 h from midnight on either side.

    `argmin` would break that tie however numpy felt; the rule is "the earlier
    one", which is the field valid on the day being asked about.  151 of the 366
    days in the real archive have only a 00:00 valid time, so this is not a
    corner case.
    """
    _write_forecast_file(tmp_path, np.datetime64("2024-03-01T00:00:00"), [0, 24])
    forcing = XarrayForcing(tmp_path, FORCING_VARIABLES, normalize=False, tolerance_hours=12)

    assert list(forcing._times) == [
        np.datetime64("2024-03-01T00:00:00"),
        np.datetime64("2024-03-02T00:00:00"),
    ]
    chosen = forcing._times[forcing._nearest(np.datetime64("2024-03-01T12:00:00"))]
    assert chosen == np.datetime64("2024-03-01T00:00:00")


def test_a_lead_time_in_seconds_is_read_as_seconds(tmp_path):
    """The unit is read from the file, not assumed.

    The same 13 h lead, written as 46800 seconds.  Code that assumed hours would
    put the field 46800 hours -- five years -- after the initialisation.
    """
    _write_forecast_file(
        tmp_path, np.datetime64("2024-01-02T12:00:00"), [13], lead_units="seconds"
    )
    forcing = XarrayForcing(tmp_path, FORCING_VARIABLES, normalize=False)
    assert forcing.time_range[0] == np.datetime64("2024-01-03T01:00:00")


def test_an_unreadable_lead_time_unit_is_refused(tmp_path):
    """Guessing would silently mis-date every field in the archive."""
    _write_forecast_file(
        tmp_path, np.datetime64("2024-01-02T12:00:00"), [13], lead_units="fortnights"
    )
    with pytest.raises(ValueError, match="fortnights"):
        XarrayForcing(tmp_path, FORCING_VARIABLES, normalize=False)


def test_a_channel_that_is_nan_everywhere_is_refused(tmp_path):
    """MEASURED: `somslpre` is NaN in all 520 records of the shipped archive.

    Left alone it reaches the model as a constant zero -- a channel that looks
    like forcing, costs an embedder channel and carries nothing.
    """
    _write_forecast_file(
        tmp_path,
        np.datetime64("2024-01-02T12:00:00"),
        [13, 37],
        all_nan=("sotemair",),
    )
    with pytest.raises(ValueError, match="sotemair.*NaN over the whole grid"):
        XarrayForcing(tmp_path, FORCING_VARIABLES, normalize=False)


def test_a_channel_that_goes_nan_later_in_the_archive_is_refused(tmp_path):
    """Both ends are probed, not only the first field.

    A second read costs 13 ms on the shipped archive, which is nothing next to
    the 430 ms the index already takes, so there is no reason to check only one
    end.  It is still a probe and not a scan: a channel that is empty only in the
    middle would reach `compute_statistics`, which reads every time.

    MUTANT: probing `{0}` instead of `{0, len - 1}` fails this.
    """
    _write_forecast_file(tmp_path, np.datetime64("2024-01-02T12:00:00"), [13, 37])
    _write_forecast_file(
        tmp_path, np.datetime64("2024-01-09T12:00:00"), [13, 37], all_nan=("sotemair",)
    )

    with pytest.raises(ValueError, match="sotemair.*NaN over the whole grid"):
        XarrayForcing(tmp_path, FORCING_VARIABLES, normalize=False)


def test_an_upside_down_latitude_axis_is_refused(tmp_path):
    """The size check cannot see it, and an upside-down atmosphere trains fine."""
    _write_forecast_file(tmp_path, np.datetime64("2024-01-02T12:00:00"), [13], flip_lat=True)

    with pytest.raises(ValueError, match="same axis reversed"):
        XarrayForcing(tmp_path, FORCING_VARIABLES, normalize=False)


def test_the_lead_time_variable_is_not_offered_as_a_channel(forecast_dir):
    with pytest.raises(ValueError, match="no lat/lon dimensions"):
        XarrayForcing(forecast_dir, ["leadtime"], normalize=False)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_statistics_written_by_the_script_are_read_back_in_channel_order(forecast_dir, tmp_path):
    """`make forcing-stats` writes it, `configs/forcing/file.yaml` points at it.

    The row order in the file is the order it was written in; the *channel* order
    is the order of `variables`.  Selecting rows by name is what keeps a
    reordered variables list from silently normalising wind with temperature.
    """
    source = XarrayForcing(forecast_dir, FORCING_VARIABLES, normalize=False)
    mean, std = source.compute_statistics()
    stats_file = tmp_path / "forcing_stats.pt"
    torch.save({"variables": list(FORCING_VARIABLES), "mean": mean, "std": std}, stats_file)

    reversed_names = list(reversed(FORCING_VARIABLES))
    loaded = XarrayForcing(forecast_dir, reversed_names, stats_path=stats_file)

    assert torch.allclose(loaded.mean.flatten(), mean.flip(0))
    assert loaded.stats_source == str(stats_file)
    # And the whole point of pinning them: normalised output is exactly what the
    # file says, not what a 64-time sample of the archive happened to give.
    field = loaded.get(np.datetime64("2024-01-03T01:00:00"))
    raw = source.get(np.datetime64("2024-01-03T01:00:00")).flip(0)
    assert torch.allclose(
        field, (raw - mean.flip(0).reshape(-1, 1, 1, 1)) / std.flip(0).reshape(-1, 1, 1, 1)
    )


def test_statistics_over_the_whole_archive_match_a_direct_computation(forecast_dir):
    """Streamed in float64 rather than stacked, so check it against the naive sum."""
    source = XarrayForcing(forecast_dir, FORCING_VARIABLES, normalize=False)
    mean, std = source.compute_statistics()

    stack = torch.stack([source._read(i) for i in range(source.n_times)])
    assert torch.allclose(mean, stack.mean(dim=(0, 2, 3, 4)), atol=1e-4)
    assert torch.allclose(std, stack.std(dim=(0, 2, 3, 4), correction=0), atol=1e-4)


def test_missing_statistics_say_which_command_writes_them(forecast_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="make forcing-stats"):
        XarrayForcing(forecast_dir, FORCING_VARIABLES, stats_path=tmp_path / "absent.pt")


def test_statistics_for_a_channel_that_is_not_in_the_file_are_refused(forecast_dir, tmp_path):
    stats_file = tmp_path / "partial.pt"
    torch.save(
        {
            "variables": ["sowinu10"],
            "mean": torch.zeros(1, 1, 1, 1),
            "std": torch.ones(1, 1, 1, 1),
        },
        stats_file,
    )
    with pytest.raises(KeyError, match="make forcing-stats"):
        XarrayForcing(forecast_dir, FORCING_VARIABLES, stats_path=stats_file)


# ---------------------------------------------------------------------------
# Caching -- the same field, however it is fetched
# ---------------------------------------------------------------------------
def test_the_cache_returns_the_same_values_as_reading_every_time(forecast_dir):
    """The cache is an I/O optimisation; it must not be a behaviour change."""
    cached = XarrayForcing(forecast_dir, FORCING_VARIABLES, cache=True)
    uncached = XarrayForcing(forecast_dir, FORCING_VARIABLES, cache=False)
    when = np.datetime64("2024-01-05T01:00:00")

    assert torch.equal(cached.get(when), uncached.get(when))
    assert torch.equal(cached.get(when), cached.get(when))  # second time, from memory


def test_the_cache_is_declined_when_the_archive_is_too_big(forecast_dir):
    forcing = XarrayForcing(forecast_dir, FORCING_VARIABLES, cache=True, cache_max_gb=1e-9)
    assert forcing._cache is None
    assert torch.isfinite(forcing.get(np.datetime64("2024-01-05T01:00:00"))).all()


# ---------------------------------------------------------------------------
# PersistenceForcing
# ---------------------------------------------------------------------------
def test_persistence_forcing_freezes_the_first_field(forcing_dir):
    source = XarrayForcing(forcing_dir, FORCING_VARIABLES)
    frozen = PersistenceForcing(source)

    first = frozen.get(FIRST_TIME)
    later = frozen.get(FIRST_TIME + np.timedelta64(3, "D"))

    assert torch.equal(first, later)
    assert not torch.equal(first, source.get(FIRST_TIME + np.timedelta64(3, "D")))
    assert frozen.frozen_at == FIRST_TIME
    assert frozen.variables == FORCING_VARIABLES


def test_persistence_forcing_can_be_reset(forcing_dir):
    frozen = PersistenceForcing(XarrayForcing(forcing_dir, FORCING_VARIABLES))
    first = frozen.get(FIRST_TIME)
    frozen.reset()
    later = frozen.get(FIRST_TIME + np.timedelta64(3, "D"))

    assert not torch.equal(first, later)


def test_persistence_forcing_over_no_forcing_still_returns_none():
    frozen = PersistenceForcing(NoForcing())
    assert frozen.get(0) is None
    assert frozen.get(1) is None
    assert frozen.n_channels == 0


# ---------------------------------------------------------------------------
# The real IFS forcing, when it is reachable
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_ifs():
    root = paths.ifs_forcing()
    if not root.exists():
        pytest.skip(f"{root} not reachable")
    return XarrayForcing(root, ["sowinu10", "sowinv10"], stats_max_times=4, cache=False)


def test_real_ifs_forcing_loads(real_ifs):
    first, last = real_ifs.time_range

    field = real_ifs.get(first)
    assert field.shape == (2, 1, N_LAT, N_LON)
    assert torch.isfinite(field).all()
    # It only covers about a year, which is why NoForcing is the default.
    assert (last - first) < np.timedelta64(500, "D")


def test_the_real_archive_is_daily_on_its_valid_times(real_ifs):
    """The measurement that overturned "520 steps, 104 distinct times, too sparse".

    Both numbers were right and the conclusion was wrong: `time_counter` is the
    *initialisation* time.  On valid times the same files are daily.
    """
    assert real_ifs.n_records == 520
    assert real_ifs.n_times == 478
    assert real_ifs.n_duplicate_valid_times == 42
    assert real_ifs.time_range == (
        np.datetime64("2024-01-03T01:00:00"),
        np.datetime64("2025-01-02T00:00:00"),
    )
    assert real_ifs.lead_time_range == (13.0, 204.0)

    days = {t.astype("datetime64[D]") for t in real_ifs._times}
    assert len(days) == 366  # 2024-01-03 to 2025-01-02, every calendar day


def test_every_glorys_day_the_archive_covers_resolves_to_that_same_day(real_ifs):
    """GLORYS is stamped 12:00, the archive is valid at 00:00/01:00.

    Every one of the 366 covered days must come back with a field valid on that
    day, and the two days before the first forecast must be refused -- not
    quietly served from 2024-01-03.
    """
    day = np.datetime64("2024-01-01")
    matched, refused = 0, []
    while day <= np.datetime64("2025-01-02"):
        noon = day.astype("datetime64[s]") + np.timedelta64(12, "h")
        try:
            chosen = real_ifs._times[real_ifs._nearest(noon)]
            if abs(chosen - noon) > real_ifs.tolerance:
                raise ValueError
            assert chosen.astype("datetime64[D]") == day, f"{noon} -> {chosen}"
            matched += 1
        except (ValueError, IndexError):
            refused.append(day)
        day += np.timedelta64(1, "D")

    assert matched == 366
    assert refused == [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]


def test_the_real_archive_refuses_a_time_from_the_training_split(real_ifs):
    """`forcing=file` on train/val/test fails at once and says what to do."""
    with pytest.raises(ValueError, match="covers valid times"):
        real_ifs.get(np.datetime64("2015-06-15T12:00:00"))
    with pytest.raises(ValueError, match="NoForcing"):
        real_ifs.get(np.datetime64("2022-06-15T12:00:00"))


def test_the_shipped_forcing_statistics_match_the_shipped_channel_list():
    """`make forcing-stats` and configs/forcing/file.yaml must not drift apart."""
    stats_file = paths.forcing_stats_file()
    if not stats_file.exists():
        pytest.skip(f"{stats_file} not found; run: make forcing-stats")
    import yaml

    config = yaml.safe_load((Path(__file__).parents[1] / "configs/forcing/file.yaml").read_text())
    stats = torch.load(stats_file, weights_only=True)

    assert config["n_channels"] == len(config["source"]["variables"])
    assert set(config["source"]["variables"]) <= set(stats["variables"])
    assert bool((stats["std"] > 0).all())


def test_the_empty_channel_in_the_shipped_archive_is_still_refused():
    """MEASURED: `somslpre` is NaN in all 520 records, so it is not a channel.

    If a future archive fixes it this test fails, which is the right way to find
    out -- and then it can go back into configs/forcing/file.yaml.
    """
    root = paths.ifs_forcing()
    if not root.exists():
        pytest.skip(f"{root} not reachable")
    with pytest.raises(ValueError, match="somslpre.*NaN over the whole grid"):
        XarrayForcing(root, ["somslpre"], normalize=False, cache=False)
