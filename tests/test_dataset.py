"""Tests for ``GlorysDataset`` / ``GlorysForecast``.

Everything here runs against the miniature archive built in ``conftest.py``,
which reproduces the real archive's two traps: a two-day gap in one year, and
sea-ice fields that change convention partway through.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import (
    ARCHIVE_YEARS,
    MISSING_DAYS,
    N_DEPTH,
    N_LAT,
    N_LON,
    SEAICE_CONVENTION_SWITCH,
)
from geoarches.main_hydra import collate_fn

from oceanarches.dataloaders.glorys import (
    SPLIT_YEARS,
    GlorysDataset,
    GlorysForecast,
    filename_filters,
    split_bounds,
)
from oceanarches.dataloaders.variables import (
    LEVEL_VARIABLES,
    NAN_MEANS_ZERO,
    PREPPED_DEPTHS,
    SURFACE_VARIABLES,
    get_component,
)

DAY = np.timedelta64(1, "D")


# ---------------------------------------------------------------------------
# Splits: filters and bounds.  Pure functions, no I/O.
# ---------------------------------------------------------------------------
def test_filename_filters_admit_the_adjacent_years():
    """A split must read one year either side, or the state on 1 January has no
    previous state.  ``era5.filename_filters`` does exactly the same."""
    keep = filename_filters["val"]  # val is 2019-2020
    assert [keep(f"glorys_1deg_{y}.nc") for y in range(2017, 2023)] == [
        False,  # 2017
        True,  # 2018 -- for the previous state of 2019-01-01
        True,  # 2019
        True,  # 2020
        True,  # 2021 -- symmetric, narrowed away by the bounds
        False,  # 2022
    ]


def test_filename_filters_ignore_non_data_files():
    for name in ("prep_manifest.json", "README.md", "glorys_1deg.nc"):
        assert not filename_filters["train"](name)
        assert not filename_filters["all"](name)


def test_every_split_has_a_filter():
    assert set(SPLIT_YEARS) <= set(filename_filters)


@pytest.mark.parametrize(
    "domain,expected", [("train", (1993, 2018)), ("val", (2019, 2020)), ("test", (2021, 2023))]
)
def test_split_bounds_cover_exactly_the_split(domain, expected):
    low, high = split_bounds(domain, load_prev=True, lead_time_hours=24)
    first, last = expected

    # `low` reaches one day back, so the first state of the split has a previous
    # state to read -- an input from the earlier split, never a target.
    assert low == np.datetime64(f"{first}-01-01T00:00:00") - np.timedelta64(24, "h")
    assert high == np.datetime64(f"{last + 1}-01-01T00:00:00")


def test_split_bounds_do_not_overlap():
    """The leakage test: no target of one split may fall inside another."""
    ordered = ["train", "val", "test", "holdout"]
    for earlier, later in zip(ordered, ordered[1:]):
        _, earlier_high = split_bounds(earlier, True, 24)
        later_first = np.datetime64(f"{SPLIT_YEARS[later][0]}-01-01T00:00:00")
        assert earlier_high <= later_first


# ---------------------------------------------------------------------------
# GlorysDataset: shapes, orientation, the sea-ice fill
# ---------------------------------------------------------------------------
def test_dataset_shapes_and_orientation(tiny_dataset_kwargs):
    dataset = GlorysDataset(domain="val", **tiny_dataset_kwargs)
    state = dataset[0]

    assert state["surface"].shape == (len(SURFACE_VARIABLES), 1, N_LAT, N_LON)
    assert state["level"].shape == (len(LEVEL_VARIABLES), N_DEPTH, N_LAT, N_LON)
    # South first, 0 degrees east first -- GLORYS' native orientation, kept on
    # purpose (Era5Dataset flips and rolls; we do not).
    assert dataset.lat[0] < dataset.lat[-1]
    assert dataset.lon[0] == 0.0


def test_dataset_keeps_land_as_nan(tiny_dataset_kwargs):
    """The un-normalised dataset is the one you look at, so land stays visible."""
    dataset = GlorysDataset(domain="val", **tiny_dataset_kwargs)
    state = dataset[0]
    masks = dataset.masks

    assert state["surface"][0, 0][~masks.wet_surface].isnan().all()
    assert not state["surface"][0, 0][masks.wet_surface].isnan().any()


def test_seaice_fill_works_on_both_sides_of_the_convention_change(tiny_dataset_kwargs):
    """GLORYS wrote 0 over ice-free ocean, then NaN.  After the fill, both are 0."""
    dataset = GlorysDataset(domain="val", **tiny_dataset_kwargs)
    times = np.array([t for _, _, t in dataset.timestamps], dtype="datetime64[s]")
    before = int(np.argmax(times >= SEAICE_CONVENTION_SWITCH - 10 * DAY))
    after = int(np.argmax(times >= SEAICE_CONVENTION_SWITCH + 10 * DAY))
    ocean = dataset.masks.wet_surface

    for index in (before, after):
        surface = dataset[index]["surface"]
        for name in NAN_MEANS_ZERO:
            channel = surface[SURFACE_VARIABLES.index(name), 0]
            assert not channel[ocean].isnan().any(), f"{name} still NaN over ocean"
            assert (channel[ocean] == 0.0).any(), f"{name} has no ice-free cells"


def test_seaice_fill_can_be_switched_off(tiny_dataset_kwargs):
    """So that you can see, in a notebook, what the raw data actually looks like."""
    dataset = GlorysDataset(domain="val", fill_seaice=False, **tiny_dataset_kwargs)
    times = np.array([t for _, _, t in dataset.timestamps], dtype="datetime64[s]")
    after = int(np.argmax(times >= SEAICE_CONVENTION_SWITCH + 10 * DAY))

    surface = dataset[after]["surface"]
    channel = surface[SURFACE_VARIABLES.index("siconc"), 0]
    assert channel[dataset.masks.wet_surface].isnan().any()


def test_depth_indices_select_a_model_preset(tiny_dataset_kwargs):
    dataset = GlorysDataset(domain="val", depth_indices=[0, 2, 4], **tiny_dataset_kwargs)

    assert dataset[0]["level"].shape == (len(LEVEL_VARIABLES), 3, N_LAT, N_LON)
    assert dataset.depths == [PREPPED_DEPTHS[i] for i in (0, 2, 4)]
    assert dataset.masks.wet_level.shape[0] == 3


def test_both_depth_selection_paths_agree(tiny_dataset_kwargs):
    """`depth_select="tensor"` is the default because it measures 3.6x faster on
    the real files; it must return exactly what `"xarray"` returns."""
    indices = [0, 2, 5, 9]
    in_tensor = GlorysDataset(domain="val", depth_indices=indices, **tiny_dataset_kwargs)
    in_xarray = GlorysDataset(
        domain="val", depth_indices=indices, depth_select="xarray", **tiny_dataset_kwargs
    )

    left, right = in_tensor[3]["level"], in_xarray[3]["level"]
    assert left.shape == (len(LEVEL_VARIABLES), len(indices), N_LAT, N_LON)
    assert torch.equal(left.nan_to_num(-1), right.nan_to_num(-1))


def test_depth_select_rejects_an_unknown_mode(tiny_dataset_kwargs):
    with pytest.raises(ValueError, match="depth_select"):
        GlorysDataset(domain="val", depth_select="magic", **tiny_dataset_kwargs)


def test_variable_subset_matches_a_component(tiny_dataset_kwargs):
    dataset = GlorysDataset(
        domain="val",
        variables=dict(surface=["siconc", "sithick"], level=["thetao"]),
        **tiny_dataset_kwargs,
    )
    state = dataset[0]
    assert state["surface"].shape[0] == 2
    assert state["level"].shape[0] == 1


def test_unknown_domain_lists_the_options(tiny_dataset_kwargs):
    with pytest.raises(KeyError, match="Unknown domain"):
        GlorysDataset(domain="nineteen_eighty_four", **tiny_dataset_kwargs)


# ---------------------------------------------------------------------------
# GlorysForecast: the sample tuple
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def forecast(tiny_forecast_kwargs):
    return GlorysForecast(domain="val", **tiny_forecast_kwargs)


def test_getitem_returns_the_geoarches_contract(forecast):
    sample = forecast[0]

    assert set(sample) == {"state", "prev_state", "next_state", "timestamp", "lead_time_hours"}
    assert sample["state"]["surface"].shape == (len(SURFACE_VARIABLES), 1, N_LAT, N_LON)
    assert sample["state"]["level"].shape == (len(LEVEL_VARIABLES), N_DEPTH, N_LAT, N_LON)
    assert sample["timestamp"].dtype is torch.int32
    assert int(sample["lead_time_hours"]) == 24


def test_multistep_adds_future_states(tiny_forecast_kwargs):
    dataset = GlorysForecast(domain="val", multistep=3, **tiny_forecast_kwargs)
    sample = dataset[0]

    assert sample["future_states"].shape[0] == 3
    assert int(sample["lead_time_hours"]) == 72
    assert torch.equal(sample["future_states"][0]["surface"], sample["next_state"]["surface"])


def test_multistep_setter_rebuilds_the_sample_index(tiny_forecast_kwargs):
    """geoarches raises ``dataset.multistep`` between epochs; longer rollouts run
    off the end of the split, so fewer samples survive."""
    dataset = GlorysForecast(domain="val", multistep=1, **tiny_forecast_kwargs)
    before = len(dataset)
    assert (dataset.n_dropped_at_edges, dataset.n_dropped_by_time_check) == (2, 4)

    dataset.multistep = 4

    assert dataset.multistep == 4
    assert dataset[0]["future_states"].shape[0] == 4
    # Four future states instead of one: three more samples run off the end of
    # the split, and the reach of each missing day grows from two neighbours to
    # five (four before it, one after).
    assert dataset.n_dropped_at_edges == 5
    assert dataset.n_dropped_by_time_check == 8  # 10 minus the one they share
    assert len(dataset) == before - 7


def test_no_nan_or_inf_reaches_the_model(forecast):
    """Sampled across the year, including the gap and the convention change."""
    indices = np.linspace(0, len(forecast) - 1, 40).astype(int)
    for index in indices:
        for key, value in forecast[int(index)].items():
            if "state" not in key:
                continue
            for name, tensor in value.items():
                assert torch.isfinite(tensor).all(), f"{key}/{name} at sample {index}"


# ---------------------------------------------------------------------------
# The 2003 gap
# ---------------------------------------------------------------------------
def test_every_sample_has_exactly_the_advertised_spacing(forecast):
    """The whole point of the timestamp validation.

    ``Era5Forecast`` steps by index, so across a missing day it silently hands
    back a state two days later and still calls it a one-day forecast.
    """
    times = np.array([t for _, _, t in forecast.timestamps], dtype="datetime64[s]")
    for sample_index, position in enumerate(forecast.valid_ids):
        assert times[position] - times[position - 1] == DAY, sample_index
        assert times[position + 1] - times[position] == DAY, sample_index


def test_the_gap_drops_the_samples_around_it(tiny_forecast_kwargs):
    dataset = GlorysForecast(domain="val", **tiny_forecast_kwargs)
    states = np.array([dataset.timestamps[p][2] for p in dataset.valid_ids], dtype="datetime64[D]")

    # Each missing day makes two neighbours unusable: the day before (its next
    # state is two days away) and the day after (its previous state is).
    for missing in MISSING_DAYS:
        assert missing not in states
        assert missing - DAY not in states
        assert missing + DAY not in states

    assert dataset.n_dropped_by_time_check == 2 * len(MISSING_DAYS)
    # Only the two ends of the split are lost to the edges.
    assert dataset.n_dropped_at_edges == 2


def test_gap_handling_scales_with_the_lead_time(tiny_forecast_kwargs):
    """A two-day lead time straddles the gap from further away."""
    dataset = GlorysForecast(domain="val", lead_time_hours=48, **tiny_forecast_kwargs)
    times = np.array([t for _, _, t in dataset.timestamps], dtype="datetime64[s]")

    for position in dataset.valid_ids:
        assert times[position] - times[position - 2] == 2 * DAY
        assert times[position + 2] - times[position] == 2 * DAY


# ---------------------------------------------------------------------------
# Splits do not leak
# ---------------------------------------------------------------------------
def test_split_timestamp_ranges_do_not_overlap(tiny_forecast_kwargs):
    ranges = {}
    for domain in ("train", "val", "test"):
        dataset = GlorysForecast(domain=domain, multistep=2, **tiny_forecast_kwargs)
        ranges[domain] = (dataset.state_timestamp_range(), dataset.target_timestamp_range())

    for domain, (states, targets) in ranges.items():
        first, last = SPLIT_YEARS[domain]
        split_start = np.datetime64(f"{first}-01-01T00:00:00")
        split_end = np.datetime64(f"{last + 1}-01-01T00:00:00")
        assert split_start <= states[0] and states[1] < split_end
        # The one that matters: no target may sit in a later split.
        assert split_start <= targets[0] and targets[1] < split_end

    assert ranges["train"][1][1] < ranges["val"][0][0]
    assert ranges["val"][1][1] < ranges["test"][0][0]


def test_previous_state_may_come_from_the_year_before(tiny_forecast_kwargs):
    """Reading an *input* from the preceding split is fine and is why the
    filename filter admits the adjacent years."""
    dataset = GlorysForecast(domain="val", **tiny_forecast_kwargs)
    first_state, _ = dataset.state_timestamp_range()

    assert first_state == np.datetime64("2019-01-01T12:00:00")
    assert len({f.split("_")[-1] for f in dataset.files}) > 1


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_denormalize_inverts_normalize_over_ocean(forecast):
    raw = forecast.__getitem__(0, normalize=False)
    mask = forecast.state_mask()

    restored = forecast.denormalize(forecast.normalize({"state": raw["state"]}))["state"]

    for key in ("surface", "level"):
        ocean = mask[key].bool().expand_as(raw["state"][key])
        original = raw["state"][key][ocean]
        # uo/vo are NaN at cells the mask calls ocean; compare where both are finite.
        finite = original.isfinite()
        difference = (original[finite] - restored[key][ocean][finite]).abs()
        scale = original[finite].abs().clamp(min=1.0)
        assert float((difference / scale).max()) < 1e-5


def _sample_index_for(dataset, date: str) -> int:
    """Sample index whose *state* falls on `date`."""
    states = np.array([dataset.timestamps[p][2] for p in dataset.valid_ids], dtype="datetime64[D]")
    hits = np.flatnonzero(states == np.datetime64(date))
    assert len(hits), f"{date} is not a usable state in this split"
    return int(hits[0])


@pytest.mark.parametrize(
    "date,convention",
    [
        # GLORYS writes an exact 0 over ice-free ocean before the switch...
        ("2019-06-01", "zero"),
        # ...and NaN after it.  Only this second case can tell the two orderings
        # apart: a cell that is already 0 on disk normalises to -mean/std either
        # way round, so a test that looks only at those cells cannot fail.
        ("2019-08-01", "nan"),
    ],
)
def test_the_seaice_fill_happens_before_normalisation(tiny_forecast_kwargs, date, convention):
    """Getting steps 2 and 3 the wrong way round is silent, so pin it down.

    Correct order: fill to a raw 0, then normalise -> the cell holds ``-mean/std``.
    Reversed order: normalise the NaN (still NaN), then fill -> the cell holds 0,
    i.e. a claim that ice-free ocean sits at the climatological mean.
    """
    dataset = GlorysForecast(domain="val", **tiny_forecast_kwargs)
    # A twin that skips step 2, so we can see exactly what is on disk.
    on_disk = GlorysForecast(domain="val", fill_seaice=False, **tiny_forecast_kwargs)
    index = _sample_index_for(dataset, date)

    channel = SURFACE_VARIABLES.index("siconc")
    mean = float(dataset.data_mean["surface"][channel])
    std = float(dataset.data_std["surface"][channel])
    expected = -mean / std
    assert abs(expected) > 1e-2, "statistics too close to zero for this test to bite"

    raw = on_disk.__getitem__(index, normalize=False)["state"]["surface"][channel, 0]
    normalised = dataset[index]["state"]["surface"][channel, 0]

    ocean = dataset.masks.wet_surface
    ice_free = (raw.isnan() if convention == "nan" else raw == 0.0) & ocean
    assert ice_free.any(), f"no ice-free ocean cell written as {convention} on {date}"

    assert torch.allclose(
        normalised[ice_free], torch.full_like(normalised[ice_free], expected), atol=1e-5
    )


def test_land_normalises_to_zero_not_to_minus_mean_over_std(forecast):
    """Step 4 is ``nan_to_num`` *after* normalisation, so land sits at the
    climatological mean -- the least disruptive value to feed a network."""
    surface = forecast[0]["state"]["surface"]
    land = ~forecast.masks.wet_surface
    assert (surface[:, 0, land] == 0.0).all()


def test_missing_statistics_say_how_to_build_them(tiny_dataset_kwargs, tmp_path):
    with pytest.raises(FileNotFoundError, match="make stats"):
        GlorysForecast(domain="val", stats_path=tmp_path / "nope.pt", **tiny_dataset_kwargs)


def test_statistics_follow_the_variable_and_depth_selection(tiny_forecast_kwargs):
    dataset = GlorysForecast(
        domain="val",
        variables=dict(surface=["siconc"], level=["thetao", "so"]),
        depth_indices=[0, 5],
        **tiny_forecast_kwargs,
    )
    assert dataset.data_mean["surface"].shape == (1, 1, 1, 1)
    assert dataset.data_mean["level"].shape == (2, 2, 1, 1)
    assert dataset.delta_std["level"].shape == (2, 2, 1, 1)


# ---------------------------------------------------------------------------
# The geoarches contract
# ---------------------------------------------------------------------------
def test_collate_fn_produces_what_geoarches_expects(tiny_forecast_kwargs):
    dataset = GlorysForecast(domain="val", multistep=2, **tiny_forecast_kwargs)
    batch = collate_fn([dataset[0], dataset[1], dataset[2]])

    assert batch["state"]["surface"].shape == (3, len(SURFACE_VARIABLES), 1, N_LAT, N_LON)
    assert batch["future_states"]["level"].shape == (
        3,
        2,
        len(LEVEL_VARIABLES),
        N_DEPTH,
        N_LAT,
        N_LON,
    )
    assert batch["timestamp"].shape == (3,)
    # geoarches' metrics compare `batch["state"][:, None]` with a rollout.
    assert batch["state"][:, None]["surface"].shape == (
        3,
        1,
        len(SURFACE_VARIABLES),
        1,
        N_LAT,
        N_LON,
    )


def test_dataloader_with_workers_returns_the_same_thing(tiny_forecast_kwargs):
    """ "The same thing" as what?  As the dataset read in this process.

    The test used to check only that a worker returned *something* finite of the
    right shape, which is true of any two random tensors -- it would have passed
    against workers that silently served the wrong sample.  The comparison is
    the point: forking has to preserve the file handles, the masks, the
    statistics and the sample index, and a mismatch there is invisible in
    training (the loss curve looks normal) and fatal to every number afterwards.
    """
    dataset = GlorysForecast(domain="val", **tiny_forecast_kwargs)
    expected = collate_fn([dataset[0], dataset[1]])

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=2, shuffle=False, collate_fn=collate_fn
    )
    batch = next(iter(loader))

    assert batch["state"]["surface"].shape[0] == 2
    assert torch.equal(batch["timestamp"], expected["timestamp"])
    for key in ("state", "prev_state", "next_state"):
        for group in ("surface", "level"):
            assert torch.equal(batch[key][group], expected[key][group]), (
                f"{key}/{group} differs between a 2-worker loader and this process"
            )
    assert torch.isfinite(batch["state"]["level"]).all()


def test_geoarches_attributes_exist(forecast):
    for name in (
        "domain",
        "multistep",
        "normalize",
        "denormalize",
        "iteration_hook",
        "convert_to_xarray",
        "convert_trajectory_to_xarray",
        "set_timestamp_bounds",
        "lead_time_hours",
    ):
        assert hasattr(forecast, name), name
    forecast.iteration_hook(model=None)  # must be a no-op, not a crash


# ---------------------------------------------------------------------------
# xarray output
# ---------------------------------------------------------------------------
def test_convert_to_xarray_has_real_coordinates_and_land(forecast):
    sample = forecast[0]
    dataset = forecast.convert_to_xarray(
        forecast.denormalize(sample["state"]), sample["timestamp"]
    )

    assert set(dataset.data_vars) == set(SURFACE_VARIABLES) | set(LEVEL_VARIABLES)
    assert dataset.sizes == {"time": 1, "depth": N_DEPTH, "lat": N_LAT, "lon": N_LON}
    assert dataset.lat.values[0] < dataset.lat.values[-1]
    assert dataset.zos.attrs["units"] == "m"
    assert np.datetime64(dataset.time.values[0]) == np.datetime64(int(sample["timestamp"]), "s")
    # Land written back as NaN: on disk, no data should look like no data.
    land = ~forecast.masks.wet_surface.numpy()
    assert np.isnan(dataset.zos.values[0][land]).all()
    assert not np.isnan(dataset.zos.values[0][~land]).any()

    # And the same for a *level* variable, where the coastline moves with depth:
    # using the surface mask at 1684 m would call the whole continental shelf
    # ocean, so this is the case that actually needs checking.
    wet_level = forecast.masks.wet_level.numpy()
    for depth_index in (0, wet_level.shape[0] // 2, wet_level.shape[0] - 1):
        dry = ~wet_level[depth_index]
        values = dataset.thetao.values[0, depth_index]
        assert np.isnan(values[dry]).all(), f"land is not NaN at depth index {depth_index}"
        assert not np.isnan(values[~dry]).any(), f"ocean is NaN at depth index {depth_index}"
    assert (wet_level[0] != wet_level[-1]).any(), (
        "the fixture's coastline does not move with depth, so this test would not "
        "distinguish the per-depth mask from the surface one"
    )


def test_convert_trajectory_to_xarray_labels_the_lead_times(tiny_forecast_kwargs):
    dataset = GlorysForecast(domain="val", multistep=3, **tiny_forecast_kwargs)
    batch = collate_fn([dataset[0], dataset[1]])

    out = dataset.convert_trajectory_to_xarray(batch["future_states"], batch["timestamp"])

    assert out.sizes["prediction_timedelta"] == 3
    assert out.sizes["time"] == 2
    assert out.prediction_timedelta.values[0] == np.timedelta64(24, "h")
    assert out.prediction_timedelta.values[-1] == np.timedelta64(72, "h")


def test_surface_only_component_round_trips(tiny_forecast_kwargs):
    """``seaice_isolated`` predicts 2-D fields only, so nothing may assume a
    "level" key exists.  The coupling task builds exactly this component."""
    component = get_component("seaice_isolated")
    dataset = GlorysForecast(
        domain="val",
        variables=dict(surface=component.prognostic_surface, level=component.prognostic_level),
        **tiny_forecast_kwargs,
    )
    sample = dataset[0]

    assert "level" not in sample["state"].keys()
    assert sample["state"]["surface"].shape == (4, 1, N_LAT, N_LON)
    assert "level" not in dataset.state_mask().keys()

    out = dataset.convert_to_xarray(dataset.denormalize(sample["state"]), sample["timestamp"])
    assert set(out.data_vars) == set(component.prognostic_surface)
    assert "depth" not in out.dims
    assert out.sizes == {"time": 1, "lat": N_LAT, "lon": N_LON}

    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["state"]["surface"].shape == (2, 4, 1, N_LAT, N_LON)


def test_empty_variable_request_is_rejected(tiny_forecast_kwargs):
    with pytest.raises(ValueError, match="No variables requested"):
        GlorysForecast(domain="val", variables=dict(surface=[], level=[]), **tiny_forecast_kwargs)


def test_convert_to_xarray_refuses_a_timestamp_batch_mismatch(tiny_forecast_kwargs):
    """Task 7 keys the output files on this coordinate, so guessing here would
    produce a plausible-looking file with the wrong dates on it."""
    dataset = GlorysForecast(domain="val", **tiny_forecast_kwargs)
    batch = collate_fn([dataset[0], dataset[1]])
    three_stamps = torch.tensor([0, 1, 2], dtype=torch.int32)

    with pytest.raises(ValueError, match="3 timestamps for a batch of 2"):
        dataset.convert_to_xarray(batch["state"], three_stamps)

    # One timestamp for the whole batch stays allowed: the trajectory writer
    # relies on it for a single initialisation time.
    out = dataset.convert_to_xarray(batch["state"], torch.tensor(0, dtype=torch.int32))
    assert out.sizes["time"] == 2
    assert (out.time.values == out.time.values[0]).all()


def test_convert_to_xarray_ignores_era5_pressure_levels(forecast):
    """geoarches' forecast module hardcodes ``levels=[300, 500, 700, 850]``.
    Our vertical coordinate is depth in metres, so warn rather than crash."""
    sample = forecast[0]
    with pytest.warns(UserWarning, match="depth in metres"):
        out = forecast.convert_to_xarray(
            sample["state"], sample["timestamp"], levels=[300, 500, 700, 850]
        )
    assert out.sizes["depth"] == N_DEPTH


def test_convert_to_xarray_selects_real_depths(forecast):
    sample = forecast[0]
    out = forecast.convert_to_xarray(
        sample["state"], sample["timestamp"], levels=[PREPPED_DEPTHS[0], PREPPED_DEPTHS[3]]
    )
    assert out.sizes["depth"] == 2


# ---------------------------------------------------------------------------
# Partial data
# ---------------------------------------------------------------------------
def test_a_split_with_no_prepared_years_says_which_years_are_missing(
    tiny_archive, tiny_masks_file, tmp_path
):
    """`make doctor` suggests `make prep-data YEARS="2015 2016"`, and `make eval`
    then scores on `test` = 2021-2023.

    geoarches' own message for that is `ValueError: ('filename_filter filtered all
    files under path:', '/.../glorys_1deg_prepped')`, which names neither the
    split nor a year, and it arrives after the module and the metrics are built.

    MUTANT: removing the `_check_split_is_present` call from
    `GlorysDataset.__init__` brings geoarches' message back and this fails on
    the `match=`.
    """
    partial = tmp_path / "partial"
    partial.mkdir()
    for year in (2015, 2016):
        (partial / f"glorys_1deg_{year}.nc").write_bytes(
            (tiny_archive / f"glorys_1deg_{ARCHIVE_YEARS[0]}.nc").read_bytes()
        )
    with pytest.raises(ValueError, match=r"'test' split \(2021-2023\)"):
        GlorysDataset(path=partial, domain="test", masks_path=tiny_masks_file)
    with pytest.raises(ValueError, match="2015-2016"):
        GlorysDataset(path=partial, domain="test", masks_path=tiny_masks_file)


# ---------------------------------------------------------------------------
# norm_scheme
# ---------------------------------------------------------------------------
def test_norm_scheme_none_the_string_is_refused(tiny_forecast_kwargs):
    """`norm_scheme: none` in YAML is the *string* "none", which is truthy.

    Everything downstream only asks `if self.norm_scheme`, so the string sailed
    through and the data came back normalised -- the opposite of what was asked,
    silently.

    MUTANT: deleting the check in `GlorysForecast.__init__` makes this pass a
    dataset that normalises, and the test fails.
    """
    with pytest.raises(ValueError, match="norm_scheme must be 'glorys' or null"):
        GlorysForecast(domain="train", norm_scheme="none", **tiny_forecast_kwargs)
    with pytest.raises(ValueError, match="norm_scheme"):
        GlorysForecast(domain="train", norm_scheme="zscore", **tiny_forecast_kwargs)
    # The two spellings that are real still work.
    assert (
        GlorysForecast(domain="train", norm_scheme=None, **tiny_forecast_kwargs).norm_scheme
        is None
    )
    assert (
        GlorysForecast(domain="train", norm_scheme="glorys", **tiny_forecast_kwargs).norm_scheme
        == "glorys"
    )
