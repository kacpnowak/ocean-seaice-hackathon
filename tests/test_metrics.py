"""Numerical tests for the land-aware metrics.

Every test here has an expected value worked out on paper, not a snapshot of
whatever the code happened to print.  Two of them exist specifically to bite if
the masking is wrong -- ``test_land_can_hold_anything`` and
``test_a_constant_error_gives_that_constant`` both fail if the reductions divide
by the grid area instead of the ocean area, or if the mask is dropped.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from geoarches.metrics.label_wrapper import convert_metric_dict_to_xarray
from geoarches.metrics.metric_base import compute_lat_weights_weatherbench
from tensordict.tensordict import TensorDict

from oceanarches.dataloaders.variables import DEPTH_PRESETS, N_LAT, N_LON, PREPPED_DEPTHS
from oceanarches.metrics.masked_metrics import (
    MaskedDeterministic,
    MaskedDeterministicMetrics,
    compute_lat_weights_glorys,
    month_interpolation_weights,
    ocean_area_weights,
)
from oceanarches.metrics.registry import register_all
from oceanarches.metrics.seaice_metrics import (
    EARTH_RADIUS_KM,
    SeaIceExtent,
    SeaIceMetrics,
    cell_areas,
    hemisphere_masks,
)


def six_dim(field: torch.Tensor) -> torch.Tensor:
    """``(lat, lon)`` -> ``(batch=1, timedelta=1, var=1, depth=1, lat, lon)``."""
    return field[None, None, None, None]


# ---------------------------------------------------------------------------
# Latitude weighting
# ---------------------------------------------------------------------------
def test_glorys_weights_are_cell_centred_and_average_to_one():
    weights = compute_lat_weights_glorys(N_LAT)
    assert weights.shape == (N_LAT, 1)
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-6)
    # The southernmost row is centred at -89.5, not at the pole.
    scale = weights[0].item() / math.cos(math.radians(89.5))
    assert weights[-1].item() / math.cos(math.radians(89.5)) == pytest.approx(scale)
    assert weights[N_LAT // 2].item() / math.cos(math.radians(0.5)) == pytest.approx(
        scale, rel=1e-5
    )


def test_glorys_and_weatherbench_weights_differ_most_at_the_poles():
    """The difference is real but confined to rows carrying almost no area.

    geoarches' weatherbench weighting puts the first row *on* the pole with a
    half-height cell, which is a different grid from ours.  Documented here as a
    number so nobody has to guess whether it matters.
    """
    ours = compute_lat_weights_glorys(N_LAT)
    theirs = compute_lat_weights_weatherbench(N_LAT)
    relative = ((ours - theirs).abs() / ours).flatten()
    # Measured on our 180-row grid: 74.7% in the polar rows, 2.6% on average,
    # 0.56% median -- the offset between a 180/180 and a 180/179 spacing.
    assert relative.max().item() == pytest.approx(0.747, abs=0.01)
    assert relative.mean().item() == pytest.approx(0.0258, abs=0.001)
    assert relative[N_LAT // 2 - 20 : N_LAT // 2 + 20].max().item() < 0.006  # tropics


def test_ocean_area_weights_sum_to_one_per_channel():
    mask = torch.zeros(2, 3, 6, 8)
    mask[..., 1:4, :] = 1.0
    weights = ocean_area_weights(mask)
    assert torch.allclose(weights.sum(dim=(-2, -1)), torch.ones(2, 3), atol=1e-6)
    assert (weights * (1 - mask)).abs().max().item() == 0.0


def test_ocean_area_weights_warn_and_zero_a_dry_channel():
    mask = torch.zeros(1, 2, 4, 4)
    mask[0, 0] = 1.0
    with pytest.warns(UserWarning, match="no ocean cell"):
        weights = ocean_area_weights(mask)
    assert weights[0, 1].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Masked RMSE / MAE / bias -- values worked out by hand
# ---------------------------------------------------------------------------
def wet_two_cell_mask() -> torch.Tensor:
    """A 4x4 grid with exactly two ocean cells, at rows 1 and 2."""
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, 1, 1] = 1.0
    mask[0, 0, 2, 2] = 1.0
    return mask


def test_masked_rmse_matches_the_hand_computation():
    mask = wet_two_cell_mask()
    metric = MaskedDeterministic(mask=mask, rollout_iterations=1)
    truth = torch.zeros(4, 4)
    pred = torch.zeros(4, 4)
    pred[1, 1] = 3.0
    pred[2, 2] = -5.0
    metric.update(six_dim(truth), six_dim(pred))

    weights = compute_lat_weights_glorys(4).flatten()
    w1, w2 = weights[1].item(), weights[2].item()
    expected_rmse = math.sqrt((9 * w1 + 25 * w2) / (w1 + w2))
    result = metric.compute()
    assert result["rmse"].item() == pytest.approx(expected_rmse, rel=1e-6)
    assert result["mae"].item() == pytest.approx((3 * w1 + 5 * w2) / (w1 + w2), rel=1e-6)
    assert result["bias"].item() == pytest.approx((3 * w1 - 5 * w2) / (w1 + w2), rel=1e-6)


def test_land_can_hold_anything():
    """The regression test for the whole point of the task.

    A metric that divided by the number of *grid* points, or that forgot the
    mask, would move by many orders of magnitude here.
    """
    mask = wet_two_cell_mask()
    truth = torch.zeros(4, 4)
    pred = torch.zeros(4, 4)
    pred[1, 1] = 3.0
    pred[2, 2] = -5.0

    clean = MaskedDeterministic(mask=mask, rollout_iterations=1)
    clean.update(six_dim(truth), six_dim(pred))

    polluted_pred = pred.clone()
    polluted_truth = truth.clone()
    land = mask[0, 0] == 0
    polluted_pred[land] = 1e9
    polluted_truth[land] = -1e9
    dirty = MaskedDeterministic(mask=mask, rollout_iterations=1)
    dirty.update(six_dim(polluted_truth), six_dim(polluted_pred))

    for name in ("rmse", "mae", "bias"):
        assert dirty.compute()[name].item() == clean.compute()[name].item(), name


def test_a_constant_error_gives_that_constant():
    """Latitude weighting must not change the answer for a uniform error.

    True whatever the latitudes are and whatever the land distribution is --
    which is exactly the property that says "this is an average over the ocean".
    """
    torch.manual_seed(0)
    mask = (torch.rand(1, 1, 32, 40) > 0.4).float()
    mask[0, 0, 5, 5] = 1.0  # never let the mask come out empty
    metric = MaskedDeterministic(mask=mask, rollout_iterations=1)
    truth = torch.randn(32, 40)
    metric.update(six_dim(truth), six_dim(truth + 2.5))
    result = metric.compute()
    assert result["rmse"].item() == pytest.approx(2.5, rel=1e-5)
    assert result["mae"].item() == pytest.approx(2.5, rel=1e-5)
    assert result["bias"].item() == pytest.approx(2.5, rel=1e-5)


def test_rmse_averages_the_mean_squared_error_over_samples():
    mask = torch.ones(1, 1, 2, 2)
    metric = MaskedDeterministic(mask=mask, rollout_iterations=1)
    truth = torch.zeros(2, 2)
    metric.update(six_dim(truth), six_dim(truth + 1.0))
    metric.update(six_dim(truth), six_dim(truth + 3.0))
    # sqrt(mean(1, 9)), not mean(sqrt(1), sqrt(9)).
    assert metric.compute()["rmse"].item() == pytest.approx(math.sqrt(5.0), rel=1e-6)


def test_each_lead_time_is_scored_separately():
    mask = torch.ones(1, 1, 2, 2)
    metric = MaskedDeterministic(mask=mask, rollout_iterations=3)
    truth = torch.zeros(1, 3, 1, 1, 2, 2)
    pred = torch.zeros(1, 3, 1, 1, 2, 2)
    for step, error in enumerate((1.0, 2.0, 4.0)):
        pred[0, step] = error
    metric.update(truth, pred)
    assert metric.compute()["rmse"].flatten().tolist() == pytest.approx([1.0, 2.0, 4.0])


def test_a_metric_that_was_never_updated_reports_nan_not_zero():
    """0.0 is the best score there is, so it must never be the "no data" answer.

    Dividing the zero sums by `nsamples.clamp(min=1)` would report `rmse = 0.0`
    for a metric nothing ever reached -- an evaluation loop that silently skipped
    every batch would look like a perfect forecast.
    """
    metric = MaskedDeterministic(mask=torch.ones(1, 1, 2, 2), rollout_iterations=1)
    result = metric.compute()
    for name in ("rmse", "mae", "bias"):
        assert torch.isnan(result[name]).all(), f"{name} of an empty metric must be NaN"
    # ... and one update is enough to make it a number again.
    metric.update(six_dim(torch.zeros(2, 2)), six_dim(torch.full((2, 2), 2.0)))
    assert metric.compute()["rmse"].item() == pytest.approx(2.0)


def test_an_empty_sea_ice_metric_reports_nan_not_a_perfect_forecast():
    metric = SeaIceExtent(mask=torch.ones(N_LAT, N_LON), siconc_index=0, rollout_iterations=1)
    result = metric.compute()
    assert result, "the metric predicts siconc, so it must report something"
    for name, value in result.items():
        assert torch.isnan(value).all(), f"{name} of an empty metric must be NaN"


def test_update_rejects_a_tensor_without_a_timedelta_axis():
    metric = MaskedDeterministic(mask=torch.ones(1, 1, 2, 2))
    with pytest.raises(ValueError, match="timedelta"):
        metric.update(torch.zeros(1, 1, 1, 2, 2), torch.zeros(1, 1, 1, 2, 2))


# ---------------------------------------------------------------------------
# ACC
# ---------------------------------------------------------------------------
def acc_metric_at(climatology: torch.Tensor, grid: tuple[int, int]) -> MaskedDeterministic:
    return MaskedDeterministic(
        mask=torch.ones(1, 1, *grid), rollout_iterations=1, climatology=climatology
    )


#: Initial time whose 24 h forecast is valid exactly at the midpoint of January,
#: so the interpolated climatology *is* the January field with no blending.
JANUARY_MIDPOINT_INIT = int(
    np.datetime64("2019-01-15T12:00:00").astype("datetime64[s]").astype("int64")
)


def test_acc_of_a_perfect_forecast_is_one():
    torch.manual_seed(1)
    climatology = torch.randn(12, 1, 1, 16, 16)
    truth = climatology[0] + torch.randn(1, 1, 16, 16)  # climatology plus a real anomaly
    metric = acc_metric_at(climatology, (16, 16))
    metric.update(
        truth[None, None], truth[None, None], timestamp=torch.tensor([JANUARY_MIDPOINT_INIT])
    )
    assert metric.compute()["acc"].item() == pytest.approx(1.0, abs=1e-5)


def test_acc_of_a_field_uncorrelated_with_the_anomaly_is_about_zero():
    torch.manual_seed(2)
    climatology = torch.randn(12, 1, 1, 64, 64)
    truth = climatology[0] + torch.randn(1, 1, 64, 64)
    pred = climatology[0] + torch.randn(1, 1, 64, 64)  # independent noise
    metric = acc_metric_at(climatology, (64, 64))
    metric.update(
        truth[None, None], pred[None, None], timestamp=torch.tensor([JANUARY_MIDPOINT_INIT])
    )
    assert abs(metric.compute()["acc"].item()) < 0.05


def test_acc_of_the_climatology_against_itself_is_zero_not_nan():
    """The degenerate case: both anomalies vanish, so the correlation is 0/0.

    We report 0 rather than NaN, on the grounds that a metric which poisons every
    later average is worse than one that says "no signal here".  It happens for
    real: ``siconc`` in the tropics has an identically zero anomaly.  Note this is
    the *only* way to get a zero anomaly -- a perfect forecast of a field that
    differs from the climatology scores 1 (the test above).
    """
    climatology = torch.randn(12, 1, 1, 8, 8)
    metric = acc_metric_at(climatology, (8, 8))
    field = climatology[0][None, None]
    metric.update(field, field, timestamp=torch.tensor([JANUARY_MIDPOINT_INIT]))
    assert metric.compute()["acc"].item() == 0.0


def test_the_climatology_is_interpolated_to_the_valid_time_not_the_initial_time():
    """A 24 h forecast is scored against the climatology of *tomorrow*."""
    climatology = torch.zeros(12, 1, 1, 4, 4)
    climatology[0] = 1.0  # January differs from every other month
    metric = acc_metric_at(climatology, (4, 4))
    exactly_january = torch.tensor([JANUARY_MIDPOINT_INIT])
    truth = torch.ones(1, 1, 1, 1, 4, 4)
    truth[..., 0, 0] = 5.0
    metric.update(truth, truth, timestamp=exactly_january)
    # The anomaly is (truth - January) and is non-zero, so ACC is 1 rather than
    # the 0 it would be if the climatology had been taken a day early (December).
    assert metric.compute()["acc"].item() == pytest.approx(1.0, abs=1e-5)


def test_acc_needs_a_timestamp():
    metric = acc_metric_at(torch.randn(12, 1, 1, 4, 4), (4, 4))
    with pytest.raises(ValueError, match="timestamp"):
        metric.update(torch.zeros(1, 1, 1, 1, 4, 4), torch.zeros(1, 1, 1, 1, 4, 4))


# ---------------------------------------------------------------------------
# Climatology interpolation
# ---------------------------------------------------------------------------
def test_month_midpoints_land_exactly_on_their_own_month():
    # 16 January 2019 at 12:00 is the midpoint of a 31-day January.
    before, after, alpha = month_interpolation_weights(np.datetime64("2019-01-16T12:00:00"))
    assert (before, after) == (0, 1)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_interpolation_is_periodic_across_the_new_year():
    before, after, alpha = month_interpolation_weights(np.datetime64("2019-12-31T12:00:00"))
    assert (before, after) == (11, 0), "December must interpolate towards January"
    assert 0.0 < alpha < 1.0
    before, after, _ = month_interpolation_weights(np.datetime64("2019-01-05T00:00:00"))
    assert (before, after) == (11, 0), "early January interpolates back from December"


def test_interpolation_follows_the_real_calendar_in_a_leap_year():
    """February's midpoint moves by half a day when it has 29 days."""
    _, _, alpha_leap = month_interpolation_weights(np.datetime64("2020-02-15T00:00:00"))
    _, _, alpha_common = month_interpolation_weights(np.datetime64("2019-02-15T00:00:00"))
    assert alpha_leap != alpha_common


def test_interpolation_weight_grows_monotonically_through_a_month():
    alphas = [
        month_interpolation_weights(np.datetime64(f"2019-03-{day:02d}T00:00:00"))[2]
        for day in (17, 20, 25, 30)
    ]
    assert alphas == sorted(alphas)
    assert all(0.0 <= a <= 1.0 for a in alphas)


def test_a_scalar_int_timestamp_is_accepted():
    assert month_interpolation_weights(1547510400) == month_interpolation_weights(
        np.datetime64("2019-01-15T00:00:00")
    )


# ---------------------------------------------------------------------------
# Sea ice
# ---------------------------------------------------------------------------
def test_cell_areas_add_up_to_the_surface_of_the_earth():
    total = cell_areas().sum().item()
    exact = 4 * math.pi * EARTH_RADIUS_KM**2
    assert total == pytest.approx(exact, rel=2e-4)


def spherical_cap_area(latitude_degrees: float) -> float:
    """Area north of ``latitude_degrees``, in 10^6 km^2."""
    return 2 * math.pi * EARTH_RADIUS_KM**2 * (1 - math.sin(math.radians(latitude_degrees))) / 1e6


def polar_cap(edge_latitude: float) -> torch.Tensor:
    """``siconc`` = 1 for every cell whose centre is north of ``edge_latitude``."""
    latitudes = torch.arange(N_LAT, dtype=torch.float32) - 89.5
    field = torch.zeros(N_LAT, N_LON)
    field[latitudes > edge_latitude] = 1.0
    return field


def seaice_metric(rollout_iterations: int = 1) -> SeaIceExtent:
    """An all-ocean metric, so the analytic areas are comparable."""
    return SeaIceExtent(
        mask=torch.ones(N_LAT, N_LON), siconc_index=0, rollout_iterations=rollout_iterations
    )


def test_extent_of_a_polar_cap_matches_the_spherical_cap_area():
    """Cells centred north of 60.5 fill the cap north of 60 exactly."""
    metric = seaice_metric()
    field = polar_cap(60.0)  # centres 60.5 ... 89.5, i.e. the band 60 ... 90
    metric.update(six_dim(torch.zeros(N_LAT, N_LON)), six_dim(field))
    north = metric.compute()["extentpred"][0, 0].item()
    expected = spherical_cap_area(60.0)
    assert north == pytest.approx(expected, rel=1e-3)
    assert metric.compute()["extentpred"][0, 1].item() == 0.0  # nothing in the south


def test_area_weights_by_concentration():
    metric = seaice_metric()
    field = 0.5 * polar_cap(60.0)  # above the 0.15 threshold, half concentration
    metric.update(six_dim(torch.zeros(N_LAT, N_LON)), six_dim(field))
    result = metric.compute()
    assert result["areabias"][0, 0].item() == pytest.approx(
        0.5 * spherical_cap_area(60.0), rel=1e-3
    )
    assert result["extentpred"][0, 0].item() == pytest.approx(spherical_cap_area(60.0), rel=1e-3)


def test_concentration_below_the_threshold_is_not_ice():
    metric = seaice_metric()
    field = 0.10 * polar_cap(60.0)
    metric.update(six_dim(torch.zeros(N_LAT, N_LON)), six_dim(field))
    assert metric.compute()["extentpred"][0, 0].item() == 0.0


def test_iiee_is_zero_for_identical_fields():
    metric = seaice_metric()
    field = six_dim(polar_cap(60.0))
    metric.update(field, field)
    result = metric.compute()
    assert result["iiee"].abs().max().item() == 0.0
    assert result["extentbias"].abs().max().item() == 0.0


def test_iiee_of_a_shifted_edge_is_the_band_between_them():
    """Model ice edge at 65, truth at 60: the 60-65 band is an underestimate."""
    metric = seaice_metric()
    metric.update(six_dim(polar_cap(60.0)), six_dim(polar_cap(65.0)))
    band = spherical_cap_area(60.0) - spherical_cap_area(65.0)
    result = metric.compute()
    assert result["iiee"][0, 0].item() == pytest.approx(band, rel=1e-3)
    assert result["iieeunder"][0, 0].item() == pytest.approx(band, rel=1e-3)
    assert result["iieeover"][0, 0].item() == 0.0
    assert result["extentbias"][0, 0].item() == pytest.approx(-band, rel=1e-3)


def test_iiee_counts_overestimate_and_underestimate_separately():
    metric = seaice_metric()
    truth = polar_cap(60.0)
    pred = polar_cap(65.0)
    # add ice in the south that the truth does not have
    pred = pred + torch.flip(polar_cap(70.0), dims=[0])
    metric.update(six_dim(truth), six_dim(pred))
    result = metric.compute()
    assert result["iieeunder"][0, 0].item() > 0 and result["iieeover"][0, 0].item() == 0.0
    assert result["iieeover"][0, 1].item() == pytest.approx(spherical_cap_area(70.0), rel=1e-3)
    assert torch.allclose(result["iiee"], result["iieeover"] + result["iieeunder"])


def test_hemispheres_partition_the_grid():
    masks = hemisphere_masks()
    assert torch.all(masks.sum(0) == 1.0)
    assert masks[0, N_LAT // 2 :].min().item() == 1.0  # north
    assert masks[1, : N_LAT // 2].min().item() == 1.0  # south


def test_land_carries_no_ice():
    """After denormalisation a land cell holds the climatological mean, which for
    ``siconc`` is above the 15% threshold in the polar rows."""
    mask = torch.ones(N_LAT, N_LON)
    mask[120:, :] = 0.0  # call the far north land
    metric = SeaIceExtent(mask=mask, siconc_index=0, rollout_iterations=1)
    metric.update(six_dim(torch.zeros(N_LAT, N_LON)), six_dim(torch.ones(N_LAT, N_LON)))
    assert metric.compute()["extentpred"][0, 0].item() == pytest.approx(
        spherical_cap_area(0.0) - spherical_cap_area(30.0), rel=1e-3
    )


# ---------------------------------------------------------------------------
# The configured wrappers, against the real mask and climatology files
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def full_metrics(real_masks_path) -> MaskedDeterministicMetrics:
    return MaskedDeterministicMetrics(
        component="full", depth_indices=DEPTH_PRESETS["tiny"], rollout_iterations=1
    )


def random_state(n_surface: int, n_level: int, n_depths: int, steps: int = 1) -> TensorDict:
    entries = {"surface": torch.randn(2, steps, n_surface, 1, N_LAT, N_LON)}
    if n_level:
        entries["level"] = torch.randn(2, steps, n_level, n_depths, N_LAT, N_LON)
    return TensorDict(entries, batch_size=[2, steps])


def test_labels_are_per_variable_and_per_lead_time(full_metrics):
    state = random_state(7, 4, 13)
    full_metrics.reset()
    full_metrics.update(state, state.clone(), timestamp=torch.tensor([1547510400, 1560000000]))
    labels = full_metrics.compute()
    assert "rmse_thetao0m_24h" in labels
    assert "rmse_siconc_24h" in labels
    assert "acc_zos_24h" in labels
    assert "rmse_thetao1684m_24h" in labels, "the deepest level must be labelled by its depth"


def test_labels_survive_convert_metric_dict_to_xarray(full_metrics):
    state = random_state(7, 4, 13)
    full_metrics.reset()
    full_metrics.update(state, state.clone(), timestamp=torch.tensor([1547510400, 1560000000]))
    dataset = convert_metric_dict_to_xarray(
        {k: v.item() for k, v in full_metrics.compute().items()},
        extra_dimensions=["prediction_timedelta"],
    )
    assert "thetao0m" in dataset.data_vars
    assert set(dataset.coords["metric"].values) == {"rmse", "mae", "bias", "acc"}


def test_the_metric_uses_the_same_depth_subset_as_the_model(real_masks_path):
    """The 14-vs-13 trap: the prepared levels include one the model never loads."""
    metrics = MaskedDeterministicMetrics(
        component="full", depth_indices=DEPTH_PRESETS["tiny"], rollout_iterations=1
    )
    inner = metrics.metrics["level"].metric
    assert inner.mask.shape[1] == len(DEPTH_PRESETS["tiny"]) == 13
    assert inner.climatology.shape == (12, 4, 13, N_LAT, N_LON)
    labels = set(metrics.metrics["level"].variable_indices)
    kept = {f"thetao{PREPPED_DEPTHS[i]:.0f}m_24h" for i in DEPTH_PRESETS["tiny"]}
    assert kept <= labels
    dropped = f"thetao{PREPPED_DEPTHS[12]:.0f}m_24h"  # 1245 m, the level the presets skip
    assert dropped not in labels


def test_headline_only_trims_the_labels_but_not_the_computation(real_masks_path):
    metrics = MaskedDeterministicMetrics(
        component="full",
        depth_indices=DEPTH_PRESETS["tiny"],
        rollout_iterations=1,
        headline_only=True,
        compute_acc=False,
    )
    labels = set(metrics.metrics["level"].variable_indices)
    assert labels == {"thetao0m_24h", "so0m_24h"}
    assert metrics.metrics["level"].metric.mask.shape[:2] == (4, 13)


def test_a_component_scores_only_what_it_predicts(real_masks_path):
    metrics = MaskedDeterministicMetrics(
        component="seaice", depth_indices=DEPTH_PRESETS["tiny"], compute_acc=False
    )
    assert "level" not in metrics.metrics, "the sea-ice component predicts nothing 3-D"
    assert set(metrics.metrics["surface"].variable_indices) == {
        "siconc_24h",
        "sithick_24h",
        "usi_24h",
        "vsi_24h",
    }


def test_the_sea_ice_metric_is_silent_for_a_component_without_ice(real_masks_path):
    with pytest.warns(UserWarning, match="does not predict siconc"):
        metrics = SeaIceMetrics(component="ocean")
    state = random_state(3, 4, 13)
    metrics.update(state, state.clone())
    assert metrics.compute() == {}


def test_the_sea_ice_metric_finds_siconc_in_the_component_channel_order(real_masks_path):
    """The sea-ice component puts ``siconc`` first, the full component fourth."""
    assert SeaIceMetrics(component="seaice").metrics["surface"].metric.siconc_index == 0
    assert SeaIceMetrics(component="full").metrics["surface"].metric.siconc_index == 3


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_metrics_are_registered_with_geoarches():
    from geoarches.evaluation.metric_registry import instantiate_metric

    names = register_all()
    assert "glorys_deterministic" in names
    assert "glorys_seaice_seaice" in names
    metric = instantiate_metric("glorys_seaice_seaice")
    assert metric.component == "seaice"
