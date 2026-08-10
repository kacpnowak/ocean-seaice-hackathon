"""Fast, CPU-only tests for the evaluation pipeline.

Nothing here loads a checkpoint or rolls a real model out; the fixtures build
small synthetic arrays and a stub module.  A test suite that needed a GPU would
not be run, and a test that takes four minutes is a test nobody runs twice.

Several tests exist specifically because the property they pin is one a reviewer
cannot check by eye, and each was confirmed to **fail** against a deliberately
broken implementation before being kept.  Those are marked below with a
``MUTANT:`` note saying exactly what was broken and what happened.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import typing
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from tensordict.tensordict import TensorDict

from oceanarches.dataloaders import variables as V
from oceanarches.evaluation import (
    animate,
    baselines,
    plots,
    provenance,
    render_cache,
    report,
    rollout,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
N_LAT, N_LON = 12, 24
N_SURFACE, N_LEVEL, N_DEPTH = 3, 2, 4


class StubModule:
    """The smallest thing the scoring loop needs: statistics, a mask and a loss.

    Not a mock: every method does the real arithmetic the real module does, on
    tensors small enough to be instant.  A mock would let the loop under test
    pass while doing nothing.
    """

    def __init__(self, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.state_mean_surface = torch.rand((N_SURFACE, 1, 1, 1), generator=generator)
        self.state_std_surface = 1.0 + torch.rand((N_SURFACE, 1, 1, 1), generator=generator)
        self.state_mean_level = torch.rand((N_LEVEL, N_DEPTH, 1, 1), generator=generator)
        self.state_std_level = 1.0 + torch.rand((N_LEVEL, N_DEPTH, 1, 1), generator=generator)
        self.mask_surface = (
            torch.rand((N_SURFACE, 1, N_LAT, N_LON), generator=generator) > 0.3
        ).float()
        self.mask_level = (
            torch.rand((N_LEVEL, N_DEPTH, N_LAT, N_LON), generator=generator) > 0.3
        ).float()
        self.n_surface_out, self.n_level_out = N_SURFACE, N_LEVEL
        self.lead_time_hours = 24
        self.surface_variables = V.SURFACE_VARIABLES[:N_SURFACE]
        self.level_variables = V.LEVEL_VARIABLES[:N_LEVEL]
        self.depth_indices = list(range(N_DEPTH))

    def _like(self, entries, like):
        return TensorDict(entries, batch_size=like.batch_size)

    def select_prognostic(self, state):
        return state

    def apply_wet_mask(self, state):
        entries = {"surface": state["surface"] * self.mask_surface}
        if "level" in state.keys():
            entries["level"] = state["level"] * self.mask_level
        return self._like(entries, state)

    def denormalize_state(self, state):
        entries = {"surface": state["surface"] * self.state_std_surface + self.state_mean_surface}
        if "level" in state.keys():
            entries["level"] = state["level"] * self.state_std_level + self.state_mean_level
        return self._like(entries, state)

    def loss(self, pred, gt, multistep=False, **kwargs):
        return float(((pred - gt) ** 2).mean().sum().values().__iter__().__next__())


def _state(batch: int, generator=None) -> TensorDict:
    return TensorDict(
        {
            "surface": torch.randn((batch, N_SURFACE, 1, N_LAT, N_LON), generator=generator),
            "level": torch.randn((batch, N_LEVEL, N_DEPTH, N_LAT, N_LON), generator=generator),
        },
        batch_size=(batch,),
    )


def _batch(batch: int = 2, iters: int = 3, seed: int = 1) -> dict:
    generator = torch.Generator().manual_seed(seed)
    state = _state(batch, generator)
    futures = TensorDict(
        {
            key: torch.stack(
                [_state(batch, generator)[key] for _ in range(iters)],
                dim=1,
            )
            for key in ("surface", "level")
        },
        batch_size=(batch, iters),
    )
    return {
        "state": state,
        "prev_state": _state(batch, generator),
        "future_states": futures,
        "timestamp": torch.tensor([1_577_000_000 + 86400 * i for i in range(batch)]),
    }


class RecordingMetric:
    """Records the exact tensors it was updated with, so a test can compare them."""

    def __init__(self):
        self.calls = []

    def update(self, targets, preds, timestamp=None):
        self.calls.append((targets, preds, timestamp))


def _metric_dataset(
    variables: dict[str, np.ndarray], metrics: list[str], leads: list[int]
) -> xr.Dataset:
    """A metric dataset shaped exactly like ``convert_metric_dict_to_xarray``'s output."""
    return xr.Dataset(
        {
            name: (["metric", "prediction_timedelta"], np.asarray(values, dtype=float))
            for name, values in variables.items()
        },
        coords={
            "metric": metrics,
            "prediction_timedelta": [np.timedelta64(24 * lead, "h") for lead in leads],
        },
    )


@pytest.fixture
def synthetic_result(tmp_path):
    """A ``RolloutResult`` with real metric datasets and no cached fields."""
    leads = [1, 2, 3, 4]
    n = len(leads)
    rng = np.random.default_rng(0)

    def block(scale, growth):
        rmse = scale * (1 + growth * np.arange(n))
        return np.stack([rmse, 0.7 * rmse, 0.05 * rmse, 0.99 - 0.02 * np.arange(n)])

    deterministic = {}
    for key, scale in (
        ("thetao0m", 0.12),
        ("thetao1684m", 0.001),
        ("so0m", 0.07),
        ("so1684m", 0.0012),
        ("zos", 0.018),
        ("siconc", 0.011),
        ("sithick", 0.011),
        ("mlotst", 9.0),
    ):
        deterministic[key] = block(scale, 0.4 + 0.1 * rng.random())
    metrics_names = ["rmse", "mae", "bias", "acc"]

    seaice = {
        name: np.stack([0.24 + 0.05 * np.arange(n), 0.05 + 0.01 * np.arange(n)])
        for name in ("seaiceNH", "seaiceSH")
    }

    def scaled(source, factor):
        return {k: v * factor for k, v in source.items()}

    result = rollout.RolloutResult(
        spec=rollout.RolloutSpec(
            experiment="synthetic",
            domain="test",
            lead_days=n,
            n_inits=4,
            selection="spread",
            save_depths="shallow",
            checkpoint="ckpt",
            inits=(0, 1, 2, 3),
        ),
        directory=tmp_path,
        losses={"model": 0.7, "persistence": 0.81, "climatology": 39.0},
        labels={"model": "Model", "persistence": "Persistence", "climatology": "Climatology"},
        init_times=["2021-01-01T12:00:00", "2021-06-01T12:00:00"],
        seconds=1.5,
        n_samples=4,
    )
    for key, factor in (("model", 1.0), ("persistence", 1.2), ("climatology", 6.0)):
        result.metrics[key] = {
            "glorys_deterministic_metrics": _metric_dataset(
                scaled(deterministic, factor), metrics_names, leads
            ),
            "glorys_seaice_metrics": _metric_dataset(
                scaled(seaice, factor), ["iiee", "extentbias"], leads
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def test_persistence_returns_the_initial_state_at_every_lead():
    """Persistence is *today*, repeated -- never tomorrow.

    MUTANT: returning ``batch["next_state"]`` instead of ``batch["state"]``
    (a baseline that peeks at the answer, which would make every model look
    hopeless). This test FAILS against it -- the mismatch is ~5.7 in max
    absolute difference on the fixture -- while a test that only checked the
    *shape* would pass.
    """
    module = StubModule()
    batch = _batch(batch=2, iters=3)
    prediction = baselines.PersistenceForecast(module).predict(batch, 3)

    assert prediction.batch_size == torch.Size([2, 3])
    for group in ("surface", "level"):
        for step in range(3):
            torch.testing.assert_close(prediction[group][:, step], batch["state"][group])


def test_persistence_does_not_look_at_the_future():
    """The complement of the test above, stated as an inequality.

    Random future states differ from the initial state, so a persistence
    forecast that matched them would be evidence of leakage.
    """
    module = StubModule()
    batch = _batch(batch=2, iters=3)
    prediction = baselines.PersistenceForecast(module).predict(batch, 3)
    difference = (prediction["surface"] - batch["future_states"]["surface"]).abs().max()
    assert float(difference) > 1e-3


def test_targets_follow_the_dataloader_shape():
    module = StubModule()
    batch = _batch(batch=2, iters=3)
    targets = baselines.targets_for(module, batch, 3)
    torch.testing.assert_close(targets["surface"], batch["future_states"]["surface"])

    single = {k: v for k, v in batch.items() if k != "future_states"}
    single["next_state"] = _state(2)
    one_step = baselines.targets_for(module, single, 1)
    assert one_step["surface"].shape[1] == 1
    torch.testing.assert_close(one_step["surface"][:, 0], single["next_state"]["surface"])

    with pytest.raises(ValueError, match="future_states"):
        baselines.targets_for(module, single, 5)


def test_as_trajectory_declares_the_batch_and_timedelta_axes():
    """TensorDict arithmetic broadcasts the declared batch size, not the shapes.

    Without this the baselines' predictions cannot be subtracted from the
    targets at all, which is how the bug was found.
    """
    entries = {"surface": torch.zeros((2, 5, N_SURFACE, 1, N_LAT, N_LON))}
    trajectory = baselines.as_trajectory(entries, 2, 5)
    assert trajectory.batch_size == torch.Size([2, 5])
    (trajectory - trajectory).batch_size  # must not raise


def test_month_interpolation_is_on_midpoints_and_wraps_at_the_new_year():
    """The interpolation the ACC metric uses, pinned at the two edge cases.

    A monthly field sits at the *midpoint* of its month, so at that instant the
    weight is entirely on it; and December interpolates forwards into January
    rather than off the end of the array.
    """
    from oceanarches.metrics.masked_metrics import month_interpolation_weights

    mid_july = np.datetime64("2021-07-16T12:00:00").astype("datetime64[s]").astype(int)
    assert month_interpolation_weights(mid_july) == (6, 7, 0.0)

    before, after, alpha = month_interpolation_weights(
        np.datetime64("2021-12-25T00:00:00").astype("datetime64[s]").astype(int)
    )
    assert (before, after) == (11, 0)
    assert 0.0 < alpha < 1.0


def _write_climatology(path: Path, module: "StubModule") -> Path:
    """A 12-month climatology whose month ``m`` is filled with the value ``m``."""
    surface = np.stack(
        [np.full((len(module.surface_variables), N_LAT, N_LON), month) for month in range(12)]
    ).astype("float32")
    level = np.stack(
        [
            np.full((len(module.level_variables), N_DEPTH, N_LAT, N_LON), month)
            for month in range(12)
        ]
    ).astype("float32")
    data = {
        name: (["month", "lat", "lon"], surface[:, i])
        for i, name in enumerate(module.surface_variables)
    }
    data.update(
        {
            name: (["month", "depth", "lat", "lon"], level[:, i])
            for i, name in enumerate(module.level_variables)
        }
    )
    dataset = xr.Dataset(
        data,
        coords=dict(
            month=np.arange(1, 13),
            lat=np.linspace(-89.5, 89.5, N_LAT),
            lon=np.linspace(0, 359, N_LON),
            depth=np.arange(N_DEPTH, dtype="float32"),
        ),
    )
    dataset.to_netcdf(path)
    return path


def test_climatology_baseline_interpolates_between_month_midpoints(tmp_path):
    """The baseline field is the monthly climatology, linearly interpolated.

    Each month of the synthetic file is filled with its own index, so the
    expected value at any instant is just the interpolation weight -- which makes
    a wrong interpolation impossible to hide.

    MUTANT: rounding to the nearest month (``alpha = round(alpha)``), the
    obvious "close enough" simplification. This test FAILS: halfway between the
    June and July midpoints the correct answer is 5.5 and the mutant gives 6.0.
    """
    module = StubModule()
    module.mask_surface = torch.ones_like(module.mask_surface)
    module.mask_level = torch.ones_like(module.mask_level)
    path = _write_climatology(tmp_path / "clim.nc", module)
    forecaster = baselines.ClimatologyForecast(module, climatology_path=path)

    # Midway between the June (2021-06-16T00:00) and July (2021-07-16T12:00)
    # midpoints. The lead is one day, so the *valid* time is what matters.
    midway = np.datetime64("2021-07-01T06:00:00") - np.timedelta64(24, "h")
    batch = {"timestamp": torch.tensor([int(midway.astype("datetime64[s]").astype(int))])}
    prediction = forecaster.predict(batch, 1)
    physical = module.denormalize_state(prediction)

    assert physical["surface"].shape[:2] == (1, 1)
    torch.testing.assert_close(
        physical["surface"][0, 0].mean(), torch.tensor(5.5), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        physical["level"][0, 0].mean(), torch.tensor(5.5), rtol=1e-4, atol=1e-4
    )


def test_climatology_baseline_lands_exactly_on_a_month_at_its_midpoint(tmp_path):
    module = StubModule()
    module.mask_surface = torch.ones_like(module.mask_surface)
    module.mask_level = torch.ones_like(module.mask_level)
    path = _write_climatology(tmp_path / "clim.nc", module)
    forecaster = baselines.ClimatologyForecast(module, climatology_path=path)

    valid = np.datetime64("2021-07-16T12:00:00") - np.timedelta64(24, "h")
    batch = {"timestamp": torch.tensor([int(valid.astype("datetime64[s]").astype(int))])}
    physical = module.denormalize_state(forecaster.predict(batch, 1))
    torch.testing.assert_close(
        physical["surface"][0, 0].mean(), torch.tensor(6.0), rtol=1e-4, atol=1e-4
    )


def test_build_forecasters_puts_the_model_first_and_can_drop_the_baselines():
    class Bare(StubModule):
        def parameters(self):
            yield torch.zeros(1)

    module = Bare()
    only_model = baselines.build_forecasters(module, include_baselines=False)
    assert [f.key for f in only_model] == ["model"]


# ---------------------------------------------------------------------------
# The scoring loop
# ---------------------------------------------------------------------------
def test_every_forecaster_is_scored_against_the_identical_target():
    """The single most important property of this pipeline.

    The truth is built once and the *same tensor* is handed to the model's
    metrics and to both baselines'.  If it were not, the reported skill would be
    partly a difference between two pieces of our own code.

    MUTANT: giving the baselines ``targets * 1.01`` instead of ``targets``
    (a 1% miscalibration, far too small to notice in any plot). This test FAILS
    against it on the first baseline comparison.
    """
    module = StubModule()
    batch = _batch(batch=2, iters=3)
    forecasters = [
        baselines.ModelForecast(module),
        baselines.PersistenceForecast(module),
    ]
    metrics = {f.key: {"m": RecordingMetric()} for f in forecasters}

    # A "model" that predicts the persistence state plus a constant, so the two
    # forecasters produce different predictions but must see one truth.
    class Constant(baselines.Forecaster):
        key, label, is_model = "model", "Model", True

        def predict(self, batch, iters):
            state = batch["state"]
            entries = {
                k: state[k].unsqueeze(1).expand(state[k].shape[0], iters, *state[k].shape[1:])
                + 0.5
                for k in state.keys()
            }
            return baselines.as_trajectory(entries, state["surface"].shape[0], iters)

    forecasters[0] = Constant()
    losses: dict[str, float] = {}
    rollout.score_batch(module, forecasters, batch, 3, metrics, losses)

    model_targets = metrics["model"]["m"].calls[0][0]
    baseline_targets = metrics["persistence"]["m"].calls[0][0]
    for group in ("surface", "level"):
        torch.testing.assert_close(model_targets[group], baseline_targets[group])
    # ...and the predictions genuinely differ, so the assertion above is not
    # vacuously true because nothing was computed.
    assert not torch.allclose(
        metrics["model"]["m"].calls[0][1]["surface"],
        metrics["persistence"]["m"].calls[0][1]["surface"],
    )


def test_scoring_reports_a_loss_for_every_forecaster_weighted_by_batch_size():
    module = StubModule()
    batch = _batch(batch=2, iters=3)
    forecasters = [baselines.ModelForecast(module), baselines.PersistenceForecast(module)]
    forecasters[0] = baselines.PersistenceForecast(module)
    forecasters[0].key, forecasters[0].is_model = "model", True
    metrics = {f.key: {} for f in forecasters}
    losses: dict[str, float] = {}
    rollout.score_batch(module, forecasters, batch, 3, metrics, losses)
    assert set(losses) == {"model", "persistence"}
    # Same forecaster twice -> the same loss, and it is scaled by the batch size.
    assert losses["model"] == pytest.approx(losses["persistence"])
    assert losses["model"] > 0


def test_metrics_see_denormalised_physical_values():
    """The metrics are documented to take physical units, not normalised ones."""
    module = StubModule()
    batch = _batch(batch=1, iters=2)
    forecaster = baselines.PersistenceForecast(module)
    metrics = {"persistence": {"m": RecordingMetric()}}
    rollout.score_batch(module, [forecaster], batch, 2, metrics, {})
    seen = metrics["persistence"]["m"].calls[0][1]["surface"]
    expected = batch["state"]["surface"] * module.state_std_surface + module.state_mean_surface
    torch.testing.assert_close(seen[:, 0], expected)


# ---------------------------------------------------------------------------
# Initialisation choice and the cache
# ---------------------------------------------------------------------------
def test_choose_initialisations_spread_covers_the_period():
    picks = rollout.choose_initialisations(100, 5, "spread")
    assert picks[0] == 0 and picks[-1] == 99
    assert picks == sorted(picks)
    assert len(picks) == 5


def test_choose_initialisations_first_is_dataset_order():
    assert rollout.choose_initialisations(100, 5, "first") == [0, 1, 2, 3, 4]


def test_choose_initialisations_clips_and_validates():
    assert rollout.choose_initialisations(3, 10, "first") == [0, 1, 2]
    assert rollout.choose_initialisations(10, 1, "spread") == [0]
    with pytest.raises(ValueError):
        rollout.choose_initialisations(0, 4, "spread")
    with pytest.raises(ValueError, match="spread"):
        rollout.choose_initialisations(10, 4, "banana")


def test_a_cache_for_a_different_question_is_not_reused(tmp_path):
    """A stale cache that silently answers a different question is the worst case.

    MUTANT: making ``RolloutSpec.matches`` return True unconditionally (the
    naive "the directory exists, so reuse it" cache). This test FAILS against it
    on all four mismatched fields.
    """
    spec = rollout.RolloutSpec(
        experiment="e",
        domain="test",
        lead_days=10,
        n_inits=8,
        selection="spread",
        save_depths="shallow",
        checkpoint="a.ckpt",
        inits=(0, 1),
        checkpoint_fingerprint="164:aaaa",
        config_hash="cccc",
        save_fields=True,
    )
    manifest = spec.as_manifest()
    manifest.update(metric_groups={}, saved_fields=False)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    assert rollout.load_cached(spec, tmp_path) is not None
    for field, value in (
        ("lead_days", 5),
        ("n_inits", 16),
        ("checkpoint", "b.ckpt"),
        ("domain", "holdout"),
        ("checkpoint_fingerprint", "164:bbbb"),
        ("config_hash", "dddd"),
        ("save_fields", False),
    ):
        import dataclasses

        other = dataclasses.replace(spec, **{field: value})
        assert rollout.load_cached(other, tmp_path) is None, field


def test_missing_cache_returns_none(tmp_path):
    spec = rollout.RolloutSpec("e", "test", 1, 1, "first", "shallow", "c", (0,))
    assert rollout.load_cached(spec, tmp_path) is None
    (tmp_path / "manifest.json").write_text("not json")
    assert rollout.load_cached(spec, tmp_path) is None


# ---------------------------------------------------------------------------
# RolloutSpec.from_manifest -- the reverse of as_manifest
# ---------------------------------------------------------------------------
def _fully_populated_spec() -> rollout.RolloutSpec:
    """A `RolloutSpec` with every field set to a value that is neither its
    class default nor equal to any other field's value.

    Built by walking `dataclasses.fields`/`typing.get_type_hints`, not typed
    out as a literal, so a field added to `RolloutSpec` in the future is
    populated here automatically -- which is what makes
    `test_from_manifest_round_trips_as_manifest_structurally` actually
    structural rather than merely exercising today's field list.
    """
    hints = typing.get_type_hints(rollout.RolloutSpec)
    kwargs = {}
    for index, f in enumerate(dataclasses.fields(rollout.RolloutSpec)):
        hint = hints.get(f.name)
        default = None if f.default is dataclasses.MISSING else f.default
        if hint is bool:
            kwargs[f.name] = not bool(default)
        elif hint is int:
            kwargs[f.name] = 1000 + index
        elif hint is str:
            kwargs[f.name] = f"distinct-{f.name}-{index}"
        else:  # tuple[int, ...] and anything else not-yet-imagined
            kwargs[f.name] = (index, index + 1, index + 2)
    return rollout.RolloutSpec(**kwargs)


def test_from_manifest_round_trips_as_manifest_structurally():
    """`from_manifest(spec.as_manifest())` must reproduce `spec` exactly,
    field for field -- checked by walking `dataclasses.fields`, not a
    hand-written list, so this test is not the thing that goes stale the next
    time the spec gains a field (the way `statistics_fingerprint` did).

    Every field of `_fully_populated_spec` is non-default AND distinct from
    every other field's value, so a field that quietly keeps its default
    across the round trip, or two fields that get swapped, both fail here. A
    test that typed out today's field names as a literal list would not catch
    a new field silently defaulting -- that is exactly the gap that let a
    notebook's own hand-written reconstruction miss `statistics_fingerprint`.

    MUTANT: `from_manifest` (or `as_manifest`) dropping a field, or the next
    field added to `RolloutSpec` not being threaded through both, fails here.
    """
    spec = _fully_populated_spec()
    rebuilt = rollout.RolloutSpec.from_manifest(spec.as_manifest())
    assert rebuilt == spec
    for f in dataclasses.fields(rollout.RolloutSpec):
        assert getattr(rebuilt, f.name) == getattr(spec, f.name), f.name


def test_from_manifest_missing_identity_field_raises():
    """An identity field (experiment, domain, lead_days, ...) has no default,
    and every manifest this project has ever written has recorded it -- a
    manifest missing one is damaged, not merely old, and must not be guessed
    at with a silent fallback.

    MUTANT: catching the KeyError internally and filling every missing field
    with `""` (the shape of the bug this project has already shipped once: a
    silent stand-in for a genuinely absent, load-bearing value) makes this
    pass.
    """
    manifest = _fully_populated_spec().as_manifest()
    del manifest["experiment"]
    with pytest.raises(KeyError, match="experiment"):
        rollout.RolloutSpec.from_manifest(manifest)


def test_from_manifest_missing_optional_fields_use_their_own_class_default():
    """Every field that has a default falls back to *that* default when the
    manifest lacks it -- not one hardcoded stand-in for all of them.
    `save_fields` must come back `True`, not `""`.

    Walks `dataclasses.fields`, so a future optional field is exercised here
    without a line being added.

    MUTANT: `from_manifest` defaulting every absent field to `""` (or to
    `None`) instead of reading `field.default` breaks this for `save_fields`
    and `inits`, whose real defaults are `True` and `()`.
    """
    manifest = _fully_populated_spec().as_manifest()
    for f in dataclasses.fields(rollout.RolloutSpec):
        if f.default is dataclasses.MISSING:
            continue
        trimmed = {k: v for k, v in manifest.items() if k != f.name}
        rebuilt = rollout.RolloutSpec.from_manifest(trimmed)
        assert getattr(rebuilt, f.name) == f.default, f.name


def test_from_manifest_missing_statistics_fingerprint_cannot_pass_as_a_match(tmp_path):
    """A manifest written before `statistics_fingerprint` existed must come
    back from `from_manifest` as the class's own declared default (`""`), not
    as a guess that happens to equal whatever a caller is comparing against --
    that guess is exactly the shape of bug this project already shipped once
    (the cache keyed on the checkpoint's file name, so retraining under the
    same name served the previous model's scores).

    Checked at both levels: the field value itself, and the practical
    consequence -- `load_cached` given a spec with today's real fingerprint
    must still refuse this cache, on the same terms as
    `test_a_cache_written_before_the_statistics_were_keyed_is_not_reused`.

    MUTANT: defaulting the missing field to `rollout.statistics_fingerprint()`
    (today's real digest) instead of the class default of `""` makes the
    final assertion below pass a stale cache off as current.
    """
    run = _fake_run(tmp_path / "run", b"weights")
    real_spec = dataclasses.replace(_spec_for(run), statistics_fingerprint="3:the-real-one")
    manifest = real_spec.as_manifest()
    del manifest["statistics_fingerprint"]

    rebuilt = rollout.RolloutSpec.from_manifest(manifest)
    assert rebuilt.statistics_fingerprint == ""

    directory = tmp_path / "cache"
    directory.mkdir()
    manifest.update(losses={}, labels={}, metric_groups={}, n_samples=1)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    assert rollout.load_cached(real_spec, directory) is None


def _fake_run(root: Path, weights: bytes, config: str = "module: {}\n") -> Path:
    """A minimal ``modelstore/<name>`` directory: one checkpoint and a config."""
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints" / "checkpoint_global_step=4500.ckpt").write_bytes(weights)
    (root / "config.yaml").write_text(config)
    return root


def _spec_for(run: Path, **overrides) -> rollout.RolloutSpec:
    fields = dict(
        experiment="my_run",
        domain="test",
        lead_days=10,
        n_inits=16,
        selection="spread",
        save_depths="shallow",
        save_fields=True,
        inits=(0, 1),
    )
    fields.update(overrides)
    # An explicit digest: these tests are about the checkpoint half of the key,
    # and hashing the real oceanarches/stats/ would make them depend on a 95 MB
    # artefact that may not exist in a fresh clone. The auto-filled default has
    # its own test.
    return rollout.RolloutSpec.with_checkpoint(
        rollout.checkpoint_identity(run), statistics_digest="3:testdigest", **fields
    )


def _write_manifest(directory: Path, spec: rollout.RolloutSpec) -> None:
    """A cache directory that ``load_cached`` will accept for ``spec``."""
    manifest = spec.as_manifest()
    manifest.update(metric_groups={}, saved_fields=spec.save_fields)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if spec.save_fields:
        for name in ("predictions.zarr", "targets.zarr"):
            (directory / name).mkdir(exist_ok=True)


def test_retraining_a_run_invalidates_its_cache_even_under_the_same_file_name(tmp_path):
    """The failure this pipeline could not be allowed to have.

    ``make train NAME=my_run`` writes ``checkpoint_global_step=4500.ckpt``.
    Retraining ``my_run`` with the same ``max_steps`` writes *different weights
    to a file with the identical name*, and then ``make eval NAME=my_run``
    reports the previous model's scores from the cache with the reassuring line
    "Reusing cached rollout". That is silently wrong in the one direction that
    matters: it tells a participant their change did nothing.

    MUTANT: keying ``RolloutSpec`` on the checkpoint's file name alone -- i.e.
    dropping ``checkpoint_fingerprint`` and ``config_hash`` from ``as_manifest``,
    which is what this pipeline shipped first. This test FAILS on the
    ``load_cached(...) is None`` assertion, because the file name is unchanged.
    """
    run = _fake_run(tmp_path / "my_run", b"weights at step 1500")
    cache = tmp_path / "evalstore" / "lead10d"

    before = _spec_for(run)
    _write_manifest(cache, before)
    assert rollout.load_cached(before, cache) is not None

    # Retrain: same file name, same directory, different weights.
    (run / "checkpoints" / "checkpoint_global_step=4500.ckpt").write_bytes(b"weights at step 4500")
    after = _spec_for(run)
    assert after.checkpoint == before.checkpoint == "checkpoint_global_step=4500.ckpt"
    assert after.checkpoint_fingerprint != before.checkpoint_fingerprint
    assert rollout.load_cached(after, cache) is None

    # And the same for the other half of what load_module reads.
    _write_manifest(cache, after)
    assert rollout.load_cached(after, cache) is not None
    (run / "config.yaml").write_text("module: {lr: 0.001}\n")
    assert rollout.load_cached(_spec_for(run), cache) is None


def test_a_checkpoint_that_only_moved_keeps_its_cache(tmp_path):
    """The other half of a cache key: it must not invalidate for nothing.

    A key made of the modification time would throw away a good 226-second
    evaluation whenever a checkpoint was copied, touched or restored from a
    backup. The contents are what the scores depend on, so the contents are the
    key.
    """
    import os
    import shutil

    run = _fake_run(tmp_path / "my_run", b"weights")
    cache = tmp_path / "evalstore" / "lead10d"
    _write_manifest(cache, _spec_for(run))

    copy = shutil.copytree(run, tmp_path / "copied_run")
    os.utime(copy / "checkpoints" / "checkpoint_global_step=4500.ckpt", (1, 1))
    assert rollout.load_cached(_spec_for(copy), cache) is not None


def test_a_cache_built_with_skip_fields_is_not_reused_when_the_maps_are_wanted(tmp_path):
    """``--skip-fields`` answers a smaller question and must not answer the big one.

    Without ``save_fields`` in the key, a metrics-only run poisons the cache: the
    next full ``make eval`` reuses it and quietly produces 4 figures where it
    produced 10, because the map, polar and spectra figures need
    ``predictions.zarr`` and there is none.

    MUTANT: dropping ``save_fields`` from ``as_manifest``. This test FAILS on the
    ``is None`` assertion.
    """
    run = _fake_run(tmp_path / "my_run", b"weights")
    cache = tmp_path / "evalstore" / "lead10d"
    _write_manifest(cache, _spec_for(run, save_fields=False))

    assert rollout.load_cached(_spec_for(run, save_fields=False), cache) is not None
    assert rollout.load_cached(_spec_for(run, save_fields=True), cache) is None


def test_checkpoint_identity_reports_missing_pieces_instead_of_raising(tmp_path):
    identity = rollout.checkpoint_identity(tmp_path / "nothing_here")
    assert identity.name == "unknown"
    assert identity.fingerprint == "missing" and identity.config_hash == "missing"


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def _luminance(rgba) -> float:
    return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]


def test_sequential_colormap_is_one_hue_and_monotone_in_lightness():
    cmap = plots.sequential_colormap()
    lightness = [_luminance(cmap(x)) for x in np.linspace(0, 1, 9)]
    assert all(a > b for a, b in zip(lightness, lightness[1:])), lightness


def test_diverging_colormap_is_cool_low_warm_high_with_a_neutral_middle():
    """The orientation of the error colour scale.

    Every error map in the report is read through this: cool means the model is
    *below* the truth, warm means above.  Reversing it silently inverts the
    meaning of every error panel and every scorecard cell, and nothing else in
    the pipeline would notice.

    MUTANT: reversing the colour list in ``diverging_colormap``. This test FAILS
    on both the low-end and high-end assertions.
    """
    cmap = plots.diverging_colormap()
    low = cmap(0.0)
    middle = cmap(0.5)
    high = cmap(1.0)
    # Cool low: more blue than red. Warm high: more red than blue.
    assert low[2] > low[0], low
    assert high[0] > high[2], high
    # Neutral middle: the three channels are within a few percent of each other,
    # so "no difference" reads as no colour.
    assert max(middle[:3]) - min(middle[:3]) < 0.05, middle
    # ...and the middle is light, so zero recedes rather than shouting.
    assert _luminance(middle) > 0.8


def test_the_two_arms_of_the_diverging_map_are_perceptually_symmetric():
    """Equal distances from zero must be equally loud, or the map lies.

    The warm arm is generated from the cool one in OKLCh, so their lightness
    profiles have to match step for step.
    """
    cmap = plots.diverging_colormap()
    for offset in (0.1, 0.2, 0.3, 0.45):
        cool = _luminance(cmap(0.5 - offset))
        warm = _luminance(cmap(0.5 + offset))
        assert abs(cool - warm) < 0.06, (offset, cool, warm)


def test_land_is_neither_zero_nor_a_data_colour():
    cmap = plots.sequential_colormap()
    land = cmap(np.nan)
    assert tuple(round(c, 4) for c in land[:3]) == pytest.approx(
        tuple(round(c, 4) for c in _hex_rgb(plots.LAND)), abs=1e-3
    )
    # And it is not the value the ramp gives to 0.
    assert not np.allclose(land[:3], cmap(0.0)[:3], atol=0.02)


def _hex_rgb(colour: str) -> tuple[float, float, float]:
    text = colour.lstrip("#")
    return tuple(int(text[i : i + 2], 16) / 255 for i in (0, 2, 4))


def test_identity_colours_are_distinct_under_simulated_colour_blindness():
    """Every pair of forecast colours clears the documented separation floors.

    Runs the same computation as the data-viz palette validator: OKLab distance
    x100 under Machado-Oliveira-Fernandes protanopia and deuteranopia at
    severity 1.0, over *all* pairs.
    """
    machado = {
        "protan": [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ],
        "deutan": [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ],
    }

    def simulate(colour, matrix):
        linear = [plots._srgb_to_linear(c) for c in _hex_rgb(colour)]
        return [
            min(1.0, max(0.0, sum(matrix[i][j] * linear[j] for j in range(3)))) for i in range(3)
        ]

    def oklab_of(linear):
        r, g, b = linear
        lc = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
        mc = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
        sc = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
        return (
            0.2104542553 * lc + 0.7936177850 * mc - 0.0040720468 * sc,
            1.9779984951 * lc - 2.4285922050 * mc + 0.4505937099 * sc,
            0.0259040371 * lc + 0.7827717662 * mc - 0.8086757660 * sc,
        )

    identity = [
        plots.SERIES_COLOURS[k] for k in ("model", "persistence", "climatology", "variant")
    ]
    for i, first in enumerate(identity):
        for second in identity[i + 1 :]:
            normal = 100 * math.dist(
                oklab_of([plots._srgb_to_linear(c) for c in _hex_rgb(first)]),
                oklab_of([plots._srgb_to_linear(c) for c in _hex_rgb(second)]),
            )
            assert normal >= 15.0, (first, second, normal)
            for kind, matrix in machado.items():
                distance = 100 * math.dist(
                    oklab_of(simulate(first, matrix)), oklab_of(simulate(second, matrix))
                )
                assert distance >= 8.0, (kind, first, second, distance)


# ---------------------------------------------------------------------------
# Labelling and the numbers behind the figures
# ---------------------------------------------------------------------------
def test_parse_metric_variable_round_trips_the_metric_labels():
    assert plots.parse_metric_variable("thetao0m") == ("thetao", 0.0)
    assert plots.parse_metric_variable("so1684m") == ("so", 1684.0)
    assert plots.parse_metric_variable("siconc") == ("siconc", None)
    # A name that merely ends in a digit and an m must not be misread.
    assert plots.parse_metric_variable("mlotst") == ("mlotst", None)


def test_variable_labels_always_carry_a_unit():
    for label in ("thetao0m", "siconc", "zos", "so1684m", "mlotst"):
        assert plots.variable_units(label), label
        assert "[" in plots.variable_label(label), label


def test_relative_error_is_positive_when_the_model_is_worse():
    """Sign convention: warm means more error.

    MUTANT: the skill-score form ``1 - model/reference``. This test FAILS on
    both assertions, and without it the scorecard would paint a model that is
    *worse* than persistence in the same blue as one that is better.
    """
    worse = plots.relative_error(np.array([1.2]), np.array([1.0]))
    better = plots.relative_error(np.array([0.8]), np.array([1.0]))
    assert worse[0] == pytest.approx(0.2)
    assert better[0] == pytest.approx(-0.2)
    assert np.isnan(plots.relative_error(np.array([1.0]), np.array([0.0]))[0])


def test_scorecard_table_is_the_figure_in_numbers(synthetic_result):
    rows, lead, matrix = plots.scorecard_table(synthetic_result)
    assert rows and matrix.shape == (len(rows), len(lead))
    # The fixture's persistence is 1.2x the model everywhere, so every cell is
    # 1/1.2 - 1 = -16.7%.
    finite = matrix[np.isfinite(matrix)]
    assert np.allclose(finite, 1 / 1.2 - 1, atol=1e-9)
    # Rows follow the canonical variable order, not the alphabet.
    assert rows[0] == "zos"


def test_headline_table_flags_a_model_that_loses_to_persistence(synthetic_result):
    header, rows = report.headline_table(synthetic_result, day=1)
    assert "Persistence" in header and "Climatology" in header
    assert all("(better)" in row[-1] for row in rows)

    # Make persistence better than the model and the verdict must flip.
    for group in synthetic_result.metrics["persistence"].values():
        for name in group.data_vars:
            group[name] = group[name] * 0.5
    _, rows = report.headline_table(synthetic_result, day=1)
    assert all("(WORSE)" in row[-1] for row in rows)


def test_forecast_horizon_finds_the_crossover_with_climatology(synthetic_result):
    assert forecast_is_beyond(synthetic_result)
    # Push the model's error above climatology from lead 3 onwards.
    group = synthetic_result.metrics["model"]["glorys_deterministic_metrics"]
    group["thetao0m"][:, 2:] = 1e6
    assert report.forecast_horizon(synthetic_result, "thetao0m") == "3 days"


def forecast_is_beyond(result) -> bool:
    return report.forecast_horizon(result, "thetao0m").startswith(">")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _axes_of(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 5_000, path


def test_the_headline_figure_renders_with_labelled_axes(synthetic_result, tmp_path):
    """No unlabelled axes: every panel names its quantity and its unit.

    Checked on the live axes objects rather than on the PNG, which is the only
    way to assert it rather than to look at it.
    """
    import matplotlib.pyplot as plt

    path = plots.plot_rmse_vs_lead(synthetic_result, tmp_path / "rmse.png")
    _axes_of(path)

    # Rebuild without saving so the axes can be inspected.
    plots.apply_theme()
    figure_numbers_before = set(plt.get_fignums())
    plots.plot_rmse_vs_lead(synthetic_result, tmp_path / "rmse2.png")
    assert set(plt.get_fignums()) == figure_numbers_before  # figures are closed


def test_every_figure_axis_has_a_label(synthetic_result, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    captured = []
    original = plt.Figure.savefig

    def spy(self, *args, **kwargs):
        captured.append([(ax.get_xlabel(), ax.get_ylabel()) for ax in self.axes])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    plots.plot_rmse_vs_lead(synthetic_result, tmp_path / "a.png")
    plots.plot_seaice_vs_lead(synthetic_result, tmp_path / "b.png")
    assert captured
    for figure in captured:
        # Every data axes (i.e. one that has any label at all) must have both.
        labelled = [pair for pair in figure if pair[0] or pair[1]]
        assert labelled
        for xlabel, ylabel in labelled:
            assert xlabel and ylabel, (xlabel, ylabel)


def test_scorecard_and_hovmoller_render(synthetic_result, tmp_path):
    _axes_of(plots.plot_scorecard(synthetic_result, tmp_path / "scorecard.png"))
    _axes_of(plots.plot_depth_hovmoller(synthetic_result, tmp_path / "hov.png"))


def test_overlay_hook_takes_two_result_sets(synthetic_result, tmp_path):
    """The hook Task 8 calls with a coupled and an uncoupled result."""
    import copy

    other = copy.deepcopy(synthetic_result)
    for group in other.metrics["model"].values():
        for name in group.data_vars:
            group[name] = group[name] * 0.9
    path = plots.plot_overlay(
        {"Uncoupled": synthetic_result, "Coupled": other},
        tmp_path / "overlay.png",
        variables=["thetao0m", "siconc"],
    )
    _axes_of(path)


def test_render_all_skips_field_figures_when_there_is_no_cached_rollout(
    synthetic_result, tmp_path
):
    """The fixture has no cached fields, so the maps and the spectra cannot be
    drawn; the curve figures must still appear."""
    written = plots.render_all(synthetic_result, out_dir=tmp_path / "figures", dpi=70)
    names = {p.name for p in written}
    assert {"01_rmse_vs_lead.png", "02_scorecard.png", "05_depth_hovmoller.png"} <= names
    assert not any("map" in n or "polar" in n or "spectra" in n for n in names)


def test_render_all_names_every_figure_it_could_not_draw(synthetic_result, tmp_path):
    """A figure that vanishes without a word is how a run silently loses 6 of 10.

    After a ``--skip-fields`` run the three map triptychs, the polar figure and
    the spectra cannot be drawn. Before this was fixed they were dropped by a
    bare ``if predictions_path.exists()`` that returned before anything warned,
    so a full ``make eval`` reusing that cache produced 4 figures where the
    previous one produced 10, with nothing on screen to say so.

    MUTANT: guarding the field figures with a plain ``if ...exists():`` and no
    ``else``. This test FAILS: no warning is raised at all.
    """
    with pytest.warns(UserWarning) as caught:
        plots.render_all(synthetic_result, out_dir=tmp_path / "figures", dpi=70)
    said = " ".join(str(w.message) for w in caught)
    for name in ("map_thetao", "map_siconc", "map_zos", "seaice_polar", "power_spectra"):
        assert name in said, name
    assert "--skip-fields" in said


def test_render_all_warns_and_continues_when_one_figure_fails(synthetic_result, tmp_path):
    """A single bad figure must not abort an evaluation.

    A participant staring at a traceback learns nothing about their model, so
    ``render_all`` warns and carries on. Here the sea-ice metrics are removed,
    which makes exactly one figure impossible.
    """
    for group in synthetic_result.metrics.values():
        group.pop("glorys_seaice_metrics", None)
    with pytest.warns(UserWarning, match="seaice_vs_lead"):
        written = plots.render_all(synthetic_result, out_dir=tmp_path / "figures", dpi=70)
    names = {p.name for p in written}
    assert "03_seaice_vs_lead.png" not in names
    assert "01_rmse_vs_lead.png" in names


def test_power_spectrum_refuses_a_grid_it_cannot_transform():
    pytest.importorskip("pyshtools")
    with pytest.warns(UserWarning, match="nlon"):
        assert plots.spherical_power_spectrum(np.zeros((10, 10))) is None


def test_power_spectrum_of_a_constant_field_is_all_in_degree_zero():
    pytest.importorskip("pyshtools")
    spectrum = plots.spherical_power_spectrum(np.full((16, 32), 3.0))
    assert spectrum is not None
    assert spectrum[0] == pytest.approx(9.0, rel=1e-6)
    assert np.allclose(spectrum[1:], 0.0, atol=1e-8)


def _write_free_rollout(directory: Path, n_lead: int = 3) -> rollout.RolloutResult:
    """A cached free-running rollout: SST and sea ice, land as NaN, on disk."""
    lat = np.linspace(-89.5, 89.5, N_LAT)
    lon = np.linspace(0, 359, N_LON)
    leads = np.array([np.timedelta64(24 * (i + 1), "h") for i in range(n_lead)])
    times = np.array([np.datetime64("2021-06-20T12:00:00")])
    land = np.zeros((N_LAT, N_LON), dtype=bool)
    land[0, :] = True  # a row of land, to prove it is excluded everywhere

    def field(values):
        block = np.broadcast_to(values, (1, n_lead, N_LAT, N_LON)).astype("float32").copy()
        block[:, :, land] = np.nan
        return block

    sst_truth = field(10.0)
    # The model drifts away from the truth: what the figure exists to show.
    sst_model = field(10.0) - np.arange(n_lead, dtype="float32")[None, :, None, None]
    ice = field(0.5)

    for name, sst in (("targets", sst_truth), ("predictions", sst_model)):
        xr.Dataset(
            {
                "thetao": (
                    ["time", "prediction_timedelta", "depth", "lat", "lon"],
                    sst[:, :, None],
                ),
                "siconc": (["time", "prediction_timedelta", "lat", "lon"], ice),
            },
            coords=dict(time=times, prediction_timedelta=leads, depth=[0.494], lat=lat, lon=lon),
        ).to_zarr(directory / f"{name}.zarr", mode="w")

    return rollout.RolloutResult(
        spec=rollout.RolloutSpec(
            experiment="synthetic",
            domain="test",
            lead_days=n_lead,
            n_inits=1,
            selection="spread",
            save_depths="shallow",
            checkpoint="ckpt",
            inits=(0,),
        ),
        directory=directory,
        n_samples=1,
    )


def _write_free_climatology(path: Path) -> Path:
    """Month ``m`` of ``thetao`` holds the value ``m``; ``siconc`` is 0.5 all year.

    Land carries 999 rather than NaN, which is the point: if the climatology is
    not masked with the truth's own land mask, the SST mean and the ice extent
    are both wildly wrong and no assertion below can pass.
    """
    months = np.arange(12, dtype="float32")
    thetao = np.broadcast_to(months[:, None, None, None], (12, 1, N_LAT, N_LON)).copy()
    siconc = np.full((12, N_LAT, N_LON), 0.5, dtype="float32")
    thetao[:, :, 0, :] = 999.0
    siconc[:, 0, :] = 999.0
    xr.Dataset(
        {
            "thetao": (["month", "depth", "lat", "lon"], thetao),
            "siconc": (["month", "lat", "lon"], siconc),
        },
        coords=dict(
            month=np.arange(1, 13),
            depth=[0.494],
            lat=np.linspace(-89.5, 89.5, N_LAT),
            lon=np.linspace(0, 359, N_LON),
        ),
    ).to_netcdf(path)
    return path


def test_the_drift_figure_carries_a_climatology_curve(tmp_path, monkeypatch):
    """Figure 07 needs three lines, not two.

    Drift towards the climatology is a forecast running out of information;
    drift away from both is a model that has left the attractor. With only the
    truth on the axes a beginner cannot tell those apart -- which is exactly the
    judgement the brief asks this figure to support.

    MUTANT: dropping ``climatology`` from ``free_running_series``. This test
    FAILS on the missing key.
    """
    from oceanarches import paths
    from oceanarches.metrics.masked_metrics import month_interpolation_weights

    free = _write_free_rollout(tmp_path)
    clim = _write_free_climatology(tmp_path / "clim.nc")
    monkeypatch.setattr(paths, "climatology_file", lambda: clim)

    series = plots.free_running_series(free)

    # SST: every ocean cell of month m holds m, so the expected curve is just
    # the interpolation weight at each valid day -- an independent computation.
    init = np.datetime64("2021-06-20T12:00:00")
    expected = []
    for step in range(len(series["days"])):
        valid = init + np.timedelta64(24 * (step + 1), "h")
        before, after, alpha = month_interpolation_weights(valid)
        expected.append((1 - alpha) * before + alpha * after)
    np.testing.assert_allclose(series["sst"]["climatology"], expected, rtol=1e-5)
    # Land carries 999 in the climatology file and must not reach the mean.
    assert np.all(series["sst"]["climatology"] < 12)

    # Sea ice: the climatology holds the same 0.5 concentration as the truth, so
    # its extent must come out *identical* -- one area integration, not two.
    for key in ("extent_nh", "extent_sh"):
        np.testing.assert_allclose(series[key]["climatology"], series[key]["truth"], rtol=1e-6)
        assert np.all(series[key]["climatology"] > 0)


def test_the_drift_figure_plots_all_three_series(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from oceanarches import paths

    free = _write_free_rollout(tmp_path)
    monkeypatch.setattr(
        paths, "climatology_file", lambda: _write_free_climatology(tmp_path / "clim.nc")
    )

    captured = []
    original = plt.Figure.savefig

    def spy(self, *args, **kwargs):
        captured.append(
            [
                ([line.get_color() for line in ax.get_lines()], ax.get_xlabel(), ax.get_ylabel())
                for ax in self.axes
            ]
        )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    path = plots.plot_free_timeseries(free, tmp_path / "07.png")
    assert path.exists()
    (panels,) = captured
    assert len(panels) == 3
    for colours, _, ylabel in panels:
        assert plots.SERIES_COLOURS["climatology"] in colours, colours
        assert (
            plots.SERIES_COLOURS["truth"] in colours and plots.SERIES_COLOURS["model"] in colours
        )
        assert ylabel


def test_the_extent_bias_axis_states_which_sign_means_too_much_ice(
    synthetic_result, tmp_path, monkeypatch
):
    """A signed quantity with an unsigned label is a coin toss.

    ``extentbias`` is extent(forecast) - extent(truth), so positive means the
    model makes too much ice -- the headline defect of the shipped model. The
    axis used to say only "Area", which does not say that.

    MUTANT: restoring the shared ``ax.set_ylabel("Area [10^6 km^2]")``. This
    test FAILS on both assertions.
    """
    import matplotlib.pyplot as plt

    captured = []
    original = plt.Figure.savefig

    def spy(self, *args, **kwargs):
        captured.append([(ax.get_ylabel(), [t.get_text() for t in ax.texts]) for ax in self.axes])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    plots.plot_seaice_vs_lead(synthetic_result, tmp_path / "03.png")

    (panels,) = captured
    bias_panels = [(label, texts) for label, texts in panels if "bias" in label.lower()]
    assert bias_panels, [label for label, _ in panels]
    for label, texts in bias_panels:
        assert "forecast - truth" in label and "10$^6$ km$^2$" in label
        assert any("positive = too much ice" in text for text in texts), texts


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------
def test_frames_are_padded_to_a_macroblock_without_losing_content():
    frame = np.full((37, 45, 3), 200, dtype=np.uint8)
    frame[0, 0] = [1, 2, 3]
    padded = animate._pad_to_macroblock(frame)
    assert padded.shape[0] % 16 == 0 and padded.shape[1] % 16 == 0
    assert padded.shape[0] >= 37 and padded.shape[1] >= 45
    np.testing.assert_array_equal(padded[:37, :45], frame)


def test_an_already_aligned_frame_is_returned_unchanged():
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    assert animate._pad_to_macroblock(frame) is frame


def test_write_animation_produces_a_playable_file(tmp_path):
    frames = [
        animate._pad_to_macroblock(
            np.full((32, 64, 3), 40 * i, dtype=np.uint8),
        )
        for i in range(1, 4)
    ]
    path = animate.write_animation(frames, tmp_path / "clip.mp4", fps=4)
    assert path.exists() and path.stat().st_size > 0
    assert path.suffix in (".mp4", ".gif")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def test_report_html_is_genuinely_self_contained(synthetic_result, tmp_path):
    """Every image must be a data: URI, or the file breaks the moment it moves.

    MUTANT: replacing ``_data_uri(path)`` with the relative path. This test
    FAILS on the ``src="data:`` assertion.
    """
    figure = tmp_path / "figures" / "01_rmse_vs_lead.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plots.plot_rmse_vs_lead(synthetic_result, figure)

    written = report.write_report(
        synthetic_result,
        out_dir=tmp_path,
        figures=[figure],
        timings={"total": 12.3},
        checkpoint="checkpoint_global_step=4500.ckpt",
    )
    markdown, html_file = written
    assert markdown.name == "report.md" and html_file.name == "report.html"

    text = html_file.read_text()
    assert 'src="data:image/png;base64,' in text
    assert 'src="figures/' not in text
    # The report answers the question it exists to answer, with the baselines.
    assert "Persistence" in text and "Climatology" in text
    assert "checkpoint_global_step=4500.ckpt" in text
    assert markdown.read_text().startswith("# Evaluation report")


def test_report_html_is_well_formed_and_every_asset_decodes(synthetic_result, tmp_path):
    """The nearest thing to opening it, without a browser.

    `report.html` has never been rendered by a browser engine -- there is none on
    JUPITER and the environment is pinned, so this is a real limitation and is
    stated as one in docs/06.  What CAN be checked mechanically is checked here:
    that every tag is closed in the right order, that every embedded asset
    decodes and that its bytes are the type it claims, and that nothing at all
    is fetched from the network (a report that needs a CDN is not a report you
    can read on a plane, and this one is meant to be emailed).

    MUTANT (all three caught, all three in `_data_uri`): dropping one byte from
    every payload; labelling every asset `video/mp4`; returning the path instead
    of embedding. The last one is also caught by
    `test_report_html_is_genuinely_self_contained`; the first two are not caught
    by anything else.
    """
    import base64
    from html.parser import HTMLParser

    figure = tmp_path / "figures" / "01_rmse_vs_lead.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plots.plot_rmse_vs_lead(synthetic_result, figure)
    # A PNG, a GIF and an mp4: one of each branch `_data_uri` and the <img> /
    # <video> choice can take. With figures alone this decoded a single PNG and
    # the mp4 branch never ran, so "every embedded asset decodes" was a promise
    # about one asset.
    media = tmp_path / "animations"
    media.mkdir()
    gif = _tiny_clip(media / "seaice_arctic.gif")
    mp4 = _tiny_clip(media / "sst_rollout.mp4")
    html_file = report.write_report(
        synthetic_result, out_dir=tmp_path, figures=[figure], animations=[gif, mp4]
    )[1]

    void = {"br", "hr", "img", "meta", "link", "input", "source", "col", "area", "base", "embed"}

    class Checker(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack, self.problems, self.sources = [], [], []

        def handle_starttag(self, tag, attrs):
            self.sources += [(tag, v) for k, v in attrs if k in ("src", "href") and v]
            if tag not in void:
                self.stack.append((tag, self.getpos()))

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.stack:
                self.problems.append(f"</{tag}> at {self.getpos()} closes nothing")
                return
            opened, position = self.stack.pop()
            if opened != tag:
                self.problems.append(
                    f"</{tag}> at {self.getpos()} closes <{opened}> opened at {position}"
                )

    checker = Checker()
    checker.feed(html_file.read_text(encoding="utf-8"))
    checker.close()
    assert not checker.problems, checker.problems
    assert not checker.stack, [f"<{t}> never closed" for t, _ in checker.stack]

    magic = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/gif": b"GIF8",
        "image/jpeg": b"\xff\xd8\xff",
    }
    embedded = 0
    seen_types: set[str] = set()
    for tag, value in checker.sources:
        assert not value.startswith(("http://", "https://", "//")), (
            f"{tag} fetches {value} from the network; report.html has to work offline"
        )
        if not value.startswith("data:"):
            continue
        embedded += 1
        head, _, payload = value.partition(",")
        mime = head[len("data:") : head.find(";")]
        seen_types.add(mime)
        blob = base64.b64decode(payload, validate=True)  # raises if it is not valid base64
        assert blob, f"{tag}: empty {mime} payload"
        if mime in magic:
            assert blob.startswith(magic[mime]), f"{tag} declares {mime}, bytes say otherwise"
        if mime == "video/mp4":
            assert b"ftyp" in blob[:32], "declared video/mp4 with no ftyp box"
    assert seen_types >= {"image/png", "image/gif", "video/mp4"}, (
        f"only {sorted(seen_types)} were embedded, so the other branches of the "
        "embedding code went untested and this checked less than it claims"
    )


def test_report_links_media_that_blows_the_embed_budget(synthetic_result, tmp_path):
    figure = tmp_path / "figures" / "01_rmse_vs_lead.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plots.plot_rmse_vs_lead(synthetic_result, figure)
    written = report.write_report(
        synthetic_result, out_dir=tmp_path, figures=[figure], embed_budget=10
    )
    text = written[1].read_text()
    assert "data:image/png" not in text
    assert "over the size budget" in text


def test_report_tables_are_the_table_view_for_the_figures(synthetic_result, tmp_path):
    """The relief for a below-contrast identity colour is that every number is
    also printed. Assert it is."""
    written = report.write_report(synthetic_result, out_dir=tmp_path)
    markdown = written[0].read_text()
    assert "| Variable | Unit | Model | Persistence | Climatology |" in markdown
    assert "Error relative to persistence, in full" in markdown
    assert "day 1 | day 2" in markdown


def test_the_drift_table_is_the_table_view_for_the_climatology_curve(
    synthetic_result, tmp_path, monkeypatch
):
    """Figure 07's climatology curve is aqua, which sits at 2.7:1 on the figure
    surface -- below the mark-contrast target. The documented relief in this
    project is the table view, so the drift table has to carry that curve.

    MUTANT: dropping the climatology column from the drift table. This test
    FAILS on the missing header.
    """
    from oceanarches import paths

    free_dir = tmp_path / "free90d"
    free_dir.mkdir()
    free = _write_free_rollout(free_dir)
    monkeypatch.setattr(
        paths, "climatology_file", lambda: _write_free_climatology(tmp_path / "clim.nc")
    )

    markdown_path, _ = report.write_report(synthetic_result, free=free, out_dir=tmp_path)
    markdown = markdown_path.read_text()
    assert "| Climatology day 3 |" in markdown
    assert "settles onto the climatology" in markdown


def _contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.1 contrast between two ``#rrggbb`` colours."""

    def relative_luminance(hex_colour: str) -> float:
        channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    light, dark = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (light + 0.05) / (dark + 0.05)


def test_figure_captions_stay_readable_in_dark_mode():
    """The captions sit on a card that stays light in both themes.

    The figures are PNGs on a light surface, so their card is pinned to #fcfcfb
    even in dark mode -- but the caption inside it used to inherit the dark
    theme's --ink-2 (#c3c2b7), which is 1.75:1 on that card. All 16 captions in
    the report were effectively invisible to a reader in dark mode.

    MUTANT: deleting the ``figcaption`` rule from the dark-mode block. This test
    FAILS at 1.75:1 against the 4.5:1 floor.
    """
    import re

    dark = report._CSS[report._CSS.index("@media (prefers-color-scheme: dark)") :]
    card = re.search(r"figure\s*{[^}]*background:\s*(#[0-9a-fA-F]{6})", dark)
    caption = re.search(r"figcaption\s*{[^}]*color:\s*(#[0-9a-fA-F]{6})", dark)
    assert card and caption, "the dark theme must pin both the card and its caption"
    ratio = _contrast_ratio(caption.group(1), card.group(1))
    assert ratio >= 4.5, f"{caption.group(1)} on {card.group(1)} is only {ratio:.2f}:1"


def _tiny_clip(path: Path) -> Path:
    """A real, decodable three-frame clip -- GIF or mp4, by extension.

    The mp4 branch used to write 64 zero bytes, on the grounds that "only the
    extension matters to the report". It does to the report; it does not to
    `test_report_html_is_well_formed_and_every_asset_decodes`, which checks that
    an asset declared `video/mp4` really is one. A placeholder makes that check
    unsatisfiable, and a check nobody can satisfy gets deleted rather than fixed.
    """
    import imageio.v2 as imageio

    frames = [np.full((16, 16, 3), 40 * i, dtype=np.uint8) for i in range(1, 4)]
    if path.suffix == ".gif":
        imageio.mimsave(path, frames, duration=200, loop=0)
    else:
        animate.write_animation(frames, path, fps=2)
        assert path.exists(), (
            f"{path} was not written; write_animation fell back to a GIF, which means "
            "no ffmpeg is reachable -- see `make doctor`"
        )
    return path


def test_a_gif_fallback_is_embedded_as_an_image_and_an_mp4_as_a_video(synthetic_result, tmp_path):
    """``write_animation`` falls back to a GIF wherever ffmpeg cannot be found.

    A GIF inside ``<video src="data:image/gif;...">`` is an element no browser
    decodes, so on exactly the machine where the fallback matters every
    animation would render as a black box; and ``![](clip.mp4)`` is a broken
    image in every markdown viewer there is.

    MUTANT: emitting ``<video>`` for every animation and ``![...]`` for every
    one in markdown -- what shipped. This test FAILS on the ``<img`` assertion
    and on the markdown link.
    """
    media = tmp_path / "animations"
    media.mkdir()
    gif = _tiny_clip(media / "seaice_arctic.gif")
    mp4 = _tiny_clip(media / "sst_rollout.mp4")

    markdown_path, html_path = report.write_report(
        synthetic_result, out_dir=tmp_path, animations=[gif, mp4]
    )
    text = html_path.read_text()
    assert '<img alt="Arctic sea-ice concentration' in text
    assert 'src="data:image/gif;base64,' in text
    assert text.count("<video") == 1  # the mp4, and only the mp4
    assert '<video controls loop muted playsinline src="data:video/mp4;base64,' in text

    markdown = markdown_path.read_text()
    assert "![Arctic sea-ice concentration" in markdown  # a GIF really is an image
    assert "](animations/seaice_arctic.gif)" in markdown
    assert "![Sea surface temperature rollout" not in markdown  # an mp4 is not
    assert "[Sea surface temperature rollout" in markdown
    assert "](animations/sst_rollout.mp4)" in markdown


# ---------------------------------------------------------------------------
# Opt-in: the pipeline against the Lightning module's own validation path
# ---------------------------------------------------------------------------
#
# This is the check the Task 7 brief calls the most valuable one -- that the
# numbers this pipeline reports are the numbers the module logs during
# validation -- and it is the one test here that needs a real checkpoint. It is
# therefore opt-in rather than deleted: the default suite stays fast and CPU
# only, and anyone who touches the scoring loop can run
#
#     OCEANARCHES_EVAL_CHECKPOINT=modelstore/task6_tiny \
#         .venv/bin/python -m pytest tests/test_evaluation.py -k lightning_module -q
#
# to prove the two paths still agree.
_CHECKPOINT_ENV = "OCEANARCHES_EVAL_CHECKPOINT"


@pytest.mark.skipif(
    not os.environ.get(_CHECKPOINT_ENV),
    reason=f"set {_CHECKPOINT_ENV}=modelstore/<run> to run the end-to-end agreement check",
)
def test_pipeline_agrees_with_the_lightning_modules_own_validation(tmp_path):
    """Score the same samples twice: through this pipeline, and through the module.

    ``OceanForecastModule._predict`` is exactly what ``validation_step`` calls.
    If the pipeline's persistence baseline or its denormalisation differed from
    the module's by so much as a scaling, this would catch it -- and a
    disagreement here means one of the two is wrong, not that the tolerance is
    too tight.
    """
    from geoarches.lightning_modules.base_module import load_module

    from oceanarches.evaluation.run_eval import build_dataset

    experiment = os.environ[_CHECKPOINT_ENV]
    n_samples = int(os.environ.get("OCEANARCHES_EVAL_SAMPLES", "8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module, cfg = load_module(experiment, device=device)
    module.eval()

    dataset = build_dataset(cfg, os.environ.get("OCEANARCHES_EVAL_DOMAIN"), 1)
    subset = torch.utils.data.Subset(dataset, list(range(n_samples)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=4, shuffle=False, collate_fn=rollout.collate_fn, num_workers=0
    )

    forecasters = baselines.build_forecasters(module)
    pipeline_metrics = {f.key: rollout.build_metrics(cfg, 1, device) for f in forecasters}
    pipeline_losses: dict[str, float] = {}

    module_metrics = rollout.build_metrics(cfg, 1, device)
    module_loss = 0.0
    seen = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            rollout.score_batch(module, forecasters, batch, 1, pipeline_metrics, pipeline_losses)
            # The module's own path, verbatim.
            loss, targets, preds = module._predict(batch, 1)
            module_loss += float(loss) * batch["timestamp"].shape[0]
            for metric in module_metrics.values():
                metric.update(targets, preds, timestamp=batch["timestamp"])
            seen += batch["timestamp"].shape[0]

    assert pipeline_losses["model"] / seen == pytest.approx(module_loss / seen, rel=1e-6)
    for name, metric in module_metrics.items():
        expected = metric.compute()
        got = pipeline_metrics["model"][name].compute()
        assert set(expected) == set(got)
        for label, value in expected.items():
            torch.testing.assert_close(
                got[label].float(), value.float(), rtol=1e-5, atol=1e-8, msg=label
            )


def test_the_report_prefers_the_models_own_reproduction_command(synthetic_result):
    """A model that is not one checkpoint has to say how to re-run itself.

    `run_eval --exp <name>` is right for a checkpoint under `modelstore/` and
    wrong for a coupled system, whose experiment name is an `evalstore/`
    directory holding no checkpoint at all. `build_report` therefore asks the
    module, and falls back to the old text when it has nothing to say.

    MUTANT: dropping the `evaluation_command` branch from `build_report` prints
    `run_eval --exp <name>` for the coupled case and this fails.
    """

    class Speaks:
        lead_time_hours = 24

        def evaluation_command(self, spec):
            return "the coupled command"

    built = report.build_report(synthetic_result, module=Speaks())
    text = "\n".join(str(block) for block in built.blocks)
    assert "the coupled command" in text
    assert f"--exp {synthetic_result.spec.experiment}" not in text

    plain = "\n".join(str(block) for block in report.build_report(synthetic_result).blocks)
    assert f"make eval NAME={synthetic_result.spec.experiment}" in plain


def test_a_report_scored_on_a_holdout_split_says_so_at_the_top(synthetic_result):
    """The report is the artefact that gets pasted into a slide.

    By then the command line that named the split is long gone, so a score from
    `holdout` -- or from one of the `ifs_forced_*` windows inside it, which is
    the only place `forcing=file` can run -- has to carry its own warning. The
    kit says it in six other places; this is the one a participant who skipped
    all six still reads.

    MUTANT: dropping the `holdout_caution` call from `build_report` fails this.
    """
    for domain, must_contain in (
        ("holdout", "Do not report them as a score"),
        ("ifs_forced_train", "plumbing demonstration"),
        ("ifs_forced_val", "plumbing demonstration"),
    ):
        result = dataclasses.replace(
            synthetic_result, spec=dataclasses.replace(synthetic_result.spec, domain=domain)
        )
        text = "\n".join(str(block) for block in report.build_report(result).blocks)
        assert must_contain in text, domain
        assert "holdout" in text.lower(), domain

    # ... and a normal split is not decorated with a warning it does not need.
    plain = "\n".join(str(block) for block in report.build_report(synthetic_result).blocks)
    assert synthetic_result.spec.domain == "test"
    assert "Do not report" not in plain
    assert "plumbing demonstration" not in plain


# ---------------------------------------------------------------------------
# `--exp`: the first thing a beginner gets wrong
# ---------------------------------------------------------------------------
def test_a_run_that_does_not_exist_lists_the_runs_that_do(fake_modelstore):
    """`make eval NAME=typo`, and plain `make eval` after `make train-tiny
    NAME=my_run` (both targets default `NAME` to `tiny`), used to end in
    `FileNotFoundError: <repo-root>/typo/config.yaml` -- a path under the
    repository root, with the `modelstore/` prefix silently dropped by
    `rollout.py`, naming no valid run.

    MUTANT: removing the `resolve_run(args.exp)` call from
    `load_forecast_system` brings the FileNotFoundError back and the
    `pytest.raises(SystemExit)` fails.
    """
    from oceanarches.evaluation.run_eval import resolve_run

    fake_modelstore("my_first_run", "ocean_tiny")
    with pytest.raises(SystemExit) as caught:
        resolve_run("my_frist_run")
    message = str(caught.value)
    assert "my_frist_run" in message
    assert "my_first_run" in message and "ocean_tiny" in message
    assert "make train-tiny" in message


def test_a_run_with_an_empty_checkpoints_directory_names_save_step_frequency(fake_modelstore):
    """A run killed before its first checkpoint gave `IndexError: list index out
    of range` from geoarches, with no mention of checkpoints at all.

    MUTANT: dropping the `if not checkpoints:` branch from `resolve_run` makes
    this return a path instead of raising.
    """
    from oceanarches.evaluation.run_eval import resolve_run

    root = fake_modelstore("killed")
    for ckpt in (root / "killed" / "checkpoints").glob("*.ckpt"):
        ckpt.unlink()
    with pytest.raises(SystemExit, match="save_step_frequency"):
        resolve_run("killed")


def test_a_run_without_a_config_says_where_it_went(fake_modelstore):
    """`log=False` trains and checkpoints happily and writes no config.yaml."""
    from oceanarches.evaluation.run_eval import resolve_run

    root = fake_modelstore("no_config")
    (root / "no_config" / "config.yaml").unlink()
    with pytest.raises(SystemExit, match="log=False"):
        resolve_run("no_config")


def test_a_run_given_as_a_path_resolves_to_that_path(fake_modelstore, tmp_path):
    """`--help` says --exp may be a path to a run directory; it must work."""
    from oceanarches.evaluation.run_eval import resolve_run

    root = fake_modelstore("elsewhere")
    assert resolve_run(str(root / "elsewhere")) == root / "elsewhere"


def test_an_absolute_exp_path_does_not_write_into_the_checkpoint_directory():
    """`evalstore / args.exp` discards the left operand when the right is absolute.

    So `--exp /somewhere/my_run` wrote report.html, the figures, the animations
    and the rollout cache into `/somewhere/my_run/` -- the *checkpoint*
    directory -- and said nothing. `--help` explicitly supports that form.

    MUTANT: making `output_name` return its argument unchanged (which is what
    `evaluate` used to do) fails the second assertion.
    """
    from pathlib import Path as _Path

    from oceanarches.evaluation.run_eval import output_name

    evalstore = _Path("evalstore")
    assert evalstore / output_name("my_run") == _Path("evalstore/my_run")
    assert evalstore / output_name("/scratch/models/my_run") == _Path("evalstore/my_run")
    assert evalstore / output_name("modelstore/my_run") == _Path("evalstore/my_run")


# ---------------------------------------------------------------------------
# Statistics provenance: what `make stats-quick` used to leave no trace of
# ---------------------------------------------------------------------------
def _sampled_provenance() -> "provenance.StatisticsProvenance":
    """What `make stats-quick` records: 20 dates, 3 of 33 years."""
    return provenance.StatisticsProvenance(
        stats_years="1993-2025",
        stats_n_dates=20,
        stats_n_years=33,
        stats_sampling="sampled",
        climatology_years="1993,2004,2015",
        climatology_n_years=3,
        climatology_n_years_available=33,
        climatology_sampling="sampled",
    )


def _full_provenance() -> "provenance.StatisticsProvenance":
    return provenance.StatisticsProvenance(
        stats_years="1993-2025",
        stats_n_dates=400,
        stats_n_years=33,
        stats_sampling="full",
        climatology_years="1993-2025",
        climatology_n_years=33,
        climatology_n_years_available=33,
        climatology_sampling="full",
    )


def test_a_report_built_on_sampled_statistics_says_so_above_its_first_table(synthetic_result):
    """`make stats-quick` used to leave no trace anywhere downstream.

    `make doctor` passed identically, and the report then printed
    five-significant-figure scorecards against a climatology built from three
    sampled years with nothing saying so -- a participant could beat a sampled
    baseline in front of a room and never know. This is the same treatment
    `holdout_caution` gives a split that must not be quoted, for the same
    reason: by the time the report is on a slide, the command that built the
    statistics is long out of the scrollback.

    MUTANT: dropping the `statistics.caution()` block from `build_report` leaves
    the sampled report indistinguishable from the full one and fails this.
    """
    sampled = "\n".join(
        str(block)
        for block in report.build_report(synthetic_result, statistics=_sampled_provenance()).blocks
    )
    assert "sampled statistics" in sampled
    assert "make stats-quick" in sampled and "make stats" in sampled
    # The numbers, so a reader can judge how sampled it was.
    assert "20 dates" in sampled and "3 of 33 years" in sampled
    # And why it matters: the climatology is one of the two scored baselines.
    assert "baselines" in sampled

    full = "\n".join(
        str(block)
        for block in report.build_report(synthetic_result, statistics=_full_provenance()).blocks
    )
    assert "sampled statistics" not in full

    # A report built with no provenance at all must not invent a warning either.
    plain = "\n".join(str(block) for block in report.build_report(synthetic_result).blocks)
    assert "sampled statistics" not in plain


def test_the_report_names_the_statistics_that_produced_its_numbers(synthetic_result):
    """Provenance rows in "How this was produced", and the *real* climatology
    years in the opening paragraph.

    That paragraph used to state "the 1993-2025 monthly mean" as a constant --
    a claim about a file the report had never read, and false for both
    `--years 1993-2018` and `--quick`.

    MUTANT: dropping `*statistics.rows()` from the provenance table fails the
    first assertion; restoring the hard-coded "1993-2025" fails the third.
    """
    text = "\n".join(
        str(block)
        for block in report.build_report(synthetic_result, statistics=_sampled_provenance()).blocks
    )
    assert "Normalisation statistics" in text
    assert "Climatology (also a scored baseline)" in text
    assert "1993,2004,2015" in text
    assert "1993-2025 monthly mean" not in text

    # Nothing known, nothing claimed: no rows rather than rows reading "not recorded".
    plain = "\n".join(str(block) for block in report.build_report(synthetic_result).blocks)
    assert "Normalisation statistics" not in plain


def _write_statistics(directory: Path, stats: dict, climatology_attrs: dict) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stats_file = directory / "glorys_1deg_stats.pt"
    climatology_file = directory / "glorys_1deg_climatology.nc"
    torch.save(stats, stats_file)
    xr.Dataset(attrs=climatology_attrs).to_netcdf(climatology_file)
    return stats_file, climatology_file


def test_a_quick_build_is_readable_as_sampled_from_the_artefacts_alone(tmp_path):
    """The provenance has to survive on disk, not only in the terminal.

    MUTANT: making `StatisticsProvenance.stats_sampled` and
    `climatology_sampled` return False fails the first two assertions.
    """
    quick = _write_statistics(
        tmp_path,
        {"n_dates": 20, "n_years": 33, "years": "1993-2025", "sampling": "sampled"},
        {"years": "1993,2004,2015", "n_years": 3, "n_years_available": 33, "sampling": "sampled"},
    )
    read = provenance.read_statistics_provenance(*quick)
    assert read.sampled and read.stats_sampled and read.climatology_sampled
    assert read.caution()
    assert read.unreadable == ()

    full = _write_statistics(
        tmp_path / "full",
        {"n_dates": 400, "n_years": 33, "years": "1993-2025", "sampling": "full"},
        {"years": "1993-2025", "n_years": 33, "n_years_available": 33, "sampling": "full"},
    )
    read_full = provenance.read_statistics_provenance(*full)
    assert not read_full.sampled
    assert read_full.caution() == ""


def test_statistics_written_before_this_provenance_existed_are_still_judged(tmp_path):
    """An artefact from an older `compute_stats.py` records `years` and
    `n_dates` and no `sampling` key at all. A quick build from back then is
    still a quick build, and the date count alone settles it.

    MUTANT: dropping the `0 < n_dates < DEFAULT_N_DATES` fallback from
    `stats_sampled` reads the old quick build as full and fails this.
    """
    old_quick = _write_statistics(
        tmp_path, {"n_dates": 20, "years": "1993-2025"}, {"years": "1993-2025"}
    )
    assert provenance.read_statistics_provenance(*old_quick).sampled

    old_full = _write_statistics(
        tmp_path / "full", {"n_dates": 400, "years": "1993-2025"}, {"years": "1993-2025"}
    )
    read = provenance.read_statistics_provenance(*old_full)
    assert not read.sampled
    # ... and it does not pretend to know what it was not told.
    assert read.climatology_sampling == ""


def test_unreadable_statistics_never_stop_an_evaluation(tmp_path):
    """This runs on the way to a 250-second evaluation. It reports, never raises."""
    read = provenance.read_statistics_provenance(tmp_path / "absent.pt", tmp_path / "absent.nc")
    assert set(read.unreadable) == {"normalisation statistics", "climatology"}
    assert not read.known
    assert read.rows()  # it does say the artefacts could not be read
    assert read.caution() == ""


def test_rebuilt_statistics_invalidate_the_cache(tmp_path):
    """The Critical this project already fixed once, in its other half.

    The cache used to be keyed on the checkpoint's *file name*, so retraining a
    run under the same name reported the previous model's scores -- 1.0167 read
    out as 0.7053. Rebuilding `oceanarches/stats/` is the same shape: the file
    names do not change and every number does. A participant runs
    `make stats-quick`, evaluates, later runs the full `make stats` because they
    now want to report the numbers, re-runs `make eval` -- and used to get the
    sampled baseline back, with a printed warning as the only defence.

    It has to be the *contents*: `--quick` and a full build differ in every
    number in the file, and two builds can carry the identical `years` label
    over a changed archive.

    MUTANT: dropping `statistics_fingerprint` from `RolloutSpec.as_manifest`
    serves the stale cache and the last assertion fails.
    """
    stats_file, climatology_file = _write_statistics(
        tmp_path / "stats",
        {"n_dates": 20, "n_years": 33, "years": "1993-2025", "sampling": "sampled"},
        {"years": "1993,2004,2015", "n_years": 3, "n_years_available": 33, "sampling": "sampled"},
    )
    artefacts = [stats_file, climatology_file]
    quick_digest = rollout.statistics_fingerprint(artefacts)

    run = _fake_run(tmp_path, b"weights")
    directory = tmp_path / "cache"
    directory.mkdir()
    spec = dataclasses.replace(_spec_for(run), statistics_fingerprint=quick_digest)
    _write_manifest(directory, spec)
    assert rollout.load_cached(spec, directory) is not None

    # `make stats`: same file names, same years label, every number different.
    _write_statistics(
        tmp_path / "stats",
        {"n_dates": 400, "n_years": 33, "years": "1993-2025", "sampling": "full"},
        {"years": "1993-2025", "n_years": 33, "n_years_available": 33, "sampling": "full"},
    )
    full_digest = rollout.statistics_fingerprint(artefacts)
    assert full_digest != quick_digest

    rebuilt = dataclasses.replace(spec, statistics_fingerprint=full_digest)
    assert rollout.load_cached(rebuilt, directory) is None, (
        "the cache answered with numbers built from statistics that no longer exist"
    )


def test_a_spec_cannot_be_built_without_the_statistics_it_depends_on(tmp_path, monkeypatch):
    """Forgetting the field is the failure being guarded against, so saying
    nothing must give the safe value rather than the empty one.

    MUTANT: defaulting `statistics_digest` to `""` instead of hashing makes the
    first assertion fail, and every cache portable across a rebuild again.
    """
    stats_file, climatology_file = _write_statistics(
        tmp_path / "stats", {"n_dates": 400}, {"years": "1993-2025"}
    )
    monkeypatch.setattr(rollout.paths, "stats_file", lambda: stats_file)
    monkeypatch.setattr(rollout.paths, "climatology_file", lambda: climatology_file)
    monkeypatch.setattr(rollout.paths, "masks_file", lambda: tmp_path / "stats" / "masks.nc")

    identity = rollout.CheckpointIdentity(name="c.ckpt", fingerprint="1:a", config_hash="b")
    spec = rollout.RolloutSpec.with_checkpoint(
        identity,
        experiment="e",
        domain="test",
        lead_days=1,
        n_inits=1,
        selection="spread",
        save_depths="shallow",
    )
    assert spec.statistics_fingerprint == rollout.statistics_fingerprint()
    assert spec.as_manifest()["statistics_fingerprint"] == spec.statistics_fingerprint

    # A missing artefact is a fact about the cache, not a crash: the spec is
    # built before anything has checked the artefacts are there.
    stats_file.unlink()
    assert rollout.statistics_fingerprint() != spec.statistics_fingerprint


def test_the_provenance_still_travels_with_a_cached_rollout(tmp_path):
    """The words, as opposed to the key: a report built from a cache has to say
    which statistics produced those numbers, and an older manifest that cannot
    say is not a crash."""
    run = _fake_run(tmp_path, b"weights")
    spec = _spec_for(run)
    directory = tmp_path / "cache"
    directory.mkdir()

    manifest = spec.as_manifest()
    manifest.update(
        losses={"model": 1.0},
        labels={"model": "Model"},
        metric_groups={},
        n_samples=1,
        statistics=_sampled_provenance().to_dict(),
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    cached = rollout.load_cached(spec, directory)
    assert cached is not None
    assert provenance.StatisticsProvenance.from_dict(cached.statistics).sampled

    older = spec.as_manifest()
    older.update(losses={"model": 1.0}, labels={"model": "Model"}, metric_groups={}, n_samples=1)
    (directory / "manifest.json").write_text(json.dumps(older))
    still_cached = rollout.load_cached(spec, directory)
    assert still_cached is not None
    assert still_cached.statistics == {}


def test_a_cache_written_before_the_statistics_were_keyed_is_not_reused(tmp_path):
    """Every cache in existence when this landed becomes one recompute.

    That is the intended price, asserted rather than assumed: a manifest that
    cannot say which statistics made its numbers is exactly the manifest that
    must not be trusted.
    """
    run = _fake_run(tmp_path, b"weights")
    spec = dataclasses.replace(_spec_for(run), statistics_fingerprint="3:abc")
    directory = tmp_path / "cache"
    directory.mkdir()
    old_manifest = {
        key: value for key, value in spec.as_manifest().items() if key != "statistics_fingerprint"
    }
    old_manifest.update(losses={}, labels={}, metric_groups={}, n_samples=1)
    (directory / "manifest.json").write_text(json.dumps(old_manifest))
    assert rollout.load_cached(spec, directory) is None


# ---------------------------------------------------------------------------
# The figures and the animations are cached too
# ---------------------------------------------------------------------------
def test_an_identical_second_render_is_served_from_the_cache(synthetic_result, tmp_path):
    """ "Just re-run the cell" cost ~200 s of a 240 s run: the rollout was
    cached and the figures and animations were not, so they re-rendered
    unconditionally.

    MUTANT: making `cached_render` always return None re-renders every time and
    the second `calls == 1` assertion fails.
    """
    from oceanarches.evaluation.run_eval import cached_or_render

    directory = tmp_path / "figures"
    key = render_cache.render_key(
        "figures", synthetic_result.spec, forecasters=synthetic_result.metrics, dpi=150
    )
    calls = []

    def render():
        calls.append(1)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "01_rmse_vs_lead.png").write_bytes(b"png")
        return [directory / "01_rmse_vs_lead.png"]

    assert cached_or_render("figures", directory, key, render) == [
        directory / "01_rmse_vs_lead.png"
    ]
    assert len(calls) == 1
    cached_or_render("figures", directory, key, render)
    assert len(calls) == 1, "the second render was not served from the cache"

    # --force ignores it, exactly as it ignores the rollout cache.
    cached_or_render("figures", directory, key, render, force=True)
    assert len(calls) == 2

    # A figure someone deleted is not a cache hit.
    (directory / "01_rmse_vs_lead.png").unlink()
    cached_or_render("figures", directory, key, render)
    assert len(calls) == 3


def test_the_render_key_changes_with_everything_that_changes_the_picture(synthetic_result):
    """A cache that answers a different question is worse than no cache.

    MUTANT: dropping `options` from `render_key`'s payload makes the dpi and the
    fps assertions fail; dropping `spec` makes the lead-days one fail.
    """
    spec = synthetic_result.spec
    base = render_cache.render_key("figures", spec, forecasters=["model"], dpi=150)
    assert base == render_cache.render_key("figures", spec, forecasters=["model"], dpi=150)

    different = {
        "dpi": render_cache.render_key("figures", spec, forecasters=["model"], dpi=300),
        "kind": render_cache.render_key("animations", spec, forecasters=["model"], dpi=150),
        "forecasters": render_cache.render_key(
            "figures", spec, forecasters=["model", "persistence"], dpi=150
        ),
        "lead_days": render_cache.render_key(
            "figures", dataclasses.replace(spec, lead_days=20), forecasters=["model"], dpi=150
        ),
        "checkpoint": render_cache.render_key(
            "figures",
            dataclasses.replace(spec, checkpoint_fingerprint="other"),
            forecasters=["model"],
            dpi=150,
        ),
        "free_rollout": render_cache.render_key(
            "figures", spec, free_spec=spec, forecasters=["model"], dpi=150
        ),
    }
    for what, key in different.items():
        assert key != base, f"{what} did not change the render key"


# ---------------------------------------------------------------------------
# Saying what is happening, and what it cost
# ---------------------------------------------------------------------------
def test_every_figure_is_named_as_it_lands(synthetic_result, tmp_path, capsys):
    """`make eval` printed nothing for ~200 s of a 240 s run and a participant
    killed it at 180 s believing it had hung.

    MUTANT: dropping the `print` from `render_all`'s `attempt` leaves the stage
    silent and fails this.
    """
    plots.render_all(synthetic_result, out_dir=tmp_path / "figures", skip_spectra=True)
    printed = capsys.readouterr().out
    assert "01_rmse_vs_lead.png" in printed and "02_scorecard.png" in printed
    assert "s)" in printed  # each line carries what it cost

    plots.render_all(
        synthetic_result, out_dir=tmp_path / "quiet", skip_spectra=True, progress=False
    )
    assert capsys.readouterr().out == ""


def test_each_animation_is_named_before_it_starts_not_after(tmp_path, capsys):
    """An animation takes tens of seconds. A line printed only when it finishes
    is a line that arrives after the participant has already decided the run
    has hung, so the name goes out first and the cost follows.

    MUTANT: moving the `animating ...` print to after the render makes the
    ordering assertion fail.
    """
    directory = tmp_path / "free"
    directory.mkdir()
    result = _write_free_rollout(directory)
    animate.render_all(result, out_dir=tmp_path / "animations", fps=2, dpi=40)
    printed = capsys.readouterr().out
    assert "animating sst" in printed
    assert printed.index("animating sst") < printed.index("animation sst_rollout")


def test_the_run_says_what_it_left_on_disk(tmp_path):
    """One `LEAD_DAYS=10` evaluation leaves ~750 MB in `evalstore/`, which is
    fine once and alarming after a dozen runs on a shared quota. The run ended
    on `Done in 238.4s` and said nothing about it.

    MUTANT: making `directory_size` return 0 fails the size assertions.
    """
    from oceanarches.evaluation.run_eval import cost_lines, directory_size, human_size

    evalstore = tmp_path / "evalstore"
    out_dir = evalstore / "my_run"
    (out_dir / "lead10d").mkdir(parents=True)
    (out_dir / "lead10d" / "predictions.zarr").write_bytes(b"x" * 3_000_000)
    (evalstore / "other_run").mkdir()
    (evalstore / "other_run" / "blob").write_bytes(b"x" * 1_000_000)

    assert directory_size(out_dir) == 3_000_000
    assert human_size(3_000_000) == "3.0 MB"
    assert human_size(750_000_000) == "750.0 MB"

    lines = "\n".join(cost_lines(out_dir, evalstore))
    assert "3.0 MB" in lines and str(out_dir) in lines
    assert "4.0 MB" in lines and "2 run(s)" in lines
    assert "--skip-fields" in lines


def test_the_closing_lines_name_the_cheap_path_and_what_it_measured():
    """`--skip-animations` existed in `--help` and in docs/06 and nowhere a
    participant re-running an evaluation would meet it.

    MUTANT: dropping the animations branch of `next_time_hint` fails the first
    two assertions.
    """
    from oceanarches.evaluation.run_eval import next_time_hint, stage_breakdown

    class Args:
        force = False
        skip_animations = False

    timings = {"load_module": 12.0, "rollout": 19.6, "figures": 61.2, "animations": 128.0}
    hint = next_time_hint(Args(), timings)
    assert "--skip-animations" in hint
    assert "128s" in hint  # what it actually cost here, not a general claim
    assert "reuses" in hint

    class Skipped(Args):
        skip_animations = True

    assert "--skip-animations" not in next_time_hint(Skipped(), {"rollout": 19.6})

    breakdown = stage_breakdown(timings)
    assert "rollout 19.6s" in breakdown and "animations 128.0s" in breakdown
    assert "free rollout" not in breakdown  # a stage that did not run is not listed


# ---------------------------------------------------------------------------
# The allocation, at the entry point the hydra guard cannot see
# ---------------------------------------------------------------------------
def test_the_evaluation_entry_point_warns_when_there_is_no_allocation():
    """`make eval` and `make couple` never reach hydra's task function, so the
    `StartupGuard` callback registered in `configs/config.yaml` -- which covers
    every training route -- covers none of the evaluation ones. Three
    participants ran evaluation off an allocation; one had a coupled eval die of
    `torch.OutOfMemoryError` against 85 of 98 GiB held by another job.

    The signal must be `SLURM_JOB_ID`, not CUDA visibility: the login node has a
    real card, so `torch.cuda.is_available()` is True there and says nothing
    about whether it is yours.

    MUTANT: making `allocation_complaint` return None unconditionally fails the
    first assertion; keying it on `torch.cuda.is_available()` instead of the
    environment cannot distinguish the first case from the second and fails one
    of them on any machine with a GPU.
    """
    from oceanarches.evaluation.run_eval import allocation_complaint

    complaint = allocation_complaint("cuda", env={}, hostname="jpbl-s02-02")
    assert complaint is not None
    assert "SLURM_JOB_ID is unset" in complaint
    assert "jpbl-s02-02" in complaint
    assert "srun" in complaint and "--ntasks=1" in complaint
    # The escape hatch has to be a flag this entry point actually takes.
    assert "--device cpu" in complaint and "cluster=local" not in complaint
    # ... and there is no cluster config here to name.
    assert "cluster=" not in complaint

    # Inside an allocation there is nothing to say.
    assert allocation_complaint("cuda", env={"SLURM_JOB_ID": "1289795"}) is None
    # ... and neither is there when no GPU was asked for. `--device cpu` is this
    # entry point's `cluster=local`.
    assert allocation_complaint("cpu", env={}, hostname="jpbl-s02-02") is None


def test_the_evaluation_guard_is_the_training_guard_and_not_a_second_one():
    """Two notions of "am I on an allocation" would be worse than one.

    The two texts are not byte-identical any more and should not be: `make eval`
    has no cluster config to name and its CPU escape is `--device cpu`, not
    `cluster=local`. Those two substitutions are exactly what
    `guards.allocation_warning` takes as arguments, so the test pins that the
    evaluation text *is* the training text with those two changes and nothing
    else -- a reimplementation would differ somewhere they do not cover.

    MUTANT: reimplementing the check inside `run_eval` -- rather than delegating
    to `guards.allocation_warning` -- fails the reconstruction.
    """
    from oceanarches import guards
    from oceanarches.evaluation.run_eval import allocation_complaint

    evaluation = allocation_complaint("cuda", env={}, hostname="host")
    training = guards.allocation_warning(env={}, hostname="host", cluster_name="jupiter_1gpu")

    assert evaluation is not None and training is not None
    assert evaluation != training
    assert evaluation == guards.allocation_warning(
        env={}, hostname="host", cluster_name=None, cpu_hint="--device cpu"
    )
    # Same words everywhere the two callers do not differ: the signal, the
    # symptoms and the srun block are one text.
    shared = "A visible GPU is not an allocated one."
    assert shared in evaluation and shared in training
    for line in guards.SRUN_LINES:
        assert line in evaluation and line in training


def test_the_evaluation_path_does_not_print_the_warning_twice():
    """`warn_without_allocation` delegates the printing; it must not also print.

    MUTANT: putting a `print` back next to the `guards.warn_allocation` call
    doubles the message and fails this.
    """
    import io
    from contextlib import redirect_stdout

    from oceanarches.evaluation.run_eval import warn_without_allocation

    captured = io.StringIO()
    with redirect_stdout(captured):
        warn_without_allocation("cuda")
    printed = captured.getvalue()
    if "SLURM_JOB_ID is unset" in printed:  # only meaningful off an allocation
        assert printed.count("SLURM_JOB_ID is unset") == 1
        assert printed.count("A visible GPU is not an allocated one.") == 1

    # On the CPU there is nothing to say at all.
    captured = io.StringIO()
    with redirect_stdout(captured):
        warn_without_allocation("cpu")
    assert captured.getvalue() == ""
