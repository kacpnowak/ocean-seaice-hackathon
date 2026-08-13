"""Tests for ``OceanForecastModule``.

Built from the real shipped configs (shrunk to ``emb_dim=32`` so a forward pass
on the 180x360 grid runs on a CPU in about a second), because what has to be
right is the *composition* of module, embedder, masks and statistics -- not a
hand-wired object that only exists in this file.

Most tests here replace the transformer with :func:`stub_backbone`, because what
they check -- shapes, the mask, the residual, the clamp, the timestamps a
rollout hands forward -- is what ``forward`` does *around* the backbone, not the
backbone's arithmetic.  Two are deliberately left running the real thing, so the
file still proves that a composed preset does a real forward pass on the real
grid: ``test_forward_returns_the_prognostic_channels_on_the_input_grid`` and
``test_forward_output_is_exactly_zero_over_land``.

The two tests that matter most are ``test_land_values_do_not_change_the_loss``
and ``test_the_module_takes_the_depth_levels_the_preset_names``: the first is the
regression test for the whole task, the second for the 14-prepared-levels /
13-used-levels trap.  Both were checked against deliberately broken
implementations -- see the Task 6 report.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from oceanarches import paths
from oceanarches.dataloaders.forcing import ForcingSource
from oceanarches.dataloaders.glorys import GlorysForecast
from oceanarches.dataloaders.variables import DEPTH_PRESETS, N_LAT, N_LON, get_component
from oceanarches.lightning_modules.ocean_forecast import (
    LossDiverged,
    OceanForecastModule,
    _select_statistics,
    compute_lat_weights_glorys,
)

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

#: Shrink the model to something a CPU can run, without touching anything the
#: tests are about (masks, statistics, components, depth levels).
SMALL = [
    "++module.backbone.emb_dim=32",
    "++module.backbone.num_heads=[2,4,4,2]",
    "++module.embedder.emb_dim=32",
    "++module.embedder.out_emb_dim=64",
    # ACC needs a 180 MB climatology per metric instance and nothing here tests it.
    "++module.metrics.glorys_deterministic_metrics.compute_acc=False",
]


def build_module(dataloader: str = "glorys_tiny", extra: list[str] = ()) -> OceanForecastModule:
    """Instantiate a shipped preset exactly the way ``main_hydra`` does."""
    overrides = ["module=tiny", f"dataloader={dataloader}", *SMALL, *extra]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR, job_name="test_module"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)
    return instantiate(cfg.module.module, cfg.module)


def make_batch(module: OceanForecastModule, batch: int = 1, seed: int = 0) -> dict:
    """A normalised batch shaped like the dataloader's output, land already zeroed.

    Batch 1 by default: a forward pass on the real 180x360 grid costs a couple of
    seconds per sample on a CPU and nothing here needs more than one.
    """
    torch.manual_seed(seed)
    entries = {"surface": torch.randn(batch, module.n_surface_in, 1, N_LAT, N_LON)}
    if module.n_level_in:
        entries["level"] = torch.randn(
            batch, module.n_level_in, module.embedder.n_depths, N_LAT, N_LON
        )
    state = module.apply_wet_mask(TensorDict(entries, batch_size=batch), inputs=True)
    return dict(
        state=state,
        prev_state=state.clone(),
        next_state=state.clone(),
        timestamp=torch.tensor(
            [1_547_510_400, 1_560_000_000, 1_570_000_000][:batch], dtype=torch.int32
        ),
        lead_time_hours=torch.tensor([24] * batch, dtype=torch.int32),
    )


@pytest.fixture(scope="module")
def full_module(real_masks_path) -> OceanForecastModule:
    return build_module()


@pytest.fixture(scope="module")
def full_batch(full_module) -> dict:
    return make_batch(full_module)


@pytest.fixture(scope="module")
def ocean_module(real_masks_path) -> OceanForecastModule:
    """The ocean specialist: predicts 3 surface fields, reads 7."""
    return build_module(dataloader="glorys_ocean")


@pytest.fixture(scope="module")
def seaice_module(real_masks_path) -> OceanForecastModule:
    """The surface-only component: ``level_ch_out == 0``."""
    return build_module(dataloader="glorys_seaice")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def test_the_optimiser_state_carries_no_omegaconf(full_module):
    """The fix at source: `betas` must reach AdamW as a plain tuple.

    `betas: [0.9, 0.98]` in configs/module/*.yaml composes to an OmegaConf
    `ListConfig`; hydra hands it to the module, geoarches passes it to
    `torch.optim.AdamW`, and AdamW stores it verbatim in `param_groups` -- so it
    lands in `optimizer_states` in every checkpoint. Since torch 2.6 that made
    the checkpoint unreadable by `torch.load(weights_only=True)`, which is the
    default Lightning's resume path uses, so **every run in this kit was
    unresumable**. Nothing noticed for six tasks: fresh runs load nothing, and
    both `load_ckpt` and `load_module` pass `weights_only=False`.

    MUTANT: `**kwargs` instead of `**plain_betas(kwargs)` in
    `OceanForecastModule.__init__`. This test fails; nothing else does.
    """
    from omegaconf import ListConfig

    assert not isinstance(full_module.betas, ListConfig)
    configured = full_module.configure_optimizers()
    # geoarches returns Lightning's ([optimizers], [schedulers]) form.
    if isinstance(configured, dict):
        optimizer = configured["optimizer"]
    elif isinstance(configured, (list, tuple)):
        optimizer = configured[0][0] if isinstance(configured[0], (list, tuple)) else configured[0]
    else:
        optimizer = configured
    assert isinstance(optimizer, torch.optim.Optimizer), type(optimizer)
    betas = optimizer.state_dict()["param_groups"][0]["betas"]
    assert not isinstance(betas, ListConfig), (
        "AdamW is holding an OmegaConf ListConfig; it will be pickled into every "
        "checkpoint and the run will not be resumable"
    )
    assert tuple(betas) == (0.9, 0.98)


def test_a_checkpoint_holding_omegaconf_still_resumes(tmp_path):
    """The fix on load, for the checkpoints already on disk.

    This is the shape a REAL checkpoint has -- verified by walking
    `modelstore/t10_tiny/checkpoints/checkpoint_global_step=4000.ckpt`, which
    contains exactly two OmegaConf objects, both at
    `optimizer_states[0]["param_groups"][*]["betas"]`, and no `hyper_parameters`
    key at all (geoarches has `save_hyperparameters()` commented out at
    `lightning_modules/forecast.py:48`).

    `plain_betas` stops new checkpoints looking like this, but the four shipped
    ones under `modelstore/` were written before it and participants are told to
    start from them, so the allowlist has to keep working too.

    MUTANT: removing the `allow_omegaconf_in_checkpoints()` call from
    `oceanarches/lightning_modules/__init__.py`. This test fails; nothing else
    does.
    """
    from omegaconf import OmegaConf

    import oceanarches.lightning_modules  # noqa: F401 - registers the allowlist

    checkpoint = {
        "state_dict": {"backbone.w": torch.zeros(2, 2)},
        "global_step": 500,
        "epoch": 2,
        "optimizer_states": [
            {
                "state": {},
                "param_groups": [
                    {
                        "lr": 3e-4,
                        "betas": OmegaConf.create([0.9, 0.98]),
                        "weight_decay": 0.05,
                        "params": [0],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "checkpoint_global_step=500.ckpt"
    torch.save(checkpoint, path)

    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored["global_step"] == 500
    assert tuple(restored["optimizer_states"][0]["param_groups"][0]["betas"]) == (0.9, 0.98)


def test_the_allowlist_still_refuses_code_execution(tmp_path):
    """It is an allowlist, not `weights_only=False`.

    Widening what loads is only acceptable if it does not widen what *runs*.
    """
    import pickle

    import oceanarches.lightning_modules  # noqa: F401

    class Payload:
        def __reduce__(self):
            import os

            return (os.system, ("echo pwned",))

    path = tmp_path / "evil.ckpt"
    with open(path, "wb") as handle:
        pickle.dump({"state_dict": Payload()}, handle)
    with pytest.raises(Exception, match="(?i)weights only|unsupported global|forbidden"):
        torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
def test_the_shipped_preset_builds_our_module(full_module):
    assert isinstance(full_module, OceanForecastModule)
    assert full_module.component.name == "full"
    assert full_module.add_input_state is True, "residual prediction is the default here"
    assert full_module.clamp_physical_bounds is True
    assert full_module.loss_delta_normalization is True
    assert full_module.lead_time_hours == 24


def test_the_era5_loss_coefficients_are_gone(full_module):
    """The parent builds coefficients for 121 latitudes and 13 pressure levels."""
    assert not hasattr(full_module, "loss_coeffs")
    assert full_module.loss_coeff_surface.shape == (7, 1, N_LAT, N_LON)
    assert full_module.loss_coeff_level.shape == (4, 13, N_LAT, N_LON)


def test_the_loss_coefficients_are_zero_over_land(full_module):
    assert (full_module.loss_coeff_surface * (1 - full_module.mask_surface)).abs().max() == 0.0
    assert (full_module.loss_coeff_level * (1 - full_module.mask_level)).abs().max() == 0.0


def test_lat_weights_come_from_our_own_grid(full_module):
    weights = full_module.compute_lat_weights_glorys(N_LAT)
    assert weights.shape == (N_LAT, 1)
    assert torch.equal(weights, compute_lat_weights_glorys(N_LAT))


def test_a_component_the_embedder_was_not_built_for_is_fatal(real_masks_path):
    """Hydra wraps our ValueError, so match on the message, not the type."""
    with pytest.raises(InstantiationException, match="Component 'seaice' needs surface 7 in"):
        build_module(extra=["++module.module.component=seaice"])


# ---------------------------------------------------------------------------
# Statistics: the same subset as the dataloader
# ---------------------------------------------------------------------------
def test_the_module_and_the_dataloader_select_the_same_statistics(tiny_forecast_kwargs):
    """The two selections live in two files; if they drift, nothing else complains."""
    indices = DEPTH_PRESETS["tiny"]
    dataset = GlorysForecast(domain="tiny_val", depth_indices=indices, **tiny_forecast_kwargs)
    stats = torch.load(tiny_forecast_kwargs["stats_path"], weights_only=True)
    for group, names in (
        ("surface", dataset.surface_variables),
        ("level", dataset.level_variables),
    ):
        selected = _select_statistics(stats, group, names, indices)
        assert torch.equal(selected["mean"], dataset.data_mean[group]), group
        assert torch.equal(selected["std"], dataset.data_std[group]), group
        assert torch.equal(selected["delta_std"], dataset.delta_std[group]), group


def test_the_module_takes_the_depth_levels_the_preset_names(full_module):
    """The 14-vs-13 trap: the statistics file has one level the model never sees."""
    stats = torch.load(paths.stats_file(), weights_only=True)
    indices = DEPTH_PRESETS["tiny"]
    assert stats["level_mean"].shape[1] == 14, "the prepared statistics carry 14 levels"
    assert full_module.state_mean_level.shape[1] == 13
    assert torch.equal(full_module.state_mean_level, stats["level_mean"][:, indices])
    assert torch.equal(full_module.state_std_level, stats["level_std"][:, indices])
    # The deepest used level is prepared index 13, not 12 (1245 m, which the
    # presets drop).  An off-by-one here would be invisible everywhere else.
    assert torch.equal(full_module.state_mean_level[:, -1], stats["level_mean"][:, 13])
    assert not torch.equal(full_module.state_mean_level[:, -1], stats["level_mean"][:, 12])


def test_denormalisation_inverts_the_dataloader_normalisation(full_module, full_batch):
    normalised = full_batch["state"]
    physical = full_module.denormalize_state(full_module.select_prognostic(normalised))
    back = (physical["surface"] - full_module.state_mean_surface) / full_module.state_std_surface
    assert torch.allclose(back, normalised["surface"], atol=1e-5)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def test_forward_returns_the_prognostic_channels_on_the_input_grid(full_module, full_batch):
    with torch.no_grad():
        out = full_module.forward(full_batch)
    assert out["surface"].shape == (1, 7, 1, N_LAT, N_LON)
    assert out["level"].shape == (1, 4, 13, N_LAT, N_LON)


def test_forward_output_is_exactly_zero_over_land(full_module, full_batch):
    with torch.no_grad():
        out = full_module.forward(full_batch)
    assert (out["surface"] * (1 - full_module.mask_surface)).abs().max().item() == 0.0
    assert (out["level"] * (1 - full_module.mask_level)).abs().max().item() == 0.0


def stub_backbone(module: OceanForecastModule, monkeypatch):
    """Replace the transformer with zeros of the right latent shape.

    Used by every test here that cares about what ``forward`` does with the
    backbone's output rather than about the backbone itself.  It takes a full
    forward pass on the 180x360 grid from ~2.4 s to ~30 ms; ``encode`` and
    ``decode`` still run, so all the shape plumbing, the mask, the residual and
    the clamp are still exercised on the real grid.

    It is *not* a way to make an assertion cheap by making it vacuous: a stubbed
    backbone still produces a non-zero prediction, because the decoder's own
    weights and biases act on the zero tokens.
    """
    latent = (module.embedder.out_emb_dim, module.embedder.z_dim, *module.embedder.latent_grid)
    # NOT zeros.  geoarches' level/surface deconvolutions have no bias, so zero
    # tokens decode to an exactly zero field, the residual then returns the
    # (already masked) input state, and every "land is 0" assertion downstream
    # passes whether or not the mask is applied -- verified: with a zeros stub,
    # deleting `apply_wet_mask` from `forward` leaves
    # test_forward_multistep_preserves_shape_and_mask green.  A constant
    # non-zero token decodes to a non-zero field over land, so the assertions
    # still have something to catch.
    #
    # A constant (rather than random) tensor keeps the stub deterministic, which
    # test_the_first_rollout_step_equals_a_single_forward relies on.
    # Patch `forward`, not the attribute: nn.Module refuses to have a submodule
    # replaced by a plain function.
    monkeypatch.setattr(
        module.backbone,
        "forward",
        lambda tokens, *args, **kwargs: torch.full((tokens.shape[0], *latent), 0.5),
    )


def zero_decoder(module: OceanForecastModule):
    """Replace the decoder with one that predicts no change at all."""

    def decode(tokens):
        batch = tokens.shape[0]
        entries = {"surface": torch.zeros(batch, module.n_surface_out, 1, N_LAT, N_LON)}
        if module.n_level_out:
            entries["level"] = torch.zeros(
                batch, module.n_level_out, module.embedder.n_depths, N_LAT, N_LON
            )
        return TensorDict(entries, batch_size=batch)

    return decode


def test_the_residual_connection_returns_the_input_state(full_module, full_batch, monkeypatch):
    """With a zero tendency, ``add_input_state`` makes the model persistence.

    The clamp is switched off here so that the test isolates the residual: the
    random input state is out of range for siconc, and clamping it -- correctly --
    would make the output differ from the input.
    """
    stub_backbone(full_module, monkeypatch)
    monkeypatch.setattr(full_module.embedder, "decode", zero_decoder(full_module))
    monkeypatch.setattr(full_module, "clamp_physical_bounds", False)
    with torch.no_grad():
        out = full_module.forward(full_batch)
    expected = full_module.select_prognostic(full_batch["state"])
    assert torch.allclose(out["surface"], expected["surface"], atol=1e-6)
    assert torch.allclose(out["level"], expected["level"], atol=1e-6)


def test_without_the_residual_a_zero_tendency_gives_a_zero_state(
    full_module, full_batch, monkeypatch
):
    stub_backbone(full_module, monkeypatch)
    monkeypatch.setattr(full_module.embedder, "decode", zero_decoder(full_module))
    monkeypatch.setattr(full_module, "add_input_state", False)
    with torch.no_grad():
        out = full_module.forward(full_batch)
    assert out["surface"].abs().max().item() == 0.0


def constant_decoder(module: OceanForecastModule, value: float):
    base = zero_decoder(module)

    def decode(tokens):
        return base(tokens).apply(lambda x: x + value)

    return decode


@pytest.mark.parametrize("value", [50.0, -50.0])
def test_clamping_keeps_sea_ice_physical(full_module, full_batch, monkeypatch, value):
    """Bounds are enforced in normalised space; check them after denormalising."""
    stub_backbone(full_module, monkeypatch)
    monkeypatch.setattr(full_module.embedder, "decode", constant_decoder(full_module, value))
    monkeypatch.setattr(full_module, "add_input_state", False)
    with torch.no_grad():
        physical = full_module.denormalize_state(full_module.forward(full_batch))
    ocean = full_module.mask_surface[3].bool()  # siconc
    siconc = physical["surface"][:, 3][:, ocean]
    thickness = physical["surface"][:, 4][:, full_module.mask_surface[4].bool()]
    assert siconc.min().item() >= -1e-5 and siconc.max().item() <= 1 + 1e-5
    assert thickness.min().item() >= -1e-5


def test_without_the_clamp_it_leaves_the_physical_range(full_module, full_batch, monkeypatch):
    """The complement of the test above: the clamp is what does the work."""
    stub_backbone(full_module, monkeypatch)
    monkeypatch.setattr(full_module.embedder, "decode", constant_decoder(full_module, 50.0))
    monkeypatch.setattr(full_module, "add_input_state", False)
    monkeypatch.setattr(full_module, "clamp_physical_bounds", False)
    with torch.no_grad():
        physical = full_module.denormalize_state(full_module.forward(full_batch))
    ocean = full_module.mask_surface[3].bool()
    assert physical["surface"][:, 3][:, ocean].max().item() > 1.0


def test_an_unbounded_variable_is_not_clamped(full_module, full_batch, monkeypatch):
    """`zos`, `bottomT`, `thetao`, `uo` and `vo` carry no bounds in variables.py."""
    stub_backbone(full_module, monkeypatch)
    monkeypatch.setattr(full_module.embedder, "decode", constant_decoder(full_module, 50.0))
    monkeypatch.setattr(full_module, "add_input_state", False)
    with torch.no_grad():
        out = full_module.forward(full_batch)
    ocean = full_module.mask_surface[0].bool()  # zos
    assert out["surface"][:, 0][:, ocean].min().item() == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------
def persistence_sized_error(module: OceanForecastModule, batch: int = 1) -> TensorDict:
    """A normalised error whose physical size is exactly one ``delta_std``.

    That is what a forecast which simply repeats today's state produces, so the
    loss must come out at 1.0 by construction.
    """
    entries = {}
    for group in ("surface", "level"):
        if group == "level" and not module.n_level_out:
            continue
        statistics = module._statistics[group]
        ratio = statistics["delta_std"] / statistics["std"]
        shape = getattr(module, f"mask_{group}").shape
        entries[group] = ratio.expand(shape)[None].repeat(batch, *([1] * 4))
    return TensorDict(entries, batch_size=batch)


def test_a_persistence_sized_error_scores_one(full_module):
    """The loss is normalised so that this number means something."""
    error = persistence_sized_error(full_module)
    zeros = error.apply(torch.zeros_like)
    assert float(full_module.loss(error, zeros)) == pytest.approx(1.0, rel=1e-5)


def test_land_values_do_not_change_the_loss(full_module):
    """The regression test for the entire task.

    Writing 10^6 on every land cell of both the prediction and the target must
    leave the loss bit-identical.  It does not if the wet mask is dropped, if the
    reduction divides by the grid area, or if the mask is applied to only one of
    the two state groups.
    """
    error = persistence_sized_error(full_module)
    zeros = error.apply(torch.zeros_like)
    clean = float(full_module.loss(error, zeros))

    polluted_pred = error.clone()
    polluted_truth = zeros.clone()
    for group in error.keys():
        land = 1 - getattr(full_module, f"mask_{group}")
        polluted_pred[group] = polluted_pred[group] + land * 1e6
        polluted_truth[group] = polluted_truth[group] - land * 1e6
    assert float(full_module.loss(polluted_pred, polluted_truth)) == clean


def test_the_loss_falls_when_the_prediction_improves(full_module):
    error = persistence_sized_error(full_module)
    zeros = error.apply(torch.zeros_like)
    worse = float(full_module.loss(error.apply(lambda x: 2 * x), zeros))
    better = float(full_module.loss(error.apply(lambda x: 0.5 * x), zeros))
    assert better < 1.0 < worse
    # pow=2, so halving the error quarters the loss.
    assert better == pytest.approx(0.25, rel=1e-5)


def test_the_loss_slices_a_full_target_down_to_the_component(ocean_module, monkeypatch):
    """A component reads more than it predicts; the loss must not broadcast-fail."""
    module = ocean_module
    stub_backbone(module, monkeypatch)
    batch = make_batch(module)
    with torch.no_grad():
        prediction = module.forward(batch)
    assert prediction["surface"].shape[1] == 3
    assert float(module.loss(prediction, batch["next_state"])) >= 0.0


def test_the_multistep_discount_is_applied_and_normalised(full_module):
    """geoarches computes this discount and then throws the result away."""
    error = persistence_sized_error(full_module, batch=1)
    stacked = torch.stack([error, error.apply(lambda x: 0.0 * x)], dim=1)
    target = stacked.apply(torch.zeros_like)
    loss = float(full_module.loss(stacked, target, multistep=True))
    # Discounts 1 and 1/4, mean-normalised to 1.6 and 0.4; the second step is
    # perfect, so only the first contributes: 1.6 * 1.0 / 2 steps = 0.8.
    assert loss == pytest.approx(0.8, rel=1e-4)


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------
def test_forward_multistep_preserves_shape_and_mask(full_module, full_batch, monkeypatch):
    stub_backbone(full_module, monkeypatch)
    with torch.no_grad():
        rollout = full_module.forward_multistep(full_batch, iters=3)
    assert rollout["surface"].shape == (1, 3, 7, 1, N_LAT, N_LON)
    assert rollout["level"].shape == (1, 3, 4, 13, N_LAT, N_LON)
    assert (rollout["surface"] * (1 - full_module.mask_surface)).abs().max().item() == 0.0
    assert (rollout["level"] * (1 - full_module.mask_level)).abs().max().item() == 0.0


def test_the_first_rollout_step_equals_a_single_forward(full_module, full_batch, monkeypatch):
    stub_backbone(full_module, monkeypatch)
    with torch.no_grad():
        single = full_module.forward(full_batch)
        rollout = full_module.forward_multistep(full_batch, iters=2)
    assert torch.equal(rollout["surface"][:, 0], single["surface"])
    assert torch.equal(rollout["level"][:, 0], single["level"])


def test_the_rollout_feeds_its_own_output_back_in(full_module, full_batch, monkeypatch):
    """Step 2 must be computed from step 1's output, not from the initial state.

    Nothing else in this file would notice a rollout that handed the *same*
    state to every iteration: the shapes, the mask and the timestamps would all
    still be right and every step would simply repeat the same forecast.
    Checked against exactly that mutation (`state=loop_batch["state"]` instead
    of `state=self.advance_state(...)`), which left the whole of this file green
    and was only caught, indirectly, by the coupling suite.
    """
    stub_backbone(full_module, monkeypatch)
    with torch.no_grad():
        rollout = full_module.forward_multistep(full_batch, iters=3)
    for step in (1, 2):
        assert not torch.equal(rollout["surface"][:, step], rollout["surface"][:, step - 1]), (
            f"rollout step {step} is bit-identical to step {step - 1}: the state is not "
            "being advanced between iterations"
        )


def test_the_rollout_advances_one_lead_time_per_step(full_module, full_batch, monkeypatch):
    """Not ``batch['lead_time_hours']``, which the dataloader sets to lead x multistep.

    Following geoarches here would move the month-of-year conditioning forward by
    ten days per step on a ten-day rollout.
    """
    stub_backbone(full_module, monkeypatch)
    seen = []
    original = full_module.forward

    def record(batch, *args, **kwargs):
        seen.append(int(batch["timestamp"][0]))
        return original(batch, *args, **kwargs)

    monkeypatch.setattr(full_module, "forward", record)
    batch = dict(full_batch)
    batch["lead_time_hours"] = torch.tensor([240], dtype=torch.int32)  # 10-day rollout
    with torch.no_grad():
        full_module.forward_multistep(batch, iters=3)
    day = 24 * 3600
    assert seen == [seen[0], seen[0] + day, seen[0] + 2 * day]


def test_a_prescribed_forcing_source_is_fetched_for_every_valid_time(real_masks_path, monkeypatch):
    """The external-forcing path Task 8 needs: one field per step's valid time.

    ``NoForcing`` (no forcing at all) is the default and every other test here
    exercises it; this one is the only check that the non-empty path works.
    """

    class StampForcing(ForcingSource):
        """Two channels, both filled with the timestamp it was asked for."""

        variables = ["a", "b"]

        def __init__(self):
            self.asked = []

        def get(self, timestamp):
            self.asked.append(int(timestamp))
            return torch.full((2, 1, N_LAT, N_LON), float(timestamp) / 1e9)

    source = StampForcing()
    module = build_module(extra=["++module.embedder.forcing_ch=2"])
    stub_backbone(module, monkeypatch)
    module.forcing_source = source
    batch = make_batch(module)
    with torch.no_grad():
        module.forward_multistep(batch, iters=3)
    day = 24 * 3600
    assert source.asked == [source.asked[0] + i * day for i in range(3)]


@pytest.fixture(scope="module")
def small_masks(real_masks_path, tmp_path_factory) -> tuple[Path, int, int]:
    """The real masks cropped to a 36x60 corner: the smallest grid the stack takes.

    36/3 = 12 and 60/3 = 20 give a latent grid the `[1, 6, 10]` attention window
    tiles exactly at both stages (12/6 = 2, 20/10 = 2, then 6/6 = 1, 10/10 = 1).
    Smaller than `cropped_masks` because the test below only has to see a number
    move, not resolve anything.
    """
    import xarray as xr

    lat, lon = 36, 60
    with xr.open_dataset(real_masks_path) as masks:
        cropped = masks.isel(lat=slice(0, lat), lon=slice(0, lon)).load()
    path = tmp_path_factory.mktemp("small_masks") / f"masks_{lat}x{lon}.nc"
    cropped.to_netcdf(path)
    return path, lat, lon


def test_changing_the_forcing_changes_the_prediction(small_masks):
    """Finding 1 of Task 12: the forcing must actually reach the network.

    The two ends of the file-forcing path -- ``OceanForecastModule.forcing_source``
    and ``OceanEncodeDecodeLayer(forcing_ch=...)`` -- both existed for several
    tasks without anything ever checking that they meet.  A config that composes,
    a source that instantiates and a tensor of the right shape are all compatible
    with the channels being dropped on the floor.

    So: the **real** backbone (:func:`stub_backbone` would make this vacuous),
    identical weights, identical state, identical timestamp, and only the forcing
    field changed.  If the prediction does not move, the wiring is broken however
    well it composes.  On the real masks cropped to 36x60 rather than the whole
    globe, for
    the same reason ``depth_coupling`` is: this probe cannot stub the backbone,
    so it shrinks the map instead (10 s on the globe, 2 s here).

    MUTANT: zeroing ``forcing`` in ``OceanForecastModule.forward`` fails here.
    """
    masks_file, lat, lon = small_masks

    class ConstantForcing(ForcingSource):
        variables = ["a", "b"]

        def __init__(self, value: float):
            self.value = value

        def get(self, timestamp):
            return torch.full((2, 1, lat, lon), self.value)

    module = build_module(
        extra=[
            "++module.embedder.forcing_ch=2",
            f"++module.module.masks_path={masks_file}",
            f"++module.embedder.masks_path={masks_file}",
            f"++module.embedder.img_size=[{lat},{lon}]",
            f"++module.backbone.tensor_size=[8,{lat // 3},{lon // 3}]",
        ]
    )
    module.eval()  # droppath is stochastic in training mode
    torch.manual_seed(0)
    state = TensorDict(
        {
            "surface": torch.randn(1, module.n_surface_in, 1, lat, lon),
            "level": torch.randn(1, module.n_level_in, module.embedder.n_depths, lat, lon),
        },
        batch_size=1,
    )
    state = module.apply_wet_mask(state, inputs=True)
    batch = dict(
        state=state,
        prev_state=state.clone(),
        timestamp=torch.tensor([1_547_510_400], dtype=torch.int32),
    )

    with torch.no_grad():
        module.forcing_source = ConstantForcing(0.0)
        calm = module.forward(batch)
        again = module.forward(batch)
        module.forcing_source = ConstantForcing(1.5)
        stormy = module.forward(batch)

    # The control: same forcing, same everything -> bit-identical.  Without it
    # the assertion below could be passing on nothing but nondeterminism.
    assert torch.equal(calm["surface"], again["surface"])

    ocean = module.mask_surface.expand_as(calm["surface"]) > 0
    moved = (stormy["surface"] - calm["surface"])[ocean].abs()
    typical = calm["surface"][ocean].abs().mean()
    assert float(moved.max()) > 0, "the forcing never reached the network"
    # Not merely non-zero: a change worth at least a per-cent of the signal.
    assert float(moved.mean()) > 0.01 * float(typical)


def test_the_no_forcing_default_reserves_nothing_and_fetches_nothing(full_module):
    """`forcing=none` must stay exactly what it was: no channels, no source.

    MUTANT: `configs/forcing/none.yaml` with `n_channels: 1` fails here.
    """
    assert full_module.forcing_source is None
    assert full_module.embedder.forcing_ch == 0
    assert full_module.external_forcing(torch.tensor([1_547_510_400], dtype=torch.int32)) is None
    # The embedder must also refuse a forcing tensor it has no channels for,
    # rather than silently ignoring it.
    batch = make_batch(full_module)
    with pytest.raises(ValueError, match="forcing_ch=0"):
        full_module.embedder.encode(
            batch["state"], batch["prev_state"], torch.zeros(1, 2, 1, N_LAT, N_LON)
        )


def test_the_rollout_carries_a_components_forcing_channels_forward(ocean_module, monkeypatch):
    """The ocean model predicts 3 surface fields and reads 4 more; those 4 persist."""
    module = ocean_module
    stub_backbone(module, monkeypatch)
    batch = make_batch(module)
    with torch.no_grad():
        prediction = module.forward(batch)
        advanced = module.advance_state(batch["state"], prediction)
    assert advanced["surface"].shape[1] == module.n_surface_in == 7
    assert torch.equal(advanced["surface"][:, :3], prediction["surface"])
    assert torch.equal(advanced["surface"][:, 3:], batch["state"]["surface"][:, 3:])


def test_the_rollout_state_stays_a_full_physical_state(ocean_module, monkeypatch):
    """Task 8 swaps another model's prediction into it, so it must not go latent."""
    module = ocean_module
    stub_backbone(module, monkeypatch)
    batch = make_batch(module)
    with torch.no_grad():
        rollout = module.forward_multistep(batch, iters=2)
    assert rollout["surface"].shape[-2:] == (N_LAT, N_LON)
    assert rollout["surface"].shape[2] == module.n_surface_out


# ---------------------------------------------------------------------------
# Components, including the surface-only one
# ---------------------------------------------------------------------------
def test_the_sea_ice_component_predicts_nothing_three_dimensional(seaice_module, monkeypatch):
    module = seaice_module
    assert module.n_level_out == 0
    assert not hasattr(module, "mask_level")
    stub_backbone(module, monkeypatch)
    batch = make_batch(module)
    with torch.no_grad():
        out = module.forward(batch)
    assert set(out.keys()) == {"surface"}
    assert out["surface"].shape == (1, 4, 1, N_LAT, N_LON)
    assert float(module.loss(out, batch["next_state"])) >= 0.0
    with torch.no_grad():
        rollout = module.forward_multistep(batch, iters=2)
    assert set(rollout.keys()) == {"surface"}
    assert rollout["surface"].shape == (1, 2, 4, 1, N_LAT, N_LON)


def test_a_surface_only_component_still_reads_the_whole_ocean(seaice_module):
    module = seaice_module
    assert module.n_level_in == 4, "it reads thetao, so, uo and vo"
    assert module.component.prognostic == get_component("seaice").prognostic


# ---------------------------------------------------------------------------
# Physical bounds on the ocean fields
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [-50.0, 50.0])
def test_mixed_layer_depth_and_salinity_cannot_go_negative(
    full_module, full_batch, monkeypatch, value
):
    """`mlotst` and `so` carry `bounds=(0.0, None)`, like the two ice fields.

    Without them the shipped 10-day forecast contained physically impossible
    values: measured on `t10_tiny` over 16 spread initialisations of the test
    split, 79 ocean cells held a negative mixed-layer depth at day 1 and 15333
    at day 10 (minimum -53.9 m), and 30 to 84 cells held negative salinity. The
    metrics -- `rmse_mlotst` among them -- were computed over exactly those
    cells.

    MUTANT: setting `bounds=(None, None)` back on either variable in
    `variables.py` fails this at `value=-50.0`; the positive case is here so
    that a clamp accidentally applied as an *upper* bound is caught too.
    """
    stub_backbone(full_module, monkeypatch)
    monkeypatch.setattr(full_module.embedder, "decode", constant_decoder(full_module, value))
    monkeypatch.setattr(full_module, "add_input_state", False)
    with torch.no_grad():
        physical = full_module.denormalize_state(full_module.forward(full_batch))

    mlotst_index = full_module.surface_variables.index("mlotst")
    ocean = full_module.mask_surface[mlotst_index].bool()
    mlotst = physical["surface"][:, mlotst_index][:, ocean]
    assert mlotst.min().item() >= -1e-4, "mixed layer depth went negative"

    so_index = full_module.level_variables.index("so")
    wet = full_module.mask_level[so_index].bool()
    salinity = physical["level"][:, so_index][wet.expand_as(physical["level"][:, so_index])]
    assert salinity.min().item() >= -1e-4, "salinity went negative"


# ---------------------------------------------------------------------------
# The multi-step training curriculum
# ---------------------------------------------------------------------------
def _attach_fake_trainer(module, multistep: int, epoch: int):
    """The two attributes `on_train_epoch_start` reads, and nothing else."""
    dataset = SimpleNamespace(multistep=multistep)
    module._trainer = SimpleNamespace(
        train_dataloader=SimpleNamespace(dataset=dataset), current_epoch=epoch
    )
    return dataset


def test_the_training_rollout_length_is_the_one_the_config_asked_for(full_module):
    """`++module.train.rollout_iterations=2` must train on a 2-step rollout.

    geoarches' own `on_train_epoch_start` overwrites the dataset's multistep with
    `2 + epoch // increase_multistep_period` on every epoch, unconditionally, so
    the four documents that recommend `=2` as the cure for the 90-day divergence
    were describing a run whose rollout climbs to 10 over `tiny`'s 18 epochs --
    a mean of 5.88 steps per training step, and about six times the compute the
    number implies. `=3` was silently *reduced* to 2 at epoch 0.

    MUTANT: deleting `OceanForecastModule.on_train_epoch_start` (so geoarches'
    runs) makes the epoch-16 case below come back 10 instead of 2.
    """
    try:
        for epoch in (0, 1, 8, 16):
            dataset = _attach_fake_trainer(full_module, multistep=2, epoch=epoch)
            full_module.on_train_epoch_start()
            assert dataset.multistep == 2, (
                f"epoch {epoch} moved the rollout to {dataset.multistep}"
            )
        # A longer rollout is equally untouched.
        dataset = _attach_fake_trainer(full_module, multistep=5, epoch=9)
        full_module.on_train_epoch_start()
        assert dataset.multistep == 5
    finally:
        full_module._trainer = None


def test_geoarches_curriculum_is_still_available_when_it_is_asked_for(full_module, monkeypatch):
    """Opt in with `++module.module.multistep_curriculum=True` and the ramp returns."""
    monkeypatch.setattr(full_module, "multistep_curriculum", True)
    try:
        dataset = _attach_fake_trainer(full_module, multistep=2, epoch=16)
        full_module.on_train_epoch_start()
        assert dataset.multistep == 2 + 16 // full_module.increase_multistep_period
    finally:
        full_module._trainer = None


def test_the_rollout_does_not_invent_a_previous_state_the_embedder_cannot_take(monkeypatch):
    """`load_prev: False` is a natural ablation, and it used to die at step 2.

    `forward_multistep` carried `prev_state` into every subsequent step whatever
    the embedder was built for, so a model built with
    `n_concatenated_states=0` trained happily and then raised "encode() was given
    a cond_state but the embedder was built with n_concatenated_states=0" the
    first time anyone rolled it out past one day -- after the training run.

    MUTANT: putting `prev_state` back into the loop batch unconditionally makes
    this raise that ValueError at step 2.
    """
    module = build_module(extra=["++module.embedder.n_concatenated_states=0"])
    assert module.embedder.n_concatenated_states == 0
    batch = make_batch(module)
    del batch["prev_state"]
    stub_backbone(module, monkeypatch)
    with torch.no_grad():
        out = module.forward_multistep(batch, iters=3)
    assert out["surface"].shape[1] == 3


# ---------------------------------------------------------------------------
# Scaling the model
# ---------------------------------------------------------------------------
def test_raising_emb_dim_alone_is_refused_at_construction(real_masks_path):
    """docs/04 invites you to scale the model up, and the obvious try is
    `HYDRA_ARGS="++module.backbone.emb_dim=192"`.

    That instantiated silently and died in the first forward pass with
    `RuntimeError: mat1 and mat2 shapes cannot be multiplied (7200x768 and
    1536x1536)` -- after the dataloader had spun up, on a GPU you had queued for.
    `tests/test_configs.py::test_backbone_and_embedder_agree` pins the same rule
    for the four shipped presets but cannot see an override.

    Hydra wraps the ValueError in an `InstantiationException` whose message is
    `repr(e)`, so the text still reaches the participant -- which is the whole
    difference from the coupling case, where a bare `FileNotFoundError` stringified
    to nothing useful.

    MUTANT: removing the `self._check_backbone_widths()` call from
    `OceanForecastModule.__init__` lets every one of these build, and all three
    `pytest.raises` fail.
    """
    for override, wanted in [
        # `SMALL` in this file sets the embedder to 32; moving the backbone alone
        # is exactly the mistake docs/04 used to invite.
        ("++module.backbone.emb_dim=64", "not the same size"),
        ("++module.backbone.num_heads=[5,4,4,2]", "num_heads"),
        ("++module.embedder.out_emb_dim=128", "out_emb_dim"),
    ]:
        with pytest.raises((ValueError, InstantiationException)) as caught:
            build_module(extra=[override])
        assert wanted in str(caught.value), override


def test_the_error_names_the_three_numbers_that_move_together(real_masks_path):
    with pytest.raises((ValueError, InstantiationException)) as caught:
        build_module(extra=["++module.backbone.emb_dim=64"])
    message = str(caught.value)
    for wanted in ("emb_dim", "num_heads", "out_emb_dim", "docs/04_scaling_finetuning.md"):
        assert wanted in message


def test_an_empty_depth_selection_is_refused(tiny_stats_file):
    """`[]` is "no levels", not "every level", and the difference was invisible.

    `_select_statistics` sliced the depth axis to length zero and its own
    positivity check, `(std > 0).all()`, came back True on the empty tensor.

    MUTANT: deleting the `len(depth_indices) == 0` branch returns statistics of
    shape [1, 0, 1, 1] and this stops raising.
    """
    stats = torch.load(tiny_stats_file, weights_only=True)
    with pytest.raises(ValueError, match="empty sequence"):
        _select_statistics(stats, "level", ["thetao"], [])
    # None still means every prepared level.
    selected = _select_statistics(stats, "level", ["thetao"], None)
    assert selected["std"].shape[1] > 0


# ---------------------------------------------------------------------------
# The divergence guard, where it is wired in
# ---------------------------------------------------------------------------
# The rule itself, and the measured trajectory it was chosen against, live in
# tests/test_divergence.py. What is checked here is that `training_step` -- the
# one piece of code `make train-tiny`, the notebooks, scripts/finetune.slurm,
# `geoarches.main_hydra` and `oceanarches.main_multinode` all share -- really
# consults it, and that the config key really turns it off.
def test_a_non_finite_training_loss_aborts_the_run(full_module, monkeypatch, tmp_path, capsys):
    """A NaN loss must stop the run there, naming the checkpoint and the way back.

    MUTANT: delete the `self.check_divergence(loss)` line from `training_step`
    and this trains on happily, which is exactly the eight wasted hours this
    guard exists to prevent.
    """
    written = tmp_path / "checkpoints" / "checkpoint_global_step=15000.ckpt"
    written.parent.mkdir(parents=True)
    written.touch()
    monkeypatch.setattr(paths, "run_dir", lambda name: tmp_path)

    def diverged(batch, rollout_iterations):
        return torch.tensor(float("nan")), None, None

    monkeypatch.setattr(full_module, "_predict", diverged)
    with pytest.raises(LossDiverged) as raised:
        full_module.training_step(make_batch(full_module), 0)

    message = str(raised.value)
    assert "NaN" in message
    assert str(written) in message, "the abort did not name the last checkpoint written"
    assert "++module.module.lr=" in message, "the abort did not name the way back"
    # Notebooks 02 and 03 capture stdout and discard stderr.
    assert "TRAINING DIVERGED" in capsys.readouterr().out


def test_the_guard_is_on_by_default(full_module):
    """Every route gets it without editing a config.

    MUTANT: `divergence_guard: bool = False` in the constructor.
    """
    assert full_module.divergence_guard is not None


def test_the_guard_can_be_turned_off_from_the_module_config(real_masks_path):
    """`++module.module.divergence_guard=False`, for somebody who means to ride a spike.

    A plain `**kwargs` catch-all would swallow this override silently -- and
    `oceanarches.guards.override_problems` would then reject the key as one
    nothing reads -- so it is a named constructor argument.

    MUTANT: rename the constructor argument and this build raises instead.
    """
    module = build_module(extra=["++module.module.divergence_guard=False"])
    assert module.divergence_guard is None
    module.check_divergence(torch.tensor(float("nan")))  # a no-op, not a refusal


def test_the_thresholds_are_configurable_too(real_masks_path):
    """MUTANT: hardcode `DivergenceGuard()` in the constructor and these come back 100/50."""
    module = build_module(
        extra=["++module.module.divergence_factor=3", "++module.module.divergence_patience=2"]
    )
    assert (module.divergence_guard.factor, module.divergence_guard.patience) == (3.0, 2)
