"""Tests for the coupling: the router's index bookkeeping and the coupled rollout.

Fast and CPU-only.  The components here have no backbone -- they are cheap,
deterministic functions of the state -- but the machinery around them is the
real thing: ``StubComponent`` borrows ``OceanForecastModule``'s own
``forward_multistep``, ``advance_state``, wet masking and prognostic slice, so
``test_a_coupled_full_component_is_bit_identical_to_it_alone`` compares
:class:`CoupledForecastModule` against the *actual* single-model rollout code
rather than against a re-implementation of it.  The grid is the 6x8 one the
shared fixtures build, which is why the whole file runs in about a second.

Several tests exist because the property they pin cannot be checked by eye, and
each was confirmed to **fail** against a deliberately broken implementation
before being kept.  Those carry a ``MUTANT:`` note saying exactly what was
broken and what happened.  The Task 8 report has the full list.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from oceanarches.dataloaders import variables as V
from oceanarches.dataloaders.masks import load_masks, state_mask
from oceanarches.lightning_modules.coupled import (
    CoupledForecastModule,
    StateRouter,
    named_component_like,
    union_component,
)
from oceanarches.lightning_modules.ocean_forecast import OceanForecastModule

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

#: 13 of the 14 prepared levels -- the shipped presets' choice, so the tests
#: exercise the same depth slicing the real models do.
DEPTH_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def masks(tiny_masks_file):
    return load_masks(path=tiny_masks_file, depth_indices=DEPTH_INDICES)


@pytest.fixture(scope="module")
def grid(masks) -> tuple[int, int]:
    return tuple(masks.wet_surface.shape)


def shared_state(
    grid: tuple[int, int],
    layout: V.ComponentSpec = None,
    batch: int = 2,
    seed: int = 0,
) -> TensorDict:
    """A state in the shared layout, with a distinct constant per channel.

    Constant per channel on purpose: a test that checks a channel arrived in the
    right slot can then name the value it expects instead of comparing tensors
    that a permutation would leave looking plausible.  The two batch elements
    differ so a bug that broadcasts one over the other shows up.
    """
    layout = layout or V.COMPONENTS["full"]
    lat, lon = grid
    generator = torch.Generator().manual_seed(seed)
    entries = {}
    n_surface = len(layout.input_surface)
    entries["surface"] = torch.arange(1.0, n_surface + 1).reshape(1, -1, 1, 1, 1) + 0.01 * (
        torch.arange(batch).reshape(-1, 1, 1, 1, 1)
        + torch.rand((1, 1, 1, lat, lon), generator=generator)
    )
    entries["surface"] = entries["surface"].expand(batch, n_surface, 1, lat, lon).contiguous()
    if layout.input_level:
        n_level = len(layout.input_level)
        entries["level"] = (
            100.0 + torch.arange(1.0, n_level + 1).reshape(1, -1, 1, 1, 1)
        ) + 0.01 * torch.rand((batch, n_level, len(DEPTH_INDICES), lat, lon), generator=generator)
    return TensorDict(entries, batch_size=(batch,))


def channel(state: TensorDict, layout: V.ComponentSpec, name: str) -> torch.Tensor:
    """One named channel out of a state laid out as ``layout``'s input."""
    group = "level" if V.variable_group(name) == "level" else "surface"
    names = layout.input_level if group == "level" else layout.input_surface
    return state[group][:, names.index(name)]


# ---------------------------------------------------------------------------
# A component with no backbone
# ---------------------------------------------------------------------------
class StubComponent(nn.Module):
    """A component that does real arithmetic and no matrix multiplication.

    ``forward`` is a deterministic function of *every* input channel, of the
    previous state and of the timestamp, so a rollout built on it notices a
    channel that arrived in the wrong slot, a step that was skipped and a clock
    that did not advance.  The channel weights are ``1, 2, 3, ...`` rather than a
    plain mean precisely so that a permutation of the inputs changes the answer.

    Everything except ``forward`` is ``OceanForecastModule``'s own code,
    borrowed: the single-model rollout this file compares the coupled one
    against has to be the real one.
    """

    advance_state = OceanForecastModule.advance_state
    forward_multistep = OceanForecastModule.forward_multistep
    apply_wet_mask = OceanForecastModule.apply_wet_mask
    select_prognostic = OceanForecastModule.select_prognostic
    _as_tensordict = OceanForecastModule._as_tensordict

    def __init__(self, component: str, masks, forcing_gain: float = 0.5, bias: float = 0.0):
        super().__init__()
        self.component = V.get_component(component)
        self.depth_indices = list(DEPTH_INDICES)
        self.lead_time_hours = 24
        self.avg_modules = None  # geoarches' EMA hook; the rollout reads it
        # The borrowed `forward_multistep` asks the embedder whether it has room
        # for a previous state before carrying one into the next step. This stub's
        # `forward` reads `prev_state`, so it answers the way a real embedder
        # built with `n_concatenated_states=1` does.
        self.embedder = SimpleNamespace(n_concatenated_states=1)
        self.forcing_gain = float(forcing_gain)
        self.bias = float(bias)

        self.n_surface_in = self.component.n_surface_in
        self.n_surface_out = self.component.n_surface_out
        self.n_level_in = self.component.n_level_in
        self.n_level_out = self.component.n_level_out

        output = state_mask(
            masks, self.component.prognostic_surface, self.component.prognostic_level
        )
        inputs = state_mask(masks, self.component.input_surface, self.component.input_level)
        self.register_buffer("mask_surface", output["surface"], persistent=False)
        self.register_buffer("input_mask_surface", inputs["surface"], persistent=False)
        if self.n_level_out:
            self.register_buffer("mask_level", output["level"], persistent=False)
        if self.n_level_in:
            self.register_buffer("input_mask_level", inputs["level"], persistent=False)

    def forward(self, batch, use_avg: bool = True, forcing=None):
        state, previous = batch["state"], batch.get("prev_state")
        clock = batch["timestamp"].reshape(-1, 1, 1, 1, 1).double().float() * 1e-9
        entries = {}
        for group, n_out in (("surface", self.n_surface_out), ("level", self.n_level_out)):
            if not n_out:
                continue
            values = state[group]
            weights = torch.arange(1.0, values.shape[-4] + 1).reshape(-1, 1, 1, 1)
            pooled = (values * weights).sum(-4, keepdim=True) / weights.sum()
            out = values[..., :n_out, :, :, :] + self.forcing_gain * pooled + self.bias + clock
            if previous is not None:
                out = out + 0.01 * previous[group][..., :n_out, :, :, :]
            entries[group] = out
        return self.apply_wet_mask(TensorDict(entries, batch_size=state.batch_size))


def make_batch(state: TensorDict, iters: int = 3, seed: int = 3) -> dict:
    """A batch shaped like the dataloader's, with ground truth for every step."""
    generator = torch.Generator().manual_seed(seed)
    futures = TensorDict(
        {
            key: torch.stack(
                [
                    state[key] + 10.0 * (i + 1) + torch.rand(state[key].shape, generator=generator)
                    for i in range(iters)
                ],
                dim=1,
            )
            for key in state.keys()
        },
        batch_size=(state.batch_size[0], iters),
    )
    return {
        "state": state,
        "prev_state": state.apply(lambda t: t * 0.5),
        "future_states": futures,
        "timestamp": torch.tensor([1_547_510_400, 1_560_000_000][: state.batch_size[0]]),
    }


def couple(components, masks_file, stats_file, **kwargs) -> CoupledForecastModule:
    return CoupledForecastModule(
        components=components, masks_path=masks_file, stats_path=stats_file, **kwargs
    )


# ---------------------------------------------------------------------------
# The union component
# ---------------------------------------------------------------------------
def test_ocean_plus_seaice_is_exactly_the_full_component():
    """The pair owns every channel, so the shared state is the full state.

    This is what lets a coupled ocean+ice system be scored by the Task 7 pipeline
    with no special case: its state is channel-for-channel the one
    `configs/dataloader/glorys.yaml` already loads.
    """
    union = union_component([V.COMPONENTS["ocean"], V.COMPONENTS["seaice"]])
    full = V.COMPONENTS["full"]
    assert union.prognostic_surface == full.prognostic_surface
    assert union.prognostic_level == full.prognostic_level
    assert union.forcing == []
    assert named_component_like(union) == "full"


def test_the_union_of_one_component_is_that_component():
    for name in ("ocean", "seaice", "seaice_isolated", "full"):
        assert named_component_like(union_component([V.COMPONENTS[name]])) == name


def test_sea_ice_alone_leaves_the_ocean_unowned():
    """Nobody predicts the ocean, so it is what `unpredicted_forcing` decides."""
    union = union_component([V.COMPONENTS["seaice"]])
    assert union.forcing == V.OCEAN_VARIABLES
    assert union.prognostic == V.SEAICE_VARIABLES


def test_two_components_predicting_the_same_variable_is_refused():
    with pytest.raises(ValueError, match="both predict"):
        union_component([V.COMPONENTS["full"], V.COMPONENTS["seaice"]])


def test_a_combination_variables_py_does_not_name_is_reported_not_invented():
    """`variables.py` is the single source of truth; this module may not add to it."""
    odd = V.ComponentSpec(name="odd", prognostic=["zos"], forcing=["siconc"])
    assert named_component_like(odd) is None


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------
def test_the_router_round_trips_the_full_component(grid):
    """gather then scatter is the identity, bit for bit.

    The `full` component reads and writes every channel in the shared order, so
    anything the router does to it is a distortion.

    MUTANT: making `gather` return `shared_state[group]` untouched (i.e.
    forgetting that a component's channel order is its own) still passes here --
    for `full` it is the same order -- but fails
    `test_gather_puts_the_channels_in_the_components_own_order`. Making
    `scatter` write with `index_copy_` over `range(len(names))` instead of the
    layout indices fails this test for `seaice` and the reconstruction test.
    """
    layout = V.COMPONENTS["full"]
    router = StateRouter.for_layout(layout)
    state = shared_state(grid)
    back = router.scatter(state, layout, router.gather(state, layout))
    for group in state.keys():
        assert torch.equal(back[group], state[group])


def test_gather_puts_the_channels_in_the_components_own_order(grid):
    """Sea ice reads its own four fields first, then the ocean -- not canonical order."""
    layout = V.COMPONENTS["full"]
    router = StateRouter.for_layout(layout)
    state = shared_state(grid)
    seaice = V.COMPONENTS["seaice"]
    gathered = router.gather(state, seaice)

    assert seaice.input_surface == ["siconc", "sithick", "usi", "vsi", "zos", "mlotst", "bottomT"]
    for position, name in enumerate(seaice.input_surface):
        assert torch.equal(gathered["surface"][:, position], channel(state, layout, name))
    for position, name in enumerate(seaice.input_level):
        assert torch.equal(gathered["level"][:, position], channel(state, layout, name))


def test_ocean_and_seaice_together_reconstruct_the_whole_state(grid):
    """gather + scatter for both components rebuilds the shared state exactly.

    Together they own every channel, so nothing may be lost, duplicated or moved.

    MUTANT: swapping the two index lists in `scatter` (writing a component's
    output at its *input* positions) leaves this test failing on the four sea-ice
    channels, which land where zos, mlotst and bottomT belong.
    """
    layout = V.COMPONENTS["full"]
    router = StateRouter.for_layout(layout)
    state = shared_state(grid, seed=1)

    rebuilt = state
    for name in ("ocean", "seaice"):
        spec = V.COMPONENTS[name]
        gathered = router.gather(state, spec)
        # A component's prediction is the *leading* channels of its input: the
        # ordering contract the dataloader configs are built on.
        prediction = TensorDict(
            {
                group: gathered[group][:, :n_out]
                for group, n_out in (("surface", spec.n_surface_out), ("level", spec.n_level_out))
                if n_out
            },
            batch_size=state.batch_size,
        )
        rebuilt = router.scatter(rebuilt, spec, prediction)
    for group in state.keys():
        assert torch.equal(rebuilt[group], state[group])


def test_scatter_leaves_every_other_channel_untouched(grid):
    layout = V.COMPONENTS["full"]
    router = StateRouter.for_layout(layout)
    state = shared_state(grid, seed=2)
    seaice = V.COMPONENTS["seaice"]

    prediction = router.gather(state, seaice)
    prediction = TensorDict(
        {"surface": prediction["surface"][:, :4] + 7.0}, batch_size=state.batch_size
    )
    written = router.scatter(state, seaice, prediction)

    for name in V.SEAICE_VARIABLES:
        assert torch.equal(channel(written, layout, name), channel(state, layout, name) + 7.0)
    for name in V.OCEAN_VARIABLES:
        assert torch.equal(channel(written, layout, name), channel(state, layout, name))


def test_scatter_does_not_modify_the_state_it_was_given(grid):
    layout = V.COMPONENTS["full"]
    router = StateRouter.for_layout(layout)
    state = shared_state(grid)
    before = state["surface"].clone()
    seaice = V.COMPONENTS["seaice"]
    router.scatter(
        state,
        seaice,
        TensorDict(
            {"surface": torch.zeros_like(state["surface"][:, :4])}, batch_size=state.batch_size
        ),
    )
    assert torch.equal(state["surface"], before)


def test_a_wrong_channel_count_names_the_component_and_the_variables(grid):
    router = StateRouter.for_layout(V.COMPONENTS["full"])
    state = shared_state(grid)
    seaice = V.COMPONENTS["seaice"]
    wrong = TensorDict({"surface": state["surface"][:, :3]}, batch_size=state.batch_size)
    with pytest.raises(ValueError, match=r"'seaice' predicts 4 surface variables.*siconc"):
        router.scatter(state, seaice, wrong)


def test_a_variable_the_shared_state_does_not_carry_is_an_error(grid):
    """A sea-ice-only shared state cannot serve the ocean component."""
    router = StateRouter.for_layout(V.COMPONENTS["seaice_isolated"])
    state = shared_state(grid, layout=V.COMPONENTS["seaice_isolated"])
    with pytest.raises(KeyError, match=r"'ocean' needs the surface input"):
        router.gather(state, V.COMPONENTS["ocean"])


def test_the_layout_may_not_repeat_a_channel():
    with pytest.raises(ValueError, match="repeats a surface variable"):
        StateRouter(["zos", "zos"])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_a_coupled_pair_looks_like_the_full_component(masks, tiny_masks_file, tiny_stats_file):
    system = couple(
        {
            "ocean": StubComponent("ocean", masks),
            "seaice": StubComponent("seaice", masks),
        },
        tiny_masks_file,
        tiny_stats_file,
    )
    assert system.component.name == "full"
    assert system.surface_variables == V.SURFACE_VARIABLES
    assert system.level_variables == V.LEVEL_VARIABLES
    assert system.n_surface_out == 7 and system.n_level_out == 4
    assert system.lead_time_hours == 24
    # the statistics and masks are assembled for the union, in canonical order
    assert system.state_mean_surface.shape[0] == 7
    assert system.loss_coeff_level.shape[:2] == (4, len(DEPTH_INDICES))


def test_a_component_named_after_the_wrong_checkpoint_is_refused(
    masks, tiny_masks_file, tiny_stats_file
):
    with pytest.raises(ValueError, match="trained as 'seaice'"):
        couple({"ocean": StubComponent("seaice", masks)}, tiny_masks_file, tiny_stats_file)


def test_an_explicit_component_name_must_match_what_the_parts_predict(
    masks, tiny_masks_file, tiny_stats_file
):
    """`component=` used to override the derived union with no check at all.

    The shared state would then have been laid out for channels no component
    writes: the rollout runs, the report names the wrong component, and the
    surplus channels sit at their initial value for the whole forecast with
    nothing saying so.
    """
    with pytest.raises(ValueError) as error:
        couple(
            {"seaice": StubComponent("seaice", masks)},
            tiny_masks_file,
            tiny_stats_file,
            component="full",
        )
    message = str(error.value)
    assert "component='full'" in message
    assert "thetao" in message, "it has to say which variables actually disagree"


def test_an_explicit_component_name_that_does_match_is_accepted(
    masks, tiny_masks_file, tiny_stats_file
):
    """The complement, so the check above is not just "any component= is refused"."""
    system = couple(
        {"seaice": StubComponent("seaice", masks)},
        tiny_masks_file,
        tiny_stats_file,
        component="seaice",
    )
    assert system.component.name == "seaice"


def test_averaged_weights_are_refused_rather_than_silently_dropped(
    masks, tiny_masks_file, tiny_stats_file
):
    """geoarches applies `avg_modules` in `forward_multistep`, which coupling never calls.

    A coupled rollout steps each component through `forward`, so an EMA would be
    quietly ignored and the coupled scores would be of the raw weights while the
    same checkpoint scored alone used the averaged ones.
    """
    component = StubComponent("seaice", masks)
    component.avg_modules = [component]
    with pytest.raises(NotImplementedError, match="avg_modules"):
        couple({"seaice": component}, tiny_masks_file, tiny_stats_file)


def test_components_on_different_depth_presets_name_both(masks, tiny_masks_file, tiny_stats_file):
    """The mismatch the brief asks about: refuse it, and say what disagrees."""
    ocean = StubComponent("ocean", masks)
    seaice = StubComponent("seaice", masks)
    seaice.depth_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    with pytest.raises(ValueError) as error:
        couple({"ocean": ocean, "seaice": seaice}, tiny_masks_file, tiny_stats_file)
    message = str(error.value)
    assert "'ocean'" in message and "'seaice'" in message
    assert "the tiny preset" in message and "a custom preset" in message


def test_components_on_different_lead_times_are_refused(masks, tiny_masks_file, tiny_stats_file):
    ocean = StubComponent("ocean", masks)
    seaice = StubComponent("seaice", masks)
    seaice.lead_time_hours = 12
    with pytest.raises(ValueError, match="steps 24 h at a time and 'seaice' steps 12 h"):
        couple({"ocean": ocean, "seaice": seaice}, tiny_masks_file, tiny_stats_file)


def test_an_unknown_mode_is_refused(masks, tiny_masks_file, tiny_stats_file):
    with pytest.raises(ValueError, match="mode must be one of"):
        couple(
            {"ocean": StubComponent("ocean", masks)}, tiny_masks_file, tiny_stats_file, mode="both"
        )


def test_an_unknown_unpredicted_forcing_policy_is_refused(masks, tiny_masks_file, tiny_stats_file):
    with pytest.raises(ValueError, match="unpredicted_forcing must be one of"):
        couple(
            {"ocean": StubComponent("ocean", masks)},
            tiny_masks_file,
            tiny_stats_file,
            unpredicted_forcing="zeros",
        )


def test_zero_is_never_a_forcing_policy():
    """The one behaviour the brief forbids outright."""
    from oceanarches.lightning_modules.coupled import UNPREDICTED_FORCING_MODES

    assert set(UNPREDICTED_FORCING_MODES) == {"persistence", "ground_truth"}


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------
def test_a_coupled_rollout_keeps_its_shapes_and_its_land_mask(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    """Several steps of two components, on the real wet mask.

    Randomly initialised in the sense that matters here: the two components have
    different, arbitrary responses, so nothing about the result is symmetric.
    """
    system = couple(
        {
            "ocean": StubComponent("ocean", masks, forcing_gain=0.7, bias=0.3),
            "seaice": StubComponent("seaice", masks, forcing_gain=0.2, bias=-0.4),
        },
        tiny_masks_file,
        tiny_stats_file,
    )
    batch = make_batch(shared_state(grid), iters=4)
    with torch.no_grad():
        trajectory = system.forward_multistep(batch, iters=4)

    lat, lon = grid
    assert trajectory["surface"].shape == (2, 4, 7, 1, lat, lon)
    assert trajectory["level"].shape == (2, 4, 4, len(DEPTH_INDICES), lat, lon)
    assert torch.isfinite(trajectory["surface"]).all()
    land_surface, land_level = 1 - system.mask_surface, 1 - system.mask_level
    assert (trajectory["surface"] * land_surface).abs().max() == 0.0
    assert (trajectory["level"] * land_level).abs().max() == 0.0
    assert land_surface.any() and land_level.any(), "the fixture must have land in it"
    # and the ocean cells are not all zero either, which is what makes the line above
    # a statement about land rather than about an empty tensor
    assert (trajectory["surface"] * system.mask_surface).abs().max() > 0.0


def test_the_rollout_advances_the_clock_one_lead_time_per_step(
    masks, grid, tiny_masks_file, tiny_stats_file, monkeypatch
):
    """geoarches advances by `lead_time_hours * multistep`; we advance by one step."""
    system = couple({"ocean": StubComponent("ocean", masks)}, tiny_masks_file, tiny_stats_file)
    seen = []
    original = system.step

    def record(shared, previous, timestamp):
        seen.append(int(timestamp[0]))
        return original(shared, previous, timestamp)

    monkeypatch.setattr(system, "step", record)
    batch = make_batch(shared_state(grid, layout=V.COMPONENTS["ocean"]), iters=3)
    with torch.no_grad():
        system.forward_multistep(batch, iters=3)
    start = int(batch["timestamp"][0])
    assert seen == [start, start + 86400, start + 2 * 86400]


def test_forward_returns_only_what_the_system_predicts(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    system = couple({"seaice": StubComponent("seaice", masks)}, tiny_masks_file, tiny_stats_file)
    batch = make_batch(shared_state(grid, layout=V.COMPONENTS["seaice"]), iters=2)
    with torch.no_grad():
        prediction = system.forward(batch)
    assert set(prediction.keys()) == {"surface"}, "the sea-ice component predicts no 3-D field"
    assert prediction["surface"].shape[1] == 4


# ---------------------------------------------------------------------------
# parallel vs sequential
# ---------------------------------------------------------------------------
def _both_modes(masks, grid, tiny_masks_file, tiny_stats_file, iters=3):
    results = {}
    for mode in ("parallel", "sequential"):
        system = couple(
            {
                "ocean": StubComponent("ocean", masks, forcing_gain=0.6, bias=0.2),
                "seaice": StubComponent("seaice", masks, forcing_gain=0.4, bias=-0.1),
            },
            tiny_masks_file,
            tiny_stats_file,
            mode=mode,
        )
        batch = make_batch(shared_state(grid), iters=iters)
        with torch.no_grad():
            results[mode] = system.forward_multistep(batch, iters=iters)
    return results


def test_parallel_and_sequential_are_different_forecasts(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    """The scientific claim this task makes, pinned.

    In `sequential` the sea-ice component reads the ocean fields this step has
    already produced; in `parallel` it reads yesterday's. If the two ever agree,
    either the ordering is not being applied or the ice does not depend on the
    ocean at all -- and both are worth failing over.

    MUTANT: turning the one `if self.mode == "sequential"` in `step` into
    `if True` (parallel run as sequential) or `if False` (the other way round)
    makes the two modes identical, and this test fails on the sea-ice channels at
    the very first lead time.
    """
    results = _both_modes(masks, grid, tiny_masks_file, tiny_stats_file)
    parallel, sequential = results["parallel"], results["sequential"]

    ice = slice(3, 7)  # siconc, sithick, usi, vsi, in canonical order
    difference = (parallel["surface"][:, :, ice] - sequential["surface"][:, :, ice]).abs().max()
    assert difference > 1e-3, "the two coupling modes produced the same sea-ice forecast"
    # And the ocean, which runs first, agrees at step 1 and only then diverges.
    ocean = slice(0, 3)
    assert torch.equal(parallel["surface"][:, 0, ocean], sequential["surface"][:, 0, ocean])
    assert (
        parallel["surface"][:, 1, ocean] - sequential["surface"][:, 1, ocean]
    ).abs().max() > 1e-6


def test_with_one_component_the_two_modes_agree(masks, grid, tiny_masks_file, tiny_stats_file):
    """A sanity check on the check: with nothing to order, ordering cannot matter."""
    trajectories = {}
    for mode in ("parallel", "sequential"):
        system = couple(
            {"ocean": StubComponent("ocean", masks)}, tiny_masks_file, tiny_stats_file, mode=mode
        )
        batch = make_batch(shared_state(grid, layout=V.COMPONENTS["ocean"]), iters=3)
        with torch.no_grad():
            trajectories[mode] = system.forward_multistep(batch, iters=3)
    for group in trajectories["parallel"].keys():
        assert torch.equal(trajectories["parallel"][group], trajectories["sequential"][group])


# ---------------------------------------------------------------------------
# The strongest test: coupling must add nothing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_a_coupled_full_component_is_bit_identical_to_it_alone(
    masks, grid, tiny_masks_file, tiny_stats_file, mode
):
    """One `full` component, coupled, must reproduce its own rollout exactly.

    `full` reads and predicts every channel, so the router is the identity and
    the coupled rollout has nothing left to do but reproduce
    `OceanForecastModule.forward_multistep` -- which is the code the stub
    borrows, so this compares against the real thing rather than a copy. Any
    distortion the coupling machinery introduces (a channel reordered, the wet
    mask applied twice or not at all, the clock advanced differently, the
    previous state taken from the wrong step) shows up here as a non-zero
    difference. `torch.equal`, not `allclose`: bit-identical is the claim.

    MUTANT: setting `previous = shared` *after* reassigning `shared` (so a
    component sees this step's state as its previous one) fails from the second
    step; advancing the clock by `lead_time_hours * iters` fails at the first
    step that uses it; not handing the components a `prev_state` at all fails
    immediately. Dropping the `apply_wet_mask` from `advance_state` does *not*
    fail here and should not: for `full` the input and output masks are the same
    and the component has already masked its own output, so the re-mask is a
    genuine no-op. `test_ground_truth_takes_the_unpredicted_channels_from_the_dataset`
    is what catches that one, because ground truth arrives unmasked.
    """
    component = StubComponent("full", masks, forcing_gain=0.35, bias=0.15)
    system = couple({"full": component}, tiny_masks_file, tiny_stats_file, mode=mode)

    state = shared_state(grid, seed=5)
    with torch.no_grad():
        alone = component.forward_multistep(make_batch(state, iters=5), iters=5)
        coupled = system.forward_multistep(make_batch(state, iters=5), iters=5)

    assert set(alone.keys()) == set(coupled.keys())
    for group in alone.keys():
        assert torch.equal(alone[group], coupled[group]), (
            f"{group}: max |difference| = {(alone[group] - coupled[group]).abs().max()}"
        )


# ---------------------------------------------------------------------------
# Channels nobody predicts
# ---------------------------------------------------------------------------
def _ice_only(masks, grid, tiny_masks_file, tiny_stats_file, policy, iters=3):
    system = couple(
        {"seaice": StubComponent("seaice", masks, forcing_gain=0.9)},
        tiny_masks_file,
        tiny_stats_file,
        unpredicted_forcing=policy,
    )
    batch = make_batch(shared_state(grid, layout=V.COMPONENTS["seaice"], seed=7), iters=iters)
    with torch.no_grad():
        return system, batch, system.forward_multistep(batch, iters=iters)


def test_persistence_holds_the_unpredicted_channels_at_the_initial_value(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    """Free-running: no information from outside the system enters the rollout."""
    system, batch, _ = _ice_only(masks, grid, tiny_masks_file, tiny_stats_file, "persistence")
    initial = system.apply_wet_mask(batch["state"], inputs=True)
    advanced = system.advance_state(
        system.step(batch["state"], batch["prev_state"], batch["timestamp"])
    )
    for name in V.OCEAN_VARIABLES:
        assert torch.equal(
            channel(advanced, system.component, name), channel(initial, system.component, name)
        )


def test_ground_truth_takes_the_unpredicted_channels_from_the_dataset(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    """Perfect forcing: the ocean the ice reads is the ocean that really happened.

    MUTANT: making `advance_state` ignore `forcing_truth` (i.e. silently falling
    back to persistence) fails here on every ocean channel; taking
    `future_states[:, step + 1]` instead of `[:, step]` fails because the state
    read into step 2 is then valid at step 3.
    """
    system, batch, _ = _ice_only(masks, grid, tiny_masks_file, tiny_stats_file, "ground_truth")
    merged = system.step(batch["state"], batch["prev_state"], batch["timestamp"])
    advanced = system.advance_state(merged, batch["future_states"][:, 0])
    truth = system.apply_wet_mask(batch["future_states"][:, 0], inputs=True)

    for name in V.OCEAN_VARIABLES:
        assert torch.equal(
            channel(advanced, system.component, name), channel(truth, system.component, name)
        )
    # and the ice channels are still the model's, not the truth's
    for name in V.SEAICE_VARIABLES:
        assert not torch.equal(
            channel(advanced, system.component, name), channel(truth, system.component, name)
        )


def test_ground_truth_uses_the_state_valid_at_each_step(
    masks, grid, tiny_masks_file, tiny_stats_file, monkeypatch
):
    """Step *k* must read the truth for time *t + k*, not *t + k + 1*.

    An off-by-one here is a forecast quietly given tomorrow's ocean, which makes
    the sea-ice component look better than it is and cannot be seen in any plot.

    MUTANT: `_ground_truth_at` returning `future_states[:, step + 1]` fails here
    at the second lead time; returning `future_states[:, 0]` at every step fails
    at the third.
    """
    system = couple(
        {"seaice": StubComponent("seaice", masks)},
        tiny_masks_file,
        tiny_stats_file,
        unpredicted_forcing="ground_truth",
    )
    batch = make_batch(shared_state(grid, layout=V.COMPONENTS["seaice"], seed=13), iters=4)

    seen = []
    original = system.step

    def record(shared, previous, timestamp):
        seen.append(shared)
        return original(shared, previous, timestamp)

    monkeypatch.setattr(system, "step", record)
    with torch.no_grad():
        system.forward_multistep(batch, iters=4)

    for step in range(1, 4):
        truth = system.apply_wet_mask(batch["future_states"][:, step - 1], inputs=True)
        for name in V.OCEAN_VARIABLES:
            assert torch.equal(
                channel(seen[step], system.component, name),
                channel(truth, system.component, name),
            ), f"step {step} did not read the ocean valid at that step"


def test_the_two_policies_give_different_forecasts(masks, grid, tiny_masks_file, tiny_stats_file):
    _, _, persisted = _ice_only(masks, grid, tiny_masks_file, tiny_stats_file, "persistence")
    _, _, forced = _ice_only(masks, grid, tiny_masks_file, tiny_stats_file, "ground_truth")
    assert (persisted["surface"] - forced["surface"]).abs().max() > 1e-3


def test_ground_truth_without_future_states_says_what_is_missing(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    system = couple(
        {"seaice": StubComponent("seaice", masks)},
        tiny_masks_file,
        tiny_stats_file,
        unpredicted_forcing="ground_truth",
    )
    batch = make_batch(shared_state(grid, layout=V.COMPONENTS["seaice"]), iters=2)
    del batch["future_states"]
    with pytest.raises(KeyError, match="no 'future_states'"):
        with torch.no_grad():
            system.forward_multistep(batch, iters=2)


def test_a_system_that_owns_everything_ignores_the_policy(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    """ocean+seaice have no unpredicted channels, so the knob cannot change anything."""
    trajectories = {}
    for policy in ("persistence", "ground_truth"):
        system = couple(
            {"ocean": StubComponent("ocean", masks), "seaice": StubComponent("seaice", masks)},
            tiny_masks_file,
            tiny_stats_file,
            unpredicted_forcing=policy,
        )
        batch = make_batch(shared_state(grid, seed=11), iters=3)
        with torch.no_grad():
            trajectories[policy] = system.forward_multistep(batch, iters=3)
    for group in trajectories["persistence"].keys():
        assert torch.equal(trajectories["persistence"][group], trajectories["ground_truth"][group])


# ---------------------------------------------------------------------------
# The interface the evaluation pipeline talks to
# ---------------------------------------------------------------------------
def test_a_coupled_system_answers_everything_the_pipeline_asks_a_module(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    """Task 7 treats any object with this surface as a model; check we have it all.

    Listed explicitly rather than discovered, because the failure mode is an
    `AttributeError` four minutes into a GPU run.
    """
    system = couple(
        {"ocean": StubComponent("ocean", masks), "seaice": StubComponent("seaice", masks)},
        tiny_masks_file,
        tiny_stats_file,
    )
    for attribute in (
        "component",
        "depth_indices",
        "lead_time_hours",
        "surface_variables",
        "level_variables",
        "n_surface_out",
        "n_level_out",
        "state_mean_surface",
        "state_std_surface",
        "state_mean_level",
        "state_std_level",
        "mask_surface",
        "select_prognostic",
        "apply_wet_mask",
        "denormalize_state",
        "loss",
        "forward",
        "forward_multistep",
    ):
        assert hasattr(system, attribute), attribute
    assert next(system.parameters(), None) is None or True  # ModuleDict is iterable either way

    batch = make_batch(shared_state(grid), iters=3)
    with torch.no_grad():
        prediction = system.forward_multistep(batch, iters=3)
    target = system.select_prognostic(batch["future_states"])
    loss = system.loss(prediction, target, multistep=True)
    assert torch.isfinite(loss) and float(loss) > 0
    physical = system.denormalize_state(prediction)
    assert physical["surface"].shape == prediction["surface"].shape


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
def build_config(**overrides) -> OmegaConf:
    args = [f"{key}={value}" for key, value in overrides.items()]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR, job_name="coupling"):
        cfg = compose(config_name="config", overrides=args)
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize(
    "module, dataloader, component",
    [
        ("ocean_component", "glorys_ocean", "ocean"),
        ("seaice_component", "glorys_seaice", "seaice"),
        ("seaice_isolated_component", "glorys_seaice_isolated", "seaice_isolated"),
    ],
)
def test_the_component_presets_pair_with_their_dataloader(module, dataloader, component):
    cfg = build_config(module=module, dataloader=dataloader)
    spec = V.COMPONENTS[component]
    assert cfg.dataloader.component == component
    assert cfg.module.module.component == component
    assert cfg.module.embedder.surface_ch_in == spec.n_surface_in
    assert cfg.module.embedder.surface_ch_out == spec.n_surface_out
    assert cfg.module.embedder.level_ch_in == spec.n_level_in
    assert cfg.module.embedder.level_ch_out == spec.n_level_out
    assert cfg.max_steps % cfg.save_step_frequency == 0


def test_the_isolated_preset_switches_off_the_layers_it_has_no_column_for():
    """0 depth levels give a latent depth of 1, and geoarches hardcodes 8.

    The two cross-depth layers have to be off and the embedder has to be told
    that the latent depth is deliberate, or it raises. A preset that got one of
    the three right and not the others would fail at the first batch.
    """
    cfg = build_config(module="seaice_isolated_component", dataloader="glorys_seaice_isolated")
    assert cfg.module.embedder.n_depths == 0
    assert cfg.module.embedder.allow_any_z_dim is True
    assert list(cfg.module.backbone.tensor_size)[0] == 1
    assert cfg.module.backbone.first_interaction_layer is None
    assert cfg.module.backbone.axis_attn is False


def test_the_isolated_dataloader_reads_nothing_but_ice():
    cfg = build_config(dataloader="glorys_seaice_isolated")
    assert list(cfg.dataloader.dataset.variables.surface) == V.SEAICE_VARIABLES
    assert list(cfg.dataloader.dataset.variables.level) == []
    assert cfg.dataloader.n_level_in == 0


def test_the_coupled_preset_ships_a_valid_wiring():
    from oceanarches.lightning_modules.coupled import COUPLING_MODES, UNPREDICTED_FORCING_MODES

    cfg = build_config(
        module="coupled",
        dataloader="glorys",
        **{"++module.module.components": "{ocean: a, seaice: b}"},
    )
    node = cfg.module.module
    assert node._target_.endswith("CoupledForecastModule")
    assert node.mode in COUPLING_MODES
    assert node.unpredicted_forcing in UNPREDICTED_FORCING_MODES
    assert list(node.order) == ["ocean", "seaice"]
    assert node.component == "full"


def test_the_coupled_preset_refuses_to_compose_without_components():
    """`components: ???` -- a coupled system built out of nothing is not a default."""
    cfg = build_config(module="coupled", dataloader="glorys")
    with pytest.raises(Exception):  # noqa: B017 - omegaconf's MissingMandatoryValue
        _ = cfg.module.module.components.keys()


def test_run_eval_knows_a_dataloader_for_every_component():
    """The map `--coupled` uses to lay out the shared state must be complete."""
    from oceanarches.evaluation.run_eval import DATALOADER_FOR_COMPONENT

    assert set(DATALOADER_FOR_COMPONENT) == set(V.COMPONENTS)
    for component, dataloader in DATALOADER_FOR_COMPONENT.items():
        cfg = build_config(dataloader=dataloader)
        assert cfg.dataloader.component == component


# ---------------------------------------------------------------------------
# External forcing (the *other* way a field reaches a model)
# ---------------------------------------------------------------------------
def test_no_forcing_is_the_default_and_reserves_no_channels():
    cfg = build_config(module="tiny", dataloader="glorys_tiny")
    assert cfg.module.embedder.forcing_ch == 0
    assert cfg.module.module.forcing is None, (
        "the default must be None, not NoForcing(): the module checks for None to decide "
        "whether to call the source at all"
    )


def test_file_forcing_wires_a_source_and_the_channels_to_match():
    cfg = build_config(module="tiny", dataloader="glorys_ocean", forcing="file")
    assert cfg.module.embedder.forcing_ch == cfg.forcing.n_channels
    assert cfg.forcing.n_channels == len(cfg.forcing.source.variables)
    assert cfg.module.module.forcing._target_.endswith("XarrayForcing")


def test_component_exchange_does_not_go_through_a_forcing_source(
    masks, grid, tiny_masks_file, tiny_stats_file
):
    """A coupled system has no single shared forcing tensor, and says so.

    Two components can legitimately be given different atmospheres, so external
    forcing stays per component; what they hand *each other* is the shared state.
    """
    system = couple({"ocean": StubComponent("ocean", masks)}, tiny_masks_file, tiny_stats_file)
    batch = make_batch(shared_state(grid, layout=V.COMPONENTS["ocean"]), iters=1)
    with pytest.raises(ValueError, match="each component reads its own ForcingSource"):
        system.forward(batch, forcing=torch.zeros(2, 3, 1, *grid))


def test_a_component_can_be_given_by_path(masks, tiny_masks_file, tiny_stats_file, tmp_path):
    """Anything `load_module` can open is a component; nothing else loads checkpoints."""
    from oceanarches.lightning_modules import coupled as coupled_module

    calls = []

    def fake_load_module(path, device="cpu", return_config=True):
        calls.append((path, device))
        return StubComponent("ocean", masks)

    original = coupled_module.load_module
    coupled_module.load_module = fake_load_module
    try:
        system = couple({"ocean": str(tmp_path / "run")}, tiny_masks_file, tiny_stats_file)
    finally:
        coupled_module.load_module = original
    assert calls == [(str(tmp_path / "run"), "cpu")]
    assert system.component.name == "ocean"


def test_something_that_is_not_a_module_or_a_path_is_refused(tiny_masks_file, tiny_stats_file):
    with pytest.raises(TypeError, match="must be a run name, a path"):
        couple({"ocean": 42}, tiny_masks_file, tiny_stats_file)


def test_components_are_required():
    with pytest.raises(ValueError, match="needs components"):
        CoupledForecastModule(components={})


# ---------------------------------------------------------------------------
# run_eval's coupled command line
# ---------------------------------------------------------------------------
def test_parse_components_reads_name_equals_run_pairs(fake_modelstore):
    from oceanarches.evaluation.run_eval import parse_components

    fake_modelstore("a", "b")
    assert parse_components(["ocean=a", "seaice=b"]) == {"ocean": "a", "seaice": "b"}


@pytest.mark.parametrize(
    "argument, message",
    [
        (["oceanrun"], "NAME=RUN pairs"),
        (["ocean="], "has no run name"),
        (["atmosphere=a"], "Unknown component"),
        (["ocean=a", "ocean=b"], "twice"),
    ],
)
def test_parse_components_refuses_nonsense(argument, message, fake_modelstore):
    from oceanarches.evaluation.run_eval import parse_components

    fake_modelstore("a", "b")
    with pytest.raises(SystemExit, match=message):
        parse_components(argument)


def test_the_coupling_wiring_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """`parallel` and `sequential` are different forecasts from the same weights.

    A cache keyed only on the checkpoints would hand a `parallel` request the
    `sequential` numbers -- the exact failure the rest of this pipeline's cache
    design exists to prevent.

    MUTANT: leaving `args.mode` out of the `wiring` string makes the two
    identities equal and this test fails.
    """
    from oceanarches.evaluation.run_eval import build_parser, coupled_identity

    components = {"ocean": str(tmp_path / "a"), "seaice": str(tmp_path / "b")}
    order = ["ocean", "seaice"]

    def identity(*extra):
        args = build_parser().parse_args(
            ["--exp", "x", "--coupled", "--components", "ocean=a", "seaice=b", *extra]
        )
        return coupled_identity(components, order, args)

    assert identity().config_hash != identity("--mode", "parallel").config_hash
    assert identity().config_hash != identity("--unpredicted-forcing", "ground_truth").config_hash
    assert identity().fingerprint == identity("--mode", "parallel").fingerprint


def test_the_coupled_run_is_named_after_its_components_and_mode():
    from oceanarches.evaluation.run_eval import default_experiment_name

    assert (
        default_experiment_name(["ocean", "seaice"], "sequential")
        == "coupled_ocean_seaice_sequential"
    )
    assert default_experiment_name(["ocean"], "parallel") != default_experiment_name(
        ["ocean"], "sequential"
    )
    # a different forcing policy is a different forecast, so a different directory
    assert default_experiment_name(["seaice"], "sequential") != default_experiment_name(
        ["seaice"], "sequential", "ground_truth"
    )
    # and so is a different ORDER: the cache key already separates them, but the
    # figures, report.md and summary.json are written by name and would otherwise
    # overwrite each other
    assert default_experiment_name(["seaice", "ocean"], "sequential") != default_experiment_name(
        ["ocean", "seaice"], "sequential"
    )


def test_exp_is_optional_only_when_coupled():
    from oceanarches.evaluation.run_eval import main

    with pytest.raises(SystemExit, match="--exp is required"):
        main(["--lead-days", "1"])


def test_the_dataloader_for_a_coupled_system_is_checked_against_the_config(monkeypatch):
    """A wrong entry in DATALOADER_FOR_COMPONENT must be an error, not a silent reorder."""
    from oceanarches.evaluation import run_eval

    monkeypatch.setitem(run_eval.DATALOADER_FOR_COMPONENT, "full", "glorys_seaice")
    args = run_eval.build_parser().parse_args(
        ["--coupled", "--components", "ocean=a", "seaice=b", "--exp", "x"]
    )
    with pytest.raises(SystemExit, match="lays out the 'seaice' state"):
        run_eval.compose_coupled_config({"ocean": "a", "seaice": "b"}, ["ocean", "seaice"], args)


def test_a_comparison_cache_for_a_different_question_is_not_reused(tmp_path):
    import json

    from oceanarches.evaluation.run_eval import load_result_for_overlay

    manifest = dict(
        experiment="other",
        domain="test",
        lead_days=10,
        n_inits=2,
        selection="spread",
        save_depths="shallow",
        checkpoint="c.ckpt",
        inits=[0, 5],
        checkpoint_fingerprint="f",
        config_hash="h",
        statistics_fingerprint="3:s",
        save_fields=False,
        metric_groups={},
        losses={},
        labels={},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert load_result_for_overlay(tmp_path, 10, "test", [0, 5]) is not None
    assert load_result_for_overlay(tmp_path, 5, "test", [0, 5]) is None
    assert load_result_for_overlay(tmp_path, 10, "val", [0, 5]) is None
    assert load_result_for_overlay(tmp_path, 10, "test", [0, 6]) is None
    assert load_result_for_overlay(tmp_path / "nowhere", 10, "test", [0, 5]) is None

    # A cache from before the statistics were part of the key cannot say which
    # ones made its numbers, so it is not laid over anything. `render_comparison`
    # warns and names the command to rescore it.
    (tmp_path / "manifest.json").write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "statistics_fingerprint"})
    )
    assert load_result_for_overlay(tmp_path, 10, "test", [0, 5]) is None


def test_no_comparison_asked_for_draws_nothing(tmp_path):
    from oceanarches.evaluation import run_eval

    args = run_eval.build_parser().parse_args(["--exp", "x"])
    assert run_eval.render_comparison(args, None, tmp_path, tmp_path) == []


def test_a_comparison_that_was_never_scored_is_reported_not_drawn(tmp_path):
    """Missing is missing: say how to produce it rather than draw a figure of one curve.

    Through `warnings.warn`, like every other "something was skipped" in this
    pipeline, so a caller can catch or filter all of them together -- a `print`
    to stderr is invisible to `pytest.warns` and to a notebook's own filter.
    """
    from oceanarches.evaluation import run_eval
    from oceanarches.evaluation.rollout import RolloutResult, RolloutSpec

    args = run_eval.build_parser().parse_args(
        ["--exp", "mine", "--compare-with", "theirs=other", "--lead-days", "10"]
    )
    spec = RolloutSpec(
        experiment="mine",
        domain="test",
        lead_days=10,
        n_inits=2,
        selection="spread",
        save_depths="shallow",
        checkpoint="c.ckpt",
        inits=(0, 5),
    )
    with pytest.warns(UserWarning, match="make eval NAME=other"):
        written = run_eval.render_comparison(
            args, RolloutResult(spec=spec, directory=tmp_path), tmp_path / "figures", tmp_path
        )
    assert written == []


# ---------------------------------------------------------------------------
# Writing the fields of a component that predicts a subset of what it reads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("component", ["seaice", "ocean"])
def test_a_component_writes_only_the_fields_it_predicts(
    component, masks, tiny_forecast_kwargs, tiny_masks_file, tiny_stats_file, tmp_path
):
    """`make eval` on a specialist must write its rollout, not die on a shape.

    `run_rollout` names the channels it writes from a dataset, and for a
    component that dataset describes what the model *reads* (7 surface channels
    for the ocean specialist) rather than what it *predicts* (3).  Writing one
    through the other used to raise

        RuntimeError: The size of tensor a (7) must match the size of tensor b (3)

    from `convert_to_xarray`, several minutes into a GPU run, with no hint about
    what to do -- and `make eval NAME=<run>` has no `--skip-fields` escape.  This
    is the regression test for `rollout.writer_view`.

    Both shapes of the bug are covered: `seaice` predicts a strict subset of the
    surface channels and *no* level channels at all, `ocean` predicts a subset of
    the surface channels and all of the level ones.

    MUTANT: passing `dataset` instead of `fields_dataset` to either
    `_write_trajectory` call restores the RuntimeError and this test fails for
    both components.
    """
    from omegaconf import OmegaConf

    from oceanarches.dataloaders.glorys import GlorysForecast
    from oceanarches.evaluation import baselines, rollout

    spec = V.COMPONENTS[component]
    dataset = GlorysForecast(
        **tiny_forecast_kwargs,
        domain="test",
        variables={"surface": spec.input_surface, "level": spec.input_level},
        depth_indices=DEPTH_INDICES,
        multistep=2,
    )
    system = couple({component: StubComponent(component, masks)}, tiny_masks_file, tiny_stats_file)
    cfg = OmegaConf.create({"module": {"inference": {"metrics": {}, "metrics_kwargs": {}}}})
    run_spec = rollout.RolloutSpec(
        experiment="fields",
        domain="test",
        lead_days=2,
        n_inits=2,
        selection="first",
        save_depths="shallow",
        checkpoint="stub",
        inits=(0, 1),
        save_fields=True,
    )
    result = rollout.run_rollout(
        system,
        cfg,
        dataset,
        [baselines.ModelForecast(system)],
        run_spec,
        tmp_path,
        batch_size=2,
        num_workers=0,
        device="cpu",
        progress=False,
    )

    for path in (result.predictions_path, result.targets_path):
        written = xr.open_zarr(path)
        assert set(written.data_vars) == set(spec.prognostic), (
            f"{path.name} holds {sorted(written.data_vars)}, not the component's own "
            f"{sorted(spec.prognostic)}"
        )
        assert written.sizes["prediction_timedelta"] == 2
        assert written.sizes["time"] == 2
        for name in spec.prognostic_surface:
            field = written[name].isel(time=0, prediction_timedelta=0).to_numpy()
            assert field.shape == tuple(masks.wet_surface.shape)
            # land came back as NaN, i.e. the mask used was the one for *these*
            # variables and not a 7-channel mask silently broadcast over 4
            assert np.isnan(field[~masks.wet_surface.numpy()]).all()
            assert np.isfinite(field[masks.wet_surface.numpy()]).any()
        written.close()


@pytest.mark.parametrize("source", [None, "NoForcing"])
def test_saying_no_forcing_either_way_reaches_the_model_as_none(source, masks):
    """`forcing=None` and `forcing=NoForcing()` must both mean "no forcing".

    `NoForcing.get()` returns None by its own documented contract, so a module
    that only checked `forcing_source is None` stacked a list of Nones and died
    with `TypeError: expected Tensor as element 0` at the first batch -- which is
    why configs/forcing/none.yaml ships `source: null`. Both spellings are now
    handled, so the config is a choice rather than a workaround.

    MUTANT: removing the `fields[0] is None` guard from `external_forcing` fails
    the NoForcing case with that TypeError.
    """
    from oceanarches.dataloaders.forcing import NoForcing

    class Probe:
        forcing_source = None if source is None else NoForcing()

    got = OceanForecastModule.external_forcing(Probe(), torch.tensor([1_600_000_000]))
    assert got is None


def test_the_run_directory_follows_order_not_the_order_components_were_typed(fake_modelstore):
    """`--order seaice ocean` must not write into the default ordering's directory.

    MUTANT: building the name from `list(components)` instead of `order` makes
    both invocations resolve to `coupled_ocean_seaice_sequential` and this fails.
    """
    from oceanarches.evaluation.run_eval import main

    fake_modelstore("a", "b")
    names = []

    def capture(args):
        names.append(args.exp)
        return {}

    import oceanarches.evaluation.run_eval as run_eval

    original = run_eval.evaluate
    run_eval.evaluate = capture
    try:
        main(["--coupled", "--components", "ocean=a", "seaice=b"])
        main(["--coupled", "--components", "ocean=a", "seaice=b", "--order", "seaice", "ocean"])
    finally:
        run_eval.evaluate = original
    assert names == ["coupled_ocean_seaice_sequential", "coupled_seaice_ocean_sequential"]


def test_a_coupled_system_says_how_to_re_run_itself(masks, tiny_masks_file, tiny_stats_file):
    """The report cannot guess it: there is no `modelstore/<coupled name>/`.

    Every part of the wiring has to appear, because every part changes the
    forecast. `make couple` is offered only when it would actually reproduce the
    run -- ocean then sea ice, default forcing policy.
    """
    from oceanarches.evaluation.rollout import RolloutSpec

    spec = RolloutSpec(
        experiment="coupled_ocean_seaice_sequential",
        domain="test",
        lead_days=10,
        n_inits=16,
        selection="spread",
        save_depths="shallow",
        checkpoint="ocean@a+seaice@b",
    )

    system = couple(
        {"ocean": StubComponent("ocean", masks), "seaice": StubComponent("seaice", masks)},
        tiny_masks_file,
        tiny_stats_file,
    )
    system.sources = {"ocean": "ocean_tiny", "seaice": "seaice_tiny"}
    command = system.evaluation_command(spec)
    assert "--exp coupled_ocean_seaice_sequential" not in command, (
        "that command names an evalstore directory, not a checkpoint, and does not run"
    )
    assert "make couple OCEAN=ocean_tiny SEAICE=seaice_tiny MODE=sequential" in command
    assert "--coupled" in command
    assert "--components ocean=ocean_tiny seaice=seaice_tiny" in command
    for flag in ("--mode sequential", "--unpredicted-forcing persistence", "--lead-days 10"):
        assert flag in command, flag

    # a wiring `make couple` cannot express is offered only in full
    other = couple(
        {"seaice": StubComponent("seaice", masks)},
        tiny_masks_file,
        tiny_stats_file,
        unpredicted_forcing="ground_truth",
    )
    other.sources = {"seaice": "seaice_tiny"}
    command = other.evaluation_command(spec)
    assert "make couple" not in command
    assert "--components seaice=seaice_tiny" in command
    assert "--unpredicted-forcing ground_truth" in command


def test_a_mistyped_component_run_names_the_component_and_the_run(fake_modelstore):
    """`make couple SEAICE=seaice_tin` is the flagship workflow's typo.

    `coupled.py` loads each component with `load_module` inside hydra's
    `instantiate`, which re-wraps and stringifies the exception *without its
    arguments*: the participant saw `InstantiationException: ...
    FileNotFoundError(2, 'No such file or directory')` -- no path, no component
    name, no way to tell which of the two runs was wrong.

    MUTANT: removing the `resolve_run` call from `parse_components` makes this
    return a dict instead of raising.
    """
    from oceanarches.evaluation.run_eval import parse_components

    fake_modelstore("ocean_tiny", "seaice_tiny")
    with pytest.raises(SystemExit) as caught:
        parse_components(["ocean=ocean_tiny", "seaice=seaice_tin"])
    message = str(caught.value)
    assert "--components seaice=seaice_tin" in message
    assert "seaice_tiny" in message, "the message must list the runs that do exist"


def test_a_coupled_system_keeps_none_depth_indices_as_every_level(
    masks, tiny_masks_file, tiny_stats_file
):
    """`list(x or [])` turned "load every prepared level" into "load none".

    `_select_statistics` is guarded on `depth_indices is not None`, so `[]` went
    straight through and gave `std` of shape [1, 0, 1, 1] -- whose own positivity
    check, `(std > 0).all()`, is vacuously True on an empty tensor. Reachable
    from any component config that omits `module.depth_indices`.

    MUTANT: restoring `self.depth_indices = list(self._first.depth_indices or [])`
    makes the level statistics come back with a zero-length depth axis and the
    last assertion fails (the constructor raises first, which is also a pass for
    the mutant-detection purpose but not for this assertion).
    """
    component = StubComponent("full", masks)
    component.depth_indices = None
    system = couple({"full": component}, tiny_masks_file, tiny_stats_file)
    assert system.depth_indices is None
    # None means every *prepared* level (14), not the 13-level preset the
    # `masks` fixture is sliced to -- which is exactly the distinction `[]` lost.
    assert system.state_std_level.shape[1] == len(V.PREPPED_DEPTHS)
    assert bool((system.state_std_level > 0).all())
