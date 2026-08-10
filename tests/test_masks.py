"""Tests for the land/ocean masks and the tensor operations built on them."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import N_DEPTH, N_LAT, N_LON
from tensordict.tensordict import TensorDict

from oceanarches.dataloaders.masks import (
    apply_wet_mask,
    fill_seaice_nans,
    load_masks,
    state_mask,
)
from oceanarches.dataloaders.variables import (
    LEVEL_VARIABLES,
    NAN_MEANS_ZERO,
    SURFACE_VARIABLES,
)


# ---------------------------------------------------------------------------
# load_masks
# ---------------------------------------------------------------------------
def test_load_masks_shapes_and_types(tiny_masks_file):
    masks = load_masks(tiny_masks_file)

    assert masks.wet_level.shape == (N_DEPTH, N_LAT, N_LON)
    assert masks.wet_surface.shape == (N_LAT, N_LON)
    assert masks.wet_seaice.shape == (N_LAT, N_LON)
    assert masks.constants.shape == (6, 1, N_LAT, N_LON)
    assert masks.wet_level.dtype is torch.bool
    assert masks.constants.dtype is torch.float32
    assert masks.channel_names[0] == "land_sea_mask"
    assert len(masks.depths) == N_DEPTH


def test_load_masks_is_cached(tiny_masks_file):
    # Every dataloader worker, the loss and every metric call this; re-reading
    # the file each time would show up in the profile.
    assert load_masks(tiny_masks_file) is load_masks(tiny_masks_file)


def test_load_masks_depth_indices_selects_a_preset(tiny_masks_file):
    full = load_masks(tiny_masks_file)
    subset = load_masks(tiny_masks_file, depth_indices=[0, 3, 5])

    assert subset.wet_level.shape[0] == 3
    assert subset.depths == tuple(full.depths[i] for i in (0, 3, 5))
    assert torch.equal(subset.wet_level[1], full.wet_level[3])


def test_load_masks_missing_file_says_how_to_fix_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make stats"):
        load_masks(tmp_path / "not_here.nc")


def test_masks_are_frozen(tiny_masks_file):
    # The object is shared between workers, the loss and the metrics.
    masks = load_masks(tiny_masks_file)
    with pytest.raises(dataclasses.FrozenInstanceError):
        masks.wet_surface = torch.ones(1)


# ---------------------------------------------------------------------------
# fill_seaice_nans
# ---------------------------------------------------------------------------
def _surface_with_nans(wet_surface: torch.Tensor) -> torch.Tensor:
    """A surface tensor that is NaN over land everywhere and, for every variable,
    NaN at one particular ocean cell."""
    surface = torch.ones(len(SURFACE_VARIABLES), 1, N_LAT, N_LON)
    surface[:, :, ~wet_surface] = float("nan")
    ocean = wet_surface.nonzero()[0]
    surface[:, :, ocean[0], ocean[1]] = float("nan")
    return surface


def test_fill_seaice_nans_fills_ocean_and_keeps_land(tiny_masks_file):
    masks = load_masks(tiny_masks_file)
    surface = _surface_with_nans(masks.wet_surface)
    ocean = masks.wet_surface.nonzero()[0]

    filled = fill_seaice_nans(surface, masks.wet_surface)

    for i, name in enumerate(SURFACE_VARIABLES):
        at_ocean_cell = filled[i, 0, ocean[0], ocean[1]]
        if name in NAN_MEANS_ZERO:
            assert at_ocean_cell == 0.0, f"{name} should have been filled"
        else:
            assert at_ocean_cell.isnan(), f"{name} must not be touched"
        # Land is still NaN for every variable: step 4 deals with it, after
        # normalisation, and turning it into a physical 0 here would be wrong.
        assert filled[i, 0][~masks.wet_surface].isnan().all()


def test_fill_seaice_nans_is_driven_by_the_variable_table(tiny_masks_file):
    """A component that loads only a couple of variables still gets it right."""
    masks = load_masks(tiny_masks_file)
    names = ["zos", "siconc"]
    surface = torch.full((2, 1, N_LAT, N_LON), float("nan"))

    filled = fill_seaice_nans(surface, masks.wet_surface, names)

    assert filled[0][:, masks.wet_surface].isnan().all()  # zos untouched
    assert (filled[1][:, masks.wet_surface] == 0.0).all()  # siconc filled


def test_fill_seaice_nans_rejects_a_wrong_channel_count(tiny_masks_file):
    masks = load_masks(tiny_masks_file)
    with pytest.raises(ValueError, match="channels"):
        fill_seaice_nans(torch.zeros(3, 1, N_LAT, N_LON), masks.wet_surface)


# ---------------------------------------------------------------------------
# state_mask / apply_wet_mask
# ---------------------------------------------------------------------------
def test_state_mask_shapes(tiny_masks_file):
    mask = state_mask(load_masks(tiny_masks_file))

    assert mask["surface"].shape == (len(SURFACE_VARIABLES), 1, N_LAT, N_LON)
    assert mask["level"].shape == (len(LEVEL_VARIABLES), N_DEPTH, N_LAT, N_LON)
    assert mask["surface"].dtype is torch.float32
    assert set(mask["surface"].unique().tolist()) <= {0.0, 1.0}


def test_state_mask_uses_the_per_depth_coastline(tiny_masks_file):
    """The coastline moves with depth; using the surface mask at 1684 m would
    call the whole continental shelf ocean."""
    masks = load_masks(tiny_masks_file)
    mask = state_mask(masks)

    for depth_index in range(N_DEPTH):
        expected = masks.wet_level[depth_index].float()
        for var_index in range(len(LEVEL_VARIABLES)):
            assert torch.equal(mask["level"][var_index, depth_index], expected)

    # The deepest level really is drier than the surface, so this matters.
    assert masks.wet_level[-1].sum() < masks.wet_surface.sum()


def test_state_mask_follows_a_variable_subset(tiny_masks_file):
    mask = state_mask(load_masks(tiny_masks_file), ["siconc", "sithick"], ["thetao"])
    assert mask["surface"].shape[0] == 2
    assert mask["level"].shape[0] == 1


def test_apply_wet_mask_zeroes_land(tiny_masks_file):
    masks = load_masks(tiny_masks_file)
    mask = state_mask(masks)
    state = TensorDict(
        surface=torch.ones(len(SURFACE_VARIABLES), 1, N_LAT, N_LON),
        level=torch.ones(len(LEVEL_VARIABLES), N_DEPTH, N_LAT, N_LON),
    )

    masked = apply_wet_mask(state, mask)

    assert (masked["surface"][:, 0, ~masks.wet_surface] == 0).all()
    assert (masked["surface"][:, 0, masks.wet_surface] == 1).all()
    assert masked["level"].sum() == masks.wet_level.sum() * len(LEVEL_VARIABLES)


def test_apply_wet_mask_broadcasts_over_batches(tiny_masks_file):
    """The mask is unbatched; states arrive with batch and rollout axes."""
    mask = state_mask(load_masks(tiny_masks_file))
    state = TensorDict(
        surface=torch.ones(3, 2, len(SURFACE_VARIABLES), 1, N_LAT, N_LON),
        level=torch.ones(3, 2, len(LEVEL_VARIABLES), N_DEPTH, N_LAT, N_LON),
        batch_size=[3, 2],
    )

    masked = apply_wet_mask(state, mask)

    assert masked["surface"].shape == state["surface"].shape
    assert torch.equal(masked["surface"][0, 0], mask["surface"])


# ---------------------------------------------------------------------------
# The generated artefact, when it is there
# ---------------------------------------------------------------------------
def test_real_masks_have_the_expected_ocean_area(real_masks_path):
    masks = load_masks(real_masks_path)
    assert masks.wet_surface.shape == (180, 360)
    assert masks.n_ocean_surface == 45115
    # wet_seaice is derived from wet_surface, so the two must agree exactly.
    assert torch.equal(masks.wet_seaice, masks.wet_surface)


def test_bathymetry_is_zero_over_land_and_never_nan(tiny_masks_file):
    """`masks.py` documented `bathymetry` as "NaN over land"; the shipped file has none.

    `scripts/compute_stats.py` writes 0.0 over land (`np.where(n_wet_levels > 0,
    ..., 0.0)`), because the constants channel takes `log1p` of it and a NaN
    there would poison an input channel. Anyone using `masks.bathymetry.isnan()`
    to find land gets an empty mask. The fixture used to be built with NaN --
    kinder than the real artefact -- which is why nothing caught the docstring.

    MUTANT: putting `np.nan` back in `tests/conftest.py::tiny_masks_file` fails
    both assertions.
    """
    masks = load_masks(path=tiny_masks_file)
    assert not torch.isnan(masks.bathymetry).any(), "bathymetry must carry no NaN"
    land = ~masks.wet_surface
    assert torch.equal(masks.bathymetry[land], torch.zeros_like(masks.bathymetry[land])), (
        "land must be exactly 0.0 in bathymetry"
    )


def test_the_real_bathymetry_is_zero_over_land_too(real_masks_path):
    """The same claim against the artefact `make stats` actually writes."""
    masks = load_masks(path=real_masks_path)
    assert not torch.isnan(masks.bathymetry).any()
    land = ~masks.wet_surface
    assert float(masks.bathymetry[land].abs().max()) == 0.0


def test_the_land_fraction_the_docs_quote_is_the_one_in_the_file(real_masks_path):
    """30.4% land at the surface, 42.5% at 1684 m -- quoted in README, docs/02,
    the cheatsheet and four module docstrings, which disagreed with each other
    (59% and "about 75%") until this was measured.

    MUTANT: changing either bound below by 1 percentage point fails.
    """
    masks = load_masks(path=real_masks_path)
    surface_land = 100.0 * (1.0 - masks.wet_surface.float().mean().item())
    deepest_land = 100.0 * (1.0 - masks.wet_level[-1].float().mean().item())
    assert surface_land == pytest.approx(30.4, abs=0.1)
    assert deepest_land == pytest.approx(42.5, abs=0.1)
    assert float(masks.depths[-1]) == pytest.approx(1684.28, abs=0.01)
