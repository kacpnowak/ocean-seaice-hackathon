"""Shape tests for the embedder.  CPU only, and deliberately paranoid.

Every failure mode tested here is one that would otherwise show up as a
plausible-looking model that is quietly wrong: a forecast on 181 latitudes, a
sea-ice model that thinks it predicts temperature, a water column read upside
down.  The tests run against the *real* mask file, because the embedder is tied
to the 180x360 grid that file describes.
"""

from __future__ import annotations

import pytest
import torch
from geoarches.backbones.archesweather import ArchesWeatherCondBackbone
from tensordict.tensordict import TensorDict

from oceanarches.backbones.ocean_embedder import (
    GEOARCHES_Z_DIM,
    OceanEncodeDecodeLayer,
    latent_grid,
    latent_z_dim,
    level_padding,
    usable_depth_counts,
)
from oceanarches.dataloaders.variables import DEPTH_PRESETS, N_LAT, N_LON

EMB = 32  # small enough that a full forward pass on CPU is instant


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_depths, expected",
    [
        (6, 5),
        (10, 7),
        (12, 8),  # usable
        (13, 8),  # usable, and what every shipped preset loads
        (14, 9),  # the trap: 14 levels do NOT give 8
        (0, 1),  # a component with no 3-D input at all: surface token only
    ],
)
def test_latent_z_dim(n_depths, expected):
    assert latent_z_dim(n_depths, check=False) == expected


def test_latent_z_dim_is_not_one_plus_half():
    """The trap: geoarches pads by a whole patch when n_depths is already even."""
    assert latent_z_dim(14, check=False) != 1 + 14 // 2


def test_only_twelve_and_thirteen_levels_are_usable():
    assert usable_depth_counts() == [12, 13]


def test_every_shipped_depth_preset_gives_geoarches_eight():
    for preset, indices in DEPTH_PRESETS.items():
        assert latent_z_dim(len(indices)) == GEOARCHES_Z_DIM, preset


@pytest.mark.parametrize("n_depths", [6, 10, 14])
def test_latent_z_dim_raises_an_actionable_error_off_eight(n_depths):
    """A participant who changes the depth preset must be told, not silently punished."""
    with pytest.raises(ValueError) as excinfo:
        latent_z_dim(n_depths)
    message = str(excinfo.value)
    assert f"{n_depths} depth levels" in message
    assert f"latent depth of {latent_z_dim(n_depths, check=False)}" in message
    assert "LinVert" in message and "axial attention" in message
    assert "[12, 13]" in message
    assert "allow_any_z_dim" in message


def test_the_embedder_constructor_enforces_it_too():
    with pytest.raises(ValueError, match="only works at 8"):
        make_embedder(n_depths=6)
    # ...and the escape hatch is real
    assert make_embedder(n_depths=6, allow_any_z_dim=True).z_dim == 5


@pytest.mark.parametrize("n_depths, pads", [(6, (1, 1)), (10, (1, 1)), (13, (0, 1)), (14, (1, 1))])
def test_level_padding(n_depths, pads):
    assert level_padding(n_depths) == pads
    front, back = pads
    assert latent_z_dim(n_depths, check=False) == 1 + (n_depths + front + back) // 2


def test_latent_grid_matches_the_backbone_window():
    lat, lon = latent_grid()
    assert (lat, lon) == (60, 120)
    # window_size = (1, 6, 10) has to divide the latent grid and the halved one
    for size in ((lat, lon), (lat // 2, lon // 2)):
        assert size[0] % 6 == 0 and size[1] % 10 == 0


def test_latent_grid_rejects_a_patch_that_does_not_divide():
    with pytest.raises(ValueError, match="does not divide"):
        latent_grid((180, 360), (2, 7, 7))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def make_embedder(**kwargs) -> OceanEncodeDecodeLayer:
    args = dict(
        surface_ch_in=7,
        surface_ch_out=7,
        level_ch_in=4,
        level_ch_out=4,
        n_depths=13,
        emb_dim=EMB,
        out_emb_dim=2 * EMB,
    )
    args.update(kwargs)
    return OceanEncodeDecodeLayer(**args)


def make_state(batch: int, surface_ch: int, level_ch: int, n_depths: int) -> TensorDict:
    fields = {"surface": torch.randn(batch, surface_ch, 1, N_LAT, N_LON)}
    if level_ch:
        fields["level"] = torch.randn(batch, level_ch, n_depths, N_LAT, N_LON)
    return TensorDict(fields, batch_size=batch)


def test_constants_are_a_buffer_not_an_attribute():
    """A buffer follows .to(device) and lands in the checkpoint; an attribute does not."""
    embedder = make_embedder()
    assert "constant_masks" in dict(embedder.named_buffers())
    assert "constant_masks" in embedder.state_dict()
    constants = embedder.constant_masks
    assert constants.shape == (6, 1, N_LAT, N_LON), "our masks, not ERA5's 121x240"
    assert embedder.constant_names[:2] == ("land_sea_mask", "log_bathymetry")


def test_input_projection_channel_count_is_the_documented_sum():
    embedder = make_embedder(forcing_ch=3, n_concatenated_states=1)
    # state surface + constants + prev-state surface + forcing
    assert embedder.surface_proj.in_channels == 7 + 6 + 7 + 3
    assert embedder.level_proj.in_channels == 4 + 4


def test_no_previous_state_shrinks_the_projection():
    embedder = make_embedder(n_concatenated_states=0)
    assert embedder.surface_proj.in_channels == 7 + 6
    assert embedder.level_proj.in_channels == 4


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(surface_ch_out=0), "predicts at least one"),
        (dict(surface_ch_out=9), "cannot predict more channels"),
        (dict(level_ch_out=9), "cannot predict more channels"),
        (dict(level_ch_in=0, level_ch_out=0), "disagree"),  # no 3-D vars but n_depths=13
        (dict(n_depths=0, allow_any_z_dim=True), "disagree"),  # n_depths=0 but 4 3-D vars
        (dict(n_concatenated_states=2), "0 or 1"),
        (dict(patch_size=(3, 3, 3)), "vertical patch of 2"),
    ],
)
def test_construction_refuses_inconsistent_wiring(kwargs, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        make_embedder(**kwargs)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_depths", [12, 13])
def test_encode_shape(n_depths):
    embedder = make_embedder(n_depths=n_depths)
    state = make_state(2, 7, 4, n_depths)
    x = embedder.encode(state, state)
    assert x.shape == (2, EMB, latent_z_dim(n_depths), 60, 120)


def test_encode_needs_the_previous_state_it_was_built_for():
    embedder = make_embedder(n_concatenated_states=1)
    state = make_state(1, 7, 4, 13)
    with pytest.raises(ValueError, match="previous state"):
        embedder.encode(state)


def test_encode_refuses_a_forcing_it_has_no_channels_for():
    embedder = make_embedder()
    state = make_state(1, 7, 4, 13)
    with pytest.raises(ValueError, match="forcing_ch=0"):
        embedder.encode(state, state, forcing=torch.randn(1, 3, 1, N_LAT, N_LON))


def test_encode_accepts_forcing_with_or_without_the_depth_axis():
    embedder = make_embedder(forcing_ch=3)
    state = make_state(1, 7, 4, 13)
    for shape in ((1, 3, 1, N_LAT, N_LON), (1, 3, N_LAT, N_LON)):
        assert embedder.encode(state, state, torch.zeros(shape)).shape[1] == EMB


def test_encode_shouts_about_the_wrong_number_of_channels():
    embedder = make_embedder()
    state = make_state(1, 6, 4, 13)  # one surface variable short
    with pytest.raises(ValueError, match="has 6 channels"):
        embedder.encode(state, state)


def test_encode_shouts_about_the_wrong_number_of_depths():
    embedder = make_embedder(n_depths=13)
    state = make_state(1, 7, 4, 10)
    with pytest.raises(ValueError, match="depth levels"):
        embedder.encode(state, state)


def test_encode_shouts_about_the_wrong_grid():
    embedder = make_embedder()
    state = TensorDict(
        {
            "surface": torch.randn(1, 7, 1, 121, 240),
            "level": torch.randn(1, 4, 13, 121, 240),
        },
        batch_size=1,
    )
    with pytest.raises(ValueError, match="grid"):
        embedder.encode(state, state)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------
def backbone_output(embedder: OceanEncodeDecodeLayer, batch: int) -> torch.Tensor:
    """A stand-in for what ArchesWeatherCondBackbone returns, hardcoded 8 and all."""
    channels = embedder.out_emb_dim * embedder.z_dim // 8
    return torch.randn(batch, channels, 8, *embedder.latent_grid)


@pytest.mark.parametrize("n_depths, allow", [(13, False), (6, True)])
def test_unpack_backbone_output_is_an_exact_inverse(n_depths, allow):
    """geoarches reshapes the token axis to a literal 8; we reshape it back.

    At the shipped 13 levels z_dim is already 8 and this is a no-op -- the case
    that matters. The 6-level case exercises the repair path that
    `allow_any_z_dim=True` opens up, and proves it is exact rather than
    approximate: build the tensor the backbone *means*, push it through the same
    two reshapes the backbone applies, and check every element comes back where
    it started.
    """
    embedder = make_embedder(n_depths=n_depths, allow_any_z_dim=allow)
    batch, channels, z_dim = 2, embedder.out_emb_dim, embedder.z_dim
    intended = torch.arange(batch * channels * z_dim * 60 * 120, dtype=torch.float32)
    intended = intended.reshape(batch, channels, z_dim, 60, 120)
    tokens = intended.reshape(batch, channels, -1).transpose(1, 2)  # (b, tokens, channel)
    mangled = tokens.transpose(1, 2).reshape(batch, -1, 8, 60, 120)  # geoarches' last line
    assert mangled.shape[1:3] == (channels * z_dim // 8, 8)
    assert torch.equal(embedder._unpack_backbone_output(mangled), intended)


@pytest.mark.parametrize("n_depths", [12, 13])
def test_decode_returns_the_input_grid_with_no_fake_pole(n_depths):
    embedder = make_embedder(n_depths=n_depths)
    out = embedder.decode(backbone_output(embedder, 2))
    assert out["surface"].shape == (2, 7, 1, N_LAT, N_LON)
    assert out["level"].shape == (2, 4, n_depths, N_LAT, N_LON)
    assert out["surface"].shape[-2] == 180, "181 latitudes means the fake south pole came back"


def test_decode_returns_only_the_predicted_channels():
    """The ocean component reads 7 surface variables and predicts 3."""
    embedder = make_embedder(surface_ch_in=7, surface_ch_out=3)
    out = embedder.decode(backbone_output(embedder, 2))
    assert out["surface"].shape == (2, 3, 1, N_LAT, N_LON)


def test_decode_rejects_tokens_of_the_wrong_size():
    embedder = make_embedder()
    with pytest.raises(ValueError, match="not compatible"):
        embedder.decode(torch.randn(2, 7, 3, 60, 120))


# ---------------------------------------------------------------------------
# The surface-only component -- required by Task 8, easiest thing to break
# ---------------------------------------------------------------------------
def test_level_ch_out_zero_builds_no_level_deconvolution():
    embedder = make_embedder(surface_ch_out=4, level_ch_in=4, level_ch_out=0)
    assert not hasattr(embedder, "level_deconv")
    assert embedder.level_proj.in_channels == 8, "it still *reads* the ocean"


def test_level_ch_out_zero_decodes_to_a_surface_only_tensordict():
    embedder = make_embedder(surface_ch_out=4, level_ch_out=0)
    out = embedder.decode(backbone_output(embedder, 3))
    assert set(out.keys()) == {"surface"}
    assert out["surface"].shape == (3, 4, 1, N_LAT, N_LON)


def test_surface_only_input_and_output():
    """`seaice_isolated`: no 3-D variables at all, so the latent column is one token.

    This needs `allow_any_z_dim=True` (z_dim is 1, not geoarches' 8) and a
    backbone built with `first_interaction_layer: null` and `axis_attn: false`.
    It is an ablation, not a shipped config.
    """
    embedder = make_embedder(
        surface_ch_in=4,
        surface_ch_out=4,
        level_ch_in=0,
        level_ch_out=0,
        n_depths=0,
        allow_any_z_dim=True,
    )
    assert embedder.z_dim == 1
    assert not hasattr(embedder, "level_proj")
    state = make_state(2, 4, 0, 0)
    x = embedder.encode(state, state)
    assert x.shape == (2, EMB, 1, 60, 120)
    out = embedder.decode(backbone_output(embedder, 2))
    assert set(out.keys()) == {"surface"}
    assert out["surface"].shape == (2, 4, 1, N_LAT, N_LON)


# ---------------------------------------------------------------------------
# The whole path, through the real backbone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "surface_out, level_out, n_depths",
    [
        (7, 4, 13),  # full -- the shipped depth preset
        (3, 4, 13),  # ocean component
        (4, 0, 13),  # sea-ice component: surface-only output
        (7, 4, 12),  # the other depth count that lands on z_dim 8
    ],
)
def test_encode_backbone_decode_round_trip(surface_out, level_out, n_depths):
    """The real backbone, with the vertical mixing the shipped configs turn on."""
    embedder = make_embedder(surface_ch_out=surface_out, level_ch_out=level_out, n_depths=n_depths)
    backbone = ArchesWeatherCondBackbone(
        tensor_size=(embedder.z_dim, 60, 120),
        emb_dim=EMB,
        cond_dim=16,
        num_heads=(2, 4, 4, 2),
        window_size=(1, 6, 10),
        depth_multiplier=1,
        use_skip=True,
        # These two are what only work at z_dim == 8, i.e. what the depth preset
        # exists to make possible.
        first_interaction_layer="linear",
        axis_attn=True,
        mlp_layer="swiglu",
    )
    state = make_state(1, 7, 4, n_depths)
    with torch.no_grad():
        out = embedder.decode(backbone(embedder.encode(state, state), torch.randn(1, 16)))

    assert out["surface"].shape == (1, surface_out, 1, N_LAT, N_LON)
    if level_out:
        assert out["level"].shape == (1, level_out, n_depths, N_LAT, N_LON)
    else:
        assert "level" not in out.keys()
    assert torch.isfinite(out["surface"]).all()


def test_embedder_moves_to_a_device_with_its_constants():
    """Not a GPU test: `.to(dtype)` exercises the same buffer machinery."""
    embedder = make_embedder().to(torch.float64)
    assert embedder.constant_masks.dtype == torch.float64
