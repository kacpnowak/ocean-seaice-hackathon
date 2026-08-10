"""How a 3-D ocean state becomes tokens, and tokens become a state again.

This file is meant to be *read*.  Everything else in the model -- the attention
blocks, the conditioning, the optimiser -- comes from geoarches unchanged.  The
embedder is the only place where "what the ocean is" meets "what a transformer
eats", so it is the only place where you have to think about the grid.

The picture
-----------
A GLORYS state is two tensors::

    surface  (batch, n_surface, 1,        180, 360)   2-D fields
    level    (batch, n_level,   n_depths, 180, 360)   3-D fields

``encode`` turns them into one tensor of tokens::

    x        (batch, emb_dim, z_dim, 60, 120)

by cutting the map into ``patch_size[1:] = (3, 3)`` degree tiles (180/3 = 60,
360/3 = 120) and the water column into ``patch_size[0] = 2``-level slabs.  Token
0 along the ``z`` axis is the surface; the rest are the depth slabs.  The
backbone shuffles those tokens around, and ``decode`` inverts the whole thing.

Three things are ours and not geoarches'
----------------------------------------
1. **The static fields.**  ``WeatherEncodeDecodeLayer`` loads ERA5's constant
   masks (a 121x240 grid) from its own package.  We load ours from
   ``oceanarches/stats/glorys_1deg_masks.nc``: land-sea mask, log bathymetry and
   sin/cos of latitude and longitude, on our 180x360 grid.  They are registered
   as a *buffer*, so they follow ``.to(device)`` and are written into the
   checkpoint -- a model is not reproducible without the geometry it was
   trained on.
2. **No fake south pole.**  ERA5 has 121 latitudes, an odd number, so geoarches'
   ``encode`` drops the last row and its ``decode`` glues a copy of the last row
   back on.  We have 180 latitudes, nothing is dropped, and gluing a row on
   would return 181.  Our ``decode`` does not do it.
3. **Input and output channels are separate.**  geoarches uses one number,
   ``surface_ch``, for both.  A *component* (see ``ComponentSpec`` in
   ``dataloaders/variables.py``) reads more than it predicts: the sea-ice model
   reads the ocean and predicts only ice.  So we take ``surface_ch_in`` and
   ``surface_ch_out`` separately, and ``level_ch_out == 0`` -- a model that
   predicts nothing three-dimensional -- is a supported configuration.

Why the latent depth is always 8
--------------------------------
geoarches' backbone is not actually general in the vertical.  It writes the
number 8 out as a literal in three places:

* ``LinVert`` -- the ``first_interaction_layer``, and the layer that mixes the
  whole water column -- builds ``nn.Linear(8 * C, 8 * C)`` and reshapes to
  ``(batch, 8, -1, C)``;
* the axial attention's ``AxialPositionalEmbedding(dim=dim, shape=(8,))``;
* ``ArchesWeatherCondBackbone.forward``'s last line,
  ``output.transpose(1, 2).reshape(batch, -1, 8, *layer1_shape)``.

At any other latent depth the second of those raises and the first silently
mixes the wrong axis -- which would cost the model all of its vertical mixing,
because ``window_size = (1, 6, 10)`` means the attention windows never span
depth and the down/up-sampling stages act on latitude and longitude alone.

So we do not fight it: every preset loads 13 depth levels, which pad to 14,
patch into 7 tokens, and give ``1 + 7 == 8``.  :func:`latent_z_dim` raises a
readable error if a depth count ever stops satisfying that, rather than letting
someone discover it as a mysteriously bad model.  Model size is scaled by
``emb_dim`` and ``depth_multiplier``; the sequence length is the same for all
four presets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from geoarches.backbones.archesweather import WeatherEncodeDecodeLayer
from tensordict.tensordict import TensorDict

from ..dataloaders.masks import load_masks
from ..dataloaders.variables import N_LAT, N_LON

__all__ = [
    "GEOARCHES_Z_DIM",
    "OceanEncodeDecodeLayer",
    "latent_grid",
    "latent_z_dim",
    "level_padding",
    "usable_depth_counts",
]

#: (depth, lat, lon) patch used by every shipped preset.  3 divides 180 and 360.
DEFAULT_PATCH_SIZE: tuple[int, int, int] = (2, 3, 3)

#: The latent vertical size geoarches' backbone hardcodes.  Not a preference.
GEOARCHES_Z_DIM: int = 8


# ---------------------------------------------------------------------------
# Geometry.  Small functions, but getting them wrong costs an afternoon.
# ---------------------------------------------------------------------------
def level_padding(
    n_depths: int, patch_size: Sequence[int] = DEFAULT_PATCH_SIZE
) -> tuple[int, int]:
    """``(front, back)`` zero rows added to the depth axis before patching.

    This reproduces geoarches' formula exactly, including its oddity: it pads by
    a *whole patch* when ``n_depths`` is already a multiple of ``patch_size[0]``.
    14 levels with a patch of 2 therefore become 16, not 14.  We keep the
    behaviour rather than "fixing" it so that a checkpoint trained with
    geoarches' own embedder stays loadable, and so that the two code paths
    cannot drift apart.
    """
    if n_depths == 0:
        return (0, 0)
    l_pad = patch_size[0] - n_depths % patch_size[0]
    return (l_pad // 2, l_pad - l_pad // 2)


def usable_depth_counts(patch_size: Sequence[int] = DEFAULT_PATCH_SIZE) -> list[int]:
    """Every ``n_depths`` that geoarches' backbone can actually be used with.

    For the shipped ``patch_size=(2, 3, 3)`` this is ``[12, 13]``.
    """
    return [n for n in range(1, 200) if _raw_latent_z_dim(n, patch_size) == GEOARCHES_Z_DIM]


def _raw_latent_z_dim(n_depths: int, patch_size: Sequence[int]) -> int:
    if n_depths == 0:
        return 1
    front, back = level_padding(n_depths, patch_size)
    return 1 + (n_depths + front + back) // patch_size[0]


def latent_z_dim(
    n_depths: int, patch_size: Sequence[int] = DEFAULT_PATCH_SIZE, check: bool = True
) -> int:
    """Size of the latent vertical axis: 1 surface token + one token per depth slab.

    This is the number the backbone must be configured with (``tensor_size[0]``),
    and geoarches only works when it is :data:`GEOARCHES_Z_DIM` (8).  Because of
    the full-patch padding above it is *not* ``1 + n_depths // 2``:

    ======== ========= ======== ============= ================ =======
    n_depths ``l_pad`` padded   level tokens  ``latent_z_dim`` usable?
    ======== ========= ======== ============= ================ =======
    6        2         8        4             5                no
    10       2         12       6             7                no
    12       2         14       7             8                yes
    13       1         14       7             8                yes
    14       2         16       8             9                no
    ======== ========= ======== ============= ================ =======

    Note the trap in the last row: 14 levels do *not* give 8, because geoarches
    pads by a whole patch when ``n_depths`` is already even.

    Args:
        n_depths: Depth levels the dataloader hands the model.
        patch_size: ``(depth, lat, lon)``.
        check: Raise if the result is not the 8 that geoarches requires.  Pass
            ``False`` only when you want the raw arithmetic (the tests do) or
            when you have configured the backbone with
            ``first_interaction_layer: null`` and ``axis_attn: false`` and
            accepted the loss of vertical mixing that implies.
    """
    z_dim = _raw_latent_z_dim(n_depths, patch_size)
    if check and z_dim != GEOARCHES_Z_DIM:
        usable = usable_depth_counts(patch_size)
        raise ValueError(
            f"{n_depths} depth levels give a latent depth of {z_dim}, but geoarches' "
            f"ArchesWeather backbone only works at {GEOARCHES_Z_DIM}. It writes that 8 out "
            "as a literal in three places: LinVert (the first_interaction_layer), the axial "
            "attention's positional embedding, and the final reshape in "
            "ArchesWeatherCondBackbone.forward. At any other latent depth the axial attention "
            "raises and LinVert silently mixes the wrong axis, which costs the model all of "
            "its cross-depth mixing.\n"
            f"Fix: pick a depth count from {usable} -- with patch_size={tuple(patch_size)} "
            f"those are the only ones that patch into {GEOARCHES_Z_DIM - 1} level tokens. "
            "DEPTH_PRESETS in oceanarches/dataloaders/variables.py ships 13.\n"
            "If you really mean to run at another latent depth, set the backbone's "
            "first_interaction_layer to null and axis_attn to false, and build the embedder "
            "with allow_any_z_dim=True to say so out loud."
        )
    return z_dim


def latent_grid(
    img_size: Sequence[int] = (N_LAT, N_LON), patch_size: Sequence[int] = DEFAULT_PATCH_SIZE
) -> tuple[int, int]:
    """Horizontal size of the token grid, ``(lat, lon)``.

    With the shipped ``patch_size=(2, 3, 3)`` this is ``(60, 120)``, which the
    backbone's ``window_size=(1, 6, 10)`` divides -- and so does the halved
    ``(30, 60)`` it works on after the down-sampling stage.  Change the patch
    size and you have to re-check both.
    """
    lat, lon = img_size[-2], img_size[-1]
    if lat % patch_size[1] or lon % patch_size[2]:
        raise ValueError(
            f"patch_size {tuple(patch_size)} does not divide the {lat}x{lon} grid; "
            "the encoder convolution would silently crop it."
        )
    return lat // patch_size[1], lon // patch_size[2]


# ---------------------------------------------------------------------------
# The embedder
# ---------------------------------------------------------------------------
class OceanEncodeDecodeLayer(WeatherEncodeDecodeLayer):
    """GLORYS state <-> tokens.

    Subclasses geoarches' layer so that everything geoarches does with an
    embedder keeps working, and reuses its decoder convolutions, its pixel
    shuffle and its depth padder verbatim.  ``encode`` and ``decode`` are
    rewritten, because those are exactly the two methods that assume ERA5.

    Channel order fed to ``surface_proj`` -- fixed, and relied on by Task 8's
    coupling::

        [ state surface | constants | prev-state surface | forcing ]

    and to ``level_proj``::

        [ state level | prev-state level ]

    The state's *own* channels come first in both, so that a component's
    prognostic variables are always the leading channels of the input and the
    leading channels of the output.
    """

    def __init__(
        self,
        surface_ch_in: int,
        surface_ch_out: int,
        level_ch_in: int,
        level_ch_out: int,
        n_depths: int,
        emb_dim: int = 192,
        out_emb_dim: int = 384,
        patch_size: Sequence[int] = DEFAULT_PATCH_SIZE,
        forcing_ch: int = 0,
        masks_path: str | Path | None = None,
        n_concatenated_states: int = 1,
        img_size: Sequence[int] = (N_LAT, N_LON),
        allow_any_z_dim: bool = False,
    ) -> None:
        """
        Args:
            surface_ch_in: 2-D variables the model *reads* (own + forcing).
            surface_ch_out: 2-D variables it *predicts*.  Must be >= 1.
            level_ch_in: 3-D variables it reads.  0 for a model with no 3-D input.
            level_ch_out: 3-D variables it predicts.  0 is legal and supported --
                that is the sea-ice component.
            n_depths: Depth levels loaded by the dataloader, i.e.
                ``len(DEPTH_PRESETS[preset])``.  0 iff ``level_ch_in == 0``.
            emb_dim: Token width produced by ``encode``.
            out_emb_dim: Token width the backbone hands to ``decode``.  With the
                backbone's ``use_skip=True`` this is ``2 * emb_dim``.
            patch_size: ``(depth, lat, lon)``.  Only ``(2, *, *)`` is supported
                (see below).
            forcing_ch: Extra 2-D channels from a
                :class:`~oceanarches.dataloaders.forcing.ForcingSource`.
            masks_path: Mask file; defaults to ``oceanarches/stats/glorys_1deg_masks.nc``.
            n_concatenated_states: 1 to concatenate the previous state (geoarches'
                ``cond_state``), 0 to run on a single snapshot.
            img_size: ``(lat, lon)`` of the data.
            allow_any_z_dim: Skip the check that ``n_depths`` gives the latent
                depth of 8 that geoarches' backbone hardcodes.  Only pass True
                if you have also set ``first_interaction_layer: null`` and
                ``axis_attn: false`` on the backbone -- see :func:`latent_z_dim`.
        """
        patch_size = tuple(int(p) for p in patch_size)
        img_size = tuple(int(s) for s in img_size)

        # -- argument checking, before anything is allocated --------------
        if patch_size[0] != 2:
            # geoarches' level_deconv is built with `out_emb_dim // 2` input
            # channels because it unpacks each token into exactly 2 depth
            # slices.  Supporting another vertical patch size means rebuilding
            # that convolution, which is more than a subclass should do quietly.
            # Checked first, because every depth calculation below assumes it.
            raise NotImplementedError(
                f"patch_size[0]={patch_size[0]}: geoarches' decoder assumes a vertical "
                "patch of 2. Use (2, h, w)."
            )
        # This one next: it is the mistake a participant is most likely to make
        # (changing a depth preset) and the one with the least obvious symptom.
        z_dim = latent_z_dim(n_depths, patch_size, check=not allow_any_z_dim)

        if surface_ch_out < 1:
            raise ValueError(
                f"surface_ch_out={surface_ch_out}: every component predicts at least one "
                "2-D field. A level-only model has no way to report a forecast here."
            )
        if surface_ch_out > surface_ch_in or level_ch_out > level_ch_in:
            raise ValueError(
                f"A model cannot predict more channels than it reads: got "
                f"surface {surface_ch_in} in / {surface_ch_out} out, "
                f"level {level_ch_in} in / {level_ch_out} out."
            )
        if (level_ch_in == 0) != (n_depths == 0):
            raise ValueError(
                f"level_ch_in={level_ch_in} and n_depths={n_depths} disagree: a model with "
                "no 3-D variables has no depth axis, and vice versa."
            )
        if n_concatenated_states not in (0, 1):
            raise ValueError(
                f"n_concatenated_states={n_concatenated_states}: encode() is handed at most "
                "one cond_state (the previous state), so this is 0 or 1."
            )
        if out_emb_dim % 2:
            raise ValueError(f"out_emb_dim={out_emb_dim} must be even (it is split in two).")

        # -- build the geoarches layer, then repair the three ERA5 bits ---
        # `surface_ch` / `level_ch` are the parent's *output* channel counts: we
        # hand it the OUT counts so that surface_deconv and level_deconv (and
        # their ICNR initialisation) come out right, and rebuild the two
        # encoder projections below with the IN counts.
        super().__init__(
            img_size=(n_depths, *img_size),
            emb_dim=emb_dim,
            out_emb_dim=out_emb_dim,
            patch_size=patch_size,
            surface_ch=surface_ch_out,
            level_ch=max(level_ch_out, 1),  # 0 output channels is not a valid Conv2d
            n_concatenated_states=0,
            final_interpolation=False,
        )

        self.surface_ch_in = int(surface_ch_in)
        self.surface_ch_out = int(surface_ch_out)
        self.level_ch_in = int(level_ch_in)
        self.level_ch_out = int(level_ch_out)
        self.level_ch = int(level_ch_out)  # the parent's name; keep it honest
        self.n_depths = int(n_depths)
        self.forcing_ch = int(forcing_ch)
        self.n_concatenated_states = int(n_concatenated_states)
        self.grid = img_size
        self.level_pads = level_padding(n_depths, patch_size)
        self.z_dim = z_dim  # checked at the top of __init__
        self.latent_grid = latent_grid(img_size, patch_size)

        # (1) our static fields instead of ERA5's -------------------------
        masks = load_masks(path=masks_path)
        constants = masks.constants.float()  # (channel, 1, lat, lon)
        if tuple(constants.shape[-2:]) != img_size:
            raise ValueError(
                f"The masks in {masks_path or 'oceanarches/stats/glorys_1deg_masks.nc'} are on a "
                f"{constants.shape[-2]}x{constants.shape[-1]} grid but the model is configured "
                f"for {img_size[0]}x{img_size[1]}. Re-run: make stats"
            )
        # The parent assigned a plain tensor attribute; register_buffer refuses
        # to shadow one, so drop it first.  A buffer (rather than an attribute)
        # is what makes the constants follow .to(device) and land in the
        # checkpoint next to the weights that were trained against them.
        del self.constant_masks
        self.register_buffer("constant_masks", constants.clone(), persistent=True)
        self.constant_names: tuple[str, ...] = tuple(masks.channel_names)
        self.n_constants = int(constants.shape[0])
        self.auto_move_to_device = False  # the buffer does it for us

        # (2) encoder projections with the real input channel counts ------
        n_states = 1 + self.n_concatenated_states
        self.surface_proj_ch = n_states * self.surface_ch_in + self.n_constants + self.forcing_ch
        self.level_proj_ch = n_states * self.level_ch_in
        self.surface_proj = nn.Conv2d(
            self.surface_proj_ch, emb_dim, kernel_size=patch_size[1:], stride=patch_size[1:]
        )
        if self.level_proj_ch:
            self.level_proj = nn.Conv3d(
                self.level_proj_ch, emb_dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            # Surface-only input: no depth axis at all, so no 3-D convolution
            # and nothing to pad.  z_dim is then 1 -- just the surface token.
            del self.level_proj
            del self.level_padder

        # (3) surface-only output: no level deconvolution ------------------
        if self.level_ch_out == 0:
            del self.level_deconv

    # -- encode ----------------------------------------------------------
    def encode(
        self,
        state: TensorDict,
        cond_state: TensorDict | None = None,
        forcing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """State (+ previous state, + forcing) -> tokens ``(batch, emb_dim, z_dim, 60, 120)``.

        Args:
            state: TensorDict with ``surface`` ``(b, surface_ch_in, 1, lat, lon)``
                and, unless the component is surface-only, ``level``
                ``(b, level_ch_in, n_depths, lat, lon)``.
            cond_state: The state one lead time earlier, same shapes.  Required
                iff the layer was built with ``n_concatenated_states=1``.
            forcing: ``(b, forcing_ch, 1, lat, lon)`` or ``(b, forcing_ch, lat, lon)``.
                Required iff ``forcing_ch > 0``.
        """
        surface = self._as_2d(state, "surface", self.surface_ch_in, "state")
        batch = surface.shape[0]

        # Order matters and is documented on the class: own state, geometry,
        # previous state, forcing.
        surface_parts = [surface, self.constant_masks[None, :, 0].expand(batch, -1, -1, -1)]
        level_parts = []
        if self.level_ch_in:
            level_parts.append(self._as_3d(state, self.level_ch_in, "state"))

        if self.n_concatenated_states:
            if cond_state is None:
                raise ValueError(
                    "This embedder was built with n_concatenated_states=1 and expects the "
                    "previous state, but encode() got cond_state=None. Either build the "
                    "dataloader with load_prev=True or set n_concatenated_states=0."
                )
            surface_parts.append(
                self._as_2d(cond_state, "surface", self.surface_ch_in, "cond_state")
            )
            if self.level_ch_in:
                level_parts.append(self._as_3d(cond_state, self.level_ch_in, "cond_state"))
        elif cond_state is not None:
            raise ValueError(
                "encode() was given a cond_state but the embedder was built with "
                "n_concatenated_states=0, so there is no room for it in surface_proj."
            )

        if self.forcing_ch:
            if forcing is None:
                raise ValueError(
                    f"This embedder reserves {self.forcing_ch} forcing channels but encode() "
                    "got forcing=None. Pass ForcingSource.get(timestamp), or rebuild with "
                    "forcing_ch=0."
                )
            surface_parts.append(self._forcing_as_2d(forcing, batch))
        elif forcing is not None:
            raise ValueError(
                "encode() was given a forcing tensor but the embedder was built with "
                "forcing_ch=0, so surface_proj has no channels for it."
            )

        surface_tokens = self.surface_proj(torch.cat(surface_parts, dim=1))
        if not level_parts:
            x = surface_tokens.unsqueeze(2)
        else:
            level = self.level_padder(torch.cat(level_parts, dim=1))
            x = torch.cat([surface_tokens.unsqueeze(2), self.level_proj(level)], dim=2)

        if x.shape[2] != self.z_dim:  # pragma: no cover - guards a coding error
            raise RuntimeError(
                f"encode() produced {x.shape[2]} latent levels but the layer was built for "
                f"{self.z_dim}. The backbone's tensor_size[0] must equal "
                f"latent_z_dim({self.n_depths})."
            )
        return x

    # -- decode ----------------------------------------------------------
    def decode(self, x: torch.Tensor) -> TensorDict:
        """Tokens -> a TensorDict holding only the channels this model *predicts*.

        Returns ``surface`` ``(b, surface_ch_out, 1, 180, 360)`` and, when
        ``level_ch_out > 0``, ``level`` ``(b, level_ch_out, n_depths, 180, 360)``.
        No fake south pole: the grid comes back exactly as it went in.
        """
        x = self._unpack_backbone_output(x)
        batch = x.shape[0]

        surface = self.pixelshuffle(self.surface_deconv(x[:, :, 0])).unsqueeze(-3)
        out = {"surface": surface}

        if self.level_ch_out:
            tokens = x[:, :, 1:]  # (b, out_emb_dim, level tokens, 60, 120)
            b, c, n_tokens, h, w = tokens.shape
            # Each token carries two depth slices, stacked in the channel axis.
            # Unstack them *token-major* -- slice index = 2 * token + half --
            # so that the latent column reads in the same order as the water
            # column, and the zero padding added by `level_padder` sits at the
            # two ends where `level_pads` says it does.
            level = tokens.reshape(b, c // 2, 2, n_tokens, h, w)
            level = level.permute(0, 1, 3, 2, 4, 5).reshape(b, c // 2, 2 * n_tokens, h, w)
            front, _ = self.level_pads
            level = level[:, :, front : front + self.n_depths]
            # One 2-D deconvolution per depth: fold depth into the batch axis.
            level = level.movedim(-3, 1).flatten(0, 1)
            level = self.pixelshuffle(self.level_deconv(level))
            out["level"] = level.reshape(batch, self.n_depths, *level.shape[1:]).movedim(1, -3)

        return TensorDict(out, batch_size=batch).to(x.device)

    # -- helpers ---------------------------------------------------------
    def _unpack_backbone_output(self, x: torch.Tensor) -> torch.Tensor:
        """Undo ``ArchesWeatherCondBackbone``'s hardcoded 8 latent levels.

        The backbone finishes with ``.reshape(batch, -1, 8, lat, lon)``.  At the
        shipped ``z_dim == 8`` that is the identity and this method returns ``x``
        untouched -- which is the whole reason the depth presets are chosen to
        land on 8.  The repair below only runs for an embedder built with
        ``allow_any_z_dim=True``, where the reshape moves the boundary between
        the channel axis and the depth axis (``(384, 5)`` comes back as
        ``(240, 8)``).  The tensor being reshaped is laid out
        ``[batch][channel][z][lat][lon]`` in memory and the reshape permutes
        nothing, so reshaping once more with the sizes we expect is exact, not
        an approximation.
        """
        if x.dim() != 5:
            raise ValueError(
                f"decode() expects a 5-D (batch, channel, z, lat, lon) tensor, got {x.dim()}-D "
                f"{tuple(x.shape)}."
            )
        batch, channels, z_dim, h, w = x.shape
        if (channels, z_dim) == (self.out_emb_dim, self.z_dim):
            return x
        if channels * z_dim != self.out_emb_dim * self.z_dim or (h, w) != self.latent_grid:
            raise ValueError(
                f"decode() got tokens of shape {tuple(x.shape)}, which is not compatible with "
                f"(batch, {self.out_emb_dim}, {self.z_dim}, {self.latent_grid[0]}, "
                f"{self.latent_grid[1]}). Check the backbone's tensor_size and emb_dim against "
                "the embedder's."
            )
        return x.reshape(batch, self.out_emb_dim, self.z_dim, h, w)

    def _as_2d(self, state: TensorDict, key: str, expected: int, what: str) -> torch.Tensor:
        """``(b, ch, 1, lat, lon)`` -> ``(b, ch, lat, lon)``, with loud checks."""
        if key not in state.keys():
            raise KeyError(f"{what} has no {key!r} key; it holds {sorted(state.keys())}.")
        tensor = state[key]
        if tensor.dim() != 5 or tensor.shape[2] != 1:
            raise ValueError(
                f"{what}['{key}'] should be (batch, {expected}, 1, lat, lon), got "
                f"{tuple(tensor.shape)}."
            )
        self._check_channels(tensor.shape[1], expected, f"{what}['{key}']")
        self._check_grid(tensor.shape[-2:], f"{what}['{key}']")
        return tensor.squeeze(-3)

    def _as_3d(self, state: TensorDict, expected: int, what: str) -> torch.Tensor:
        if "level" not in state.keys():
            raise KeyError(
                f"{what} has no 'level' key but the embedder reads {expected} 3-D variables; "
                f"it holds {sorted(state.keys())}."
            )
        tensor = state["level"]
        if tensor.dim() != 5:
            raise ValueError(
                f"{what}['level'] should be (batch, {expected}, {self.n_depths}, lat, lon), got "
                f"{tuple(tensor.shape)}."
            )
        self._check_channels(tensor.shape[1], expected, f"{what}['level']")
        if tensor.shape[2] != self.n_depths:
            raise ValueError(
                f"{what}['level'] has {tensor.shape[2]} depth levels, the embedder was built "
                f"for {self.n_depths}. The dataloader's depth_indices and the module preset "
                "have to agree."
            )
        self._check_grid(tensor.shape[-2:], f"{what}['level']")
        return tensor

    def _forcing_as_2d(self, forcing: torch.Tensor, batch: int) -> torch.Tensor:
        if forcing.dim() == 5 and forcing.shape[2] == 1:
            forcing = forcing.squeeze(-3)
        if forcing.dim() != 4:
            raise ValueError(
                f"forcing should be (batch, {self.forcing_ch}, [1,] lat, lon), got "
                f"{tuple(forcing.shape)}."
            )
        if forcing.shape[0] != batch:
            raise ValueError(f"forcing has batch {forcing.shape[0]}, the state has {batch}.")
        self._check_channels(forcing.shape[1], self.forcing_ch, "forcing")
        self._check_grid(forcing.shape[-2:], "forcing")
        return forcing

    @staticmethod
    def _check_channels(got: int, expected: int, what: str) -> None:
        if got != expected:
            raise ValueError(
                f"{what} has {got} channels, the embedder was built for {expected}. "
                "A silent mismatch here trains a model on the wrong variables, so this is "
                "fatal on purpose."
            )

    def _check_grid(self, got, what: str) -> None:
        if tuple(int(s) for s in got) != self.grid:
            raise ValueError(
                f"{what} is on a {got[0]}x{got[1]} grid, the embedder was built for "
                f"{self.grid[0]}x{self.grid[1]}."
            )

    # -- introspection ----------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"surface {self.surface_ch_in} in / {self.surface_ch_out} out, "
            f"level {self.level_ch_in} in / {self.level_ch_out} out, "
            f"n_depths={self.n_depths}, z_dim={self.z_dim}, "
            f"latent_grid={self.latent_grid}, forcing_ch={self.forcing_ch}"
        )
