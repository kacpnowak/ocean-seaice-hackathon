"""Land/ocean masks and every tensor operation that uses them.

The ocean is not a rectangle.  Measured from the shipped mask file, 30.4% of the
cells on a 1-degree grid are land at the surface and 42.5% are at 1684 m, the
deepest prepared level -- so a large minority of what a convolution or an
attention window sees is not ocean at all.  Everything in this project therefore
passes through one of three ideas:

``wet_surface``   which cells are ocean at the surface (2-D fields),
``wet_level``     which cells are ocean at each of the prepared depths (3-D
                  fields) -- the coastline moves with depth, and using the
                  surface mask for a 1684 m field would call the whole
                  continental shelf "ocean",
``wet_seaice``    where sea-ice variables are defined.  Currently identical to
                  ``wet_surface``: after :func:`fill_seaice_nans` an ice-free
                  ocean cell carries a real value (0), so the domain of the
                  ice fields is the whole ocean surface.

The masking order used by the dataloader (see :mod:`oceanarches.dataloaders.glorys`)
is fixed and matters:

1. read the raw fields.  Land is NaN.  The four ``NAN_MEANS_ZERO`` variables are
   *also* NaN over ice-free ocean, at least from 2015-12-30 onwards.
2. :func:`fill_seaice_nans` -- set those variables to 0 where the cell is ocean.
3. normalise: ``(x - mean) / std``.
4. ``nan_to_num(0.0)`` -- whatever is still NaN is land, and 0 *after*
   normalisation is the climatological mean, the least disruptive value to feed
   a network.

Steps 2 and 3 must stay in that order.  Filling with 0 *after* normalisation
would put a raw 0 into a normalised field, i.e. silently claim that ice-free
ocean has a sea-ice concentration of ``mean + 0 * std``.  Filling land *before*
normalisation would pull every land cell to ``-mean/std`` instead of 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import xarray as xr
from tensordict.tensordict import TensorDict

from .. import paths
from .variables import (
    LEVEL_VARIABLES,
    NAN_MEANS_ZERO,
    SURFACE_VARIABLES,
)

__all__ = [
    "Masks",
    "load_masks",
    "fill_seaice_nans",
    "state_mask",
    "apply_wet_mask",
]


@dataclass(frozen=True)
class Masks:
    """The contents of ``oceanarches/stats/glorys_1deg_masks.nc``, as tensors.

    ``frozen=True`` because the instance is shared (and cached) between
    dataloader workers, the loss and every metric.  Note what that does and does
    not buy: it blocks *rebinding* a field (``masks.wet_surface = other`` raises)
    and nothing more.  The tensors themselves stay writable -- torch has no
    read-only tensor -- so ``masks.wet_surface[0] = False`` would silently change
    the coastline for every holder of the cached object.  Do not write into them.

    The three ``wet_*`` fields are boolean -- ``True`` means ocean.  Convert to
    float only where you need to multiply (:func:`state_mask` does).
    """

    #: ``(depth, lat, lon)`` -- ocean at each prepared depth level.
    wet_level: torch.Tensor
    #: ``(lat, lon)`` -- ocean at the surface.
    wet_surface: torch.Tensor
    #: ``(lat, lon)`` -- where sea-ice variables are defined.
    wet_seaice: torch.Tensor
    #: ``(lat, lon)`` -- depth of the deepest *prepared* level that is ocean in
    #: this column, in metres, and therefore capped at 1684 m rather than being
    #: the true sea floor.  Land is **0.0, not NaN** (``compute_stats.py`` writes
    #: it that way so that ``log1p`` of it is finite for the constants channel),
    #: so ``bathymetry.isnan()`` finds nothing -- use ``wet_surface`` to find land.
    bathymetry: torch.Tensor
    #: ``(channel, 1, lat, lon)`` -- static fields fed to the network alongside
    #: the state (land-sea mask, log bathymetry, sin/cos of lat and lon).
    constants: torch.Tensor
    #: Names of the ``constants`` channels, in order.
    channel_names: tuple[str, ...]
    #: Depths (metres) of the ``wet_level`` axis, in order.
    depths: tuple[float, ...]

    @property
    def n_depth(self) -> int:
        return self.wet_level.shape[0]

    @property
    def n_constants(self) -> int:
        return self.constants.shape[0]

    @property
    def n_ocean_surface(self) -> int:
        """Number of ocean cells at the surface -- a handy sanity number (45115)."""
        return int(self.wet_surface.sum())


@lru_cache(maxsize=4)
def _load_masks_cached(path: str, depth_indices: tuple[int, ...] | None) -> Masks:
    """The real loader.  Cached on its (hashable) arguments -- see :func:`load_masks`."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Masks file not found: {path}\nrun: make stats   (or: make stats-quick)"
        )

    with xr.open_dataset(path) as ds:
        wet_level = torch.from_numpy(ds["wet_level"].to_numpy()).bool()
        wet_surface = torch.from_numpy(ds["wet_surface"].to_numpy()).bool()
        wet_seaice = torch.from_numpy(ds["wet_seaice"].to_numpy()).bool()
        bathymetry = torch.from_numpy(ds["bathymetry"].to_numpy()).float()
        constants = torch.from_numpy(ds["constants"].to_numpy()).float()
        channel_names = tuple(str(c) for c in ds["channel"].to_numpy())
        depths = tuple(float(d) for d in ds["depth"].to_numpy())

    if depth_indices is not None:
        idx = list(depth_indices)
        wet_level = wet_level[idx]
        depths = tuple(depths[i] for i in idx)

    # There used to be a `requires_grad_(False)` loop here "to make them
    # read-only".  It did nothing: a tensor built by `torch.from_numpy` already
    # has requires_grad=False, and requires_grad has nothing to do with
    # mutability anyway.  torch offers no read-only tensor, so the protection is
    # the convention documented on `Masks` -- treat these as immutable.

    return Masks(
        wet_level=wet_level,
        wet_surface=wet_surface,
        wet_seaice=wet_seaice,
        bathymetry=bathymetry,
        constants=constants,
        channel_names=channel_names,
        depths=depths,
    )


def load_masks(
    path: str | Path | None = None,
    depth_indices: list[int] | tuple[int, ...] | None = None,
) -> Masks:
    """Load the mask file once and share it.

    Args:
        path: Mask netCDF file.  Defaults to ``oceanarches/stats/glorys_1deg_masks.nc``.
        depth_indices: Indices into the 14 *prepared* depth levels, i.e. what a
            model preset selects (see ``DEPTH_PRESETS`` in ``variables.py``).
            ``None`` keeps all of them.

    The result is cached, because every dataloader worker, the loss and every
    metric ask for it, and re-reading a 3 MB netCDF file per call would show up
    in the profile.
    """
    path = Path(path) if path is not None else paths.masks_file()
    key = tuple(depth_indices) if depth_indices is not None else None
    return _load_masks_cached(str(path), key)


def fill_seaice_nans(
    surface: torch.Tensor,
    wet_surface: torch.Tensor,
    surface_variables: list[str] | None = None,
) -> torch.Tensor:
    """Replace "no ice here" NaNs with 0, in place, and return ``surface``.

    GLORYS gives all four sea-ice fields the cell method
    ``area: mean where sea_ice``, so over *ice-free ocean* they are undefined.
    The fill is the same for all four, but the reason differs, and only two of
    them changed convention:

    * ``usi`` and ``vsi`` are the ones the 2015-12-30 switch is about.  GLORYS
      wrote an exact 0 over ice-free ocean until 2015-12-29 and NaN from
      2015-12-30 on.  Without this fill the model would see two input channels
      that jump to mostly-NaN halfway through the archive -- and every year from
      2016 on is validation, test or holdout.
    * ``siconc`` and ``sithick`` are NaN over ice-free ocean throughout the whole
      archive, not only after 2015 (``docs/02`` section 2.9 says so).  For them
      the fill is not a convention repair, it is the statement that "no ice" is
      a concentration and a thickness of zero.

    Land is left NaN, so that step 4 of the masking order (``nan_to_num`` after
    normalisation) still puts it at the climatological mean.

    Args:
        surface: ``(var, 1, lat, lon)`` (or any shape whose first axis is the
            variable axis and whose last two are lat/lon).
        wet_surface: ``(lat, lon)`` boolean, ``True`` over ocean.
        surface_variables: Names of the channels of ``surface``, in order.
            Defaults to the full canonical list.  Pass the component's own list
            when a model only loads a subset.
    """
    names = surface_variables if surface_variables is not None else SURFACE_VARIABLES
    if len(names) != surface.shape[0]:
        raise ValueError(
            f"surface has {surface.shape[0]} channels but {len(names)} variable names"
        )

    # Driven by the table in variables.py, never by hardcoded names: adding a
    # variable with nan_means_zero=True there is all it should take.
    channels = [i for i, name in enumerate(names) if name in NAN_MEANS_ZERO]
    if not channels:
        return surface

    ocean = wet_surface.to(torch.bool)
    selected = surface[channels]
    surface[channels] = torch.where(selected.isnan() & ocean, torch.zeros_like(selected), selected)
    return surface


def state_mask(
    masks: Masks,
    surface_variables: list[str] | None = None,
    level_variables: list[str] | None = None,
) -> TensorDict:
    """A float mask that broadcasts onto a state TensorDict: 1 over ocean, 0 over land.

    Shapes match a single (unbatched) state, so it broadcasts against batched
    states too:

        ``surface (var, 1, lat, lon)``   ``level (var, depth, lat, lon)``

    Tasks 6 and 7 multiply the loss and the metrics by this, which is the whole
    point of it: a model scored over land is scored on cells that carry no
    information, and the score is then dominated by how well it reproduces
    zeros.

    All surface variables -- ocean *and* sea ice -- use ``wet_surface``.  After
    :func:`fill_seaice_nans` the ice fields are defined everywhere the ocean
    surface is, so ``wet_seaice`` (which is currently identical) would add
    nothing.  Level variables use the per-depth ``wet_level``.
    """
    surface_variables = surface_variables if surface_variables is not None else SURFACE_VARIABLES
    level_variables = level_variables if level_variables is not None else LEVEL_VARIABLES

    out = {}
    if surface_variables:
        surface = masks.wet_surface.float()[None, None]  # (1, 1, lat, lon)
        out["surface"] = surface.expand(
            len(surface_variables), 1, *surface.shape[-2:]
        ).contiguous()
    if level_variables:
        level = masks.wet_level.float()[None]  # (1, depth, lat, lon)
        out["level"] = level.expand(len(level_variables), *level.shape[1:]).contiguous()
    return TensorDict(out)


def apply_wet_mask(state: TensorDict, mask: TensorDict) -> TensorDict:
    """Zero everything over land.  Returns a new TensorDict.

    ``mask`` is unbatched (see :func:`state_mask`); ``state`` may carry any
    number of leading batch/time axes, and broadcasting lines the two up.
    """
    return state * mask
