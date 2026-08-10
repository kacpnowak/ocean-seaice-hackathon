"""Model input/output layers.

The transformer *backbone* is geoarches' ``ArchesWeatherCondBackbone``, used
unchanged.  What lives here is the layer on either side of it: the embedder that
turns a GLORYS state into tokens and the tokens back into a state.

If you are here because you changed a depth preset and something shouted at you,
:data:`GEOARCHES_Z_DIM` and :func:`usable_depth_counts` are the two names you
want; :func:`latent_z_dim` explains the rest.
"""

from .ocean_embedder import (
    GEOARCHES_Z_DIM,
    OceanEncodeDecodeLayer,
    latent_grid,
    latent_z_dim,
    level_padding,
    usable_depth_counts,
)

__all__ = [
    "GEOARCHES_Z_DIM",
    "OceanEncodeDecodeLayer",
    "latent_grid",
    "latent_z_dim",
    "level_padding",
    "usable_depth_counts",
]
