"""Lightning modules for OceanArches.

Importing this package also makes ``torch.load(weights_only=True)`` able to read
our checkpoints -- see :func:`allow_omegaconf_in_checkpoints`, which is called at
the bottom.  This is the right home for it because *every* path that opens a
checkpoint imports a module from here first: `main_hydra` instantiates
`cfg.module.module` before `trainer.fit`, and geoarches' `load_module`
instantiates before it calls `init_from_ckpt`.
"""

from .checkpoints import allow_omegaconf_in_checkpoints
from .coupled import CoupledForecastModule, StateRouter
from .ocean_forecast import OceanForecastModule, compute_lat_weights_glorys

__all__ = [
    "CoupledForecastModule",
    "OceanForecastModule",
    "StateRouter",
    "allow_omegaconf_in_checkpoints",
    "compute_lat_weights_glorys",
]

allow_omegaconf_in_checkpoints()
