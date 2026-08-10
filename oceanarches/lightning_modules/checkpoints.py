"""Making our checkpoints readable by ``torch.load(weights_only=True)``.

**Without this, resuming a run is impossible.**  Since torch 2.6 ``torch.load``
defaults to ``weights_only=True``, and Lightning's
``_CheckpointConnector.resume_start`` takes that default, so relaunching a run
died before the first batch with::

    _pickle.UnpicklingError: Weights only load failed.
    WeightsUnpickler error: Unsupported global: GLOBAL
    omegaconf.listconfig.ListConfig was not an allowed global by default.

Where the OmegaConf actually comes from
--------------------------------------
Not from ``hyper_parameters``: geoarches has ``save_hyperparameters()``
commented out (``geoarches/lightning_modules/forecast.py:48``) and no checkpoint
this kit writes has that key at all.  The leak is the **optimiser state**::

    optimizer_states[0]["param_groups"][0]["betas"]  ->  ListConfig([0.9, 0.98])

``betas: [0.9, 0.98]`` in ``configs/module/*.yaml`` composes to an OmegaConf
``ListConfig``, hydra hands it straight to ``OceanForecastModule``, geoarches'
``configure_optimizers`` passes it to ``torch.optim.AdamW``, and AdamW keeps it
verbatim in ``param_groups``.  Verified by walking a real 4000-step checkpoint:
two ``ListConfig`` values, both at that path, and nothing else.

Two fixes, and both are wanted
------------------------------
1. **At source.** :func:`plain_betas` in ``ocean_forecast.py`` casts ``betas``
   to a ``tuple`` of floats before it reaches the parent, so checkpoints written
   from now on contain no OmegaConf anywhere and load under
   ``weights_only=True`` with no allowlist at all.  That is the narrower fix and
   it is the primary one.
2. **On load**, below.  Kept deliberately, for two reasons the cast cannot
   cover: the checkpoints already on disk (`modelstore/task6_tiny` and the three
   specialists, which participants are told to start from) still carry the
   ``ListConfig``, and geoarches could reinstate ``save_hyperparameters()`` --
   it is one comment character away -- which would put the whole hydra config
   back into every checkpoint.

Security
--------
This is deliberately *not* ``weights_only=False``, which would disable the
protection entirely.  Every name allowlisted below is a data container that
runs nothing on unpickling.  A pickle carrying, say, ``os.system`` is still
refused, which is the property ``weights_only=True`` exists to give.
``DictConfig`` is the one entry no checkpoint currently needs (checked by
leave-one-out against a real one); it is kept for case 2 above and is inert.
"""

from __future__ import annotations

__all__ = ["allow_omegaconf_in_checkpoints"]

_DONE = False


def allow_omegaconf_in_checkpoints() -> None:
    """Allowlist the OmegaConf container types our checkpoints contain.  Idempotent.

    Called for its side effect when :mod:`oceanarches.lightning_modules` is
    imported, which is before any checkpoint is opened on every path there is.
    The call site that needs it is inside Lightning, reached through geoarches,
    so there is no argument we could pass instead.
    """
    global _DONE
    if _DONE:
        return

    import collections
    import typing

    import torch
    from omegaconf import base as omegaconf_base
    from omegaconf import dictconfig, listconfig, nodes

    torch.serialization.add_safe_globals(
        [
            listconfig.ListConfig,  # the one a real checkpoint has, via `betas`
            dictconfig.DictConfig,  # only if geoarches reinstates save_hyperparameters
            omegaconf_base.ContainerMetadata,
            omegaconf_base.Metadata,
            nodes.AnyNode,
            typing.Any,
            collections.defaultdict,
            list,
            dict,
            int,
        ]
    )
    _DONE = True
