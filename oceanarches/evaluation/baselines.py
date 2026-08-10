"""The forecasts a model has to beat, and the interface that keeps them honest.

A number with nothing beside it does not answer "is my model any good?".  These
are the two things it has to be compared against:

**Persistence** -- tomorrow looks like today.  For a one-day ocean forecast this
is a very strong baseline (the ocean has a memory of weeks to months), and it is
the line every model must clear.  A day-1 RMSE worse than persistence means
something is wrong with the model, the masking or the evaluation.

**Climatology** -- the 1993-2025 monthly mean from
``oceanarches/stats/glorys_1deg_climatology.nc``, interpolated to the valid day.
It knows nothing about today, so it is hopeless at day 1 and unbeatable at day
1000.  *Where the model's error curve crosses the climatology curve is the
honest estimate of the useful forecast horizon* -- past that point the forecast
carries no more information than "it is March".

The one design rule in this file
--------------------------------
The model is wrapped in the *same* :class:`Forecaster` interface as the two
baselines, and the scoring loop in :mod:`oceanarches.evaluation.run_eval` only
ever talks to that interface.  There is therefore exactly one code path from a
prediction to a metric, and it is impossible for the model and its baselines to
be denormalised differently, masked differently or scored on different samples.
That is not decoration: a baseline computed through a second, parallel path is
how you end up publishing a skill score that measures the difference between two
pieces of your own code.

Every :meth:`Forecaster.predict` returns **normalised** tensors shaped
``(batch, timedelta, var, depth, lat, lon)`` -- the model's native space.  The
caller denormalises once, with the module's own statistics, before scoring.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from ..metrics.masked_metrics import _load_climatology, month_interpolation_weights

__all__ = [
    "as_trajectory",
    "Forecaster",
    "ModelForecast",
    "PersistenceForecast",
    "ClimatologyForecast",
    "build_forecasters",
    "targets_for",
    "BASELINE_KEYS",
]

#: The two baselines, in the order they appear in every legend and table.
BASELINE_KEYS = ("persistence", "climatology")


def as_trajectory(entries: dict[str, torch.Tensor] | TensorDict, n_batch: int, iters: int):
    """Stamp a rollout with ``batch_size = (batch, timedelta)``.

    Not cosmetic.  ``TensorDict`` arithmetic broadcasts the *declared* batch size,
    not the tensor shapes, so subtracting a prediction whose batch size is
    ``(2,)`` from a target whose batch size is ``(2, 10)`` raises even though
    every tensor inside them has the identical shape.  ``forward_multistep``
    produces ``(batch, timedelta)`` and this is what makes the baselines match
    it, so ``module.loss`` accepts all three without a special case.
    """
    if isinstance(entries, TensorDict):
        entries = {key: entries[key] for key in entries.keys()}
    return TensorDict(entries, batch_size=(int(n_batch), int(iters)))


def targets_for(module, batch: dict, iters: int) -> TensorDict:
    """Ground truth for ``iters`` steps, normalised, with a ``timedelta`` axis.

    Mirrors ``OceanForecastModule._predict``: ``future_states`` when the
    dataloader was built with ``multistep > 1``, otherwise the single
    ``next_state`` with a length-1 timedelta axis inserted.

    Args:
        module: the loaded :class:`~oceanarches.lightning_modules.ocean_forecast.OceanForecastModule`.
        batch: one collated sample dict from ``GlorysForecast``.
        iters: number of lead times to keep.

    Returns:
        ``(batch, timedelta, var, depth, lat, lon)``, normalised, this
        component's prognostic channels only.
    """
    if "future_states" in batch:
        return module.select_prognostic(batch["future_states"])[:, :iters]
    if iters != 1:
        raise ValueError(
            f"The batch has no 'future_states', so only a 1-step target exists, but "
            f"{iters} were asked for. Build the dataset with multistep={iters}."
        )
    return module.select_prognostic(batch["next_state"])[:, None]


class Forecaster(abc.ABC):
    """One thing that can produce a rollout: the model, or a baseline.

    Subclasses implement :meth:`predict` only.  ``key`` is what appears in file
    names and metric dictionaries; ``label`` is what appears in a figure legend.
    """

    #: Short machine-readable name.
    key: str = "forecaster"
    #: Human-readable name for legends and tables.
    label: str = "Forecaster"
    #: True for the trained model, False for a baseline -- used only for styling.
    is_model: bool = False

    @abc.abstractmethod
    def predict(self, batch: dict, iters: int) -> TensorDict:
        """``(batch, timedelta, var, depth, lat, lon)``, normalised."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(key={self.key!r})"


class ModelForecast(Forecaster):
    """The trained model, rolled out autoregressively.

    A thin wrapper on ``module.forward_multistep``.  It exists so that the model
    and the baselines are the same kind of object to the scoring loop.
    """

    key = "model"
    is_model = True

    def __init__(self, module, label: str | None = None):
        self.module = module
        self.label = label or "Model"

    @torch.no_grad()
    def predict(self, batch: dict, iters: int) -> TensorDict:
        return self.module.forward_multistep(batch, iters=iters)


class PersistenceForecast(Forecaster):
    """Repeat the initial state at every lead time.

    The prediction is the *initial* state's prognostic channels, broadcast along
    the timedelta axis with ``expand`` rather than copied -- a 10-day rollout at
    batch 8 is 1.2 GB of state, and the metrics only ever read it.

    No masking or clamping is applied and none is needed: the dataloader already
    fills land with 0 in normalised space (step 4 of the masking order), and the
    state is by construction inside the physical bounds.
    """

    key = "persistence"
    label = "Persistence"

    def __init__(self, module):
        self.module = module

    @torch.no_grad()
    def predict(self, batch: dict, iters: int) -> TensorDict:
        state = self.module.select_prognostic(batch["state"])
        entries = {
            key: state[key].unsqueeze(1).expand(state[key].shape[0], iters, *state[key].shape[1:])
            for key in state.keys()
        }
        return as_trajectory(entries, next(iter(entries.values())).shape[0], iters)


class ClimatologyForecast(Forecaster):
    """The monthly climatology, interpolated to each step's valid day.

    Uses :func:`~oceanarches.metrics.masked_metrics.month_interpolation_weights`
    -- the *same* periodic linear interpolation between month midpoints that the
    ACC metric uses -- so the climatology baseline and the climatology the model
    is scored against are the identical field.  Anything else would make ACC and
    the climatology curve tell two different stories about the same file.

    The file is in physical units; the result is normalised with the module's own
    statistics, which is a round trip of ~1e-7 relative error against the
    denormalisation the scorer applies afterwards.  That is six orders of
    magnitude below the smallest metric difference in the report, and it buys the
    single-code-path guarantee described in the module docstring.

    Args:
        module: the loaded forecast module (for the statistics, the mask and the
            variable list).
        climatology_path: override; defaults to the shipped file.
    """

    key = "climatology"
    label = "Climatology"

    def __init__(self, module, climatology_path: str | Path | None = None):
        from .. import paths

        self.module = module
        path = str(climatology_path or paths.climatology_file())
        depth_indices = module.depth_indices
        self.climatology: dict[str, torch.Tensor] = {}
        groups = [("surface", module.surface_variables, None)]
        if module.n_level_out:
            groups.append(
                ("level", module.level_variables, tuple(depth_indices) if depth_indices else None)
            )
        for group, names, indices in groups:
            if not names:
                continue
            self.climatology[group] = _load_climatology(path, group, tuple(names), indices)
        self.lead_time_hours = int(module.lead_time_hours)

    def to(self, device) -> "ClimatologyForecast":
        """Move the climatology to a device.  Returns self, so it chains."""
        self.climatology = {k: v.to(device) for k, v in self.climatology.items()}
        return self

    def _field(self, group: str, valid_seconds: int) -> torch.Tensor:
        i0, i1, alpha = month_interpolation_weights(valid_seconds)
        clim = self.climatology[group]
        return torch.lerp(clim[i0], clim[i1], alpha)

    @torch.no_grad()
    def predict(self, batch: dict, iters: int) -> TensorDict:
        stamps = np.atleast_1d(torch.as_tensor(batch["timestamp"]).detach().cpu().numpy()).astype(
            "int64"
        )
        step_seconds = self.lead_time_hours * 3600

        entries = {}
        for group in self.climatology:
            per_sample = []
            for start in stamps:
                per_sample.append(
                    torch.stack(
                        [
                            self._field(group, int(start) + (i + 1) * step_seconds)
                            for i in range(iters)
                        ]
                    )
                )
            entries[group] = torch.stack(per_sample)

        n_batch = len(stamps)
        physical = as_trajectory(entries, n_batch, iters)
        normalised = as_trajectory(
            {
                "surface": (physical["surface"] - self.module.state_mean_surface)
                / self.module.state_std_surface,
                **(
                    {
                        "level": (physical["level"] - self.module.state_mean_level)
                        / self.module.state_std_level
                    }
                    if "level" in physical.keys()
                    else {}
                ),
            },
            n_batch,
            iters,
        )
        # Land back to exactly 0 in normalised space, which is what the model
        # emits and what the dataloader writes.  The climatology file stores land
        # as NaN and `_load_climatology` turns it into a physical 0, which
        # normalises to -mean/std -- a big number over a continent that would
        # otherwise reach the sea-ice area integral.
        return self.module.apply_wet_mask(normalised)


def build_forecasters(
    module,
    include_baselines: bool = True,
    model_label: str = "Model",
    climatology_path: str | Path | None = None,
) -> list[Forecaster]:
    """The model first, then the baselines in :data:`BASELINE_KEYS` order.

    Args:
        module: loaded forecast module.
        include_baselines: set False for ``--skip-baselines``.
        model_label: legend label for the model curve.
        climatology_path: override, for tests.
    """
    forecasters: list[Forecaster] = [ModelForecast(module, label=model_label)]
    if include_baselines:
        device = next(module.parameters()).device
        forecasters.append(PersistenceForecast(module))
        forecasters.append(
            ClimatologyForecast(module, climatology_path=climatology_path).to(device)
        )
    return forecasters
