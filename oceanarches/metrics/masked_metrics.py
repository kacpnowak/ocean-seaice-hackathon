"""Land-aware deterministic metrics: RMSE, MAE, bias and ACC.

Why this file exists at all
--------------------------
30.4% of the 1-degree grid is land at the surface and 42.5% at 1684 m, and
after normalisation the dataloader fills land with 0.  A metric that averages
over *grid points* therefore measures, for a third of its samples, how well the
model reproduces a constant -- which it does perfectly.  A model that predicts
nothing but zeros would score respectably, and the RMSE you report would be a
number about the coastline rather than about the ocean.

So every reduction here divides by the ocean area, not by the grid area:

    metric[var, depth] = sum_(lat,lon) w(lat) m(var,depth,lat,lon) f(x, y)
                         ------------------------------------------------
                              sum_(lat,lon) w(lat) m(var,depth,lat,lon)

with ``m`` the wet mask from :mod:`oceanarches.dataloaders.masks` (per depth for
3-D fields -- the coastline moves down the continental shelf) and ``w`` the
latitude weight.  Written that way a *constant* error field of size ``e`` gives
exactly ``e`` whatever the land distribution and whatever the latitudes are,
which is the property tests/test_metrics.py pins down.

What is reused from geoarches, and why
--------------------------------------
:class:`~geoarches.metrics.metric_base.MetricBase` for the pluggable latitude
weighting, :class:`~geoarches.metrics.metric_base.TensorDictMetricBase` to run
one metric per state group, and
:class:`~geoarches.metrics.label_wrapper.LabelDictWrapper` +
:func:`~geoarches.metrics.label_wrapper.add_timedelta_index` for the labels, so
the output keys (``rmse_thetao0m_24h``) drop straight into the training logs and
into ``convert_metric_dict_to_xarray`` without a translation layer.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import torch
import xarray as xr
from geoarches.metrics.label_wrapper import LabelDictWrapper, add_timedelta_index
from geoarches.metrics.metric_base import MetricBase, TensorDictMetricBase
from tensordict.tensordict import TensorDict
from torchmetrics import Metric

from .. import paths
from ..dataloaders import variables as V
from ..dataloaders.masks import load_masks, state_mask

__all__ = [
    "MaskedDeterministic",
    "MaskedDeterministicMetrics",
    "compute_lat_weights_glorys",
    "ocean_area_weights",
    "sample_count_or_nan",
]

#: Variables logged every training step.  The full set is 7 surface variables
#: plus 4 x 13 level channels, i.e. ~200 scalars per step through
#: ``self.log(..., sync_dist=True)``; that is a real cost under DDP and it makes
#: the progress bar unreadable.  ``headline_only=True`` (the default for the
#: train and val metric instances) cuts it to these.  Same idea as geoarches'
#: ``era5.get_headline_level_variable_indices``.
HEADLINE_SURFACE_VARIABLES = ("zos", "siconc", "sithick")
HEADLINE_LEVEL_VARIABLES = ("thetao", "so")


def sample_count_or_nan(nsamples: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """The denominator of a running mean: the sample count, or NaN if there is none.

    A metric that was never updated has all-zero sums, so dividing by
    ``nsamples.clamp(min=1)`` would report ``rmse = 0.0`` -- the best score a
    forecast can get, and indistinguishable from a perfect one.  An evaluation
    loop that silently skipped every batch would look like a triumph.  NaN says
    "no value" instead, and it propagates through any later average rather than
    dragging it towards zero.

    Args:
        nsamples: scalar integer state.
        like: a state tensor whose dtype and device the denominator should share.

    Returns:
        A scalar tensor: ``float(nsamples)``, or NaN when ``nsamples == 0``.
    """
    count = nsamples.to(dtype=like.dtype, device=like.device)
    # torch.where rather than a python `if`: this keeps compute() free of a
    # device-to-host synchronisation, which under DDP would be a stall.
    return torch.where(count > 0, count, torch.full_like(count, float("nan")))


# ---------------------------------------------------------------------------
# Latitude weighting
# ---------------------------------------------------------------------------
def compute_lat_weights_glorys(latitude_resolution: int) -> torch.Tensor:
    """``cos(lat)`` at the true GLORYS cell centres, normalised to mean 1.

    Our grid is *cell centred*: 180 rows at -89.5, -88.5, ... 89.5.  geoarches'
    :func:`~geoarches.metrics.metric_base.compute_lat_weights_weatherbench`
    assumes ``linspace(-90, 90, n)``, i.e. rows sitting on the cell *edges*, and
    ERA5 really is like that (121 rows from -90 to 90 inclusive).  Handing it 180
    would place the first row at the pole with a cell only half as tall, which is
    a different grid from the one our data is on.

    The two disagree by up to 75% in the polar rows (their weights differ by a
    factor of 4 at 89.5 degrees) but by only ~0.2% in an ocean-area-weighted mean
    of a smooth field, because those rows carry almost no area and, in the
    Southern Ocean, almost no water.  Numbers in the Task 6 report.

    Args:
        latitude_resolution: number of latitude rows.

    Returns:
        ``(latitude_resolution, 1)`` weights, mean 1 -- the shape
        ``MetricBase`` multiplies onto ``(..., lat, lon)``.
    """
    if latitude_resolution == 1:
        return torch.tensor(1.0)
    dlat = 180.0 / latitude_resolution
    # Cell centres, south first: -90 + dlat/2 ... 90 - dlat/2.
    lat = torch.arange(latitude_resolution, dtype=torch.float32) * dlat - 90.0 + dlat / 2
    weights = torch.cos(torch.deg2rad(lat))
    return (weights / weights.mean())[:, None]


def ocean_area_weights(
    mask: torch.Tensor,
    compute_lat_weights_fn: Callable[[int], torch.Tensor] = compute_lat_weights_glorys,
) -> torch.Tensor:
    """Per-channel spatial weights that sum to 1 over the ocean.

    This is the single definition of "average over the ocean" used by both the
    metrics and the training loss, so the two cannot drift apart.

    Args:
        mask: ``(var, depth, lat, lon)`` float, 1 over ocean and 0 over land.
        compute_lat_weights_fn: latitude weighting, ``(lat, 1)``.

    Returns:
        ``(var, depth, lat, lon)``, summing to 1 over ``(lat, lon)`` for every
        ``(var, depth)``.  A channel with no ocean at all gets all-zero weights
        (and a warning) rather than a division by zero; its metrics come out NaN,
        which is the honest answer.
    """
    lat_weights = compute_lat_weights_fn(mask.shape[-2]).to(mask.dtype)
    weights = mask * lat_weights
    total = weights.sum(dim=(-2, -1), keepdim=True)
    dry = total <= 0
    if bool(dry.any()):
        warnings.warn(
            f"{int(dry.sum())} of {dry.numel()} (variable, depth) channels have no ocean "
            "cell at all; their metrics will be NaN.",
            stacklevel=2,
        )
    return torch.where(dry, torch.zeros_like(weights), weights / total.clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_climatology(
    path: str, group: str, names: tuple[str, ...], depth_indices: tuple[int, ...] | None
) -> torch.Tensor:
    """Monthly climatology as ``(12, var, depth, lat, lon)``, land filled with 0.

    Cached: the train, val and inference metric instances all ask for the same
    thing, and the file is 95 MB on disk.

    Land is NaN in the file (that is what "no data" looks like on disk).  We
    replace it with 0 because ``0 * NaN`` is NaN, not 0 -- masking a NaN out does
    not remove it.  The value does not matter; the wet mask discards those cells.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Climatology not found: {path}\nrun: make stats   (or: make stats-quick)"
        )
    with xr.open_dataset(path) as ds:
        fields = []
        for name in names:
            if name not in ds:
                raise KeyError(
                    f"{Path(path).name} has no climatology for {name!r}. "
                    f"It holds {sorted(ds.data_vars)}. Re-run: make stats"
                )
            array = ds[name].to_numpy()
            if group == "surface":
                array = array[:, None]  # (12, 1, lat, lon) -- the length-1 depth axis
            elif depth_indices is not None:
                # The file carries all 14 prepared levels; take exactly the subset
                # the dataloader and the model took.  A mismatch here would score
                # 5 m water against a 15 m climatology and look merely "bad".
                array = array[:, list(depth_indices)]
            fields.append(array)
    return torch.from_numpy(np.stack(fields, axis=1)).float().nan_to_num(0.0)


def month_interpolation_weights(timestamp) -> tuple[int, int, float]:
    """``(month_before, month_after, alpha)`` for a day of the year.

    The climatology holds one field per calendar month.  We place each monthly
    field at the *midpoint* of its month and interpolate linearly between the two
    surrounding midpoints, wrapping from December to January -- so the result is
    periodic and continuous across the new year, and a field is only ever equal
    to the raw monthly mean in the middle of its own month.

    Working in real datetimes rather than in day-of-year numbers means February
    is 28 or 29 days long exactly when it should be, with no leap-year fudge.

    Args:
        timestamp: anything ``numpy.datetime64`` accepts, or seconds since the
            epoch as an int.

    Returns:
        ``(i0, i1, alpha)`` with 0-based month indices and
        ``clim = (1 - alpha) * clim[i0] + alpha * clim[i1]``.
    """
    if isinstance(timestamp, (int, np.integer)):
        time = np.datetime64(int(timestamp), "s")
    elif isinstance(timestamp, torch.Tensor):
        time = np.datetime64(int(timestamp.item()), "s")
    else:
        time = np.datetime64(timestamp).astype("datetime64[s]")

    def midpoint(month: np.datetime64) -> np.datetime64:
        start = month.astype("datetime64[s]")
        end = (month + 1).astype("datetime64[s]")
        return start + (end - start) // 2

    month = time.astype("datetime64[M]")
    if time >= midpoint(month):
        before, after = month, month + 1
    else:
        before, after = month - 1, month
    low, high = midpoint(before), midpoint(after)
    alpha = float((time - low) / (high - low))
    # .astype(object) turns datetime64[M] into a datetime.date, whose .month is
    # 1-12; the wrap from December to January is then automatic.
    return before.astype(object).month - 1, after.astype(object).month - 1, alpha


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------
class MaskedDeterministic(Metric, MetricBase):
    """RMSE / MAE / bias / ACC over ocean cells only, for one state group.

    Expects **denormalised** (physical unit) tensors shaped
    ``(batch, timedelta, var, depth, lat, lon)``, which is what geoarches'
    forecast modules hand their metrics.

    Args:
        mask: ``(var, depth, lat, lon)`` float wet mask for this group.
        lead_time_hours: gap between rollout steps, used to work out the valid
            time of each step for the ACC climatology.
        rollout_iterations: size of the ``timedelta`` axis.
        climatology: ``(12, var, depth, lat, lon)`` monthly climatology in the
            same units, or None to skip ACC.
        compute_lat_weights_fn: latitude weighting; see
            :class:`~geoarches.metrics.metric_base.MetricBase`.
    """

    #: torchmetrics: our compute() is not a simple elementwise function of the
    #: inputs, so forward() (update + compute in one) is not supported.
    full_state_update: bool = False

    def __init__(
        self,
        mask: torch.Tensor,
        lead_time_hours: int = 24,
        rollout_iterations: int = 1,
        climatology: torch.Tensor | None = None,
        compute_lat_weights_fn: Callable[[int], torch.Tensor] = compute_lat_weights_glorys,
    ):
        Metric.__init__(self)
        MetricBase.__init__(self, compute_lat_weights_fn=compute_lat_weights_fn)

        if mask.dim() != 4:
            raise ValueError(f"mask should be (var, depth, lat, lon), got {tuple(mask.shape)}")
        self.lead_time_hours = int(lead_time_hours)
        self.rollout_iterations = int(rollout_iterations)
        n_var, n_depth = mask.shape[0], mask.shape[1]
        data_shape = (self.rollout_iterations, n_var, n_depth)

        # Not persistent: both are derived from files that ship with the repo, and
        # writing 15 MB of mask into every checkpoint helps nobody.
        self.register_buffer("mask", mask.float(), persistent=False)
        self.register_buffer(
            "area_weights",
            ocean_area_weights(mask.float(), compute_lat_weights_fn),
            persistent=False,
        )
        if climatology is not None:
            if climatology.shape[1:] != mask.shape:
                raise ValueError(
                    f"climatology {tuple(climatology.shape)} does not match the mask "
                    f"{tuple(mask.shape)} on (var, depth, lat, lon)."
                )
            self.register_buffer("climatology", climatology.float(), persistent=False)
        else:
            self.climatology = None

        self.add_state("nsamples", default=torch.tensor(0), dist_reduce_fx="sum")
        for name in ("sum_mse", "sum_mae", "sum_bias"):
            self.add_state(name, default=torch.zeros(data_shape), dist_reduce_fx="sum")
        if self.has_climatology:
            self.add_state("sum_acc", default=torch.zeros(data_shape), dist_reduce_fx="sum")

    @property
    def has_climatology(self) -> bool:
        return getattr(self, "climatology", None) is not None

    # -- the ocean average --------------------------------------------------
    def ocean_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Latitude-weighted mean over ocean cells: ``(..., var, depth)``.

        The weights already sum to 1 per channel (see :func:`ocean_area_weights`),
        so this is a weighted *average* over the ocean, not over the grid.
        """
        return (x * self.area_weights).sum(dim=(-2, -1))

    # -- update -------------------------------------------------------------
    def update(
        self, targets: torch.Tensor, preds: torch.Tensor, timestamp: torch.Tensor | None = None
    ) -> None:
        """Accumulate one batch.

        Args:
            targets: ``(batch, timedelta, var, depth, lat, lon)``, denormalised.
            preds: same shape, denormalised.
            timestamp: ``(batch,)`` seconds since the epoch of the *initial*
                state.  Required when ACC is switched on: step ``i`` is valid at
                ``timestamp + (i + 1) * lead_time_hours``.
        """
        # float32 on purpose: training runs in bf16-mixed, and bf16 carries about
        # three decimal digits -- enough for a gradient, not for an RMSE.
        targets = targets.float()
        preds = preds.float()
        for name, tensor in (("targets", targets), ("preds", preds)):
            if tensor.dim() != 6:
                raise ValueError(
                    f"{name} should be (batch, timedelta, var, depth, lat, lon), got "
                    f"{tuple(tensor.shape)}. Single-step callers pass state[:, None]."
                )
        if targets.shape != preds.shape:
            raise ValueError(f"targets {tuple(targets.shape)} != preds {tuple(preds.shape)}")

        self.nsamples += preds.shape[0]
        error = preds - targets
        self.sum_mse = self.sum_mse + self.ocean_mean(error.pow(2)).sum(0)
        self.sum_mae = self.sum_mae + self.ocean_mean(error.abs()).sum(0)
        self.sum_bias = self.sum_bias + self.ocean_mean(error).sum(0)

        if self.has_climatology:
            self.sum_acc = self.sum_acc + self._anomaly_correlation(targets, preds, timestamp)

    def _anomaly_correlation(
        self, targets: torch.Tensor, preds: torch.Tensor, timestamp: torch.Tensor | None
    ) -> torch.Tensor:
        """ACC per (timedelta, var, depth), summed over the batch.

        ACC is computed per sample and per lead time and then averaged over
        samples, so a single number is a plain mean of correlations rather than a
        pooled correlation over a batch of different seasons.

        The climatology is interpolated to the *valid* time of each step (see
        :func:`month_interpolation_weights`), which is why this is a loop: a
        vectorised gather would materialise a (batch, timedelta, var, depth, lat,
        lon) climatology, about 1 GB for a 10-day rollout at batch 8.
        """
        if timestamp is None:
            raise ValueError(
                "ACC needs the initial time of each sample: call "
                "metric.update(targets, preds, timestamp=batch['timestamp']). "
                "Build the metric with compute_acc=False if you do not have it."
            )
        stamps = np.atleast_1d(torch.as_tensor(timestamp).detach().cpu().numpy()).astype("int64")
        n_batch, n_lead = preds.shape[0], preds.shape[1]
        if len(stamps) != n_batch:
            raise ValueError(f"got {len(stamps)} timestamps for a batch of {n_batch}")

        out = torch.zeros_like(self.sum_acc)
        for b in range(n_batch):
            for step in range(n_lead):
                valid = stamps[b] + (step + 1) * self.lead_time_hours * 3600
                i0, i1, alpha = month_interpolation_weights(int(valid))
                clim = torch.lerp(self.climatology[i0], self.climatology[i1], alpha)
                pred_anomaly = preds[b, step] - clim
                true_anomaly = targets[b, step] - clim
                covariance = self.ocean_mean(pred_anomaly * true_anomaly)
                pred_var = self.ocean_mean(pred_anomaly.pow(2))
                true_var = self.ocean_mean(true_anomaly.pow(2))
                # A channel whose anomaly is identically zero (an ice-free month
                # for siconc, say) has no correlation to speak of; report 0
                # rather than 0/0.
                denominator = (pred_var * true_var).sqrt()
                out[step] += torch.where(
                    denominator > 0,
                    covariance / denominator.clamp(min=1e-12),
                    torch.zeros_like(covariance),
                )
        return out

    # -- compute ------------------------------------------------------------
    def compute(self) -> Dict[str, torch.Tensor]:
        """Metric name -> ``(timedelta, var, depth)`` tensor.

        Names carry no underscore on purpose:
        :func:`~geoarches.metrics.label_wrapper.convert_metric_dict_to_xarray`
        splits labels on ``_`` and would mis-parse ``rmse_before_time_avg``.
        """
        count = sample_count_or_nan(self.nsamples, self.sum_mse)
        metrics = {
            "rmse": (self.sum_mse / count).sqrt(),
            "mae": self.sum_mae / count,
            "bias": self.sum_bias / count,
        }
        if self.has_climatology:
            metrics["acc"] = self.sum_acc / count
        return metrics


# ---------------------------------------------------------------------------
# The configured wrapper
# ---------------------------------------------------------------------------
class MaskedDeterministicMetrics(TensorDictMetricBase):
    """Masked RMSE / MAE / bias / ACC for a whole GLORYS state.

    One :class:`MaskedDeterministic` per state group, each wrapped in geoarches'
    :class:`~geoarches.metrics.label_wrapper.LabelDictWrapper` so that
    ``compute()`` returns flat, logger-friendly labels::

        {"rmse_siconc_24h": ..., "rmse_thetao0m_24h": ..., "acc_zos_24h": ...}

    Args:
        component: name in :data:`oceanarches.dataloaders.variables.COMPONENTS`.
            The metric scores the component's *prognostic* variables, which are
            exactly the channels the model returns.
        depth_indices: which of the 14 prepared levels the model uses.  Must be
            the same list the dataloader was built with -- the mask, the
            climatology and the depth labels all follow it.
        lead_time_hours: hours between rollout steps.
        rollout_iterations: number of rollout steps scored.
        headline_only: log only the handful of variables worth watching every
            step (see :data:`HEADLINE_SURFACE_VARIABLES`).  The metric is still
            computed for every channel; this only trims the labels.
        compute_acc: score the anomaly correlation against the monthly
            climatology.  Costs ~180 MB of device memory per instance.
        compute_lat_weights_fn: latitude weighting.
        masks_path, climatology_path: overrides, for tests.
    """

    def __init__(
        self,
        component: str = "full",
        depth_indices: Sequence[int] | None = None,
        lead_time_hours: int = 24,
        rollout_iterations: int = 1,
        headline_only: bool = False,
        compute_acc: bool = True,
        compute_lat_weights_fn: Callable[[int], torch.Tensor] = compute_lat_weights_glorys,
        masks_path: str | Path | None = None,
        climatology_path: str | Path | None = None,
    ):
        spec = V.get_component(component)
        depth_indices = list(depth_indices) if depth_indices is not None else None
        masks = load_masks(path=masks_path, depth_indices=depth_indices)
        surface_names, level_names = spec.prognostic_surface, spec.prognostic_level
        mask = state_mask(masks, surface_names, level_names)
        climatology_path = str(climatology_path or paths.climatology_file())

        groups = {}
        for group, names in (("surface", surface_names), ("level", level_names)):
            if not names:
                continue
            climatology = None
            if compute_acc:
                climatology = _load_climatology(
                    climatology_path,
                    group,
                    tuple(names),
                    tuple(depth_indices)
                    if depth_indices is not None and group == "level"
                    else None,
                )
            groups[group] = LabelDictWrapper(
                MaskedDeterministic(
                    mask=mask[group],
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                    climatology=climatology,
                    compute_lat_weights_fn=compute_lat_weights_fn,
                ),
                variable_indices=add_timedelta_index(
                    self._variable_indices(group, names, list(masks.depths), headline_only),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
            )
        super().__init__(**groups)
        self.component = spec.name
        self.depth_indices = depth_indices
        self.compute_acc = bool(compute_acc)

    @staticmethod
    def _variable_indices(
        group: str, names: list[str], depths: list[float], headline_only: bool
    ) -> dict[str, tuple]:
        """Label -> ``(var, depth)`` index, in the *component's* channel order.

        Deliberately not ``variables.headline_variable_indices()``: that helper
        looks names up in the canonical ``SURFACE_VARIABLES`` order, which is not
        the channel order of a component (the sea-ice component puts ``siconc``
        first).  Enumerating the component's own list is the only safe way.
        """
        if group == "surface":
            indices = V.surface_variable_indices(names)
            if headline_only:
                indices = {k: v for k, v in indices.items() if k in HEADLINE_SURFACE_VARIABLES}
            return indices
        indices = {}
        for var_index, name in enumerate(names):
            for depth_index, depth in enumerate(depths):
                if headline_only and not (name in HEADLINE_LEVEL_VARIABLES and depth_index == 0):
                    continue
                indices[f"{name}{depth:.0f}m"] = (var_index, depth_index)
        return indices

    def update(
        self, targets: TensorDict, preds: TensorDict, timestamp: torch.Tensor | None = None
    ) -> None:
        """Same contract as ``TensorDictMetricBase.update``, plus the timestamp.

        The timestamp is what lets ACC find the right day of the year; geoarches'
        signature has no room for it, so we add it as an optional keyword and
        forward it.
        """
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)
        for key, metric in self.metrics.items():
            metric.update(targets=targets[key], preds=preds[key], timestamp=timestamp)
