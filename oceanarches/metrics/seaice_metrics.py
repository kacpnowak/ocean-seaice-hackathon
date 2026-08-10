"""Sea-ice diagnostics on the 15% concentration contour, per hemisphere.

Sea ice is not scored the way temperature is.  What people publish, compare and
argue about is the position of the ice *edge* and the total area inside it, in
millions of square kilometres -- not a grid-point RMSE.  So this file computes
the three numbers the sea-ice literature uses:

``extent``  total area of every ocean cell whose concentration exceeds 15%.
            The 15% contour is the community's definition of "ice edge"; it comes
            from what passive-microwave satellites can reliably detect.
``area``    the same cells weighted by their concentration, i.e. how much ice
            there actually is.  Extent minus area is a measure of how broken up
            the pack is.
``IIEE``    the Integrated Ice-Edge Error (Goessling et al. 2016): the total area
            where model and truth disagree about whether there is ice.  Split
            into *overestimate* (model says ice, truth says open water) and
            *underestimate*, because a model that puts too much ice in one place
            and too little in another is wrong in a different way from one that
            is simply too icy.

Everything is reported in 10^6 km^2 so the numbers can be read against published
ones directly: Arctic September minimum is about 4-5, March maximum about 15.

**Expect this dataset to sit high against those published figures, and do not go
looking for a bug when it does.**  The Arctic March 2019 daily maximum computed
here is 18.1, not 15.  The area integral is right -- the surplus is the 1-degree
regrid (``cdo remap,r360x180`` from GLORYS' native 1/12 degree) smearing the ice
edge across whole 1-degree cells: 3.6 x 10^6 km^2 of the March total sits in
cells whose concentration is between 0.15 and 0.8, and raising the threshold to
0.8 brings the same days back to 14.6, right where the published number is.
Compare a model against the truth *on this grid*, which is what every metric in
this file does, rather than against a satellite product on another one.

Areas are real spherical cell areas, ``R^2 * dlon * dlat * cos(lat)`` with
R = 6371 km.  On a 1-degree grid that under-integrates ``cos`` by about 0.01%
against the exact ``R^2 * dlon * (sin(lat_north) - sin(lat_south))``; the tests
quantify it rather than hiding it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from geoarches.metrics.label_wrapper import LabelDictWrapper, add_timedelta_index
from geoarches.metrics.metric_base import MetricBase, TensorDictMetricBase
from tensordict.tensordict import TensorDict
from torchmetrics import Metric

from ..dataloaders import variables as V
from ..dataloaders.masks import load_masks
from .masked_metrics import compute_lat_weights_glorys, sample_count_or_nan

__all__ = ["SeaIceExtent", "SeaIceMetrics", "cell_areas", "EARTH_RADIUS_KM", "ICE_EDGE_THRESHOLD"]

#: Mean Earth radius, km.  The value used throughout the sea-ice literature.
EARTH_RADIUS_KM = 6371.0

#: The concentration that counts as "there is ice here".
ICE_EDGE_THRESHOLD = 0.15

#: Hemisphere labels, in the order of the metric's last axis.  No underscore --
#: `convert_metric_dict_to_xarray` splits labels on it.
HEMISPHERES = ("seaiceNH", "seaiceSH")

#: Reported when ``headline_only`` is on: the ice-edge error and the extent bias
#: say most of what there is to say, and a training log does not need eleven
#: sea-ice numbers per lead time.
HEADLINE_METRICS = ("iiee", "extentbias")


def cell_areas(n_lat: int = V.N_LAT, n_lon: int = V.N_LON) -> torch.Tensor:
    """Area of every grid cell in km^2, ``(n_lat, n_lon)``.

    ``R^2 * dlon * dlat * cos(lat)`` at the cell centres, with dlon and dlat in
    radians.  Cell centres, south first, matching
    :data:`oceanarches.dataloaders.variables.LAT`.
    """
    dlat = 180.0 / n_lat
    dlon = 360.0 / n_lon
    lat = torch.arange(n_lat, dtype=torch.float64) * dlat - 90.0 + dlat / 2
    row = EARTH_RADIUS_KM**2 * np.deg2rad(dlon) * np.deg2rad(dlat) * torch.cos(torch.deg2rad(lat))
    return row[:, None].expand(n_lat, n_lon).float().contiguous()


def hemisphere_masks(n_lat: int = V.N_LAT, n_lon: int = V.N_LON) -> torch.Tensor:
    """``(2, n_lat, n_lon)`` float masks: northern hemisphere first."""
    dlat = 180.0 / n_lat
    lat = torch.arange(n_lat, dtype=torch.float32) * dlat - 90.0 + dlat / 2
    north = (lat >= 0).float()[:, None].expand(n_lat, n_lon)
    return torch.stack([north, 1.0 - north]).contiguous()


class SeaIceExtent(Metric, MetricBase):
    """Extent, area and ice-edge error, per hemisphere and lead time.

    Expects **denormalised** surface tensors,
    ``(batch, timedelta, var, 1, lat, lon)``.

    Args:
        mask: ``(lat, lon)`` float ocean mask -- land carries no ice, and after
            denormalisation a land cell holds the climatological mean rather than
            zero, so leaving it in would invent an ice shelf over Africa.
        siconc_index: channel of ``siconc`` in the tensor, or None if this
            component does not predict sea ice (then the metric reports nothing).
        rollout_iterations: size of the timedelta axis.
        threshold: concentration that counts as ice.
        headline_only: report only :data:`HEADLINE_METRICS`.
    """

    full_state_update: bool = False

    def __init__(
        self,
        mask: torch.Tensor,
        siconc_index: int | None,
        rollout_iterations: int = 1,
        threshold: float = ICE_EDGE_THRESHOLD,
        headline_only: bool = False,
    ):
        Metric.__init__(self)
        MetricBase.__init__(self, compute_lat_weights_fn=compute_lat_weights_glorys)

        if mask.dim() != 2:
            raise ValueError(f"mask should be (lat, lon), got {tuple(mask.shape)}")
        n_lat, n_lon = mask.shape
        self.siconc_index = None if siconc_index is None else int(siconc_index)
        self.threshold = float(threshold)
        self.rollout_iterations = int(rollout_iterations)
        self.headline_only = bool(headline_only)

        # (2, lat, lon) in units of 10^6 km^2, ocean only: multiply an indicator
        # field by this and sum to get an area straight away.
        weights = cell_areas(n_lat, n_lon) * mask.float() / 1e6
        self.register_buffer(
            "hemisphere_areas", weights * hemisphere_masks(n_lat, n_lon), persistent=False
        )

        shape = (self.rollout_iterations, len(HEMISPHERES))
        self.add_state("nsamples", default=torch.tensor(0), dist_reduce_fx="sum")
        for name in self._state_names():
            self.add_state(name, default=torch.zeros(shape), dist_reduce_fx="sum")

    @staticmethod
    def _state_names() -> tuple[str, ...]:
        return (
            "sum_extent_pred",
            "sum_extent_truth",
            "sum_extent_bias",
            "sum_extent_mae",
            "sum_area_bias",
            "sum_area_mae",
            "sum_iiee",
            "sum_iiee_over",
            "sum_iiee_under",
        )

    def _integrate(self, field: torch.Tensor) -> torch.Tensor:
        """``(..., lat, lon)`` -> ``(..., hemisphere)`` area integral in 10^6 km^2."""
        return torch.einsum("...ij,hij->...h", field, self.hemisphere_areas)

    def update(
        self, targets: torch.Tensor, preds: torch.Tensor, timestamp: torch.Tensor | None = None
    ) -> None:
        """Accumulate one batch.

        Args:
            targets: ``(batch, timedelta, var, 1, lat, lon)``, denormalised.
            preds: same shape.
            timestamp: unused; accepted so that every metric in a configured list
                takes the same call.
        """
        if targets.dim() != 6 or preds.dim() != 6:
            raise ValueError(
                "targets and preds should be (batch, timedelta, var, 1, lat, lon), got "
                f"{tuple(targets.shape)} and {tuple(preds.shape)}."
            )
        self.nsamples += preds.shape[0]
        if self.siconc_index is None:
            return

        # (batch, timedelta, lat, lon), float32: the concentrations arrive in bf16
        # during mixed-precision training, and an area integral over 60000 cells
        # needs more than three decimal digits.
        truth = targets[:, :, self.siconc_index, 0].float()
        pred = preds[:, :, self.siconc_index, 0].float()
        ice_truth = (truth > self.threshold).float()
        ice_pred = (pred > self.threshold).float()

        extent_truth = self._integrate(ice_truth)
        extent_pred = self._integrate(ice_pred)
        # Concentration is clamped into [0, 1] before integrating: a model without
        # the physical-bounds clamp can emit -0.2, and a negative ice area is not
        # a diagnostic, it is a bug that hides itself by cancellation.
        area_truth = self._integrate(truth.clamp(0.0, 1.0))
        area_pred = self._integrate(pred.clamp(0.0, 1.0))

        over = self._integrate(ice_pred * (1.0 - ice_truth))
        under = self._integrate((1.0 - ice_pred) * ice_truth)

        self.sum_extent_pred = self.sum_extent_pred + extent_pred.sum(0)
        self.sum_extent_truth = self.sum_extent_truth + extent_truth.sum(0)
        self.sum_extent_bias = self.sum_extent_bias + (extent_pred - extent_truth).sum(0)
        self.sum_extent_mae = self.sum_extent_mae + (extent_pred - extent_truth).abs().sum(0)
        self.sum_area_bias = self.sum_area_bias + (area_pred - area_truth).sum(0)
        self.sum_area_mae = self.sum_area_mae + (area_pred - area_truth).abs().sum(0)
        self.sum_iiee = self.sum_iiee + (over + under).sum(0)
        self.sum_iiee_over = self.sum_iiee_over + over.sum(0)
        self.sum_iiee_under = self.sum_iiee_under + under.sum(0)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Metric name -> ``(timedelta, hemisphere)`` tensor, in 10^6 km^2.

        Metric names carry no underscore: geoarches'
        ``convert_metric_dict_to_xarray`` splits labels on it.
        """
        if self.siconc_index is None:
            return {}
        count = sample_count_or_nan(self.nsamples, self.sum_iiee)
        metrics = {
            "iiee": self.sum_iiee / count,
            "iieeover": self.sum_iiee_over / count,
            "iieeunder": self.sum_iiee_under / count,
            "extentbias": self.sum_extent_bias / count,
            "extentmae": self.sum_extent_mae / count,
            "extentpred": self.sum_extent_pred / count,
            "extenttruth": self.sum_extent_truth / count,
            "areabias": self.sum_area_bias / count,
            "areamae": self.sum_area_mae / count,
        }
        if self.headline_only:
            return {name: metrics[name] for name in HEADLINE_METRICS}
        return metrics


class SeaIceMetrics(TensorDictMetricBase):
    """Sea-ice extent, area and ice-edge error for a configured component.

    Produces labels like ``iiee_seaiceNH_24h`` and ``extentbias_seaiceSH_240h``.

    A component that does not predict ``siconc`` (the ocean specialist) is not an
    error: the metric is built, reports nothing, and stays in the configured list
    so that the same ``module=...`` works with every dataloader.

    Args:
        component: name in :data:`oceanarches.dataloaders.variables.COMPONENTS`.
        lead_time_hours, rollout_iterations: rollout labelling.
        headline_only: report only :data:`HEADLINE_METRICS`.
        threshold: concentration that counts as ice.
        depth_indices: unused (sea ice is 2-D); accepted so that every metric in a
            config group takes the same arguments.
        masks_path: override, for tests.
    """

    def __init__(
        self,
        component: str = "full",
        lead_time_hours: int = 24,
        rollout_iterations: int = 1,
        headline_only: bool = False,
        threshold: float = ICE_EDGE_THRESHOLD,
        depth_indices: Sequence[int] | None = None,
        masks_path: str | Path | None = None,
    ):
        spec = V.get_component(component)
        names = spec.prognostic_surface
        siconc_index = names.index("siconc") if "siconc" in names else None
        if siconc_index is None:
            warnings.warn(
                f"Component {spec.name!r} does not predict siconc, so the sea-ice metrics "
                "will report nothing. Drop `glorys_seaice` from the module's metric list "
                "if you want that made explicit.",
                stacklevel=2,
            )
        masks = load_masks(path=masks_path)

        variable_indices = (
            {name: (index,) for index, name in enumerate(HEMISPHERES)}
            if siconc_index is not None
            else {}
        )
        super().__init__(
            surface=LabelDictWrapper(
                SeaIceExtent(
                    mask=masks.wet_surface.float(),
                    siconc_index=siconc_index,
                    rollout_iterations=rollout_iterations,
                    threshold=threshold,
                    headline_only=headline_only,
                ),
                variable_indices=add_timedelta_index(
                    variable_indices,
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
            )
        )
        self.component = spec.name

    def update(
        self, targets: TensorDict, preds: TensorDict, timestamp: torch.Tensor | None = None
    ) -> None:
        """As ``TensorDictMetricBase.update``; ``timestamp`` is accepted and ignored."""
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)
        for key, metric in self.metrics.items():
            metric.update(targets=targets[key], preds=preds[key], timestamp=timestamp)
