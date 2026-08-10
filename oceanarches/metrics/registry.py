"""Make the GLORYS metrics reachable by name from geoarches' evaluation script.

``geoarches.evaluation.eval_multistep`` takes ``--metrics <name> <name> ...`` and
looks each name up in :mod:`geoarches.evaluation.metric_registry`, which ships
only the ERA5 metrics.  Registering ours there means

    python -m geoarches.evaluation.eval_multistep ... --metrics glorys_deterministic

works on our data with no change to geoarches, and Task 7's evaluation pipeline
can discover what is available instead of hardcoding a list.

Registration is idempotent and happens on ``import oceanarches.metrics``.
"""

from __future__ import annotations

from geoarches.evaluation.metric_registry import available_metrics, register_metric

from ..dataloaders import variables as V
from .masked_metrics import MaskedDeterministicMetrics
from .seaice_metrics import SeaIceMetrics

__all__ = ["register_all", "available_metrics"]


def register_all() -> list[str]:
    """Register every GLORYS metric and return the names now available.

    One entry per metric and one per (metric, component) pair: the evaluation
    script passes only ``--metrics <name>`` and has nowhere to say which
    component a checkpoint belongs to, so the component has to be baked into the
    name.  Metrics for the ``full`` component keep the short name.
    """
    register_metric("glorys_deterministic", MaskedDeterministicMetrics)
    register_metric("glorys_seaice", SeaIceMetrics)
    for component in V.COMPONENTS:
        register_metric(
            f"glorys_deterministic_{component}", MaskedDeterministicMetrics, component=component
        )
        register_metric(f"glorys_seaice_{component}", SeaIceMetrics, component=component)
    return available_metrics()
