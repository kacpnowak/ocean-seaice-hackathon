"""Land-aware metrics for GLORYS forecasts.

Importing this package registers every metric with geoarches' evaluation
registry (see :mod:`oceanarches.metrics.registry`), so that
``geoarches.evaluation.eval_multistep`` can build them by name.
"""

from .masked_metrics import (
    MaskedDeterministic,
    MaskedDeterministicMetrics,
    compute_lat_weights_glorys,
)
from .registry import register_all
from .seaice_metrics import SeaIceExtent, SeaIceMetrics, cell_areas

register_all()

__all__ = [
    "MaskedDeterministic",
    "MaskedDeterministicMetrics",
    "SeaIceExtent",
    "SeaIceMetrics",
    "cell_areas",
    "compute_lat_weights_glorys",
    "register_all",
]
