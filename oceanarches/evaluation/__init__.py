"""The evaluation pipeline: rollout, baselines, figures, animations, report.

One command turns a trained checkpoint into an answer to *is my model any good?*::

    make eval NAME=my_run LEAD_DAYS=10

The pieces, in the order :mod:`~oceanarches.evaluation.run_eval` uses them:

``baselines``   the forecasts every model has to beat -- persistence and
                climatology -- behind the same interface as the model itself, so
                all three go through one scoring loop and cannot drift apart.
``provenance``  how deeply ``oceanarches/stats/`` was sampled, and the warning a
                report built on ``make stats-quick`` statistics has to carry.
``render_cache`` the figures and the animations, cached on the rollout's terms.
``plots``       every figure.
``animate``     mp4 (with a GIF fallback) rollout animations.
``report``      ``report.md`` and a self-contained ``report.html``.
``run_eval``    the command line that stitches them together.
"""

from __future__ import annotations

__all__ = [
    "baselines",
    "plots",
    "animate",
    "provenance",
    "render_cache",
    "report",
    "run_eval",
]
