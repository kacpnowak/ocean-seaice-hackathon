"""Every figure the evaluation produces, on one colour system.

The figures are the main thing a participant looks at, so they follow one set of
rules rather than matplotlib's defaults:

**Colour is assigned by the job it does, not by taste.**

* *Identity* (which forecast) is categorical: the model is blue, persistence is
  orange, climatology is aqua, and a second model variant -- what Task 8's
  coupled/uncoupled overlay needs -- is violet.  Those four hexes are the
  documented categorical slots 1, 2, 3 and 7, and the set clears the
  colour-vision checks on the strictest (all-pairs) list: worst simulated
  protan/deutan dE 9.2 and worst normal-vision dE 16.3, against thresholds of 8
  and 15.  The *observed* series is not a forecast and does not take an identity
  slot -- it is drawn in primary ink, solid, and named in the legend.
* *Magnitude* (a concentration, a speed, an error size) is one hue, light to
  dark: the documented blue ramp, steps 100 to 700.
* *Polarity* (an anomaly, a difference, a signed error, a skill score) is
  diverging and always centred on zero, with a neutral grey midpoint so that
  "no difference" reads as nothing.  The warm arm is generated in OKLCh from the
  cool arm -- same lightness, same chroma, the documented red's hue -- so the two
  sides are perceptually symmetric and a reader cannot mistake which one is
  bigger.  See :func:`_diverging_colormap`.

**Land is never left as a value.**  30.4% of the surface grid and 42.5% of the
deepest prepared level (1684 m) is land; the cached fields carry it as NaN and every map paints it
in a neutral grey through ``cmap.set_bad``.  Land drawn as 0 would read as a real
sea surface height of zero and as a real ice-free ocean.

**Every axis carries its unit**, taken from
:mod:`oceanarches.dataloaders.variables`, which is the single source of truth for
what a variable is and what it is measured in.

Two accessibility notes.  Aqua (climatology) sits at 2.7:1 against the figure
surface, below the 3:1 mark-contrast target.  The documented relief for that is a
visible direct label *or* a table view; this figure set uses the table view --
``report.md`` and ``report.html`` tabulate the numbers behind every curve and
every heatmap cell -- because with three converging error curves per panel,
direct end-labels would collide and detach from their lines, which the same
guidance rules out.  Direct end-labels are used where the series do separate (the
free-running time series).  Each forecast additionally carries its own dash
pattern and marker, so the figures survive greyscale printing and full-severity
colour blindness without relying on hue at all.

One convention holds across every panel that uses the diverging map: **warm means
"more of the bad thing"**.  A warm error cell is a prediction that is too high; a
warm scorecard cell is a forecast with more error than the baseline.  The
alternative -- plotting a skill score, where warm would mean *good* -- was
rejected for exactly that reason.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from ..dataloaders import variables as V  # noqa: E402
from ..metrics.seaice_metrics import ICE_EDGE_THRESHOLD  # noqa: E402

__all__ = [
    "SERIES_COLOURS",
    "sequential_colormap",
    "diverging_colormap",
    "apply_theme",
    "plot_rmse_vs_lead",
    "plot_scorecard",
    "scorecard_table",
    "relative_error",
    "plot_seaice_vs_lead",
    "plot_map_triptych",
    "plot_depth_hovmoller",
    "plot_seaice_polar",
    "plot_free_timeseries",
    "plot_power_spectra",
    "plot_overlay",
    "render_all",
    "parse_metric_variable",
    "variable_label",
]

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
#: Land, and any other "no data here" cell.  Neutral, recessive, and not a step
#: of any data ramp, so it can never be mistaken for a value.
#:
#: Known shortfall, left as it is on purpose.  Against the diverging map's
#: neutral midpoint (#f0efec) this grey is 1.55:1, below the 3:1 this file asks
#: of everything else.  The darkest grey that still clears 3:1 there is about
#: #8a8a86 (3.01:1) -- and that is exactly the problem: its luminance is 0.253,
#: which sits *inside* the ramp's lightness range, 1.05:1 against the cool step
#: #3987e5 and 1.12:1 against the warm step #d75853.  A land colour indis-
#: tinguishable in lightness from a mid-ramp value is a worse failure than a
#: quiet one, because it reads AS a value.
#:
#: What actually separates land from the ramp here is chroma, not lightness:
#: #c3c2b7 is neutral, and every step of the ramp is saturated.  (Its luminance,
#: 0.536, is nearly identical to the cool step #9ec5f4 at 0.537, and nobody
#: confuses them.)  So the shortfall is against a lightness target the design
#: does not actually rely on.
#:
#: The relief is textual and is now on every map figure: the title says
#: "Grey is land: no data, not a value", and every figure has a table of the
#: same numbers in report.md.
LAND = "#c3c2b7"
NEUTRAL_MIDPOINT = "#f0efec"

#: Identity slots.  Fixed per entity, never reassigned by rank -- a reader who
#: learned "the model is blue" in figure 1 must still be right in figure 8.
SERIES_COLOURS = {
    "model": "#2a78d6",  # categorical slot 1
    "persistence": "#eb6834",  # slot 2
    "climatology": "#1baf7a",  # slot 3
    "variant": "#4a3aa7",  # slot 7 -- Task 8's second model
    "variant2": "#a8358f",  # slot 8 -- a third arm (e.g. ground-truth-forced sea ice)
    "truth": INK,  # observed: chrome ink, not an identity slot
}
SERIES_DASHES = {
    "model": "-",
    "persistence": "--",
    "climatology": ":",
    "variant": "-.",
    "variant2": (0, (3, 1, 1, 1, 1, 1)),
    "truth": "-",
}
SERIES_MARKERS = {
    "model": "o",
    "persistence": "s",
    "climatology": "^",
    "variant": "D",
    "variant2": "v",
    "truth": None,
}

#: Model identity slots, in the order :func:`plot_overlay` hands them out.  There
#: are three and no more: a fourth arm would have to reuse one of these, and two
#: curves in the same colour and dash with different legend labels is worse than
#: no figure at all.
MODEL_ROLES = ("model", "variant", "variant2")

#: Documented blue ramp, steps 100 -> 700.  Used for magnitude.
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: The documented red whose hue the warm arm of the diverging map borrows.
_RED_POLE = "#e34948"
#: Two steps of the blue ramp for "same entity, different lead time".
ORDINAL_BLUE = ("#6da7ec", "#184f95")


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _hex_to_oklab(colour: str) -> tuple[float, float, float]:
    text = colour.lstrip("#")
    r, g, b = (_srgb_to_linear(int(text[i : i + 2], 16) / 255) for i in (0, 2, 4))
    lc = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    mc = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    sc = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * lc + 0.7936177850 * mc - 0.0040720468 * sc,
        1.9779984951 * lc - 2.4285922050 * mc + 0.4505937099 * sc,
        0.0259040371 * lc + 0.7827717662 * mc - 0.8086757660 * sc,
    )


def _oklab_to_hex(lightness: float, a: float, b: float) -> str:
    l_, m_, s_ = (
        lightness + 0.3963377774 * a + 0.2158037573 * b,
        lightness - 0.1055613458 * a - 0.0638541728 * b,
        lightness - 0.0894841775 * a - 1.2914855480 * b,
    )
    lc, mc, sc = l_**3, m_**3, s_**3
    channels = (
        4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc,
        -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc,
        -0.0041960863 * lc - 0.7034186147 * mc + 1.7076147010 * sc,
    )
    return "#" + "".join(f"{round(255 * _linear_to_srgb(c)):02x}" for c in channels)


def _mirror_hue(colour: str, hue_source: str) -> str:
    """``colour`` re-hued to ``hue_source``'s hue, keeping its lightness and chroma.

    This is how the warm arm of the diverging map is built from the documented
    cool ramp: every warm step has the *same* OKLCh lightness and chroma as its
    cool partner, so a reader comparing +0.4 against -0.4 is comparing two colours
    that are equally far from the neutral midpoint.  Hand-picking a red ramp would
    almost certainly make one arm louder than the other, which on an error map is
    a claim about the data.

    The seven derived warm steps are, light to dark: ``#fad6d2 #f1aea8 #e4857e
    #d75853 #b13f3c #892b2a #621b1a``.
    """
    lightness, a, b = _hex_to_oklab(colour)
    chroma = math.hypot(a, b)
    _, ha, hb = _hex_to_oklab(hue_source)
    hue = math.atan2(hb, ha)
    return _oklab_to_hex(lightness, chroma * math.cos(hue), chroma * math.sin(hue))


def sequential_colormap(name: str = "oceanarches_seq"):
    """One hue, light to dark: magnitude.  Land (NaN) renders in neutral grey."""
    return LinearSegmentedColormap.from_list(name, _BLUE_RAMP).with_extremes(bad=LAND)


def diverging_colormap(name: str = "oceanarches_div"):
    """Two opposed hues around a neutral grey midpoint: polarity, centred on zero."""
    warm = [_mirror_hue(step, _RED_POLE) for step in _BLUE_RAMP]
    colours = list(reversed(_BLUE_RAMP)) + [NEUTRAL_MIDPOINT] + warm
    return LinearSegmentedColormap.from_list(name, colours).with_extremes(bad=LAND)


SEQUENTIAL = sequential_colormap()
DIVERGING = diverging_colormap()

# Cartopy refuses to wrap a pcolormesh whose masked colour is opaque, and warns
# on every call. The map variants therefore leave masked cells *transparent* and
# `_draw_field` paints the axes background in LAND instead, which puts the same
# grey on screen with none of the wrapping trouble.
SEQUENTIAL_MAP = sequential_colormap("oceanarches_seq_map").with_extremes(bad=(0, 0, 0, 0))
DIVERGING_MAP = diverging_colormap("oceanarches_div_map").with_extremes(bad=(0, 0, 0, 0))


def apply_theme() -> None:
    """Recessive chrome, generous padding, hairline solid grid.  Idempotent."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS_RULE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # never dashed: a dashed grid reads as a threshold
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "figure.constrained_layout.use": True,
        }
    )


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
_LEVEL_LABEL = re.compile(r"^([a-zA-Z]+)(\d+)m$")


def parse_metric_variable(label: str) -> tuple[str, float | None]:
    """``"thetao0m"`` -> ``("thetao", 0.0)``; ``"siconc"`` -> ``("siconc", None)``.

    The metric labels are built by ``variables.level_variable_indices`` as
    ``f"{name}{depth:.0f}m"``, so the depth is recoverable but rounded -- 0.494 m
    is written ``0m``.  Callers that need the exact depth take it from the
    dataset, not from the label.
    """
    match = _LEVEL_LABEL.match(label)
    if match and match.group(1) in V.VARIABLES:
        return match.group(1), float(match.group(2))
    return label, None


def variable_units(label: str) -> str:
    name, _ = parse_metric_variable(label)
    meta = V.VARIABLES.get(name)
    return meta.units if meta else ""


def variable_label(label: str, with_units: bool = True) -> str:
    """A human title: ``"thetao0m"`` -> ``"Sea water potential temperature @ 0 m"``."""
    name, depth = parse_metric_variable(label)
    meta = V.VARIABLES.get(name)
    title = meta.long_name if meta else name
    if depth is not None:
        title = f"{title} @ {depth:g} m"
    if with_units and meta and meta.units:
        title = f"{title} [{meta.units}]"
    return title


def _is_diverging_variable(label: str) -> bool:
    name, _ = parse_metric_variable(label)
    meta = V.VARIABLES.get(name)
    return bool(meta and meta.diverging)


# ---------------------------------------------------------------------------
# Small helpers over a RolloutResult
# ---------------------------------------------------------------------------
def _lead_days(dataset: xr.Dataset) -> np.ndarray:
    return dataset["prediction_timedelta"].to_numpy().astype("timedelta64[h]").astype(float) / 24.0


def _series(result, key: str, metric: str, variable: str):
    """``(lead_days, values)`` for one forecaster, or None when absent."""
    for group in result.metrics.get(key, {}).values():
        if variable in group.data_vars and metric in list(group["metric"].values):
            values = group[variable].sel(metric=metric).to_numpy()
            return _lead_days(group), np.asarray(values, dtype=float)
    return None


def _forecaster_order(result) -> list[str]:
    order = ["model", "persistence", "climatology"]
    keys = list(result.metrics)
    return [k for k in order if k in keys] + [k for k in keys if k not in order]


def _label_for(result, key: str) -> str:
    return result.labels.get(key, key.capitalize())


def _style(key: str) -> dict:
    role = key if key in SERIES_COLOURS else "variant"
    return dict(
        color=SERIES_COLOURS[role],
        linestyle=SERIES_DASHES[role],
        marker=SERIES_MARKERS[role],
        markersize=4.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
    )


def _end_label(ax, x, y, text: str) -> None:
    """One direct label at the last point.

    Selective by construction -- exactly one per series, at the end -- which is
    what makes direct labels readable.  It is also the documented relief for the
    two identity hues that sit below 3:1 against the figure surface.
    """
    if len(x) == 0 or not np.isfinite(y[-1]):
        return
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=7,
        color=INK_SECONDARY,
        clip_on=False,
    )


def _legend(fig, keys: Sequence[str], labels: Sequence[str], ncol: int | None = None) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=_style(k)["color"],
            linestyle=_style(k)["linestyle"],
            marker=_style(k)["marker"],
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            linewidth=2.0,
        )
        for k in keys
    ]
    # "outside lower center" makes constrained_layout reserve a band for the
    # legend; plain "lower center" draws it on top of the x-axis labels.
    fig.legend(
        handles,
        list(labels),
        loc="outside lower center",
        ncol=ncol or min(len(keys), 4),
        labelcolor=INK_SECONDARY,
    )


#: The variables the headline figure and the report lead with.  Surface first,
#: then the shallowest level of the two tracers -- the ones a participant checks.
HEADLINE_VARIABLES = ("thetao0m", "siconc", "zos", "sithick", "so0m", "mlotst")


def _resolve_headline(result) -> list[str]:
    """Headline labels that exist in this run, with the depth suffix repaired.

    A component with different depth levels writes ``thetao5m`` rather than
    ``thetao0m``, so the shallowest available level of each level variable is
    substituted instead of silently dropping the panel.
    """
    available = set()
    for group in result.metrics.get("model", {}).values():
        available.update(group.data_vars)
    resolved = []
    for wanted in HEADLINE_VARIABLES:
        if wanted in available:
            resolved.append(wanted)
            continue
        name, depth = parse_metric_variable(wanted)
        if depth is None:
            continue
        candidates = [
            (parse_metric_variable(label)[1], label)
            for label in available
            if parse_metric_variable(label)[0] == name
            and parse_metric_variable(label)[1] is not None
        ]
        if candidates:
            resolved.append(min(candidates)[1])
    return resolved


# ---------------------------------------------------------------------------
# 1. RMSE against lead time -- the headline figure
# ---------------------------------------------------------------------------
def plot_rmse_vs_lead(result, out_path: Path, variables: Sequence[str] | None = None) -> Path:
    """RMSE per variable against lead time, with both baselines on the same axes.

    Reading it: the model curve should start below persistence (a day-1 forecast
    worse than "nothing changes" means something is broken) and should stay below
    climatology for as long as it is useful.  **Where the model crosses the
    climatology curve is the honest end of its forecast horizon.**
    """
    apply_theme()
    variables = list(variables or _resolve_headline(result))
    keys = _forecaster_order(result)
    n = len(variables)
    ncols = min(3, n) or 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows), squeeze=False)

    for index, variable in enumerate(variables):
        ax = axes[index // ncols][index % ncols]
        for key in keys:
            series = _series(result, key, "rmse", variable)
            if series is None:
                continue
            lead, values = series
            ax.plot(lead, values, **_style(key), zorder=3 if key == "model" else 2)
        ax.set_title(variable_label(variable, with_units=False), color=INK)
        ax.set_xlabel("Lead time [days]")
        ax.set_ylabel(f"RMSE [{variable_units(variable) or '1'}]")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
    for index in range(n, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    fig.suptitle(
        "Forecast error against lead time, ocean cells only",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    _legend(fig, keys, [_label_for(result, k) for k in keys])
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# 2. Scorecard
# ---------------------------------------------------------------------------
def relative_error(model: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``model / reference - 1``: **positive means the model has more error**.

    Deliberately not a skill score.  A skill score is positive when the model is
    good, which on the diverging colour map would paint "good" warm and "bad"
    cool -- the opposite of what every reader expects, and the opposite of what
    the error maps in this same report do.  Framing it as a relative error keeps
    one rule for the whole figure set: warm is more of the bad thing.

    Read a cell as a percentage: ``-16`` means the model's error is 16% smaller
    than the baseline's; ``+9`` means it is 9% larger.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(model, dtype=float) / np.asarray(reference, dtype=float) - 1.0
    return np.where(np.isfinite(out), out, np.nan)


def _canonical_order(labels: list[str]) -> list[str]:
    """Sort metric labels by the canonical variable order, then by depth.

    Alphabetical order would interleave sea ice with the deep ocean and put
    ``bottomT`` first; ``variables.py`` already defines the order the whole
    project uses, so the scorecard follows it.
    """
    order = {name: i for i, name in enumerate(V.ALL_VARIABLES)}

    def key(label: str):
        name, depth = parse_metric_variable(label)
        return (order.get(name, len(order)), depth if depth is not None else -1.0)

    return sorted(labels, key=key)


def scorecard_table(
    result, reference: str = "persistence"
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """``(row labels, lead days, matrix)`` -- the numbers behind the scorecard.

    The matrix is :func:`relative_error`, so a negative entry is a model that
    beats the baseline.  Returned separately from the figure because it is also
    the figure's table view: ``report.md`` prints it verbatim.

    Level variables appear at their shallowest and deepest level only.  The whole
    depth structure is the Hovmoller's job; putting 52 rows on a scorecard would
    make the one-glance panel unreadable.
    """
    rows, values = [], []
    deterministic = result.metrics.get("model", {}).get("glorys_deterministic_metrics")
    if deterministic is not None:
        surface_names = [v for v in deterministic.data_vars if parse_metric_variable(v)[1] is None]
        level_names = [
            v for v in deterministic.data_vars if parse_metric_variable(v)[1] is not None
        ]
        depths = sorted({parse_metric_variable(v)[1] for v in level_names})
        keep = {depths[0], depths[-1]} if depths else set()
        chosen = _canonical_order(surface_names) + _canonical_order(
            [v for v in level_names if parse_metric_variable(v)[1] in keep]
        )
        for name in chosen:
            model = _series(result, "model", "rmse", name)
            other = _series(result, reference, "rmse", name)
            if model is None or other is None:
                continue
            rows.append(name)
            values.append(relative_error(model[1], other[1]))
    seaice = result.metrics.get("model", {}).get("glorys_seaice_metrics")
    if seaice is not None:
        for name in sorted(seaice.data_vars):
            model = _series(result, "model", "iiee", name)
            other = _series(result, reference, "iiee", name)
            if model is None or other is None:
                continue
            rows.append(f"iiee {name}")
            values.append(relative_error(model[1], other[1]))
    lead = _lead_days(deterministic) if deterministic is not None else np.array([])
    return rows, lead, (np.vstack(values) if values else np.zeros((0, len(lead))))


def plot_scorecard(result, out_path: Path, reference: str = "persistence") -> Path:
    """Variables x lead times, coloured by error relative to a baseline.

    The one-glance "is my model good" panel.  **Blue is better** (less error than
    the baseline), red is worse, and the neutral midpoint is exactly "no
    different".  Every cell carries its own number, so the panel is readable in
    greyscale, is not gated on colour, and needs no separate legend.
    """
    apply_theme()
    rows, lead, matrix = scorecard_table(result, reference=reference)
    if not rows:
        raise ValueError("Nothing to score: no metrics for the model and the reference.")

    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    limit = max(limit, 0.05)
    fig, ax = plt.subplots(figsize=(3.2 + 0.55 * len(lead), 1.4 + 0.30 * len(rows)))
    mesh = ax.imshow(
        matrix,
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(lead)), [f"{d:g}" for d in lead])
    ax.set_yticks(range(len(rows)), [variable_label(r, with_units=False) for r in rows])
    ax.set_xlabel("Lead time [days]")
    ax.grid(False)
    ax.set_title(
        f"Error relative to {_label_for(result, reference).lower()}"
        "\nblue = the model has less error, red = more",
        color=INK,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            # Ink or white by the cell's own luminance, so a label inside a fill
            # always clears contrast.
            rgba = mesh.cmap(mesh.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                j,
                i,
                f"{100 * value:+.0f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=INK if luminance > 0.5 else "#ffffff",
            )
    bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.03, aspect=30)
    bar.set_label(f"Error relative to {reference} [%]", color=INK_SECONDARY)
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:+.0f}"))
    bar.outline.set_edgecolor(AXIS_RULE)
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# 3. Sea-ice diagnostics against lead time
# ---------------------------------------------------------------------------
def plot_seaice_vs_lead(result, out_path: Path) -> Path:
    """Ice-edge error and extent bias per hemisphere, in 10^6 km^2.

    Sea ice is not judged by grid-point RMSE.  IIEE is the total area where the
    forecast and the truth disagree about whether there is ice; extent bias is
    signed, so the zero line is the target and both signs are failures.
    """
    apply_theme()
    seaice = result.metrics.get("model", {}).get("glorys_seaice_metrics")
    if seaice is None:
        raise ValueError("This run has no sea-ice metrics.")
    # Northern hemisphere first, matching HEMISPHERES in seaice_metrics.py;
    # `data_vars` order is not guaranteed and came out south-first.
    from ..metrics.seaice_metrics import HEMISPHERES

    hemispheres = [h for h in HEMISPHERES if h in seaice.data_vars]
    hemispheres += [h for h in seaice.data_vars if h not in hemispheres]
    keys = _forecaster_order(result)
    fig, axes = plt.subplots(
        2, len(hemispheres), figsize=(3.8 * len(hemispheres), 5.4), squeeze=False
    )
    for column, hemisphere in enumerate(hemispheres):
        for row, (metric, title) in enumerate(
            (("iiee", "Integrated ice-edge error"), ("extentbias", "Sea-ice extent bias"))
        ):
            ax = axes[row][column]
            for key in keys:
                series = _series(result, key, metric, hemisphere)
                if series is None:
                    continue
                lead, values = series
                ax.plot(lead, values, **_style(key), zorder=3 if key == "model" else 2)
            if metric == "extentbias":
                # The metric is extent(forecast) - extent(truth), so the sign is
                # a physical statement and the reader has to be told which way
                # round it goes: an unlabelled signed quantity is a coin toss.
                ax.axhline(0.0, color=AXIS_RULE, linewidth=1.0, zorder=1)
                ax.set_ylabel("Extent bias, forecast - truth [10$^6$ km$^2$]")
                # Headroom for the note below, so it never sits on a curve.
                low, high = ax.get_ylim()
                ax.set_ylim(low, high + 0.16 * (high - low))
                ax.annotate(
                    "positive = too much ice",
                    xy=(0.02, 0.96),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                    fontsize=7.5,
                    color=INK_SECONDARY,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.6),
                    zorder=5,
                )
            else:
                ax.set_ylabel("Area [10$^6$ km$^2$]")
            ax.set_title(f"{title} -- {hemisphere.replace('seaice', '')}", color=INK)
            ax.set_xlabel("Lead time [days]")
            ax.set_xlim(left=0)
    fig.suptitle(
        "Sea-ice edge diagnostics on the 15% concentration contour",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    _legend(fig, keys, [_label_for(result, k) for k in keys])
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# Map machinery
# ---------------------------------------------------------------------------
def _projection(kind: str = "global"):
    import cartopy.crs as ccrs

    if kind == "north":
        return ccrs.NorthPolarStereo()
    if kind == "south":
        return ccrs.SouthPolarStereo()
    return ccrs.Robinson(central_longitude=180)


def _draw_field(ax, field: xr.DataArray, cmap, norm=None, vmin=None, vmax=None):
    """One pcolormesh in the data's own coordinates, land already NaN."""
    import cartopy.crs as ccrs

    mesh = ax.pcolormesh(
        field["lon"].to_numpy(),
        field["lat"].to_numpy(),
        np.ma.masked_invalid(field.to_numpy()),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        shading="auto",
        rasterized=True,
    )
    # Land is NaN in the cached fields and transparent in the map colormaps, so
    # what shows through is this: a neutral grey that is not a step of any data
    # ramp and can never be read as a value.
    ax.set_facecolor(LAND)
    ax.gridlines(linewidth=0.5, color=GRIDLINE, alpha=0.8)
    return mesh


def _symmetric_limit(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), 99.0))
    return limit if limit > 0 else 1.0


def _select(dataset: xr.Dataset, variable: str, lead_day: float, init_index: int) -> xr.DataArray:
    """One 2-D field. For a 3-D variable this is the SHALLOWEST level, not 0 m.

    GLORYS' first level is centred at 0.494 m, so a map of ``thetao`` here is a
    map of the top half-metre. :func:`_depth_note` turns that into words for the
    figure, because "sea surface temperature" and "temperature at 0.49 m" are not
    quite the same claim.
    """
    field = dataset[variable].isel(time=init_index)
    field = field.sel(prediction_timedelta=np.timedelta64(int(round(lead_day * 24)), "h"))
    if "depth" in field.dims:
        field = field.isel(depth=0)
    return field


def _depth_note(field: xr.DataArray) -> str:
    """`` at 0.49 m`` for a level field, empty for a genuinely 2-D one."""
    depth = field.coords.get("depth")
    if depth is None or depth.size != 1:
        return ""
    return f" at {float(depth):.2f} m"


def plot_map_triptych(
    result,
    variable: str,
    out_path: Path,
    lead_days: Sequence[float] | None = None,
    init_index: int = 0,
) -> Path:
    """Truth | prediction | error, one row per lead time.

    Truth and prediction share a colour scale, so the two panels are directly
    comparable; the error panel is diverging and centred on zero, so a positive
    and a negative error of the same size are equally loud.  Land is grey in
    every panel.
    """
    apply_theme()
    predictions = result.open_predictions()
    targets = result.open_targets()
    available = _lead_days(predictions)
    if lead_days is None:
        picks = sorted({available[0], available[len(available) // 2], available[-1]})
    else:
        picks = [d for d in lead_days if d in set(available)]
    if not picks:
        picks = [available[-1]]

    units = variable_units(variable) or "1"
    projection = _projection("global")
    fig, axes = plt.subplots(
        len(picks),
        3,
        figsize=(12.0, 2.7 * len(picks) + 0.9),
        subplot_kw={"projection": projection},
        squeeze=False,
    )

    truths = [_select(targets, variable, d, init_index) for d in picks]
    preds = [_select(predictions, variable, d, init_index) for d in picks]
    stacked = np.concatenate([t.to_numpy().ravel() for t in truths])
    finite = stacked[np.isfinite(stacked)]
    if _is_diverging_variable(variable):
        limit = _symmetric_limit(finite)
        field_cmap, vmin, vmax = DIVERGING_MAP, -limit, limit
    else:
        field_cmap = SEQUENTIAL_MAP
        vmin = float(np.nanpercentile(finite, 1)) if finite.size else 0.0
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    error_limit = _symmetric_limit(
        np.concatenate([(p - t).to_numpy().ravel() for p, t in zip(preds, truths)])
    )

    field_mesh = error_mesh = None
    for row, day in enumerate(picks):
        truth, prediction = truths[row], preds[row]
        field_mesh = _draw_field(axes[row][0], truth, field_cmap, vmin=vmin, vmax=vmax)
        _draw_field(axes[row][1], prediction, field_cmap, vmin=vmin, vmax=vmax)
        error_mesh = _draw_field(
            axes[row][2],
            prediction - truth,
            DIVERGING_MAP,
            norm=TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit),
        )
        for column, title in enumerate(
            ("GLORYS truth", "Model prediction", "Error (model - truth)")
        ):
            axes[row][column].set_title(
                f"{title}  --  day {day:g}" if row == 0 else f"day {day:g}", color=INK, fontsize=9
            )
        axes[row][0].set_global()
        axes[row][1].set_global()
        axes[row][2].set_global()

    # `extend` because vmin/vmax are the 1st and 99th percentile of the truth,
    # so ~2% of the ocean is outside the scale and is drawn at the end colour.
    # Without the arrows a saturated cell reads as one exactly at the limit.
    bar = fig.colorbar(
        field_mesh,
        ax=axes[:, :2].ravel().tolist(),
        shrink=0.75,
        pad=0.015,
        fraction=0.02,
        extend="both",
    )
    bar.set_label(f"{variable_label(variable, with_units=False)} [{units}]", color=INK_SECONDARY)
    bar.outline.set_edgecolor(AXIS_RULE)
    bar_error = fig.colorbar(
        error_mesh,
        ax=axes[:, 2].ravel().tolist(),
        shrink=0.75,
        pad=0.015,
        fraction=0.04,
        extend="both",
    )
    bar_error.set_label(f"Error [{units}]", color=INK_SECONDARY)
    bar_error.outline.set_edgecolor(AXIS_RULE)

    stamp = str(np.datetime64(targets["time"].to_numpy()[init_index], "D"))
    fig.suptitle(
        f"{variable_label(variable, with_units=False)}{_depth_note(truths[0])}"
        f" -- forecast from {stamp}"
        "\nGrey is land: no data, not a value.",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# 4. Depth / lead-time Hovmoller
# ---------------------------------------------------------------------------
def depth_matrix(result, key: str, name: str, metric: str = "rmse"):
    """``(depths, lead days, values[depth, lead])`` for one level variable."""
    group = result.metrics.get(key, {}).get("glorys_deterministic_metrics")
    if group is None:
        return None
    labels = [
        (parse_metric_variable(v)[1], v)
        for v in group.data_vars
        if parse_metric_variable(v)[0] == name and parse_metric_variable(v)[1] is not None
    ]
    if not labels:
        return None
    labels.sort()
    values = np.vstack([group[v].sel(metric=metric).to_numpy() for _, v in labels])
    # The metric label rounds the depth to whole metres, so the shallowest level
    # is written "0m" -- and a 0 on a log depth axis is not plottable. Snap each
    # rounded label back onto the prepared depth it came from.
    prepared = np.asarray(V.PREPPED_DEPTHS, dtype=float)
    depths = np.array(
        [float(prepared[np.argmin(np.abs(prepared - d))]) for d, _ in labels], dtype=float
    )
    return depths, _lead_days(group), values


def plot_depth_hovmoller(result, out_path: Path, names: Sequence[str] = ("thetao", "so")) -> Path:
    """Error and relative error against depth and lead time.

    The top row is the raw RMSE on a log colour scale, because the error at
    1684 m is three orders of magnitude smaller than at the surface and one
    linear scale would render the whole deep ocean as a single colour.  The
    bottom row is the error *relative to persistence*, which is dimensionless and
    therefore comparable across depths -- and it is the row that answers the real
    question: **is the model only good at the surface?**  Blue is better than
    persistence, red is worse, on the same convention as the scorecard.

    Depth increases downwards, as in every ocean section.
    """
    apply_theme()
    panels = [(name, depth_matrix(result, "model", name)) for name in names]
    panels = [(name, data) for name, data in panels if data is not None]
    if not panels:
        raise ValueError("No level variables in this run, so there is no depth structure to plot.")

    fig, axes = plt.subplots(
        2, len(panels), figsize=(4.8 * len(panels), 7.2), squeeze=False, sharey=True
    )
    for column, (name, (depths, lead, values)) in enumerate(panels):
        units = V.VARIABLES[name].units
        positive = values[np.isfinite(values) & (values > 0)]
        norm = (
            LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
            if positive.size
            else None
        )
        ax = axes[0][column]
        mesh = ax.pcolormesh(lead, depths, values, cmap=SEQUENTIAL, norm=norm, shading="nearest")
        ax.set_title(f"RMSE -- {V.VARIABLES[name].long_name}", color=INK)
        bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
        bar.set_label(
            f"RMSE [{units}]" + (", log scale" if norm is not None else ""),
            color=INK_SECONDARY,
        )
        bar.outline.set_edgecolor(AXIS_RULE)

        reference = depth_matrix(result, "persistence", name)
        ax = axes[1][column]
        if reference is not None:
            relative = relative_error(values, reference[2])
            limit = max(
                float(np.nanmax(np.abs(relative))) if np.isfinite(relative).any() else 0.1, 0.05
            )
            mesh = ax.pcolormesh(
                lead,
                depths,
                relative,
                cmap=DIVERGING,
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                shading="nearest",
            )
            bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
            bar.set_label("Error relative to persistence [%]", color=INK_SECONDARY)
            bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:+.0f}"))
            bar.outline.set_edgecolor(AXIS_RULE)
            ax.set_title(f"Error relative to persistence -- {name}  (blue = better)", color=INK)
        else:
            ax.set_title(f"No persistence baseline for {name}", color=INK)
        for row in (0, 1):
            axes[row][column].set_yscale("log")
            axes[row][column].set_xlabel("Lead time [days]")
            # Both scales here are logarithmic and an unlabelled log axis is a
            # way to make a small difference look large, so both say so.
            axes[row][column].set_ylabel("Depth [m], log scale")
            axes[row][column].grid(False)

    # Once, at the end: the y axis is shared, so inverting it inside the loop
    # would toggle it back and leave the deep ocean at the top.
    axes[0][0].set_ylim(float(max(panels[0][1][0])) * 1.15, float(min(panels[0][1][0])) * 0.85)

    fig.suptitle(
        "Where in the water column is the model good?",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# 5. Polar sea-ice maps
# ---------------------------------------------------------------------------
def plot_seaice_polar(
    result,
    out_path: Path,
    lead_day: float | None = None,
    init_index: int = 0,
    threshold: float = ICE_EDGE_THRESHOLD,
) -> Path:
    """Arctic and Antarctic concentration with both 15% ice edges drawn on top.

    The contour is the number the sea-ice community argues about: where the model
    puts the ice edge against where it actually was.  The two edges are drawn on
    every panel so the disagreement is visible without flicking between images.
    """
    import cartopy.crs as ccrs

    apply_theme()
    predictions = result.open_predictions()
    targets = result.open_targets()
    if "siconc" not in predictions:
        raise ValueError("This run does not predict siconc.")
    available = _lead_days(predictions)
    day = available[-1] if lead_day is None else lead_day

    truth = _select(targets, "siconc", day, init_index)
    prediction = _select(predictions, "siconc", day, init_index)
    lon, lat = truth["lon"].to_numpy(), truth["lat"].to_numpy()
    # Ice never lives on land, and the contour routine needs a number, not NaN.
    truth_values = np.nan_to_num(truth.to_numpy(), nan=0.0)
    prediction_values = np.nan_to_num(prediction.to_numpy(), nan=0.0)

    hemispheres = (("north", "Arctic", (50, 90)), ("south", "Antarctic", (-90, -50)))
    fig = plt.figure(figsize=(11.5, 8.0))
    error_limit = _symmetric_limit((prediction - truth).to_numpy())
    field_mesh = error_mesh = None
    for row, (kind, title, extent) in enumerate(hemispheres):
        projection = _projection(kind)
        for column, (field, panel) in enumerate(
            (
                (truth, "GLORYS truth"),
                (prediction, "Model prediction"),
                (prediction - truth, "Error"),
            )
        ):
            ax = fig.add_subplot(2, 3, row * 3 + column + 1, projection=projection)
            ax.set_extent([-180, 180, extent[0], extent[1]], ccrs.PlateCarree())
            if panel == "Error":
                error_mesh = _draw_field(
                    ax,
                    field,
                    DIVERGING_MAP,
                    norm=TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit),
                )
            else:
                mesh = _draw_field(ax, field, SEQUENTIAL_MAP, vmin=0.0, vmax=1.0)
                field_mesh = field_mesh or mesh
            for values, colour, style in (
                (truth_values, INK, "-"),
                (prediction_values, SERIES_COLOURS["model"], "--"),
            ):
                ax.contour(
                    lon,
                    lat,
                    values,
                    levels=[threshold],
                    colors=[colour],
                    linewidths=1.4,
                    linestyles=[style],
                    transform=ccrs.PlateCarree(),
                )
            ax.set_title(f"{title} -- {panel}", color=INK, fontsize=9)

    bar = fig.colorbar(field_mesh, ax=fig.axes[:2] + fig.axes[3:5], shrink=0.7, pad=0.02)
    bar.set_label("Sea-ice concentration [1]", color=INK_SECONDARY)
    bar.outline.set_edgecolor(AXIS_RULE)
    bar_error = fig.colorbar(error_mesh, ax=[fig.axes[2], fig.axes[5]], shrink=0.7, pad=0.02)
    bar_error.set_label("Concentration error [1]", color=INK_SECONDARY)
    bar_error.outline.set_edgecolor(AXIS_RULE)
    fig.legend(
        handles=[
            Line2D([], [], color=INK, linestyle="-", linewidth=1.4),
            Line2D([], [], color=SERIES_COLOURS["model"], linestyle="--", linewidth=1.4),
        ],
        labels=[
            f"Observed {threshold:.0%} ice edge",
            f"Predicted {threshold:.0%} ice edge",
        ],
        loc="outside lower center",
        ncol=2,
        labelcolor=INK_SECONDARY,
    )
    stamp = str(np.datetime64(targets["time"].to_numpy()[init_index], "D"))
    fig.suptitle(
        f"Sea-ice concentration at day {day:g}, forecast from {stamp}"
        "\nGrey is land: no data, not a value.",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# 6. Free-running time series -- the drift check
# ---------------------------------------------------------------------------
def _monthly_climatology(variable: str, depth: float | None) -> np.ndarray:
    """``(12, lat, lon)`` of the shipped monthly climatology, land NaN."""
    from .. import paths

    with xr.open_dataset(paths.climatology_file()) as dataset:
        if variable not in dataset:
            raise KeyError(f"The climatology file holds no {variable!r}.")
        field = dataset[variable]
        if "depth" in field.dims:
            field = (
                field.isel(depth=0) if depth is None else field.sel(depth=depth, method="nearest")
            )
        return field.to_numpy()


def climatology_trajectory(variable: str, valid_times, depth: float | None = None) -> np.ndarray:
    """The monthly climatology interpolated to each valid day: ``(lead, lat, lon)``.

    Uses :func:`~oceanarches.metrics.masked_metrics.month_interpolation_weights`
    -- the same periodic interpolation between month midpoints that the ACC
    metric and the climatology *baseline* use -- so the curve drawn on the drift
    figure is the identical field the model is scored against.  Anything else
    would put two different climatologies in one report.
    """
    from ..metrics.masked_metrics import month_interpolation_weights

    monthly = _monthly_climatology(variable, depth)
    out = np.empty((len(valid_times),) + monthly.shape[1:], dtype=float)
    for step, valid in enumerate(valid_times):
        before, after, alpha = month_interpolation_weights(valid)
        out[step] = (1.0 - alpha) * monthly[before] + alpha * monthly[after]
    return out


def free_running_series(free, init_index: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Global-mean SST and per-hemisphere ice extent along a free rollout.

    Computed from the cached fields, with the same latitude weighting and the
    same cell areas the metrics use, so the curves and the scores are the same
    quantity.  Each quantity carries three series -- ``model``, ``truth`` and,
    when the climatology file is readable, ``climatology`` -- because a curve
    that walks away from the truth *towards* the climatology is a forecast losing
    information, while one that walks away from both is a model leaving the
    attractor, and a beginner cannot tell those apart without the third line.
    """
    from ..metrics.masked_metrics import compute_lat_weights_glorys
    from ..metrics.seaice_metrics import cell_areas, hemisphere_masks

    predictions = free.open_predictions()
    targets = free.open_targets()
    days = _lead_days(predictions)
    lat = predictions["lat"].to_numpy()
    weights = compute_lat_weights_glorys(len(lat)).numpy()[:, 0][:, None]
    areas = cell_areas(len(lat), len(predictions["lon"])).numpy() / 1e6
    hemispheres = hemisphere_masks(len(lat), len(predictions["lon"])).numpy()

    def surface(dataset, variable):
        """``((lead, lat, lon) values, depth of the level taken)``."""
        field = dataset[variable].isel(time=init_index)
        depth = None
        if "depth" in field.dims:
            if "depth" in field.coords:
                depth = float(np.atleast_1d(field["depth"].to_numpy())[0])
            field = field.isel(depth=0)
        return field.to_numpy(), depth

    def mean_over_ocean(values):
        mask = np.isfinite(values)
        weighted = np.where(mask, values, 0.0) * weights
        return weighted.sum(axis=(-2, -1)) / (mask * weights).sum(axis=(-2, -1))

    def ice_extent(values, hemisphere_index):
        indicator = (np.nan_to_num(values, nan=0.0) > ICE_EDGE_THRESHOLD).astype(float)
        return (indicator * areas * hemispheres[hemisphere_index]).sum(axis=(-2, -1))

    model_sst, depth = surface(predictions, "thetao")
    truth_sst, _ = surface(targets, "thetao")
    model_ice, _ = surface(predictions, "siconc")
    truth_ice, _ = surface(targets, "siconc")

    out = {"days": days}
    out["sst"] = {"model": mean_over_ocean(model_sst), "truth": mean_over_ocean(truth_sst)}
    out["extent_nh"] = {"model": ice_extent(model_ice, 0), "truth": ice_extent(truth_ice, 0)}
    out["extent_sh"] = {"model": ice_extent(model_ice, 1), "truth": ice_extent(truth_ice, 1)}

    # -- the climatology, on the same axes -----------------------------------
    try:
        init = targets["time"].to_numpy()[init_index]
        valid = [
            np.datetime64(init + step, "s") for step in targets["prediction_timedelta"].to_numpy()
        ]
        # Masked with the truth's own land mask, so all three curves average over
        # exactly the same ocean cells and a difference between them is physics.
        land = ~np.isfinite(truth_sst)
        clim_sst = np.where(land, np.nan, climatology_trajectory("thetao", valid, depth))
        clim_ice = np.where(
            ~np.isfinite(truth_ice), np.nan, climatology_trajectory("siconc", valid)
        )
    except Exception as error:  # noqa: BLE001 - the drift figure is worth more than the third curve
        warnings.warn(
            f"The climatology curve was left off the drift figure: "
            f"{type(error).__name__}: {error}",
            stacklevel=2,
        )
    else:
        out["sst"]["climatology"] = mean_over_ocean(clim_sst)
        out["extent_nh"]["climatology"] = ice_extent(clim_ice, 0)
        out["extent_sh"]["climatology"] = ice_extent(clim_ice, 1)
    return out


def plot_free_timeseries(free, out_path: Path, init_index: int = 0) -> Path:
    """Global-mean SST and hemispheric ice extent over a long free rollout.

    This is the drift check, and it is what exposes a model that has learned to
    blur: a blurred forecast keeps a respectable RMSE while the global mean walks
    away from the truth and the ice edge dissolves.
    """
    apply_theme()
    series = free_running_series(free, init_index=init_index)
    days = series["days"]
    stamp = np.datetime64(free.open_targets()["time"].to_numpy()[init_index], "D")

    panels = (
        ("sst", "Global-mean sea surface temperature", f"SST [{V.VARIABLES['thetao'].units}]"),
        ("extent_nh", "Northern-hemisphere sea-ice extent", "Extent [10$^6$ km$^2$]"),
        ("extent_sh", "Southern-hemisphere sea-ice extent", "Extent [10$^6$ km$^2$]"),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.0, 8.4), sharex=True)
    roles = [("truth", "GLORYS truth"), ("model", "Model")]
    if "climatology" in series["sst"]:
        roles.append(("climatology", "Climatology"))
    for ax, (key, title, ylabel) in zip(axes, panels):
        for role, label in roles:
            values = series[key].get(role)
            if values is None:
                continue
            ax.plot(
                days,
                values,
                color=SERIES_COLOURS[role],
                linestyle=SERIES_DASHES[role],
                zorder=3 if role == "model" else 2,
            )
            # Well-separated series: direct end-labels work here, which is why
            # this is the one figure that carries them.
            _end_label(ax, days, values, label)
        ax.set_title(title, color=INK)
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
    axes[-1].set_xlabel("Lead time [days]")
    legend_labels = ["GLORYS truth", "Model, free running"]
    if any(role == "climatology" for role, _ in roles):
        legend_labels.append("Climatology (a forecast with no skill)")
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=SERIES_COLOURS[role],
                linestyle=SERIES_DASHES[role],
                linewidth=2.0,
            )
            for role, _ in roles
        ],
        labels=legend_labels,
        loc="outside lower center",
        ncol=len(legend_labels),
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        f"Free-running rollout from {stamp}: does the model drift?",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# 7. Power spectra
# ---------------------------------------------------------------------------
def spherical_power_spectrum(field: np.ndarray) -> np.ndarray | None:
    """Spherical-harmonic power per degree of a ``(lat, lon)`` field, or None.

    Reuses geoarches' ``PowerSpectrum`` machinery (``pyshtools``) rather than
    reimplementing it, but **not** ``Era5PowerSpectrum``: that wrapper's
    preprocess drops the southernmost latitude row, which is right for ERA5's 121
    rows (120 x 240 satisfies ``nlon == 2 * nlat``) and wrong for our 180 (179 x
    360 does not).  Our grid already satisfies the requirement, so no row is
    dropped.

    ``pyshtools`` places the first row at the north pole while ours is a
    south-first cell-centred grid.  A north-south flip does not change power by
    degree, and the half-cell offset is a small approximation; this is a
    diagnostic of effective resolution, not a spectral analysis, so it is stated
    rather than corrected.
    """
    try:
        import pyshtools as pysh
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    n_lat, n_lon = field.shape[-2:]
    if n_lon != 2 * n_lat:
        warnings.warn(
            f"The power spectrum needs nlon == 2 * nlat; this grid is {n_lat} x {n_lon}. "
            "Skipping.",
            stacklevel=2,
        )
        return None
    return np.asarray(pysh.SHGrid.from_array(np.asarray(field, dtype=float)).expand().spectrum())


def _fill_land_with_ocean_mean(field: xr.DataArray) -> np.ndarray:
    """Land -> the field's own ocean mean, so the coastline adds no step change.

    Truth and prediction share one land mask, so both get the identical
    treatment and the difference between their spectra is still the model's.
    """
    values = field.to_numpy().astype(float)
    ocean = np.isfinite(values)
    if not ocean.any():
        return np.zeros_like(values)
    return np.where(ocean, values, values[ocean].mean())


def plot_power_spectra(
    result,
    out_path: Path,
    variables: Sequence[str] = ("thetao", "zos", "siconc"),
    init_index: int = 0,
) -> Path | None:
    """Truth against prediction, power by spherical-harmonic degree.

    A model that has learned to blur loses power at high degree -- small scales --
    while keeping a respectable RMSE, because a smooth field is a safe bet.  The
    top row shows both spectra at the longest lead; on a log-log axis spanning
    four decades a 20% loss of power is invisible there, so the **bottom row
    plots the ratio**, prediction over truth, where 1.0 is "the right amount of
    structure" and anything below it is missing variance.  Day 1 and the longest
    lead are drawn as two steps of one hue, because they are the same model at
    two lead times rather than two different things.

    Returns None (with a warning) when ``pyshtools`` is not importable, rather
    than failing the run.
    """
    apply_theme()
    predictions = result.open_predictions()
    targets = result.open_targets()
    available = _lead_days(predictions)
    picks = sorted({available[0], available[-1]})
    variables = [v for v in variables if v in predictions]
    if not variables:
        return None

    probe = spherical_power_spectrum(
        _fill_land_with_ocean_mean(_select(targets, variables[0], picks[0], init_index))
    )
    if probe is None:
        warnings.warn(
            "pyshtools is not available (or the grid is unsuitable), so the power "
            "spectra were skipped. Everything else in the evaluation is unaffected.",
            stacklevel=2,
        )
        return None

    fig, axes = plt.subplots(2, len(variables), figsize=(4.2 * len(variables), 6.6), squeeze=False)
    for column, variable in enumerate(variables):
        spectra = {}
        for day in picks:
            spectra[("truth", day)] = spherical_power_spectrum(
                _fill_land_with_ocean_mean(_select(targets, variable, day, init_index))
            )
            spectra[("model", day)] = spherical_power_spectrum(
                _fill_land_with_ocean_mean(_select(predictions, variable, day, init_index))
            )
        degree = np.arange(len(spectra[("truth", picks[-1])]))

        ax = axes[0][column]
        ax.plot(
            degree[1:],
            spectra[("truth", picks[-1])][1:],
            color=SERIES_COLOURS["truth"],
            linewidth=1.8,
        )
        ax.plot(
            degree[1:],
            spectra[("model", picks[-1])][1:],
            color=SERIES_COLOURS["model"],
            linewidth=1.8,
            linestyle="--",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"{variable_label(variable, with_units=False)} -- day {picks[-1]:g}", color=INK
        )
        ax.set_xlabel("Spherical harmonic degree $\\ell$ [-]")
        ax.set_ylabel(f"Power [({variable_units(variable) or '1'})$^2$]")

        ax = axes[1][column]
        for step, day in zip(ORDINAL_BLUE, picks):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = spectra[("model", day)] / spectra[("truth", day)]
            ax.plot(degree[1:], ratio[1:], color=step, linewidth=1.8, label=f"day {day:g}")
        ax.axhline(1.0, color=AXIS_RULE, linewidth=1.0)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.6)
        ax.set_title(f"{variable} -- power ratio, model / truth", color=INK)
        ax.set_xlabel("Spherical harmonic degree $\\ell$ [-]")
        ax.set_ylabel("Power ratio [-]")
        ax.legend(loc="lower left", labelcolor=INK_SECONDARY)

    fig.legend(
        handles=[
            Line2D([], [], color=SERIES_COLOURS["truth"], linestyle="-", linewidth=2.0),
            Line2D([], [], color=SERIES_COLOURS["model"], linestyle="--", linewidth=2.0),
        ],
        labels=["GLORYS truth", "Model prediction"],
        loc="outside lower center",
        ncol=2,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "Power spectrum: a model that blurs loses the small scales",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# 8. Coupled vs uncoupled overlay -- the hook Task 8 calls
# ---------------------------------------------------------------------------
def plot_overlay(
    results: Mapping[str, object],
    out_path: Path,
    variables: Sequence[str] | None = None,
    metric: str = "rmse",
    baselines_from: str | None = None,
    title: str = "Coupled against uncoupled",
) -> Path:
    """Two or more result sets on one axes, plus the baselines drawn once.

    The hook Task 8 uses to put a coupled rollout beside an uncoupled one.  Each
    result set's *model* curve gets its own identity colour (blue, then violet,
    then magenta -- :data:`MODEL_ROLES`); the baselines are taken from one result
    set only and drawn once, because persistence and climatology do not depend on
    which model produced the forecast and drawing them twice would imply they do.

    Raises:
        ValueError: with more than ``len(MODEL_ROLES)`` result sets.  There is no
            fourth identity slot, and silently giving the fourth curve the first
            one's colour and dash -- which is what this used to do -- produces a
            figure whose legend is wrong.

    Args:
        results: ``{legend label: RolloutResult}``, in the order to draw them.
        out_path: where to write the PNG.
        variables: metric labels to panel over; defaults to the headline set of
            the first result.
        metric: which metric to plot (``rmse``, ``mae``, ``acc``, ``iiee``, ...).
        baselines_from: which entry supplies the baseline curves.  Defaults to
            the first.
        title: figure title.
    """
    apply_theme()
    items = list(results.items())
    if not items:
        raise ValueError("plot_overlay needs at least one result set.")
    if len(items) > len(MODEL_ROLES):
        raise ValueError(
            f"plot_overlay was given {len(items)} result sets but only {len(MODEL_ROLES)} "
            f"model identity slots exist ({', '.join(MODEL_ROLES)}). A further curve would "
            "be drawn in the same colour and dash as the first one under a different "
            "legend label. Split the comparison across two figures."
        )
    baseline_key = baselines_from or items[0][0]
    variables = list(variables or _resolve_headline(items[0][1]))
    model_roles = list(MODEL_ROLES[: len(items)])

    n = len(variables)
    ncols = min(3, n) or 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows), squeeze=False)

    legend_keys, legend_labels = [], []
    for index, variable in enumerate(variables):
        ax = axes[index // ncols][index % ncols]
        for position, (label, result) in enumerate(items):
            role = model_roles[position]
            series = _series(result, "model", metric, variable)
            if series is None:
                continue
            lead, values = series
            ax.plot(lead, values, **_style(role), zorder=4)
            if index == 0:
                legend_keys.append(role)
                legend_labels.append(label)
        for baseline in ("persistence", "climatology"):
            series = _series(results[baseline_key], baseline, metric, variable)
            if series is None:
                continue
            lead, values = series
            ax.plot(lead, values, **_style(baseline), zorder=2)
            if index == 0:
                legend_keys.append(baseline)
                legend_labels.append(_label_for(results[baseline_key], baseline))
        ax.set_title(variable_label(variable, with_units=False), color=INK)
        ax.set_xlabel("Lead time [days]")
        ax.set_ylabel(f"{metric.upper()} [{variable_units(variable) or '1'}]")
        ax.set_xlim(left=0)
    for index in range(n, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    fig.suptitle(title, fontsize=12, color=INK, fontweight="bold")
    _legend(fig, legend_keys, legend_labels)
    fig.savefig(out_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
#: Variables that get a map triptych.  Surface tracers a participant recognises.
MAP_VARIABLES = ("thetao", "siconc", "zos")


def render_all(
    result,
    free=None,
    out_dir: Path = Path("figures"),
    dpi: int = 150,
    skip_spectra: bool = False,
    progress: bool = True,
) -> list[Path]:
    """Render every figure that this result supports; skip the rest with a warning.

    A missing figure never fails the run -- a component without sea ice, a run
    without cached fields, an environment without ``pyshtools`` -- because a
    participant staring at a traceback learns nothing about their model.

    Args:
        progress: print each figure as it lands.  On by default because this
            stage and the animations are most of ``make eval``'s wall clock and
            used to print nothing at all: a participant killed a run at 180 s
            believing it had hung, when it needed 257 s.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["savefig.dpi"] = dpi
    written: list[Path] = []

    def attempt(name: str, function, *args, **kwargs) -> None:
        started = time.time()
        try:
            path = function(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - one bad figure must not stop the rest
            warnings.warn(
                f"Figure {name!r} was skipped: {type(error).__name__}: {error}", stacklevel=2
            )
            return
        if path is not None:
            written.append(Path(path))
            if progress:
                print(
                    f"  figure {Path(path).name} ({time.time() - started:.1f}s)",
                    flush=True,
                )

    def has_fields(name: str, source) -> bool:
        """True if ``source`` has cached fields; warn and return False if not.

        Every figure that draws a map, a polar plot or a spectrum reads
        ``predictions.zarr``, which ``--skip-fields`` does not write.  Skipping
        those *silently* is how a run comes back with 4 figures where the last
        one had 10 and nobody notices: the count is not on screen and the
        missing figures are exactly the ones a participant looks at first.
        """
        if source is not None and source.predictions_path.exists():
            return True
        where = "" if source is None else f" ({source.predictions_path} is not there)"
        warnings.warn(
            f"Figure {name!r} needs the cached fields and there are none{where}. "
            "It was written by a run with --skip-fields, or not written at all; "
            "re-run without --skip-fields, adding --force if a cache is being reused.",
            stacklevel=2,
        )
        return False

    attempt("rmse_vs_lead", plot_rmse_vs_lead, result, out_dir / "01_rmse_vs_lead.png")
    attempt("scorecard", plot_scorecard, result, out_dir / "02_scorecard.png")
    attempt("seaice_vs_lead", plot_seaice_vs_lead, result, out_dir / "03_seaice_vs_lead.png")
    for variable in MAP_VARIABLES:
        if has_fields(f"map_{variable}", result):
            attempt(
                f"map_{variable}",
                plot_map_triptych,
                result,
                variable,
                out_dir / f"04_map_{variable}.png",
            )
    attempt("depth_hovmoller", plot_depth_hovmoller, result, out_dir / "05_depth_hovmoller.png")
    if has_fields("seaice_polar", result):
        attempt("seaice_polar", plot_seaice_polar, result, out_dir / "06_seaice_polar.png")
    if free is None:
        warnings.warn(
            "Figure 'free_timeseries' was skipped: there is no free-running rollout. "
            "It is the drift check; drop --skip-free-rollout to get it.",
            stacklevel=2,
        )
    elif has_fields("free_timeseries", free):
        attempt("free_timeseries", plot_free_timeseries, free, out_dir / "07_free_timeseries.png")
    if not skip_spectra and has_fields("power_spectra", result):
        attempt("power_spectra", plot_power_spectra, result, out_dir / "08_power_spectra.png")
    return written
