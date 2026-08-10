"""``report.md`` and a self-contained ``report.html``.

The report is the artefact a participant actually reads, so it is built around
one question -- *is my model any good?* -- and it answers it in the first table,
before any figure.

``report.html`` embeds every figure and every animation as a ``data:`` URI, so
the single file can be scp'd off the cluster, mailed, or opened from a USB stick
and still work.  That costs about a third more bytes than linking (base64 is 4/3)
and it is worth it: a report whose images 404 the moment it leaves the directory
it was written in is not a report.

Both files are generated from one list of blocks, so the Markdown and the HTML
cannot drift apart.  The tables are not decoration either -- they are the
*table view* that makes every figure readable without relying on colour, which
is what lets the figure set use a hue that sits below the 3:1 contrast target.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import plots
from .provenance import StatisticsProvenance

__all__ = ["write_report", "headline_table", "forecast_horizon", "Block", "Report"]

#: Embed media up to this many bytes in total; past it, link relatively and say
#: so, rather than writing a 200 MB HTML file nobody can open.
DEFAULT_EMBED_BUDGET = 32 * 1024 * 1024


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """One piece of the report, renderable to both Markdown and HTML."""

    kind: str  # "heading" | "text" | "table" | "figure" | "video" | "code"
    text: str = ""
    level: int = 2
    header: Sequence[str] = field(default_factory=list)
    rows: Sequence[Sequence[str]] = field(default_factory=list)
    path: Path | None = None
    caption: str = ""


@dataclass
class Report:
    title: str
    blocks: list[Block] = field(default_factory=list)

    def heading(self, text: str, level: int = 2) -> None:
        self.blocks.append(Block("heading", text=text, level=level))

    def text(self, text: str) -> None:
        self.blocks.append(Block("text", text=text))

    def code(self, text: str) -> None:
        self.blocks.append(Block("code", text=text))

    def table(
        self, header: Sequence[str], rows: Sequence[Sequence[str]], caption: str = ""
    ) -> None:
        self.blocks.append(
            Block("table", header=list(header), rows=[list(r) for r in rows], caption=caption)
        )

    def figure(self, path: Path, caption: str = "") -> None:
        self.blocks.append(Block("figure", path=Path(path), caption=caption))

    def video(self, path: Path, caption: str = "") -> None:
        self.blocks.append(Block("video", path=Path(path), caption=caption))


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def _format(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1000 or (magnitude < 1e-3 and magnitude > 0):
        return f"{value:.4g}"
    if magnitude >= 10:
        return f"{value:.3f}"
    return f"{value:.5g}"


def headline_table(
    result, day: float, variables: Sequence[str] | None = None, metric: str = "rmse"
) -> tuple[list[str], list[list[str]]]:
    """Model against both baselines at one lead time.

    This is the table the whole pipeline exists to produce: a number, the two
    numbers it has to be read against, and the percentage between them.
    """
    variables = list(variables or plots._resolve_headline(result))
    header = [
        "Variable",
        "Unit",
        "Model",
        "Persistence",
        "Climatology",
        "Model vs persistence",
    ]
    rows = []
    for variable in variables:
        values = {}
        for key in ("model", "persistence", "climatology"):
            series = plots._series(result, key, metric, variable)
            if series is None:
                values[key] = None
                continue
            lead, data = series
            index = int(np.argmin(np.abs(lead - day)))
            values[key] = float(data[index])
        if values["model"] is None:
            continue
        if values["persistence"]:
            relative = 100.0 * (values["model"] / values["persistence"] - 1.0)
            verdict = f"{relative:+.1f}%" + ("  (better)" if relative < 0 else "  (WORSE)")
        else:
            verdict = "n/a"
        rows.append(
            [
                plots.variable_label(variable, with_units=False),
                plots.variable_units(variable) or "1",
                _format(values["model"]),
                _format(values["persistence"]),
                _format(values["climatology"]),
                verdict,
            ]
        )
    return header, rows


def forecast_horizon(result, variable: str, metric: str = "rmse") -> str:
    """First lead time at which the model is no better than climatology.

    Past that point the forecast carries no more information than "it is March",
    which is the honest end of its useful range.  Returns a string because the
    answer is often "longer than we rolled out".
    """
    model = plots._series(result, "model", metric, variable)
    climatology = plots._series(result, "climatology", metric, variable)
    if model is None or climatology is None:
        return "n/a"
    lead, model_values = model
    _, climatology_values = climatology
    worse = np.where(model_values >= climatology_values)[0]
    if worse.size == 0:
        return f"> {lead[-1]:g} days"
    return f"{lead[worse[0]]:g} days"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _markdown(report: Report, media_root: Path) -> str:
    lines = [f"# {report.title}", ""]
    for block in report.blocks:
        if block.kind == "heading":
            lines += ["#" * block.level + f" {block.text}", ""]
        elif block.kind == "text":
            lines += [block.text, ""]
        elif block.kind == "code":
            lines += ["```", block.text, "```", ""]
        elif block.kind == "table":
            if block.caption:
                lines += [f"**{block.caption}**", ""]
            lines.append("| " + " | ".join(block.header) + " |")
            lines.append("|" + "|".join("---" for _ in block.header) + "|")
            for row in block.rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("")
        elif block.kind in ("figure", "video"):
            relative = _relative(block.path, media_root)
            # An mp4 in an image tag is a broken image in every markdown viewer
            # there is; a link to it plays. A GIF fallback really is an image.
            if block.kind == "figure" or _is_image(Path(block.path)):
                lines += [f"![{block.caption}]({relative})", "", f"*{block.caption}*", ""]
            else:
                lines += [f"[{block.caption}]({relative})", "", f"*{block.caption}*", ""]
    return "\n".join(lines)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


_CSS = """
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --rule: #c3c2b7; --accent: #2a78d6;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.25rem 5rem; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.6;
}
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.9rem; margin: 0 0 .35rem; }
h2 { font-size: 1.3rem; margin: 2.6rem 0 .6rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--grid); }
h3 { font-size: 1.05rem; margin: 1.8rem 0 .4rem; color: var(--ink-2); }
p { margin: .5rem 0 .9rem; color: var(--ink-2); max-width: 78ch; }
code, pre { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .84rem; }
pre { background: var(--surface); border: 1px solid var(--grid); border-radius: 8px;
      padding: .85rem 1rem; overflow-x: auto; color: var(--ink); }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--grid);
              border-radius: 10px; margin: .8rem 0 1.4rem; }
table { border-collapse: collapse; width: 100%; font-size: .86rem;
        font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .5rem .85rem; border-bottom: 1px solid var(--grid);
         white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; background: var(--page); position: sticky; top: 0; }
tbody tr:last-child td { border-bottom: 0; }
td.better { color: #006300; }
td.worse  { color: #b3261e; font-weight: 600; }
figure { margin: 1.2rem 0 2rem; background: var(--surface); border: 1px solid var(--grid);
         border-radius: 10px; padding: .9rem; }
figure img, figure video { display: block; width: 100%; height: auto; max-width: 100%;
                           border-radius: 6px; background: var(--surface); }
figcaption { margin-top: .6rem; color: var(--ink-2); font-size: .85rem; }
.caption { color: var(--muted); font-size: .8rem; margin: .3rem 0 .8rem; }
.lede { font-size: 1.02rem; color: var(--ink); }
@media (prefers-color-scheme: dark) {
  :root { --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7; --grid: #2c2c2a; --rule: #383835; }
  /* The figures are rendered on a light surface, so their card stays light --
     a PNG cannot follow the reader's theme, and inverting the card around it
     would put a hard white rectangle in a dark page.  The caption sits *inside*
     that light card, so it must keep the light-theme ink: the dark theme's
     --ink-2 on #fcfcfb is 1.75:1, which is not readable text. */
  figure { background: #fcfcfb; border-color: #2c2c2a; }
  figcaption { color: #52514e; }
  .table-wrap { background: #1a1a19; border-color: #2c2c2a; }
  th { background: #141413; }
  pre { background: #1a1a19; border-color: #2c2c2a; color: #ffffff; }
  td.better { color: #0ca30c; }
  td.worse { color: #e66767; }
}
"""


def _mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _is_image(path: Path) -> bool:
    """True for a still or an animated GIF, False for an mp4.

    ``animate.write_animation`` falls back to a GIF wherever ffmpeg cannot be
    found, and returns the path it actually wrote.  A GIF inside ``<video>`` is
    an element no browser can decode, so on exactly the machine where the
    fallback matters every animation would be a black box.
    """
    return _mime(path).startswith("image/")


def _data_uri(path: Path) -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{_mime(path)};base64,{payload}"


def _cell_class(value: str) -> str:
    if "(better)" in value:
        return ' class="better"'
    if "(WORSE)" in value:
        return ' class="worse"'
    return ""


def _html(report: Report, media_root: Path, embed_budget: int = DEFAULT_EMBED_BUDGET) -> str:
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(report.title)}</title>",
        f"<style>{_CSS}</style>",
        "</head><body><main>",
        f"<h1>{html.escape(report.title)}</h1>",
    ]
    budget = int(embed_budget)
    for block in report.blocks:
        if block.kind == "heading":
            level = min(max(block.level, 1), 6)
            parts.append(f"<h{level}>{html.escape(block.text)}</h{level}>")
        elif block.kind == "text":
            parts.append(f"<p>{html.escape(block.text)}</p>")
        elif block.kind == "code":
            parts.append(f"<pre><code>{html.escape(block.text)}</code></pre>")
        elif block.kind == "table":
            if block.caption:
                parts.append(f'<p class="caption">{html.escape(block.caption)}</p>')
            head = "".join(f"<th>{html.escape(str(c))}</th>" for c in block.header)
            body = "".join(
                "<tr>"
                + "".join(f"<td{_cell_class(str(c))}>{html.escape(str(c))}</td>" for c in row)
                + "</tr>"
                for row in block.rows
            )
            parts.append(
                f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>"
            )
        elif block.kind in ("figure", "video"):
            path = Path(block.path)
            if not path.exists():
                continue
            size = path.stat().st_size
            embed = size <= budget
            if embed:
                budget -= size
                source = _data_uri(path)
                note = ""
            else:
                source = _relative(path, media_root)
                note = " (linked, not embedded: over the size budget)"
            caption = html.escape(block.caption) + html.escape(note)
            if block.kind == "figure" or _is_image(path):
                parts.append(
                    f'<figure><img alt="{html.escape(block.caption)}" src="{source}">'
                    f"<figcaption>{caption}</figcaption></figure>"
                )
            else:
                parts.append(
                    f'<figure><video controls loop muted playsinline src="{source}"></video>'
                    f"<figcaption>{caption}</figcaption></figure>"
                )
    parts.append("</main></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------
_FIGURE_CAPTIONS = {
    "01_rmse_vs_lead.png": "Forecast error against lead time for the model and both baselines. "
    "The model has to start below persistence; where it crosses climatology is the end of its "
    "useful horizon.",
    "02_scorecard.png": "Error relative to persistence for every variable and lead time. "
    "Blue means the model has less error than persistence, red means more.",
    "03_seaice_vs_lead.png": "Ice-edge error and extent bias per hemisphere, in 10^6 km^2. "
    "Extent bias is forecast minus truth, so positive means too much ice: the zero line is the "
    "target and both signs are failures.",
    "04_map_thetao.png": "Sea water potential temperature: truth, prediction and error at three "
    "lead times. Land is grey.",
    "04_map_siconc.png": "Sea-ice concentration: truth, prediction and error at three lead times.",
    "04_map_zos.png": "Sea surface height: truth, prediction and error at three lead times.",
    "05_depth_hovmoller.png": "Error against depth and lead time. The lower row is relative to "
    "persistence and answers whether the model is only good at the surface.",
    "06_seaice_polar.png": "Arctic and Antarctic concentration with the observed and predicted "
    "15% ice edges drawn over it.",
    "07_free_timeseries.png": "Free-running rollout: global-mean SST and hemispheric ice extent "
    "against the truth and the climatology. This is the drift check: a model settling onto the "
    "climatology has lost its information, one leaving both has left the attractor.",
    "08_power_spectra.png": "Power by spherical-harmonic degree, and the model/truth ratio. "
    "A blurring model loses power at high degree.",
}


def _animation_caption(path: Path) -> str:
    name = path.name
    if "sst" in name:
        return "Sea surface temperature rollout: truth, prediction and error, with the valid date."
    if "seaice_arctic" in name:
        return "Arctic sea-ice concentration with the moving 15% ice edge."
    if "seaice_antarctic" in name:
        return "Antarctic sea-ice concentration with the moving 15% ice edge."
    if "current" in name:
        return "Surface current speed: truth, prediction and error."
    return name


#: Splits a score must never be quoted from, and why.  `holdout` is the year kept
#: back for the end of the challenge; the `ifs_forced_*` windows are inside it and
#: exist only so `forcing=file` can be exercised on the one year of atmosphere the
#: kit ships (docs/05 section 5.6).
HOLDOUT_DOMAINS = ("holdout", "ifs_forced_train", "ifs_forced_val")


def holdout_caution(domain: str) -> str:
    """The warning to print above a report scored on a holdout split, or "".

    A report is the artefact that gets pasted into a slide, and by then the
    command line that named the split is long gone.  The kit says "2024 is
    holdout, never report it" in six places; this is the one place a participant
    who skipped all six will still read it.
    """
    if domain not in HOLDOUT_DOMAINS:
        return ""
    if domain == "holdout":
        return (
            "**These numbers are from the `holdout` split (2024-2025), which is kept back "
            "for the end of the challenge. Do not report them as a score.**"
        )
    return (
        f"**These numbers are from `{domain}`, a window inside the `holdout` split "
        "(2024). It exists so that file-based forcing can be exercised on the one year "
        "of atmosphere this kit ships -- a model trained on it has seen holdout data. "
        "This is a plumbing demonstration, not a score, and must not be reported as one. "
        "See docs/05_coupling.md section 5.6.**"
    )


def build_report(
    result,
    free=None,
    module=None,
    figures: Sequence[Path] = (),
    animations: Sequence[Path] = (),
    timings: dict | None = None,
    checkpoint: str = "",
    statistics: StatisticsProvenance | None = None,
) -> Report:
    """Assemble the report blocks from a scored rollout.

    Args:
        statistics: where ``oceanarches/stats/`` came from, from
            :func:`oceanarches.evaluation.provenance.read_statistics_provenance`.
            A sampled build puts a caution above the first table, in the same
            place and for the same reason as :func:`holdout_caution`.
    """
    timings = timings or {}
    statistics = statistics or StatisticsProvenance()
    spec = result.spec
    lead = plots._lead_days(next(iter(next(iter(result.metrics.values())).values())))
    first_day, last_day = float(lead[0]), float(lead[-1])
    report = Report(title=f"Evaluation report: {spec.experiment}")

    report.text(
        f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from checkpoint "
        f"{checkpoint or spec.checkpoint} on split '{spec.domain}', "
        f"{result.n_samples} initialisations x {spec.lead_days} days."
    )
    caution = holdout_caution(spec.domain)
    if caution:
        report.text(caution)
    # Same treatment, different silence: a score quoted against a climatology
    # built from three sampled years is not the score it looks like, and the
    # `make stats-quick` that produced it is long out of the scrollback by the
    # time this report reaches a slide.
    sampled = statistics.caution()
    if sampled:
        report.text(sampled)

    # -- the answer ---------------------------------------------------------
    # The years were hard-coded as "1993-2025" here, which was a claim about a
    # file this report had not read: `--years 1993-2018` and `--quick` both make
    # it false. Say what the climatology actually records, or say nothing.
    climatology = (
        f"the {statistics.climatology_years} monthly mean"
        if statistics.climatology_years
        else "the monthly mean"
    )
    report.heading("Is this model any good?", 2)
    report.text(
        "A forecast error means nothing on its own, so every number below sits next to the two "
        "baselines it has to beat. Persistence -- 'tomorrow looks like today' -- is the line a "
        f"one-day forecast must clear; climatology -- {climatology} interpolated to the valid "
        "day -- is the line a long forecast eventually falls back to. All three go through "
        "exactly the same metric code, on exactly the same samples."
    )
    for day in sorted({first_day, last_day}):
        header, rows = headline_table(result, day)
        report.table(header, rows, caption=f"RMSE at day {day:g}, ocean cells only.")

    if result.losses:
        report.table(
            ["Forecast", "Training loss"],
            [[result.labels.get(k, k), _format(v)] for k, v in result.losses.items()],
            caption="The module's own loss, on the same samples. Lower is better; a value near "
            "1 is roughly what a one-day persistence forecast scores.",
        )

    # The crossover with climatology is the honest end of the forecast horizon,
    # and a 10-day rollout usually cannot see it. Use the long free-running
    # rollout when there is one, and say which rollout the number came from.
    horizon_source = free if free is not None and free.metrics else result
    horizons = [
        [plots.variable_label(v, with_units=False), forecast_horizon(horizon_source, v)]
        for v in plots._resolve_headline(horizon_source)
    ]
    report.table(
        ["Variable", "Model no better than climatology from"],
        horizons,
        caption=(
            "Useful forecast horizon: the first lead time at which the model's error reaches "
            f"the climatology's. Measured on the {horizon_source.spec.lead_days}-day rollout "
            f"from {horizon_source.n_samples} initialisation(s)"
            + (
                " -- one initialisation is an estimate, not a statistic."
                if horizon_source.n_samples < 4
                else "."
            )
        ),
    )

    # -- scorecard table (the figure's table view) --------------------------
    rows_labels, lead_days, matrix = plots.scorecard_table(result)
    if rows_labels:
        report.heading("Error relative to persistence, in full", 2)
        report.text(
            "The numbers behind the scorecard figure. Negative is better: -16 means the model's "
            "error is 16% smaller than persistence's."
        )
        report.table(
            ["Variable"] + [f"day {d:g}" for d in lead_days],
            [
                [plots.variable_label(name, with_units=False)]
                + [("n/a" if not np.isfinite(v) else f"{100 * v:+.0f}") for v in row]
                for name, row in zip(rows_labels, matrix)
            ],
        )

    # -- figures ------------------------------------------------------------
    if figures:
        report.heading("Figures", 2)
        for path in figures:
            report.figure(path, _FIGURE_CAPTIONS.get(Path(path).name, Path(path).name))

    # -- animations ---------------------------------------------------------
    if animations:
        report.heading("Animations", 2)
        report.text(
            "A still frame says the error is large; an animation says how it got large. Drift and "
            "blurring are visible here and invisible in a scalar."
        )
        for path in animations:
            report.video(path, _animation_caption(Path(path)))

    # -- free rollout -------------------------------------------------------
    if free is not None:
        report.heading("Free-running rollout", 2)
        report.text(
            f"{free.spec.lead_days} days from {free.n_samples} initialisation(s), with no "
            "correction at any point. A model that has learned to blur keeps a respectable RMSE "
            "while its global mean walks away from the truth."
        )
        try:
            series = plots.free_running_series(free)
            days = series["days"]
            rows = []
            # The climatology curve on figure 07 is drawn in the aqua identity
            # colour, which sits below the 3:1 mark-contrast target; the
            # documented relief is this table view, so it carries that curve too.
            has_climatology = "climatology" in series["sst"]
            for label, key in (
                ("Global-mean SST [degC]", "sst"),
                ("NH sea-ice extent [10^6 km^2]", "extent_nh"),
                ("SH sea-ice extent [10^6 km^2]", "extent_sh"),
            ):
                model, truth = series[key]["model"], series[key]["truth"]
                row = [
                    label,
                    _format(float(truth[0])),
                    _format(float(model[0])),
                    _format(float(truth[-1])),
                    _format(float(model[-1])),
                ]
                if has_climatology:
                    row.append(_format(float(series[key]["climatology"][-1])))
                row.append(_format(float(model[-1] - truth[-1])))
                rows.append(row)
            header = [
                "Quantity",
                "Truth day 1",
                "Model day 1",
                f"Truth day {days[-1]:g}",
                f"Model day {days[-1]:g}",
            ]
            if has_climatology:
                header.append(f"Climatology day {days[-1]:g}")
            header.append("Drift at the end")
            report.table(
                header,
                rows,
                caption="Drift over the free-running rollout. A model that has merely run out "
                "of information settles onto the climatology; one that has left the attractor "
                "walks past it.",
            )
            horizon_rows = []
            for variable in plots._resolve_headline(free):
                series = plots._series(free, "model", "rmse", variable)
                baseline = plots._series(free, "climatology", "rmse", variable)
                if series is None:
                    continue
                lead, values = series
                horizon_rows.append(
                    [
                        plots.variable_label(variable, with_units=False),
                        plots.variable_units(variable) or "1",
                        _format(float(values[0])),
                        _format(float(values[-1])),
                        _format(float(baseline[1][-1])) if baseline is not None else "n/a",
                    ]
                )
            if horizon_rows:
                report.table(
                    [
                        "Variable",
                        "Unit",
                        "RMSE day 1",
                        f"RMSE day {days[-1]:g}",
                        f"Climatology day {days[-1]:g}",
                    ],
                    horizon_rows,
                    caption="Free-running error growth. An RMSE far above the climatology's is "
                    "not a degraded forecast, it is a model that has left the attractor.",
                )
        except Exception as error:  # noqa: BLE001 - a missing cache must not break the report
            report.text(
                f"(The drift table could not be computed: {type(error).__name__}: {error})"
            )

    # -- provenance ---------------------------------------------------------
    report.heading("How this was produced", 2)
    # A model that is not one checkpoint has to say how to re-run itself: a
    # coupled system's experiment name is a directory under evalstore/, not a run
    # under modelstore/, so `run_eval --exp <name>` would not work.
    own_command = getattr(module, "evaluation_command", None)
    report.code(
        own_command(spec)
        if callable(own_command)
        else (
            f"make eval NAME={spec.experiment} LEAD_DAYS={spec.lead_days}\n"
            f"# or, in full:\n"
            f".venv/bin/python -m oceanarches.evaluation.run_eval --exp {spec.experiment} "
            f"--lead-days {spec.lead_days} --n-inits {spec.n_inits} "
            f"--init-selection {spec.selection} --domain {spec.domain}"
        )
    )
    initial = result.init_times
    report.table(
        ["Setting", "Value"],
        [
            ["Checkpoint", checkpoint or spec.checkpoint],
            # The file name is not unique across retrainings; this is.
            ["Checkpoint fingerprint", spec.checkpoint_fingerprint or "n/a"],
            ["Config hash", spec.config_hash or "n/a"],
            ["Split", spec.domain],
            ["Initialisations", str(result.n_samples)],
            ["First / last initial time", f"{initial[0]} / {initial[-1]}" if initial else "n/a"],
            ["Lead time", f"{spec.lead_days} days at {getattr(module, 'lead_time_hours', 24)} h"],
            *statistics.rows(),
            ["Rollout wall clock", f"{result.seconds:.1f} s"],
            # Everything up to this report, which is all that can be known while
            # writing it; `summary.json` records the whole run.
            ["Wall clock to this report", f"{timings.get('total', float('nan')):.1f} s"],
            ["Cache", str(result.directory)],
        ],
        caption="Provenance. Every number in this report was measured on exactly this.",
    )
    report.heading("What this evaluation does not tell you", 2)
    report.text(
        "The scores are averages over the initialisations listed above and over ocean cells "
        "only, weighted by cos(latitude) and normalised by ocean area -- land contributes "
        "nothing. Sea-ice extent on this 1-degree grid sits high against published satellite "
        "figures (the March Arctic maximum computes to about 18.1 x 10^6 km^2 against a "
        "published 14-16) because the regrid smears the ice edge across whole cells; compare a "
        "model against the truth on this grid, which is what every metric here does, rather "
        "than against a satellite product on another one."
    )
    return report


def write_report(
    result,
    free=None,
    module=None,
    out_dir: Path = Path("."),
    figures: Sequence[Path] = (),
    animations: Sequence[Path] = (),
    timings: dict | None = None,
    checkpoint: str = "",
    statistics: StatisticsProvenance | None = None,
    embed_budget: int = DEFAULT_EMBED_BUDGET,
) -> list[Path]:
    """Write ``report.md`` and a self-contained ``report.html``.  Returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        result,
        free=free,
        module=module,
        figures=figures,
        animations=animations,
        timings=timings,
        checkpoint=checkpoint,
        statistics=statistics,
    )
    markdown_path = out_dir / "report.md"
    html_path = out_dir / "report.html"
    markdown_path.write_text(_markdown(report, out_dir))
    html_path.write_text(_html(report, out_dir, embed_budget=embed_budget))
    return [markdown_path, html_path]
