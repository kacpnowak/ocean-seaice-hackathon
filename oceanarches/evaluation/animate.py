"""Rollout animations: mp4 through the bundled ffmpeg, GIF if that fails.

A still frame at day 10 tells you the error is large.  An animation tells you
*how* it got large -- whether the model drifts, whether the ice edge dissolves,
whether eddies stop moving and start smearing.  That is the failure mode a
30-minute model has, and it is invisible in a scalar.

Everything is drawn from the cached rollout (see
:mod:`oceanarches.evaluation.rollout`), never from the model, so re-rendering an
animation costs no GPU.

Two implementation notes worth knowing before you edit this file:

* **There may be no system ffmpeg.**  ``imageio_ffmpeg.get_ffmpeg_exe()``
  unpacks a binary from the wheel and ``imageio`` picks it up automatically; that
  is the only reason mp4 works here.  If it ever does not, every writer falls
  back to an animated GIF rather than failing the evaluation.
* **Frames are padded to a multiple of 16 pixels**, in the figure's own
  background colour.  H.264 needs macroblock-aligned dimensions; letting imageio
  *resize* to fit them instead (its default) resamples every frame and turns
  crisp 1-degree cells into mush.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import xarray as xr

from ..dataloaders import variables as V
from ..metrics.seaice_metrics import ICE_EDGE_THRESHOLD
from . import plots

__all__ = [
    "write_animation",
    "animate_field_triptych",
    "animate_seaice",
    "animate_current_speed",
    "render_all",
]

#: Colour written into the padding strip; matches the figure surface so the pad
#: is invisible.
_PAD_RGB = (252, 252, 251)


def _pad_to_macroblock(frame: np.ndarray, block: int = 16) -> np.ndarray:
    """Pad an ``(h, w, 3)`` frame up to a multiple of ``block`` in both axes."""
    height, width = frame.shape[:2]
    pad_h = (-height) % block
    pad_w = (-width) % block
    if not pad_h and not pad_w:
        return frame
    out = np.empty((height + pad_h, width + pad_w, 3), dtype=frame.dtype)
    out[:] = np.array(_PAD_RGB, dtype=frame.dtype)
    out[:height, :width] = frame
    return out


def _figure_to_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    return _pad_to_macroblock(buffer[..., :3].copy())


def write_animation(frames: Sequence[np.ndarray], out_path: Path, fps: int = 6) -> Path:
    """Write ``frames`` as mp4, or as a GIF if no ffmpeg can be found.

    Returns the path actually written, which may have a different suffix from
    ``out_path``.
    """
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("Nothing to animate: no frames were rendered.")

    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
        writer = imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,  # frames are already aligned; do not resample
            ffmpeg_log_level="error",
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return out_path
    except Exception as error:  # noqa: BLE001 - any ffmpeg problem falls back
        warnings.warn(
            f"mp4 encoding failed ({type(error).__name__}: {error}); writing a GIF instead.",
            stacklevel=2,
        )
        gif_path = out_path.with_suffix(".gif")
        imageio.mimsave(gif_path, list(frames), duration=1000 / max(fps, 1), loop=0)
        return gif_path


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------
def _valid_time(dataset: xr.Dataset, init_index: int, lead_index: int) -> str:
    init = dataset["time"].to_numpy()[init_index]
    lead = dataset["prediction_timedelta"].to_numpy()[lead_index]
    return str(np.datetime64(init + lead, "D"))


def _surface(dataset: xr.Dataset, variable: str, init_index: int) -> xr.DataArray:
    field = dataset[variable].isel(time=init_index)
    if "depth" in field.dims:
        field = field.isel(depth=0)
    return field


def animate_field_triptych(
    result,
    variable: str,
    out_path: Path,
    init_index: int = 0,
    fps: int = 6,
    dpi: int = 100,
    transform: Callable[[xr.Dataset, int], xr.DataArray] | None = None,
    title: str | None = None,
    units: str | None = None,
    diverging: bool | None = None,
) -> Path:
    """Truth | prediction | error, animated over lead time, with the valid date.

    The colour scales are fixed over the whole animation -- computed once from
    the truth -- because a per-frame rescale makes a drifting forecast look
    stationary, which is the exact thing this animation exists to expose.

    Args:
        result: a :class:`~oceanarches.evaluation.rollout.RolloutResult` with
            cached fields.
        variable: name in the cached dataset.
        out_path: destination (``.mp4``; a ``.gif`` is written if ffmpeg fails).
        init_index: which initialisation to animate.
        fps, dpi: frame rate and resolution.
        transform: optional ``(dataset, init_index) -> DataArray`` to derive a
            field that is not stored directly (surface current speed).
        title: overrides the panel title.
        units: overrides the unit taken from ``variables.py``.
        diverging: overrides whether the *field* panels use the diverging map.
            A derived field can have a different polarity from the variable it
            was built from -- current *speed* is a magnitude even though ``uo``
            is signed -- and getting this wrong would centre a strictly positive
            field on zero and throw away half the colour range.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plots.apply_theme()
    predictions, targets = result.open_predictions(), result.open_targets()
    extract = transform or (lambda ds, i: _surface(ds, variable, i))
    truth = extract(targets, init_index).load()
    prediction = extract(predictions, init_index).load()
    n_frames = truth.sizes["prediction_timedelta"]

    finite = truth.to_numpy()[np.isfinite(truth.to_numpy())]
    diverging_field = (
        plots._is_diverging_variable(variable) if diverging is None else bool(diverging)
    )
    if diverging_field:
        limit = plots._symmetric_limit(finite)
        vmin, vmax, field_cmap = -limit, limit, plots.DIVERGING_MAP
    else:
        vmin = float(np.nanpercentile(finite, 1)) if finite.size else 0.0
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        field_cmap = plots.SEQUENTIAL_MAP
    error_limit = plots._symmetric_limit((prediction - truth).to_numpy())
    units = units or plots.variable_units(variable) or "1"
    name = title or plots.variable_label(variable, with_units=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 3.4),
        subplot_kw={"projection": plots._projection("global")},
        squeeze=True,
    )
    # `dpi` has to be set on the figure, not only on rcParams: `render_all` sets
    # the rcParam, but calling this function directly (a notebook does) would
    # otherwise accept a dpi argument and quietly ignore it.
    fig.set_dpi(dpi)
    meshes = []
    for column, (field, panel) in enumerate(
        (
            (truth.isel(prediction_timedelta=0), "GLORYS truth"),
            (prediction.isel(prediction_timedelta=0), "Model prediction"),
            (
                (prediction - truth).isel(prediction_timedelta=0),
                "Error (model - truth)",
            ),
        )
    ):
        ax = axes[column]
        if panel.startswith("Error"):
            mesh = plots._draw_field(
                ax,
                field,
                plots.DIVERGING_MAP,
                norm=TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit),
            )
        else:
            mesh = plots._draw_field(ax, field, field_cmap, vmin=vmin, vmax=vmax)
        ax.set_global()
        ax.set_title(panel, color=plots.INK, fontsize=9)
        meshes.append(mesh)
    # `extend` because the scale is the 1st-99th percentile of the truth: about
    # 2% of the cells, and any drift the model develops, are outside it. Without
    # the arrows a saturated cell is indistinguishable from one exactly at the
    # limit.
    bar = fig.colorbar(
        meshes[0],
        ax=axes[:2].tolist(),
        shrink=0.85,
        pad=0.015,
        fraction=0.02,
        extend="both",
    )
    bar.set_label(f"{name} [{units}]", color=plots.INK_SECONDARY)
    bar.outline.set_edgecolor(plots.AXIS_RULE)
    bar_error = fig.colorbar(
        meshes[2], ax=axes[2], shrink=0.85, pad=0.015, fraction=0.04, extend="both"
    )
    bar_error.set_label(f"Error [{units}]", color=plots.INK_SECONDARY)
    bar_error.outline.set_edgecolor(plots.AXIS_RULE)
    suptitle = fig.suptitle("", fontsize=12, color=plots.INK, fontweight="bold")

    _ = ccrs  # imported for its side effect of registering the projection classes
    frames = []
    for index in range(n_frames):
        fields = (
            truth.isel(prediction_timedelta=index),
            prediction.isel(prediction_timedelta=index),
            (prediction - truth).isel(prediction_timedelta=index),
        )
        for mesh, field in zip(meshes, fields):
            mesh.set_array(np.ma.masked_invalid(field.to_numpy()).ravel())
        suptitle.set_text(
            f"{name} -- valid {_valid_time(targets, init_index, index)}  "
            f"(day {index + 1} of the forecast)"
        )
        frames.append(_figure_to_frame(fig))
    plt.close(fig)
    return write_animation(frames, out_path, fps=fps)


def animate_seaice(
    result,
    out_path: Path,
    hemisphere: str = "north",
    init_index: int = 0,
    fps: int = 6,
    dpi: int = 100,
    threshold: float = ICE_EDGE_THRESHOLD,
) -> Path:
    """Polar concentration with both 15% ice edges moving over the rollout.

    Both edges are on every frame, so what you watch is the *gap* between them
    opening or closing -- which is what IIEE measures, drawn.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plots.apply_theme()
    predictions, targets = result.open_predictions(), result.open_targets()
    if "siconc" not in predictions:
        raise ValueError("This run does not predict siconc.")
    truth = _surface(targets, "siconc", init_index).load()
    prediction = _surface(predictions, "siconc", init_index).load()
    lon, lat = truth["lon"].to_numpy(), truth["lat"].to_numpy()
    extent = [-180, 180, 50, 90] if hemisphere == "north" else [-180, 180, -90, -50]
    label = "Arctic" if hemisphere == "north" else "Antarctic"

    fig = plt.figure(figsize=(9.0, 4.6))
    fig.set_dpi(dpi)  # see animate_field_triptych: rcParams alone ignores the argument
    axes = []
    meshes = []
    for column, panel in enumerate(("GLORYS truth", "Model prediction")):
        ax = fig.add_subplot(1, 2, column + 1, projection=plots._projection(hemisphere))
        ax.set_extent(extent, ccrs.PlateCarree())
        field = (truth if column == 0 else prediction).isel(prediction_timedelta=0)
        meshes.append(plots._draw_field(ax, field, plots.SEQUENTIAL_MAP, vmin=0.0, vmax=1.0))
        ax.set_title(panel, color=plots.INK, fontsize=9)
        axes.append(ax)
    bar = fig.colorbar(meshes[0], ax=axes, shrink=0.85, pad=0.02, fraction=0.03)
    bar.set_label("Sea-ice concentration [1]", color=plots.INK_SECONDARY)
    bar.outline.set_edgecolor(plots.AXIS_RULE)
    fig.legend(
        handles=[
            Line2D([], [], color=plots.INK, linestyle="-", linewidth=1.4),
            Line2D([], [], color=plots.SERIES_COLOURS["model"], linestyle="--", linewidth=1.4),
        ],
        labels=[f"Observed {threshold:.0%} edge", f"Predicted {threshold:.0%} edge"],
        loc="outside lower center",
        ncol=2,
        labelcolor=plots.INK_SECONDARY,
    )
    suptitle = fig.suptitle("", fontsize=12, color=plots.INK, fontweight="bold")

    frames = []
    contours: list = []
    for index in range(truth.sizes["prediction_timedelta"]):
        truth_values = np.nan_to_num(truth.isel(prediction_timedelta=index).to_numpy(), nan=0.0)
        prediction_values = np.nan_to_num(
            prediction.isel(prediction_timedelta=index).to_numpy(), nan=0.0
        )
        meshes[0].set_array(
            np.ma.masked_invalid(truth.isel(prediction_timedelta=index).to_numpy()).ravel()
        )
        meshes[1].set_array(
            np.ma.masked_invalid(prediction.isel(prediction_timedelta=index).to_numpy()).ravel()
        )
        for contour in contours:
            contour.remove()
        contours = []
        for ax in axes:
            for values, colour, style in (
                (truth_values, plots.INK, "-"),
                (prediction_values, plots.SERIES_COLOURS["model"], "--"),
            ):
                contours.append(
                    ax.contour(
                        lon,
                        lat,
                        values,
                        levels=[threshold],
                        colors=[colour],
                        linewidths=1.3,
                        linestyles=[style],
                        transform=ccrs.PlateCarree(),
                    )
                )
        suptitle.set_text(
            f"{label} sea ice -- valid {_valid_time(targets, init_index, index)}  "
            f"(day {index + 1} of the forecast)"
        )
        frames.append(_figure_to_frame(fig))
    plt.close(fig)
    return write_animation(frames, out_path, fps=fps)


def _surface_current_speed(dataset: xr.Dataset, init_index: int) -> xr.DataArray:
    """``sqrt(uo^2 + vo^2)`` at the shallowest level, as a named DataArray."""
    u = _surface(dataset, "uo", init_index)
    v = _surface(dataset, "vo", init_index)
    speed = np.sqrt(u**2 + v**2)
    return speed.rename("current_speed")


def animate_current_speed(result, out_path: Path, **kwargs) -> Path:
    """Surface current speed: truth | prediction | error.

    Velocity is where a blurring model shows itself first -- speed is a quadratic
    of two fields the model is only asked to get right linearly, so a small
    smoothing of ``uo`` and ``vo`` becomes a visible loss of the western boundary
    currents.
    """
    return animate_field_triptych(
        result,
        "uo",
        out_path,
        transform=_surface_current_speed,
        title="Surface current speed",
        units=V.VARIABLES["uo"].units,
        # `uo` is signed and takes a diverging map; its magnitude is not.
        diverging=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def render_all(
    result,
    free=None,
    out_dir: Path = Path("animations"),
    fps: int = 6,
    dpi: int = 100,
    progress: bool = True,
) -> list[Path]:
    """Every animation this result supports; a failure warns and is skipped.

    Args:
        progress: name each animation *before* rendering it, and again with its
            wall clock when it lands.  This is the most expensive stage of
            ``make eval`` and it used to print nothing for minutes; the line
            before the work is the one that tells a participant it has not hung.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.dpi"] = dpi
    written: list[Path] = []

    def attempt(name: str, function, *args, **kwargs) -> None:
        if progress:
            print(f"  animating {name} ...", flush=True)
        started = time.time()
        try:
            path = function(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - one bad animation is not fatal
            warnings.warn(
                f"Animation {name!r} was skipped: {type(error).__name__}: {error}", stacklevel=2
            )
            return
        if path is not None:
            written.append(Path(path))
            if progress:
                print(
                    f"  animation {Path(path).name} ({time.time() - started:.1f}s)",
                    flush=True,
                )

    if result is not None and result.predictions_path.exists():
        attempt(
            "sst",
            animate_field_triptych,
            result,
            "thetao",
            out_dir / "sst_rollout.mp4",
            fps=fps,
            dpi=dpi,
        )
        attempt(
            "seaice_arctic",
            animate_seaice,
            result,
            out_dir / "seaice_arctic.mp4",
            hemisphere="north",
            fps=fps,
            dpi=dpi,
        )
        attempt(
            "seaice_antarctic",
            animate_seaice,
            result,
            out_dir / "seaice_antarctic.mp4",
            hemisphere="south",
            fps=fps,
            dpi=dpi,
        )
        attempt(
            "current_speed",
            animate_current_speed,
            result,
            out_dir / "surface_current_speed.mp4",
            fps=fps,
            dpi=dpi,
        )
    if free is not None and free.predictions_path.exists():
        days = free.spec.lead_days
        attempt(
            "free_sst",
            animate_field_triptych,
            free,
            "thetao",
            out_dir / f"free_{days}d_sst.mp4",
            fps=fps,
            dpi=dpi,
        )
        attempt(
            "free_seaice_arctic",
            animate_seaice,
            free,
            out_dir / f"free_{days}d_seaice_arctic.mp4",
            hemisphere="north",
            fps=fps,
            dpi=dpi,
        )
    return written
