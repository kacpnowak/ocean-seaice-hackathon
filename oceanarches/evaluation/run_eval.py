"""``make eval`` -- one command from a trained checkpoint to a report.

    make eval NAME=my_run LEAD_DAYS=10
    .venv/bin/python -m oceanarches.evaluation.run_eval --exp my_run --lead-days 10

What it does, in order:

1. loads the trained module with
   :func:`geoarches.lightning_modules.base_module.load_module`, which reads
   ``modelstore/<name>/config.yaml`` and the newest checkpoint;
2. builds the test dataset for the domain that run was configured with, and
   rolls the model out ``--lead-days`` days from ``--n-inits`` initial
   conditions spread across the period;
3. writes the predictions and the truth to zarr with geoarches'
   ``ZarrIterativeWriter``;
4. scores the model **and both baselines** with the same metric classes the
   Lightning module logs during validation;
5. runs a long free-running rollout (``--free-days``) for the drift check;
6. renders the figures and the animations;
7. writes ``evalstore/<name>/report.md`` and a self-contained ``report.html``.

Every stage can be skipped (``--skip-baselines``, ``--skip-figures``,
``--skip-animations``, ``--skip-free-rollout``, ``--skip-report``) and the two
rollouts are cached, so re-rendering a figure does not re-run the model.  The
cache is keyed on the checkpoint's *contents* and on ``config.yaml``, not on the
checkpoint's file name, so retraining a run under the same name recomputes rather
than reporting the previous model's scores.  Pass ``--force`` to recompute
regardless.

Reproducing a reference number
------------------------------
``--init-selection first`` takes the initial conditions in dataset order instead
of spreading them over the period, which is what makes a run comparable against
a fixed reference set.  The Task 6 validation numbers are the first 128 samples
of ``tiny_val`` at one day::

    python -m oceanarches.evaluation.run_eval --exp task6_tiny \\
        --domain tiny_val --lead-days 1 --n-inits 128 --init-selection first \\
        --skip-figures --skip-animations --skip-free-rollout

and ``tests/test_evaluation.py`` pins the agreement between that path and the
module's own ``validation_step``.

Coupled systems
---------------
``--coupled`` builds the model out of several trained components instead of
loading one checkpoint::

    make couple OCEAN=my_ocean_run SEAICE=my_ice_run
    python -m oceanarches.evaluation.run_eval --coupled \\
        --components ocean=my_ocean_run seaice=my_ice_run --lead-days 10

Everything after that is unchanged, because
:class:`~oceanarches.lightning_modules.coupled.CoupledForecastModule` exposes the
same ``forward`` / ``forward_multistep`` and the same state helpers as a single
module -- there is no branch anywhere below on whether the model is coupled.  The
wiring (``--mode``, ``--order``, ``--unpredicted-forcing``) is part of the cache
key, because it changes the forecast.

``--compare-with LABEL=RUN`` lays previously scored runs over this one's curves,
which is how "coupled against uncoupled" and the three-way sea-ice comparison are
drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import torch
from geoarches.lightning_modules.base_module import load_module
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Importing the metrics package registers every GLORYS metric with geoarches'
# evaluation registry, so `python -m geoarches.evaluation.eval_multistep
# --metrics glorys_deterministic` also works on the files this pipeline writes.
from .. import guards, paths
from .. import metrics as _metrics  # noqa: F401
from ..dataloaders import variables as V
from ..lightning_modules.coupled import (
    COUPLING_MODES,
    UNPREDICTED_FORCING_MODES,
    named_component_like,
    union_component,
)
from . import provenance as provenance_module
from . import render_cache
from .baselines import build_forecasters
from .rollout import (
    CheckpointIdentity,
    RolloutSpec,
    cache_dir_for,
    checkpoint_identity,
    choose_initialisations,
    load_cached,
    run_rollout,
    statistics_fingerprint,
)

__all__ = [
    "main",
    "evaluate",
    "build_dataset",
    "load_forecast_system",
    "parse_components",
    "resolve_run",
    "output_name",
]

#: Repository configs, for `--coupled`: it composes the real
#: `configs/module/coupled.yaml` rather than reproducing its defaults here.
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")

#: Which dataloader config lays out the shared state of a coupled system,
#: by the name of the component that system is equivalent to.  Checked against
#: the composed config at load time, so a wrong entry is an error and not a
#: silently mis-ordered state.
DATALOADER_FOR_COMPONENT = {
    "full": "glorys",
    "ocean": "glorys_ocean",
    "seaice": "glorys_seaice",
    "seaice_isolated": "glorys_seaice_isolated",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataset(cfg, domain: str | None, multistep: int, limit_examples: int | None = None):
    """The evaluation dataset: the run's own ``test_args``, with the lead time set.

    Reuses whatever the training run was configured with -- variables, depth
    levels, normalisation, lead time -- so the model is never scored on a state
    assembled differently from the one it was trained on.  Only ``domain`` and
    ``multistep`` are overridden.
    """
    test_args = dict(getattr(cfg.dataloader, "test_args", {}) or {})
    test_args["domain"] = domain or test_args.get("domain", "test")
    test_args["multistep"] = int(multistep)
    if limit_examples is not None:
        test_args["limit_examples"] = int(limit_examples)
    return instantiate(cfg.dataloader.dataset, **test_args)


# ---------------------------------------------------------------------------
# What is being scored: one checkpoint, or several coupled together
# ---------------------------------------------------------------------------
def output_name(experiment: str) -> str:
    """The directory name under ``evalstore/`` for a run named ``experiment``.

    ``--help`` says ``--exp`` may be "a path to a directory holding config.yaml
    and checkpoints/", and ``Path("evalstore") / "/abs/run"`` is ``/abs/run`` --
    pathlib discards the left operand when the right is absolute.  So passing a
    path wrote the report, the figures, the animations and the whole rollout
    cache into the *checkpoint* directory, silently.  The output is named after
    the run, wherever the run itself lives.
    """
    return Path(experiment).name or str(experiment)


def resolve_run(run: str, label: str | None = None) -> Path:
    """``modelstore/<run>`` (or ``<run>`` as a path), checked, or a good SystemExit.

    Two mistakes account for most of the failed first evaluations, and neither
    used to say anything useful:

    * a run name that does not exist -- ``make eval NAME=typo``, and also plain
      ``make eval`` after ``make train-tiny NAME=my_run``, because both targets
      default ``NAME`` to ``tiny``.  ``load_module`` then raises
      ``FileNotFoundError: <repo-root>/typo/config.yaml``: a path under the
      repository root, with the ``modelstore/`` prefix silently dropped, naming
      no valid run.
    * a run killed before its first checkpoint -- an empty ``checkpoints/``
      directory, which geoarches reports as ``IndexError: list index out of
      range`` with no mention of checkpoints at all.

    Args:
        run: run name under ``modelstore/``, or a path to a run directory.
        label: how to name the offending option in the message, e.g.
            ``"--components seaice=my_ice_run"``.  Defaults to ``--exp <run>``.

    Raises:
        SystemExit: with the run, the path that was looked for, and what to do.
    """
    label = label or f"--exp {run}"
    modelstore = Path(paths.setting("MODELSTORE"))
    given = Path(str(run))
    # A name and a path are different mistakes and deserve different answers.
    # They also have to be told apart explicitly, because `modelstore / run` is
    # NOT a name lookup when `run` is absolute: pathlib discards the left operand,
    # so `root` came out as the path itself and the message offered the reader the
    # same directory twice as two places it had looked, then listed the runs in
    # modelstore/ as though a name had been typed.
    looks_like_path = given.is_absolute() or len(given.parts) > 1
    root = given if looks_like_path else modelstore / str(run)
    if not root.is_dir():
        if looks_like_path:
            raise SystemExit(
                f"{label}: that directory does not exist.\n"
                "As a path, `--exp` wants the run directory itself -- the one holding "
                "config.yaml and checkpoints/.\n"
                "If this is running under SLURM, check the path is on a shared "
                "filesystem: /tmp is local to each node, so a directory staged on the "
                "login node is not there when the job runs.\n"
                "A bare run name is looked up under "
                f"{modelstore}/ instead."
            )
        available = (
            sorted(p.name for p in modelstore.iterdir() if (p / "config.yaml").is_file())
            if modelstore.is_dir()
            else []
        )
        listing = "\n  ".join(available) if available else "(none -- train one first)"
        raise SystemExit(
            f"{label}: no such run. Looked for {root}/.\n"
            f"Runs in {modelstore}/:\n  {listing}\n"
            "Train one with: make train-tiny NAME=my_first_run"
        )

    if not (root / "config.yaml").is_file():
        raise SystemExit(
            f"{label}: {root}/config.yaml is missing, so there is nothing to load. "
            "A run trained with `log=False` never writes it -- retrain with the default "
            "`log=True` (see docs/03 section 3.8)."
        )

    checkpoints = sorted((root / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise SystemExit(
            f"{label}: {root}/checkpoints/ holds no *.ckpt file, so the run has no "
            "weights to score. geoarches checkpoints on "
            "`global_step % save_step_frequency == 0` and writes nothing at the end of "
            "fit, so a run killed before its first save leaves this directory empty. "
            "Train for at least `save_step_frequency` steps, or lower it: "
            "make train-tiny NAME=" + str(run) + ' HYDRA_ARGS="++save_step_frequency=500"'
        )
    return root


def parse_components(pairs) -> dict[str, str]:
    """``["ocean=my_run", "seaice=other"]`` -> ``{"ocean": "my_run", ...}``.

    The key is the *component* the checkpoint was trained as, not a free label:
    it is what tells the router which channels of the shared state that model
    reads and writes.  ``CoupledForecastModule`` re-checks it against the
    checkpoint's own config.
    """
    components: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(
                f"--components takes NAME=RUN pairs, got {pair!r}. "
                "For example: --components ocean=my_ocean_run seaice=my_ice_run"
            )
        name, run = pair.split("=", 1)
        name, run = name.strip(), run.strip()
        if not run:
            raise SystemExit(
                f"--components {name}= has no run name. Point it at a directory under "
                "modelstore/, or at a path holding config.yaml and checkpoints/."
            )
        if name not in V.COMPONENTS:
            raise SystemExit(
                f"Unknown component {name!r}. Available: {sorted(V.COMPONENTS)} "
                "(they are defined in oceanarches/dataloaders/variables.py)."
            )
        if name in components:
            raise SystemExit(f"--components names {name!r} twice.")
        # Validate here, not at instantiate time. `coupled.py` calls
        # `load_module` inside hydra's `instantiate`, which re-wraps and
        # stringifies the exception without its arguments: the participant sees
        # `InstantiationException: ... FileNotFoundError(2, 'No such file or
        # directory')` with no path and no component name, on the kit's flagship
        # coupling workflow.
        resolve_run(run, label=f"--components {name}={run}")
        components[name] = run
    return components


def coupled_identity(components: dict[str, str], order, args) -> CheckpointIdentity:
    """A cache key for a coupled system: every component's checkpoint, and the wiring.

    The wiring is in the *config* hash on purpose.  ``parallel`` and
    ``sequential`` give different forecasts from the identical checkpoints, and
    so do the two ``unpredicted_forcing`` policies, so a cache that ignored them
    would answer a question it was not asked -- which is the failure this
    pipeline's cache design exists to prevent.
    """
    parts = {name: checkpoint_identity(path) for name, path in components.items()}
    name = "+".join(f"{key}@{parts[key].name}" for key in order)
    fingerprint = "|".join(f"{key}:{parts[key].fingerprint}" for key in order)
    wiring = "|".join(
        [args.mode, args.unpredicted_forcing, ",".join(order)]
        + [f"{key}:{parts[key].config_hash}" for key in order]
    )
    return CheckpointIdentity(
        name=name,
        fingerprint=hashlib.sha256(fingerprint.encode()).hexdigest()[:16],
        config_hash=hashlib.sha256(wiring.encode()).hexdigest()[:16],
    )


def compose_coupled_config(components: dict[str, str], order, args):
    """Compose ``configs/module/coupled.yaml`` with the components filled in.

    Composing the shipped config rather than building an equivalent one here is
    the point: ``mode``, the metric set and the shared-state layout are the
    file's, so editing ``configs/module/coupled.yaml`` changes what
    ``make couple`` does.
    """
    union = union_component([V.get_component(name) for name in order])
    union_name = named_component_like(union)
    if union_name is None:
        raise SystemExit(
            f"The components {order} together predict {union.prognostic} and read "
            f"{union.forcing}, which is not one of the COMPONENTS entries "
            f"{sorted(V.COMPONENTS)}. Add the combination to COMPONENTS in "
            "oceanarches/dataloaders/variables.py -- it is the single source of truth "
            "for what a state contains."
        )
    dataloader = DATALOADER_FOR_COMPONENT.get(union_name)
    if dataloader is None:
        raise SystemExit(
            f"No dataloader config lays out the shared state of a {union_name!r} system. "
            f"Add configs/dataloader/glorys_{union_name}.yaml and list it in "
            "run_eval.DATALOADER_FOR_COMPONENT."
        )

    listing = "{" + ", ".join(f"{key}: {run}" for key, run in components.items()) + "}"
    overrides = [
        "module=coupled",
        f"dataloader={dataloader}",
        f"++module.module.components={listing}",
        f"++module.module.order=[{','.join(order)}]",
        f"++module.module.mode={args.mode}",
        f"++module.module.unpredicted_forcing={args.unpredicted_forcing}",
        f"++module.inference.rollout_iterations={int(args.lead_days)}",
        f"++name={args.exp}",
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR, job_name="couple"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)

    if cfg.dataloader.component != union_name:
        raise SystemExit(
            f"configs/dataloader/{dataloader}.yaml lays out the {cfg.dataloader.component!r} "
            f"state, but these components need the {union_name!r} one. Fix "
            "run_eval.DATALOADER_FOR_COMPONENT."
        )
    return cfg


def load_forecast_system(args, device: str):
    """The thing being scored, its config, and its cache identity.

    One checkpoint or a coupled system of several -- the rest of the pipeline
    cannot tell the difference, which is the whole design: a coupled system
    exposes the same ``forward`` / ``forward_multistep`` and the same state
    helpers as a single module.
    """
    if not args.coupled:
        resolve_run(args.exp)
        print(f"Loading {args.exp} on {device} ...", flush=True)
        module, cfg = load_module(args.exp, device=device)
        return module, cfg, checkpoint_identity(args.exp)

    components = parse_components(args.components)
    if not components:
        raise SystemExit(
            "--coupled needs --components NAME=RUN [NAME=RUN ...], e.g. "
            "--components ocean=my_ocean_run seaice=my_ice_run"
        )
    order = list(args.order or components)
    if sorted(order) != sorted(components):
        raise SystemExit(f"--order {order} does not name the components {sorted(components)}.")

    print(
        f"Coupling {' -> '.join(order)} ({args.mode}, unpredicted forcing: "
        f"{args.unpredicted_forcing}) on {device} ...",
        flush=True,
    )
    cfg = compose_coupled_config(components, order, args)
    module = instantiate(cfg.module.module, device=device).to(device)

    # `or []` on both sides, and only for the comparison: None means "every
    # prepared level" in both places, so the two spellings must compare equal --
    # but `module.depth_indices` keeps its None (see CoupledForecastModule).
    dataset_depths = list(cfg.dataloader.dataset.depth_indices or [])
    module_depths = list(module.depth_indices or [])
    if dataset_depths != module_depths:
        raise SystemExit(
            f"The shared state is built with depth_indices={dataset_depths} but the "
            f"components were trained on {module_depths}. They would be "
            "handed the wrong levels. Retrain, or fix configs/module/coupled.yaml."
        )
    return module, cfg, coupled_identity(components, order, args)


# ---------------------------------------------------------------------------
# Putting two result sets on one axes
# ---------------------------------------------------------------------------
def load_result_for_overlay(directory: Path, lead_days: int, domain: str, inits) -> object | None:
    """A previously computed rollout, if it answers the same question.

    Read back through the cache's own manifest, so a run scored earlier -- a
    single model, or a differently wired coupled system -- can be laid over this
    one without recomputing it.  Anything that would make the two curves
    incomparable (a different split, a different lead time, different initial
    conditions) returns None rather than a misleading figure.
    """
    manifest_path = Path(directory) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    if (
        manifest.get("lead_days") != int(lead_days)
        or manifest.get("domain") != domain
        or list(manifest.get("inits", [])) != list(inits)
    ):
        return None
    # Not compared against ours: an overlay is a different run and may
    # legitimately have been scored from another checkpoint. A manifest that
    # does not record its statistics at all fails to match below, via
    # `RolloutSpec.from_manifest`'s own default handling, and `render_comparison`
    # says so with the command to rescore it -- the same treatment every other
    # incomparable cache gets.
    spec = RolloutSpec.from_manifest(manifest)
    return load_cached(spec, Path(directory))


def render_comparison(args, result, out_dir: Path, evalstore: Path) -> list[Path]:
    """``--compare-with LABEL=EXP`` -> one figure with every model curve on it.

    This is the figure the coupling questions are answered with: coupled against
    uncoupled, or sea ice coupled against sea ice given the true ocean against
    sea ice that never sees the ocean.  It reuses ``plots.plot_overlay``, which
    draws the baselines once because persistence and climatology do not depend on
    which model produced the forecast.
    """
    if not args.compare_with:
        return []
    from . import plots

    sets = {args.model_label or args.exp: result}
    for pair in args.compare_with:
        if "=" not in pair:
            raise SystemExit(f"--compare-with takes LABEL=RUN pairs, got {pair!r}.")
        label, experiment = (part.strip() for part in pair.split("=", 1))
        directory = cache_dir_for(evalstore, output_name(experiment), f"lead{args.lead_days}d")
        other = load_result_for_overlay(
            directory, args.lead_days, result.spec.domain, result.spec.inits
        )
        if other is None:
            # `warnings.warn`, like every other "something was skipped" in this
            # pipeline (plots.render_all, animate.render_all). One channel means
            # a caller can catch or filter all of them together; a `print` to
            # stderr is invisible to `pytest.warns` and to a notebook's own
            # warning filter.
            warnings.warn(
                f"No comparable cached rollout for {experiment!r} in {directory}, so it "
                f"was left off the comparison figure. Score it the same way first, e.g. "
                f"`make eval NAME={experiment} LEAD_DAYS={args.lead_days}`, with the same "
                "--domain, --n-inits and --init-selection.",
                stacklevel=2,
            )
            continue
        sets[label] = other
    if len(sets) < 2:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric, variables in (
        ("rmse", None),
        ("rmse", ["siconc", "sithick"]),
    ):
        name = "comparison_rmse" + ("_seaice" if variables else "")
        try:
            written.append(
                plots.plot_overlay(
                    sets,
                    out_dir / f"{name}.png",
                    variables=variables,
                    metric=metric,
                    title="Coupled against uncoupled",
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad panel must not lose the run
            warnings.warn(
                f"Comparison figure {name!r} was skipped: {type(error).__name__}: {error}",
                stacklevel=2,
            )
    return written


# ---------------------------------------------------------------------------
# The allocation
# ---------------------------------------------------------------------------
def allocation_complaint(device: str, env: dict | None = None, hostname: str | None = None):
    """:func:`oceanarches.guards.allocation_warning`, applied to this entry point.

    The `StartupGuard` hydra callback in ``configs/config.yaml`` covers every
    *training* route, and covers none of the evaluation ones: ``make eval`` and
    ``make couple`` call this module directly and never reach hydra's task
    function.  Three participants ran evaluation without an allocation and one of
    them had a coupled eval die of ``torch.OutOfMemoryError`` on a card that
    already had 85 of 98 GiB in use by somebody else's job -- a traceback that
    mentions neither SLURM nor the step that was skipped.

    The signal and the words come from :mod:`oceanarches.guards`, deliberately
    unwrapped and unrepeated: two notions of "am I on an allocation" would be
    worse than one, and ``SLURM_JOB_ID`` is the one that was decided.

    Args:
        device: the resolved device.  Anything but ``cuda`` is exempt -- running
            an evaluation on the CPU here is what ``--device cpu`` is for, and it
            is this entry point's equivalent of ``cluster=local``.
        env, hostname: injected by the tests; default to the real ones.

    Returns:
        The complaint to print, or None when there is nothing to say.
    """
    if str(device) != "cuda":
        return None
    return guards.allocation_warning(
        env=env, hostname=hostname, cluster_name=None, cpu_hint="--device cpu"
    )


def warn_without_allocation(device: str) -> None:
    """Say it, and pause so Ctrl-C is possible.

    The printing, the ten-second pause and the ``OCEANARCHES_SKIP_GUARDS`` rule
    all belong to :func:`oceanarches.guards.warn_allocation`, which
    ``make train-*`` goes through as well.  One implementation, so a participant
    cannot meet two slightly different phrasings of "you have no allocation" and
    conclude they are two different problems.
    """
    if allocation_complaint(device) is None:
        return
    guards.warn_allocation(cluster_name=None, cpu_hint="--device cpu")


# ---------------------------------------------------------------------------
# Saying what is happening, and what it cost
# ---------------------------------------------------------------------------
def announce(stage: str, started: float, detail: str = "") -> None:
    """One line per stage, carrying the elapsed wall clock.

    ``make eval`` printed nothing at all between the rollout and the finished
    report -- about 200 s of a 240 s run, all of it in the figures and the
    animations.  A participant killed one at 180 s believing it had hung; it
    needed 257 s.  Every stage now says it has started, and how far into the run
    that is, so the next person can tell waiting from hanging.
    """
    print(
        f"[{time.time() - started:6.1f}s] {stage}" + (f" -- {detail}" if detail else ""),
        flush=True,
    )


def directory_size(path: Path) -> int:
    """Bytes of real files under ``path``.  Symlinks are not followed and a file
    that vanishes mid-walk is skipped, because this must never be the thing that
    fails an evaluation that has already succeeded."""
    total = 0
    for entry in Path(path).rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def human_size(n_bytes: float) -> str:
    """``752 MB``.  Decimal units, matching what `du -h --si` and quotas report."""
    size = float(n_bytes)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "kB") else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"


def cost_lines(out_dir: Path, evalstore: Path) -> list[str]:
    """What this run left on disk, and what the store now holds.

    One `LEAD_DAYS=10` evaluation of a toy checkpoint leaves ~750 MB, nearly all
    of it `predictions.zarr` and `targets.zarr` in the two rollout caches.  That
    is fine once and alarming after a dozen runs on a shared quota, and nothing
    said it: the run ended on `Done in 238.4s`.
    """
    out_dir, evalstore = Path(out_dir), Path(evalstore)
    lines = [f"This run's output in {out_dir} is {human_size(directory_size(out_dir))}."]
    if evalstore.exists() and evalstore.resolve() != out_dir.resolve():
        runs = sum(1 for entry in evalstore.iterdir() if entry.is_dir())
        lines.append(
            f"{evalstore}/ now holds {human_size(directory_size(evalstore))} across {runs} run(s)."
        )
    lines.append(
        "Most of it is the cached rollouts. Delete a run's directory to reclaim it, or "
        "score without the maps with --skip-fields (which costs every map, polar and "
        "spectrum figure)."
    )
    return lines


def stage_breakdown(timings: dict) -> str:
    """``rollout 19.6s, figures 61.2s, animations 128.0s`` -- where the time went."""
    labels = {
        "load_module": "load",
        "rollout": "rollout",
        "free_rollout": "free rollout",
        "figures": "figures",
        "animations": "animations",
        "report": "report",
    }
    # A stage served from cache costs ~0 s and saying "figures 0.0s" would read
    # as a failure rather than as a hit; the cache prints its own line already.
    parts = [
        f"{label} {timings[key]:.1f}s"
        for key, label in labels.items()
        if timings.get(key, 0.0) >= 0.05
    ]
    return ", ".join(parts) or "everything was served from cache"


def next_time_hint(args, timings: dict) -> str:
    """What a re-run of this exact command now costs, and how to make it cheaper.

    ``--skip-animations`` was in ``--help`` and in docs/06 and nowhere a
    participant re-running an evaluation would meet it, so re-runs cost the full
    animation stage over and over.  The command that just spent the time is the
    right place to say so, and it can quote what it actually measured.
    """
    if getattr(args, "force", False):
        return "Re-running without --force reuses the rollouts and the renders."
    animations = timings.get("animations")
    hint = "Re-running this command reuses the rollouts and the renders (--force recomputes)."
    if animations and not getattr(args, "skip_animations", False):
        hint += (
            f" Change anything and the animations re-render: that was {animations:.0f}s "
            'here, and EVAL_ARGS="--skip-animations" drops it.'
        )
    return hint


def cached_or_render(kind: str, directory: Path, key: str, render, force: bool = False):
    """Reuse an identical previous render, or draw it and record what it drew.

    The rollout has always been cached; the figures and the animations were not,
    so "just re-run the cell" cost ~200 s of a 240 s run for a result that was
    already on disk.  Same key discipline as the rollout cache: the inputs, never
    a file name, and ``--force`` bypasses it.
    """
    directory = Path(directory)
    if not force:
        cached = render_cache.cached_render(directory, key)
        if cached is not None:
            print(f"  reusing {len(cached)} cached {kind} in {directory}", flush=True)
            return cached
    paths = render()
    render_cache.record_render(directory, key, paths)
    return paths


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def evaluate(args) -> dict:
    """Run every requested stage and return a summary dict."""
    started = time.time()
    #: What the numbers cost when they were made: `timings["rollout"]` is the
    #: cached rollout's own wall clock, and the report quotes it.
    timings: dict[str, float] = {}
    #: What *this* process spent, which for a reused cache is nearly nothing.
    #: Kept apart so the closing breakdown of a 5-second warm run cannot claim a
    #: 20-second rollout it did not run.
    spent: dict[str, float] = {}

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Before the checkpoint, the dataset or the GPU: a visible card is not an
    # allocated one, and this is the entry point the hydra guard cannot see.
    warn_without_allocation(device)

    # Read before anything expensive, and print it: a report built on
    # `make stats-quick` statistics is a sampled comparison, and until now
    # nothing between that command and the finished scorecard said so.
    statistics = provenance_module.read_statistics_provenance()
    if statistics.known or statistics.unreadable:
        announce(statistics.summary_line(), started)
    # Hashed once and shared by both rollouts: rebuilding oceanarches/stats/ has
    # to invalidate the cache, and a warning printed next to a reused one is not
    # a guard -- it is read by the people who did not need it.
    statistics_digest = statistics_fingerprint()

    t0 = time.time()
    # Identity, not name: retraining a run rewrites the same checkpoint file, and
    # a cache keyed on the file name would answer with the previous model.
    module, cfg, identity = load_forecast_system(args, device)
    module.eval()
    timings["load_module"] = spent["load_module"] = time.time() - t0
    checkpoint = identity.name
    print(f"  checkpoint {checkpoint}, component {module.component.name}", flush=True)

    evalstore = Path(args.out or paths.setting("EVALSTORE"))
    run_label = output_name(args.exp)
    out_dir = evalstore / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    forecasters = build_forecasters(
        module,
        include_baselines=not args.skip_baselines,
        model_label=args.model_label or args.exp,
    )

    # -- the scored rollout --------------------------------------------------
    dataset = build_dataset(cfg, args.domain, args.lead_days)
    inits = choose_initialisations(len(dataset), args.n_inits, args.init_selection)
    spec = RolloutSpec.with_checkpoint(
        identity,
        statistics_digest=statistics_digest,
        experiment=args.exp,
        domain=str(dataset.domain),
        lead_days=int(args.lead_days),
        n_inits=len(inits),
        selection=args.init_selection,
        save_depths=args.save_depths,
        save_fields=not args.skip_fields,
        inits=tuple(int(i) for i in inits),
    )
    directory = cache_dir_for(evalstore, run_label, f"lead{args.lead_days}d")
    t0 = time.time()
    cached = None if args.force else load_cached(spec, directory)
    if cached is not None and set(cached.metrics) >= {f.key for f in forecasters}:
        announce(f"Reusing cached rollout in {directory}", started)
        result = cached
    else:
        announce(
            f"Rolling out {len(inits)} initialisations x {args.lead_days} days "
            f"on domain {dataset.domain!r} ({len(dataset)} samples available)",
            started,
        )
        result = run_rollout(
            module,
            cfg,
            dataset,
            forecasters,
            spec,
            directory,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            statistics=statistics.to_dict(),
        )
    timings["rollout"] = result.seconds
    spent["rollout"] = time.time() - t0

    for key in result.metrics:
        print(f"  loss[{key}] = {result.losses.get(key, float('nan')):.4f}", flush=True)

    # -- the long free-running rollout --------------------------------------
    free = None
    if not args.skip_free_rollout and args.free_days > 0:
        t0 = time.time()
        free_directory = cache_dir_for(evalstore, run_label, f"free{args.free_days}d")
        free_dataset = build_dataset(cfg, args.domain, args.free_days)
        free_inits = choose_initialisations(len(free_dataset), args.n_free_inits, "spread")
        free_spec = RolloutSpec.with_checkpoint(
            identity,
            statistics_digest=statistics_digest,
            experiment=args.exp,
            domain=str(free_dataset.domain),
            lead_days=int(args.free_days),
            n_inits=len(free_inits),
            selection="spread",
            save_depths="shallow",
            save_fields=True,
            inits=tuple(int(i) for i in free_inits),
        )
        free = None if args.force else load_cached(free_spec, free_directory)
        if free is None:
            announce(f"Free-running rollout: {args.free_days} days", started)
            free = run_rollout(
                module,
                cfg,
                free_dataset,
                forecasters,
                free_spec,
                free_directory,
                batch_size=1,
                # A 90-day sample is 90 stacked states, ~1.4 GB in the worker
                # before it is even collated; more than a couple of prefetching
                # workers is a straightforward way to run the node out of RAM.
                num_workers=min(args.num_workers, 2),
                device=device,
                statistics=statistics.to_dict(),
            )
        else:
            announce(f"Reusing cached free rollout in {free_directory}", started)
        timings["free_rollout"] = free.seconds
        spent["free_rollout"] = time.time() - t0

    # -- figures -------------------------------------------------------------
    figures: list[Path] = []
    if not args.skip_figures:
        from . import plots

        t0 = time.time()
        announce("Rendering figures", started)
        figure_dir = out_dir / "figures"
        # `--compare-with` reads *another* run's cache directory, which this key
        # cannot see change, so those figures are never served from a cache.
        comparing = bool(args.compare_with)
        figure_key = render_cache.render_key(
            "figures",
            spec,
            free_spec=free.spec if free is not None else None,
            forecasters=result.metrics,
            dpi=args.dpi,
            skip_spectra=bool(args.skip_spectra),
        )
        figures = cached_or_render(
            "figures",
            figure_dir,
            figure_key,
            lambda: (
                plots.render_all(
                    result,
                    free=free,
                    out_dir=figure_dir,
                    dpi=args.dpi,
                    skip_spectra=args.skip_spectra,
                )
                + render_comparison(args, result, figure_dir, evalstore)
            ),
            force=args.force or comparing,
        )
        timings["figures"] = spent["figures"] = time.time() - t0

    # -- animations ----------------------------------------------------------
    animations: list[Path] = []
    if not args.skip_animations:
        from . import animate

        t0 = time.time()
        announce("Rendering animations", started, "the most expensive stage of the run")
        animation_dir = out_dir / "animations"
        animation_key = render_cache.render_key(
            "animations",
            spec,
            free_spec=free.spec if free is not None else None,
            forecasters=result.metrics,
            fps=args.fps,
            dpi=args.anim_dpi,
        )
        animations = cached_or_render(
            "animations",
            animation_dir,
            animation_key,
            lambda: animate.render_all(
                result,
                free=free,
                out_dir=animation_dir,
                fps=args.fps,
                dpi=args.anim_dpi,
            ),
            force=args.force,
        )
        timings["animations"] = spent["animations"] = time.time() - t0

    # -- report --------------------------------------------------------------
    # The wall clock the report will quote: everything up to the report itself.
    timings["total"] = time.time() - started
    # The statistics that produced these numbers, which for a reused cache is
    # not necessarily what is in oceanarches/stats/ now.
    scored_with = provenance_module.StatisticsProvenance.from_dict(result.statistics)
    if not scored_with.known:
        scored_with = statistics
    summary = {
        "experiment": args.exp,
        "checkpoint": checkpoint,
        "domain": spec.domain,
        "lead_days": args.lead_days,
        "n_inits": len(inits),
        "init_selection": args.init_selection,
        "losses": result.losses,
        "statistics": scored_with.to_dict(),
        "timings": timings,
        "figures": [str(p) for p in figures],
        "animations": [str(p) for p in animations],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if not args.skip_report:
        from . import report as report_module

        announce("Writing the report", started)
        t0 = time.time()
        written = report_module.write_report(
            result,
            free=free,
            module=module,
            out_dir=out_dir,
            figures=figures,
            animations=animations,
            timings=timings,
            checkpoint=checkpoint,
            statistics=scored_with,
        )
        spent["report"] = time.time() - t0
        summary["reports"] = [str(p) for p in written]
        for path in written:
            print(f"Wrote {path}", flush=True)

    # Written last, so that `total` is the whole run and not the part of it that
    # happened to precede the report. The report itself quotes the wall clock up
    # to the moment it was built, and says so.
    timings["total"] = time.time() - started
    summary["spent"] = spent
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone in {timings['total']:.1f}s: " + stage_breakdown(spent), flush=True)
    print(next_time_hint(args, spent), flush=True)
    for line in cost_lines(out_dir, evalstore):
        print(line, flush=True)
    if scored_with.sampled:
        print(
            "The report carries a warning: these numbers were produced with sampled "
            "statistics. See the top of report.md.",
            flush=True,
        )
    return summary


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oceanarches.evaluation.run_eval",
        description="Score a trained checkpoint against persistence and climatology, "
        "render the figures and animations, and write a self-contained report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        default=None,
        help="Run name under modelstore/, or a path to a directory holding "
        "config.yaml and checkpoints/. Required unless --coupled, which names "
        "its output directory after the components instead.",
    )
    coupling = parser.add_argument_group(
        "coupling",
        "Score several trained components run together. The coupled system exposes the "
        "same interface as one model, so every option above applies to it unchanged.",
    )
    coupling.add_argument(
        "--coupled",
        action="store_true",
        help="Build the model from --components instead of loading one checkpoint.",
    )
    coupling.add_argument(
        "--components",
        nargs="+",
        default=None,
        metavar="NAME=RUN",
        help="Components to couple, e.g. `ocean=my_ocean_run seaice=my_ice_run`. NAME "
        "is the COMPONENTS entry in variables.py that the checkpoint was trained as.",
    )
    coupling.add_argument(
        "--mode",
        choices=COUPLING_MODES,
        default="sequential",
        help="'sequential': components run in --order and later ones see the fields "
        "earlier ones already updated. 'parallel': everyone sees time t.",
    )
    coupling.add_argument(
        "--order",
        nargs="+",
        default=None,
        help="Order the components run in. Defaults to the order --components lists "
        "them, i.e. ocean before sea ice.",
    )
    coupling.add_argument(
        "--unpredicted-forcing",
        choices=UNPREDICTED_FORCING_MODES,
        default="persistence",
        help="What happens to shared-state channels no component predicts. "
        "'persistence' holds them at the initial value (free-running); "
        "'ground_truth' reads them from the dataset at each valid time "
        "(perfect forcing). Never zeroed.",
    )
    parser.add_argument(
        "--compare-with",
        nargs="+",
        default=None,
        metavar="LABEL=RUN",
        help="Overlay previously scored runs on this one's curves. They must have been "
        "scored on the same --domain, --lead-days, --n-inits and --init-selection.",
    )
    parser.add_argument("--lead-days", type=int, default=10, help="Rollout length, in days.")
    parser.add_argument(
        "--n-inits",
        type=int,
        default=16,
        help="Initial conditions to score. More is a steadier number and a longer run.",
    )
    parser.add_argument(
        "--init-selection",
        choices=("spread", "first"),
        default="spread",
        help="'spread' walks the whole period; 'first' takes dataset order from the start.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Split to score on. Defaults to the run's own test_args.domain.",
    )
    parser.add_argument("--free-days", type=int, default=90, help="Free-running rollout length.")
    parser.add_argument(
        "--n-free-inits", type=int, default=1, help="Initial conditions for the free rollout."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-depths",
        choices=("shallow", "all"),
        default="shallow",
        help="'shallow' writes the surface fields and the shallowest level "
        "(what the figures need); 'all' writes every depth and is ~13x larger.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution.")
    parser.add_argument("--anim-dpi", type=int, default=100, help="Animation resolution.")
    parser.add_argument("--fps", type=int, default=6, help="Animation frame rate.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output ROOT, not the run directory: everything lands in "
        "<out>/<run>/, where <run> is --exp or the coupled system's generated "
        "name. `--out evalstore/arm1` therefore gives evalstore/arm1/<run>/, "
        "which is what keeps two arms of the same experiment apart.",
    )
    parser.add_argument("--model-label", default=None, help="Legend label for the model curve.")
    parser.add_argument(
        "--force", action="store_true", help="Recompute the rollouts even if a cache matches."
    )
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-animations", action="store_true")
    parser.add_argument("--skip-free-rollout", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-spectra", action="store_true")
    parser.add_argument(
        "--skip-fields",
        action="store_true",
        help="Do not write predictions.zarr / targets.zarr (metrics only).",
    )
    return parser


def default_experiment_name(
    order: Sequence[str], mode: str, unpredicted_forcing: str = "persistence"
) -> str:
    """``coupled_ocean_seaice_sequential`` -- readable, and different per wiring.

    The wiring is in the *name*, not only in the cache key, because two wirings
    of the same checkpoints are two different forecasts: sharing one output
    directory would leave a report whose figures and numbers came from different
    runs. The forcing policy is appended only when it is not the default, so the
    common case stays short.

    Args:
        order: the components in the order they run -- ``order``, not the order
            ``--components`` happened to list them in. ``--order seaice ocean``
            is a different forecast from the default and gets a different
            directory; taking the name from the component *set* would have let
            the two overwrite each other's figures and report, which is the one
            thing this function exists to prevent.
    """
    parts = list(order) + [mode]
    if unpredicted_forcing != "persistence":
        parts.append(unpredicted_forcing)
    return "coupled_" + "_".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.coupled:
        components = parse_components(args.components)
        if not components:
            raise SystemExit(
                "--coupled needs --components NAME=RUN [NAME=RUN ...], e.g. "
                "--components ocean=my_ocean_run seaice=my_ice_run"
            )
        order = list(args.order or components)
        if sorted(order) != sorted(components):
            raise SystemExit(f"--order {order} does not name the components {sorted(components)}.")
        args.exp = args.exp or default_experiment_name(order, args.mode, args.unpredicted_forcing)
    elif not args.exp:
        raise SystemExit("--exp is required (or use --coupled --components ...).")
    if args.skip_baselines:
        print(
            "WARNING: --skip-baselines. A forecast error with nothing beside it "
            "does not say whether the model is any good.",
            file=sys.stderr,
        )
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
