"""Rolling a forecaster out, scoring it, and caching the result.

This is the part of the pipeline that touches the GPU.  Everything downstream --
figures, animations, the report -- reads what this module wrote, so re-rendering
a figure never re-runs the model.

The scoring loop is deliberately boring, and there is only one of it.  The model
and both baselines are :class:`~oceanarches.evaluation.baselines.Forecaster`
objects; the loop denormalises with the module's own statistics, updates the
*same* metric classes the Lightning module logs during validation, and never
special-cases which forecaster it is holding.  That is what makes the comparison
apples to apples, and it is what lets the pipeline reproduce the training run's
validation numbers exactly -- pinned by
``tests/test_evaluation.py::test_pipeline_agrees_with_the_lightning_modules_own_validation``.

Cache layout, under ``evalstore/<experiment>/``::

    lead10d/manifest.json          what produced this cache
    lead10d/metrics_model_glorys_deterministic_metrics.nc
    lead10d/metrics_persistence_...nc
    lead10d/predictions.zarr       model rollout, physical units, land NaN
    lead10d/targets.zarr           ground truth over the same window
    free90d/...                    the long free-running rollout

The manifest records the experiment, domain, lead time, initialisation indices,
whether the fields were written, how deeply ``oceanarches/stats/`` was sampled
when these numbers were produced, and -- the part that is easy to get wrong --
the *identity* of the checkpoint and of the statistics rather than their file
names: both are hashed by content, because retraining a model and rebuilding the
statistics both leave the file names untouched and change every number.  A cache whose manifest
does not match the requested run is recomputed rather than reused: a stale cache
that silently answers a different question is worse than no cache at all.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from geoarches.dataloaders.zarr import ZarrIterativeWriter
from geoarches.metrics.label_wrapper import convert_metric_dict_to_xarray
from hydra.utils import instantiate

from .. import paths
from .baselines import Forecaster, targets_for

__all__ = [
    "CheckpointIdentity",
    "checkpoint_identity",
    "statistics_fingerprint",
    "RolloutSpec",
    "RolloutResult",
    "collate_fn",
    "choose_initialisations",
    "build_metrics",
    "score_batch",
    "run_rollout",
    "load_cached",
    "cache_dir_for",
    "writer_view",
]


#: geoarches' own collate: our samples are TensorDicts, which ``torch.stack``
#: handles and ``default_collate`` does not.
def collate_fn(samples: list[dict]) -> dict:
    return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}


# ---------------------------------------------------------------------------
# What a rollout is
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointIdentity:
    """What ``load_module`` will actually load, not merely what it is called."""

    #: File name of the checkpoint, for the report and for a human reading a manifest.
    name: str
    #: ``"<size>:<short sha-256 of the contents>"``, or ``"missing"``.
    fingerprint: str
    #: Short SHA-256 of ``config.yaml``, or ``"missing"``.
    config_hash: str


def _digest(path: Path) -> str:
    """Short SHA-256 of a file, read in 4 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def checkpoint_identity(experiment: str | Path) -> CheckpointIdentity:
    """Identify the checkpoint and config ``load_module`` will pick for a run.

    **A checkpoint file name is not a cache key.**  geoarches' checkpoint
    callback names its files after the global step, so retraining a run with the
    same ``max_steps`` overwrites ``checkpoint_global_step=4500.ckpt`` with
    different weights under the identical name.  Keying the cache on the name
    alone means ``make eval NAME=my_run`` silently reports the *previous* model's
    scores -- the ordinary hackathon workflow, and the worst possible failure for
    a pipeline whose whole job is to say whether a model got better.

    So the cache is keyed on the checkpoint's *contents*: its size and a SHA-256
    of the bytes, plus a hash of ``config.yaml``, which covers everything else
    ``load_module`` reads.  A modification time would be cheaper, but it is both
    too weak and too strong -- too weak because a file system whose timestamps
    are coarser than a write can leave it unchanged, and too strong because
    copying or touching a checkpoint would throw a good cache away.  Hashing this
    project's 164 MB checkpoint takes 0.26 s against a 270 s evaluation, and the
    size alone catches nothing: every checkpoint of a given model has it.

    Mirrors geoarches' own choice of checkpoint: the newest by modification time
    (``BaseLightningModule.init_from_ckpt``), under ``modelstore/<experiment>/``
    or under ``<experiment>/`` when that is a path.

    Args:
        experiment: run name under ``modelstore/``, or a path to a directory
            holding ``config.yaml`` and ``checkpoints/``.

    Returns:
        A :class:`CheckpointIdentity`.  Missing files give ``"missing"`` rather
        than raising, because this function is only ever reached after
        :func:`oceanarches.evaluation.run_eval.resolve_run` has checked the run
        exists and has a checkpoint.  It is that function, not ``load_module``,
        that produces the readable message -- ``load_module`` raises
        ``FileNotFoundError`` on a path with the ``modelstore/`` prefix dropped.
    """
    root = Path("modelstore") / str(experiment)
    if not root.exists():
        root = Path(experiment)

    checkpoints = sorted((root / "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        name = checkpoints[-1].name
        fingerprint = f"{checkpoints[-1].stat().st_size}:{_digest(checkpoints[-1])}"
    else:
        name, fingerprint = "unknown", "missing"

    config = root / "config.yaml"
    config_hash = _digest(config) if config.exists() else "missing"
    return CheckpointIdentity(name=name, fingerprint=fingerprint, config_hash=config_hash)


#: The generated artefacts every scored number depends on.  The masks decide
#: which cells are scored, the normalisation statistics decide what the model
#: reads and what "physical units" means, and the climatology *is* one of the two
#: baselines.
STATISTICS_ARTEFACTS = ("masks_file", "stats_file", "climatology_file")


def statistics_fingerprint(files: Sequence[Path] | None = None) -> str:
    """``"3:<short sha-256>"`` over ``oceanarches/stats/``, for the cache key.

    **Contents, for the same reason the checkpoint is keyed on its contents.**
    This project already paid for the other answer once: the cache was keyed on
    the checkpoint's *file name*, so retraining a run under the same name served
    the previous model's scores -- 1.0167 reported as 0.7053 -- and it read as a
    result rather than as a bug.

    Rebuilding the statistics is the same shape of mistake and was until now
    unguarded: a participant runs ``make stats-quick`` to save eight minutes,
    evaluates, later runs the full ``make stats`` because they now want to report
    the numbers, re-runs ``make eval``, and gets the sampled baseline back
    because the cache still matched.  A printed warning is not a guard -- the
    warning is read by the people who did not need it.

    Metadata would not do here either: ``--quick`` and a full build differ in
    every number in the file, and two builds can carry the identical ``years``
    label over a changed archive.  The bytes are the only honest key.

    Measured at 0.06 s for the three files -- 95 MB of it the climatology --
    against a 250 s evaluation, paid once per run.  The same trade the checkpoint
    hash already makes, and an order of magnitude cheaper than it.

    Args:
        files: the artefacts to hash.  Defaults to the three
            :mod:`oceanarches.paths` accessors named in
            :data:`STATISTICS_ARTEFACTS`.

    Returns:
        A short digest over ``(name, size, contents)`` of each artefact.  A
        missing file contributes ``"missing"`` rather than raising: a spec is
        built before anything has checked the artefacts exist, and "the
        statistics were absent" is itself a distinguishing fact about a cache.
    """
    if files is None:
        files = [getattr(paths, name)() for name in STATISTICS_ARTEFACTS]
    digest = hashlib.sha256()
    for file in files:
        file = Path(file)
        digest.update(file.name.encode())
        if file.is_file():
            digest.update(f"{file.stat().st_size}:{_digest(file)}".encode())
        else:
            digest.update(b"missing")
    return f"{len(list(files))}:{digest.hexdigest()[:16]}"


@dataclass(frozen=True)
class RolloutSpec:
    """Everything that changes the numbers, and therefore invalidates a cache."""

    experiment: str
    domain: str
    lead_days: int
    n_inits: int
    selection: str  # "spread" or "first"
    save_depths: str  # "shallow" or "all"
    checkpoint: str  # file name -- for humans; NOT enough to key a cache on
    inits: tuple[int, ...] = ()
    #: Size and content hash of that file: what makes retrained weights a
    #: different run even when they land under the identical name.
    checkpoint_fingerprint: str = ""
    #: Hash of ``config.yaml``: the rest of what ``load_module`` reads.
    config_hash: str = ""
    #: Digest of ``oceanarches/stats/`` -- masks, normalisation statistics and
    #: climatology.  Every scored number depends on all three, and the
    #: climatology is itself one of the two baselines, so rebuilding them makes
    #: this a different question and must not be answered from the old cache.
    statistics_fingerprint: str = ""
    #: Whether ``predictions.zarr`` / ``targets.zarr`` were written.  A cache
    #: built with ``--skip-fields`` cannot answer a request that needs the maps.
    save_fields: bool = True

    @classmethod
    def with_checkpoint(
        cls,
        identity: CheckpointIdentity,
        statistics_digest: str | None = None,
        **kwargs,
    ) -> "RolloutSpec":
        """Build a spec from a :class:`CheckpointIdentity`, so no caller can
        record the file name and forget the fields that make it a key.

        ``statistics_digest`` defaults to hashing ``oceanarches/stats/``
        rather than to ``""``: forgetting it is the failure this guards against,
        so the safe value is the one a caller gets by saying nothing.  Pass it
        explicitly to reuse one digest across the two specs of a run.
        """
        return cls(
            checkpoint=identity.name,
            checkpoint_fingerprint=identity.fingerprint,
            config_hash=identity.config_hash,
            statistics_fingerprint=(
                statistics_fingerprint() if statistics_digest is None else statistics_digest
            ),
            **kwargs,
        )

    def as_manifest(self) -> dict:
        return {
            "experiment": self.experiment,
            "domain": self.domain,
            "lead_days": self.lead_days,
            "n_inits": self.n_inits,
            "selection": self.selection,
            "save_depths": self.save_depths,
            "checkpoint": self.checkpoint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "config_hash": self.config_hash,
            "statistics_fingerprint": self.statistics_fingerprint,
            "save_fields": self.save_fields,
            "inits": list(self.inits),
        }

    def matches(self, manifest: dict) -> bool:
        mine = self.as_manifest()
        return all(manifest.get(k) == v for k, v in mine.items())

    @classmethod
    def from_manifest(cls, manifest: dict) -> "RolloutSpec":
        """The exact reverse of :meth:`as_manifest`: rebuild a spec from a saved
        manifest, so nothing reading a cache back has to open-code its own copy
        of the field list.

        This is the fix for a bug that already happened once:
        ``statistics_fingerprint`` was added to this class, and the one
        reconstruction that used a hand-written list of manifest keys (a
        notebook) simply did not gain it -- its cache reload started failing
        with ``load_cached`` refusing a perfectly good cache, silently, because
        the rebuilt spec could never match its own manifest. Every caller should
        use this instead of typing the field list out.

        The field list is read off the dataclass itself with
        :func:`dataclasses.fields`, not copied by hand, so the next field this
        class gains is picked up here automatically.

        A field with **no default on the class** -- the run's identity:
        ``experiment``, ``domain``, ``lead_days``, ``n_inits``, ``selection``,
        ``save_depths``, ``checkpoint`` -- has no sensible fallback, and every
        manifest this project has ever written has always recorded them, so a
        manifest missing one of those raises rather than guessing.

        A field **with** a default (``checkpoint_fingerprint``,
        ``config_hash``, ``statistics_fingerprint``, ``save_fields``,
        ``inits``) falls back to that same class default when the key is
        absent -- exactly what reading an *old* manifest, written before the
        field existed, needs. That is not a loophole a stale cache can walk
        through: :meth:`matches` compares the rebuilt spec's value against
        ``manifest.get(name)``, which is ``None`` for a key that was never
        written, and a real default (``""``, ``True``, ``()``) is never equal
        to ``None``. So a manifest that cannot say still fails the match, on
        the same terms every other incomparable cache does -- see
        :func:`statistics_fingerprint` for why that has to be true rather than
        merely usual.

        Args:
            manifest: a ``manifest.json`` already parsed to a dict (or any
                dict shaped like :meth:`as_manifest`'s output).

        Returns:
            A :class:`RolloutSpec` equal to the one that would reproduce this
            exact manifest via :meth:`as_manifest`.

        Raises:
            KeyError: naming the field(s) with no default that the manifest
                does not record.
        """
        values: dict = {}
        missing: list[str] = []
        for f in fields(cls):
            if f.name in manifest:
                value = manifest[f.name]
            elif f.default is not MISSING:
                value = f.default
            else:
                missing.append(f.name)
                continue
            values[f.name] = tuple(value) if f.name == "inits" else value
        if missing:
            raise KeyError(
                f"manifest is missing required field(s) {sorted(missing)}, which have no "
                "default on RolloutSpec -- this is not a valid rollout manifest, or it is "
                "damaged rather than merely old."
            )
        return cls(**values)


@dataclass
class RolloutResult:
    """Scores and file paths for one rollout, for every forecaster."""

    spec: RolloutSpec
    directory: Path
    #: ``{forecaster_key: {metric_group_name: xr.Dataset}}``
    metrics: dict[str, dict[str, xr.Dataset]] = field(default_factory=dict)
    #: ``{forecaster_key: mean loss}`` -- the module's own training loss.
    losses: dict[str, float] = field(default_factory=dict)
    #: ``{forecaster_key: legend label}``
    labels: dict[str, str] = field(default_factory=dict)
    #: Initial times of the scored samples, ISO strings.
    init_times: list[str] = field(default_factory=list)
    #: Seconds spent in the rollout itself.
    seconds: float = 0.0
    n_samples: int = 0
    #: How deeply ``oceanarches/stats/`` was sampled when these numbers were
    #: produced, as :class:`oceanarches.evaluation.provenance.StatisticsProvenance`
    #: records it.  Informational, and deliberately **not** part of
    #: :class:`RolloutSpec`: it must not invalidate a cache, but a report built
    #: from a cache has to describe the statistics that made those numbers
    #: rather than whatever is in ``oceanarches/stats/`` today.
    statistics: dict = field(default_factory=dict)

    @property
    def predictions_path(self) -> Path:
        return self.directory / "predictions.zarr"

    @property
    def targets_path(self) -> Path:
        return self.directory / "targets.zarr"

    def open_predictions(self) -> xr.Dataset:
        return xr.open_zarr(self.predictions_path)

    def open_targets(self) -> xr.Dataset:
        return xr.open_zarr(self.targets_path)

    def deterministic(self, key: str) -> xr.Dataset | None:
        return self.metrics.get(key, {}).get("glorys_deterministic_metrics")

    def seaice(self, key: str) -> xr.Dataset | None:
        return self.metrics.get(key, {}).get("glorys_seaice_metrics")

    @property
    def forecaster_keys(self) -> list[str]:
        return list(self.metrics)


def cache_dir_for(root: Path, experiment: str, tag: str) -> Path:
    """``evalstore/<experiment>/<tag>`` -- the cache key the brief asks for."""
    return Path(root) / experiment / tag


# ---------------------------------------------------------------------------
# Choosing initial conditions
# ---------------------------------------------------------------------------
def choose_initialisations(n_available: int, n_inits: int, selection: str) -> list[int]:
    """Sample indices to initialise from.

    ``spread`` walks the whole period at even spacing, so the score is not a
    statement about one season.  ``first`` takes them in dataset order from the
    start, which is what makes a run reproducible against a fixed reference set
    (the Task 6 numbers are "the first 128 samples of ``tiny_val``").

    Args:
        n_available: ``len(dataset)``.
        n_inits: how many to score.  Clipped to what exists.
        selection: ``"spread"`` or ``"first"``.
    """
    if n_available <= 0:
        raise ValueError("The dataset has no usable samples for this domain and lead time.")
    n = min(int(n_inits), n_available)
    if selection == "first":
        return list(range(n))
    if selection != "spread":
        raise ValueError(f"selection must be 'spread' or 'first', got {selection!r}")
    if n == 1:
        return [0]
    return sorted(set(np.linspace(0, n_available - 1, n).round().astype(int).tolist()))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def build_metrics(cfg, iters: int, device) -> dict[str, torch.nn.Module]:
    """One fresh copy of every configured inference metric.

    Taken from ``cfg.module.inference.metrics`` -- the same entries the Lightning
    module builds its ``test_metrics`` from -- with ``rollout_iterations``
    overridden to the requested lead time and ``headline_only`` forced off, so
    every variable and every depth is scored rather than the handful worth
    watching during training.
    """
    kwargs = dict(cfg.module.inference.metrics_kwargs)
    kwargs["rollout_iterations"] = int(iters)
    kwargs["headline_only"] = False
    return {
        name: instantiate(metric_cfg, **kwargs).to(device)
        for name, metric_cfg in cfg.module.inference.metrics.items()
    }


def _metric_to_xarray(labelled: dict[str, torch.Tensor]) -> xr.Dataset:
    plain = {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in labelled.items()}
    return convert_metric_dict_to_xarray(plain, ["prediction_timedelta"])


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------
def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def writer_view(dataset, module):
    """A view of the dataset that describes what the module **predicts**.

    ``convert_to_xarray`` names the channels it is handed from the dataset's own
    variable lists and masks them with the dataset's own ``state_mask``.  For a
    *component* those describe what the model reads -- 7 surface channels for the
    ocean specialist -- while a prediction carries only what it predicts, which
    is 3.  Writing one through the other used to die mid-rollout with

        RuntimeError: The size of tensor a (7) must match the size of tensor b (3)

    after several minutes of GPU time, and ``make eval`` has no way to opt out.
    ``OceanForecastModule.test_step`` guards the identical case; this is the same
    guard turned into the thing it should have done instead.

    A shallow copy is enough and is deliberately cheap: ``convert_to_xarray``
    reads only the variable lists, the cached mask, the depths and the
    coordinates, and touches no file, so nothing here re-opens the archive or
    disturbs the dataset the loader workers are using.

    Args:
        dataset: the ``GlorysForecast`` the rollout is running over.
        module: the forecast module (or coupled system) being scored.

    Returns:
        ``dataset`` itself when the model predicts everything it reads (the
        ``full`` component, and every coupled system that owns the whole state),
        otherwise a view restricted to the model's prognostic variables.
    """
    surface = list(getattr(module, "surface_variables", dataset.surface_variables))
    level = list(getattr(module, "level_variables", dataset.level_variables))
    if surface == dataset.surface_variables and level == dataset.level_variables:
        return dataset

    # The dataloader contract is prognostic channels first (tests/test_configs.py
    # checks it), which is what makes the leading channels of a sample the ones
    # the model predicts. If that ever stopped holding, the fields would be
    # written to disk under the wrong names -- a file that looks fine and is
    # labelled wrong -- so say so instead.
    for group, wanted, available in (
        ("surface", surface, dataset.surface_variables),
        ("level", level, dataset.level_variables),
    ):
        if wanted != available[: len(wanted)]:
            raise ValueError(
                f"The model predicts the {group} variables {wanted}, which are not the "
                f"leading channels of the {group} variables the dataset loads "
                f"({available}). The predictions would be written to disk under the wrong "
                "names. Order the dataloader's variables prognostic-first, or re-run with "
                "--skip-fields to score without writing the fields."
            )

    view = copy.copy(dataset)
    view.surface_variables = surface
    view.level_variables = level
    view._state_mask = None  # rebuilt on demand for exactly these variables
    return view


def _write_trajectory(writer, dataset, physical_state, timestamps, levels) -> None:
    xr_dataset = dataset.convert_trajectory_to_xarray(
        physical_state.cpu(), timestamp=timestamps.cpu(), denormalize=False, levels=levels
    )
    writer.write(xr_dataset, append_dim="time")


def score_batch(
    module,
    forecasters: Sequence[Forecaster],
    batch: dict,
    iters: int,
    metrics: dict[str, dict[str, object]],
    losses: dict[str, float],
    on_model_prediction=None,
) -> object:
    """Score every forecaster on one batch, through one code path.

    Factored out of :func:`run_rollout` so it can be tested without a GPU, a
    checkpoint or hydra -- and because it is the single most important property
    of this pipeline: **the target tensor is built once, denormalised once, and
    the identical object is handed to every forecaster's metrics.** There is no
    branch on which forecaster is being scored, so the model cannot be measured
    against a different truth from its baselines.

    Args:
        module: the forecast module (statistics, masks, loss).
        forecasters: model and baselines.
        batch: one collated sample dict, already on the right device.
        iters: rollout length.
        metrics: ``{forecaster key: {metric name: metric}}``; updated in place.
        losses: ``{forecaster key: running sum}``; updated in place, weighted by
            the batch size so the caller can divide by the sample count.
        on_model_prediction: optional callback given the *model's* denormalised
            prediction, used to stream it to the zarr writer.

    Returns:
        The denormalised targets, so the caller can write them out too.
    """
    n_batch = int(batch["timestamp"].shape[0])
    targets = targets_for(module, batch, iters)
    targets_physical = module.denormalize_state(targets)

    for forecaster in forecasters:
        predictions = forecaster.predict(batch, iters)
        losses[forecaster.key] = losses.get(forecaster.key, 0.0) + (
            float(module.loss(predictions, targets, multistep=True)) * n_batch
        )
        predictions_physical = module.denormalize_state(predictions)
        for metric in metrics.get(forecaster.key, {}).values():
            metric.update(targets_physical, predictions_physical, timestamp=batch["timestamp"])
        if forecaster.is_model and on_model_prediction is not None:
            on_model_prediction(predictions_physical)
        del predictions, predictions_physical
    return targets_physical


def run_rollout(
    module,
    cfg,
    dataset,
    forecasters: Sequence[Forecaster],
    spec: RolloutSpec,
    directory: Path,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "cuda",
    progress: bool = True,
    statistics: dict | None = None,
) -> RolloutResult:
    """Roll every forecaster out over ``spec.inits`` and score all of them.

    Args:
        module: the loaded forecast module (also supplies the statistics, the
            wet masks and the loss).
        cfg: the run config, for the metric definitions.
        dataset: a ``GlorysForecast`` built with ``multistep = lead_days``.
        forecasters: the model first, then the baselines.
        spec: what is being run, including whether to write the fields; written
            to the cache manifest.
        directory: cache directory, created if missing.
        batch_size, num_workers: dataloader settings.
        progress: print one line per batch.
        statistics: provenance of ``oceanarches/stats/``, recorded in the
            manifest so a later report can say what produced these numbers.

    Returns:
        A :class:`RolloutResult` whose metric datasets are also on disk.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    iters = int(spec.lead_days)
    levels = None if spec.save_depths == "all" else [dataset.depths[0]]

    subset = torch.utils.data.Subset(dataset, list(spec.inits))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    metrics = {f.key: build_metrics(cfg, iters, device) for f in forecasters}
    losses = {f.key: 0.0 for f in forecasters}

    writers = {}
    # Both the predictions and the targets carry the *model's* channels
    # (`targets_for` selects them), so both are written through this view.
    fields_dataset = writer_view(dataset, module) if spec.save_fields else dataset
    if spec.save_fields:
        for name in ("predictions", "targets"):
            path = directory / f"{name}.zarr"
            if path.exists():
                shutil.rmtree(path)
            writers[name] = ZarrIterativeWriter(path, force=True)

    # A context manager, not `torch.set_grad_enabled(False)`: that call is
    # PROCESS-WIDE and has no matching restore, so a notebook that rolled a model
    # out and then trained one got zero gradients and a loss that never moved,
    # with nothing to see. `no_grad` puts it back however this function exits.
    with torch.no_grad():
        return _rollout_loop(
            module=module,
            loader=loader,
            subset=subset,
            spec=spec,
            directory=directory,
            forecasters=forecasters,
            metrics=metrics,
            losses=losses,
            iters=iters,
            levels=levels,
            writers=writers,
            fields_dataset=fields_dataset,
            progress=progress,
            device=device,
            statistics=dict(statistics or {}),
        )


def _rollout_loop(
    *,
    module,
    loader,
    subset,
    spec,
    directory,
    forecasters,
    metrics,
    losses,
    iters,
    levels,
    writers,
    fields_dataset,
    progress,
    device,
    statistics=None,
):
    """The body of :func:`run_rollout`, inside its ``no_grad`` block."""
    n_samples = 0
    init_times: list[str] = []
    start = time.time()
    for batch_index, batch in enumerate(loader):
        batch = _to_device(batch, device)
        n_batch = int(batch["timestamp"].shape[0])
        init_times.extend(
            np.asarray(batch["timestamp"].cpu().numpy(), dtype="int64")
            .astype("datetime64[s]")
            .astype(str)
            .tolist()
        )

        def write_predictions(predictions_physical, _batch=batch):
            if writers:
                _write_trajectory(
                    writers["predictions"],
                    fields_dataset,
                    predictions_physical,
                    _batch["timestamp"],
                    levels,
                )

        targets_physical = score_batch(
            module,
            forecasters,
            batch,
            iters,
            metrics,
            losses,
            on_model_prediction=write_predictions,
        )
        if writers:
            _write_trajectory(
                writers["targets"], fields_dataset, targets_physical, batch["timestamp"], levels
            )
        n_samples += n_batch
        del targets_physical
        if progress:
            print(
                f"  batch {batch_index + 1}/{len(loader)}  "
                f"({n_samples}/{len(subset)} initialisations, {time.time() - start:.1f}s)",
                flush=True,
            )
    seconds = time.time() - start

    result = RolloutResult(
        spec=spec,
        directory=directory,
        losses={k: v / max(n_samples, 1) for k, v in losses.items()},
        labels={f.key: f.label for f in forecasters},
        init_times=init_times,
        seconds=seconds,
        n_samples=n_samples,
        statistics=dict(statistics or {}),
    )
    for key, group in metrics.items():
        result.metrics[key] = {}
        for name, metric in group.items():
            labelled = metric.compute()
            if not labelled:
                continue
            xr_dataset = _metric_to_xarray(labelled)
            xr_dataset.to_netcdf(directory / f"metrics_{key}_{name}.nc")
            result.metrics[key][name] = xr_dataset

    manifest = spec.as_manifest()
    manifest.update(
        losses=result.losses,
        labels=result.labels,
        init_times=init_times,
        seconds=seconds,
        n_samples=n_samples,
        metric_groups={k: sorted(v) for k, v in result.metrics.items()},
        saved_fields=bool(writers),
        statistics=result.statistics,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result


def load_cached(spec: RolloutSpec, directory: Path) -> RolloutResult | None:
    """Reuse a cache, or None if there is none that answers this exact question."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    if not spec.matches(manifest):
        return None

    result = RolloutResult(
        spec=spec,
        directory=directory,
        losses=manifest.get("losses", {}),
        labels=manifest.get("labels", {}),
        init_times=manifest.get("init_times", []),
        seconds=float(manifest.get("seconds", 0.0)),
        n_samples=int(manifest.get("n_samples", 0)),
        # Absent from every manifest written before this existed, which is why
        # it is read with a default rather than being required: an old cache is
        # still a valid cache, it just cannot say what its statistics were.
        statistics=dict(manifest.get("statistics") or {}),
    )
    for key, names in manifest.get("metric_groups", {}).items():
        result.metrics[key] = {}
        for name in names:
            path = directory / f"metrics_{key}_{name}.nc"
            if path.exists():
                result.metrics[key][name] = xr.load_dataset(path)
    if manifest.get("saved_fields") and not result.predictions_path.exists():
        return None
    return result
