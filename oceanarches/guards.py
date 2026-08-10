"""Startup guards -- the checks that turn an expensive silence into a cheap refusal.

Six people ran this kit as first-time participants.  Every one of their costly
findings was a *silence*, not an error: a short run that trains and saves nothing,
an override that lands on a key nothing reads, a re-run that trains zero steps and
exits 0, a login node that trains on somebody else's GPU until the fork bomb hits.
Each of those printed nothing at all and cost between ten and forty minutes.  This
module says the thing out loud, before the GPU time is spent.

---------------------------------------------------------------------------
Why a hydra callback, and not the Makefile
---------------------------------------------------------------------------
`make train-tiny` runs `geoarches.main_hydra` directly -- not
`oceanarches.main_multinode` -- so a guard written into either entry point covers
only half the routes, and a guard written into the Makefile recipe covers none of
the documented raw command lines (docs/01, docs/05, the notebooks and
scripts/*.slurm all invoke python themselves).  What every one of those routes
*does* share is our `configs/config.yaml`, so the guard is registered there as a
`hydra.callbacks` entry and hydra runs it before the task function on every
route.  `Hydra.run()` calls `on_run_start` immediately after composition and
before `run_job`, so nothing has touched the data, the GPU or `modelstore/` yet.

Two consequences of that placement are load-bearing:

* hydra's `Callbacks._notify` wraps each callback in `except Exception` and
  downgrades it to a `warnings.warn` -- which is precisely the silence this
  module exists to remove.  So a refusal is raised as `SystemExit`, a
  `BaseException`, which `_notify` does not catch and `run_and_report` does not
  swallow.
* the message is printed to **stdout** and only then raised.  Notebook 02 and 03
  capture subprocess stdout and discard stderr, so a guard that spoke only
  through the exception would be invisible in exactly the place beginners read.

`hydra.show_cfg` (`--cfg job`) does not run callbacks, so `make -n` plus
`--cfg job --resolve` -- what tests/test_configs.py uses to check composition --
is unaffected.

---------------------------------------------------------------------------
Escape hatch
---------------------------------------------------------------------------
`OCEANARCHES_SKIP_GUARDS=1` turns every refusal below into a printed warning.  It
is deliberately not advertised in the messages themselves: each refusal names a
one-line fix, and a bypass offered next to the fix is the one people copy.
"""

from __future__ import annotations

import atexit
import difflib
import inspect
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = [
    "SRUN_LINES",
    "SRUN_ONE_LINE",
    "StartupGuard",
    "acquire_run_lock",
    "allocation_warning",
    "already_trained_refusal",
    "concurrent_run_refusal",
    "checkpoint_steps",
    "created_keys",
    "keywords_read_by",
    "latest_checkpoint",
    "oom_advice",
    "no_checkpoint_refusal",
    "override_problems",
    "preset_swap_warning",
    "release_run_lock",
    "resume_line",
    "resumed_architecture_refusal",
    "run_lock_path",
    "skip_guards",
    "srun_hint",
    "warn_allocation",
    "suggest_save_frequency",
]

#: The allocation every GPU target in this kit needs.  Quoted verbatim from
#: configs/cluster/jupiter_1gpu.yaml, including the two flags that are not
#: optional there (`--ntasks=1`, and `CUDA_VISIBLE_DEVICES=0` because the booster
#: hands back all four cards whatever you asked for).
SRUN_LINES = (
    "srun --account=hclimrep --partition=booster --gres=gpu:1 --ntasks=1 \\",
    "     --cpus-per-task=16 --time=01:00:00 --pty bash",
    "export CUDA_VISIBLE_DEVICES=0",
)
#: The same thing on one line, for `make doctor`, whose fix column is one line.
SRUN_ONE_LINE = (
    "srun --account=hclimrep --partition=booster --gres=gpu:1 --ntasks=1 "
    "--cpus-per-task=16 --time=01:00:00 --pty bash  (then: export CUDA_VISIBLE_DEVICES=0)"
)


def srun_hint(indent: str = "      ") -> str:
    """`SRUN_LINES` indented as a block, continuation lines included."""
    return "\n".join(indent + line for line in SRUN_LINES)


#: Top-level keys `geoarches.main_hydra` reads with `hasattr`/`getattr` and that
#: our `configs/config.yaml` therefore does NOT define.  Creating one of these is
#: the documented way to use them (`+load_ckpt=...` in scripts/finetune.slurm), so
#: they are not orphans.
GEOARCHES_OPTIONAL_KEYS = frozenset(
    {
        "load_ckpt",
        "ckpt_filename_match",
        "limit_train_batches",
        "limit_test_batches",
        "profiler",
        "cli_overrides",
    }
)

_STEP_IN_FILENAME = re.compile(r"global_step=(\d+)")


def skip_guards() -> bool:
    """Is `OCEANARCHES_SKIP_GUARDS` set to something truthy?

    Public because the evaluation entry point needs the same answer, and two
    spellings of the same truthiness rule is exactly the drift this module exists
    to prevent -- one of them would eventually accept `0` or reject `yes`.
    """
    return os.environ.get("OCEANARCHES_SKIP_GUARDS", "") not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# 1. The allocation
# ---------------------------------------------------------------------------
def allocation_warning(
    env: dict[str, str] | None = None,
    hostname: str | None = None,
    cluster_name: str | None = "?",
    cpu_hint: str = "cluster=local",
) -> str | None:
    """The complaint to print when a GPU cluster config is being run off SLURM.

    The signal is `SLURM_JOB_ID`, **not** `torch.cuda.is_available()`.  Measured on
    the login node `jpbl-s02-02`: it carries a real GPU at index 0, so CUDA is
    available and `make doctor` used to PASS there, while `nvidia-smi` showed that
    same card at 99% utilisation and 49 GiB in use by another user's job.  CUDA
    visibility answers "is there a card", never "is it yours".

    `cluster=local` is exempt: that config exists to say "no GPU, one worker, one
    sample", and running it on a laptop or a login node is what it is for.
    """
    env = os.environ if env is None else env
    if env.get("SLURM_JOB_ID"):
        return None
    if cluster_name == "local":
        return None
    host = hostname or socket.gethostname()
    # `cluster=` is the training routes' vocabulary; the evaluation entry point
    # has no cluster config and passes None rather than printing `cluster=?`.
    where = f"host {host}" + (f", cluster={cluster_name}" if cluster_name is not None else "")
    return (
        f"no SLURM allocation: SLURM_JOB_ID is unset ({where}).\n"
        "\n"
        "    A visible GPU is not an allocated one. The login nodes carry a real card,\n"
        "    so torch.cuda.is_available() is True there and says nothing about whether\n"
        "    the memory, the process slots or the cores are free -- they are shared with\n"
        "    everyone else logged in. What that costs, measured on this cluster:\n"
        "\n"
        "      RuntimeError: can't start new thread              (dataloader workers)\n"
        "      BlockingIOError: [Errno 11] ... from os.fork      (epoch boundary)\n"
        "      torch.OutOfMemoryError                            (85 of 98 GiB was not yours)\n"
        "\n"
        "    None of those tracebacks mention SLURM, and all three land minutes in.\n"
        "    Get a node of your own first:\n"
        "\n"
        f"{srun_hint()}\n"
        "\n"
        f"    Then re-run this command inside it. `{cpu_hint}` if you really do mean\n"
        "    to run on the CPU here."
    )


def warn_allocation(
    cluster_name: str | None = "?",
    env: dict[str, str] | None = None,
    hostname: str | None = None,
    cpu_hint: str = "cluster=local",
    pause: bool = True,
) -> str | None:
    """Print the allocation warning if there is one, pause, and return what was said.

    The one place that decides *how* a missing allocation is presented, so that
    `make train-*` and `make eval` cannot drift apart -- same signal, same words,
    same pause, same `OCEANARCHES_SKIP_GUARDS` rule.  `StartupGuard.check` and
    `oceanarches.evaluation.run_eval` both go through here.

    A warning and a pause rather than a refusal: `cluster=local` (and the
    evaluation path's `--device cpu`) is exempt above, and there are legitimate
    GPU-less machines.  Ten seconds is enough to read the message and press
    Ctrl-C.  The pause is skipped when stdout is not a tty -- a log file, a
    notebook cell, a queued job -- so nothing can hang on it, and skipped
    entirely when the guards are turned off.

    Args:
        cluster_name: the cluster config selected, `None` for an entry point that
            has none.  `"local"` is exempt.
        env, hostname: injected by the tests; default to the real ones.
        cpu_hint: how *this* entry point spells "I really do mean the CPU".
        pause: set False when the caller has already decided not to wait.

    Returns:
        The complaint printed, or None when there was nothing to say.
    """
    complaint = allocation_warning(
        env=env, hostname=hostname, cluster_name=cluster_name, cpu_hint=cpu_hint
    )
    if complaint is None:
        return None
    _warn(complaint)
    if pause and sys.stdout.isatty() and not skip_guards():
        _emit(["    Continuing in 10 s -- Ctrl-C to stop and get a node first.", ""])
        time.sleep(10)
    return complaint


def _allocation_detail(env: dict[str, str] | None = None, hostname: str | None = None) -> str:
    env = os.environ if env is None else env
    host = hostname or socket.gethostname()
    job = env.get("SLURM_JOB_ID")
    if not job:
        return f"none (SLURM_JOB_ID unset) on {host}"
    return f"SLURM job {job}, {env.get('SLURM_NNODES', '1')} node(s), on {host}"


# ---------------------------------------------------------------------------
# 2. A run that saves no checkpoint
# ---------------------------------------------------------------------------
def checkpoint_steps(max_steps: int, save_step_frequency: int) -> list[int]:
    """The global steps geoarches would actually write a checkpoint at.

    `CheckpointEveryNSteps.on_train_batch_end` fires on
    `trainer.global_step % save_step_frequency == 0` and there is no end-of-fit
    save, so this is the complete list -- and it is empty whenever the run is
    shorter than one save interval.
    """
    if save_step_frequency <= 0 or max_steps <= 0:
        return []
    return list(range(save_step_frequency, max_steps + 1, save_step_frequency))


def suggest_save_frequency(max_steps: int) -> int:
    """A `save_step_frequency` that divides `max_steps` and gives about four checkpoints.

    Four is a judgement call, not a measurement: enough that an interrupted run
    keeps something, few enough that a 700 MB `tiny` checkpoint does not fill a
    quota.  Falls back to `max_steps` itself (one checkpoint, the finished model)
    when no divisor gives eight or fewer -- a prime `max_steps` has none.
    """
    if max_steps <= 0:
        return 1
    divisors = set()
    for candidate in range(1, int(max_steps**0.5) + 1):
        if max_steps % candidate == 0:
            divisors.add(candidate)
            divisors.add(max_steps // candidate)
    usable = [d for d in divisors if max_steps // d <= 8]
    if not usable:
        return max_steps
    # Closest to four checkpoints; on a tie prefer the larger interval, i.e. the
    # smaller number of files.
    return min(usable, key=lambda d: (abs(max_steps // d - 4), -d))


def no_checkpoint_refusal(
    max_steps: int, save_step_frequency: int, preset: str = "?", name: str = "<run>"
) -> str | None:
    """The refusal for a run whose finished model would never reach disk.

    Two cases, both silent today and both measured by a participant:

    * `max_steps` shorter than one save interval -- five runs at 200/250/300
      steps produced `modelstore/<name>/config.yaml` and no `checkpoints/`
      directory at all, while printing normal metrics and
      ``Trainer.fit stopped: max_steps=200 reached``.  `make help`'s own example
      override, `++max_steps=1000`, sits exactly on the `tiny` boundary.
    * `save_step_frequency` not dividing `max_steps` -- checkpoints appear, but
      the last of them is not the finished model and nothing says so.
    """
    if save_step_frequency <= 0:
        return (
            f"save_step_frequency is {save_step_frequency}, so no checkpoint would ever "
            "be written: geoarches' callback fires on "
            "`global_step % save_step_frequency == 0`, which is a ZeroDivisionError "
            "at 0 and never true below it. Pass a positive value."
        )
    steps = checkpoint_steps(max_steps, save_step_frequency)
    if steps and steps[-1] == max_steps:
        return None

    suggestion = suggest_save_frequency(max_steps)
    written = (
        "no checkpoint at all -- not even a `checkpoints/` directory"
        if not steps
        else f"checkpoints at steps {', '.join(str(s) for s in steps[:4])}"
        + (", ..." if len(steps) > 4 else "")
        + f" -- the last at {steps[-1]}, which is NOT the finished model"
    )
    return (
        f"this run would train {max_steps} steps and leave {written}.\n"
        "\n"
        f"      max_steps            {max_steps}\n"
        f"      save_step_frequency  {save_step_frequency}   (from module={preset})\n"
        f"      {max_steps} % {save_step_frequency} = {max_steps % save_step_frequency}\n"
        "\n"
        "    geoarches' CheckpointEveryNSteps writes only when\n"
        "    `global_step % save_step_frequency == 0`, and it writes nothing at the end\n"
        "    of `fit`. So the run would finish, print\n"
        f'    "`Trainer.fit` stopped: `max_steps={max_steps}` reached", look completely\n'
        f"    normal, and `make eval NAME={name}` would be the first thing to tell you\n"
        "    the model is gone.\n"
        "\n"
        "    Fix -- pick a save_step_frequency that DIVIDES max_steps:\n"
        "\n"
        f'      HYDRA_ARGS="++max_steps={max_steps} ++save_step_frequency={suggestion}"\n'
        "\n"
        f"    ({suggestion} gives {max_steps // suggestion} checkpoint(s), the last of them at "
        f"step {max_steps}.)\n"
        "    Or leave max_steps alone and keep the preset's own budget."
    )


# ---------------------------------------------------------------------------
# 3. A re-run that trains nothing
# ---------------------------------------------------------------------------
def latest_checkpoint(exp_dir: str | Path) -> Path | None:
    """The checkpoint geoarches would resume from: newest by mtime, as it sorts them.

    Note that `resume: False` does NOT change this. `main_hydra` picks the
    newest checkpoint in `<exp_dir>/checkpoints/` whenever that directory exists;
    `resume` only decides whether the *config* is taken from the previous run.
    """
    ckpt_dir = Path(exp_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    files = [p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt"]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def checkpoint_step(path: Path | None) -> int | None:
    """The step in `checkpoint_global_step=4000.ckpt`, or None if the name does not say."""
    if path is None:
        return None
    match = _STEP_IN_FILENAME.search(path.name)
    return int(match.group(1)) if match else None


def already_trained_refusal(
    name: str, exp_dir: str | Path, step: int | None, max_steps: int
) -> str | None:
    """The refusal for `make train-tiny NAME=<a run that is already finished>`.

    Measured: `make train-tiny NAME=task6_tiny` loads the shipped checkpoint, sees
    `max_steps` already behind its step count, and **exits 0 having trained and
    saved nothing**.  The participant believes they have trained a model; they
    have a copy of somebody else's.
    """
    if step is None or step < max_steps:
        return None
    # A budget that is BOTH past the existing checkpoint and still a whole number
    # of `max_steps`, so it keeps whatever save_step_frequency divides max_steps
    # and does not trip the no-checkpoint guard two lines later.
    longer = max_steps * (step // max_steps + 1)
    return (
        f"{exp_dir} already holds a checkpoint at step {step}, and max_steps is "
        f"{max_steps}.\n"
        "\n"
        "    Lightning would resume from that checkpoint, find the budget already\n"
        "    spent, stop before the first batch and exit 0 -- training nothing and\n"
        "    saving nothing, while looking exactly like a successful run. That is how\n"
        "    a shipped baseline gets mistaken for your own model.\n"
        "\n"
        "    Pick one:\n"
        "\n"
        "      make train-tiny NAME=my_own_run\n"
        "          -- your own run, trained from scratch\n"
        f'      make train-tiny NAME={name} HYDRA_ARGS="++max_steps={longer}"\n'
        "          -- keep training this one; the budget has to be past the checkpoint\n"
        f"      make eval NAME={name}\n"
        "          -- score the checkpoint that is already there"
    )


# ---------------------------------------------------------------------------
# 4. Overrides that land on nothing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CreatedKey:
    """A `+`/`++` override and the dotted key it would bring into existence."""

    override: str
    key: str
    forced: bool  # True for `++` (add-or-override), False for `+` (add, deliberate)


def created_keys(overrides: Iterable[str], exists: Any) -> list[CreatedKey]:
    """The `+`/`++` overrides whose key does not already exist.

    `exists(key)` answers "was this key in the config *before* the plus overrides".
    Only `+` and `++` can invent a key: a bare `key=value` for an unknown key is
    already a hydra error, which is why `++lr=1e-4` is the spelling that hurts --
    `++` means add-or-override, so a wrong path is created rather than rejected.
    """
    found: list[CreatedKey] = []
    for override in overrides:
        if not override.startswith("+") or "=" not in override:
            continue
        forced = override.startswith("++")
        key = override.lstrip("+").split("=", 1)[0].strip()
        # `~key` deletions and group overrides (`+group@pkg=option`) are not
        # config keys and are left alone.
        if not key or "@" in key:
            continue
        if exists(key):
            continue
        found.append(CreatedKey(override=override, key=key, forced=forced))
    return found


def _key_exists(cfg: DictConfig, key: str) -> bool:
    """Is `key` present in `cfg` -- present, not merely non-null.

    `OmegaConf.select(cfg, key) is not None` is the obvious spelling and is wrong
    here: `entity`, `forcing.source` and `profiler` all ship as null, so
    `++entity=my_team` would be reported as inventing a key that configs/config.yaml
    plainly defines.
    """
    parent_path, _, leaf = key.rpartition(".")
    parent = OmegaConf.select(cfg, parent_path) if parent_path else cfg
    return isinstance(parent, DictConfig) and leaf in parent


def _leaves_named(cfg: DictConfig, leaf: str, limit: int = 6) -> list[str]:
    """Every dotted path in `cfg` whose last component is `leaf`."""
    hits: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if len(hits) >= limit or not isinstance(node, DictConfig):
            return
        for key in node.keys():
            path = f"{prefix}{key}"
            if str(key) == leaf:
                hits.append(path)
            walk(node._get_node(key), f"{path}.")

    walk(cfg, "")
    return hits


def keywords_read_by(target: str) -> set[str] | None:
    """Every constructor keyword the callable at `target` accepts, across its MRO.

    `_target_` is what hydra instantiates a config node into, so its signature is
    the honest answer to "would a new key under this node be read by anything".
    The MRO matters and the `**kwargs` do not: `OceanForecastModule.__init__`
    takes `multistep_curriculum` by name and forwards everything else to
    geoarches' base module, which is where `lr`, `betas` and `weight_decay` are
    named -- so the union over the MRO accepts the documented overrides and still
    rejects `embed_dim`, which no class in that chain has ever heard of.

    Returns None -- "cannot tell" -- when the target will not import.  A guard
    must not refuse out of its own ignorance, so None is treated as a warning
    upstream rather than as a refusal.
    """
    try:
        from hydra.utils import get_class

        cls = get_class(target)
    except Exception:  # noqa: BLE001 - an unimportable target is "cannot tell"
        return None
    names: set[str] = set()
    for base in inspect.getmro(cls):
        initialiser = base.__dict__.get("__init__")
        if initialiser is None:
            continue
        try:
            signature = inspect.signature(initialiser)
        except (TypeError, ValueError):  # C-level __init__, e.g. object's
            continue
        names.update(
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
        )
    return names


def _node_target(reference: DictConfig, path: str) -> str | None:
    """The `_target_` of the config node at `path`, if it has one."""
    node = OmegaConf.select(reference, path) if path else reference
    if not isinstance(node, DictConfig):
        return None
    target = node.get("_target_", None)
    return str(target) if target else None


def _all_leaf_paths(cfg: DictConfig, limit: int = 4000) -> list[str]:
    """Every dotted path in `cfg` that names a leaf value."""
    paths: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, DictConfig) or len(paths) >= limit:
            return
        for key in node.keys():
            path = f"{prefix}{key}"
            child = node._get_node(key)
            if isinstance(child, DictConfig):
                walk(child, f"{path}.")
            else:
                paths.append(path)

    walk(cfg, "")
    return paths


def _nearest_keys(key: str, reference: DictConfig, limit: int = 3) -> list[str]:
    """The existing keys a mistyped one most looks like.

    `++module.embed_dim=512` and `++dataloader.n_levels=20` are the measured
    guesses; the keys that were meant are `module.backbone.emb_dim` and
    `dataloader.n_level_in`.  Matching on the leaf name alone finds them, and
    matching on the full dotted path does not, because the guess is in the wrong
    part of the tree -- which is exactly the mistake.
    """
    leaf = key.rsplit(".", 1)[-1]
    candidates = _all_leaf_paths(reference)
    by_leaf: dict[str, list[str]] = {}
    for path in candidates:
        by_leaf.setdefault(path.rsplit(".", 1)[-1], []).append(path)
    close = difflib.get_close_matches(leaf, list(by_leaf), n=limit, cutoff=0.6)
    return [path for name in close for path in by_leaf[name]][:limit]


def unread_key_refusal(item: CreatedKey, reference: DictConfig) -> str:
    """The refusal for a `++` key that nothing on any code path would read."""
    parent_path = item.key.rsplit(".", 1)[0] if "." in item.key else ""
    target = _node_target(reference, parent_path)
    nearest = _nearest_keys(item.key, reference)
    if target:
        why = (
            f"      {parent_path} is instantiated as {target},\n"
            f"      which takes no `{item.key.rsplit('.', 1)[-1]}` -- so nothing would read it"
        )
    elif parent_path and isinstance(OmegaConf.select(reference, parent_path), DictConfig):
        why = (
            f"      {parent_path} has no `_target_`: nothing instantiates it, so a new key\n"
            "      there is read by no constructor and by no interpolation"
        )
    else:
        why = f"      there is no `{parent_path}` node in the composed config at all"
    suggestion = (
        "      did you mean   ++" + "=...\n      or             ++".join(nearest) + "=...\n"
        if nearest
        else ""
    )
    return (
        f"`{item.override}` creates a new key `{item.key}`. Nothing reads it.\n"
        "\n"
        f"      you wrote      {item.override}\n"
        f"      key created    {item.key}   (absent from the composed config)\n"
        + suggestion
        + why
        + "\n\n"
        "    `++` means add-or-override, so a wrong path is CREATED rather than rejected,\n"
        "    and the run then trains at the preset's own value with no warning at all.\n"
        "    `make train-tiny ... --cfg job --resolve` prints every key there is.\n"
        f"    If you really do want a new key, spell it with a single `+`: `+{item.key}=...`."
    )


def override_problems(
    created: Sequence[CreatedKey], reference: DictConfig
) -> tuple[list[str], list[str]]:
    """`(refusals, warnings)` for keys a `+`/`++` override would invent.

    Three tiers, because "created a key" is not always a mistake:

    * a **top-level** key that is not one geoarches reads optionally -- `++lr=`,
      `++max_step=` -- is a refusal.  `configs/config.yaml` enumerates every
      top-level key the kit uses, so there is nothing else it could be.
    * a nested key whose **leaf name already exists somewhere else** --
      `++module.lr=` against the real `module.module.lr` -- is a refusal, and the
      message names the path that was meant.
    * anything else is a warning.  `++module.module.multistep_curriculum=True` is
      documented in docs/cheatsheet.md and creates a key on purpose: the lightning
      module takes it as a constructor keyword with a default, so it is absent
      from the yaml and still read.  A refusal there would break a documented
      workflow.

    A single `+` is hydra's "I mean to add this" spelling and is never a refusal.
    """
    refusals: list[str] = []
    warnings: list[str] = []
    for item in created:
        leaf = item.key.rsplit(".", 1)[-1]
        elsewhere = [p for p in _leaves_named(reference, leaf) if p != item.key]
        top_level = "." not in item.key
        if not item.forced:
            warnings.append(
                f"`{item.override}` adds a new key `{item.key}` (a single `+` is hydra's "
                "spelling for 'add this on purpose', so this is taken as deliberate)."
            )
            continue
        if top_level and item.key not in GEOARCHES_OPTIONAL_KEYS:
            detail = f"      did you mean   ++{elsewhere[0]}=...\n" if elsewhere else ""
            refusals.append(
                f"`{item.override}` creates a new top-level key `{item.key}`. Nothing reads it.\n"
                "\n"
                f"      you wrote      {item.override}\n"
                f"      key created    {item.key}   (configs/config.yaml defines no such key)\n"
                + detail
                + "\n"
                "    `++` means add-or-override, so a wrong path is CREATED rather than\n"
                "    rejected, and the run then trains at the preset's own value with no\n"
                "    warning at all -- for a full `tiny` run that is 31 minutes at the\n"
                "    learning rate you thought you had changed.\n"
                + (
                    "\n    Existing keys with that name: " + ", ".join(elsewhere) + "\n"
                    if elsewhere
                    else ""
                )
                + "    If you really want a new key, spell it with a single `+`: "
                f"`+{item.key}=...`."
            )
        elif elsewhere:
            refusals.append(
                f"`{item.override}` creates a new key `{item.key}`. Nothing reads it.\n"
                "\n"
                f"      you wrote      {item.override}\n"
                f"      key created    {item.key}   (absent from the composed config)\n"
                f"      did you mean   ++{elsewhere[0]}=...\n"
                "\n"
                "    `++` means add-or-override, so a wrong path is CREATED rather than\n"
                "    rejected. The real value stays exactly where it was and nothing says so.\n"
                "    Existing keys named `" + leaf + "`: " + ", ".join(elsewhere) + "\n"
                f"    If you really want a new key, spell it with a single `+`: `+{item.key}=...`."
            )
        else:
            parent_path = item.key.rsplit(".", 1)[0]
            target = _node_target(reference, parent_path)
            accepted = keywords_read_by(target) if target else None
            if accepted is not None and leaf in accepted:
                warnings.append(
                    f"`{item.override}` creates a new key `{item.key}`; no config file defines "
                    f"it, but {target} takes `{leaf}` as a constructor keyword, so it IS read. "
                    "That is how docs/cheatsheet.md passes "
                    "`++module.module.multistep_curriculum=True` and "
                    "`++module.backbone.gradient_checkpointing=True`."
                )
            elif target is not None and accepted is None:
                warnings.append(
                    f"`{item.override}` creates a new key `{item.key}`, and {target} -- which "
                    "would be the thing to read it -- could not be imported here, so this guard "
                    "cannot tell whether anything reads it. Check the run plan above."
                )
            else:
                refusals.append(unread_key_refusal(item, reference))
    return refusals, warnings


# ---------------------------------------------------------------------------
# 5. What the run is actually about to do
# ---------------------------------------------------------------------------
def resume_line(exp_dir: str | Path, max_steps: int, resume: bool) -> str:
    """One line saying whether this is a fresh start or a resume, and from where.

    `resume: True` prints on every run whether or not there is anything to resume,
    so five separate runs under the same name each looked like a continuation and
    each started from scratch, losing the previous one's work.
    """
    ckpt = latest_checkpoint(exp_dir)
    if ckpt is None:
        return f"FRESH START -- {exp_dir} holds no checkpoint"
    step = checkpoint_step(ckpt)
    if step is None:
        known = "step unknown from the filename"
    elif step >= max_steps:
        known = f"step {step}, already at or past the {max_steps}-step budget"
    else:
        known = f"step {step} of {max_steps}, {max_steps - step} to go"
    note = (
        ""
        if resume
        else "  (resume=False does not prevent this: geoarches loads the newest "
        "checkpoint whenever checkpoints/ exists)"
    )
    return f"RESUMING from {ckpt.name} -- {known}{note}"


def preset_swap_warning(
    overrides: Sequence[str], previous: DictConfig | None, max_steps: int, batch_size: int
) -> str | None:
    """Warn that a preset carries a whole recipe, not only a network size.

    Two detections, both measured:

    * the same group named twice on one command line -- `make train-tiny
      HYDRA_ARGS="module=base"` expands to `module=tiny ... module=base`, hydra
      takes the last silently, and batch_size goes 8 -> 2 and max_steps
      4000 -> 110000 along with the architecture;
    * a re-run of an existing name under a different preset, read off the
      `modelstore/<name>/config.yaml` the previous run wrote.
    """
    chosen = [o.split("=", 1)[1] for o in overrides if o.split("=", 1)[0] == "module"]
    lines: list[str] = []
    if len(chosen) > 1 and len(set(chosen)) > 1:
        lines.append(
            f"the command line names module= more than once ({' then '.join(chosen)}); "
            f"hydra silently keeps the last."
        )
    if previous is not None:
        old_steps = OmegaConf.select(previous, "max_steps")
        old_batch = OmegaConf.select(previous, "batch_size")
        changed = []
        if old_steps is not None and old_steps != max_steps:
            changed.append(f"max_steps {old_steps} -> {max_steps}")
        if old_batch is not None and old_batch != batch_size:
            changed.append(f"batch_size {old_batch} -> {batch_size}")
        if changed:
            lines.append(
                "this run name was last used with a different budget: " + ", ".join(changed) + "."
            )
    if not lines:
        return None
    return (
        " ".join(lines)
        + "\n"
        + "    A preset is a whole recipe -- architecture, per-GPU batch size and step\n"
        "    budget move together (tiny: 8/4000, base: 2/110000, large: 1/300000). If you\n"
        "    wanted the same run with a bigger network, pin the rest yourself:\n"
        '      HYDRA_ARGS="module=base ++batch_size=8 ++max_steps=4000"'
    )


# ---------------------------------------------------------------------------
# 6. A preset swap onto a name that already has checkpoints
# ---------------------------------------------------------------------------
# `NAME=baseline0 MODULE=small` trained the *tiny* network: 13.5M parameters
# where `small` is 45M, warned only about the budget, and overwrote step 30 of
# `baseline0` in place.  The cause is geoarches' resume path
# (`geoarches/main_hydra.py`), which on finding `<exp_dir>/checkpoints`
#
#     cfg.module = exp_cfg.module
#     cfg.dataloader = exp_cfg.dataloader
#
# -- the stored config wins over the one you asked for, and only the `+`/`++`
# dotlist overrides are merged back on top afterwards.  So a *group* swap
# (`module=small`, `dataloader=glorys_seaice`) is silently undone, while
# `++module.backbone.emb_dim=192` on the same command line would survive.  Both
# `resume: True` (the default) and `resume: False` reach a stored checkpoint;
# with `resume: False` geoarches prints "Module config mismatch" and carries on
# regardless, loading the old weights into whatever it built.
#
#: The keys that decide what the network *is*.  A run that differs in any of them
#: is a different model wearing the same name.
ARCHITECTURE_KEYS = (
    "module._target_",
    "n_depths",
    "depth_indices",
    "backbone._target_",
    "backbone.emb_dim",
    "backbone.num_heads",
    "backbone.depth_multiplier",
    "backbone.window_size",
    "backbone.mlp_ratio",
    "backbone.mlp_layer",
    "backbone.first_interaction_layer",
    "backbone.axis_attn",
    "backbone.use_skip",
    "backbone.tensor_size",
    "embedder._target_",
    "embedder.emb_dim",
    "embedder.out_emb_dim",
    "embedder.patch_size",
    "embedder.n_concatenated_states",
    "embedder.forcing_ch",
)
#: The dataloader keys that change what the network is fed -- the channel counts
#: the embedder is built from, and the component wiring.  `domain` is NOT here:
#: training on another split under the same name is a real thing to do.
DATA_KEYS = (
    "component",
    "n_surface_in",
    "n_surface_out",
    "n_level_in",
    "n_level_out",
    "dataset._target_",
    "dataset.lead_time_hours",
    "dataset.load_prev",
)


def _value(cfg: DictConfig | None, key: str) -> Any:
    """`OmegaConf.select`, resolved, with an unresolvable key reading as absent."""
    if cfg is None:
        return None
    try:
        value = OmegaConf.select(cfg, key)
    except Exception:  # noqa: BLE001 - a guard must never be the thing that breaks a run
        return None
    return (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, DictConfig)
        else (list(value) if isinstance(value, ListConfig) else value)
    )


def config_differences(
    previous: DictConfig | None, current: DictConfig | None, keys: Sequence[str]
) -> list[tuple[str, Any, Any]]:
    """`(key, stored, requested)` for every key the two configs disagree on.

    A key that is missing or null on either side is skipped rather than reported:
    presets gain and lose optional keys between versions of the kit, and "you did
    not have `mlp_layer` last week" is not a difference anybody wants stopping a
    run.
    """
    changed = []
    for key in keys:
        old, new = _value(previous, key), _value(current, key)
        if old is None or new is None or old == new:
            continue
        changed.append((key, old, new))
    return changed


def resumed_architecture_refusal(
    name: str,
    exp_dir: str | Path,
    previous: DictConfig | None,
    current: DictConfig,
    module_choice: str = "?",
    dataloader_choice: str = "?",
) -> str | None:
    """Refuse a run whose architecture would be silently replaced by the stored one.

    Fires only when `<exp_dir>/checkpoints` exists -- that is geoarches' own
    trigger for the substitution -- and only when the stored `module` or
    `dataloader` config really differs in a shape-defining key.  Resuming the
    same run with the same preset, with a longer budget or a different learning
    rate goes through untouched: those are the documented ways to continue a run,
    and every one of them survives the substitution intact.
    """
    if not Path(exp_dir).joinpath("checkpoints").is_dir() or previous is None:
        return None
    changed = [
        (f"module.{key}", old, new)
        for key, old, new in config_differences(
            _value_node(previous, "module"), _value_node(current, "module"), ARCHITECTURE_KEYS
        )
    ] + [
        (f"dataloader.{key}", old, new)
        for key, old, new in config_differences(
            _value_node(previous, "dataloader"),
            _value_node(current, "dataloader"),
            DATA_KEYS,
        )
    ]
    if not changed:
        return None
    step = checkpoint_step(latest_checkpoint(exp_dir))
    at = f" (its newest checkpoint is step {step})" if step is not None else ""
    listing = "\n".join(
        f"      {key}   stored {old}  ->  you asked {new}" for key, old, new in changed
    )
    return (
        f"run `{name}` already exists and was trained with a DIFFERENT architecture.\n"
        "\n"
        f"      you asked      module={module_choice}  dataloader={dataloader_choice}\n"
        f"      {exp_dir}/ holds a different one{at}:\n"
        f"{listing}\n"
        "\n"
        "    geoarches would train the STORED one, not the one you asked for. Its resume\n"
        "    path does `cfg.module = exp_cfg.module` as soon as <exp_dir>/checkpoints\n"
        "    exists, so a `module=`/`dataloader=` group swap is silently undone -- the\n"
        "    parameter count stays what it was -- and the new checkpoints overwrite the\n"
        "    old ones in place. `++resume=False` does not help: geoarches still loads the\n"
        "    newest checkpoint from that directory.\n"
        "\n"
        "    Train it under a name of its own, or move the old run aside:\n"
        f"      make train MODULE={module_choice} NAME={name}_{module_choice}\n"
        f"      mv {exp_dir} {exp_dir}.old"
    )


def _value_node(cfg: DictConfig | None, path: str) -> DictConfig | None:
    node = OmegaConf.select(cfg, path) if cfg is not None else None
    return node if isinstance(node, DictConfig) else None


# ---------------------------------------------------------------------------
# 7. Two runs, one name, at the same time
# ---------------------------------------------------------------------------
# Two `make train-tiny NAME=x` at once both printed FRESH START and both aimed at
# `modelstore/x/`: two terminals and a forgotten name is an ordinary hackathon
# accident, and the result is two processes writing one checkpoint directory and
# one `config.yaml`.  A lock file, written only after every other check has
# passed, so a refused run still leaves nothing behind.
LOCK_NAME = ".training.lock"
#: How long a lock left by a run on ANOTHER host is believed.  Liveness is only
#: answerable for our own host (`os.kill(pid, 0)`); across hosts the choice is
#: between an age and nothing, and a wall-clock day covers the longest queue slot
#: this kit documents.
LOCK_STALE_AFTER_SECONDS = 24 * 3600


def run_lock_path(exp_dir: str | Path) -> Path:
    return Path(exp_dir) / LOCK_NAME


def _is_worker_rank(env: dict[str, str] | None = None) -> bool:
    """Is this process a rank that did NOT launch the job?

    DDP reaches this code twice over: `srun --ntasks=4` runs the whole command
    line once per rank, and Lightning's subprocess launcher re-runs it once per
    local device with `LOCAL_RANK` set.  Every one of those composes the config
    and therefore reaches this guard, so a lock taken by every rank would refuse
    the second rank of every multi-GPU run -- the guard crying wolf at the exact
    configuration the cluster configs are for.
    """
    env = os.environ if env is None else env
    return env.get("SLURM_PROCID", "0") != "0" or env.get("LOCAL_RANK", "0") != "0"


def read_run_lock(exp_dir: str | Path) -> dict[str, Any] | None:
    """The lock file's contents, or None when there is none (or it is unreadable)."""
    path = run_lock_path(exp_dir)
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def lock_is_live(
    info: dict[str, Any], hostname: str | None = None, now: float | None = None
) -> bool:
    """Is the process that wrote this lock plausibly still running?

    On this host the question is answerable exactly, and a dead pid means the
    lock is rubbish left by a crash or a `scancel` -- taking it over silently is
    the only behaviour that does not punish somebody for a node failure.  From
    another host, only the age is knowable.
    """
    host = hostname or socket.gethostname()
    now = time.time() if now is None else now
    if info.get("host") == host:
        pid = info.get("pid")
        if not isinstance(pid, int):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # somebody else's process, but a process
            return True
        return True
    started = info.get("started")
    if not isinstance(started, (int, float)):
        return False
    return (now - started) < LOCK_STALE_AFTER_SECONDS


def concurrent_run_refusal(
    name: str, exp_dir: str | Path, info: dict[str, Any], now: float | None = None
) -> str:
    """What to say when this run name is already being trained."""
    now = time.time() if now is None else now
    started = info.get("started")
    age = (
        f"{(now - started) / 60:.0f} min ago"
        if isinstance(started, (int, float))
        else "at an unknown time"
    )
    job = info.get("slurm_job_id") or "no SLURM job id"
    return (
        f"run `{name}` is already being trained by another process.\n"
        "\n"
        f"      started        {age}\n"
        f"      host / pid     {info.get('host', '?')} / {info.get('pid', '?')}   ({job})\n"
        f"      command        {info.get('command', '?')}\n"
        f"      lock file      {run_lock_path(exp_dir)}\n"
        "\n"
        "    Two runs under one name write one `checkpoints/` directory and one\n"
        "    `config.yaml`: whichever saves last wins, and the loser's GPU hours are\n"
        "    gone. Give this one a name of its own:\n"
        f"      make train-tiny NAME={name}_2\n"
        "\n"
        "    If that job is dead -- a crashed node, a `scancel` on another host -- the\n"
        "    lock is stale and deleting it is safe:\n"
        f"      rm {run_lock_path(exp_dir)}"
    )


def acquire_run_lock(
    exp_dir: str | Path, env: dict[str, str] | None = None, hostname: str | None = None
) -> Path | None:
    """Take the lock for `exp_dir`, or raise `SystemExit` if somebody else holds it.

    Returns the lock path (so the caller can release it), or None when this
    process is a DDP rank that must not lock.  Registered with `atexit` as well
    as released by the callback: `on_job_end` does not run when the task raises
    `SystemExit`, and a lock nobody releases is a lock that refuses the next run.
    """
    env = os.environ if env is None else env
    if _is_worker_rank(env):
        return None
    directory = Path(exp_dir)
    path = run_lock_path(directory)
    info = read_run_lock(directory)
    if info is not None and lock_is_live(info, hostname=hostname):
        _refuse("already training", concurrent_run_refusal(directory.name, directory, info))
        return None  # only reached with OCEANARCHES_SKIP_GUARDS set
    if info is not None:
        _emit(
            [
                f"[oceanarches] a stale lock from {info.get('host', '?')} pid "
                f"{info.get('pid', '?')} was left behind; taking it over.",
            ]
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": hostname or socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "started": time.time(),
        "started_text": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
    }
    path.write_text(json.dumps(payload, indent=1))
    atexit.register(release_run_lock, path)
    return path


def release_run_lock(path: str | Path | None) -> None:
    """Remove a lock this process owns.  Idempotent, and never raises."""
    if path is None:
        return
    path = Path(path)
    info = read_run_lock(path.parent)
    if info is not None and info.get("pid") != os.getpid():
        return  # somebody else's lock; taken over after we were declared stale
    try:
        path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 8. The error PyTorch reports, and the fix this kit actually has
# ---------------------------------------------------------------------------
def oom_advice(error: BaseException, batch_size: Any = None, preset: str = "?") -> str | None:
    """Translate a CUDA out-of-memory death into the knob that causes it here.

    `++batch_size=64` died with PyTorch's stock suggestion -- set
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` -- which is about
    fragmentation and is not what an eight-fold batch is.  Returns None for every
    other failure, so an unrelated crash is never dressed up as an OOM.
    """
    text = str(error)
    is_oom = type(error).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError") or (
        "CUDA out of memory" in text or "CUDA error: out of memory" in text
    )
    if not is_oom:
        return None
    try:
        smaller = max(1, int(batch_size) // 2)
        asked = f"{int(batch_size)}"
    except (TypeError, ValueError):
        smaller, asked = 4, "?"
    return (
        "the GPU ran out of memory. In this kit that is `batch_size`, not the allocator.\n"
        "\n"
        f"      you ran with   batch_size {asked}   (module={preset} ships the batch size it "
        "was sized for)\n"
        f'      halve it       HYDRA_ARGS="++batch_size={smaller}"\n'
        "      then           ++module.backbone.gradient_checkpointing=True   (slower, much "
        "less memory)\n"
        "      measure first  make benchmark   -- peak memory of every preset, before you "
        "queue for a GPU\n"
        "\n"
        "    PyTorch's own suggestion above (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)\n"
        "    is about fragmentation. It will not fit a batch that does not fit.\n"
        "    `tiny` at its shipped batch_size 8 already peaks at 42.6 GiB of 96."
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
_RULE = "=" * 78


def _emit(lines: Sequence[str]) -> None:
    """Print to stdout and flush.

    stdout, not stderr, and flushed: notebooks 02 and 03 capture only the
    subprocess' stdout, and an unflushed guard message would come out after the
    traceback it is meant to explain.
    """
    print("\n".join(lines), flush=True)


def _provenance(key: str, overrides: Sequence[str], fallback: str) -> str:
    """Where a value came from: the command line, or the config group that supplies it.

    Worth being exact about. `max_steps` and `save_step_frequency` interpolate
    `${module.*}`, so they follow the size preset; `batch_size` interpolates
    `${cluster.batch_size}`, and only the jupiter clusters pass that on to
    `${module.batch_size}` -- `cluster=local` sets 1 outright. Saying "from the
    preset" there would be wrong, and this line exists precisely so that nobody
    has to guess which number a preset swap moved.
    """
    for override in overrides:
        if override.lstrip("+").split("=", 1)[0] == key:
            return "command line"
    return fallback


def run_plan(cfg: DictConfig, choices: dict[str, Any], overrides: Sequence[str] = ()) -> list[str]:
    """The block printed at the top of every run: what is about to happen, in one place."""
    name = OmegaConf.select(cfg, "name")
    exp_dir = OmegaConf.select(cfg, "exp_dir")
    max_steps = int(OmegaConf.select(cfg, "max_steps") or 0)
    frequency = int(OmegaConf.select(cfg, "save_step_frequency") or 0)
    batch_size = OmegaConf.select(cfg, "batch_size")
    mode = OmegaConf.select(cfg, "mode")
    groups = "  ".join(
        f"{g}={choices.get(g, '?')}" for g in ("module", "dataloader", "cluster", "forcing")
    )
    steps_from = _provenance("max_steps", overrides, f"module={choices.get('module', '?')}")
    batch_from = _provenance("batch_size", overrides, f"cluster={choices.get('cluster', '?')}")
    steps = checkpoint_steps(max_steps, frequency)
    if not steps:
        schedule = "NONE -- see the refusal below"
    else:
        shown = ", ".join(str(s) for s in steps[:4]) + (", ..." if len(steps) > 4 else "")
        schedule = f"every {frequency} steps -> {shown}"
    lines = [
        "",
        _RULE,
        f"[oceanarches] run plan -- {name}   (mode={mode})",
        f"  config       {groups}",
    ]
    # `mode=test` reads a checkpoint and writes none, so the budget and the save
    # schedule are noise there -- and a "checkpoints every 25000 steps" line on a
    # scoring run is worse than noise.
    if mode == "train":
        lines += [
            f"  budget       max_steps {max_steps} ({steps_from}), "
            f"batch_size {batch_size} ({batch_from})",
            f"  checkpoints  {schedule}",
            f"               in {exp_dir}/checkpoints",
        ]
    lines += [
        f"  start        {resume_line(exp_dir, max_steps, bool(OmegaConf.select(cfg, 'resume')))}",
        f"  allocation   {_allocation_detail()}",
        _RULE,
    ]
    return lines


def _refuse(kind: str, message: str) -> None:
    _emit(["", _RULE, f"FATAL: {message}", _RULE, ""])
    if skip_guards():
        _emit([f"OCEANARCHES_SKIP_GUARDS is set -- continuing anyway ({kind}).", ""])
        return
    raise SystemExit(f"oceanarches: refusing to start ({kind}); the reason is printed above.")


def _warn(message: str) -> None:
    _emit(["", f"[oceanarches] WARNING: {message}", ""])


def _reference_config(config_name: str | None, overrides: Sequence[str]) -> DictConfig | None:
    """The config as it would compose WITHOUT the `+`/`++` overrides.

    That is the only honest answer to "did this key already exist": the composed
    config always contains the key, because the override put it there.  hydra is
    already initialised inside a callback (`GlobalHydra` is set by
    `create_main_hydra2` before the run), so this is a re-compose off the same
    search path and costs a few milliseconds of yaml.
    """
    from hydra import compose

    plain = [o for o in overrides if not o.startswith(("+", "~"))]
    try:
        return compose(config_name=config_name or "config", overrides=list(plain))
    except Exception:  # noqa: BLE001 - a guard must never be the thing that breaks a run
        return None


class StartupGuard(Callback):
    """The hydra callback registered by `configs/config.yaml`.

    Everything it can say is said before `run_job`, i.e. before the dataloaders,
    the model or the GPU exist.

    It subclasses hydra's `Callback` rather than duck-typing the one hook it
    needs: `Callbacks._notify` calls `on_job_end` and `on_run_end` on every
    registered callback and turns the resulting `AttributeError` into a
    `UserWarning`, so a bare class ends a perfectly good run with two warnings
    about itself.  The base class supplies the no-ops.
    """

    #: Set by :meth:`check` when this process takes the run lock, so that
    #: :meth:`on_job_end` releases the one it took and never somebody else's.
    lock: Path | None = None

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        self.check(config, config_name=kwargs.get("config_name"))

    def on_job_end(self, config: DictConfig, job_return: Any, **kwargs: Any) -> None:
        """Explain a CUDA OOM in this kit's vocabulary, and drop the run lock.

        hydra calls this for a FAILED job as well as a completed one, with the
        exception in `job_return._return_value` (the public property re-raises
        it, which inside a callback would only bury the original traceback).
        """
        error = getattr(job_return, "_return_value", None)
        if isinstance(error, BaseException):
            choices = (
                OmegaConf.to_container(OmegaConf.select(config, "hydra.runtime.choices"))
                if OmegaConf.select(config, "hydra.runtime.choices") is not None
                else {}
            )
            advice = oom_advice(
                error,
                OmegaConf.select(config, "batch_size"),
                str(choices.get("module", "?")),
            )
            if advice is not None:
                _emit(["", _RULE, f"FATAL: {advice}", _RULE, ""])
        release_run_lock(self.lock)
        self.lock = None

    # `on_multirun_start` deliberately not implemented: a sweep composes one
    # config per job and `on_job_start` would be the hook, but nothing in this
    # kit sweeps and a guard that has never run is worse than no guard.

    def check(self, config: DictConfig, config_name: str | None = None) -> None:
        choices = OmegaConf.to_container(
            OmegaConf.select(config, "hydra.runtime.choices") or OmegaConf.create({})
        )
        overrides = list(OmegaConf.select(config, "hydra.overrides.task") or [])
        cfg = config  # the job keys live at the top level alongside `hydra`
        mode = OmegaConf.select(cfg, "mode")
        max_steps = int(OmegaConf.select(cfg, "max_steps") or 0)
        frequency = int(OmegaConf.select(cfg, "save_step_frequency") or 0)
        exp_dir = OmegaConf.select(cfg, "exp_dir") or "modelstore/?"
        name = OmegaConf.select(cfg, "name") or "?"

        _emit(run_plan(cfg, choices, overrides))

        warn_allocation(cluster_name=str(choices.get("cluster", "?")))

        reference = _reference_config(config_name, overrides)
        if reference is not None:
            refusals, warnings = override_problems(
                created_keys(overrides, lambda key: _key_exists(reference, key)),
                reference,
            )
            for message in warnings:
                _warn(message)
            if refusals:
                _refuse("orphan override", "\n\nFATAL: ".join(refusals))

        if mode != "train":
            return

        previous = _previous_config(exp_dir)
        swap = preset_swap_warning(
            overrides,
            previous,
            max_steps,
            OmegaConf.select(cfg, "batch_size"),
        )
        if swap is not None:
            _warn(swap)

        stale = resumed_architecture_refusal(
            str(name),
            exp_dir,
            previous,
            cfg,
            str(choices.get("module", "?")),
            str(choices.get("dataloader", "?")),
        )
        if stale is not None:
            _refuse("architecture would be replaced by the stored one", stale)

        step = checkpoint_step(latest_checkpoint(exp_dir))
        finished = already_trained_refusal(str(name), exp_dir, step, max_steps)
        if finished is not None:
            _refuse("nothing to train", finished)

        no_checkpoint = no_checkpoint_refusal(
            max_steps, frequency, str(choices.get("module", "?")), str(name)
        )
        if no_checkpoint is not None:
            _refuse("no checkpoint would be written", no_checkpoint)

        # Last, and only here: everything above refuses without creating
        # `modelstore/<name>/`, and taking the lock creates it.
        self.lock = acquire_run_lock(exp_dir)


def _previous_config(exp_dir: str | Path) -> DictConfig | None:
    """`modelstore/<name>/config.yaml` from an earlier run of this name, if any."""
    path = Path(exp_dir) / "config.yaml"
    if not path.is_file():
        return None
    try:
        loaded = OmegaConf.load(path)
    except Exception:  # noqa: BLE001
        return None
    return loaded if isinstance(loaded, DictConfig) else None
