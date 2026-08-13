"""Project paths, resolved from ``config.env``.

Everything that needs to know where the data lives asks this module, so there is
exactly one file to edit when you move to another machine: ``config.env``.

Resolution order for every setting:

1. a real environment variable (``export GLORYS_PREPPED=...``) -- wins,
2. the value in ``config.env``,
3. the built-in default below.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

# Repository root == the directory containing this package.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ENV = REPO_ROOT / "config.env"

# Where generated artefacts (masks, normalisation stats, climatology) are written.
STATS_DIR = Path(__file__).resolve().parent / "stats"

_DEFAULTS = {
    "GLORYS_RAW": "/e/data1/climateai/hclimrep/data/glorys_1deg",
    "DATA_ROOT": str(REPO_ROOT / "data"),
    "GLORYS_PREPPED": str(REPO_ROOT / "data" / "glorys_1deg_prepped"),
    "IFS_FORCING": "/e/data1/climateai/hclimrep/data/glorys_forcings/ifs_1deg",
    "MODELSTORE": "modelstore",
    "EVALSTORE": "evalstore",
    "SLURM_ACCOUNT": "hclimrep",
    "SLURM_PARTITION": "booster",
}

_ASSIGNMENT = re.compile(r'^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*"?(.*?)"?\s*$')


@lru_cache(maxsize=1)
def _config_env() -> dict[str, str]:
    """Parse ``config.env``: ``KEY="value"`` lines, with ``${OTHER_KEY}`` expansion."""
    values: dict[str, str] = {}
    if not CONFIG_ENV.exists():
        return values
    for line in CONFIG_ENV.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ASSIGNMENT.match(line)
        if not match:
            continue
        key, raw = match.groups()
        # Expand ${VAR} against values seen so far, then against the real environment.
        expanded = re.sub(
            r"\$\{([A-Z_][A-Z0-9_]*)\}",
            lambda m: values.get(m.group(1), os.environ.get(m.group(1), "")),
            raw,
        )
        values[key] = expanded
    return values


def setting(key: str, default: str | None = None) -> str:
    """Look up one setting (environment > config.env > built-in default)."""
    if key in os.environ and os.environ[key]:
        return os.environ[key]
    from_file = _config_env().get(key)
    if from_file:
        return from_file
    if default is not None:
        return default
    if key in _DEFAULTS:
        return _DEFAULTS[key]
    raise KeyError(f"Unknown setting {key!r}. Add it to {CONFIG_ENV} or pass a default.")


def path(key: str, default: str | None = None) -> Path:
    """Same as :func:`setting` but returns a :class:`~pathlib.Path`."""
    return Path(setting(key, default))


# --- convenience accessors --------------------------------------------------
def glorys_raw() -> Path:
    """Read-only tree of raw daily GLORYS files: ``<root>/YYYY/MM/*.nc``."""
    return path("GLORYS_RAW")


def glorys_prepped() -> Path:
    """Prepared yearly files written by ``scripts/prepare_glorys.py``."""
    return path("GLORYS_PREPPED")


def ifs_forcing() -> Path:
    """Optional atmospheric forcing (only covers ~2024 -- see docs/05_coupling.md)."""
    return path("IFS_FORCING")


def masks_file() -> Path:
    return STATS_DIR / "glorys_1deg_masks.nc"


def stats_file() -> Path:
    return STATS_DIR / "glorys_1deg_stats.pt"


def climatology_file() -> Path:
    return STATS_DIR / "glorys_1deg_climatology.nc"


def forcing_stats_file() -> Path:
    """Normalisation statistics for the prescribed atmosphere.

    Separate from ``stats_file()`` on purpose: the forcing is not part of the
    GLORYS state, is not scored, and is averaged over land as well as ocean.
    Written by ``scripts/compute_forcing_stats.py`` (``make forcing-stats``).
    """
    return STATS_DIR / "ifs_1deg_forcing_stats.pt"


# ---------------------------------------------------------------------------
# Run directories, and deleting one without deleting somebody else's
# ---------------------------------------------------------------------------
# docs/01 section 1.1 tells a participant to build `modelstore/` out of symlinks
# into the tutor's shared store, one per shipped run.  Two participants linked
# the whole directory instead (`ln -s $SHARED modelstore`), which the same
# section warns against and which nothing enforces.  Either way, a path under
# `modelstore/` can resolve into a directory shared with the whole room -- and
# notebook 02 opened with
#
#     shutil.rmtree(f"modelstore/{RUN}", ignore_errors=True)
#
# on `RUN = "notebook_probe"`, a name that also exists in the shared store.  Run
# top to bottom against a writable share that deletes a shipped checkpoint out
# from under everyone else, and `ignore_errors=True` means it says nothing
# either way.  It survived the rehearsal only because the share was mounted
# read-only.
#
# So the deletion goes through one function that refuses anything which is not
# the caller's own directory, and reports what it did.
#: The runs docs/01 tells you to link in from the shared store.  Never ours to delete.
SHIPPED_RUNS = frozenset(
    {"task6_tiny", "ocean_tiny", "seaice_tiny", "seaice_isolated_tiny", "large_pretrained"}
)


class UnsafeRunDirectory(RuntimeError):
    """`modelstore/<name>` is not this clone's own directory to remove."""


def modelstore() -> Path:
    """The run store, absolute.  ``MODELSTORE`` is relative to the repository."""
    store = path("MODELSTORE")
    return store if store.is_absolute() else REPO_ROOT / store


def run_dir(name: str) -> Path:
    """``modelstore/<name>``, with ``name`` checked to be a plain run name.

    Not a path: `..`, an absolute path or anything with a separator in it would
    let a run name address a directory the store does not contain.
    """
    text = str(name)
    if not text or text in (".", "..") or text != Path(text).name:
        raise UnsafeRunDirectory(
            f"{text!r} is not a run name. A run name is a single directory name "
            "under modelstore/ -- no '/', no '..', not an absolute path."
        )
    return modelstore() / text


def check_own_run_dir(name: str) -> Path:
    """Return ``modelstore/<name>`` if it is this clone's own, else raise.

    "Ours" means: a real directory under a real store, reached without following
    a symlink.  Both of the routes docs/01 section 1.1 describes put other
    people's checkpoints within reach of that path -- linking the shipped runs in
    one by one makes `modelstore/<name>` a symlink for those names, and linking
    the whole directory (which that section warns against, and which two
    participants did anyway) makes *every* name one.  Neither is ours to write to
    or delete.

    An absolute ``MODELSTORE`` in config.env is taken at its word: that is a
    deliberate statement about where this clone keeps its runs, not the
    accidental sharing this function exists to stop.

    Raises:
        UnsafeRunDirectory: naming the path it resolved to and what to do instead.
    """
    directory = run_dir(name)
    store = modelstore()
    if store.is_symlink() or (_within(store, REPO_ROOT.resolve()) and store.resolve() != store):
        raise UnsafeRunDirectory(
            f"{store} is a SYMLINK to {store.resolve()}.\n"
            "That is `ln -s $SHARED modelstore`, which docs/01 section 1.1 warns against: "
            "every run name under it -- including this one -- is the shared store's, so a "
            "run written or deleted here is everybody's.\n"
            "Make modelstore/ a real directory of your own and link the shipped runs into "
            "it one by one (docs/01 section 1.1)."
        )
    if directory.is_symlink():
        raise UnsafeRunDirectory(
            f"{directory} is a SYMLINK to {os.readlink(directory)}.\n"
            "That is how docs/01 section 1.1 links the shipped runs in, so this run belongs "
            "to the shared store and not to you.\n"
            f"Pick a run name of your own (not one of: {', '.join(sorted(SHIPPED_RUNS))})."
        )
    resolved = directory.resolve() if directory.exists() else _resolve_parent(directory)
    if not _within(resolved, store.resolve()):
        raise UnsafeRunDirectory(
            f"{directory} resolves to {resolved}, which is outside {store.resolve()}."
        )
    if str(name) in SHIPPED_RUNS:
        raise UnsafeRunDirectory(
            f"{name!r} is one of the runs the kit ships (docs/01 section 1.1). "
            "Even where it is a directory of your own it is the copy the documents and "
            "the notebooks load by name. Use a name of your own."
        )
    return directory


def remove_run_dir(name: str) -> str:
    """Delete ``modelstore/<name>`` -- and only ever this clone's own copy.

    The counterpart of :func:`check_own_run_dir`, with no ``ignore_errors``: a
    failed delete is raised, because the whole failure this replaces was a
    deletion that said nothing whether it worked or not.

    Returns:
        One line saying what happened, for a notebook to print.
    """
    import shutil

    directory = check_own_run_dir(name)
    if not directory.exists():
        return f"nothing to remove: {directory} does not exist"
    entries = sum(1 for _ in directory.rglob("*"))
    shutil.rmtree(directory)
    return f"removed {directory} ({entries} files and directories)"


def _resolve_parent(directory: Path) -> Path:
    """Where a path that does not exist yet would be created.

    `Path.resolve()` on a missing path still resolves the symlinks in its
    parents, but resolving the parent explicitly keeps the answer honest when
    only the leaf is absent.
    """
    return directory.parent.resolve() / directory.name


def _within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents
