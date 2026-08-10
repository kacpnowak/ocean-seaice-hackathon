"""Caching the figures and the animations, on the same terms as the rollout.

The rollout has been cached since the beginning, which made "re-run the eval
cell" sound cheap.  It was not: figures and animations re-rendered
unconditionally, so a second run of the same command cost ~200 s against 243 s
cold -- a participant measured exactly that and concluded the cache did not
work.  Nearly all of the saving was in the two stages that were never cached.

The design is the rollout's, deliberately:

* the key is the **inputs**, not a file name or a timestamp -- the rollout spec
  (which already carries the checkpoint's content hash and the config hash),
  which forecasters were scored, and the render options;
* a manifest that does not match is ignored rather than patched, because a cache
  that silently answers a different question is worse than no cache;
* ``--force`` bypasses it, exactly as it bypasses the rollout cache.

What the key deliberately does **not** cover is the plotting code itself: editing
``plots.py`` does not invalidate a rendered figure, just as editing ``rollout.py``
does not invalidate a scored rollout.  Bump :data:`CACHE_VERSION` when a change
to the drawing has to reach caches that already exist.  ``--compare-with`` is not
cached at all: those figures read *another* run's cache directory, which this key
cannot see change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["CACHE_VERSION", "MANIFEST_NAME", "render_key", "cached_render", "record_render"]

#: Bump when a change in the rendering must invalidate existing caches.
CACHE_VERSION = 1

#: Written inside the figure/animation directory it describes.
MANIFEST_NAME = ".render_cache.json"


def render_key(kind: str, spec, free_spec=None, forecasters=(), **options) -> str:
    """A short hash of everything that changes what is drawn.

    Args:
        kind: ``"figures"`` or ``"animations"`` -- two caches, two directories.
        spec: the scored rollout's :class:`~oceanarches.evaluation.rollout.RolloutSpec`.
        free_spec: the free-running rollout's spec, or None when there is none.
        forecasters: the forecaster keys present in the result.  The same spec
            can be answered by a cache holding more of them, and the figures
            differ.
        **options: dpi, fps, and anything else the renderer was passed.
    """
    payload = {
        "version": CACHE_VERSION,
        "kind": kind,
        "spec": spec.as_manifest() if hasattr(spec, "as_manifest") else spec,
        "free_spec": (free_spec.as_manifest() if hasattr(free_spec, "as_manifest") else free_spec),
        "forecasters": sorted(forecasters),
        "options": options,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def cached_render(directory: Path, key: str) -> list[Path] | None:
    """The files a previous render with this exact key left behind, or None.

    An empty render is reported as a miss.  Nothing here is expensive enough for
    that to matter -- ``render_all`` returns an empty list in milliseconds when
    there are no cached fields to draw from -- and it means a run that produced
    no figures at all never looks like a successful cached one.
    """
    directory = Path(directory)
    manifest = directory / MANIFEST_NAME
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("key") != key or data.get("version") != CACHE_VERSION:
        return None
    paths = [directory / name for name in data.get("files", [])]
    if not paths or not all(path.exists() for path in paths):
        return None
    return paths


def record_render(directory: Path, key: str, paths) -> None:
    """Record what this render produced, so the next identical one can skip it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = [Path(p).name for p in paths]
    (directory / MANIFEST_NAME).write_text(
        json.dumps({"version": CACHE_VERSION, "key": key, "files": files}, indent=2)
    )
