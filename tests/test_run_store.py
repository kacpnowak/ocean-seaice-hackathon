"""Deleting a run directory, without deleting somebody else's.

Notebook 02 opened its training cell with

    shutil.rmtree(f"modelstore/{RUN}", ignore_errors=True)

on ``RUN = "notebook_probe"``.  docs/01 section 1.1 builds ``modelstore/`` out of
symlinks into a checkpoint store shared with the whole room, and
``notebook_probe`` exists in that store as well: a participant running the
notebook top to bottom against a *writable* share deleted a shipped checkpoint
out from under everybody else, and ``ignore_errors=True`` swallowed the evidence
whether it worked or not.  The rehearsal survived it only because the share
happened to be mounted read-only.

Both of the shapes docs/01 produces are reconstructed here with real symlinks and
a fake shared store -- never against the real one -- and both must be refused
*before* anything is removed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from oceanarches import paths

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "notebooks"


@pytest.fixture
def clone(tmp_path, monkeypatch):
    """A participant's clone and the tutor's store, side by side.

    Returns ``(clone_root, shared_store)``.  The shared store holds
    ``notebook_probe`` -- the name notebook 02 uses -- with a checkpoint in it,
    exactly as the real one does.
    """
    clone_root = tmp_path / "clone"
    (clone_root / "notebooks").mkdir(parents=True)
    shared = tmp_path / "shared" / "modelstore"
    (shared / "notebook_probe" / "checkpoints").mkdir(parents=True)
    (shared / "notebook_probe" / "checkpoints" / "checkpoint_global_step=4500.ckpt").write_bytes(
        b"the tutor's weights"
    )
    (shared / "notebook_probe" / "config.yaml").write_text("name: notebook_probe\n")
    monkeypatch.setattr(paths, "REPO_ROOT", clone_root)
    monkeypatch.setenv("MODELSTORE", "modelstore")
    return clone_root, shared


def _refusal_from(call):
    """Run `call`, returning the refusal it raised -- or None if it raised nothing.

    `pytest.raises` would abort the test at the missing exception, before the
    assertion that says whether the shared store survived. That assertion is the
    point of this file, so it has to come first.
    """
    try:
        call()
    except paths.UnsafeRunDirectory as error:
        return error
    return None


def _shared_intact(shared: Path) -> bool:
    ckpt = shared / "notebook_probe" / "checkpoints" / "checkpoint_global_step=4500.ckpt"
    return ckpt.is_file() and ckpt.read_bytes() == b"the tutor's weights"


def test_a_whole_directory_symlink_cannot_be_deleted_through(clone):
    """`ln -s $SHARED modelstore` -- the route two participants took.

    Every run name under it is the shared store's.  `shutil.rmtree` follows the
    symlink into the target's contents and removes them, so this is the shape
    that loses the checkpoint.

    MUTANT: dropping the symlink test from `check_own_run_dir` deletes the
    tutor's `notebook_probe` and fails the last assertion.
    """
    clone_root, shared = clone
    (clone_root / "modelstore").symlink_to(shared)
    assert (clone_root / "modelstore" / "notebook_probe").is_dir()  # the trap is armed

    # Not `pytest.raises`: the deletion is what matters, so the shared store is
    # checked FIRST -- a version that deletes and then raises would pass a bare
    # `raises` block, and a version that deletes without raising would fail on
    # the wrong assertion and never say what it cost.
    raised = _refusal_from(lambda: paths.remove_run_dir("notebook_probe"))
    assert _shared_intact(shared), "the tutor's checkpoint was deleted"
    assert raised is not None, "the deletion was allowed to reach the shared store"
    assert "SYMLINK" in str(raised)
    assert "docs/01 section 1.1" in str(raised)


def test_a_per_run_symlink_cannot_be_deleted_through(clone):
    """The route docs/01 actually recommends: one symlink per shipped run.

    `shutil.rmtree` on a symlink raises `OSError: Cannot call rmtree on a
    symbolic link` -- which `ignore_errors=True` hid, so the notebook then
    trained *into the shared run* believing it had started clean.

    MUTANT: removing the per-run symlink test lets `remove_run_dir` raise
    OSError instead of `UnsafeRunDirectory`, and says nothing about the name.
    """
    clone_root, shared = clone
    store = clone_root / "modelstore"
    store.mkdir()
    (store / "notebook_probe").symlink_to(shared / "notebook_probe")

    raised = _refusal_from(lambda: paths.remove_run_dir("notebook_probe"))
    assert _shared_intact(shared), "the tutor's checkpoint was deleted"
    assert (store / "notebook_probe").is_symlink(), "the link itself was removed"
    assert raised is not None, "rmtree was left to fail on its own, silently"
    assert "SYMLINK" in str(raised)


def test_a_real_directory_of_your_own_is_removed_and_says_so(clone):
    """The case the notebook is actually for, and the half that must keep working.

    A guard that refuses this would be worse than the line it replaced.
    """
    clone_root, shared = clone
    run = clone_root / "modelstore" / "my_probe" / "checkpoints"
    run.mkdir(parents=True)
    (run / "checkpoint_global_step=100.ckpt").write_bytes(b"mine")

    said = paths.remove_run_dir("my_probe")

    assert not (clone_root / "modelstore" / "my_probe").exists()
    assert "removed" in said and "my_probe" in said
    assert _shared_intact(shared)


def test_removing_a_run_that_is_not_there_is_not_an_error_but_is_reported(clone):
    """Re-running the notebook cell on a clean clone must not raise -- but the
    old `ignore_errors=True` reported nothing in *either* direction, and that is
    the half being removed."""
    said = paths.remove_run_dir("never_trained")
    assert "nothing to remove" in said


def test_a_shipped_run_name_is_refused_even_in_a_directory_of_your_own(clone):
    """`task6_tiny` is loaded by name from four documents and three notebooks."""
    clone_root, _ = clone
    (clone_root / "modelstore" / "task6_tiny" / "checkpoints").mkdir(parents=True)
    with pytest.raises(paths.UnsafeRunDirectory) as raised:
        paths.remove_run_dir("task6_tiny")
    assert "ships" in str(raised.value)
    assert (clone_root / "modelstore" / "task6_tiny").is_dir()


@pytest.mark.parametrize("name", ["../evil", "/etc", "a/b", "", ".", ".."])
def test_a_run_name_that_is_really_a_path_is_refused(clone, name):
    """A run name addresses one directory under the store, and nothing else."""
    with pytest.raises(paths.UnsafeRunDirectory):
        paths.remove_run_dir(name)


# ---------------------------------------------------------------------------
# Wiring: the notebook has to be the thing that goes through it
# ---------------------------------------------------------------------------
def _code(notebook: Path, comments: bool = True) -> str:
    """The code cells, optionally with whole-line comments dropped.

    Dropped, because the fixed cell *names* the line it replaced -- a comment
    saying what not to do must not read as doing it.
    """
    cells = json.loads(notebook.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    if comments:
        return text
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


@pytest.mark.parametrize("notebook", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.name)
def test_no_notebook_deletes_a_path_it_has_not_checked(notebook):
    """The unguarded shapes, in every notebook rather than only the one that had it.

    `shutil.rmtree`, `Path.unlink` and `os.remove` on a `modelstore/` path are
    all one symlink away from somebody else's checkpoints, and `ignore_errors`
    makes the outcome unobservable.

    MUTANT: restoring the original `shutil.rmtree(f"modelstore/{RUN}",
    ignore_errors=True)` line in notebook 02 fails this.
    """
    code = _code(notebook, comments=False)
    assert "rmtree" not in code, f"{notebook.name} removes a directory tree by hand"
    assert "ignore_errors" not in code
    assert "os.remove" not in code and ".unlink(" not in code


def test_notebook_02_clears_its_probe_run_through_the_checked_deletion():
    """It still has to *do* the thing -- geoarches resumes an existing run silently."""
    code = _code(NOTEBOOKS / "02_train_and_rollout.ipynb")
    assert "paths.remove_run_dir(RUN)" in code
    assert 'RUN = "notebook_probe"' in code


def test_notebook_02_says_that_the_rollouts_do_not_need_the_training_run():
    """Its later cells -- the point of the notebook -- load the shipped checkpoint.

    A participant whose training cell failed read the rest as lost with it.
    """
    cells = json.loads((NOTEBOOKS / "02_train_and_rollout.ipynb").read_text())["cells"]
    markdown = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    assert "do not depend on this training run" in markdown
    assert "task6_tiny" in markdown


# ---------------------------------------------------------------------------
# The four places a newly shipped run has to be named, and the two a test can see
# ---------------------------------------------------------------------------
def test_the_documented_symlink_loop_ships_exactly_the_protected_runs():
    """`SHIPPED_RUNS` and docs/01 section 1.1 are the same list, or the kit lies.

    Adding a run to the shared store means naming it in four places (see
    docs/TUTORS.md section 2): the store itself, `SHIPPED_RUNS`, the `for run in
    ...` loop participants copy out of docs/01, and a release. Miss the loop and
    nobody links the new run in -- the checkpoint is there and every document
    that loads it by name fails. Miss `SHIPPED_RUNS` and the kit will let a
    participant delete it out of the store shared with the room, which is the
    failure the rest of this file exists to prevent.

    Nothing tied the two together, so a run added to one could sit indefinitely
    missing from the other.

    MUTANT: dropping any name from either list fails this.
    """
    text = (REPO_ROOT / "docs" / "01_setup.md").read_text()
    match = re.search(r"^for run in (.+?); do$", text, re.M)
    assert match is not None, (
        "docs/01 section 1.1 no longer has the `for run in ...; do` loop that "
        "participants copy to link the shipped runs in"
    )
    documented = set(match.group(1).split())
    protected = set(paths.SHIPPED_RUNS)
    assert documented == protected, (
        f"docs/01 tells participants to link {sorted(documented)}, but "
        f"paths.SHIPPED_RUNS protects {sorted(protected)}.\n"
        f"  in the docs but unprotected: {sorted(documented - protected) or 'none'}\n"
        f"  protected but undocumented: {sorted(protected - documented) or 'none'}"
    )


def test_every_shipped_run_is_really_in_the_store():
    """A name in both lists and nothing on disk is the same failure, one step later."""
    store = paths.modelstore()
    if not store.is_dir():  # a clone that has not linked anything in yet
        pytest.skip(f"{store} does not exist in this checkout")
    missing = [name for name in sorted(paths.SHIPPED_RUNS) if not (store / name).exists()]
    assert not missing, f"{store} is missing shipped runs: {missing}"
