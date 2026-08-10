"""`scripts/publish.sh`: the guards, and the shape of the history it writes.

The public repository must never contain the development history.  That is not
something you can check by reading the script once -- it is a property of what
the script writes, so every test here runs the real script in a throwaway git
repository and then inspects the commits it produced.

The expensive failure this file exists to prevent is a *silent* one: a publish
that works, pushes, and quietly carries either the 70-odd development commits or
a gitignored file out to a public URL.  Neither can be taken back.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISH = REPO_ROOT / "scripts" / "publish.sh"


def git(repo: Path, *args: str) -> str:
    """Run git in `repo` and return its stdout, stripped."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_fails(repo: Path, *args: str) -> bool:
    """True when git exits non-zero -- `merge-base` answers "never" that way."""
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True).returncode != 0


def publish(repo: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    """Run the copy of publish.sh that lives in `repo`, never the one in this repo."""
    return subprocess.run(
        ["bash", str(repo / "scripts" / "publish.sh"), *args],
        capture_output=True,
        text=True,
        input=input_text,
        cwd=str(repo),
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repository shaped like this one: `main`, a .gitignore, and history.

    Three commits, so that "the published branch has one commit" is a real
    statement about squashing rather than an accident of an empty history.
    """
    root = tmp_path / "kit"
    (root / "scripts").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    shutil.copy2(PUBLISH, root / "scripts" / "publish.sh")

    (root / ".gitignore").write_text("secrets/\n*.ckpt\ndata/\n")
    (root / "README.md").write_text("first\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "first development commit")

    (root / "README.md").write_text("second\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "second development commit")

    (root / "kit.py").write_text("print('hello')\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "third development commit")

    # The things that must never be published.
    (root / "secrets").mkdir()
    (root / "secrets" / "token.txt").write_text("hunter2\n")
    (root / "model.ckpt").write_bytes(b"weights")
    (root / "data").mkdir()
    (root / "data" / "glorys.nc").write_bytes(b"92GB, pretend")
    return root


def commit_subjects(repo: Path, branch: str) -> list[str]:
    return git(repo, "log", "--format=%s", branch).splitlines()


# ---------------------------------------------------------------------------
# 1. The history it writes
# ---------------------------------------------------------------------------
def test_the_first_publish_is_exactly_one_commit_holding_mains_tree(repo: Path):
    """One root commit, no parent, and a tree identical to main's tracked tree.

    MUTANT: dropping the `-p` handling and always parenting on main would fail
    the count; committing anything other than `main^{tree}` fails the diff.
    """
    result = publish(repo, "--yes", "Initial public release")
    assert result.returncode == 0, result.stderr

    assert commit_subjects(repo, "public") == ["Initial public release"]
    assert git(repo, "rev-list", "--count", "public") == "1"
    assert git(repo, "rev-list", "--count", "main") == "3"
    # The published tree IS main's tree, byte for byte.
    assert git(repo, "diff", "--stat", "main", "public") == ""
    assert git(repo, "rev-parse", "main^{tree}") == git(repo, "rev-parse", "public^{tree}")
    # A root commit: nothing behind it to walk back into.
    assert git(repo, "rev-list", "--parents", "-1", "public").split() == [
        git(repo, "rev-parse", "public")
    ]


def test_a_second_publish_appends_exactly_one_commit(repo: Path):
    """Two releases, two commits -- and the second tree is main's new tree."""
    assert publish(repo, "--yes", "Initial public release").returncode == 0

    (repo / "kit.py").write_text("print('hello again')\n")
    (repo / "notes.md").write_text("what changed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a development commit nobody should read")

    result = publish(repo, "--yes", "Second release: notes")
    assert result.returncode == 0, result.stderr

    assert commit_subjects(repo, "public") == ["Second release: notes", "Initial public release"]
    assert git(repo, "rev-list", "--count", "public") == "2"
    assert git(repo, "diff", "--stat", "main", "public") == ""
    # And the release trail is a straight line, so a participant can pull it.
    assert git(repo, "rev-parse", "public~1") == git(repo, "rev-list", "--max-parents=0", "public")


def test_the_published_branch_shares_no_commit_with_main(repo: Path):
    """The whole point: `public` is not a descendant of `main` and never was.

    If it were, `git push public:main` would carry every development commit,
    message and author date out to a public URL.
    """
    publish(repo, "--yes", "Initial public release")
    (repo / "kit.py").write_text("2\n")
    git(repo, "commit", "-qam", "more development")
    publish(repo, "--yes", "Second release")

    assert git(repo, "rev-list", "--count", "public", "^main") == "2"  # nothing shared
    assert git_fails(repo, "merge-base", "--is-ancestor", "main", "public")
    # No merge base at all: git exits 1 because the two histories never meet.
    assert git_fails(repo, "merge-base", "--all", "main", "public")

    for subject in commit_subjects(repo, "main"):
        assert subject not in commit_subjects(repo, "public")


def test_a_clone_of_the_published_branch_sees_only_the_releases(repo: Path, tmp_path: Path):
    """What a participant actually gets: the releases, and no gitignored file."""
    bare = tmp_path / "public.git"
    # `-b main`: a bare repo's HEAD defaults to whatever init.defaultBranch says,
    # and a clone of a repo whose HEAD names a branch that was never pushed checks
    # out nothing.  GitHub sets HEAD from the first push; a local bare repo does not.
    subprocess.run(["git", "init", "-q", "-b", "main", "--bare", str(bare)], check=True)
    git(repo, "remote", "add", "origin", str(bare))
    assert publish(repo, "--yes", "--push", "origin", "Initial public release").returncode == 0

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(bare), str(clone)], check=True)

    assert git(clone, "log", "--all", "--format=%s") == "Initial public release"
    assert git(clone, "rev-list", "--all", "--count") == "1"
    assert (clone / "kit.py").exists()
    for gitignored in ("secrets", "model.ckpt", "data"):
        assert not (clone / gitignored).exists(), gitignored


# ---------------------------------------------------------------------------
# 2. Nothing gitignored, ever
# ---------------------------------------------------------------------------
def test_gitignored_paths_are_absent_from_the_published_tree(repo: Path):
    publish(repo, "--yes", "Initial public release")
    published = git(repo, "ls-tree", "-r", "--name-only", "public").splitlines()
    assert published == [".gitignore", "README.md", "kit.py", "scripts/publish.sh"]


def test_a_tracked_file_that_matches_an_ignore_rule_is_refused(repo: Path):
    """`git status` stays silent about these, so the check has to use --no-index.

    A file added before the ignore rule existed keeps being tracked, and would be
    published without a word.
    """
    (repo / "leftover.ckpt").write_bytes(b"weights")
    git(repo, "add", "-f", "leftover.ckpt")
    git(repo, "commit", "-q", "-m", "oops")

    result = publish(repo, "--yes", "Initial public release")
    assert result.returncode != 0
    assert "leftover.ckpt" in result.stderr
    assert "match a .gitignore rule" in result.stderr
    assert git(repo, "branch", "--list", "public") == ""


def test_a_tracked_development_workspace_is_refused(repo: Path):
    """`.superpowers/` by name: it is the one directory that must never go out."""
    (repo / ".superpowers").mkdir()
    (repo / ".superpowers" / "plan.md").write_text("internal\n")
    git(repo, "add", "-f", ".superpowers")
    git(repo, "commit", "-q", "-m", "accidentally added the workspace")

    result = publish(repo, "--yes", "Initial public release")
    assert result.returncode != 0
    assert ".superpowers" in result.stderr
    assert git(repo, "branch", "--list", "public") == ""


def test_an_unignored_development_workspace_is_refused(repo: Path):
    """Untracked is not enough: it has to be *ignored*, and that is asserted."""
    (repo / ".superpowers").mkdir()
    (repo / ".superpowers" / "plan.md").write_text("internal\n")

    result = publish(repo, "--yes", "Initial public release")
    assert result.returncode != 0
    assert ".superpowers" in result.stderr


def test_this_repository_keeps_its_workspace_untracked_and_self_ignoring():
    """The claim the script asserts, checked here against the real repository."""
    tracked = git(REPO_ROOT, "ls-files", "--", ".superpowers")
    assert tracked == "", f"tracked files under .superpowers: {tracked}"
    untracked = git(REPO_ROOT, "ls-files", "--others", "--exclude-standard", "--", ".superpowers")
    assert untracked == "", f"unignored files under .superpowers: {untracked}"


def test_nothing_tracked_in_this_repository_matches_an_ignore_rule():
    """The other half of the same guarantee, on the repository that will be published."""
    files = git(REPO_ROOT, "ls-tree", "-r", "--name-only", "main")
    matched = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "--no-index", "--stdin"],
        input=files,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert matched == "", f"tracked but gitignored: {matched}"


# ---------------------------------------------------------------------------
# 3. Refusals
# ---------------------------------------------------------------------------
def test_a_dirty_working_tree_is_refused(repo: Path):
    (repo / "kit.py").write_text("half an edit\n")
    result = publish(repo, "--yes", "Initial public release")
    assert result.returncode != 0
    assert "not clean" in result.stderr and "kit.py" in result.stderr
    assert git(repo, "branch", "--list", "public") == ""


def test_an_untracked_file_also_counts_as_dirty(repo: Path):
    """It is about to be committed or about to be deleted, and nobody knows which."""
    (repo / "new_module.py").write_text("unfinished\n")
    result = publish(repo, "--yes", "Initial public release")
    assert result.returncode != 0
    assert "new_module.py" in result.stderr


def test_publishing_from_the_wrong_branch_is_refused(repo: Path):
    git(repo, "switch", "-q", "-c", "experiment")
    result = publish(repo, "--yes", "Initial public release")
    assert result.returncode != 0
    assert "experiment" in result.stderr and "git switch main" in result.stderr
    assert git(repo, "branch", "--list", "public") == ""


def test_publishing_without_a_message_is_refused(repo: Path):
    result = publish(repo, "--yes")
    assert result.returncode != 0
    assert "no release message" in result.stderr
    assert git(repo, "branch", "--list", "public") == ""


def test_two_messages_are_refused_rather_than_silently_dropped(repo: Path):
    """`publish.sh Add the thing` -- unquoted -- must not publish as "Add"."""
    result = publish(repo, "--yes", "Add", "the thing")
    assert result.returncode != 0
    assert git(repo, "branch", "--list", "public") == ""


def test_a_prompt_that_cannot_be_asked_is_refused(repo: Path):
    """No terminal and no --yes: the confirmation is not optional by accident."""
    result = publish(repo, "Initial public release", input_text="")
    assert result.returncode != 0
    assert "--yes" in result.stderr
    assert git(repo, "branch", "--list", "public") == ""


def test_answering_no_publishes_nothing(repo: Path):
    result = publish(repo, "--yes", "--dry-run", "Initial public release")
    assert result.returncode == 0
    assert git(repo, "branch", "--list", "public") == ""


def test_republishing_an_unchanged_tree_is_refused(repo: Path):
    publish(repo, "--yes", "Initial public release")
    result = publish(repo, "--yes", "Same thing again")
    assert result.returncode != 0
    assert "nothing new to publish" in result.stderr
    assert git(repo, "rev-list", "--count", "public") == "1"


def test_a_public_branch_that_touches_main_is_refused(repo: Path):
    """Somebody rebased or merged the two: from here on the branch is not squashed."""
    git(repo, "branch", "public", "main")
    result = publish(repo, "--yes", "Initial public release")
    assert result.returncode != 0
    assert "shares" in result.stderr
    assert git(repo, "rev-parse", "public") == git(repo, "rev-parse", "main")


# ---------------------------------------------------------------------------
# 4. What it shows you before it does anything
# ---------------------------------------------------------------------------
def test_the_plan_shows_the_files_the_count_the_size_and_the_message(repo: Path):
    result = publish(repo, "--dry-run", "Initial public release")
    assert result.returncode == 0, result.stderr
    for expected in ("kit.py", "README.md", "scripts/publish.sh", ".gitignore"):
        assert expected in result.stdout
    assert "4 files" in result.stdout
    assert "Initial public release" in result.stdout
    assert "FIRST release" in result.stdout
    for gitignored in ("secrets/token.txt", "model.ckpt", "data/glorys.nc"):
        assert gitignored not in result.stdout


def test_dry_run_writes_nothing(repo: Path):
    head_before = git(repo, "rev-parse", "HEAD")
    result = publish(repo, "--dry-run", "Initial public release")
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert git(repo, "branch", "--list", "public") == ""
    assert git(repo, "rev-parse", "HEAD") == head_before
    assert git(repo, "status", "--porcelain") == ""


def test_publishing_leaves_the_working_tree_and_head_exactly_where_they_were(repo: Path):
    """No checkout happens, so an interrupted publish cannot strand your checkout."""
    head_before = git(repo, "rev-parse", "HEAD")
    publish(repo, "--yes", "Initial public release")
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(repo, "rev-parse", "HEAD") == head_before
    assert git(repo, "status", "--porcelain") == ""
    assert (repo / "secrets" / "token.txt").exists()  # untouched, just not published


def test_the_push_command_is_printed_and_the_remote_is_not_guessed(repo: Path):
    result = publish(repo, "--yes", "Initial public release")
    assert "Nothing has been pushed" in result.stdout
    assert "git push -u origin public:main" in result.stdout
    assert "git remote add origin" in result.stdout


def test_a_configured_remote_is_named_in_the_push_command(repo: Path, tmp_path: Path):
    bare = tmp_path / "public.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    git(repo, "remote", "add", "upstream", str(bare))

    result = publish(repo, "--yes", "Initial public release")
    assert "git push upstream public:main" in result.stdout
    assert str(bare) in result.stdout
    # Printed, not run.
    assert git(bare, "branch", "--list") == ""


def test_publishing_can_be_run_from_anywhere(repo: Path, tmp_path: Path):
    """A tutor will run it from wherever they are; it finds its own repository."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "publish.sh"), "--yes", "Initial public release"],
        capture_output=True,
        text=True,
        cwd=str(elsewhere),
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert git(repo, "rev-list", "--count", "public") == "1"
