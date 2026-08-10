"""The rule `tests/conftest.py` applies when `make stats` has never been run.

A fresh clone has no `oceanarches/stats/*`: they are generated, and git-ignored.
Before the hook in `conftest.py` existed, `make test` on such a clone gave 45
failures, all of them the same `FileNotFoundError` for the mask file, and none
of them saying "run `make stats`". They are skips now.

What is tested here is the decision itself, because the two halves of it pull in
opposite directions: an artefact that is absent must excuse a failure, and an
artefact that is *present but wrong* must not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import _missing_generated_artefact

from oceanarches import paths


def test_a_missing_artefact_is_recognised_and_names_its_command(monkeypatch, tmp_path):
    absent = tmp_path / "glorys_1deg_masks.nc"
    monkeypatch.setattr(paths, "masks_file", lambda: absent)

    error = FileNotFoundError(f"Masks file not found: {absent}\nrun: make stats")

    assert _missing_generated_artefact(error) == (absent, "make stats")


def test_the_forcing_statistics_name_their_own_target(monkeypatch, tmp_path):
    absent = tmp_path / "ifs_1deg_forcing_stats.pt"
    monkeypatch.setattr(paths, "forcing_stats_file", lambda: absent)

    error = FileNotFoundError(f"no forcing statistics at {absent}")

    assert _missing_generated_artefact(error) == (absent, "make forcing-stats")


def test_an_artefact_that_exists_is_never_excused(monkeypatch, tmp_path):
    """Present but wrong has to stay a failure -- that is the whole risk here."""
    present = tmp_path / "glorys_1deg_masks.nc"
    present.write_bytes(b"not really a netCDF file")
    monkeypatch.setattr(paths, "masks_file", lambda: present)

    error = FileNotFoundError(f"Masks file not found: {present}")

    assert _missing_generated_artefact(error) is None


def test_an_unrelated_missing_file_is_not_excused(tmp_path):
    error = FileNotFoundError(2, "No such file or directory", str(tmp_path / "elsewhere.nc"))

    assert _missing_generated_artefact(error) is None


def test_another_error_about_the_artefact_is_not_excused(monkeypatch, tmp_path):
    """Only "it is not there" is excused, not "it is there and it is broken"."""
    absent = tmp_path / "glorys_1deg_masks.nc"
    monkeypatch.setattr(paths, "masks_file", lambda: absent)

    assert _missing_generated_artefact(ValueError(f"cannot open {absent}")) is None


def test_a_wrapped_missing_artefact_is_found_through_the_chain(monkeypatch, tmp_path):
    """hydra wraps it: `InstantiationException` with the real error as its cause."""
    absent = tmp_path / "glorys_1deg_masks.nc"
    monkeypatch.setattr(paths, "masks_file", lambda: absent)

    try:
        try:
            raise FileNotFoundError(f"Masks file not found: {absent}")
        except FileNotFoundError as cause:
            raise RuntimeError("Error in call to target 'OceanEncodeDecodeLayer'") from cause
    except RuntimeError as wrapped:
        assert _missing_generated_artefact(wrapped) == (absent, "make stats")


def test_the_filename_alone_is_matched_too(monkeypatch, tmp_path):
    """`OSError.filename`, not only the message text."""
    absent = tmp_path / "glorys_1deg_climatology.nc"
    monkeypatch.setattr(paths, "climatology_file", lambda: absent)

    error = FileNotFoundError(2, "No such file or directory")
    error.filename = str(absent)

    assert _missing_generated_artefact(error) == (absent, "make stats")


def test_the_hook_itself_turns_that_failure_into_a_skip(monkeypatch, tmp_path):
    """End to end through the wrapper, driven by hand as pluggy drives it."""
    import conftest

    absent = tmp_path / "glorys_1deg_masks.nc"
    monkeypatch.setattr(paths, "masks_file", lambda: absent)

    wrapper = conftest.pytest_runtest_call(item=None)
    next(wrapper)  # up to the `yield`, i.e. the test is now running
    with pytest.raises(pytest.skip.Exception) as skipped:
        wrapper.throw(FileNotFoundError(f"Masks file not found: {absent}"))

    assert "make stats" in str(skipped.value)
    assert "glorys_1deg_masks.nc" in str(skipped.value)


def test_the_hook_re_raises_anything_else(tmp_path):
    import conftest

    wrapper = conftest.pytest_runtest_call(item=None)
    next(wrapper)
    with pytest.raises(AssertionError, match="a real failure"):
        wrapper.throw(AssertionError("a real failure"))


@pytest.mark.parametrize(
    "accessor", ["masks_file", "stats_file", "climatology_file", "forcing_stats_file"]
)
def test_every_generated_artefact_is_covered(accessor):
    """Whatever `paths` calls generated, the hook knows how to rebuild."""
    from conftest import _generated_artefacts

    covered = {path for path, _ in _generated_artefacts()}
    assert getattr(paths, accessor)() in covered
    assert all(path.parent == Path(paths.STATS_DIR) for path, _ in _generated_artefacts())
