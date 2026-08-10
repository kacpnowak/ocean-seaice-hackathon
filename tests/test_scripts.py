"""The two data-preparation scripts, which had no test references at all.

Both are pure-decision tests: nothing here reads or writes GLORYS.  The
behaviour they pin is the behaviour a participant meets on the command line.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compute_stats  # noqa: E402
import prepare_glorys  # noqa: E402


# ---------------------------------------------------------------------------
# prepare_glorys: refusing a year the archive has not finished
# ---------------------------------------------------------------------------
def test_a_year_the_archive_has_not_finished_is_refused():
    """`make prep-data` with no YEARS prepares everything it finds.

    The raw archive grows. When it gained the first 83 days of 2026, the bare
    target prepared a quarter-year file that then sat in GLORYS_PREPPED looking
    exactly like a whole year -- in the file listing, in `make doctor` and in the
    split filters. A review agent tripped this for real.

    MUTANT: making `refusal_reason` return None unconditionally fails the first
    assertion.
    """
    missing_2026 = [
        date(2026, 4, 1) + __import__("datetime").timedelta(days=i) for i in range(282)
    ]
    reason = prepare_glorys.refusal_reason(2026, missing_2026, max_missing=10)
    assert reason is not None
    assert "282 of 365" in reason and "--allow-incomplete" in reason


def test_a_year_with_a_couple_of_gaps_is_still_prepared():
    """2003 really is missing 2003-02-07 and 2003-02-11 and is a good training year.

    A rule that refused every non-continuous year would drop it, so the guard is
    about scale, not about perfection.
    """
    assert prepare_glorys.refusal_reason(2003, [date(2003, 2, 7), date(2003, 2, 11)], 10) is None
    assert prepare_glorys.refusal_reason(2015, [], 10) is None


def test_the_refusal_threshold_is_the_flag_it_names():
    missing = [date(2020, 3, 1) + timedelta(days=i) for i in range(11)]
    assert prepare_glorys.refusal_reason(2020, missing, max_missing=10) is not None
    assert prepare_glorys.refusal_reason(2020, missing, max_missing=11) is None


def test_prepare_glorys_year_spellings():
    assert prepare_glorys.parse_years(["2015"]) == [2015]
    assert prepare_glorys.parse_years(["2015", "2016"]) == [2015, 2016]
    assert prepare_glorys.parse_years(["1993-1995", "2020"]) == [1993, 1994, 1995, 2020]


# ---------------------------------------------------------------------------
# compute_stats: --years
# ---------------------------------------------------------------------------
def _archive(tmp_path: Path, years) -> Path:
    root = tmp_path / "prepped"
    root.mkdir(exist_ok=True)
    for year in years:
        (root / f"glorys_1deg_{year}.nc").write_bytes(b"x")
    return root


def test_compute_stats_years_selects_a_subset(tmp_path):
    """The statistics and the climatology are built from every prepared year by
    default -- test and holdout included -- and `--years 1993-2018` is what makes
    a strictly-train version.

    MUTANT: ignoring the `years` argument in `prepared_files` returns all 33 and
    the first assertion fails.
    """
    root = _archive(tmp_path, range(1993, 2026))
    train_only = compute_stats.prepared_files(root, compute_stats.parse_years(["1993-2018"]))
    assert len(train_only) == 26
    assert compute_stats.years_label(train_only) == "1993-2018"
    assert len(compute_stats.prepared_files(root)) == 33
    assert compute_stats.years_label(compute_stats.prepared_files(root)) == "1993-2025"


def test_compute_stats_years_that_are_not_prepared_are_an_error(tmp_path):
    root = _archive(tmp_path, [2015, 2016])
    with pytest.raises(SystemExit, match="2015-2016"):
        compute_stats.prepared_files(root, compute_stats.parse_years(["2021-2023"]))


def test_a_non_contiguous_selection_is_labelled_year_by_year(tmp_path):
    """`years_label` goes into the stats file, so it must not claim a range it
    does not have."""
    root = _archive(tmp_path, [2015, 2017, 2018])
    assert compute_stats.years_label(compute_stats.prepared_files(root)) == "2015,2017,2018"


def test_the_shipped_statistics_record_the_years_they_came_from():
    """Provenance: a checkpoint's statistics must be traceable without guessing.

    MUTANT: dropping the `years` key from `compute_statistics`'s return value
    fails this for any regenerated file.
    """
    import torch

    from oceanarches import paths

    if not paths.stats_file().exists():
        pytest.skip("statistics not generated; run: make stats")
    stats = torch.load(paths.stats_file(), weights_only=True)
    assert "years" in stats, "glorys_1deg_stats.pt records no `years`; re-run make stats"
    assert isinstance(stats["years"], str) and stats["years"]


def test_a_sampled_build_is_labelled_as_one(tmp_path):
    """`make stats-quick` used to leave nothing on disk that told it apart from
    `make stats`: the climatology's `years` attribute read `1993-2025` whether
    it came from 33 years or from 3 spread across the same span, and the
    evaluation report then quoted five-figure scores against it.

    MUTANT: making `sampling_label` return `"full"` unconditionally fails the
    first two assertions.
    """
    assert compute_stats.sampling_label(20, 400) == "sampled"
    assert compute_stats.sampling_label(3, 33) == "sampled"
    assert compute_stats.sampling_label(400, 400) == "full"
    # A short archive is not a sampled build: using everything there is is full.
    assert compute_stats.sampling_label(365, 365) == "full"


def test_the_shipped_statistics_record_how_deeply_they_were_sampled():
    """Provenance the *report* reads, not only a person.

    MUTANT: dropping `sampling` from `compute_statistics`'s return value or from
    the climatology's attributes fails this for any regenerated file.
    """
    import torch
    import xarray as xr

    from oceanarches import paths
    from oceanarches.evaluation import provenance

    if not paths.stats_file().exists() or not paths.climatology_file().exists():
        pytest.skip("statistics not generated; run: make stats")

    stats = torch.load(paths.stats_file(), weights_only=True)
    assert stats.get("sampling") in ("full", "sampled")
    assert stats["n_dates"] and stats["n_years"]

    with xr.open_dataset(paths.climatology_file()) as climatology:
        attrs = climatology.attrs
    assert attrs.get("sampling") in ("full", "sampled")
    assert int(attrs["n_years"]) and int(attrs["n_years_available"])

    # ... and the evaluation reads it back as what it is.
    read = provenance.read_statistics_provenance()
    assert read.known and not read.unreadable
    assert read.stats_sampled == (stats["sampling"] == "sampled")
