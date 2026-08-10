"""Where the statistics behind a score came from, and saying so on the report.

``make stats-quick`` builds the normalisation statistics from 20 dates instead
of 400 and the climatology from a third of the prepared years, in about a minute
instead of eight.  That is a good trade while you are getting the pipeline to
run and a bad one the moment a number leaves the terminal: **the climatology is
one of the two baselines every score in the report is quoted against**, so a
sampled climatology is a sampled comparison.

Nothing downstream could tell.  ``make doctor`` passed identically either way and
the report printed five-significant-figure scorecards with no mention of it, so a
participant could beat a sampled baseline in front of a room and have no way to
know.  This module reads the provenance the two artefacts record and
:func:`oceanarches.evaluation.report.build_report` puts it on the face of the
report -- the same treatment :func:`~oceanarches.evaluation.report.holdout_caution`
gives a score from a split that must not be quoted.

This module answers *which* statistics, in words a participant can read.  It is
not what keeps a cache honest: rebuilding ``oceanarches/stats/`` changes the
rollout cache key through
:func:`oceanarches.evaluation.rollout.statistics_fingerprint`, which hashes the
artefacts' bytes, so a cache built on sampled statistics is recomputed rather
than reused.  The human-readable provenance still travels in the cache manifest,
so a report built from a cache describes the statistics that actually produced
those numbers.

One deliberate choice: an artefact that records nothing is read as *unknown*,
never as *full*.  The depth is inferred from the numbers only where the numbers
cannot be misread -- ``n_dates`` below the full-build default is a sample
whichever version of ``compute_stats.py`` wrote it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

__all__ = [
    "DEFAULT_N_DATES",
    "StatisticsProvenance",
    "read_statistics_provenance",
]

#: ``scripts/compute_stats.py --n-dates`` default: what a full build uses.
DEFAULT_N_DATES = 400

#: What ``scripts/compute_stats.py`` writes into the ``sampling`` key/attribute.
FULL = "full"
SAMPLED = "sampled"


@dataclass(frozen=True)
class StatisticsProvenance:
    """How deeply ``oceanarches/stats/`` was sampled, as the artefacts record it."""

    stats_years: str = ""
    stats_n_dates: int = 0
    stats_n_years: int = 0
    #: ``"full"``, ``"sampled"``, or ``""`` when the artefact records nothing.
    stats_sampling: str = ""
    climatology_years: str = ""
    climatology_n_years: int = 0
    climatology_n_years_available: int = 0
    climatology_sampling: str = ""
    #: Artefacts that could not be read at all, by name.
    unreadable: tuple[str, ...] = ()

    # -- what it means ------------------------------------------------------
    @property
    def stats_sampled(self) -> bool:
        if self.stats_sampling:
            return self.stats_sampling == SAMPLED
        # Written before this provenance existed: the count alone is decisive.
        return 0 < self.stats_n_dates < DEFAULT_N_DATES

    @property
    def climatology_sampled(self) -> bool:
        if self.climatology_sampling:
            return self.climatology_sampling == SAMPLED
        return 0 < self.climatology_n_years < self.climatology_n_years_available

    @property
    def sampled(self) -> bool:
        return self.stats_sampled or self.climatology_sampled

    @property
    def known(self) -> bool:
        """True when at least one artefact said something about itself."""
        return bool(
            self.stats_years
            or self.stats_n_dates
            or self.climatology_years
            or self.climatology_n_years
        )

    # -- how it is written down ---------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data["unreadable"] = list(self.unreadable)
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "StatisticsProvenance":
        """Rebuild from a manifest entry, ignoring keys a newer version added."""
        if not data:
            return cls()
        fields = {f for f in cls.__dataclass_fields__}
        kept = {k: v for k, v in data.items() if k in fields}
        if "unreadable" in kept:
            kept["unreadable"] = tuple(kept["unreadable"] or ())
        return cls(**kept)

    def describe_stats(self, mark_sampled: bool = True) -> str:
        """``400 dates from 1993-2025``.  ``mark_sampled`` adds the verdict, which
        a sentence that has already said "sampled" does not want repeated."""
        if "normalisation statistics" in self.unreadable:
            return "not readable"
        if not self.stats_n_dates and not self.stats_years:
            return "not recorded"
        dates = f"{self.stats_n_dates} dates" if self.stats_n_dates else "an unrecorded number"
        years = f" from {self.stats_years}" if self.stats_years else ""
        depth = " (sampled)" if self.stats_sampled and mark_sampled else ""
        return f"{dates}{years}{depth}"

    def describe_climatology(self, mark_sampled: bool = True) -> str:
        """``33 of 33 years (1993-2025)``."""
        if "climatology" in self.unreadable:
            return "not readable"
        if not self.climatology_n_years and not self.climatology_years:
            return "not recorded"
        if self.climatology_n_years and self.climatology_n_years_available:
            count = f"{self.climatology_n_years} of {self.climatology_n_years_available} years"
        elif self.climatology_n_years:
            count = f"{self.climatology_n_years} years"
        else:
            count = "years the file did not record"
        years = f" ({self.climatology_years})" if self.climatology_years else ""
        depth = " (sampled)" if self.climatology_sampled and mark_sampled else ""
        return f"{count}{years}{depth}"

    def summary_line(self) -> str:
        """One line for the terminal, before the expensive part starts."""
        prefix = "statistics: SAMPLED -- " if self.sampled else "statistics: "
        return (
            f"{prefix}normalisation from {self.describe_stats()}, "
            f"climatology from {self.describe_climatology()}"
        )

    def rows(self) -> list[list[str]]:
        """Rows for the report's provenance table, or none when nothing is known.

        Two rows reading "not recorded" would be worse than no rows: they claim
        the question was asked of the artefacts, which it was not.
        """
        if not self.known and not self.unreadable:
            return []
        return [
            ["Normalisation statistics", self.describe_stats()],
            ["Climatology (also a scored baseline)", self.describe_climatology()],
        ]

    # -- the warning that goes on the face of the report --------------------
    def caution(self) -> str:
        """The warning to print above a report built on sampled statistics, or "".

        Deliberately shaped like
        :func:`~oceanarches.evaluation.report.holdout_caution`: a report is what
        gets pasted into a slide, and by then the ``make stats-quick`` that
        produced it is long out of the scrollback.
        """
        if not self.sampled:
            return ""
        pieces = []
        if self.stats_sampled:
            pieces.append(
                f"the normalisation statistics come from {self.describe_stats(False)}, "
                f"where a full build uses {DEFAULT_N_DATES} dates"
            )
        if self.climatology_sampled:
            pieces.append(f"the climatology comes from {self.describe_climatology(False)}")
        detail = "; and ".join(pieces)
        baseline = (
            " The climatology is one of the two baselines every number below is quoted "
            "against, so these are real measurements against a sampled baseline."
            if self.climatology_sampled
            else ""
        )
        return (
            f"**These numbers were produced with sampled statistics (`make stats-quick`): "
            f"{detail}.{baseline} Rebuild them with `make stats` (~8 min) and re-run the "
            "evaluation before reporting any of these numbers -- the rebuild invalidates "
            "the cached rollout by itself, so the numbers really are rescored.**"
        )


def _read_stats(path: Path) -> dict:
    import torch

    return torch.load(path, weights_only=True)


def _read_climatology_attrs(path: Path) -> dict:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return dict(ds.attrs)


def read_statistics_provenance(
    stats_file: Path | None = None, climatology_file: Path | None = None
) -> StatisticsProvenance:
    """Read what ``oceanarches/stats/`` records about how it was built.

    Never raises: a missing or unreadable artefact is reported as such, because
    this runs on the way to an evaluation that must not be stopped by it.

    Args:
        stats_file: ``glorys_1deg_stats.pt``; defaults to :func:`oceanarches.paths.stats_file`.
        climatology_file: ``glorys_1deg_climatology.nc``; defaults to
            :func:`oceanarches.paths.climatology_file`.

    Returns:
        A :class:`StatisticsProvenance`.  ``known`` is False when neither
        artefact said anything.
    """
    from .. import paths

    stats_file = Path(stats_file or paths.stats_file())
    climatology_file = Path(climatology_file or paths.climatology_file())

    provenance = StatisticsProvenance()
    unreadable: list[str] = []

    try:
        stats = _read_stats(stats_file)
    except Exception:  # noqa: BLE001 - provenance must never stop an evaluation
        unreadable.append("normalisation statistics")
    else:
        provenance = replace(
            provenance,
            stats_years=str(stats.get("years", "") or ""),
            stats_n_dates=int(stats.get("n_dates", 0) or 0),
            stats_n_years=int(stats.get("n_years", 0) or 0),
            stats_sampling=str(stats.get("sampling", "") or ""),
        )

    try:
        attrs = _read_climatology_attrs(climatology_file)
    except Exception:  # noqa: BLE001
        unreadable.append("climatology")
    else:
        provenance = replace(
            provenance,
            climatology_years=str(attrs.get("years", "") or ""),
            climatology_n_years=int(attrs.get("n_years", 0) or 0),
            climatology_n_years_available=int(attrs.get("n_years_available", 0) or 0),
            climatology_sampling=str(attrs.get("sampling", "") or ""),
        )

    return replace(provenance, unreadable=tuple(unreadable))
