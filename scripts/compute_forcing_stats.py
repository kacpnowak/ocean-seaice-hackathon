#!/usr/bin/env python
"""Normalisation statistics for the prescribed atmosphere.

    oceanarches/stats/ifs_1deg_forcing_stats.pt    mean and std per forcing channel

The GLORYS statistics in ``glorys_1deg_stats.pt`` cover the model *state*.  The
forcing is not part of that state, is never scored, and -- unlike the ocean --
is defined over land as well, so it needs its own file and its own rule:
**every cell counts, ocean and land alike.**  Averaging the atmosphere over
ocean only would give a 2 m temperature that is wrong by several kelvin over the
continents, which is exactly where the model most needs it.

    make forcing-stats                                  # the whole archive
    python scripts/compute_forcing_stats.py              # the same thing
    python scripts/compute_forcing_stats.py --quick      # 64 times, for a smoke test
    python scripts/compute_forcing_stats.py --path /elsewhere --out my_stats.pt

Measured on the shipped archive: 8.4 s for all 478 valid times x 7 channels.

The statistics are computed over the archive's **valid times** after duplicate
valid times have been resolved (shortest forecast lead wins), i.e. over exactly
the fields ``XarrayForcing`` can serve -- see
``oceanarches/dataloaders/forcing.py``.

Which times these are built from is the same choice ``scripts/compute_stats.py``
documents, and here it is not a choice at all: the shipped archive is one year
(2024, the holdout split) and the statistics are built from all of it.  Nothing
is scored on them -- they only set the scale of the model's *input* channels --
but say so anyway when you report a forced run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oceanarches import paths  # noqa: E402
from oceanarches.dataloaders.forcing import XarrayForcing  # noqa: E402

#: The seven channels ``configs/forcing/file.yaml`` reserves, in that order.
#: Keep the two lists in step: the order is the order the embedder sees.
#:
#: ``somslpre`` (mean sea-level pressure) is deliberately absent.  It is a
#: variable of the shipped archive, but it is NaN over the whole grid in all 520
#: records of all 52 files, so it carries no information at all.
DEFAULT_VARIABLES = [
    "sowinu10",
    "sowinv10",
    "sotemair",
    "sod2m",
    "sowaprec",
    "sosudosw",
    "sosudolw",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path", type=Path, default=None, help="Forcing archive. Default: IFS_FORCING."
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=DEFAULT_VARIABLES,
        help="Channels to compute statistics for. Default: the seven in configs/forcing/file.yaml.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output .pt. Default: paths.forcing_stats_file()."
    )
    parser.add_argument(
        "--max-times",
        type=int,
        default=None,
        help="Use an evenly spaced sample of this many valid times instead of all of them.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Same as --max-times 64 (smoke tests only)."
    )
    args = parser.parse_args()

    root = args.path or paths.ifs_forcing()
    out = args.out or paths.forcing_stats_file()
    max_times = 64 if args.quick else args.max_times

    # normalize=False: this script *is* where the statistics come from, so asking
    # the source to normalise itself would be circular.  cache=False because a
    # single streaming pass has nothing to re-read.
    forcing = XarrayForcing(root, args.variables, normalize=False, cache=False)
    print(f"forcing archive : {root}  ({len(forcing.files)} file(s))")
    first, last = forcing.time_range
    print(
        f"valid times     : {forcing.n_times} distinct, {first} to {last} "
        f"(from {forcing.n_records} records; {forcing.n_duplicate_valid_times} were "
        "duplicate valid times, resolved by shortest lead)"
    )
    low, high = forcing.lead_time_range
    print(f"forecast leads  : {low:g} to {high:g} h")
    print(f"channels        : {args.variables}")
    if max_times is not None:
        print(f"sampling        : {max_times} evenly spaced times (NOT the whole archive)")
    print()

    mean, std = forcing.compute_statistics(max_times=max_times)
    for name, m, s in zip(args.variables, mean.tolist(), std.tolist()):
        print(f"  {name:10s} mean {m:14.6g}   std {s:14.6g}")

    stats = {
        "variables": list(args.variables),
        "mean": mean.reshape(len(args.variables), 1, 1, 1),
        "std": std.reshape(len(args.variables), 1, 1, 1),
        "n_times": forcing.n_times if max_times is None else min(max_times, forcing.n_times),
        "time_range": [str(first), str(last)],
        "source": str(root),
        "over": "every cell, ocean and land (the atmosphere is defined everywhere)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
