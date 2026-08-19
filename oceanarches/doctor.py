"""``make doctor`` -- check that this machine can actually run the challenge.

Run this first, and run it again whenever something breaks.  It prints one line
per check with a clear PASS / WARN / FAIL and, for anything that is not PASS, the
exact command that fixes it.
"""

from __future__ import annotations

import errno
import importlib
import os
import shutil
import socket
import sys
from collections.abc import Mapping
from pathlib import Path

from . import guards, paths

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_COLOR = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
_RESET = "\033[0m"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "", fix: str = "") -> None:
        self.rows.append((status, name, detail, fix))

    def render(self) -> int:
        width = max(len(name) for _, name, _, _ in self.rows) + 2
        print()
        for status, name, detail, fix in self.rows:
            colour = _COLOR[status] if sys.stdout.isatty() else ""
            reset = _RESET if sys.stdout.isatty() else ""
            print(f"  {colour}{status:<4}{reset}  {name:<{width}} {detail}")
            if status != PASS and fix:
                print(f"        {'':<{width}} -> {fix}")
        failures = sum(1 for s, *_ in self.rows if s == FAIL)
        warnings = sum(1 for s, *_ in self.rows if s == WARN)
        print()
        if failures:
            print(f"  {failures} failure(s), {warnings} warning(s). Fix the failures above.")
        elif warnings:
            print(f"  No failures, {warnings} warning(s) -- you can start working.")
        else:
            print("  Everything looks good. Next: docs/03_first_model.md")
        print()
        return 1 if failures else 0


def _check_python(report: Report) -> None:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11) and (major, minor) < (3, 13)
    report.add(
        PASS if ok else FAIL,
        "python",
        f"{sys.version.split()[0]} ({sys.executable})",
        "geoarches needs >=3.11,<3.13. Re-run: make setup",
    )


def _check_imports(report: Report) -> None:
    required = [
        "torch",
        "lightning",
        "hydra",
        "tensordict",
        "xarray",
        "netCDF4",
        "zarr",
        "geoarches",
        "matplotlib",
        "cartopy",
    ]
    # The *reason* an import failed, not only its name.  "missing: cartopy"
    # sends a participant to `make setup`, which reinstalls a package that is
    # already installed; the real cause is usually an OSError naming a shared
    # library the node does not have, and that is the only thing that tells them
    # what to do next.
    failures: list[str] = []
    for name in required:
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - we only care that it did not import
            failures.append(f"{name} ({type(error).__name__}: {error})")
    report.add(
        PASS if not failures else FAIL,
        "imports",
        "all core packages import" if not failures else "did not import: " + "; ".join(failures),
        "make setup  (unless the message above names a missing shared library, "
        "which reinstalling will not fix)",
    )

    # Optional: the spherical power spectrum needs pyshtools, which is not
    # available on every platform.  Evaluation degrades gracefully without it.
    try:
        importlib.import_module("geoarches.metrics.spherical_power_spectrum")
        report.add(PASS, "power spectrum", "pyshtools available")
    except Exception as exc:  # noqa: BLE001
        report.add(
            WARN,
            "power spectrum",
            f"unavailable ({type(exc).__name__}) -- spectra will be skipped",
            "optional: pip install pyshtools",
        )


def have_allocation(env: Mapping[str, str] | None = None) -> bool:
    """Is this process inside a SLURM allocation?

    `SLURM_JOB_ID`, and nothing else.  Not `torch.cuda.is_available()`: measured on
    the login node `jpbl-s02-02`, CUDA *is* available -- the node carries a real
    card at index 0 -- while `nvidia-smi` showed it at 99% utilisation with 49 GiB
    held by another user's job.  Three of six trial participants trained there
    believing they were on a compute node, because this file used to PASS the GPU
    check and say nothing else.
    """
    env = os.environ if env is None else env
    return bool(env.get("SLURM_JOB_ID"))


def _check_allocation(report: Report) -> None:
    """WARN whenever there is no allocation -- whatever CUDA says.

    Skipped entirely off SLURM (no `srun` on PATH): on a laptop the warning would
    be noise, and the hostname hint below is cosmetic.
    """
    if shutil.which("srun") is None:
        return
    host = socket.gethostname()
    if have_allocation():
        detail = (
            f"SLURM job {os.environ['SLURM_JOB_ID']}, "
            f"{os.environ.get('SLURM_NNODES', '1')} node(s), on {host}"
        )
        report.add(PASS, "allocation", detail)
        return
    # Hostname is the secondary signal, never the only one: JURECA's login nodes
    # carry "login" in the name. A compute node reached by ssh has no
    # SLURM_JOB_ID either and is just as shared, which is why the WARN above does
    # not depend on this at all -- the annotation is cosmetic and an unrecognised
    # host simply loses it.
    where = " (a login node)" if host.startswith("jpbl") or "login" in host else ""
    report.add(
        WARN,
        "allocation",
        f"none -- SLURM_JOB_ID is unset, on {host}{where}",
        f"anything that trains or evaluates needs a node of your own: {guards.SRUN_ONE_LINE}",
    )


def _check_gpu(report: Report) -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        report.add(FAIL, "gpu", "torch did not import", "make setup")
        return

    if not torch.cuda.is_available():
        report.add(
            WARN,
            "gpu",
            f"no CUDA device visible (torch {torch.__version__})",
            f"expected on a login node without an allocation. Before training: {guards.SRUN_ONE_LINE}",
        )
        return

    names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    detail = f"{torch.cuda.device_count()}x {', '.join(names)} (torch {torch.__version__})"
    # How much of it is actually yours.  A card that is 85 of 98 GiB full is the
    # difference between "PASS, start training" and a CUDA OOM twenty minutes in,
    # and `is_available()` reports both identically.
    free = _free_memory(torch)
    if free is not None:
        detail += f", {free[0]:.0f} of {free[1]:.0f} GiB free on device 0"
    if not have_allocation() and shutil.which("srun") is not None:
        report.add(
            WARN,
            "gpu",
            detail + " -- visible, but NOT allocated to you",
            "a visible card is not an idle one; see the `allocation` line above",
        )
        return
    if free is not None and free[0] < 0.5 * free[1]:
        report.add(
            WARN,
            "gpu",
            detail + " -- over half of device 0 is already in use",
            "another job is on this card; `tiny` alone peaks at 48.6 GiB",
        )
        return
    report.add(PASS, "gpu", detail)


def _free_memory(torch) -> tuple[float, float] | None:
    """`(free, total)` GiB on device 0, or None if CUDA will not say."""
    try:
        free, total = torch.cuda.mem_get_info(0)
    except Exception:  # noqa: BLE001 - a busy or exclusive-mode card can refuse
        return None
    return free / 2**30, total / 2**30


def _check_ffmpeg(report: Report) -> None:
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        report.add(PASS, "ffmpeg", Path(exe).name)
    except Exception:  # noqa: BLE001
        if shutil.which("ffmpeg"):
            report.add(PASS, "ffmpeg", "system ffmpeg")
        else:
            report.add(
                WARN,
                "ffmpeg",
                "not found -- animations will fall back to GIF",
                "pip install imageio-ffmpeg",
            )


def _check_data(report: Report) -> None:
    raw = paths.glorys_raw()
    prepped = paths.glorys_prepped()
    files = sorted(prepped.glob("glorys_1deg_*.nc")) if prepped.is_dir() else []

    # The raw archive is an INPUT TO `make prep-data`, and nothing else reads it.
    # On JURECA it was deliberately not copied -- 640 GB to reproduce a 92 GB
    # output that came across already prepared -- so a hard FAIL here would fail
    # every participant's first `make doctor` for something that cannot affect
    # any run they will do. It only matters when there is nothing prepared
    # either, and then it is the real problem and says so.
    if raw.is_dir():
        years = sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name.isdigit())
        span = f"{years[0]}-{years[-1]} ({len(years)} years)" if years else "no year dirs"
        report.add(PASS, "raw GLORYS", f"{raw}  {span}")
    elif files:
        report.add(
            WARN,
            "raw GLORYS",
            f"{raw} does not exist -- not needed, the prepared data below is here",
            "only `make prep-data` reads it; set GLORYS_RAW in config.env if you must re-prep",
        )
    else:
        report.add(
            FAIL,
            "raw GLORYS",
            f"{raw} does not exist, and there is no prepared data either",
            "set GLORYS_RAW in config.env, then: make prep-data",
        )

    if not files:
        report.add(
            FAIL,
            "prepared data",
            f"nothing in {prepped}",
            'make prep-data YEARS="2015 2016"  (or: make prep-data-slurm)',
        )
        return

    detail = f"{prepped}  {len(files)} yearly files ({files[0].stem[-4:]}-{files[-1].stem[-4:]})"
    # Counting files is not the same as covering a split, and the difference is
    # the whole failure this check exists to catch: the fix line above suggests
    # two years, and two years pass a file-count check while `make eval` -- which
    # scores on `test` -- dies inside geoarches with "filename_filter filtered
    # all files under path", naming neither the split nor the years.
    missing = _splits_not_covered(files)
    if missing:
        first = missing[0]
        report.add(
            WARN,
            "prepared data",
            detail + "; incomplete: " + ", ".join(f"{name} ({years})" for name, years in missing),
            f'make prep-data YEARS="{first[1].replace("-", " ").split()[0]}..." '
            "(or: make prep-data-slurm for all 33 years). Training or scoring on a "
            "split whose years are absent fails inside the dataloader.",
        )
    else:
        report.add(PASS, "prepared data", detail)


def _splits_not_covered(files) -> list[tuple[str, str]]:
    """``[("test", "2021-2023: missing 2022, 2023"), ...]`` for the scoring splits.

    Only the four calendar splits are checked, not the ``tiny_*`` ones (they are
    subsets of ``train``/``val``) and not ``all``.
    """
    from .dataloaders.glorys import SPLIT_YEARS

    present = set()
    for file in files:
        stem = file.stem[-4:]
        if stem.isdigit():
            present.add(int(stem))
    incomplete = []
    for name in ("train", "val", "test", "holdout"):
        first, last = SPLIT_YEARS[name]
        absent = [year for year in range(first, last + 1) if year not in present]
        if absent:
            listed = ", ".join(str(year) for year in absent[:4])
            if len(absent) > 4:
                listed += f", ... ({len(absent)} years)"
            incomplete.append((name, f"{first}-{last}: missing {listed}"))
    return incomplete


def _check_stats(report: Report) -> None:
    """The three generated artefacts, opened rather than merely counted.

    "The file is there" is weak evidence: a `make stats-quick` run, an
    interrupted one and a full one all leave a file of about the right size, and
    a mask file on the wrong grid or a statistics file missing `delta_std` fails
    much later, inside a training run.  So each one is opened and the fields the
    rest of the kit actually reads are checked.
    """
    # `make stats`, never `make stats-quick`: docs/01 section 1.6 forbids the
    # quick statistics for any model whose numbers you intend to report, and a
    # fresh clone has no statistics at all, so this line is the first thing a
    # participant runs.  Pointing it at the quick path would put every clone on
    # statistics computed from a handful of dates.
    for label, file, fix in [
        ("masks", paths.masks_file(), "make stats  (~8 min)"),
        ("norm stats", paths.stats_file(), "make stats  (~8 min)"),
        ("climatology", paths.climatology_file(), "make stats  (~8 min)"),
    ]:
        if not file.exists():
            report.add(FAIL if label != "climatology" else WARN, label, "not generated", fix)
            continue
        size = _human_size(file.stat().st_size)
        problem = _inspect_artefact(label, file)
        if problem is None:
            report.add(PASS, label, f"{file.name} ({size})")
        else:
            report.add(
                FAIL if label != "climatology" else WARN,
                label,
                f"{file.name} ({size}): {problem}",
                fix,
            )

    _check_stats_depth(report)


def _check_stats_depth(report: Report) -> None:
    """How deeply the statistics were sampled -- not merely that they are readable.

    `make stats-quick` builds the normalisation statistics from 20 dates instead
    of 400 and the climatology from a third of the years, in one minute instead
    of eight.  Until now it left no trace anywhere a participant would look:
    every check above PASSed identically either way, so someone could save eight
    minutes, beat a sampled climatology in front of a room, and have nothing tell
    them or their audience that the baseline was a sample.  The climatology *is*
    one of the two baselines every score is quoted against.

    `scripts/compute_stats.py` now records the depth in both artefacts and every
    evaluation report carries it; this is the other place people look, before the
    numbers exist rather than after.  The comment above -- "`make stats`, never
    `make stats-quick`" -- was true and unenforced, and is now checkable.

    Nothing here can fail the doctor: unreadable or unrecorded artefacts are
    reported by the checks above, and `read_statistics_provenance` never raises.
    """
    from .evaluation.provenance import read_statistics_provenance

    provenance = read_statistics_provenance()
    if provenance.sampled:
        report.add(
            WARN,
            "stats depth",
            provenance.summary_line(),
            "make stats  (~8 min) before you report any number -- every evaluation "
            "report built on these says on its face that they were sampled",
        )
    elif provenance.known:
        report.add(PASS, "stats depth", provenance.summary_line())
    # Neither: the artefacts are missing or predate the provenance, and the rows
    # above have already said so.  A "not recorded" row would claim the question
    # was asked of them, which it was not.


def _human_size(n_bytes: int) -> str:
    """``3.9 KB``, not ``0.0 MB``.

    The statistics file is a healthy 3981 bytes and always will be -- it holds
    two dozen small tensors.  Rounded to one decimal in megabytes it reads
    ``0.0 MB``, i.e. the one tool whose job is to build confidence in the
    artefacts reports the good one as empty.
    """
    if n_bytes < 1e6:
        return f"{n_bytes / 1e3:.1f} KB"
    return f"{n_bytes / 1e6:.1f} MB"


def _inspect_artefact(label: str, file) -> str | None:
    """``None`` if the artefact is usable, else a one-line description of what is wrong."""
    from .dataloaders.variables import LEVEL_VARIABLES, N_LAT, N_LON, SURFACE_VARIABLES

    try:
        if label == "norm stats":
            import torch

            stats = torch.load(file, weights_only=True)
            required = [
                "surface_mean",
                "surface_std",
                "surface_delta_std",
                "level_mean",
                "level_std",
                "level_delta_std",
                "surface_variables",
                "level_variables",
                "depths",
            ]
            missing = [key for key in required if key not in stats]
            if missing:
                return f"missing {missing}"
            if list(stats["surface_variables"]) != list(SURFACE_VARIABLES):
                return "surface_variables do not match variables.py"
            if list(stats["level_variables"]) != list(LEVEL_VARIABLES):
                return "level_variables do not match variables.py"
            n_depth = len(stats["depths"])
            for key in ("level_mean", "level_std", "level_delta_std"):
                if tuple(stats[key].shape[:2]) != (len(LEVEL_VARIABLES), n_depth):
                    return (
                        f"{key} has shape {tuple(stats[key].shape)}, expected it to start "
                        f"({len(LEVEL_VARIABLES)}, {n_depth})"
                    )
            if not bool((stats["surface_std"] > 0).all()) or not bool(
                (stats["level_std"] > 0).all()
            ):
                return "a standard deviation is <= 0; normalisation would divide by zero"
            return None

        import xarray as xr

        with xr.open_dataset(file) as dataset:
            if label == "masks":
                required = ["wet_level", "wet_surface", "wet_seaice", "bathymetry", "constants"]
                missing = [name for name in required if name not in dataset]
                if missing:
                    return f"missing variables {missing}"
                shape = (dataset.sizes.get("lat"), dataset.sizes.get("lon"))
                if shape != (N_LAT, N_LON):
                    return f"grid is {shape[0]}x{shape[1]}, expected {N_LAT}x{N_LON}"
                wet = int(dataset["wet_surface"].sum())
                if not 0.5 * N_LAT * N_LON < wet < 0.9 * N_LAT * N_LON:
                    return f"{wet} ocean cells at the surface, which is not plausible"
                return None
            # climatology
            if "month" not in dataset.sizes or dataset.sizes["month"] != 12:
                return f"month axis is {dataset.sizes.get('month')}, expected 12"
            names = [n for n in SURFACE_VARIABLES if n in dataset]
            if not names:
                return "no surface variables in the file"
            finite = [n for n in names if bool(dataset[n].notnull().any())]
            if len(finite) != len(names):
                return f"all-NaN: {sorted(set(names) - set(finite))}"
            return None
    except Exception as error:  # noqa: BLE001 - any read failure is the answer
        return f"could not be read: {type(error).__name__}: {error}"


def _check_forcing(report: Report) -> None:
    """The prescribed atmosphere -- optional, so never a FAIL.

    `forcing=none` is the default and everything works without any of this.  The
    check is here so that someone who *wants* `forcing=file` finds out on the
    command line rather than at the first batch of a queued job.
    """
    root = paths.ifs_forcing()
    files = sorted(root.glob("*.nc")) if root.is_dir() else []
    if not files:
        report.add(
            WARN,
            "IFS forcing",
            f"nothing in {root} (optional: only `forcing=file` needs it)",
            "set IFS_FORCING in config.env, or stay with the default forcing=none",
        )
        return
    report.add(PASS, "IFS forcing", f"{root}  {len(files)} files")

    stats_file = paths.forcing_stats_file()
    if not stats_file.exists():
        report.add(
            WARN,
            "forcing stats",
            f"{stats_file.name} not generated (only `forcing=file` needs it)",
            "make forcing-stats  (~10 s)",
        )
        return
    problem = _inspect_forcing_stats(stats_file)
    detail = f"{stats_file.name} ({_human_size(stats_file.stat().st_size)})"
    if problem is None:
        report.add(PASS, "forcing stats", detail)
    else:
        report.add(WARN, "forcing stats", f"{detail}: {problem}", "make forcing-stats  (~10 s)")


def _inspect_forcing_stats(file) -> str | None:
    """``None`` if the forcing statistics are usable, else what is wrong with them."""
    try:
        import torch

        stats = torch.load(file, weights_only=True)
        missing = [key for key in ("variables", "mean", "std") if key not in stats]
        if missing:
            return f"missing {missing}"
        n = len(stats["variables"])
        for key in ("mean", "std"):
            if stats[key].shape[0] != n:
                return f"{key} has {stats[key].shape[0]} rows for {n} variables"
        if not bool((stats["std"] > 0).all()):
            return "a standard deviation is not positive; dividing by it gives inf"
        return None
    except Exception as error:  # noqa: BLE001 - any read failure is the answer
        return f"could not be read: {type(error).__name__}: {error}"


def _home_inodes_left(probe_dir: Path, ceiling: int = 600) -> int | None:
    """How many more files $HOME will take, by creating them until it will not.

    There is no portable way to read an inode quota here: `quota`, `lfs` and
    `jutil ... quota` are all absent on the JURECA login nodes, and `df` reports
    the filesystem's terabytes rather than the user's file count.  So this
    measures the thing directly, and stops at `ceiling` because the answer only
    has to be "comfortably more than a Python install" or "not".

    Returns None if the probe could not run at all (read-only $HOME, no space),
    which is reported as a warning rather than guessed at.
    """
    made = 0
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        for made in range(ceiling):  # noqa: B007
            (probe_dir / f"{made}").write_text("")
        return ceiling
    except OSError as exc:
        if exc.errno == errno.EDQUOT or "quota" in str(exc).lower():
            return made
        return None if made == 0 else made
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def _check_home_quota(report: Report) -> None:
    """$HOME's inode quota, which is what actually stops `make setup`.

    Measured on JURECA: ~2050 files total, of which a fresh account already uses
    ~480.  uv unpacks its managed CPython (several thousand files) into
    ~/.local/share/uv/python and its wheel cache (tens of thousands) into
    ~/.cache/uv unless told otherwise, so a stock `uv venv` dies with

        Failed to extract archive: cpython-3.12.14-...tar.gz
          Caused by: Disk quota exceeded (os error 122)

    before torch is even considered.  scripts/setup_env.sh now points both at
    $REPO_ROOT/.uv, so this check exists to catch the OTHER things that write to
    $HOME -- pip's cache, ~/.cache/huggingface, a hand-made venv -- and to name
    inodes when they do, because nothing else in the error will.
    """
    home = Path.home()
    headroom = _home_inodes_left(home / ".oceanarches_inode_probe")
    if headroom is None:
        report.add(
            WARN,
            "home quota",
            f"could not probe {home} -- is it writable?",
            "a full or read-only $HOME breaks `make setup` in ways the error will not name",
        )
        return
    if headroom >= 600:
        report.add(PASS, "home quota", f"{home} accepts 600+ more files")
        return
    report.add(
        WARN if headroom > 100 else FAIL,
        "home quota",
        f"{home} accepted only {headroom} more files before EDQUOT -- this is an "
        f"INODE quota, not a space one, so `df` will show terabytes free",
        "keep caches off $HOME. scripts/setup_env.sh already does this for uv; for the "
        "rest: export UV_CACHE_DIR=$PWD/.uv/cache XDG_CACHE_HOME=$PWD/.cache "
        "PIP_CACHE_DIR=$PWD/.cache/pip, and delete ~/.cache",
    )


def main() -> int:
    print(f"OceanArches doctor -- repo at {paths.REPO_ROOT}")
    report = Report()
    _check_python(report)
    _check_home_quota(report)
    _check_imports(report)
    _check_allocation(report)
    _check_gpu(report)
    _check_ffmpeg(report)
    _check_data(report)
    _check_stats(report)
    _check_forcing(report)
    return report.render()


if __name__ == "__main__":
    raise SystemExit(main())
