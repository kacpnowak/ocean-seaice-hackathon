"""The startup guards: every one of them replaces a silence that cost a participant.

Six people ran this kit as first-time participants.  The expensive findings were
all silences -- a short run that saved nothing, an override that landed on a key
nothing reads, a re-run that trained zero steps and exited 0, a login node that
trained on somebody else's GPU.  `oceanarches/guards.py` is the refusal; this file
is what keeps it able to fire.

Two things are tested that unit tests alone would miss, and both have their own
way of silently rotting:

* that the guard is actually WIRED IN -- it is registered as a `hydra.callbacks`
  entry in `configs/config.yaml`, and deleting those three lines would leave every
  unit test below green while turning the whole module off.  So one test runs the
  real `make train-tiny` command line in a subprocess.
* that the message reaches STDOUT.  Notebooks 02 and 03 capture subprocess stdout
  and discard stderr, so a guard that spoke only through its exception would be
  invisible in exactly the place beginners read.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from oceanarches import guards

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# 1. A run that would save no checkpoint
# ---------------------------------------------------------------------------
def test_the_checkpoint_schedule_is_the_one_geoarches_really_uses():
    """Not a restatement of our own arithmetic: geoarches' callback is driven directly.

    `CheckpointEveryNSteps.on_train_batch_end` is the only thing that writes a
    checkpoint during `fit`, and it fires on
    `trainer.global_step % save_step_frequency == 0`. If that ever changes,
    `checkpoint_steps` -- and therefore the refusal below -- is wrong, and this is
    the test that notices.
    """
    from geoarches.main_hydra import CheckpointEveryNSteps

    class _Recording(CheckpointEveryNSteps):
        def __init__(self, frequency):
            super().__init__(save_step_frequency=frequency)
            self.saved: list[int] = []

        def save(self, *args, **kwargs):
            self.saved.append(self.trainer.global_step)

    class _Trainer:
        global_step = 0

    for max_steps, frequency in [(200, 1000), (4000, 1000), (1500, 1000), (12, 4)]:
        callback, trainer = _Recording(frequency), _Trainer()
        for step in range(1, max_steps + 1):
            trainer.global_step = step
            callback.on_train_batch_end(trainer)
        assert callback.saved == guards.checkpoint_steps(max_steps, frequency), (
            f"max_steps={max_steps}, save_step_frequency={frequency}"
        )


def test_a_run_shorter_than_one_save_interval_is_refused():
    """The measured case: five runs at 200/250/300 steps left no `checkpoints/` at all."""
    message = guards.no_checkpoint_refusal(200, 1000, preset="tiny", name="my_first_run")
    assert message is not None
    assert "200" in message and "1000" in message
    assert "no checkpoint at all" in message
    # The fix has to be a command, not a diagnosis.
    assert "++save_step_frequency=" in message


def test_a_run_whose_final_step_is_never_checkpointed_is_refused():
    """4500 steps at save_step_frequency 1000 keeps step 4000 and throws the model away."""
    message = guards.no_checkpoint_refusal(4500, 1000, preset="tiny", name="x")
    assert message is not None
    assert "NOT the finished model" in message


def test_the_shipped_budgets_are_not_refused():
    """Every preset ships a `max_steps` its `save_step_frequency` divides."""
    for preset in ("tiny", "small", "base", "large", "ocean_component", "seaice_component"):
        cfg = OmegaConf.load(CONFIG_DIR / "module" / f"{preset}.yaml")
        assert guards.no_checkpoint_refusal(cfg.max_steps, cfg.save_step_frequency, preset) is None


@pytest.mark.parametrize("max_steps", [1, 2, 7, 12, 200, 250, 1000, 4000, 4500, 110000, 300000])
def test_the_suggested_save_frequency_actually_divides(max_steps):
    """A suggestion that does not divide would fail the very guard that printed it."""
    frequency = guards.suggest_save_frequency(max_steps)
    assert max_steps % frequency == 0
    assert guards.no_checkpoint_refusal(max_steps, frequency) is None


def test_a_zero_save_frequency_is_refused_rather_than_dividing_by_zero():
    assert guards.no_checkpoint_refusal(200, 0) is not None


# ---------------------------------------------------------------------------
# 2. A re-run that would train nothing
# ---------------------------------------------------------------------------
def _checkpoint(directory: Path, step: int) -> Path:
    ckpts = directory / "checkpoints"
    ckpts.mkdir(parents=True, exist_ok=True)
    path = ckpts / f"checkpoint_global_step={step}.ckpt"
    path.write_bytes(b"not a real checkpoint")
    return path


def test_reusing_a_finished_run_name_is_refused(tmp_path):
    """`make train-tiny NAME=task6_tiny` exits 0 today, having trained and saved nothing."""
    _checkpoint(tmp_path, 4000)
    step = guards.checkpoint_step(guards.latest_checkpoint(tmp_path))
    assert step == 4000
    message = guards.already_trained_refusal("task6_tiny", tmp_path, step, max_steps=4000)
    assert message is not None
    assert "4000" in message and "make eval NAME=task6_tiny" in message


def test_the_longer_budget_it_suggests_is_past_the_checkpoint_and_still_checkpoints():
    """The escape route it offers must not walk into the other refusal."""
    message = guards.already_trained_refusal("r", "modelstore/r", step=4500, max_steps=4000)
    suggested = int(message.split("++max_steps=")[1].split('"')[0])
    assert suggested > 4500
    assert guards.no_checkpoint_refusal(suggested, 1000) is None


def test_a_half_trained_run_is_allowed_to_continue(tmp_path):
    _checkpoint(tmp_path, 1000)
    step = guards.checkpoint_step(guards.latest_checkpoint(tmp_path))
    assert guards.already_trained_refusal("r", tmp_path, step, max_steps=4000) is None


def test_a_run_with_no_checkpoints_is_allowed(tmp_path):
    assert guards.latest_checkpoint(tmp_path) is None
    assert guards.already_trained_refusal("r", tmp_path, None, max_steps=4000) is None


# ---------------------------------------------------------------------------
# 3. Overrides that land on nothing
# ---------------------------------------------------------------------------
def _reference():
    """The real composed config, so a config restructure cannot leave these stale."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR), job_name="guards"):
        return compose(
            config_name="config",
            overrides=["cluster=local", "module=tiny", "dataloader=glorys_tiny"],
        )


def _problems(*overrides):
    reference = _reference()
    created = guards.created_keys(overrides, lambda key: guards._key_exists(reference, key))
    return guards.override_problems(created, reference)


def test_the_learning_rate_guesses_that_do_nothing_are_refused():
    """Measured: both of these leave lr at 0.0003 and print nothing (tweaker, 2 of 17)."""
    for override in ("++lr=1e-4", "++module.lr=1e-4"):
        refusals, _ = _problems(override)
        assert refusals, override
        assert "module.module.lr" in refusals[0], (
            f"{override} must name the path that was meant, not merely refuse"
        )


def test_a_top_level_typo_is_refused():
    """`++max_step=10` -- one letter -- leaves max_steps at the preset's 4000."""
    refusals, _ = _problems("++max_step=10")
    assert refusals and "max_step" in refusals[0]


def test_every_documented_override_is_accepted():
    """docs/cheatsheet.md's override table must survive its own guard.

    A guard that refuses the documented commands is worse than no guard, and the
    only honest way to check is to run the real ones through it.
    """
    documented = [
        "++name=my_run",
        "++max_steps=8000",
        "++save_step_frequency=1000",
        "++batch_size=4",
        "++limit_val_batches=16",
        "++module.module.lr=1e-4",
        "++module.train.rollout_iterations=2",
        "++seed=1",
        "++entity=my_team",  # ships as null: present, not absent
        "+load_ckpt=modelstore/other_run",
        "++module.module.num_training_steps=75000",
        # docs/04's scaling table and docs/01+03+cheatsheet's OOM row:
        "++module.backbone.emb_dim=192",
        "++module.backbone.num_heads=[6,12,12,6]",
        "++module.embedder.emb_dim=192",
        "++module.embedder.out_emb_dim=384",
        "++module.module.num_warmup_steps=1000",
        "++module.backbone.gradient_checkpointing=True",
    ]
    refusals, _ = _problems(*documented)
    assert not refusals, refusals


def test_a_documented_constructor_keyword_warns_but_does_not_refuse():
    """`++module.module.multistep_curriculum=True` is documented and creates a key.

    `OceanForecastModule` takes it as a keyword with a default, so no yaml file
    mentions it and it is still read.  Refusing would break docs/04 section 4.6;
    saying nothing would make it indistinguishable from a typo.
    """
    refusals, warnings = _problems("++module.module.multistep_curriculum=True")
    assert not refusals
    assert warnings and "multistep_curriculum" in warnings[0]


def test_a_single_plus_is_taken_as_deliberate():
    refusals, warnings = _problems("+something_new=1")
    assert not refusals
    assert warnings


def test_group_selections_and_plain_overrides_are_left_alone():
    """`module=tiny` and `mode=test` are not `+` overrides and cannot invent a key."""
    refusals, warnings = _problems("module=tiny", "mode=test", "cluster=local")
    assert not refusals and not warnings


# ---------------------------------------------------------------------------
# 4. The allocation
# ---------------------------------------------------------------------------
def test_a_visible_gpu_does_not_count_as_an_allocation():
    """The whole point: `jpbl-s02-02` has a real GPU and `SLURM_JOB_ID` unset.

    Keying off CUDA visibility is what made `make doctor` PASS on a login node
    while another user's job held 49 GiB of the card three participants then
    trained on.
    """
    message = guards.allocation_warning({}, hostname="jpbl-s02-02", cluster_name="jupiter_1gpu")
    assert message is not None
    assert "SLURM_JOB_ID" in message and "srun" in message


def test_an_allocation_silences_it():
    assert (
        guards.allocation_warning(
            {"SLURM_JOB_ID": "1285543"}, hostname="jpbo-b-001", cluster_name="jupiter_1gpu"
        )
        is None
    )


def test_the_cpu_cluster_config_is_exempt():
    """`cluster=local` says "no GPU, one sample" outright; warning about it is noise."""
    assert guards.allocation_warning({}, hostname="jpbl-s02-02", cluster_name="local") is None


def test_doctor_warns_about_the_missing_allocation_even_with_a_gpu(monkeypatch):
    from oceanarches import doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/srun")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "jpbl-s02-02")
    report = doctor.Report()
    doctor._check_allocation(report)
    statuses = {name: status for status, name, _, _ in report.rows}
    assert statuses.get("allocation") == doctor.WARN, report.rows
    assert "srun" in report.rows[0][3]


def test_doctor_passes_inside_an_allocation(monkeypatch):
    from oceanarches import doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/srun")
    monkeypatch.setenv("SLURM_JOB_ID", "1285543")
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "jpbo-b-001")
    report = doctor.Report()
    doctor._check_allocation(report)
    assert report.rows[0][0] == doctor.PASS
    assert "1285543" in report.rows[0][2]


def test_doctor_says_nothing_about_slurm_off_a_slurm_cluster(monkeypatch):
    from oceanarches import doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    report = doctor.Report()
    doctor._check_allocation(report)
    assert report.rows == []


# ---------------------------------------------------------------------------
# 5. Resuming, and the preset that drags a budget behind it
# ---------------------------------------------------------------------------
def test_the_run_says_whether_it_is_resuming(tmp_path):
    """`resume: True` printed on all five of one participant's runs, four of them fresh."""
    assert "FRESH START" in guards.resume_line(tmp_path, 4000, resume=True)
    _checkpoint(tmp_path, 1000)
    line = guards.resume_line(tmp_path, 4000, resume=True)
    assert "RESUMING" in line and "1000" in line and "3000 to go" in line


def test_resume_false_still_says_it_will_resume(tmp_path):
    """geoarches loads the newest checkpoint whenever `checkpoints/` exists, resume or not."""
    _checkpoint(tmp_path, 1000)
    line = guards.resume_line(tmp_path, 4000, resume=False)
    assert "RESUMING" in line and "resume=False" in line


def test_naming_two_presets_on_one_command_line_warns():
    """`make train-tiny HYDRA_ARGS="module=base"` sends hydra `module=tiny module=base`."""
    message = guards.preset_swap_warning(
        ["module=tiny", "module=base"], None, max_steps=110000, batch_size=2
    )
    assert message is not None and "batch_size" in message


def test_reusing_a_name_under_a_different_budget_warns():
    previous = OmegaConf.create({"max_steps": 4000, "batch_size": 8})
    message = guards.preset_swap_warning([], previous, max_steps=110000, batch_size=2)
    assert message is not None
    assert "4000 -> 110000" in message and "8 -> 2" in message


def test_the_same_preset_twice_is_not_a_swap():
    assert guards.preset_swap_warning(["module=tiny", "module=tiny"], None, 4000, 8) is None


# ---------------------------------------------------------------------------
# 6. Wiring -- the part unit tests cannot see
# ---------------------------------------------------------------------------
def test_the_guard_is_registered_in_the_root_config():
    cfg = OmegaConf.load(CONFIG_DIR / "config.yaml")
    targets = [entry["_target_"] for entry in cfg.hydra.callbacks.values()]
    assert "oceanarches.guards.StartupGuard" in targets, (
        "configs/config.yaml no longer registers the startup guard, so no run is checked"
    )


def _make_recipe(target: str) -> list[str]:
    make = shutil.which("make")
    if make is None:
        pytest.fail("`make` is not on PATH")
    printed = subprocess.run(
        [make, "--no-print-directory", "-n", target],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert printed.returncode == 0, printed.stderr[-2000:]
    argv = shlex.split(printed.stdout.replace("\\\n", " "))
    return [sys.executable] + argv[1:]


def test_the_real_command_line_refuses_a_short_run_on_stdout():
    """End to end: `make train-tiny HYDRA_ARGS="++max_steps=200"` must not start.

    This is the one test that fails if the `hydra.callbacks` block is deleted from
    configs/config.yaml, and the one that fails if the refusal is raised without
    being printed to stdout first.  It costs one interpreter start-up and stops
    before any data is read.
    """
    run_dir = REPO_ROOT / "modelstore" / "zz_guard_test"
    # Cleared first, not merely asserted absent: this test is the one that leaves
    # a directory behind if it ever fails, and a stale one from a previous run
    # would make the assertion below fail for the wrong reason for ever after.
    shutil.rmtree(run_dir, ignore_errors=True)
    argv = _make_recipe("train-tiny") + ["++name=zz_guard_test", "++max_steps=200"]
    result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0, "the run started; it would have saved no checkpoint"
    assert "save_step_frequency" in result.stdout, (
        "the refusal did not reach stdout, where the notebooks read it:\n"
        f"stdout={result.stdout[-800:]!r}\nstderr={result.stderr[-800:]!r}"
    )
    assert not run_dir.exists(), "the refusal came too late: modelstore/ was already written"


# ---------------------------------------------------------------------------
# 7. The Makefile's own silence
# ---------------------------------------------------------------------------
def _make(*args: str) -> subprocess.CompletedProcess:
    make = shutil.which("make")
    if make is None:
        pytest.fail("`make` is not on PATH")
    return subprocess.run(
        [make, "--no-print-directory", "-n", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


@pytest.mark.parametrize(
    "assignment", ["++max_steps=10", "module=base", "++module.module.lr=1e-4", "name=my_run"]
)
def test_make_refuses_a_hydra_override_typed_as_a_goal(assignment):
    """Measured: `make train-tiny NAME=x ++max_steps=10` launched the full 4000 steps.

    make parses every command-line word containing `=` as a variable assignment,
    so the override never reaches hydra and nothing says so.
    """
    result = _make("train-tiny", "NAME=x", assignment)
    assert result.returncode != 0, f"`make train-tiny {assignment}` still runs the default budget"
    combined = result.stdout + result.stderr
    assert assignment in combined
    assert "HYDRA_ARGS" in combined, "the refusal must name the supported spelling"


def test_make_still_takes_its_own_variables_and_the_environment():
    """The guard must not break the documented command lines, or the cure is worse."""
    for extra in (
        ["NAME=my_run"],
        ["NAME=my_run", "HYDRA_ARGS=++max_steps=1000"],
        ["CLUSTER=local", "NAME=x"],
        # `make train-tiny CUDA_VISIBLE_DEVICES=0` really does reach the recipe's
        # environment; it is not a hydra override and must keep working.
        ["CUDA_VISIBLE_DEVICES=0", "NAME=x"],
    ):
        result = _make("train-tiny", *extra)
        assert result.returncode == 0, f"make train-tiny {extra} was refused: {result.stderr}"
        assert "geoarches.main_hydra" in result.stdout


# ---------------------------------------------------------------------------
# 8. One implementation of the allocation warning, shared with the eval path
# ---------------------------------------------------------------------------
def test_warn_allocation_prints_the_complaint_and_returns_it(capsys, monkeypatch):
    """`make train-*` and `make eval` must present this identically, so both call this.

    Workstream B duplicated the `OCEANARCHES_SKIP_GUARDS` truthiness rule and the
    pause in `run_eval` because there was no public entry point.  Two copies of
    "am I on an allocation" drifting apart is the failure this whole exercise is
    about.
    """
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    said = guards.warn_allocation(cluster_name="jupiter_1gpu", env={}, hostname="jpbl-s02-02")
    out = capsys.readouterr().out
    assert said is not None
    assert "[oceanarches] WARNING:" in out
    assert "SLURM_JOB_ID" in out and "srun" in out
    assert said in out, "the returned text and the printed text must be the same text"


def test_warn_allocation_says_nothing_inside_an_allocation(capsys):
    assert guards.warn_allocation(env={"SLURM_JOB_ID": "1289794"}, hostname="jpbo-035-40") is None
    assert capsys.readouterr().out == ""


def test_the_cpu_escape_is_spelled_for_the_entry_point_that_prints_it():
    """`cluster=local` is the training routes' spelling; `run_eval` takes `--device cpu`.

    A shared message that ends by naming a flag the caller does not accept sends
    the reader to a dead end, which is why this is a parameter and not an
    addendum printed afterwards.
    """
    training = guards.allocation_warning({}, hostname="h", cluster_name="jupiter_1gpu")
    evaluation = guards.allocation_warning(
        {}, hostname="h", cluster_name=None, cpu_hint="--device cpu"
    )
    assert "`cluster=local` if you really do mean" in training
    assert "`--device cpu` if you really do mean" in evaluation
    # ... and an entry point with no cluster config does not print `cluster=?`.
    assert "cluster=" not in evaluation.splitlines()[0]
    assert "cluster=jupiter_1gpu" in training.splitlines()[0]


def test_the_pause_waits_on_a_tty_and_never_anywhere_else(monkeypatch):
    """A queued job, a log file and a notebook cell must not sit for ten seconds."""
    slept: list[float] = []
    monkeypatch.setattr(guards.time, "sleep", lambda seconds: slept.append(seconds))
    monkeypatch.delenv("OCEANARCHES_SKIP_GUARDS", raising=False)

    monkeypatch.setattr(guards.sys.stdout, "isatty", lambda: False, raising=False)
    guards.warn_allocation(env={}, hostname="jpbl-s02-02")
    assert slept == [], "paused with nothing watching"

    monkeypatch.setattr(guards.sys.stdout, "isatty", lambda: True, raising=False)
    guards.warn_allocation(env={}, hostname="jpbl-s02-02")
    assert slept == [10], "no pause on a tty, so Ctrl-C is impossible"

    # ... and the documented escape hatch turns it off, through the one public
    # truthiness rule rather than a second copy of it.
    monkeypatch.setenv("OCEANARCHES_SKIP_GUARDS", "1")
    assert guards.skip_guards() is True
    guards.warn_allocation(env={}, hostname="jpbl-s02-02")
    assert slept == [10], "OCEANARCHES_SKIP_GUARDS did not skip the pause"

    monkeypatch.setenv("OCEANARCHES_SKIP_GUARDS", "0")
    assert guards.skip_guards() is False, "`0` must not read as 'skip'"


def test_pause_false_is_honoured(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(guards.time, "sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(guards.sys.stdout, "isatty", lambda: True, raising=False)
    guards.warn_allocation(env={}, hostname="jpbl-s02-02", pause=False)
    assert slept == []


# ---------------------------------------------------------------------------
# 9. Sampled statistics have to show up in `make doctor`
# ---------------------------------------------------------------------------
def _stats_artefacts(tmp_path, n_dates: int, sampling: str, n_years: int, available: int):
    """A statistics file and a climatology carrying nothing but their provenance."""
    import torch
    import xarray as xr

    stats = tmp_path / "stats.pt"
    torch.save(
        {
            "years": "1993-2025",
            "n_dates": n_dates,
            "n_years": 33,
            "sampling": sampling,
        },
        stats,
    )
    climatology = tmp_path / "climatology.nc"
    xr.Dataset(
        attrs={
            "years": "1993-2025",
            "n_years": n_years,
            "n_years_available": available,
            "sampling": sampling,
        }
    ).to_netcdf(climatology)
    return stats, climatology


def _depth_row(monkeypatch, tmp_path, **kwargs):
    """The `stats depth` rows doctor produces for artefacts with this provenance.

    `read_statistics_provenance` resolves `paths.stats_file()` at call time, so
    pointing those two functions at a temporary pair is enough -- the real
    artefacts are never touched, and neither is the real provenance reader.
    """
    from oceanarches import doctor, paths

    stats, climatology = _stats_artefacts(tmp_path, **kwargs)
    monkeypatch.setattr(paths, "stats_file", lambda: stats)
    monkeypatch.setattr(paths, "climatology_file", lambda: climatology)
    report = doctor.Report()
    doctor._check_stats_depth(report)
    return report


def test_doctor_warns_when_the_statistics_came_from_a_sample(monkeypatch, tmp_path):
    """`make stats-quick` used to leave no trace anywhere a participant looks.

    A sampled climatology is a sampled *baseline*: every score in the report is
    quoted against it.  Someone could save eight minutes and present precise
    numbers with nothing saying what they were measured against.
    """
    report = _depth_row(
        monkeypatch, tmp_path, n_dates=20, sampling="sampled", n_years=11, available=33
    )
    from oceanarches import doctor

    assert len(report.rows) == 1
    status, name, detail, fix = report.rows[0]
    assert status == doctor.WARN, report.rows
    assert name == "stats depth"
    assert "SAMPLED" in detail and "20 dates" in detail and "11 of 33 years" in detail
    assert "make stats" in fix


def test_doctor_passes_a_full_statistics_build(monkeypatch, tmp_path):
    report = _depth_row(
        monkeypatch, tmp_path, n_dates=400, sampling="full", n_years=33, available=33
    )
    from oceanarches import doctor

    assert report.rows and report.rows[0][0] == doctor.PASS
    assert "SAMPLED" not in report.rows[0][2]


def test_doctor_catches_a_quick_build_that_predates_the_provenance(monkeypatch, tmp_path):
    """Artefacts written before `sampling` existed are judged on `n_dates` alone."""
    report = _depth_row(monkeypatch, tmp_path, n_dates=20, sampling="", n_years=0, available=0)
    from oceanarches import doctor

    assert report.rows and report.rows[0][0] == doctor.WARN


def test_doctor_adds_no_depth_row_when_the_artefacts_say_nothing(monkeypatch, tmp_path):
    """A "not recorded" row would claim the artefacts were asked, which they were not."""
    from oceanarches import doctor, paths

    monkeypatch.setattr(paths, "stats_file", lambda: tmp_path / "absent.pt")
    monkeypatch.setattr(paths, "climatology_file", lambda: tmp_path / "absent.nc")
    report = doctor.Report()
    doctor._check_stats_depth(report)
    assert report.rows == []


def test_the_real_doctor_run_includes_the_depth_row():
    """Wiring: `_check_stats` must call it, or the row exists and never appears."""
    from oceanarches import doctor, paths

    if not paths.stats_file().exists():
        pytest.skip("statistics not generated; run: make stats")
    report = doctor.Report()
    doctor._check_stats(report)
    assert any(name == "stats depth" for _, name, _, _ in report.rows), (
        "_check_stats no longer reports the sampling depth"
    )


# ---------------------------------------------------------------------------
# 10. The keys a guesser actually reaches for
# ---------------------------------------------------------------------------
# Round 2 of the participant rehearsal: `++lr=` was hard-refused with the right
# answer, while `++module.embed_dim=512`, `++module.module.embed_dim=512` and
# `++dataloader.n_levels=20` -- equally natural guesses -- got a soft,
# dismissible "may be intended" warning and then trained the unchanged network.
# The guard caught the aliases it had been told about and waved through the case
# a guesser hits.  The rule is now one rule: a `++` key that nothing would read
# is refused, and "would read it" means a constructor keyword of the `_target_`
# that instantiates the node it sits in.
@pytest.mark.parametrize(
    "override, wanted",
    [
        ("++module.embed_dim=512", "module.backbone.emb_dim"),
        ("++module.module.embed_dim=512", "module.backbone.emb_dim"),
        ("++dataloader.n_levels=20", "dataloader.n_level_in"),
    ],
)
def test_a_guessed_key_that_reaches_nothing_is_refused_and_the_real_one_named(override, wanted):
    """Measured: all three trained a 13.0M-parameter backbone, unchanged, and said so
    only as a warning you can read past.

    MUTANT: putting these back on the warning branch (the old `else`) fails the
    first assertion; dropping `_nearest_keys` fails the second.
    """
    refusals, warnings = _problems(override)
    assert refusals, f"{override} is still only a warning"
    assert wanted in refusals[0], f"{override} does not name the key that was meant"
    assert not warnings


def test_a_constructor_keyword_of_the_node_that_is_instantiated_still_only_warns():
    """The other half of the rule, and the half that must not become a refusal.

    Both of these are documented (docs/cheatsheet.md, docs/04) and both create a
    key: `OceanForecastModule` and `ArchesWeatherCondBackbone` take them as
    constructor keywords with defaults, so no yaml mentions them and they are
    still read.  Refusing here would break a documented workflow, which is the
    failure mode this exercise is most careful about.
    """
    for override in (
        "++module.module.multistep_curriculum=True",
        "++module.backbone.gradient_checkpointing=True",
    ):
        refusals, warnings = _problems(override)
        assert not refusals, override
        assert warnings and "IS read" in warnings[0]


def test_the_keyword_lookup_reads_the_whole_inheritance_chain():
    """`lr`, `betas` and `weight_decay` are named by geoarches' base module, not ours.

    A lookup that stopped at `OceanForecastModule.__init__` would refuse
    `++module.module.num_cycles=2`, which is a real keyword three classes up.
    """
    accepted = guards.keywords_read_by(
        "oceanarches.lightning_modules.ocean_forecast.OceanForecastModule"
    )
    assert accepted is not None
    assert {"multistep_curriculum", "lr", "num_cycles"} <= accepted
    # ... and `**kwargs` must not read as "accepts anything", or nothing is ever refused.
    assert "embed_dim" not in accepted


def test_a_target_that_will_not_import_is_a_warning_and_not_a_refusal():
    """A guard must not refuse out of its own ignorance."""
    assert guards.keywords_read_by("no.such.module.SomeClass") is None


# ---------------------------------------------------------------------------
# 11. A preset swap onto a name that already has checkpoints
# ---------------------------------------------------------------------------
def _module_cfg(**backbone):
    return OmegaConf.create(
        {
            "module": {"module": {"_target_": "x.Y"}, "backbone": dict(backbone)},
            "dataloader": {"n_surface_in": 7, "component": "full"},
        }
    )


def test_a_preset_swap_onto_an_existing_run_is_refused(tmp_path):
    """Measured: `NAME=baseline0 MODULE=small` kept training the *tiny* network.

    geoarches' resume path does `cfg.module = exp_cfg.module` as soon as
    `<exp_dir>/checkpoints` exists, so the stored architecture wins, the Params
    table stays 13.5M where `small` is 45M, and the old checkpoints are
    overwritten in place.  The budget warning next door noticed max_steps and
    batch_size and said nothing about the network.

    MUTANT: returning None unconditionally from `resumed_architecture_refusal`
    fails the first assertion.
    """
    _checkpoint(tmp_path, 30)
    message = guards.resumed_architecture_refusal(
        "baseline0",
        tmp_path,
        _module_cfg(emb_dim=96),
        _module_cfg(emb_dim=192),
        module_choice="small",
        dataloader_choice="glorys",
    )
    assert message is not None
    assert "emb_dim" in message and "96" in message and "192" in message
    assert "step 30" in message, "the message must say what it would overwrite"
    # A refusal has to name a command, and `++resume=False` is NOT one: geoarches
    # loads the newest checkpoint whether or not resume is set.
    assert "NAME=baseline0_small" in message
    assert "`++resume=False` does not help" in message


def test_resuming_the_same_run_with_the_same_preset_is_not_refused(tmp_path):
    """The ordinary continuation, and the one this must never break."""
    _checkpoint(tmp_path, 1000)
    same = _module_cfg(emb_dim=96)
    assert guards.resumed_architecture_refusal("r", tmp_path, same, same) is None


def test_a_longer_budget_or_a_new_learning_rate_is_not_an_architecture_change(tmp_path):
    """`++max_steps=8000 ++module.module.lr=1e-4` on a resume is documented, and both
    survive geoarches' substitution as `+`-overrides. Refusing them would be the
    guard crying wolf at the escape route another refusal recommends."""
    _checkpoint(tmp_path, 1000)
    previous = OmegaConf.create(
        {"module": {"module": {"lr": 3e-4, "_target_": "x.Y"}, "backbone": {"emb_dim": 96}}}
    )
    current = OmegaConf.create(
        {"module": {"module": {"lr": 1e-4, "_target_": "x.Y"}, "backbone": {"emb_dim": 96}}}
    )
    assert guards.resumed_architecture_refusal("r", tmp_path, previous, current) is None


def test_a_dataloader_swap_onto_an_existing_run_is_refused_too(tmp_path):
    """`cfg.dataloader = exp_cfg.dataloader` is the very next line in geoarches."""
    _checkpoint(tmp_path, 1000)
    previous = OmegaConf.create({"dataloader": {"component": "full", "n_surface_in": 7}})
    current = OmegaConf.create({"dataloader": {"component": "seaice", "n_surface_in": 4}})
    message = guards.resumed_architecture_refusal("r", tmp_path, previous, current)
    assert message is not None and "component" in message


def test_a_name_with_no_checkpoints_yet_is_left_alone(tmp_path):
    """Without `checkpoints/` geoarches does not substitute anything, so neither do we."""
    assert (
        guards.resumed_architecture_refusal(
            "r", tmp_path, _module_cfg(emb_dim=96), _module_cfg(emb_dim=192)
        )
        is None
    )


def test_the_real_presets_are_far_enough_apart_to_be_caught(tmp_path):
    """Not our own fixture arguing with itself: the shipped configs are composed.

    If `tiny` and `small` ever stop differing in a key this guard reads, the
    guard silently stops firing on the exact swap it was written for.
    """
    _checkpoint(tmp_path, 30)
    tiny = _reference()  # module=tiny dataloader=glorys_tiny
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR), job_name="guards"):
        small = compose(
            config_name="config",
            overrides=["cluster=local", "module=small", "dataloader=glorys_tiny"],
        )
    assert guards.resumed_architecture_refusal("baseline0", tmp_path, tiny, small) is not None
    assert guards.resumed_architecture_refusal("baseline0", tmp_path, tiny, tiny) is None


# ---------------------------------------------------------------------------
# 12. Two runs, one name, at the same time
# ---------------------------------------------------------------------------
def test_a_second_run_under_the_same_name_is_refused(tmp_path, monkeypatch):
    """Measured: two `make train-tiny NAME=my_first_run` at once both printed
    FRESH START, with no lock and no warning, both aiming at one directory.

    MUTANT: making `acquire_run_lock` write the lock without reading the existing
    one lets the second call through and fails this.
    """
    monkeypatch.delenv("OCEANARCHES_SKIP_GUARDS", raising=False)
    first = guards.acquire_run_lock(tmp_path)
    assert first is not None and first.is_file()

    with pytest.raises(SystemExit):
        guards.acquire_run_lock(tmp_path)

    info = guards.read_run_lock(tmp_path)
    assert info["pid"] == os.getpid(), "the second run overwrote the first run's lock"
    guards.release_run_lock(first)
    assert not first.exists()


def test_the_refusal_says_who_holds_it_and_how_to_clear_a_dead_one(tmp_path):
    info = {
        "host": "jpbo-035-40",
        "pid": 4242,
        "slurm_job_id": "1289795",
        "started": time.time() - 600,
        "command": "python -m geoarches.main_hydra ...",
    }
    message = guards.concurrent_run_refusal("my_first_run", tmp_path, info)
    assert "jpbo-035-40" in message and "4242" in message and "1289795" in message
    assert "10 min ago" in message
    assert f"rm {tmp_path}/{guards.LOCK_NAME}" in message
    assert "NAME=my_first_run_2" in message


class _Squeue:
    """Stand-in for `subprocess.run([squeue, ...])`."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr

    def __call__(self, *args, **kwargs):
        return self


def _with_squeue(monkeypatch, result):
    monkeypatch.setattr(guards.shutil, "which", lambda name: "/usr/bin/squeue")
    monkeypatch.setattr(guards.subprocess, "run", result)


def test_a_lock_left_by_a_slurm_job_that_has_ended_is_taken_over(tmp_path, monkeypatch):
    """The regression this whole check exists for.

    SLURM enforces the wall clock with SIGKILL, so `atexit` never runs and the
    lock outlives the job.  The very next thing anybody does is resubmit to
    continue from the last checkpoint -- and job 1349666, which was exactly that
    resubmission, was refused by the lock of the 12-hour run it was meant to
    continue.  The lock was 20 minutes old and from another host, so nothing in
    the age or the hostname could tell; only the job id could.

    MUTANT: dropping the `slurm_job_is_running` branch from `lock_liveness` puts
    this back on the 24-hour age rule and fails here.
    """
    monkeypatch.delenv("OCEANARCHES_SKIP_GUARDS", raising=False)
    guards.run_lock_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    guards.run_lock_path(tmp_path).write_text(
        json.dumps(
            {
                "host": "jpbo-002-25",  # another host,
                "pid": 326978,
                "slurm_job_id": "1325538",
                "started": time.time() - 20 * 60,  # and only 20 minutes old
            }
        )
    )
    _with_squeue(
        monkeypatch, _Squeue(returncode=1, stderr="slurm_load_jobs error: Invalid job id")
    )

    taken = guards.acquire_run_lock(tmp_path)  # must not raise
    assert taken is not None
    assert guards.read_run_lock(tmp_path)["pid"] == os.getpid()
    guards.release_run_lock(taken)


def test_a_lock_whose_slurm_job_is_still_queued_is_refused(tmp_path, monkeypatch):
    """The other direction: two jobs really under one name must still be caught,
    even when the pid is on a host we cannot ask about."""
    monkeypatch.delenv("OCEANARCHES_SKIP_GUARDS", raising=False)
    guards.run_lock_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    guards.run_lock_path(tmp_path).write_text(
        json.dumps(
            {
                "host": "jpbo-002-25",
                "pid": 326978,
                "slurm_job_id": "1325538",
                "started": time.time() - 20 * 60,
            }
        )
    )
    _with_squeue(monkeypatch, _Squeue(returncode=0, stdout="RUNNING\n"))
    with pytest.raises(SystemExit):
        guards.acquire_run_lock(tmp_path)


def test_slurm_outranks_a_pid_on_this_host(monkeypatch):
    """A pid can be reused; a job id within a cluster cannot.  A lock written by a
    job that is still queued is believed even if the recorded pid is gone."""
    _with_squeue(monkeypatch, _Squeue(returncode=0, stdout="RUNNING\n"))
    monkeypatch.setattr(
        guards.os, "kill", lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError())
    )
    info = {"host": "mine", "pid": 1, "slurm_job_id": "42", "started": time.time()}
    assert guards.lock_is_live(info, hostname="mine") is True


@pytest.mark.parametrize(
    "result, expected",
    [
        (_Squeue(returncode=0, stdout="RUNNING\n"), True),
        (_Squeue(returncode=0, stdout="PENDING\n"), True),
        # Still tearing down, and its processes can still be alive.
        (_Squeue(returncode=0, stdout="COMPLETING\n"), True),
        (_Squeue(returncode=0, stdout=""), False),  # gone from the queue
        (_Squeue(returncode=0, stdout="TIMEOUT\n"), False),  # what 1325538 ended as
        (_Squeue(returncode=0, stdout="CANCELLED\n"), False),
        (_Squeue(returncode=1, stderr="slurm_load_jobs error: Invalid job id"), False),
        # slurmctld down, a permissions problem: unknown, NOT dead.
        (_Squeue(returncode=1, stderr="slurm_load_jobs error: Unable to contact"), None),
    ],
)
def test_squeue_is_read_the_way_slurm_actually_answers(monkeypatch, result, expected):
    """MUTANT: treating a non-zero exit as `False` turns a slurmctld outage into
    two jobs training over each other, and fails the last case here."""
    _with_squeue(monkeypatch, result)
    assert guards.slurm_job_is_running("1325538") is expected


@pytest.mark.parametrize("job_id", [None, "", "; rm -rf /", "large_pretrained", "1234; ls"])
def test_only_a_job_id_shaped_job_id_reaches_a_subprocess(monkeypatch, job_id):
    """The id comes out of a file on disk.  Nothing that is not a job id is handed
    to `squeue`, and the caller falls back to the host and age checks instead."""
    called = []
    monkeypatch.setattr(guards.shutil, "which", lambda name: "/usr/bin/squeue")
    monkeypatch.setattr(guards.subprocess, "run", lambda *a, **k: called.append(a) or _Squeue())
    assert guards.slurm_job_is_running(job_id) is None
    assert not called, f"{job_id!r} was passed to squeue"


def test_a_slow_or_missing_squeue_falls_back_rather_than_hanging(monkeypatch):
    """Training must not wait on slurmctld, and must still run where there is no SLURM."""
    monkeypatch.setattr(guards.shutil, "which", lambda name: None)
    assert guards.slurm_job_is_running("1325538") is None

    monkeypatch.setattr(guards.shutil, "which", lambda name: "/usr/bin/squeue")

    def timeout(*args, **kwargs):
        raise guards.subprocess.TimeoutExpired(cmd="squeue", timeout=10)

    monkeypatch.setattr(guards.subprocess, "run", timeout)
    assert guards.slurm_job_is_running("1325538") is None
    # And the fallback is the old behaviour, not a refusal.
    info = {"host": "elsewhere", "pid": 1, "slurm_job_id": "1325538", "started": 1000.0}
    assert guards.lock_is_live(info, hostname="mine", now=1000.0 + 25 * 3600) is False


def test_both_messages_say_how_the_lock_was_judged(tmp_path, monkeypatch, capsys):
    """`still running: assumed, not checked` is the honest thing to print when the
    guard could not ask, and a takeover must say what it concluded -- the refusal
    that cost job 1349666 gave the reader no way to tell which it was."""
    info = {
        "host": "jpbo-002-25",
        "pid": 326978,
        "slurm_job_id": "1325538",
        "started": time.time(),
    }
    refusal = guards.concurrent_run_refusal(
        "large_pretrained", tmp_path, info, reason="SLURM job 1325538 is still in the queue"
    )
    assert "SLURM job 1325538 is still in the queue" in refusal
    assert "squeue -j 1325538" in refusal

    guards.run_lock_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    guards.run_lock_path(tmp_path).write_text(json.dumps(info))
    _with_squeue(monkeypatch, _Squeue(returncode=0, stdout="TIMEOUT\n"))
    taken = guards.acquire_run_lock(tmp_path)
    assert "no longer in the queue" in capsys.readouterr().out
    guards.release_run_lock(taken)


def test_a_lock_left_by_a_dead_process_on_this_host_is_taken_over(tmp_path, monkeypatch):
    """A crash, a `scancel` or a Ctrl-C must not lock a participant out of their own
    run name -- a guard that does that is worse than the silence it replaced."""
    lock = guards.acquire_run_lock(tmp_path)
    monkeypatch.setattr(
        guards.os, "kill", lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError())
    )
    # Run the suite inside an allocation and the lock carries a real, running job
    # id, which now outranks the pid; this test is about the pid path.
    monkeypatch.setattr(guards, "slurm_job_is_running", lambda job_id: None)
    retaken = guards.acquire_run_lock(tmp_path)  # must not raise
    assert retaken == lock
    guards.release_run_lock(retaken)


def test_a_lock_from_another_host_is_believed_until_it_is_a_day_old():
    """Liveness is only answerable for our own host; elsewhere the age is all there is."""
    fresh = {"host": "some-other-node", "pid": 1, "started": 1000.0}
    assert guards.lock_is_live(fresh, hostname="mine", now=1000.0 + 3600) is True
    assert guards.lock_is_live(fresh, hostname="mine", now=1000.0 + 25 * 3600) is False


@pytest.mark.parametrize("env", [{"SLURM_PROCID": "1"}, {"LOCAL_RANK": "2"}])
def test_a_ddp_rank_takes_no_lock(tmp_path, env):
    """`srun --ntasks=4` runs the whole command line once per rank and Lightning's
    launcher re-runs it once per device: a lock per rank would refuse every
    multi-GPU run this kit ships a cluster config for."""
    assert guards.acquire_run_lock(tmp_path, env=env) is None
    assert not guards.run_lock_path(tmp_path).exists()


def test_releasing_a_lock_somebody_else_took_over_leaves_it_alone(tmp_path):
    guards.acquire_run_lock(tmp_path)
    path = guards.run_lock_path(tmp_path)
    path.write_text(json.dumps({"host": "elsewhere", "pid": -1, "started": time.time()}))
    guards.release_run_lock(path)
    assert path.is_file(), "a stale-declared lock was deleted out from under its new owner"
    path.unlink()


# ---------------------------------------------------------------------------
# 13. The startup guard, wired end to end in-process
# ---------------------------------------------------------------------------
def _composed(*overrides: str):
    """The real config, with the `hydra` node the callback reads."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR), job_name="guards"):
        yield compose(
            config_name="config",
            overrides=["cluster=local", "module=tiny", "dataloader=glorys_tiny", *overrides],
            return_hydra_config=True,
        )


def test_the_startup_guard_takes_the_lock_it_checks(tmp_path, capsys):
    """The lock has to be taken by `check`, not merely be available to it.

    Taken LAST: every refusal above it must leave `modelstore/<name>/` untouched,
    which `test_the_real_command_line_refuses_a_short_run_on_stdout` pins from
    the other side.
    """
    exp_dir = tmp_path / "my_run"
    for config in _composed(
        f"++exp_dir={exp_dir}", "++name=my_run", "++max_steps=200", "++save_step_frequency=100"
    ):
        guard = guards.StartupGuard()
        guard.check(config, config_name="config")
        assert guards.run_lock_path(exp_dir).is_file(), "check() did not take the lock"

        # A second process, i.e. a second `check` on the same directory.
        with pytest.raises(SystemExit):
            guards.StartupGuard().check(config, config_name="config")
        assert "already being trained" in capsys.readouterr().out

        guard.on_job_end(config, job_return=None)
        assert not guards.run_lock_path(exp_dir).exists(), "the lock outlived the job"


def test_a_refused_run_takes_no_lock(tmp_path):
    """`max_steps=200` with the preset's `save_step_frequency=1000` saves nothing and
    is refused -- and must leave nothing behind, or the next attempt meets a lock
    from a run that never happened."""
    exp_dir = tmp_path / "my_run"
    for config in _composed(f"++exp_dir={exp_dir}", "++name=my_run", "++max_steps=200"):
        with pytest.raises(SystemExit):
            guards.StartupGuard().check(config, config_name="config")
        assert not exp_dir.exists()


# ---------------------------------------------------------------------------
# 14. The error PyTorch reports, and the fix this kit has
# ---------------------------------------------------------------------------
def test_an_out_of_memory_death_names_batch_size_and_not_the_allocator():
    """Measured: `++batch_size=64` died with PyTorch's stock
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` suggestion, which is about
    fragmentation. The fix here is the batch size.

    MUTANT: returning None unconditionally from `oom_advice` fails this; keying
    it on the exception type alone misses `torch.OutOfMemoryError` raised as a
    plain `RuntimeError`, which is the second case below.
    """

    class OutOfMemoryError(RuntimeError):
        pass

    advice = guards.oom_advice(OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"), 64)
    assert advice is not None
    assert "++batch_size=32" in advice
    assert "gradient_checkpointing" in advice and "make benchmark" in advice
    assert "expandable_segments" in advice, "it must say why PyTorch's own advice is not the fix"

    assert guards.oom_advice(RuntimeError("CUDA out of memory"), 8) is not None


def test_an_unrelated_crash_is_not_dressed_up_as_an_out_of_memory():
    assert guards.oom_advice(ValueError("shapes do not match"), 8) is None
    assert guards.oom_advice(KeyboardInterrupt(), 8) is None


def test_the_guard_prints_the_out_of_memory_advice_when_the_job_dies(tmp_path, capsys):
    """Wiring: hydra calls `on_job_end` for a FAILED job with the exception in
    `job_return._return_value` (the public property re-raises it). Without this
    hook the advice exists and is never printed."""

    class Return:
        _return_value = RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")

    for config in _composed("++name=x", f"++exp_dir={tmp_path / 'x'}", "++batch_size=64"):
        guards.StartupGuard().on_job_end(config, job_return=Return())
    printed = capsys.readouterr().out
    assert "++batch_size=32" in printed
    assert "module=tiny" in printed
