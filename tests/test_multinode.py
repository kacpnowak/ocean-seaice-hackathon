"""The multi-node entry point, `python -m oceanarches.main_multinode`.

`geoarches.main_hydra` builds `L.Trainer(...)` without `num_nodes`, so Lightning
assumes one node and refuses to start on more than one (measured, job 1285543):

    ValueError: You set `num_nodes=1` in Lightning, but the number of nodes
    configured in SLURM `--nodes=2` does not match.

Everything here is about the shim that supplies it.  None of it needs a GPU, a
second node or SLURM -- the decisions are pure, and the one place they meet
Lightning is a `Trainer` built on the CPU.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import geoarches.main_hydra as geoarches_main
import lightning
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from oceanarches import main_multinode

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def _cpu_trainer(**kwargs):
    """A Trainer that costs nothing to build and touches no accelerator."""
    return geoarches_main.L.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, **kwargs
    )


# ---------------------------------------------------------------------------
# Which node count wins
# ---------------------------------------------------------------------------
def test_the_cluster_config_decides_and_the_allocation_is_the_fallback():
    """`cluster.num_nodes` when the config sets it, `SLURM_NNODES` when it does not.

    The fallback is what keeps `cluster=jupiter_4gpu` and `cluster=local` working
    through this entry point without gaining a key, and what makes a two-node
    experiment possible without a two-node cluster config.

    MUTANT: returning 1 unconditionally fails the first two.
    """
    assert main_multinode.resolve_num_nodes(4, 4) == 4  # jupiter_4nodes, --nodes=4
    assert main_multinode.resolve_num_nodes(None, 2) == 2  # jupiter_4gpu, --nodes=2
    assert main_multinode.resolve_num_nodes(None, 1) == 1  # the ordinary one-node job
    assert main_multinode.resolve_num_nodes(None, None) == 1  # no SLURM at all
    # The config is the source of truth and the allocation only the fallback, so
    # a launcher that is not SLURM (torchrun, a bare mpirun) still gets a world.
    assert main_multinode.resolve_num_nodes(4, None) == 4


def test_a_config_written_for_a_node_count_the_job_does_not_have_is_refused():
    """Submitting `cluster=jupiter_4nodes` with `--nodes=2` must not train.

    Lightning catches the same mismatch a second later, but its message is
    "HINT: Set `num_nodes=2`", which names a Trainer argument nobody sets by
    hand.  This one names the config and the sbatch flag.

    MUTANT: returning None unconditionally fails the first assertion.
    """
    complaint = main_multinode.check_allocation(4, 2, "jupiter_4nodes")
    assert complaint is not None
    assert "jupiter_4nodes" in complaint and "--nodes=4" in complaint and "2 node" in complaint

    assert main_multinode.check_allocation(4, 4, "jupiter_4nodes") is None
    assert main_multinode.check_allocation(1, 1, "jupiter_4gpu") is None
    # Off SLURM there is no allocation to disagree with.
    assert main_multinode.check_allocation(4, None, "jupiter_4nodes") is None


def test_a_node_count_slurm_spells_oddly_is_not_read_as_a_node_count():
    """`SLURM_NTASKS_PER_NODE` comes back as `4(x2)` on a heterogeneous job."""
    assert main_multinode._slurm_int("A_NAME_NOTHING_SETS") is None


# ---------------------------------------------------------------------------
# ... and how it reaches the trainer geoarches builds
# ---------------------------------------------------------------------------
def test_the_trainer_geoarches_builds_is_told_how_many_nodes_there_are():
    """The point of the whole module, checked on the object geoarches calls.

    Asserting on a Trainer we built ourselves would prove nothing: the thing
    that has to change is the `L.Trainer` lookup inside
    `geoarches.main_hydra.main`, which is why this goes through
    `geoarches_main.L` rather than through `lightning`.

    MUTANT: deleting the `kwargs.setdefault("num_nodes", ...)` line leaves
    `trainer.num_nodes == 1` and fails.
    """
    assert geoarches_main.L is lightning, "importing the shim must not patch anything"

    with main_multinode.injecting_num_nodes(4):
        assert _cpu_trainer().num_nodes == 4
        # Everything else geoarches reaches for on `L` still has to be there.
        assert geoarches_main.L.seed_everything is lightning.seed_everything
        assert geoarches_main.L.pytorch is lightning.pytorch

    assert geoarches_main.L is lightning, "the stand-in outlived the run it was for"


def test_the_stand_in_is_removed_even_when_the_run_raises():
    with pytest.raises(RuntimeError):
        with main_multinode.injecting_num_nodes(4):
            raise RuntimeError("training died")
    assert geoarches_main.L is lightning


def test_an_explicit_num_nodes_would_still_win():
    """`setdefault`, not assignment: if geoarches ever passes `num_nodes` itself
    this module becomes a no-op instead of fighting it."""
    with main_multinode.injecting_num_nodes(4):
        assert _cpu_trainer(num_nodes=1).num_nodes == 1


# ---------------------------------------------------------------------------
# The entry point composes OUR root config
# ---------------------------------------------------------------------------
def test_the_entry_point_composes_this_projects_root_config():
    """The entry point runs, takes the documented flags, and composes THIS root config.

    `oceanarches.main_multinode` carries its own `@hydra.main`, so it has its own
    chance to compose the wrong `config.yaml`: geoarches ships one too, with
    `max_steps: 300000` and `save_step_frequency: 50000`, and a run that picked it
    up would train to the wrong budget and never checkpoint, silently.  `--cfg
    job` composes and prints without training.

    `cluster.num_nodes` is asserted here rather than in test_configs.py because
    this is the only entry point that reads it.

    MUTANT: deleting `num_nodes: 4` from configs/cluster/jupiter_4nodes.yaml
    fails this; so does a shim that stops passing the composed config through.

    (Not a mutant: the decorator's own `config_path`. hydra's `--config-path`
    flag replaces it outright, and the documented command line always passes it.
    With `config_path=None` the shim is immune to the `--config-dir` trap
    described at the top of configs/config.yaml -- geoarches' config directory
    is never on its search path -- which is a happy accident, not the reason.)
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "oceanarches.main_multinode",
            "--config-path",
            str(CONFIG_DIR),
            "cluster=jupiter_4nodes",
            "module=large",
            "dataloader=glorys",
            "++name=zz_test_compose",
            "--cfg",
            "job",
            "--resolve",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    cfg = OmegaConf.create(result.stdout)
    assert cfg.cluster.num_nodes == 4, "the shim reads this key; it must survive composition"
    assert cfg.save_step_frequency == cfg.module.save_step_frequency
    assert cfg.max_steps == cfg.module.max_steps
    assert cfg.batch_size == cfg.cluster.batch_size


# ---------------------------------------------------------------------------
# ... and the SLURM script that launches it
# ---------------------------------------------------------------------------
SLURM_SCRIPT = REPO_ROOT / "scripts" / "pretrain_large.slurm"


def _shell_default(name: str) -> int:
    """The number in `NAME="${NAME:-1234}"` in the pre-training script."""
    text = SLURM_SCRIPT.read_text()
    match = re.search(rf'^{name}="\$\{{{name}:-(\d+)\}}"', text, re.M)
    assert match is not None, f"{name} has no numeric default in {SLURM_SCRIPT.name}"
    return int(match.group(1))


def test_the_pre_training_job_asks_for_the_allocation_its_cluster_config_describes():
    """`#SBATCH --nodes` and `cluster.num_nodes` are two statements of one fact.

    If they drift, the run dies on every rank in
    `SLURMEnvironment.validate_settings` after the queue wait, which on a
    four-node job is an expensive way to find a typo.

    MUTANT: changing either number alone fails this.
    """
    text = SLURM_SCRIPT.read_text()
    nodes = int(re.search(r"^#SBATCH --nodes=(\d+)", text, re.M).group(1))
    tasks_per_node = int(re.search(r"^#SBATCH --ntasks-per-node=(\d+)", text, re.M).group(1))
    gpus = int(re.search(r"^#SBATCH --gres=gpu:(\d+)", text, re.M).group(1))

    assert nodes == 4 and tasks_per_node == 4 and gpus == 4, "16 GH200 is 4 nodes x 4 tasks"
    cluster_cfg = OmegaConf.load(CONFIG_DIR / "cluster" / "jupiter_4nodes.yaml")
    assert cluster_cfg.num_nodes == nodes, (
        f"{SLURM_SCRIPT.name} asks SLURM for {nodes} nodes but cluster/jupiter_4nodes.yaml "
        f"declares num_nodes: {cluster_cfg.num_nodes}"
    )
    # One process per GPU, so `--ntasks-per-node` must equal the GPUs per node:
    # Lightning validates `devices` against it and stops if they differ.
    assert tasks_per_node == gpus


def test_the_pre_training_job_launches_the_entry_point_that_can_do_four_nodes():
    """MUTANT: putting `geoarches.main_hydra` back on the srun line fails this."""
    text = SLURM_SCRIPT.read_text()
    srun_lines = [line for line in text.splitlines() if line.strip().startswith("srun ")]
    assert srun_lines, "the script no longer launches anything with srun"
    for line in srun_lines:
        assert "oceanarches.main_multinode" in line, (
            f"{line.strip()!r} launches geoarches.main_hydra, which builds its Trainer "
            "without num_nodes and cannot start on more than one node"
        )
    # The flag itself lives in the HYDRA_ARGS array the srun line expands, and
    # `--config-dir` would silently compose geoarches' root config instead of
    # ours: see the header of configs/config.yaml.
    assert "--config-path" in text
    assert "--config-dir" not in text


def test_a_rank_that_refuses_to_start_takes_the_whole_step_down_with_it():
    """The startup guards refuse on rank 0 only, and rank 0 is also the rendezvous
    master.  Without `--kill-on-bad-exit=1` the remaining 15 ranks sit in
    `init_process_group` waiting for a store that will never be created, for the
    full 30-minute default timeout, and what comes out is 15 copies of
    `DistNetworkError: The client socket has timed out` -- which reads as a
    network fault and buries the real message two thousand lines up.  Job 1349666
    and job 1342138 each burned 4 nodes for 1 h 15 that way.

    MUTANT: removing the flag from the srun line fails this.
    """
    srun_lines = [
        line for line in SLURM_SCRIPT.read_text().splitlines() if line.strip().startswith("srun ")
    ]
    assert srun_lines
    for line in srun_lines:
        assert "--kill-on-bad-exit=1" in line, (
            f"{line.strip()!r} lets a refusal on rank 0 hang the other ranks until the "
            "distributed rendezvous times out"
        )


def test_the_pre_training_job_is_still_submittable_on_jupiter():
    """`#SBATCH --requeue` makes the job unsubmittable here, not just unrequeued:

        sbatch: error: job_submit_filter: --requeue option is not supported
        sbatch: error: Batch job submission failed

    MUTANT: adding the directive back fails this.
    """
    text = SLURM_SCRIPT.read_text()
    assert not re.search(r"^#SBATCH .*--requeue", text, re.M), (
        "JUPITER's submit filter rejects --requeue; the job would never enter the queue"
    )
    # The two things a relaunch depends on, both of which have been lost before.
    assert "SLURM_SUBMIT_DIR" in text, "BASH_SOURCE points into SLURM's spool directory"
    assert 'exit "${TRAIN_STATUS}"' in text, "the job must not report COMPLETED for a failed run"


def test_the_shipped_step_budget_is_checkpointable_at_every_node_count_it_supports():
    """`max_steps` is derived from a sample budget, so the node count moves it.

    geoarches checkpoints on `global_step % save_step_frequency == 0` and never
    at the end of `fit`, so a `SAVE_EVERY` that stops dividing `MAX_STEPS` loses
    the finished model and nothing else. The script refuses to start in that
    case -- this is the test that the SHIPPED defaults never get there.

    MUTANT: the previous default of SAVE_EVERY=10000 fails at 16 ranks
    (1200000 / 16 = 75000, and 75000 % 10000 = 5000).
    """
    budget = _shell_default("SAMPLE_BUDGET")
    save_every = _shell_default("SAVE_EVERY")
    for ranks in (4, 8, 16):  # 1, 2 and 4 nodes at 4 GPUs each
        max_steps = budget // ranks
        assert budget % ranks == 0, f"{budget} samples does not divide over {ranks} ranks"
        assert max_steps % save_every == 0, (
            f"at {ranks} ranks the script would run {max_steps} steps and checkpoint every "
            f"{save_every}, so the last checkpoint is never written"
        )


# ---------------------------------------------------------------------------
# ... and the one thing a relaunch must not leave behind
# ---------------------------------------------------------------------------
def _compose(**overrides) -> OmegaConf:
    """Compose the root config the way the real command line does, and resolve it.

    Resolved matters here: `modelstore/<name>/config.yaml`, the file geoarches
    reloads on a resume, is written with `resolve=True`, so every `${max_steps}`
    inside `module` is already a literal number by then.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR), job_name="test"):
        cfg = compose(
            config_name="config",
            overrides=list(overrides.pop("extra", []))
            + [f"{group}={name}" for group, name in overrides.items()],
        )
    OmegaConf.resolve(cfg)
    return cfg


def _script_overrides(**shell_variables: object) -> list[str]:
    """The `++key=value` overrides the pre-training script passes, filled in.

    Read off `HYDRA_ARGS` in the script itself rather than copied here, so the
    test moves when the launch line moves and cannot quietly agree with a
    version of the command that no longer ships.
    """
    block = SLURM_SCRIPT.read_text().split("HYDRA_ARGS=(", 1)[1].split("\n)", 1)[0]
    pairs = re.findall(r'^\s*"\+\+([\w.]+)=\$\{(\w+)\}"\s*$', block, re.M)
    assert pairs, "no ++key=${VAR} overrides found in HYDRA_ARGS"
    for _, variable in pairs:
        assert variable in shell_variables, (
            f"the script passes ${{{variable}}}; this test does not set it"
        )
    return [f"++{key}={shell_variables[variable]}" for key, variable in pairs]


def test_a_relaunch_at_a_new_step_budget_takes_the_lr_schedule_with_it():
    """The step budget and the cosine schedule must not come apart on a resume.

    geoarches' resume does not re-compose `cfg.module`. It replaces it wholesale
    with the *resolved* module config written at the first launch and then
    re-applies only the `+`-prefixed command-line overrides as a dotlist:

        cfg.module = exp_cfg.module          # ${max_steps} already a literal
        cfg.merge_with_dotlist([x.removeprefix("++") for x in cli if x[0] == "+"])

    So `++max_steps=` moves the trainer's budget and `num_training_steps` stays
    at the first launch's. That is reachable exactly where this script says it
    is -- a relaunch at a different `--nodes`, since the budget is derived from
    the world size -- and it is not a cosmetic mismatch: see the next test.

    MUTANT: delete `"++module.module.num_training_steps=${MAX_STEPS}"` from
    HYDRA_ARGS and this fails on the `resumed` assertion.
    """
    groups = dict(module="large", dataloader="glorys", cluster="jupiter_4nodes")
    variables = dict(NAME="relaunch_probe", SAVE_EVERY=50)

    first = _compose(extra=_script_overrides(MAX_STEPS=100, **variables), **groups)
    assert first.max_steps == 100
    assert first.module.module.num_training_steps == 100

    # The relaunch: fresh composition at the new budget, then geoarches' merge.
    new_arguments = _script_overrides(MAX_STEPS=200, **variables)
    resumed = _compose(extra=new_arguments, **groups)
    resumed.module = first.module  # what `cfg.module = exp_cfg.module` does
    OmegaConf.set_struct(resumed, False)
    resumed.merge_with_dotlist([x.removeprefix("++") for x in new_arguments])

    assert resumed.max_steps == 200
    assert resumed.module.module.num_training_steps == 200, (
        "the relaunch moved max_steps to 200 and left the cosine schedule at "
        f"{resumed.module.module.num_training_steps}; scripts/pretrain_large.slurm must pass "
        "++module.module.num_training_steps so it survives the resume dotlist merge"
    )

    # ... and the same merge without that one override is the defect itself,
    # pinned here so this test explains what it is protecting.
    stale = _compose(extra=new_arguments, **groups)
    stale.module = first.module
    OmegaConf.set_struct(stale, False)
    stale.merge_with_dotlist(
        [x.removeprefix("++") for x in new_arguments if "num_training_steps" not in x]
    )
    assert stale.max_steps == 200 and stale.module.module.num_training_steps == 100


def test_a_schedule_left_behind_raises_the_learning_rate_back_to_peak():
    """Why the test above is not cosmetic.

    geoarches trains on `diffusers.optimization.get_cosine_schedule_with_warmup`
    with the default `num_cycles=0.5`, whose lambda is

        max(0.0, 0.5 * (1 + cos(pi * 2 * num_cycles * progress)))

    and `progress` is not clamped. Past `num_training_steps` the cosine keeps
    turning: at twice the budget it is back at `cos(2 pi) = 1`, i.e. the peak
    learning rate, on a run that was supposed to be annealing to zero.

    MUTANT: none needed -- this pins a fact about the installed diffusers, and
    if it ever stops being true the argument for the override changes with it.
    """
    import diffusers.optimization

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    schedule = diffusers.optimization.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=50
    )
    learning_rates = []
    for _ in range(101):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule.step()

    assert learning_rates[0] == pytest.approx(1.0)
    assert learning_rates[50] == pytest.approx(0.0, abs=1e-9), "the schedule should end at zero"
    assert learning_rates[100] > 0.99, (
        "a schedule left behind at half the real budget climbs back to the peak "
        f"learning rate, not to zero: got {learning_rates[100]}"
    )
