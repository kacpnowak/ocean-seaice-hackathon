"""Training entry point for runs that span more than one node.

Use it exactly like `geoarches.main_hydra` -- same flags, same config groups,
same `--config-path` rule::

    .venv/bin/python -m oceanarches.main_multinode --config-path $PWD/configs \\
        cluster=jureca_4nodes module=base dataloader=glorys ++name=my_run

---------------------------------------------------------------------------
Why this module exists
---------------------------------------------------------------------------
`geoarches.main_hydra` builds its trainer as

    trainer = L.Trainer(devices="auto", accelerator="auto",
                        strategy="ddp_find_unused_parameters_true", ...)

with no `num_nodes`, and Lightning defaults that to 1.  On one node that is
right and nothing here changes.  On more than one it is wrong, and Lightning
says so rather than mistraining:

    $ srun --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \\
          .venv/bin/python -m geoarches.main_hydra ... cluster=jureca_4gpu
    ValueError: You set `num_nodes=1` in Lightning, but the number of nodes
    configured in SLURM `--nodes=2` does not match. HINT: Set `num_nodes=2`.

(measured).  That check is `SLURMEnvironment.validate_settings`,
reached from `_SubprocessScriptLauncher.launch`, and it fires on every rank
before a single batch is read.  So the multi-node failure mode is a crash, not
eight processes quietly training four independent models -- but the run still
does not happen, and `num_nodes` is the only thing missing.

---------------------------------------------------------------------------
What it does, and what it deliberately does not do
---------------------------------------------------------------------------
geoarches is installed non-editable from a pinned commit and this project has
never forked or vendored any of it.  So this module does not copy
`main_hydra`'s body: it composes the same hydra config, swaps the `lightning`
module that `geoarches.main_hydra` looks `Trainer` up on for a stand-in that
fills in `num_nodes`, and then calls geoarches' own `main` with the composed
config.  Everything downstream is geoarches' code, unchanged: the resume
search through `modelstore/<name>/checkpoints/`, the `CheckpointEveryNSteps`
callback, the `modelstore/<name>/config.yaml` dump that
`base_module.load_module` needs, the wandb logger, the SIGTERM handling.

`hydra.main`'s decorated function takes an optional already-composed config and
hands it straight to the undecorated body (`hydra/main.py`, `cfg_passthrough`),
which is what makes the delegation a single call.

The stand-in is a per-call object and the swap is undone in a `finally`, so
importing this module has no effect on anything else in the process -- tests
included.

`num_nodes` comes from `cluster.num_nodes` when the cluster config sets it
(configs/cluster/jureca_4nodes.yaml does) and from the allocation otherwise,
which is why `cluster=jureca_4gpu` and `cluster=local` need no new key and
behave exactly as they did.  The two can only disagree if you submit a config
for a node count you did not ask SLURM for; `check_allocation` says so in this
project's vocabulary, and Lightning would catch it a second later anyway.

`check_allocation` here is only about the NODE COUNT.  The checks that stop a run
which would save no checkpoint, train nothing, or apply an override to a key
nothing reads live in `oceanarches/guards.py` and reach this entry point the same
way they reach `geoarches.main_hydra`: through the `hydra.callbacks` entry in
`configs/config.yaml`, which every route composes.  That is deliberate -- a guard
written into this shim would cover the four-node job and miss `make train-tiny`,
which is where every one of the measured failures happened.
"""

from __future__ import annotations

import contextlib
import os

import geoarches.main_hydra as geoarches_main
import hydra
import lightning as L  # noqa: N812
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

__all__ = ["check_allocation", "injecting_num_nodes", "main", "resolve_num_nodes"]


def _slurm_int(name: str) -> int | None:
    """`os.environ[name]` as an int, or None when it is unset or not a number.

    `SLURM_NTASKS_PER_NODE` can be spelled `4(x2)` on a heterogeneous job, so
    "not a number" is a real case and not a defensive flourish.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_num_nodes(configured: int | None, allocated: int | None) -> int:
    """How many nodes to tell Lightning about.

    `configured` is `cluster.num_nodes` or None if the cluster config does not
    set it; `allocated` is `SLURM_NNODES` or None off SLURM.  The allocation is
    the fallback rather than the override: a cluster config that names a node
    count is a deliberate statement about the job it is meant for, and
    `check_allocation` is what stops it being applied to a different one.
    """
    if configured is not None:
        return int(configured)
    if allocated is not None:
        return allocated
    return 1


def check_allocation(num_nodes: int, allocated: int | None, cluster_name: str) -> str | None:
    """The complaint to print when the config asks for a node count SLURM did not give.

    Returns None when there is nothing wrong, including off SLURM entirely.
    Lightning raises on the same mismatch a few seconds later, but its message
    ("HINT: Set `num_nodes=2`") names a `Trainer` argument no participant sets
    by hand, so it is worth saying which config and which sbatch flag disagree.
    """
    if allocated is None or allocated == num_nodes:
        return None
    return (
        f"cluster={cluster_name} is a {num_nodes}-node configuration "
        f"(cluster.num_nodes: {num_nodes}), but this allocation has {allocated} node(s). "
        f"Either submit with --nodes={num_nodes}, or pick the cluster config that "
        "matches the allocation (jureca_4gpu is one node, jureca_4nodes is four)."
    )


class _LightningWithNumNodes:
    """Stands in for the `lightning` module inside `geoarches.main_hydra`.

    Everything except `Trainer` is forwarded to the real module, so geoarches
    keeps `L.seed_everything`, `L.pytorch.loggers.WandbLogger` and anything else
    it reaches for.
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes

    def __getattr__(self, name: str):  # only called for attributes not set above
        return getattr(L, name)

    def Trainer(self, *args, **kwargs) -> L.Trainer:  # noqa: N802 - it stands in for a class
        # `setdefault`: if geoarches ever starts passing `num_nodes` itself, its
        # value wins and this module becomes a no-op rather than a conflict.
        kwargs.setdefault("num_nodes", self.num_nodes)
        return L.Trainer(*args, **kwargs)


@contextlib.contextmanager
def injecting_num_nodes(num_nodes: int):
    """Make `geoarches.main_hydra` build its trainer with `num_nodes`, then put it back.

    The swap is on the *module attribute* `geoarches.main_hydra.L`, not on
    `lightning.Trainer` itself: `L` is the real `lightning` module, shared with
    every other importer in the process, and rebinding `lightning.Trainer` would
    reach into all of them.  Restoring in a `finally` keeps a failed run from
    leaving the stand-in behind.
    """
    original = geoarches_main.L
    geoarches_main.L = _LightningWithNumNodes(num_nodes)
    try:
        yield
    finally:
        geoarches_main.L = original


# `config_path=None`: this module ships no configs of its own, and the documented
# command line always passes `--config-path $PWD/configs` (absolute, and NOT
# `--config-dir` -- see the header of configs/config.yaml).  Forgetting it here is
# a loud "cannot find primary config 'config'" rather than a silent fall-through
# to geoarches' own defaults, because geoarches' config directory is never on
# this entry point's search path.
@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    num_nodes = resolve_num_nodes(
        OmegaConf.select(cfg, "cluster.num_nodes"), _slurm_int("SLURM_NNODES")
    )
    # The *group option* actually selected, i.e. "jureca_4nodes" -- the thing a
    # participant typed -- not cfg.name, which is the run name.
    cluster_name = HydraConfig.get().runtime.choices.get("cluster", "?")
    complaint = check_allocation(num_nodes, _slurm_int("SLURM_NNODES"), cluster_name)
    if complaint is not None:
        raise SystemExit(f"FATAL: {complaint}")

    print(
        f"[oceanarches] num_nodes={num_nodes}, "
        f"SLURM_NNODES={os.environ.get('SLURM_NNODES', '<unset>')}, "
        f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS', '<unset>')}, "
        f"SLURM_PROCID={os.environ.get('SLURM_PROCID', '<unset>')}"
    )

    # `geoarches_main.main` is hydra-decorated; handed an already-composed config
    # it calls the undecorated body directly, so this is geoarches' `main(cfg)`
    # with nothing copied and nothing skipped.
    with injecting_num_nodes(num_nodes):
        geoarches_main.main(cfg)


if __name__ == "__main__":
    main()
