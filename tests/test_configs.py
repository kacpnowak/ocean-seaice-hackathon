"""Every shipped config combination must compose and instantiate.

A config that does not compose is the single most common way to lose twenty
minutes at a hackathon, and it always fails on someone else's laptop rather than
on the author's.  These tests compose all
``module x dataloader`` combinations plus every cluster, instantiate the
embedder and the backbone, and check that the shapes they agree on are the
shapes the data actually has.

They deliberately do *not* assert which Lightning module the configs point at:
Task 6 replaces it, and this file should not have to change when it does.
"""

from __future__ import annotations

import itertools
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from oceanarches.backbones.ocean_embedder import (
    GEOARCHES_Z_DIM,
    latent_grid,
    latent_z_dim,
    usable_depth_counts,
)
from oceanarches.dataloaders.variables import (
    COMPONENTS,
    DEPTH_PRESETS,
    N_LAT,
    N_LON,
)

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

MODULES = ["tiny", "small", "base"]
DATALOADERS = ["glorys", "glorys_tiny", "glorys_ocean", "glorys_seaice"]
#: `glorys_forced` is `glorys_tiny` on a different window -- same component, same
#: channel counts -- so it is composed and checked below rather than run through
#: every combination test, which would cost four more model instantiations for
#: shapes that are identical by construction.
FORCINGS = ["none", "file"]
CLUSTERS = ["local", "jureca_1gpu", "jureca_4gpu", "jureca_4nodes"]

#: Keys geoarches' main_hydra.py reads off the root config.  If one of these
#: disappears, training dies after the dataset has been opened, not before.
REQUIRED_ROOT_KEYS = [
    "mode",
    "exp_dir",
    "resume",
    "seed",
    "max_steps",
    "batch_size",
    "log",
    "log_freq",
    "limit_val_batches",
    "save_step_frequency",
    "accumulate_grad_batches",
    "debug",
    "name",
    "project",
]
#: Keys `geoarches.main_hydra` really reads off `cfg.cluster`. There is
#: deliberately no `gpus` here: geoarches builds its Trainer with
#: `devices="auto"` and never looks at a GPU count, so a `gpus:` key sitting
#: below the comment that explains how `devices="auto"` grabs all four cards
#: read exactly like the knob that prevents it. It was deleted, not wired up --
#: `export CUDA_VISIBLE_DEVICES=0` is the thing that works.
REQUIRED_CLUSTER_KEYS = [
    "wandb_mode",
    "use_custom_requeue",
    "precision",
    "batch_size",
    "cpus",
]


def build(extra: list[str] | None = None, **overrides) -> OmegaConf:
    """Compose the root config with this directory as the primary config path.

    That is what `main_hydra --config-path <abs>/configs` does, and it is *not*
    what `--config-dir configs` does -- see
    `test_the_documented_command_line_uses_this_root_config`.

    Args:
        extra: Raw hydra overrides, for dotted keys that cannot be keyword
            arguments (``"++module.backbone.emb_dim=32"``).
        overrides: ``group=name`` selections.
    """
    args = [f"{key}={value}" for key, value in overrides.items()] + list(extra or [])
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR, job_name="test"):
        cfg = compose(config_name="config", overrides=args)
    OmegaConf.resolve(cfg)
    return cfg


ALL_COMBINATIONS = list(itertools.product(MODULES, DATALOADERS))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module, dataloader", ALL_COMBINATIONS)
def test_every_combination_composes(module, dataloader):
    cfg = build(module=module, dataloader=dataloader)
    for key in REQUIRED_ROOT_KEYS:
        assert key in cfg, f"main_hydra reads cfg.{key}"
    for key in ("module", "backbone", "embedder", "train", "val", "inference"):
        assert key in cfg.module
    for block in ("train", "val", "inference"):
        assert "rollout_iterations" in cfg.module[block]
        assert cfg.module[block].metrics, f"{block} has no metrics configured"


def test_the_forced_dataloader_matches_the_one_it_is_a_window_of():
    """`glorys_forced` differs from `glorys_tiny` only in which days it serves."""
    forced = build(module="tiny", dataloader="glorys_forced")
    tiny = build(module="tiny", dataloader="glorys_tiny")

    assert forced.dataloader.component == tiny.dataloader.component
    for key in ("n_surface_in", "n_surface_out", "n_level_in", "n_level_out"):
        assert forced.dataloader[key] == tiny.dataloader[key]
    assert forced.dataloader.dataset.variables == tiny.dataloader.dataset.variables
    assert _locate(forced.dataloader.dataset._target_) is not None
    # Every split it names is one the shipped forcing can actually serve.
    from oceanarches.dataloaders.glorys import SPLIT_DATES

    for block in ("dataset", "validation_args", "test_args"):
        assert forced.dataloader[block].domain in SPLIT_DATES


@pytest.mark.parametrize("forcing", FORCINGS)
def test_every_forcing_composes_and_the_embedder_reserves_what_it_declares(forcing):
    """`forcing.n_channels` and `embedder.forcing_ch` are one number under two
    names, and the two disagreeing is a shape error at the first batch, after the
    data has loaded and the GPU has been queued for."""
    cfg = build(module="tiny", dataloader="glorys_forced", forcing=forcing)

    assert cfg.module.embedder.forcing_ch == cfg.forcing.n_channels
    assert cfg.module.module.forcing == cfg.forcing.source
    if cfg.forcing.source is None:
        assert cfg.forcing.n_channels == 0
    else:
        # The variables list IS the channel list, in order.
        assert len(cfg.forcing.source.variables) == cfg.forcing.n_channels
        assert cfg.forcing.source._target_.endswith("XarrayForcing")


def test_the_forcing_default_is_no_forcing():
    """It is the best-tested path and every later task has to work with it.

    MUTANT: `- forcing: file` in configs/config.yaml fails here -- and would make
    every `make train-tiny` on the 1993-2018 split die on the first batch, since
    the shipped archive is 2024 only.
    """
    cfg = build(module="tiny", dataloader="glorys_tiny")
    assert cfg.forcing.n_channels == 0
    assert cfg.forcing.source is None
    assert cfg.module.embedder.forcing_ch == 0


@pytest.mark.parametrize("cluster", CLUSTERS)
def test_every_cluster_composes(cluster):
    cfg = build(cluster=cluster)
    for key in REQUIRED_CLUSTER_KEYS:
        assert key in cfg.cluster, f"main_hydra reads cfg.cluster.{key}"
    assert cfg.batch_size == cfg.cluster.batch_size
    # A sanity check on the key, not a statement about the machine.
    assert 1 <= cfg.cluster.cpus <= 72, "dataloader workers per process, not per node"


def test_no_cluster_config_offers_a_gpu_count_that_does_nothing():
    """`devices="auto"` is hardcoded in geoarches; a `gpus:` key cannot change it.

    MUTANT: adding `gpus: 1` back to any configs/cluster/*.yaml fails this.
    """
    for cluster in CLUSTERS:
        cfg = build(cluster=cluster)
        assert "gpus" not in cfg.cluster, (
            f"configs/cluster/{cluster}.yaml offers a `gpus` key. Nothing reads it: "
            "geoarches builds `L.Trainer(devices='auto')`. Use CUDA_VISIBLE_DEVICES."
        )


@pytest.mark.parametrize("module, dataloader", ALL_COMBINATIONS)
def test_targets_are_importable(module, dataloader):
    """`_target_` typos only surface at instantiation time, i.e. after the data loads."""
    cfg = build(module=module, dataloader=dataloader)
    for target in (
        cfg.module.module._target_,
        cfg.module.backbone._target_,
        cfg.module.embedder._target_,
        cfg.dataloader.dataset._target_,
    ):
        assert callable(_locate(target)), target


REPO_ROOT = Path(CONFIG_DIR).parent


def make_recipe(target: str) -> list[str]:
    """The command `make <target>` would run, as an argv list.

    `make -n` expands the Makefile's own variables ($(PY), $(CONFIG_DIR),
    $(CLUSTER), ...) and prints the recipe without running it, so what comes back
    is the participant's real command line rather than a copy of it that can
    drift.  The python interpreter is swapped for the one running the tests.

    `--no-print-directory` is required, not cosmetic: when pytest is started *by*
    make (`make test`), MAKELEVEL is inherited and this nested `make -n` prefixes
    its output with `make[1]: Entering directory ...`, which then parses as the
    command.  Without the flag these tests fail under `make test` and pass when
    pytest is run directly, which is the worst of both worlds.
    """
    make = shutil.which("make")
    if make is None:
        pytest.fail("`make` is not on PATH, so the Makefile recipes cannot be checked")
    printed = subprocess.run(
        [make, "--no-print-directory", "-n", target],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert printed.returncode == 0, printed.stderr[-2000:]
    argv = shlex.split(printed.stdout.replace("\\\n", " "))
    assert argv[0].endswith("python"), f"unexpected recipe for {target}: {printed.stdout!r}"
    return [sys.executable] + argv[1:]


def test_a_fresh_checkout_is_told_to_run_make_setup(tmp_path):
    """`make doctor` is the first command docs/01 gives a participant.

    In a fresh clone there is no `.venv`, and every target runs `$(PY)`. Without
    the parse-time guard in the Makefile this failed with a bare
    `.venv/bin/python: No such file or directory` and `Error 127`, which says
    nothing about what to do next. Reproduced on a real `git clone` of this repo.

    The guard is a parse-time `$(error)` and not a prerequisite on purpose: a
    prerequisite would add a line to every recipe and `make -n train-tiny` --
    which the two tests below read -- would stop being the training command.
    """
    make = shutil.which("make")
    if make is None:
        pytest.fail("`make` is not on PATH")
    shutil.copy(REPO_ROOT / "Makefile", tmp_path / "Makefile")
    result = subprocess.run(
        [make, "--no-print-directory", "doctor"], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "make setup" in combined, combined
    assert "No such file or directory" not in combined, combined

    # ... and `make setup` and `make help` still have to work there, or the fix
    # is a deadlock: the only way out is the command it is telling you to run.
    # `setup doctor` too -- the obvious one-liner after reading that message.
    for target in (["setup"], ["help"], ["clean"], ["setup", "doctor"]):
        dry = subprocess.run(
            [make, "--no-print-directory", "-n", *target],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert dry.returncode == 0, (
            f"`make {' '.join(target)}` is blocked in a fresh checkout: {dry.stderr}"
        )

    # A DANGLING .venv/bin/python -- an interpreter that has been removed, or a
    # uv-managed python that moved -- has to be caught too. `$(wildcard)` matches
    # a broken symlink and would have sailed straight past it into Error 127.
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").symlink_to("/nonexistent/python")
    dangling = subprocess.run(
        [make, "--no-print-directory", "doctor"], capture_output=True, text=True, cwd=tmp_path
    )
    assert dangling.returncode != 0
    assert "make setup" in dangling.stdout + dangling.stderr


@pytest.mark.parametrize("target", ["train-tiny", "train"])
def test_the_makefile_hands_hydra_config_path_not_config_dir(target):
    """Read the flag off the real recipe, not off a copy of it.

    Hardcoding the command line here would validate hydra's semantics and leave
    the Makefile free to go back to `--config-dir` with the suite still green --
    which is the defect below, and it cost a 40-minute GPU run.
    """
    argv = make_recipe(target)
    assert "geoarches.main_hydra" in argv, argv
    assert "--config-dir" not in argv, (
        f"`make {target}` passes --config-dir. hydra searches it *after* the directory in "
        "@hydra.main(config_path=...), so geoarches' root config.yaml wins and ours is "
        "silently ignored. Use --config-path with an absolute path."
    )
    assert "--config-path" in argv, f"`make {target}` does not tell hydra where our configs are"
    given = Path(argv[argv.index("--config-path") + 1])
    assert given.is_absolute(), f"--config-path {given} must be absolute"
    assert given.resolve() == Path(CONFIG_DIR).resolve()


@pytest.mark.parametrize("target", ["train-tiny", "train"])
def test_the_documented_command_line_uses_this_root_config(target):
    """Run the recipe `make <target>` would run and check OUR root config wins.

    This is the test that would have caught a 40-minute wasted GPU run. hydra's
    `--config-dir` is searched *after* the directory named in
    `@hydra.main(config_path=...)`, which for `geoarches.main_hydra` is geoarches'
    own `configs/`. Both hold a `config.yaml`, so `--config-dir configs` silently
    picks geoarches': `max_steps: 300000`, `save_step_frequency: 50000`,
    `limit_val_batches: null`. The *groups* (`module=tiny` and friends) still come
    from here, because geoarches has no `module/tiny.yaml`, so the run looks
    entirely normal -- it just trains 67x too long and never checkpoints.

    Composing through `initialize_config_dir` (what `build` above does) cannot
    see this, because it makes this directory primary by construction. So this
    test takes the command out of the Makefile with `make -n` and shells out to
    the real entry point with `--cfg job`, which composes and prints without
    training.
    """
    # The recipe already carries its own ++name; --cfg job composes and prints
    # the config without training and without touching modelstore/.
    argv = make_recipe(target) + ["--cfg", "job", "--resolve"]
    result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    cfg = OmegaConf.create(result.stdout)
    selected = {
        key: value
        for key, value in (arg.split("=", 1) for arg in argv if "=" in arg and "/" not in arg)
        if key in ("cluster", "module", "dataloader")
    }
    assert set(selected) == {"cluster", "module", "dataloader"}, (
        f"could not read the group selections out of `make {target}`: {argv}"
    )
    expected = build(**selected)
    for key in ("max_steps", "save_step_frequency", "limit_val_batches", "batch_size"):
        assert cfg[key] == expected[key], (
            f"cfg.{key} is {cfg[key]!r} on the real command line but {expected[key]!r} in "
            "configs/config.yaml -- hydra is composing geoarches' root config instead of ours"
        )
    assert cfg.max_steps == cfg.module.max_steps, "max_steps must follow the size preset"


def test_the_shipped_default_writes_the_file_load_module_needs():
    """`log: True` is what makes `modelstore/<name>/config.yaml` appear.

    geoarches' `main_hydra` writes it inside `if cfg.log and main_node and ...`,
    and `load_module` reads exactly that file, so a default run with `log: False`
    trains, checkpoints, and can never be reloaded. Every cluster keeps wandb
    offline so the flag costs nothing.
    """
    cfg = build()
    assert cfg.log is True, (
        "configs/config.yaml ships log: False -- main_hydra then never writes "
        "modelstore/${name}/config.yaml and load_module() cannot reload the run"
    )
    for cluster in CLUSTERS:
        assert build(cluster=cluster).cluster.wandb_mode == "offline", (
            f"cluster/{cluster}.yaml must keep wandb offline: with log: True on by default, "
            "an online logger would ask every participant for a wandb account"
        )


@pytest.mark.parametrize("module", MODULES)
def test_the_last_step_is_checkpointed(module):
    """`save_step_frequency` must divide `max_steps`.

    geoarches' CheckpointEveryNSteps fires on
    `trainer.global_step % save_step_frequency == 0`, and nothing checkpoints at
    the end of `fit`. If the two do not divide, the *finished* model is the one
    thing that never reaches disk -- you get the second-to-last checkpoint and no
    warning at all.
    """
    cfg = build(module=module)
    assert cfg.max_steps % cfg.save_step_frequency == 0, (
        f"{module}: max_steps={cfg.max_steps} is not a multiple of "
        f"save_step_frequency={cfg.save_step_frequency}, so step {cfg.max_steps} is lost"
    )


def test_test_mode_composes():
    cfg = build(module="tiny", dataloader="glorys_tiny", mode="test")
    assert cfg.mode == "test"
    assert cfg.dataloader.test_args.domain == "test", "never score on the training years"


# ---------------------------------------------------------------------------
# The preset <-> variables.py contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", MODULES)
def test_depth_preset_matches_variables_py(module):
    cfg = build(module=module)
    assert list(cfg.module.depth_indices) == DEPTH_PRESETS[module]
    assert cfg.module.n_depths == len(DEPTH_PRESETS[module])


@pytest.mark.parametrize("module", MODULES)
def test_tensor_size_matches_latent_z_dim(module):
    """The test that stops a depth-preset change from silently breaking the model."""
    cfg = build(module=module)
    tensor_size = list(cfg.module.backbone.tensor_size)
    patch_size = list(cfg.module.embedder.patch_size)
    assert tensor_size[0] == latent_z_dim(cfg.module.n_depths, patch_size)
    assert tuple(tensor_size[1:]) == latent_grid((N_LAT, N_LON), patch_size)


@pytest.mark.parametrize("module", MODULES)
def test_backbone_and_embedder_agree(module):
    cfg = build(module=module)
    backbone, embedder = cfg.module.backbone, cfg.module.embedder
    assert embedder.emb_dim == backbone.emb_dim
    # use_skip concatenates the skip connection, doubling what decode() sees
    assert embedder.out_emb_dim == (2 if backbone.use_skip else 1) * backbone.emb_dim
    assert backbone.cond_dim == cfg.module.module.cond_dim
    # attention head counts have to divide the dims they act on
    heads = list(backbone.num_heads)
    dims = [backbone.emb_dim, 2 * backbone.emb_dim, 2 * backbone.emb_dim, embedder.out_emb_dim]
    for head, dim in zip(heads, dims):
        assert dim % head == 0, f"{dim} features do not split into {head} heads"
    # the attention window must divide the latent grid and its downsampled half
    _, win_lat, win_lon = list(backbone.window_size)
    lat, lon = list(backbone.tensor_size)[1:]
    assert lat % win_lat == 0 and lon % win_lon == 0
    assert (lat // 2) % win_lat == 0 and (lon // 2) % win_lon == 0


@pytest.mark.parametrize("module", MODULES)
def test_every_preset_lands_on_the_latent_depth_geoarches_hardcodes(module):
    """geoarches writes "8 latent levels" out as a literal in three places.

    `LinVert`, the axial attention's positional embedding and the final reshape
    in `ArchesWeatherCondBackbone.forward` all assume it. The depth presets exist
    to satisfy it, which is what lets the first two -- the backbone's only mixing
    between depth levels -- stay switched on.
    """
    cfg = build(module=module)
    assert cfg.module.backbone.tensor_size[0] == GEOARCHES_Z_DIM
    assert cfg.module.n_depths in usable_depth_counts(cfg.module.embedder.patch_size)
    assert cfg.module.backbone.first_interaction_layer == "linear"
    assert cfg.module.backbone.axis_attn is True


@pytest.mark.parametrize("module", MODULES)
def test_the_vertical_is_not_a_scaling_axis(module):
    """All four presets carry the same water column; only emb_dim and depth change."""
    cfg = build(module=module)
    assert cfg.module.n_depths == 13
    assert list(cfg.module.depth_indices) == DEPTH_PRESETS["base"]


# ---------------------------------------------------------------------------
# The dataloader <-> component contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dataloader", DATALOADERS)
def test_channel_counts_match_the_component(dataloader):
    cfg = build(dataloader=dataloader)
    component = COMPONENTS[cfg.dataloader.component]
    assert cfg.dataloader.n_surface_in == component.n_surface_in
    assert cfg.dataloader.n_surface_out == component.n_surface_out
    assert cfg.dataloader.n_level_in == component.n_level_in
    assert cfg.dataloader.n_level_out == component.n_level_out


@pytest.mark.parametrize("dataloader", DATALOADERS)
def test_variable_order_is_prognostic_then_forcing(dataloader):
    """The model's output is the *leading* channels of its input; order is the contract."""
    cfg = build(dataloader=dataloader)
    component = COMPONENTS[cfg.dataloader.component]
    assert list(cfg.dataloader.dataset.variables.surface) == component.input_surface
    assert list(cfg.dataloader.dataset.variables.level) == component.input_level
    n_out = cfg.dataloader.n_surface_out
    assert list(cfg.dataloader.dataset.variables.surface)[:n_out] == component.prognostic_surface


@pytest.mark.parametrize("module", MODULES)
def test_dataloader_depth_selection_follows_the_preset(module):
    cfg = build(module=module, dataloader="glorys")
    assert list(cfg.dataloader.dataset.depth_indices) == list(cfg.module.depth_indices)


def test_tiny_dataloader_uses_the_tiny_splits():
    cfg = build(module="tiny", dataloader="glorys_tiny")
    assert cfg.dataloader.dataset.domain == "tiny_train"
    assert cfg.dataloader.validation_args.domain == "tiny_val"


# ---------------------------------------------------------------------------
# Instantiation: the shapes have to line up for real, not just on paper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module, dataloader", ALL_COMBINATIONS)
def test_embedder_and_backbone_instantiate_with_consistent_shapes(module, dataloader):
    cfg = build(module=module, dataloader=dataloader)
    embedder = instantiate(cfg.module.embedder)
    backbone = instantiate(cfg.module.backbone)

    tensor_size = tuple(cfg.module.backbone.tensor_size)
    assert (backbone.zdim, *backbone.layer1_shape) == tensor_size
    assert embedder.z_dim == tensor_size[0]
    assert embedder.latent_grid == tensor_size[1:]

    # encode: a state shaped like the dataloader's output -> tokens the backbone wants
    n_depths = cfg.module.n_depths
    state = TensorDict(
        {
            "surface": torch.zeros(1, cfg.dataloader.n_surface_in, 1, N_LAT, N_LON),
            "level": torch.zeros(1, cfg.dataloader.n_level_in, n_depths, N_LAT, N_LON),
        },
        batch_size=1,
    )
    with torch.no_grad():
        tokens = embedder.encode(state, state)
    assert tokens.shape == (1, cfg.module.backbone.emb_dim, *tensor_size)

    # decode: stand in for the backbone (a full forward at emb_dim=384 on CPU is
    # minutes, and tests/test_embedder.py already runs the real thing).
    out_dim = cfg.module.embedder.out_emb_dim
    fake = torch.zeros(1, out_dim * tensor_size[0] // 8, 8, *tensor_size[1:])
    with torch.no_grad():
        out = embedder.decode(fake)
    assert out["surface"].shape == (1, cfg.dataloader.n_surface_out, 1, N_LAT, N_LON)
    if cfg.dataloader.n_level_out:
        assert out["level"].shape == (1, cfg.dataloader.n_level_out, n_depths, N_LAT, N_LON)
    else:
        assert "level" not in out.keys()


def test_the_lightning_module_instantiates():
    """Smoke test on one combination: the metrics and the module config are wired up.

    Only the wiring is checked here; tests/test_module.py runs real forward passes,
    losses and rollouts against the same composed configs.
    """
    cfg = build(module="tiny", dataloader="glorys_tiny")
    pl_module = instantiate(cfg.module.module, cfg.module)
    assert pl_module.embedder.z_dim == cfg.module.backbone.tensor_size[0]
    assert pl_module.backbone.zdim == cfg.module.backbone.tensor_size[0]


# ---------------------------------------------------------------------------
# Vertical mixing: does the water column actually talk to itself?
#
# This is the regression that cost a review round, and it is not catchable by
# reading YAML. `first_interaction_layer: linear` and `axis_attn: True` are only
# usable because every depth preset lands on the latent depth of 8 that geoarches
# hardcodes; if a config edit switches them off, or a geoarches upgrade turns
# `LinVert` into a no-op, the model still trains and still produces plausible
# output -- it just has no path between depth levels at all, because
# `window_size[0] == 1` and the down/up-sampling stages act on lat/lon alone.
#
# So we measure it instead: perturb the shallowest input depth only, and check
# that the deep output levels move.
# ---------------------------------------------------------------------------
#: Shrinks a preset to something a CPU can run in a second, without touching the
#: two settings under test.  ``emb_dim`` (token width) and ``depth_multiplier``
#: (block count) are the only two axes the four presets differ on, and neither
#: has anything to do with whether the water column is mixed -- that is decided
#: by ``first_interaction_layer``, ``axis_attn`` and the latent depth, all of
#: which are left exactly as the preset ships them.
SMALL_MODEL = [
    "++module.backbone.emb_dim=32",
    "++module.backbone.num_heads=[2,4,4,2]",
    "++module.backbone.depth_multiplier=1",
    "++module.embedder.emb_dim=32",
    "++module.embedder.out_emb_dim=64",
]
MIXING_OFF = [
    "++module.backbone.first_interaction_layer=null",
    "++module.backbone.axis_attn=false",
]


def depth_coupling(
    cropped_masks: tuple[Path, int, int], module: str = "tiny", extra: list[str] = ()
) -> float:
    """Largest change at output depths 4+ when only input depth 0 is perturbed.

    Input depth 0 lands in latent token 0, which the decoder unpacks back into
    output depths 0 and 1. Output depths 4 and below come from latent tokens 2
    and up. So without some layer that mixes along the latent depth axis, this
    number is *exactly* zero -- there is no numerical path at all, not merely a
    small one.

    The model and backbone are built from a composed preset config, so whatever
    the YAML says about vertical mixing is what gets measured.

    Args:
        cropped_masks: The ``cropped_masks`` fixture -- the real masks on a
            72x120 corner of the globe, and its grid.  This probe runs the
            *real* backbone, so it cannot be stubbed; shrinking the map instead
            takes it from ~5 s to ~1.5 s per preset.  Only latitude and
            longitude shrink.  The latent depth, which is the axis under test,
            stays at the 8 geoarches hardcodes, and 72x120 still divides the
            patch size and gives a latent grid (24x40, halved to 12x20) that
            the ``[1, 6, 10]`` attention window tiles exactly at both stages.
        module: Which shipped preset to compose.
        extra: Further hydra overrides, e.g. :data:`MIXING_OFF`.
    """
    masks_file, lat, lon = cropped_masks
    grid = [
        f"++module.embedder.masks_path={masks_file}",
        f"++module.embedder.img_size=[{lat},{lon}]",
        f"++module.backbone.tensor_size=[{GEOARCHES_Z_DIM},{lat // 3},{lon // 3}]",
    ]
    cfg = build(module=module, dataloader="glorys", extra=[*SMALL_MODEL, *grid, *extra])
    torch.manual_seed(0)
    embedder = instantiate(cfg.module.embedder).eval()
    backbone = instantiate(cfg.module.backbone).eval()

    # geoarches zero-initialises every adaLN gate, which makes each attention
    # block -- including the axial attention -- exactly the identity on an
    # untrained model. Left closed, this probe would only ever see `LinVert`
    # (whose residual branch is ungated) and would pass a config that had
    # quietly turned `axis_attn` off. Opening them puts the model in the regime
    # it spends training in.
    for submodule in backbone.modules():
        gates = getattr(submodule, "adaLN_modulation", None)
        if gates is not None:
            nn.init.constant_(gates[-1].bias, 0.1)

    torch.manual_seed(1)
    state = TensorDict(
        {
            "surface": torch.randn(1, cfg.dataloader.n_surface_in, 1, lat, lon),
            "level": torch.randn(1, cfg.dataloader.n_level_in, cfg.module.n_depths, lat, lon),
        },
        batch_size=1,
    )
    cond = torch.zeros(1, cfg.module.backbone.cond_dim)

    def forward(sample):
        with torch.no_grad():
            return embedder.decode(backbone(embedder.encode(sample, sample), cond))

    reference = forward(state)
    perturbed = state.clone()
    perturbed["level"][:, :, 0] += 1.0
    difference = (forward(perturbed) - reference)["level"]

    assert difference[:, :, :2].abs().max() > 1e-6, (
        "perturbing input depth 0 did not even change output depths 0-1; the probe "
        "is broken, not the model"
    )
    return float(difference[:, :, 4:].abs().max())


@pytest.mark.parametrize("module", MODULES)
def test_the_top_of_the_water_column_reaches_the_bottom(module, cropped_masks):
    """As shipped, every preset propagates a surface-level change to depth."""
    assert depth_coupling(cropped_masks, module) > 1e-6


def test_the_probe_bites_with_both_mixing_layers_off(cropped_masks):
    """Without them there is no path between depths at all -- not a small one, none."""
    assert depth_coupling(cropped_masks, "tiny", MIXING_OFF) == 0.0


@pytest.mark.parametrize(
    "disabled",
    [
        pytest.param(["++module.backbone.axis_attn=false"], id="LinVert alone"),
        pytest.param(
            ["++module.backbone.first_interaction_layer=null"], id="axial attention alone"
        ),
    ],
)
def test_each_mixing_layer_carries_the_signal_on_its_own(disabled, cropped_masks):
    """Turn one off and the other must still couple the column.

    Together with the test above this pins down what the probe can see: it fails
    if *either* layer stops working, not just if both do. That is what makes it a
    guard against a geoarches upgrade quietly no-opping one of them, rather than
    only against someone editing the YAML.
    """
    assert depth_coupling(cropped_masks, "tiny", disabled) > 1e-6


# ---------------------------------------------------------------------------
# One measurement of `make setup`, quoted the same everywhere
# ---------------------------------------------------------------------------
#: Measured with uv's wheel cache already
#: populated: 20 s to build a brand-new `.venv`, 8 s to re-check an existing one;
#: a participant timed 28 s.  The cold number is the ~2 GB PyTorch download.
SETUP_TIMING = "~30 s warm, ~5 min cold"
SETUP_TIMING_FILES = (
    "README.md",
    "Makefile",
    "scripts/setup_env.sh",
    "docs/00_start_here.md",
    "docs/01_setup.md",
    "docs/cheatsheet.md",
)


@pytest.mark.parametrize("relative", SETUP_TIMING_FILES)
def test_every_document_quotes_the_same_setup_timing(relative):
    """Three documents quoted three different numbers for one command.

    Measured: README said ~48 s, the
    Makefile's pre-setup error said about 5 minutes, and the run took 28 s. A
    reader cannot tell which of those to plan around, and the one they meet
    first -- the error message -- was the most wrong.

    MUTANT: putting `~48 s` back in README.md fails this.
    """
    text = (REPO_ROOT / relative).read_text()
    assert SETUP_TIMING in text, f"{relative} quotes no timing for `make setup`"
    for stale in ("48 s", "48s", "about 5 minutes", "~5 min first time"):
        assert stale not in text, f"{relative} still carries the old claim {stale!r}"


def test_the_fresh_checkout_error_quotes_that_timing_too(tmp_path):
    """It is the first timing anybody meets: it prints before `make setup` exists."""
    make = shutil.which("make")
    if make is None:
        pytest.fail("`make` is not on PATH")
    shutil.copy(REPO_ROOT / "Makefile", tmp_path / "Makefile")
    result = subprocess.run(
        [make, "--no-print-directory", "doctor"], capture_output=True, text=True, cwd=tmp_path
    )
    assert SETUP_TIMING in result.stdout + result.stderr


#: What each preset was measured to fit on a dc-gpu A100 (39.5 GiB usable), by
#: `make benchmark`.
JURECA_BATCH = {"tiny": 4, "small": 2, "base": 1}


@pytest.mark.parametrize("module", sorted(JURECA_BATCH))
@pytest.mark.parametrize("cluster", ["jureca_1gpu", "jureca_4gpu", "jureca_4nodes"])
def test_the_clusters_use_the_batch_measured_on_this_card(module, cluster):
    """A batch that does not fit puts `make train-tiny` -- step 5 of the README
    quickstart -- into an out-of-memory error on its first step.

    MUTANT: pinning one literal value in a cluster config fails this for all but
    one preset.
    """
    cfg = build(module=module, dataloader="glorys", cluster=cluster)
    assert cfg.batch_size == JURECA_BATCH[module], (
        f"{cluster} composes batch_size {cfg.batch_size} for {module}, "
        f"but {JURECA_BATCH[module]} is what fits the card"
    )


@pytest.mark.parametrize("module", sorted(JURECA_BATCH))
def test_every_preset_declares_a_batch_size(module):
    """A preset missing `batch_size` makes every cluster fail to compose with an
    interpolation error rather than a readable message."""
    cfg = build(module=module, dataloader="glorys", cluster="jureca_1gpu")
    assert "batch_size" in cfg.module
    assert cfg.module.batch_size >= 1
