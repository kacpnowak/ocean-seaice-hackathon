# ---------------------------------------------------------------------------
# OceanArches -- one-stop entry point.
#
#   make help          list every target
#   make setup         create .venv and install everything
#   make doctor        check that the environment and the data are usable
#
# Every target runs inside the project .venv, so you do NOT have to activate it.
# ---------------------------------------------------------------------------
SHELL := /bin/bash
.DEFAULT_GOAL := help

PY := .venv/bin/python

# A fresh clone has no .venv, and every target below runs $(PY). Without this
# check `make doctor` -- the first command docs/01 tells a participant to run --
# fails with a bare `.venv/bin/python: No such file or directory` and `Error
# 127`, which says nothing about what to do. Checked at parse time rather than as
# a prerequisite so that no recipe changes and `make -n train-tiny` still prints
# exactly the training command (tests/test_configs.py reads it).
# `test -x`, not `$(wildcard ...)`: wildcard matches a DANGLING symlink, so a
# .venv whose interpreter has been removed or whose uv-managed python moved
# would sail past this check and reproduce the raw Error 127 it exists to
# prevent. And `setup` anywhere in the goals disables the guard entirely, so the
# obvious `make setup doctor` one-liner works -- .venv exists by the time
# `doctor` runs, but make evaluates this at parse time and cannot know that.
ifeq (,$(shell test -x .venv/bin/python && echo yes))
ifeq (,$(filter setup,$(MAKECMDGOALS)))
ifneq (,$(filter-out help clean,$(or $(MAKECMDGOALS),help)))
$(info )
$(info There is no virtual environment here yet: .venv/bin/python does not exist.)
$(info This is a fresh checkout. Build it with:)
$(info )
$(info   $$ make setup      # ~30 s warm, ~5 min cold -- the 2 GB PyTorch download)
$(info )
$(error nothing to run until then)
endif
endif
endif
# Absolute, and handed to hydra as --config-path (NOT --config-dir): see the
# note at the top of configs/config.yaml. --config-dir is searched *after*
# geoarches' own configs/, so geoarches' config.yaml wins and our max_steps,
# save_step_frequency and limit_val_batches are silently ignored.
CONFIG_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))configs)

# Overridable on the command line, e.g.  make train-tiny NAME=my_run
NAME ?= tiny
MODULE ?= tiny
DATALOADER ?= glorys
# Extra hydra overrides for the two train targets.  They cannot be passed as bare
# words: make parses anything containing `=` on its command line as a VARIABLE
# ASSIGNMENT, so `make train-tiny ++max_steps=10` sets a make variable called
# `++max_steps` and the run trains for the preset's full budget with no warning.
# Quote them into this one variable instead:
#   make train-tiny NAME=my_run HYDRA_ARGS="++max_steps=1000 ++module.module.lr=1e-4"
HYDRA_ARGS ?=
# configs/cluster/*.yaml: batch size, precision and dataloader workers.
# Use CLUSTER=local on a machine without a GPU, jureca_4gpu for a whole node.
# There is a jureca_4nodes too, but `make train` cannot use it: more than one
# node needs `python -m oceanarches.main_multinode` rather than
# `geoarches.main_hydra`, which builds its Trainer without `num_nodes`. That is
# what scripts/pretrain_base.slurm runs.
CLUSTER ?= jureca_1gpu
YEARS ?=
LEAD_DAYS ?= 10
EVAL_ARGS ?=
# Extra flags for `make benchmark`, for the same make-eats-`=` reason as
# HYDRA_ARGS.  A preset that does not fit this cluster's card is measured with
#   make benchmark BENCH_ARGS="--presets small --batch-size 2"
BENCH_ARGS ?=
# `make couple`: the two trained components, and how they are stepped.
OCEAN ?= ocean_tiny
SEAICE ?= seaice_tiny
MODE ?= sequential

# --- goals that are really hydra overrides ---------------------------------
# Measured on a trial participant: `make train-tiny NAME=x ++max_steps=10` -- the
# first thing anyone types to shorten a run -- launched the full 4000-step,
# 31-minute default.  make parses every command-line word containing `=` as a
# VARIABLE ASSIGNMENT, `++max_steps` is a legal variable name, nothing reads it,
# and nothing warns.  The same trap swallows `module=base`,
# `++module.module.lr=1e-4` and the lower-case `name=my_run`, which leaves NAME
# at its default and writes into the default run directory instead of yours.
#
# `$(.VARIABLES)` + `$(origin ...)` is the only way to see these: a command-line
# assignment never reaches MAKECMDGOALS.  Checked at parse time, like the .venv
# guard above, so no recipe changes and `make -n train-tiny` still prints exactly
# the training command (tests/test_configs.py reads it).
#
# The test is the SHAPE of the name, not an allowlist: everything this Makefile
# reads is upper case (NAME, HYDRA_ARGS, LEAD_DAYS, ...), and everything hydra
# takes is lower case or carries a `+` or a `.`.  That distinction matters --
# `make train-tiny CUDA_VISIBLE_DEVICES=0` really does reach the recipe's
# environment and must keep working, while `make train-tiny mode=test` reaches
# nothing at all.
_UPPER := A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
_has_upper = $(strip $(foreach c,$(_UPPER),$(if $(findstring $(c),$(1)),x)))
_override_shaped = $(if $(findstring +,$(1))$(findstring .,$(1)),$(1),$(if $(call _has_upper,$(1)),,$(1)))
OVERRIDES_EATEN_BY_MAKE := $(strip $(foreach v,$(.VARIABLES),\
	$(if $(filter command line,$(origin $(v))),$(call _override_shaped,$(v)))))
ifneq (,$(OVERRIDES_EATEN_BY_MAKE))
$(info )
$(info make parsed these as assignments to make variables, so nothing will read them:)
$(info )
$(info >>  $(foreach v,$(OVERRIDES_EATEN_BY_MAKE),$(v)=$($(v))))
$(info )
$(info A bare hydra override on a make command line is silently dropped and the run)
$(info trains the preset's full budget -- 4000 steps, about 31 minutes for `tiny`.)
$(info Quote them into HYDRA_ARGS instead, and run_eval flags into EVAL_ARGS:)
$(info )
$(info >>  make train-tiny NAME=my_run HYDRA_ARGS="++max_steps=200 ++save_step_frequency=50")
$(info >>  make eval NAME=my_run EVAL_ARGS="--skip-animations")
$(info )
$(info The variables this Makefile reads, all upper case: NAME MODULE DATALOADER)
$(info CLUSTER HYDRA_ARGS YEARS LEAD_DAYS EVAL_ARGS OCEAN SEAICE MODE.)
$(info )
$(error refusing to run: that command line does not do what it says)
endif

# config.env is NOT included here. It is written for bash (`KEY="value"`), and
# `include config.env` + `export` would put the quotes into the environment
# verbatim: GLORYS_PREPPED came out as `""/path/to/data"/glorys_1deg_prepped"`
# and every target that touches the data failed with "Path does not exist".
# Nothing needs it exported anyway -- oceanarches/paths.py parses config.env
# itself, and its resolution order is environment > config.env > built-in
# default, so `GLORYS_PREPPED=/somewhere make train` still overrides it.
# scripts/*.slurm source config.env with bash, where the quotes are correct.

# ---------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help
	@echo "OceanArches targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- environment -----------------------------------------------------------
.PHONY: setup
setup:  ## Create .venv and install oceanarches + geoarches (~30 s warm, ~5 min cold)
	bash scripts/setup_env.sh

.PHONY: doctor
doctor:  ## Check python, GPU, geoarches, data paths and generated stats
	$(PY) -m oceanarches.doctor

# --- data ------------------------------------------------------------------
.PHONY: prep-data
prep-data:  ## Prepare GLORYS yearly files locally (use YEARS="2015 2016" for a subset)
	$(PY) scripts/prepare_glorys.py --years $(YEARS)

.PHONY: prep-data-slurm
prep-data-slurm:  ## Submit the full 1993-2025 preparation as a SLURM array job
	sbatch scripts/prepare_glorys.slurm

.PHONY: stats
stats:  ## Compute masks, normalisation stats and climatology
	$(PY) scripts/compute_stats.py

.PHONY: stats-quick
stats-quick:  ## Same as `stats` but from a handful of dates (~1 min)
	$(PY) scripts/compute_stats.py --quick

# Only needed for `forcing=file`. Separate from `stats` because it reads the
# optional IFS archive rather than the prepared GLORYS files, and because a
# clone that never runs a forced model never needs it.
.PHONY: forcing-stats
forcing-stats:  ## Compute normalisation stats for the prescribed atmosphere (~10 s)
	$(PY) scripts/compute_forcing_stats.py

# --- training --------------------------------------------------------------
.PHONY: train-tiny
train-tiny:  ## Train the 30-minute model on one GPU (add HYDRA_ARGS="++max_steps=1000")
	$(PY) -m geoarches.main_hydra --config-path $(CONFIG_DIR) \
		cluster=$(CLUSTER) module=tiny dataloader=glorys_tiny ++name=$(NAME) $(HYDRA_ARGS)

.PHONY: train
train:  ## Train any preset:  make train MODULE=base NAME=my_base_run
	$(PY) -m geoarches.main_hydra --config-path $(CONFIG_DIR) \
		cluster=$(CLUSTER) module=$(MODULE) dataloader=$(DATALOADER) ++name=$(NAME) $(HYDRA_ARGS)

.PHONY: benchmark
benchmark:  ## Measure params + step time + projected wall clock for every preset
	$(PY) scripts/benchmark_step.py $(BENCH_ARGS)

# --- evaluation ------------------------------------------------------------
# EVAL_ARGS takes the rest of run_eval's flags, for the same reason HYDRA_ARGS
# exists: make eats a bare `--domain=x` as a variable assignment.
#   make eval NAME=my_run EVAL_ARGS="--domain ifs_forced_val --skip-free-rollout"
.PHONY: eval
eval:  ## Roll out, score against persistence/climatology, and render figures + animations
	$(PY) -m oceanarches.evaluation.run_eval --exp $(NAME) --lead-days $(LEAD_DAYS) $(EVAL_ARGS)

# --- coupling --------------------------------------------------------------
# Train the two halves first (about 16 minutes each on one GPU):
#   make train MODULE=ocean_component  DATALOADER=glorys_ocean  NAME=ocean_tiny
#   make train MODULE=seaice_component DATALOADER=glorys_seaice NAME=seaice_tiny
# then run them together. MODE=parallel runs both off the state at time t;
# sequential (the default) lets the sea ice see the ocean this step already
# produced. The output directory is named after the components and the mode.
.PHONY: couple
couple:  ## Run a coupled ocean + sea-ice rollout from two trained components
	$(PY) -m oceanarches.evaluation.run_eval --coupled \
		--components ocean=$(OCEAN) seaice=$(SEAICE) \
		--mode $(MODE) --lead-days $(LEAD_DAYS)

# --- development -----------------------------------------------------------
.PHONY: test
test:  ## Run the (fast, CPU-only) test suite
	$(PY) -m pytest -q

# `ruff check` covers the notebooks too -- they are the first example code a
# participant reads, and unused imports in notebook 03 cell 1 went unnoticed for
# exactly as long as they were outside this target. `ruff format` does NOT: it
# would reflow every plotting cell in the four reviewed notebooks.
.PHONY: lint
lint:  ## Check formatting and lint
	.venv/bin/ruff check oceanarches scripts tests notebooks
	.venv/bin/ruff format --check oceanarches scripts tests

.PHONY: format
format:  ## Auto-format the code
	.venv/bin/ruff format oceanarches scripts tests
	.venv/bin/ruff check --fix oceanarches scripts tests notebooks

.PHONY: clean
clean:  ## Remove caches (keeps .venv, data, checkpoints)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
