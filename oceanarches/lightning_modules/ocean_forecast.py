"""The forecast module: what geoarches trains, taught about the ocean.

Read ``.venv/lib/python3.12/site-packages/geoarches/lightning_modules/forecast.py``
alongside this file.  Almost everything there -- the optimiser, the cosine
schedule, the conditioning on month and hour, the gradient-checkpointed rollout,
the checkpointing hooks -- is reused unchanged.  What is *not* reusable is
everything that encodes "the data is ERA5":

======================================  =====================================
geoarches' ``ForecastModuleWithCond``   ``OceanForecastModule``
======================================  =====================================
``compute_lat_weights(121)``            ``compute_lat_weights_glorys(180)``
13 pressure levels, weight ~ pressure   13 depth levels, weight 1 each
4 named ERA5 surface variables          the component's own variables
``pangu_norm_stats2_with_w.pt``         ``oceanarches/stats/glorys_1deg_stats.pt``
no land                                 30.4-42.5% land, and it must not be scored
one model predicts all it reads         a component predicts a *subset*
======================================  =====================================

The loss, in one line
---------------------
For every variable ``v`` the model predicts::

    loss = sum_v  loss_weight[v] * mean_depths( mean_ocean( (err / delta_std)^2 ) )
           ---------------------------------------------------------------------
                                   sum_v loss_weight[v]

* ``mean_ocean`` is a latitude-weighted average over *ocean cells only*, using
  the per-depth wet mask -- so land, which the dataloader fills with a constant,
  contributes nothing at all.  This is the single most important line in the
  file: 30.4% of the surface grid and 42.5% of the deepest prepared level (1684 m)
  is land, and a model scored over land is mostly scored on how well it
  reproduces a constant.
* ``err / delta_std`` is the tendency normalisation.  The model works in
  normalised units, so the error is multiplied by ``data_std / delta_std`` --
  that is exactly what geoarches' ``loss_delta_normalization`` flag does, and we
  keep its name.  Without it, deep salinity (huge spatial spread, tiny daily
  change) would be measured against how much salinity varies between the
  Baltic and the Red Sea instead of against how much it changes in a day.
* Dividing by ``sum_v loss_weight[v]`` makes the number readable: it puts a
  one-day persistence forecast *near* 1, because ``delta_std`` is by definition
  the spread of the one-day change.

  Near, not exactly.  An error of exactly one ``delta_std`` in every cell scores
  exactly 1.0 -- that is what
  ``tests/test_module.py::test_a_persistence_sized_error_scores_one`` pins -- but
  a real persistence forecast comes out **lower**, because ``delta_std`` is one
  number per (variable, depth) taken unweighted over every ocean cell and every
  date in the archive, while this loss is a latitude-weighted mean normalised by
  ocean *area*.  The two denominators do not cancel cell by cell -- measured, the
  same samples score 0.906 with uniform spatial weights and 0.814 with
  ``cos(lat)``, because the one-day tendency is largest exactly where the cosine
  weighting is smallest.  Measured through this method, on whole splits in
  dataset order (Task 6 report, "Fix round 1", has the commands)::

      0.814   the first 128 samples of tiny_val (2019)
      0.872   the whole tiny_val year            (364 samples)
      0.844   the whole tiny_train split         (2014-2018, 1825 samples)
      0.820   the whole train split              (1993-2018, 9488 samples)
      0.860   the whole test split               (2021-2023, 1094 samples)

  So **the line to beat is ~0.82-0.89, not 1.0**, and it moves with the split --
  quote the split whenever you quote the number.  For scale, on the same 128
  samples two-day persistence scores 2.08 and predicting the climatological mean
  everywhere scores 6469, so the useful range of this loss is narrow.  The
  shipped ``tiny`` checkpoint scores 0.7095 against persistence's 0.8141 on those
  128 samples, i.e. 13% better -- a real margin for 34 minutes on one GPU, and
  not a large one.

Components
----------
A *component* (see ``ComponentSpec`` in ``dataloaders/variables.py``) predicts a
subset of what it reads.  The module therefore takes a full state in and returns
only the component's prognostic variables, and every step that compares the two
-- the loss, the metrics, the residual connection, the rollout -- slices the
target down to the leading ``n_out`` channels first.  That works because the
dataloader is required to order channels prognostic-first, which
``tests/test_configs.py`` checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint as gradient_checkpoint
from geoarches.lightning_modules.forecast import ForecastModuleWithCond
from tensordict.tensordict import TensorDict

from .. import paths
from ..dataloaders import variables as V
from ..dataloaders.forcing import ForcingSource
from ..dataloaders.masks import Masks, load_masks, state_mask
from ..metrics.masked_metrics import compute_lat_weights_glorys, ocean_area_weights

__all__ = ["OceanForecastModule", "compute_lat_weights_glorys", "plain_betas"]


def _select_statistics(
    stats: dict, kind: str, names: Sequence[str], depth_indices: Sequence[int] | None
) -> dict[str, torch.Tensor]:
    """Pull ``mean``, ``std`` and ``delta_std`` for ``names`` out of the stats file.

    The prepared statistics carry all 14 depth levels; the model uses 13.  The
    subset taken here **must** be the same one the dataloader took, or the model
    would be trained against a 15 m mean at 5 m and nothing would look obviously
    wrong.  ``GlorysForecast._load_stats`` does the identical selection on the
    data side; ``tests/test_module.py`` asserts the two agree element by element.
    """
    available = list(stats[f"{kind}_variables"])
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(
            f"The statistics have no entry for {missing}; they know about {available}. "
            "Re-run: make stats"
        )
    if kind == "level" and depth_indices is not None and len(depth_indices) == 0:
        # `[]` is not "every level", it is "no levels", and the difference is
        # invisible downstream: the statistics come back with a zero-length depth
        # axis and the positivity check below, `(std > 0).all()`, is vacuously
        # True on an empty tensor. Say so here instead.
        raise ValueError(
            "depth_indices is an empty sequence, which would select no depth levels at "
            "all and give statistics with a zero-length depth axis. Pass None to mean "
            "'every prepared level'."
        )
    rows = [available.index(name) for name in names]
    selected = {}
    for key in ("mean", "std", "delta_std"):
        tensor = stats[f"{kind}_{key}"][rows].float()
        if kind == "level" and depth_indices is not None:
            tensor = tensor[:, list(depth_indices)]
        selected[key] = tensor
    if not bool((selected["std"] > 0).all()) or not bool((selected["delta_std"] > 0).all()):
        raise ValueError(
            f"Non-positive {kind} standard deviation in the statistics; the loss would "
            "divide by zero. Re-run: make stats"
        )
    return selected


def plain_betas(kwargs: dict) -> dict:
    """Return ``kwargs`` with ``betas`` as a plain tuple of floats.

    ``betas: [0.9, 0.98]`` in ``configs/module/*.yaml`` composes to an OmegaConf
    ``ListConfig``.  hydra hands it here, geoarches' ``configure_optimizers``
    passes it to ``torch.optim.AdamW``, and AdamW stores it verbatim in
    ``param_groups`` -- so it ends up inside ``optimizer_states`` in every
    checkpoint.  Since torch 2.6 that made the checkpoint unreadable by
    ``torch.load(weights_only=True)``, which is the default Lightning's resume
    path uses: **every run in this kit was unresumable** and nothing noticed,
    because fresh runs load nothing and both ``load_ckpt`` and ``load_module``
    pass ``weights_only=False``.

    Casting here is the narrow fix -- checkpoints written from now on contain no
    OmegaConf at all.  ``oceanarches.lightning_modules.checkpoints`` covers the
    two cases a cast cannot: checkpoints already on disk, and geoarches
    reinstating ``save_hyperparameters()``.

    A no-op when ``betas`` is absent or already a plain sequence.
    """
    if "betas" not in kwargs or kwargs["betas"] is None:
        return kwargs
    return {**kwargs, "betas": tuple(float(beta) for beta in kwargs["betas"])}


class OceanForecastModule(ForecastModuleWithCond):
    """Deterministic GLORYS forecast module.

    Args:
        cfg: the ``configs/module/*.yaml`` node; geoarches instantiates the
            backbone, the embedder and the metrics from it.
        name: run name, used for the checkpoint and evaluation directories.
        component: which :data:`oceanarches.dataloaders.variables.COMPONENTS`
            entry this model is.  ``full`` predicts everything it reads.
        depth_indices: which of the 14 prepared depth levels the dataloader
            hands over.  Defaults to ``cfg.depth_indices``.
        add_input_state: predict the *change* and add the input state, rather
            than predicting the state directly.  Default True and it matters:
            day-to-day SST changes by ~0.1 K against a spatial spread of ~10 K,
            so a model that has to reproduce the absolute field spends all its
            capacity on geography instead of on dynamics.
        clamp_physical_bounds: clamp ``siconc`` to [0, 1] and ``sithick`` to
            >= 0 (bounds come from ``variables.py``).  Applied in *normalised*
            space by converting the bound, not in physical space -- see
            :meth:`clamp_to_physical_bounds`.
        loss_delta_normalization: geoarches' flag, same meaning: weight the loss
            by ``(data_std / delta_std) ** pow`` so that each variable's one-day
            tendency, not its absolute value, has comparable magnitude.
        multistep_curriculum: let geoarches lengthen the training rollout as the
            run proceeds (``2 + epoch // increase_multistep_period``) instead of
            holding it at ``module.train.rollout_iterations``.  Default False --
            see :meth:`on_train_epoch_start`.
        lead_time_hours: hours between rollout steps.  The rollout advances the
            timestamp by this much per step.
        forcing: optional :class:`~oceanarches.dataloaders.forcing.ForcingSource`
            for prescribed external fields.  ``None`` (no forcing) is the
            default and the best-tested path.
        stats_path, masks_path: overrides, for tests.
        kwargs: passed to geoarches (``lr``, ``betas``, ``weight_decay``,
            ``num_warmup_steps``, ``num_training_steps``, ``cond_dim``,
            ``use_prev``, ``pow``, ...).
    """

    def __init__(
        self,
        cfg,
        name: str = "ocean_forecast",
        component: str = "full",
        depth_indices: Sequence[int] | None = None,
        add_input_state: bool = True,
        clamp_physical_bounds: bool = True,
        loss_delta_normalization: bool = True,
        multistep_curriculum: bool = False,
        lead_time_hours: int = 24,
        forcing: ForcingSource | None = None,
        stats_path: str | Path | None = None,
        masks_path: str | Path | None = None,
        **kwargs,
    ):
        # Two of geoarches' flags are switched off for the *parent's* constructor
        # and restored immediately after:
        #   loss_delta_normalization=False stops it loading ERA5's
        #     `pangu_norm_stats2_with_w.pt`, whose 6 level and 4 surface channels
        #     have nothing to do with our state;
        #   add_input_state=False stops `ForecastModule.forward` adding the whole
        #     input state to an output that only has the component's channels.
        # Everything else the parent does -- backbone, embedder, metrics,
        # optimiser, schedule -- is exactly what we want.  It still builds one set
        # of ERA5 loss coefficients from `compute_lat_weights_weatherbench(121)`
        # and throws them away; that is a few kilobytes at start-up and the only
        # alternative is to fork its `__init__`.
        super().__init__(
            cfg,
            name=name,
            add_input_state=False,
            loss_delta_normalization=False,
            lead_time_hours=lead_time_hours,
            **plain_betas(kwargs),
        )
        del self.loss_coeffs  # the ERA5 ones; ours are registered buffers below

        self.component = V.get_component(component)
        self.add_input_state = bool(add_input_state)
        self.clamp_physical_bounds = bool(clamp_physical_bounds)
        self.loss_delta_normalization = bool(loss_delta_normalization)
        self.multistep_curriculum = bool(multistep_curriculum)
        self.forcing_source = forcing

        if depth_indices is None and cfg is not None and "depth_indices" in cfg:
            depth_indices = cfg.depth_indices
        self.depth_indices = list(depth_indices) if depth_indices is not None else None

        self.surface_variables = self.component.prognostic_surface
        self.level_variables = self.component.prognostic_level
        self.n_surface_in = self.component.n_surface_in
        self.n_surface_out = self.component.n_surface_out
        self.n_level_in = self.component.n_level_in
        self.n_level_out = self.component.n_level_out

        masks = load_masks(path=masks_path, depth_indices=self.depth_indices)
        self._check_backbone_widths()
        self._check_embedder(masks)
        self._register_masks(masks)
        self._register_statistics(stats_path)
        self._register_loss_coefficients(masks)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _check_embedder(self, masks: Masks) -> None:
        """Fail loudly if the component, the dataloader and the embedder disagree.

        Every one of these mismatches is silent otherwise: wrong channel counts
        train a model on the wrong variables, and a wrong depth count lines the
        statistics up against the wrong levels.  Both produce a model that looks
        like it works.
        """
        embedder = self.embedder
        expected = (
            self.n_surface_in,
            self.n_surface_out,
            self.n_level_in,
            self.n_level_out,
        )
        found = (
            embedder.surface_ch_in,
            embedder.surface_ch_out,
            embedder.level_ch_in,
            embedder.level_ch_out,
        )
        if found != expected:
            raise ValueError(
                f"Component {self.component.name!r} needs surface {expected[0]} in / "
                f"{expected[1]} out and level {expected[2]} in / {expected[3]} out, but the "
                f"embedder was built for surface {found[0]}/{found[1]} and level "
                f"{found[2]}/{found[3]}. The module's `component` and the dataloader config "
                "have to be the same component."
            )
        if embedder.n_depths and embedder.n_depths != masks.n_depth:
            raise ValueError(
                f"The embedder has {embedder.n_depths} depth levels but the masks and "
                f"statistics were sliced to {masks.n_depth} (depth_indices="
                f"{self.depth_indices}). module.depth_indices and module.n_depths must agree."
            )

    def _check_backbone_widths(self) -> None:
        """Fail here if ``emb_dim`` was raised without ``num_heads`` and ``out_emb_dim``.

        docs/04 invites you to scale the model up, and the obvious way to try it
        is ``HYDRA_ARGS="++module.backbone.emb_dim=192"``.  That instantiates
        silently and dies in the first forward pass with ``RuntimeError: mat1 and
        mat2 shapes cannot be multiplied (7200x768 and 1536x1536)`` -- after the
        dataloader has spun up, on a GPU you queued for.  Three numbers move
        together, and this says which.

        ``tests/test_configs.py::test_backbone_and_embedder_agree`` pins the same
        rule for the four shipped presets; this one fires on overrides, which the
        config test cannot see.
        """
        backbone, embedder = self.backbone, self.embedder
        emb = int(backbone.emb_dim)
        expected_out = (2 if backbone.use_skip else 1) * emb
        problems = []
        if int(embedder.emb_dim) != emb:
            problems.append(
                f"embedder.emb_dim={embedder.emb_dim} but backbone.emb_dim={emb}; "
                "the encoder writes the tokens the backbone reads, so they are one number"
            )
        if int(embedder.out_emb_dim) != expected_out:
            problems.append(
                f"embedder.out_emb_dim={embedder.out_emb_dim} but the backbone hands the "
                f"decoder {expected_out} channels "
                f"({'2 * ' if backbone.use_skip else ''}emb_dim, because use_skip="
                f"{backbone.use_skip})"
            )
        heads = list(backbone.num_heads)
        dims = [emb, 2 * emb, 2 * emb, expected_out]
        for stage, (head, dim) in enumerate(zip(heads, dims)):
            if head and dim % head:
                problems.append(
                    f"backbone.num_heads[{stage}]={head} does not divide the {dim} features "
                    "that stage attends over"
                )
        if problems:
            raise ValueError(
                "The backbone and the embedder are not the same size:\n  - "
                + "\n  - ".join(problems)
                + "\nScaling this model means moving backbone.emb_dim, backbone.num_heads "
                "and embedder.emb_dim/out_emb_dim together. The shipped presets, as "
                "emb_dim / num_heads / out_emb_dim: tiny 96 / [3,6,6,3] / 192, "
                "small 192 / [6,12,12,6] / 384, base 192 / [6,12,12,6] / 384, "
                "large 384 / [12,24,24,12] / 768. See docs/04_scaling_finetuning.md."
            )

    def _register_masks(self, masks: Masks) -> None:
        """Wet masks for the model's input and output channels.

        Buffers, not attributes: they have to follow ``.to(device)``.  Not
        persistent, because they are derived from a file that ships with the repo
        and there is no reason to put 15 MB of coastline into every checkpoint.
        """
        output = state_mask(masks, self.surface_variables, self.level_variables)
        inputs = state_mask(masks, self.component.input_surface, self.component.input_level)
        self.register_buffer("mask_surface", output["surface"], persistent=False)
        self.register_buffer("input_mask_surface", inputs["surface"], persistent=False)
        if self.n_level_out:
            self.register_buffer("mask_level", output["level"], persistent=False)
        if self.n_level_in:
            self.register_buffer("input_mask_level", inputs["level"], persistent=False)

    def _register_statistics(self, stats_path: str | Path | None) -> None:
        """Normalisation statistics and the normalised physical bounds."""
        stats_path = Path(stats_path) if stats_path is not None else paths.stats_file()
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalisation statistics not found: {stats_path}\nrun: make stats"
            )
        stats = torch.load(stats_path, weights_only=True)

        self._statistics: dict[str, dict[str, torch.Tensor]] = {}
        for group, names in (("surface", self.surface_variables), ("level", self.level_variables)):
            if not names:
                continue
            selected = _select_statistics(stats, group, names, self.depth_indices)
            self._statistics[group] = selected
            self.register_buffer(f"state_mean_{group}", selected["mean"], persistent=False)
            self.register_buffer(f"state_std_{group}", selected["std"], persistent=False)
            # The clamp works on the model's output, which is in normalised
            # space; converting the bound once here is exact and costs nothing,
            # whereas denormalising the whole state to clamp it and normalising
            # it back would be three extra passes over 3.4 M values per step.
            low, high = [], []
            for index, name in enumerate(names):
                lower, upper = V.VARIABLES[name].bounds
                mean = selected["mean"][index]
                std = selected["std"][index]
                low.append(
                    torch.full_like(mean, -float("inf")) if lower is None else (lower - mean) / std
                )
                high.append(
                    torch.full_like(mean, float("inf")) if upper is None else (upper - mean) / std
                )
            self.register_buffer(f"clamp_low_{group}", torch.stack(low), persistent=False)
            self.register_buffer(f"clamp_high_{group}", torch.stack(high), persistent=False)
        # Nothing to do if no predicted variable has a bound at all.
        self._has_bounds = any(
            V.VARIABLES[name].bounds != (None, None)
            for name in self.surface_variables + self.level_variables
        )

    def _register_loss_coefficients(self, masks: Masks) -> None:
        """Build ``loss_coeffs`` from our grid, our masks and our statistics.

        The parent's ``loss()`` reduces with ``TensorDict.mean()``, i.e. it
        divides by ``n_variables * n_depths * n_lat * n_lon``.  We want a weighted
        mean over *variables* of an ocean-only spatial mean, so each group's
        coefficients are multiplied by that group's variable count to cancel the
        variable mean (the level group keeps its 1/n_depths, which is the mean
        over depths we want), and the spatial weights are normalised to average 1
        over the ocean.  This is exactly geoarches' own construction -- it
        multiplies its surface coefficients by 4 "because we do a mean" -- with
        ERA5's constants replaced by ours.
        """
        groups = {"surface": self.surface_variables}
        if self.n_level_out:
            groups["level"] = self.level_variables
        mask = state_mask(masks, self.surface_variables, self.level_variables)
        total_weight = sum(
            V.VARIABLES[name].loss_weight for names in groups.values() for name in names
        )

        for group, names in groups.items():
            n_lat, n_lon = mask[group].shape[-2:]
            # Sums to 1 over the ocean -> multiply by the cell count so that the
            # parent's mean over (lat, lon) gives an ocean average of 1 for a
            # uniform error field.
            spatial = ocean_area_weights(mask[group], compute_lat_weights_glorys) * (n_lat * n_lon)
            weight = torch.tensor(
                [V.VARIABLES[name].loss_weight for name in names], dtype=torch.float32
            ).reshape(-1, 1, 1, 1)
            if self.loss_delta_normalization:
                statistics = self._statistics[group]
                weight = weight * (statistics["std"] / statistics["delta_std"]).pow(self.pow)
            coefficients = spatial * weight * (len(names) / total_weight)
            self.register_buffer(f"loss_coeff_{group}", coefficients, persistent=False)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _as_tensordict(self, entries: dict[str, torch.Tensor], like: TensorDict) -> TensorDict:
        return TensorDict(entries, batch_size=like.batch_size)

    def select_prognostic(self, state: TensorDict) -> TensorDict:
        """The channels this component predicts, taken off a full state.

        Relies on the dataloader ordering channels prognostic-first (checked by
        ``tests/test_configs.py``), so a component's own variables are always the
        leading channels of both the input and the output.  For the ``full``
        component this is a no-op view.
        """
        entries = {"surface": state["surface"][..., : self.n_surface_out, :, :, :]}
        if self.n_level_out:
            entries["level"] = state["level"][..., : self.n_level_out, :, :, :]
        return self._as_tensordict(entries, state)

    def apply_wet_mask(self, state: TensorDict, inputs: bool = False) -> TensorDict:
        """Zero every land cell.  ``inputs=True`` masks a full input state."""
        prefix = "input_mask" if inputs else "mask"
        entries = {"surface": state["surface"] * getattr(self, f"{prefix}_surface")}
        if "level" in state.keys():
            entries["level"] = state["level"] * getattr(self, f"{prefix}_level")
        return self._as_tensordict(entries, state)

    def clamp_to_physical_bounds(self, state: TensorDict) -> TensorDict:
        """Enforce ``variables.py``'s bounds, in normalised space.

        The bounds live in physical units (``siconc`` in [0, 1], ``sithick`` >= 0)
        but the model's output is normalised, so the *bound* is converted once at
        construction with the same mean and standard deviation the state was
        normalised with -- see ``_register_statistics``.  Clamping here rather
        than after denormalisation keeps the constraint inside the graph, so the
        gradient of an out-of-range prediction is zero instead of being spent
        pushing further out of range.

        Note the dtype: under ``bf16-mixed`` the state arrives as bfloat16 and the
        bounds are float32, so ``clamp`` promotes the result to float32.  That is
        deliberate -- the loss and the metrics are more accurate for it, and the
        promotion happens after the last matmul, so it costs no throughput.
        """
        if not self._has_bounds:
            return state
        entries = {
            "surface": state["surface"].clamp(
                min=self.clamp_low_surface, max=self.clamp_high_surface
            )
        }
        if "level" in state.keys():
            entries["level"] = state["level"].clamp(
                min=self.clamp_low_level, max=self.clamp_high_level
            )
        return self._as_tensordict(entries, state)

    def denormalize_state(self, state: TensorDict) -> TensorDict:
        """``x * std + mean`` for the component's *own* channels.

        Not ``dataset.denormalize``: the dataset's statistics cover everything it
        reads, which for a component is more channels than the model returns.
        Using the module's own copy also means the metrics see exactly the
        statistics the loss was built from.
        """
        entries = {"surface": state["surface"] * self.state_std_surface + self.state_mean_surface}
        if "level" in state.keys():
            entries["level"] = state["level"] * self.state_std_level + self.state_mean_level
        return self._as_tensordict(entries, state)

    def loss_coefficients(self) -> TensorDict:
        """The loss weights, assembled from the registered buffers (already on device)."""
        entries = {"surface": self.loss_coeff_surface}
        if self.n_level_out:
            entries["level"] = self.loss_coeff_level
        return TensorDict(entries)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @staticmethod
    def compute_lat_weights_glorys(latitude_resolution: int) -> torch.Tensor:
        """See :func:`oceanarches.metrics.masked_metrics.compute_lat_weights_glorys`."""
        return compute_lat_weights_glorys(latitude_resolution)

    def time_conditioning(self, timestamp: torch.Tensor) -> torch.Tensor:
        """Month + hour embedding.

        Six lines lifted from ``ForecastModuleWithCond.forward``, because that
        method computes the conditioning *and* calls ``ForecastModule.forward``,
        which adds the whole input state to a component's partial output.  We need
        the first half without the second.
        """
        device = timestamp.device
        times = pd.to_datetime(timestamp.detach().cpu().numpy(), unit="s").tz_localize(None)
        month = torch.as_tensor(np.asarray(times.month), device=device)
        hour = torch.as_tensor(np.asarray(times.hour), device=device)
        return self.month_embedder(month) + self.hour_embedder(hour)

    def external_forcing(self, timestamp: torch.Tensor) -> torch.Tensor | None:
        """Prescribed forcing for a batch of times, or None when there is none.

        Two ways to say "no forcing", and both have to work: ``forcing=None``,
        and a source whose ``get`` returns None -- which is exactly what
        :class:`~oceanarches.dataloaders.forcing.NoForcing` is defined to do.
        Handling only the first made ``forcing: NoForcing()`` a trap that fired
        as ``TypeError: expected Tensor as element 0`` at the first batch.
        """
        if self.forcing_source is None:
            return None
        fields = [self.forcing_source.get(int(t)) for t in timestamp.detach().cpu().flatten()]
        if not fields or fields[0] is None:
            return None
        return torch.stack(fields).to(timestamp.device)

    def forward(self, batch, use_avg: bool = True, forcing: torch.Tensor | None = None):
        """One step: a full state in, this component's prognostic variables out.

        Args:
            batch: dict with ``state`` (and ``prev_state`` when the embedder was
                built with ``n_concatenated_states=1``) and ``timestamp``.
            use_avg: accepted for signature compatibility with geoarches.
            forcing: overrides :meth:`external_forcing`, used by the rollout.

        Returns:
            TensorDict with ``surface`` ``(batch, n_surface_out, 1, lat, lon)``
            and, unless the component is surface-only, ``level``.
        """
        if forcing is None:
            forcing = self.external_forcing(batch["timestamp"])
        tokens = self.embedder.encode(batch["state"], batch.get("prev_state", None), forcing)
        tokens = self.backbone(tokens, self.time_conditioning(batch["timestamp"]))
        prediction = self.embedder.decode(tokens)

        if self.add_input_state:
            # The residual: predict the one-day *change*, not the state.
            prediction = prediction + self.select_prognostic(batch["state"])
        if self.clamp_physical_bounds:
            prediction = self.clamp_to_physical_bounds(prediction)
        # Last, and after the clamp: a clamped land cell is no longer 0, and the
        # whole framework treats land as exactly 0 in normalised space.
        return self.apply_wet_mask(prediction)

    def advance_state(self, state: TensorDict, prediction: TensorDict) -> TensorDict:
        """The input state for the next rollout step.

        The prediction replaces this component's own channels; the *forcing*
        channels -- what it reads but does not predict -- are carried forward
        unchanged, i.e. persistence.  That is the honest default for a component
        run on its own; feeding zeros would after normalisation assert "the
        climatological mean everywhere", which is a much stronger claim.  Task 8's
        coupled module overrides this to write another component's prediction into
        those channels instead.

        For the ``full`` component there are no forcing channels and this is just
        the prediction.
        """
        entries = {}
        for group, n_out, n_in in (
            ("surface", self.n_surface_out, self.n_surface_in),
            ("level", self.n_level_out, self.n_level_in),
        ):
            if n_in == 0:
                continue
            if n_out == 0:
                entries[group] = state[group]
            elif n_out == n_in:
                entries[group] = prediction[group]
            else:
                entries[group] = torch.cat(
                    [prediction[group], state[group][..., n_out:, :, :, :]], dim=-4
                )
        return self.apply_wet_mask(self._as_tensordict(entries, state), inputs=True)

    def on_train_epoch_start(self, *args, **kwargs):
        """Hold the training rollout at the length the config asked for.

        geoarches' own hook is::

            if dataset.multistep > 1:
                dataset.multistep = 2 + self.current_epoch // self.increase_multistep_period

        -- unconditional, and it overwrites whatever the dataloader was built
        with.  So ``++module.train.rollout_iterations=2``, which four documents
        recommend as the fix for the 90-day divergence, does not train a 2-step
        model: measured against the real dataset object, the rollout climbs
        2 -> 3 -> ... -> 10 over ``tiny``'s 18 epochs, a mean of **5.88** steps
        per training step.  ``=3`` is silently *reduced* to 2 at epoch 0.  A
        participant reading "about twice the compute" queues an hour and needs
        about six.

        Holding it fixed is the honest default: you get the rollout length you
        asked for, and the cost is the one the number implies.  Pass
        ``++module.module.multistep_curriculum=True`` for geoarches' ramp, and
        budget for it -- with ``increase_multistep_period=2`` the mean rollout
        length over ``E`` epochs is about ``2 + E / 4``.
        """
        if self.multistep_curriculum:
            return super().on_train_epoch_start(*args, **kwargs)
        return None

    def forward_multistep(
        self,
        batch,
        iters: int | None = None,
        return_format: str = "tensordict",
        use_avg: bool = True,
    ):
        """Autoregressive rollout in *grid space*.

        Same structure and same gradient checkpointing as geoarches, with three
        changes:

        1. the timestamp advances by ``self.lead_time_hours`` per step.  geoarches
           advances it by ``batch["lead_time_hours"]``, which our dataloader sets
           to ``lead_time_hours * multistep`` -- for a 10-day rollout that would
           move the month-of-year conditioning forward by 100 days per step;
        2. the state handed to the next step is rebuilt by :meth:`advance_state`,
           so a component's forcing channels survive and the wet mask is
           re-applied after every step;
        3. the prescribed forcing, if any, is fetched for each step's *input*
           time, not its valid time. That is the correct behaviour -- the
           forcing is an input to the step, read alongside the state it is
           handed -- and the wording used to say "valid time", which is a
           different instant one lead time later.

        The state passed from step to step is always a full physical state on the
        180x360 grid, never a latent -- that is what lets Task 8 swap another
        model's prediction into it between steps.
        """
        if use_avg and self.avg_modules is not None:
            out = self.forward_multistep(batch, iters=iters, use_avg=False)
            for module in self.avg_modules:
                out = out + module.forward_multistep(batch, iters=iters, use_avg=False)
            return out / (1 + len(self.avg_modules))

        predictions = []
        loop_batch = {key: value for key, value in batch.items()}
        step_seconds = int(self.lead_time_hours) * 3600
        for _ in range(iters):
            predecessor = loop_batch["state"]
            if torch.is_grad_enabled():
                prediction = gradient_checkpoint.checkpoint(
                    self.forward, loop_batch, use_reentrant=False
                )
            else:
                prediction = self.forward(loop_batch)
            predictions.append(prediction)
            loop_batch = dict(
                state=self.advance_state(loop_batch["state"], prediction),
                timestamp=loop_batch["timestamp"] + step_seconds,
                lead_time_hours=batch.get("lead_time_hours"),
            )
            # Only when the embedder has channels for it. Built with
            # `n_concatenated_states=0` (the `load_prev: False` ablation) it
            # raises "encode() was given a cond_state but the embedder was built
            # with n_concatenated_states=0" -- at rollout step 2, i.e. after a
            # single-step model has trained perfectly well.
            if self.embedder.n_concatenated_states:
                loop_batch["prev_state"] = predecessor

        if return_format == "list":
            return predictions
        return torch.stack(predictions, dim=1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss(self, pred: TensorDict, gt: TensorDict, multistep: bool = False, **kwargs):
        """Masked, latitude-weighted, tendency-normalised error.  See the module docstring."""
        gt = self.select_prognostic(gt)  # a no-op view for the `full` component
        coefficients = self.loss_coefficients()

        if multistep:
            lead_iterations = next(iter(gt.values())).shape[1]
            discount = torch.tensor(
                [1 / (1 + i) ** 2 for i in range(lead_iterations)], device=self.device
            )
            # Normalised to mean 1 so that a multistep loss is on the same scale
            # as a single-step one.  geoarches computes this discount and then
            # discards it -- its `loss_coeffs.apply(...)` result is never assigned
            # -- so a multistep run there silently weights every lead time equally.
            discount = discount / discount.mean()
            coefficients = coefficients.apply(lambda x: x * discount.reshape(-1, 1, 1, 1, 1))

        weighted_error = (pred - gt).abs().pow(self.pow).mul(coefficients)
        return sum(weighted_error.mean().values())

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _predict(self, batch, rollout_iterations: int):
        """Run the model, compute the loss and return metric-ready tensors.

        Returns ``(loss, targets, preds)`` with the targets and predictions
        denormalised and carrying a leading ``timedelta`` axis, which is the shape
        every metric in ``oceanarches.metrics`` expects.
        """
        if "future_states" not in batch:
            prediction = self.forward(batch)
            target = self.select_prognostic(batch["next_state"])
            loss = self.loss(prediction, target)
            target, prediction = target[:, None], prediction[:, None]
        else:
            lead_iterations = batch["future_states"].shape[1]
            prediction = self.forward_multistep(batch, iters=lead_iterations)
            target = self.select_prognostic(batch["future_states"])
            loss = self.loss(prediction, target, multistep=True)
            target = target[:, :rollout_iterations]
            prediction = prediction[:, :rollout_iterations]
        return loss, self.denormalize_state(target), self.denormalize_state(prediction)

    # geoarches' `mylog` has a mutable default argument (`dct={}`) that it then
    # updates, so keys leak from one call to the next and from training into
    # validation. Passing an explicit dict every time keeps that default empty.
    def training_step(self, batch, batch_nb):
        for metric in self.train_metrics:
            metric.reset()
        loss, targets, preds = self._predict(
            batch, self.cfg.train.metrics_kwargs.rollout_iterations
        )
        self.mylog(dict(loss=loss))
        for metric in self.train_metrics:
            metric.update(targets, preds, timestamp=batch["timestamp"])
            self.mylog(dict(metric.compute()))
        return loss

    def validation_step(self, batch, batch_nb):
        loss, targets, preds = self._predict(batch, self.cfg.val.metrics_kwargs.rollout_iterations)
        self.mylog(dict(loss=loss))
        for metric in self.val_metrics:
            metric.update(targets, preds, timestamp=batch["timestamp"])
        return loss

    def test_step(self, batch, batch_nb):
        dataset = self.trainer.test_dataloaders.dataset
        predictions = self.forward_multistep(batch, iters=dataset.multistep)
        reference = (
            batch["future_states"] if "future_states" in batch else batch["next_state"][:, None]
        )
        targets = self.denormalize_state(self.select_prognostic(reference))
        preds = self.denormalize_state(predictions)
        for metric in self.test_metrics.values():
            metric.update(targets, preds, timestamp=batch["timestamp"])

        if self.save_test_outputs:
            if (self.n_surface_out, self.n_level_out) != (self.n_surface_in, self.n_level_in):
                raise NotImplementedError(
                    f"save_test_outputs writes through the dataset's own variable list, which "
                    f"describes the {self.n_surface_in} channels it reads, not the "
                    f"{self.n_surface_out} that component {self.component.name!r} predicts. "
                    "Use `make eval` (oceanarches.evaluation) to write component rollouts."
                )
            # Already denormalised above, and `levels` is left at None because our
            # vertical coordinate is depth in metres, not geoarches' pressure.
            self.zarr_writer.write(
                dataset.convert_trajectory_to_xarray(
                    preds, timestamp=batch["timestamp"], denormalize=False, levels=None
                ),
                append_dim="time",
            )
            if not (batch_nb + 1) % 25:
                self.zarr_writer.to_netcdf(dump_id=batch_nb)
