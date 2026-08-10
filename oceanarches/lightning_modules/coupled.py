"""Coupling: running independently trained components as one forecast system.

This is the file the challenge's headline question hangs off -- *how do you
disentangle the coupling between ocean, sea ice and atmosphere?* -- so it is
written to be read.

The whole thing rests on one property of
:meth:`~oceanarches.lightning_modules.ocean_forecast.OceanForecastModule.forward_multistep`:
the rollout is **grid-space and autoregressive**.  Every step produces a full
physical state on the 180x360 grid; nothing is carried between steps in a latent
space.  Component A's prediction is therefore an ordinary field, and an ordinary
field can be written into a shared state that component B reads on the next
step.  That is the entire coupling mechanism, and it is why two models trained on
different days by different people can be run together.

The shared state
----------------
One state, one fixed channel order, shared by every component::

    shared surface (batch, var, 1,     lat, lon)   var = layout.input_surface
    shared level   (batch, var, depth, lat, lon)   var = layout.input_level

The layout is the *union component*: what the coupled system predicts between
them (canonical order, first) followed by what it reads but nobody predicts.
For ocean + sea ice that union is exactly ``COMPONENTS["full"]``, i.e. the layout
``configs/dataloader/glorys.yaml`` already loads -- which is why a coupled
ocean+ice system can be scored by the Task 7 pipeline with no special case at
all.

Two ways forcing reaches a component, and both work:

1. **From another component**, through the shared state.  This file.
2. **From files**, through
   :class:`~oceanarches.dataloaders.forcing.XarrayForcing` -- the prescribed
   atmosphere.  Each component keeps its own ``forcing_source``, so a coupled
   system where the ocean is forced by IFS winds and the ice is not needs
   nothing here; :meth:`OceanForecastModule.forward` fetches it per component.

And a model with **no forcing at all** must work, is the default, and is the
best-tested path: with one ``full`` component the router is the identity and
:class:`CoupledForecastModule` reproduces that module's own rollout bit for bit
(``tests/test_coupling.py::test_a_coupled_full_component_is_bit_identical_to_it_alone``).

Two coupling modes
------------------
``parallel``    every component sees the state at time *t*; the outputs are
                merged afterwards.  This is what you get by running two models
                side by side and stitching the answers together.
``sequential``  components run in a configured order, and a later component sees
                the fields an earlier one has *already updated*.  Ocean first,
                then sea ice, is the natural default: the ice responds to today's
                ocean rather than to yesterday's.

The two genuinely differ -- the difference is a real (small) scientific result
about how tightly the two systems are coupled at a one-day step, and reporting it
is part of the exercise.  Measured numbers are in the Task 8 report.

What happens to channels nobody predicts
----------------------------------------
Run the sea-ice model on its own and the seven ocean channels it reads are
nobody's output.  They are never silently zeroed -- zero in normalised space
asserts "the climatological mean everywhere", which is a much stronger claim than
"I don't know".  The behaviour is an explicit choice:

``persistence``    hold them at the initial value.  The honest default for a
                   *free-running* forecast: no information from outside the
                   system enters the rollout.
``ground_truth``   take them from the dataset at each step's valid time.  A
                   perfect-forcing experiment -- it tells you how much of a
                   component's error is its own and how much it inherited.

Training a specialist needs neither knob: training is single-step, so the
forcing channels come straight off the dataloader, i.e. ground truth by
construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from geoarches.lightning_modules.base_module import load_module
from tensordict.tensordict import TensorDict

from ..dataloaders import variables as V
from ..dataloaders.masks import load_masks
from .ocean_forecast import OceanForecastModule

__all__ = [
    "StateRouter",
    "CoupledForecastModule",
    "union_component",
    "named_component_like",
    "COUPLING_MODES",
    "UNPREDICTED_FORCING_MODES",
]

#: How the components are stepped relative to each other.
COUPLING_MODES = ("parallel", "sequential")

#: What happens to shared-state channels that no component predicts.
UNPREDICTED_FORCING_MODES = ("persistence", "ground_truth")

#: Channel axis of a state tensor, counted from the right:
#: ``(..., var, depth, lat, lon)``.
_VAR_AXIS = -4


# ---------------------------------------------------------------------------
# The union component
# ---------------------------------------------------------------------------
def union_component(
    components: Sequence[V.ComponentSpec], name: str | None = None
) -> V.ComponentSpec:
    """The single component a coupled system behaves like, from its parts.

    Its prognostic variables are everything the components predict between them;
    its forcing variables are everything they read that none of them predicts.
    Both lists come back in the canonical order of ``variables.py`` because
    :class:`~oceanarches.dataloaders.variables.ComponentSpec` derives them by
    filtering the canonical lists -- so the union of ``ocean`` and ``seaice`` is
    channel-for-channel the ``full`` component.

    Args:
        components: the component specs being run together.
        name: name for the result; defaults to the parts joined with ``+``.

    Raises:
        ValueError: if two components predict the same variable.  Coupling is
            an exchange, not a vote: there is no defensible way to merge two
            forecasts of the same field into one shared state.
    """
    if not components:
        raise ValueError("A coupled system needs at least one component.")

    owner: dict[str, str] = {}
    for spec in components:
        for variable in spec.prognostic:
            if variable in owner:
                raise ValueError(
                    f"Components {owner[variable]!r} and {spec.name!r} both predict "
                    f"{variable!r}. Every variable in the shared state must have exactly "
                    "one owner; drop one of the two components, or define a component "
                    "in variables.py that splits them."
                )
            owner[variable] = spec.name

    predicted = list(owner)
    read_only = [
        variable
        for spec in components
        for variable in spec.forcing
        if variable not in owner  # supplied by another component -> not forcing any more
    ]
    return V.ComponentSpec(
        name=name or "+".join(spec.name for spec in components),
        prognostic=predicted,
        # de-duplicated; ComponentSpec re-sorts into canonical order anyway
        forcing=list(dict.fromkeys(read_only)),
    )


def named_component_like(spec: V.ComponentSpec) -> str | None:
    """Name of the ``COMPONENTS`` entry with the same variables, or None.

    The metrics, the dataloader configs and the report all identify a state by a
    component *name*, so a coupled system has to be able to say which shipped
    component it is equivalent to.  ``ocean + seaice`` is ``full``; ``seaice``
    alone is ``seaice``.  A combination that matches nothing means
    ``variables.py`` needs a new entry -- it is the single source of truth for
    what is in a state, and this module does not get to invent a parallel one.
    """
    for name, candidate in V.COMPONENTS.items():
        if (
            candidate.prognostic_surface == spec.prognostic_surface
            and candidate.prognostic_level == spec.prognostic_level
            and candidate.forcing_surface == spec.forcing_surface
            and candidate.forcing_level == spec.forcing_level
        ):
            return name
    return None


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------
class StateRouter:
    """Index bookkeeping between the shared state and one component's tensors.

    Nothing else.  There is no arithmetic here and no model: given the shared
    state's channel order, :meth:`gather` selects the channels a component reads
    and :meth:`scatter` writes the channels it predicts back.  Everything it
    knows comes from ``variables.py``.

    Args:
        surface: shared-state surface channel names, in order.
        level: shared-state level channel names, in order.  Empty for a
            surface-only system.

    Example, for the sea-ice component in a full shared state::

        router = StateRouter.for_layout(V.COMPONENTS["full"])
        ice_in = router.gather(shared, V.COMPONENTS["seaice"])   # ice first, then ocean
        shared = router.scatter(shared, V.COMPONENTS["seaice"], ice_out)
    """

    def __init__(self, surface: Sequence[str], level: Sequence[str] = ()):
        self.surface = list(surface)
        self.level = list(level)
        for group, names in (("surface", self.surface), ("level", self.level)):
            if len(set(names)) != len(names):
                raise ValueError(
                    f"The shared state layout repeats a {group} variable: {names}. "
                    "Every channel must name a different field."
                )
        self._indices: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}

    @classmethod
    def for_layout(cls, layout: V.ComponentSpec) -> "StateRouter":
        """Router for a shared state laid out as ``layout``'s input.

        That is the same order the dataloader produces for that component --
        prognostic channels first, then forcing -- so a shared state and a
        ``GlorysForecast`` sample of the same component are the same thing.
        """
        return cls(layout.input_surface, layout.input_level)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StateRouter(surface={self.surface}, level={self.level})"

    # -- indices ---------------------------------------------------------
    def index(self, group: str, names: Sequence[str], component: str, what: str) -> torch.Tensor:
        """Positions of ``names`` in the shared state's ``group`` channel axis."""
        key = (group, tuple(names))
        if key not in self._indices:
            layout = self.surface if group == "surface" else self.level
            missing = [name for name in names if name not in layout]
            if missing:
                raise KeyError(
                    f"Component {component!r} needs the {group} {what} {missing}, but the "
                    f"shared state only carries {layout}. The shared state has to be built "
                    "for a component that covers every variable its parts read -- see "
                    "union_component()."
                )
            self._indices[key] = torch.tensor([layout.index(n) for n in names], dtype=torch.long)
        return self._indices[key]

    # -- the two operations ----------------------------------------------
    def gather(self, shared_state: TensorDict, component: V.ComponentSpec) -> TensorDict:
        """A component's input state, selected out of the shared state.

        Channel order is the component's own: its prognostic variables first,
        then its forcing variables -- exactly what its embedder was wired for
        and what ``configs/dataloader/glorys_*.yaml`` load.  For the ``full``
        component this is a copy with the channels in the same order, i.e. the
        identity.

        Args:
            shared_state: ``surface`` and (unless surface-only) ``level``,
                shaped ``(..., var, depth, lat, lon)``.
            component: whose input to build.
        """
        entries = {}
        for group, names in (
            ("surface", component.input_surface),
            ("level", component.input_level),
        ):
            if not names:
                continue
            if group not in shared_state.keys():
                raise KeyError(
                    f"Component {component.name!r} reads the {group} variables {names}, but "
                    f"the shared state has no {group!r} key (it holds "
                    f"{sorted(shared_state.keys())})."
                )
            tensor = shared_state[group]
            index = self.index(group, names, component.name, "input")
            entries[group] = tensor.index_select(tensor.dim() + _VAR_AXIS, index.to(tensor.device))
        return TensorDict(entries, batch_size=shared_state.batch_size)

    def scatter(
        self,
        shared_state: TensorDict,
        component: V.ComponentSpec,
        prediction: TensorDict,
    ) -> TensorDict:
        """The shared state with this component's prognostic channels replaced.

        Every other channel is copied through untouched, which is what makes
        ``gather`` then ``scatter`` of a set of components that between them own
        every channel reconstruct the whole state exactly.

        Args:
            shared_state: the state to write into.  Not modified in place.
            component: whose output ``prediction`` is.
            prediction: the component's own channels, in its own order.
        """
        # `index_copy` is out of place, so nothing here mutates the caller's
        # tensors and a channel no component owns is carried through by
        # reference -- a 90-day rollout does not want a copy of the whole state
        # per step.
        entries = {group: shared_state[group] for group in shared_state.keys()}
        for group, names in (
            ("surface", component.prognostic_surface),
            ("level", component.prognostic_level),
        ):
            if not names:
                continue
            if group not in prediction.keys():
                raise KeyError(
                    f"Component {component.name!r} predicts the {group} variables {names}, "
                    f"but its prediction has no {group!r} key (it holds "
                    f"{sorted(prediction.keys())})."
                )
            values = prediction[group]
            found = values.shape[values.dim() + _VAR_AXIS]
            if found != len(names):
                raise ValueError(
                    f"Component {component.name!r} predicts {len(names)} {group} variables "
                    f"{names}, but its prediction has {found} channels. The module's "
                    "`component` and the checkpoint it was trained as have to agree."
                )
            target = entries[group]
            # Under bf16 autocast a component's clamp promotes its output to
            # float32 while the shared state is still bfloat16; promote the
            # destination rather than silently rounding the prediction back down.
            dtype = torch.promote_types(target.dtype, values.dtype)
            if target.dtype != dtype:
                target = target.to(dtype)
            entries[group] = target.index_copy(
                target.dim() + _VAR_AXIS,
                self.index(group, names, component.name, "output").to(target.device),
                values.to(dtype),
            )
        return TensorDict(entries, batch_size=shared_state.batch_size)


# ---------------------------------------------------------------------------
# The coupled module
# ---------------------------------------------------------------------------
class CoupledForecastModule(nn.Module):
    """N trained components, run together, presenting one model's interface.

    ``forward`` and ``forward_multistep`` have the same signatures as
    :class:`~oceanarches.lightning_modules.ocean_forecast.OceanForecastModule`,
    and so do the state helpers the evaluation pipeline uses, so Task 7 scores a
    coupled system exactly like a single model with no special case anywhere.

    Args:
        cfg: unused; accepted first so that ``load_module`` (which calls
            ``instantiate(cfg.module.module, cfg.module)``) can build this class
            too.  Everything a coupled system needs comes from the components'
            own configs.
        components: ``{component name: checkpoint}``.  A checkpoint is a run
            name under ``modelstore/``, a path to a run directory, or an
            already-built module (the tests pass modules).  The key must be the
            name of the ``COMPONENTS`` entry that checkpoint was trained as.
        mode: ``"sequential"`` (later components see earlier ones' updates) or
            ``"parallel"`` (everyone sees time *t*).
        order: the order components run in; defaults to the order they are
            given, which for ``--components ocean=... seaice=...`` is ocean
            first.  Only meaningful for ``sequential``, and validated for both.
        unpredicted_forcing: ``"persistence"`` or ``"ground_truth"`` -- see the
            module docstring.  Ignored when every channel has an owner.
        name: run name, for the evaluation directories.
        component: name of the ``COMPONENTS`` entry the union is equivalent to.
            Normally derived from the parts; pass it only to be explicit, and it
            must agree with what they predict or construction fails.
            Derived when not given.
        device: where to put components loaded from a path.
        loss_delta_normalization, pow: as in ``OceanForecastModule``; they define
            the loss the pipeline reports, and the defaults are the shipped
            module's, so a coupled loss is comparable with a single model's.
        stats_path, masks_path: overrides, for tests.
    """

    # ---- borrowed wholesale from the single-model module ----------------
    # These are the state bookkeeping of *a component*, and a coupled system is
    # a component: the union one.  Borrowing the methods rather than copying
    # them is what makes a coupled system's statistics, wet masks, loss
    # coefficients and denormalisation the *same arithmetic* a single model of
    # the same variables uses -- which is what
    # `test_a_coupled_full_component_is_bit_identical_to_it_alone` and the
    # measured agreement with `make eval NAME=task6_tiny` actually check.  They
    # read nothing but the attributes set in `__init__` below.
    _as_tensordict = OceanForecastModule._as_tensordict
    _register_statistics = OceanForecastModule._register_statistics
    _register_masks = OceanForecastModule._register_masks
    _register_loss_coefficients = OceanForecastModule._register_loss_coefficients
    select_prognostic = OceanForecastModule.select_prognostic
    apply_wet_mask = OceanForecastModule.apply_wet_mask
    denormalize_state = OceanForecastModule.denormalize_state
    loss_coefficients = OceanForecastModule.loss_coefficients
    loss = OceanForecastModule.loss

    def __init__(
        self,
        cfg=None,
        components: Mapping[str, object] | None = None,
        mode: str = "sequential",
        order: Sequence[str] | None = None,
        unpredicted_forcing: str = "persistence",
        name: str = "coupled",
        component: str | None = None,
        device: str | torch.device = "cpu",
        loss_delta_normalization: bool = True,
        pow: int = 2,
        stats_path: str | Path | None = None,
        masks_path: str | Path | None = None,
    ):
        super().__init__()
        if not components:
            raise ValueError(
                "CoupledForecastModule needs components, e.g. "
                "components={'ocean': 'my_ocean_run', 'seaice': 'my_ice_run'}."
            )
        if mode not in COUPLING_MODES:
            raise ValueError(f"mode must be one of {COUPLING_MODES}, got {mode!r}.")
        if unpredicted_forcing not in UNPREDICTED_FORCING_MODES:
            raise ValueError(
                f"unpredicted_forcing must be one of {UNPREDICTED_FORCING_MODES}, "
                f"got {unpredicted_forcing!r}."
            )

        self.name = name
        self.mode = mode
        self.unpredicted_forcing = unpredicted_forcing
        self.loss_delta_normalization = bool(loss_delta_normalization)
        self.pow = int(pow)

        #: Where each component came from, for the report's reproduction command:
        #: a coupled system has no single checkpoint under ``modelstore/``, so the
        #: only thing that can say how to re-run it is the system itself.
        self.sources = {
            key: str(value) if isinstance(value, (str, Path)) else "<module>"
            for key, value in components.items()
        }
        self.components = nn.ModuleDict(
            {key: _as_module(value, device) for key, value in components.items()}
        )
        self.order = list(order) if order is not None else list(self.components.keys())
        if sorted(self.order) != sorted(self.components.keys()):
            raise ValueError(
                f"order={self.order} does not name the components "
                f"{sorted(self.components.keys())}."
            )

        self.specs: dict[str, V.ComponentSpec] = {}
        for key, module in self.components.items():
            spec = getattr(module, "component", None)
            if spec is None:
                raise TypeError(
                    f"Component {key!r} is a {type(module).__name__}, which has no `component` "
                    "attribute. Coupling needs a module that knows which variables it predicts."
                )
            if spec.name != key:
                raise ValueError(
                    f"Component {key!r} was given a checkpoint trained as {spec.name!r}. "
                    "Name each component after the COMPONENTS entry it was trained as, e.g. "
                    f"--components {spec.name}=<run>."
                )
            if getattr(module, "avg_modules", None):
                # geoarches' weight-averaging is applied inside
                # `OceanForecastModule.forward_multistep`, and a coupled rollout
                # never calls it: `step` below drives each component's `forward`
                # directly, because the components have to be interleaved one
                # lead time at a time.  So an EMA would be silently dropped and
                # the coupled scores would be of the raw weights.  Nothing in
                # this kit configures one; say so rather than lose it quietly.
                raise NotImplementedError(
                    f"Component {key!r} carries geoarches' averaged modules "
                    "(`avg_modules`), which a coupled rollout cannot apply: the components "
                    "are stepped one lead time at a time through their `forward`, and the "
                    "averaging lives in `forward_multistep`. Couple the averaged weights "
                    "themselves instead."
                )
            self.specs[key] = spec

        self._check_depth_presets()
        self._check_lead_times()

        # -- the union: what this system looks like from outside ----------
        derived = union_component([self.specs[key] for key in self.order])
        if component is not None:
            # An explicit name has to *agree* with what the parts actually
            # predict.  It used to override silently, which would have given the
            # shared state a channel layout the components never write into: the
            # rollout would run, the report would name the wrong component, and
            # the extra channels would sit at their initial value for the whole
            # forecast without anything saying so.
            named = V.get_component(component)
            if sorted(named.prognostic) != sorted(derived.prognostic):
                raise ValueError(
                    f"component={component!r} predicts {sorted(named.prognostic)}, but the "
                    f"components {self.order} between them predict "
                    f"{sorted(derived.prognostic)}. Drop the `component` argument and let it "
                    "be derived, or name one that matches."
                )
        component = component or named_component_like(derived)
        if component is None:
            raise ValueError(
                f"The components {self.order} together predict {derived.prognostic} and read "
                f"{derived.forcing}, which is not one of the COMPONENTS entries "
                f"{sorted(V.COMPONENTS)}. The metrics, the dataloader configs and the report "
                "all identify a state by a component name, so add this combination to "
                "COMPONENTS in oceanarches/dataloaders/variables.py."
            )
        self.component = V.get_component(component)
        self.router = StateRouter.for_layout(self.component)
        #: The shared-state channels nobody predicts, as a pseudo-component, so
        #: that ground-truth forcing is written with the same router the
        #: components use rather than a second indexing scheme.
        self._unowned = V.ComponentSpec(
            name="unpredicted", prognostic=list(self.component.forcing)
        )

        self.surface_variables = self.component.prognostic_surface
        self.level_variables = self.component.prognostic_level
        self.n_surface_in = self.component.n_surface_in
        self.n_surface_out = self.component.n_surface_out
        self.n_level_in = self.component.n_level_in
        self.n_level_out = self.component.n_level_out
        # Keep None as None. `list(x or [])` turned "load every prepared level"
        # into an empty list, and `_select_statistics` -- borrowed from
        # OceanForecastModule, guarded on `depth_indices is not None` -- then
        # sliced the statistics to a zero-length depth axis. `std` came out
        # shaped [1, 0, 1, 1] and its own positivity check, `(std > 0).all()`,
        # was vacuously True. Reachable from any component config that omits
        # `module.depth_indices`.
        self.depth_indices = self._first.depth_indices
        self.lead_time_hours = int(self._first.lead_time_hours)

        masks = load_masks(path=masks_path, depth_indices=self.depth_indices)
        self._register_masks(masks)
        self._register_statistics(stats_path)
        self._register_loss_coefficients(masks)

    # -- construction checks ---------------------------------------------
    @property
    def _first(self):
        return self.components[self.order[0]]

    def _check_depth_presets(self) -> None:
        """Components must load the same depth levels, and say so if they do not.

        A mismatch is silent otherwise: the shared state would carry 13 levels,
        a component built for 12 would read the first 12 of them, and every
        depth from the second one down would be off by one -- a model that looks
        like it works.  There is no honest way to reconcile two vertical grids
        here, so this raises.
        """
        reference_key = self.order[0]
        reference = list(self.components[reference_key].depth_indices or [])
        for key in self.order[1:]:
            found = list(self.components[key].depth_indices or [])
            if found != reference:
                raise ValueError(
                    f"Components {reference_key!r} and {key!r} were trained on different "
                    f"depth presets: {reference_key!r} uses depth_indices={reference} "
                    f"({_preset_label(reference)}) and {key!r} uses depth_indices={found} "
                    f"({_preset_label(found)}). They exchange fields on one shared grid, so "
                    "they have to load the same levels. Retrain one of them with the other's "
                    "`module.depth_indices`."
                )

    def _check_lead_times(self) -> None:
        reference_key = self.order[0]
        reference = int(self.components[reference_key].lead_time_hours)
        for key in self.order[1:]:
            found = int(self.components[key].lead_time_hours)
            if found != reference:
                raise ValueError(
                    f"Component {reference_key!r} steps {reference} h at a time and {key!r} "
                    f"steps {found} h. A coupled rollout advances one clock; retrain one of "
                    "them with the other's `lead_time_hours`."
                )

    # -- what the pipeline expects off a module ---------------------------
    @property
    def device(self) -> torch.device:
        return self.mask_surface.device

    def evaluation_command(self, spec) -> str:
        """The command line that reproduces a scored rollout of this system.

        The evaluation report asks a module for this because it cannot guess:
        its own default -- ``run_eval --exp <name>`` -- names an experiment
        directory that, for a coupled system, holds no checkpoint at all.  The
        components, the mode and the forcing policy all have to appear, because
        every one of them changes the forecast.

        Args:
            spec: the :class:`~oceanarches.evaluation.rollout.RolloutSpec` that
                was scored, for the lead time, the split and the initialisations.
        """
        listing = " ".join(f"{key}={self.sources.get(key, '<module>')}" for key in self.order)
        lines = []
        # `make couple` hardcodes ocean then sea ice and the default policy, so
        # only offer it when that is genuinely what was run.
        if self.order == ["ocean", "seaice"] and self.unpredicted_forcing == "persistence":
            lines += [
                f"make couple OCEAN={self.sources.get('ocean')} "
                f"SEAICE={self.sources.get('seaice')} "
                f"MODE={self.mode} LEAD_DAYS={spec.lead_days}",
                "# or, in full:",
            ]
        lines.append(
            ".venv/bin/python -m oceanarches.evaluation.run_eval --coupled \\\n"
            f"    --components {listing} \\\n"
            f"    --mode {self.mode} --unpredicted-forcing {self.unpredicted_forcing} \\\n"
            f"    --lead-days {spec.lead_days} --n-inits {spec.n_inits} "
            f"--init-selection {spec.selection} --domain {spec.domain}"
        )
        return "\n".join(lines)

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"component={self.component.name!r}, mode={self.mode!r}, order={self.order}, "
            f"unpredicted_forcing={self.unpredicted_forcing!r}"
        )

    # ------------------------------------------------------------------
    # One coupled step
    # ------------------------------------------------------------------
    def step(
        self,
        shared_state: TensorDict,
        prev_state: TensorDict | None,
        timestamp: torch.Tensor,
    ) -> TensorDict:
        """Run every component once and return the merged shared state.

        The whole difference between the two coupling modes is the single ``if``
        below.  ``merged`` always accumulates every component's output;
        ``visible`` is what a component *reads*, and only in ``sequential`` does
        it follow ``merged`` as the loop goes on.  In ``parallel`` it stays at
        the state at time *t*, which is what running two models side by side
        gives you.

        Returns:
            A shared state whose predicted channels are this step's forecast and
            whose unpredicted channels are still the ones that came in.
        """
        visible = shared_state
        merged = shared_state

        for key in self.order:
            spec = self.specs[key]
            batch = {"state": self.router.gather(visible, spec), "timestamp": timestamp}
            # ... and only to a component whose embedder has channels for it. One
            # built with `n_concatenated_states=0` raises rather than ignoring it,
            # and a coupled system may mix the two.
            component = self.components[key]
            wants_prev = bool(
                getattr(getattr(component, "embedder", None), "n_concatenated_states", 1)
            )
            if prev_state is not None and wants_prev:
                batch["prev_state"] = self.router.gather(prev_state, spec)
            prediction = component(batch)
            merged = self.router.scatter(merged, spec, prediction)
            if self.mode == "sequential":
                visible = merged
        return merged

    def forward(self, batch, use_avg: bool = True, forcing: torch.Tensor | None = None):
        """One coupled step: a full shared state in, the predicted channels out.

        Args:
            batch: dict with ``state``, ``timestamp`` and -- when the components
                were built with ``n_concatenated_states=1``, which the shipped
                presets are -- ``prev_state``.
            use_avg: accepted for signature compatibility with geoarches.
            forcing: accepted for signature compatibility.  External forcing is
                per component: each one fetches its own from its
                ``forcing_source``, because two components can legitimately be
                given different atmospheres.
        """
        if forcing is not None:
            raise ValueError(
                "A coupled system does not take one shared forcing tensor: each component "
                "reads its own ForcingSource, so give the forcing to the components "
                "(configs/forcing/*.yaml) instead."
            )
        merged = self.step(batch["state"], batch.get("prev_state"), batch["timestamp"])
        return self.select_prognostic(merged)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def advance_state(
        self, merged: TensorDict, forcing_truth: TensorDict | None = None
    ) -> TensorDict:
        """The shared state the next step reads.

        The mirror of ``OceanForecastModule.advance_state``, and the same
        contract: the predicted channels are this step's forecast, and the wet
        mask is re-applied so that land is exactly 0 going into the next step.
        The one addition is what happens to channels nobody predicts.

        Args:
            merged: the shared state :meth:`step` returned.
            forcing_truth: the ground-truth shared state valid at the *new*
                time, when ``unpredicted_forcing='ground_truth'``.  None means
                persistence -- leave those channels at the value they carry,
                which after the first step is the initial one.
        """
        state = merged
        if forcing_truth is not None and self.component.forcing:
            state = self.router.scatter(
                state, self._unowned, self.router.gather(forcing_truth, self._unowned)
            )
        return self.apply_wet_mask(state, inputs=True)

    def _ground_truth_at(self, batch: dict, step: int) -> TensorDict:
        """The dataset's shared state valid after ``step + 1`` lead times."""
        if "future_states" not in batch:
            raise KeyError(
                "unpredicted_forcing='ground_truth' reads the unpredicted channels off the "
                "dataset, but this batch has no 'future_states'. Build the dataset with "
                "multistep >= the rollout length, or use unpredicted_forcing='persistence'."
            )
        future = batch["future_states"]
        available = future.shape[1]
        if step >= available:
            raise ValueError(
                f"unpredicted_forcing='ground_truth' needs ground truth for step {step + 1}, "
                f"but the batch only carries {available}. Build the dataset with "
                f"multistep >= {step + 1}."
            )
        return future[:, step]

    def forward_multistep(
        self,
        batch,
        iters: int | None = None,
        return_format: str = "tensordict",
        use_avg: bool = True,
    ):
        """Autoregressive coupled rollout, in grid space.

        Structurally identical to ``OceanForecastModule.forward_multistep`` --
        the same loop, the same timestamp advance, the same re-masking between
        steps -- with :meth:`step` in place of a single ``forward``.  With one
        ``full`` component the two produce bit-identical tensors.

        No gradient checkpointing: the components are frozen checkpoints being
        rolled out, so there is nothing to keep a graph for.  Wrap the call in
        ``torch.no_grad()`` (the evaluation pipeline does) if you want the memory
        back.
        """
        if iters is None:
            raise ValueError("forward_multistep needs iters=<number of steps>.")
        iters = int(iters)

        predictions = []
        shared = batch["state"]
        previous = batch.get("prev_state")
        timestamp = batch["timestamp"]
        step_seconds = int(self.lead_time_hours) * 3600
        ground_truth = self.unpredicted_forcing == "ground_truth" and bool(self.component.forcing)

        for step in range(iters):
            merged = self.step(shared, previous, timestamp)
            predictions.append(self.select_prognostic(merged))
            truth = self._ground_truth_at(batch, step) if ground_truth else None
            previous, shared = shared, self.advance_state(merged, truth)
            timestamp = timestamp + step_seconds

        if return_format == "list":
            return predictions
        return torch.stack(predictions, dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_module(entry, device) -> nn.Module:
    """A component: an already-built module, or a checkpoint loaded by path.

    Loading goes through ``geoarches.lightning_modules.base_module.load_module``
    and nothing else, so any checkpoint any participant trained -- with whatever
    backbone, size and forcing -- can be dropped in by name or by path.
    """
    if isinstance(entry, (str, Path)):
        return load_module(str(entry), device=str(device), return_config=False)
    if isinstance(entry, nn.Module):
        return entry
    raise TypeError(
        f"A component must be a run name, a path to a run directory, or a module; "
        f"got {type(entry).__name__}."
    )


def _preset_label(depth_indices: Sequence[int]) -> str:
    """``"the tiny preset"`` when the indices are a shipped one, else ``"custom"``."""
    for name, preset in V.DEPTH_PRESETS.items():
        if list(preset) == list(depth_indices):
            return f"the {name} preset"
    return "a custom preset"
