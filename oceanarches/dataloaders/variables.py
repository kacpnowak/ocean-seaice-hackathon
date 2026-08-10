"""The single source of truth for *what is in the state*.

Everything else -- data preparation, the dataloader, the model's input/output
channel counts, the metric labels, the plots -- derives from the tables here.
If you want to add a variable or change the depth levels, this is the only file
you need to touch.

Two conventions used everywhere in this project, copied from geoarches so that
its modules and metrics work on our data unchanged:

``surface``   2-D fields, tensor shape ``(batch, var, 1, lat, lon)``
              (the length-1 axis is a "depth" axis of size one -- keeping it
              means surface and level tensors have the same rank)
``level``     3-D fields, tensor shape ``(batch, var, depth, lat, lon)``

Variable order inside those tensors is the order of :data:`SURFACE_VARIABLES`
and :data:`LEVEL_VARIABLES`.  It is fixed; do not reorder without regenerating
the normalisation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
# GLORYS regridded to 1 degree.  Cell centres, latitude increasing (south first).
N_LAT, N_LON = 180, 360
LAT = [-89.5 + i for i in range(N_LAT)]  # -89.5 ... 89.5
LON = [float(i) for i in range(N_LON)]  # 0 ... 359

# GLORYS itself only covers latitudes >= -80, so the southernmost 13 rows are
# entirely missing.  We keep them (the mask deals with it) rather than cropping,
# because 180 divides cleanly by the model's patch size.
SOUTHERNMOST_VALID_LAT = -80.0

# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------
# The 50 native GLORYS levels, in metres.
DEPTHS_NATIVE = [
    0.494025,
    1.541375,
    2.645669,
    3.819495,
    5.078224,
    6.440614,
    7.929560,
    9.572997,
    11.404999,
    13.467140,
    15.810070,
    18.495560,
    21.598820,
    25.211410,
    29.444730,
    34.434150,
    40.344051,
    47.373689,
    55.764290,
    65.807266,
    77.853851,
    92.326073,
    109.729301,
    130.666000,
    155.850700,
    186.125504,
    222.475204,
    266.040314,
    318.127411,
    380.213013,
    453.937714,
    541.088928,
    643.566772,
    763.333130,
    902.339294,
    1062.439941,
    1245.291016,
    1452.250977,
    1684.284058,
    1941.892944,
    2225.077881,
    2533.336182,
    2865.702881,
    3220.820313,
    3597.031982,
    3992.483887,
    4405.224121,
    4833.291016,
    5274.784180,
    5727.917000,
]

# Level 49 (5728 m) is 100% land/NaN everywhere -- it carries no information at
# all.  The levels we keep in the prepared files: a 14-level subset that samples
# the mixed layer, the thermocline and the deep ocean.
#
# We prepare 14 so that there is one to spare; the model loads 13 of them (see
# DEPTH_PRESETS).
PREPPED_DEPTH_INDICES = [0, 4, 10, 14, 18, 21, 24, 26, 28, 30, 32, 34, 36, 38]
PREPPED_DEPTHS = [DEPTHS_NATIVE[i] for i in PREPPED_DEPTH_INDICES]

# Model presets select a subset of the *prepared* levels at load time, so you can
# change model size without re-preparing any data.  Values are indices into
# PREPPED_DEPTHS.
#
# EVERY preset loads the same 13 levels, and that is not laziness.  geoarches'
# ArchesWeather backbone hardcodes 8 latent vertical positions in three places
# (`LinVert`, the axial attention's positional embedding, and the final reshape
# in `ArchesWeatherCondBackbone.forward`), so the model only works when
#
#     1 surface token + (n_depths padded to a multiple of 2) / 2  ==  8
#
# which means 12 or 13 depth levels and nothing else.  13 keeps the most detail.
# See `latent_z_dim` in oceanarches/backbones/ocean_embedder.py, which raises a
# readable error the moment this stops holding.
#
# Dropped: PREPPED_DEPTHS[12], 1245 m.  It sits between 902 m and 1684 m, which
# are both kept, so the vertical *range* is unchanged; the gap it leaves
# (902 -> 1684, a factor of 1.87) is no wider than gaps we already accept higher
# up (15.8 -> 29.4 is 1.86, 29.4 -> 55.8 is 1.89).  Below ~900 m the water column
# is the most vertically uniform part of the prepared range, so a level there
# carries the least independent information -- far less than dropping 5 m (which
# would blind the model inside the mixed layer) or 1684 m (which would cut the
# deep ocean off entirely).
#
# Model size is scaled by `emb_dim` and `depth_multiplier` instead.  With the
# latent depth pinned at 8 the backbone's sequence length is identical for every
# preset, so carrying 13 levels in `tiny` costs almost nothing there.
_THIRTEEN_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]

DEPTH_PRESETS: dict[str, list[int]] = {
    "tiny": list(_THIRTEEN_LEVELS),
    "small": list(_THIRTEEN_LEVELS),
    "base": list(_THIRTEEN_LEVELS),
    "large": list(_THIRTEEN_LEVELS),
}


def depths_for_preset(preset: str) -> list[float]:
    """Actual depths (metres) selected by a model preset."""
    return [PREPPED_DEPTHS[i] for i in DEPTH_PRESETS[preset]]


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variable:
    """Everything we know about one GLORYS field."""

    name: str
    long_name: str
    units: str
    group: str  # "surface" (2-D) or "level" (3-D)
    component: str  # "ocean" or "seaice"
    cmap: str = "viridis"
    diverging: bool = False
    #: Physical bounds enforced on model output when `clamp_physical_bounds` is on.
    bounds: tuple[float | None, float | None] = (None, None)
    #: Weight of this variable in the training loss (relative, within its group).
    loss_weight: float = 1.0
    #: True when the raw field is NaN over *ice-free ocean* rather than only over
    #: land.  Those NaNs mean "zero", not "no data" -- see docs/02_data_and_masking.md.
    nan_means_zero: bool = False


_VARIABLE_LIST = [
    # --- ocean, 2-D ---------------------------------------------------------
    Variable(
        "zos",
        "Sea surface height",
        "m",
        "surface",
        "ocean",
        cmap="RdBu_r",
        diverging=True,
        loss_weight=1.0,
    ),
    Variable(
        "mlotst",
        "Mixed layer depth",
        "m",
        "surface",
        "ocean",
        cmap="magma_r",
        bounds=(0.0, None),
        loss_weight=0.3,
    ),
    Variable(
        "bottomT",
        "Sea floor potential temperature",
        "degC",
        "surface",
        "ocean",
        cmap="cividis",
        loss_weight=0.3,
    ),
    # --- sea ice, 2-D -------------------------------------------------------
    Variable(
        "siconc",
        "Sea ice concentration",
        "1",
        "surface",
        "seaice",
        cmap="Blues_r",
        bounds=(0.0, 1.0),
        loss_weight=1.0,
        nan_means_zero=True,
    ),
    Variable(
        "sithick",
        "Sea ice thickness",
        "m",
        "surface",
        "seaice",
        cmap="YlGnBu",
        bounds=(0.0, None),
        loss_weight=1.0,
        nan_means_zero=True,
    ),
    # usi/vsi carry the same `area: mean where sea_ice` cell method as siconc.
    # GLORYS is inconsistent about it: until 2015-12-29 it writes an exact 0 over
    # ice-free ocean, from 2015-12-30 it writes NaN there.  Filling the NaNs with
    # 0 restores the earlier convention; without it every year from 2016 on
    # (all of val, test and holdout) would feed NaN into two input channels.
    Variable(
        "usi",
        "Sea ice eastward velocity",
        "m/s",
        "surface",
        "seaice",
        cmap="RdBu_r",
        diverging=True,
        loss_weight=0.5,
        nan_means_zero=True,
    ),
    Variable(
        "vsi",
        "Sea ice northward velocity",
        "m/s",
        "surface",
        "seaice",
        cmap="RdBu_r",
        diverging=True,
        loss_weight=0.5,
        nan_means_zero=True,
    ),
    # --- ocean, 3-D ---------------------------------------------------------
    Variable(
        "thetao",
        "Sea water potential temperature",
        "degC",
        "level",
        "ocean",
        cmap="RdYlBu_r",
        loss_weight=1.0,
    ),
    Variable(
        "so",
        "Sea water salinity",
        "1e-3",
        "level",
        "ocean",
        cmap="viridis",
        bounds=(0.0, None),
        loss_weight=1.0,
    ),
    Variable(
        "uo",
        "Eastward sea water velocity",
        "m/s",
        "level",
        "ocean",
        cmap="RdBu_r",
        diverging=True,
        loss_weight=0.5,
    ),
    Variable(
        "vo",
        "Northward sea water velocity",
        "m/s",
        "level",
        "ocean",
        cmap="RdBu_r",
        diverging=True,
        loss_weight=0.5,
    ),
]

VARIABLES: dict[str, Variable] = {v.name: v for v in _VARIABLE_LIST}

#: Canonical channel order.  DO NOT REORDER (the normalisation statistics, the
#: saved checkpoints and the metric labels all assume it).
SURFACE_VARIABLES = [v.name for v in _VARIABLE_LIST if v.group == "surface"]
LEVEL_VARIABLES = [v.name for v in _VARIABLE_LIST if v.group == "level"]
ALL_VARIABLES = SURFACE_VARIABLES + LEVEL_VARIABLES

#: Variables whose raw NaNs over ocean mean "no ice here", i.e. zero.
NAN_MEANS_ZERO = [v.name for v in _VARIABLE_LIST if v.nan_means_zero]


def variable_group(name: str) -> str:
    return VARIABLES[name].group


def surface_index(name: str) -> int:
    return SURFACE_VARIABLES.index(name)


def level_index(name: str) -> int:
    return LEVEL_VARIABLES.index(name)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentSpec:
    """Which variables one model *predicts*, and which it only *reads*.

    A component is a model specialised on part of the climate system.  It
    predicts its own prognostic variables and may additionally read variables
    belonging to other components (or to a prescribed atmosphere) as input.
    Those read-only inputs are what we call *forcing*.

    The split is what makes two independently trained models couplable: they
    exchange full physical fields on the grid, so component A's prediction can
    be handed to component B as forcing at every rollout step.
    """

    name: str
    #: Predicted (prognostic) variables -- these appear in the model's output.
    prognostic: list[str]
    #: Read-only variables taken from another component / a file / ground truth.
    forcing: list[str] = field(default_factory=list)

    # -- derived views, in canonical order ---------------------------------
    @property
    def prognostic_surface(self) -> list[str]:
        return [v for v in SURFACE_VARIABLES if v in self.prognostic]

    @property
    def prognostic_level(self) -> list[str]:
        return [v for v in LEVEL_VARIABLES if v in self.prognostic]

    @property
    def forcing_surface(self) -> list[str]:
        return [v for v in SURFACE_VARIABLES if v in self.forcing]

    @property
    def forcing_level(self) -> list[str]:
        return [v for v in LEVEL_VARIABLES if v in self.forcing]

    @property
    def input_surface(self) -> list[str]:
        """Input channel order: own variables first, then forcing."""
        return self.prognostic_surface + self.forcing_surface

    @property
    def input_level(self) -> list[str]:
        return self.prognostic_level + self.forcing_level

    # -- channel counts, for wiring the embedder --------------------------
    @property
    def n_surface_in(self) -> int:
        return len(self.input_surface)

    @property
    def n_surface_out(self) -> int:
        return len(self.prognostic_surface)

    @property
    def n_level_in(self) -> int:
        return len(self.input_level)

    @property
    def n_level_out(self) -> int:
        return len(self.prognostic_level)


OCEAN_VARIABLES = [v.name for v in _VARIABLE_LIST if v.component == "ocean"]
SEAICE_VARIABLES = [v.name for v in _VARIABLE_LIST if v.component == "seaice"]

#: Ready-made component definitions used by the shipped configs.
COMPONENTS: dict[str, ComponentSpec] = {
    # The baseline: one model that predicts everything.  No forcing needed.
    "full": ComponentSpec(name="full", prognostic=ALL_VARIABLES),
    # Ocean specialist.  Reads the sea-ice state (ice insulates and freshens the
    # ocean) but does not predict it.
    "ocean": ComponentSpec(
        name="ocean",
        prognostic=OCEAN_VARIABLES,
        forcing=SEAICE_VARIABLES,
    ),
    # Sea-ice specialist.  Predicts only 2-D ice fields; reads the ocean state,
    # which is where its forcing comes from when coupled.
    "seaice": ComponentSpec(
        name="seaice",
        prognostic=SEAICE_VARIABLES,
        forcing=OCEAN_VARIABLES,
    ),
    # Sea ice with no knowledge of the ocean at all -- the ablation that shows
    # how much the ocean state actually matters.
    "seaice_isolated": ComponentSpec(name="seaice_isolated", prognostic=SEAICE_VARIABLES),
}


def get_component(name: str) -> ComponentSpec:
    if name not in COMPONENTS:
        raise KeyError(f"Unknown component {name!r}. Available: {sorted(COMPONENTS)}")
    return COMPONENTS[name]


# ---------------------------------------------------------------------------
# Metric labelling
# ---------------------------------------------------------------------------
def surface_variable_indices(variables: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """Map surface variable name -> ``(var, level)`` index, for geoarches'
    ``LabelDictWrapper``."""
    variables = variables or SURFACE_VARIABLES
    return {name: (i, 0) for i, name in enumerate(variables)}


def level_variable_indices(
    variables: list[str] | None = None,
    depths: list[float] | None = None,
) -> dict[str, tuple[int, int]]:
    """Map ``"<var><depth>m"`` -> ``(var, level)`` index, for ``LabelDictWrapper``."""
    variables = variables or LEVEL_VARIABLES
    depths = depths or PREPPED_DEPTHS
    out = {}
    for var_idx, name in enumerate(variables):
        for lev_idx, depth in enumerate(depths):
            out[f"{name}{depth:.0f}m"] = (var_idx, lev_idx)
    return out


def headline_variable_indices(
    depths: list[float] | None = None,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """The handful of variables worth logging every step during training.

    Returns ``(surface_indices, level_indices)``.  Surface temperature and sea-ice
    concentration are the two numbers most people watch; the shallowest ocean
    level stands in for SST.
    """
    depths = depths or PREPPED_DEPTHS
    surface = {
        name: (surface_index(name), 0)
        for name in ("zos", "siconc", "sithick")
        if name in SURFACE_VARIABLES
    }
    level = {}
    for name in ("thetao", "so"):
        if name in LEVEL_VARIABLES:
            level[f"{name}{depths[0]:.0f}m"] = (level_index(name), 0)
    return surface, level
