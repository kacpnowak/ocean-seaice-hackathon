"""Data loading for GLORYS, built on geoarches' ``XarrayDataset``."""

from .variables import (
    DEPTH_PRESETS,
    LEVEL_VARIABLES,
    PREPPED_DEPTHS,
    SURFACE_VARIABLES,
    VARIABLES,
    ComponentSpec,
    get_component,
)

__all__ = [
    "DEPTH_PRESETS",
    "LEVEL_VARIABLES",
    "PREPPED_DEPTHS",
    "SURFACE_VARIABLES",
    "VARIABLES",
    "ComponentSpec",
    "get_component",
]
