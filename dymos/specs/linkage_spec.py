"""
Phase linkage specification for dymos trajectories.
"""
from __future__ import annotations

from typing import Literal
from pydantic import Field

from .base_spec import DymosBaseSpec


class LinkageSpec(DymosBaseSpec):
    """
    Specification for linking variables between phases in a trajectory.

    Defines how variables are connected between consecutive phases.
    """

    phase_from: str = Field(
        ...,
        description="Name of the source phase."
    )

    phase_to: str = Field(
        ...,
        description="Name of the target phase."
    )

    vars_from: list[str] = Field(
        ...,
        description="List of variable names to link from source phase."
    )

    vars_to: list[str] | None = Field(
        default=None,
        description="List of variable names to link to in target phase (defaults to vars_from)."
    )

    loc_from: Literal['final'] = Field(
        default='final',
        description="Location in source phase (currently only 'final' supported)."
    )

    loc_to: Literal['initial'] = Field(
        default='initial',
        description="Location in target phase (currently only 'initial' supported)."
    )

    connected: bool = Field(
        default=True,
        description="If True, create a connected link; if False, add continuity constraint."
    )

    sign: float = Field(
        default=1.0,
        description="Sign multiplier for constraint (-1 or 1)."
    )
