"""
Trajectory specification for dymos.

Defines the complete specification for a dymos Trajectory.
"""
from __future__ import annotations

from typing import Optional
from pydantic import Field

from .base_spec import DymosBaseSpec
from .phase_spec import PhaseSpec
from .variable_spec import TrajParameterSpec
from .linkage_spec import LinkageSpec


class TrajectorySpec(DymosBaseSpec):
    """
    Complete specification for a dymos Trajectory.

    Contains all configuration needed to recreate a trajectory through instantiation.
    """

    name: str = Field(
        default='traj',
        description="Name of the trajectory."
    )

    phases: list[PhaseSpec] = Field(
        default_factory=list,
        description="Phases in the trajectory."
    )

    parameters: list[TrajParameterSpec] = Field(
        default_factory=list,
        description="Trajectory-level parameters shared across phases."
    )

    linkages: list[LinkageSpec] = Field(
        default_factory=list,
        description="Phase linkage specifications."
    )

    parallel_phases: bool = Field(
        default=False,
        description="If True, phases are executed in parallel (where possible)."
    )

    auto_solvers: bool = Field(
        default=True,
        description="If True, automatically configure solvers."
    )
