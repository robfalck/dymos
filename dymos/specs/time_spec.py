"""
Time specification for dymos phases.
"""
from __future__ import annotations

import numpy as np
from pydantic import Field, field_serializer

from .base_spec import DymosBaseSpec


class TimeSpec(DymosBaseSpec):
    """
    Specification for time options in a phase.

    Defines how time is configured in a dymos phase, including bounds,
    scaling, and targets.
    """

    name: str = Field(
        default='time',
        description="The name of the time variable."
    )

    units: str | None = Field(
        default=None,
        description="The units in which time is defined."
    )

    fix_initial: bool = Field(
        default=False,
        description="If True, the initial time value is fixed."
    )

    fix_duration: bool = Field(
        default=False,
        description="If True, the phase duration is fixed."
    )

    input_initial: bool = Field(
        default=False,
        description="If True, the initial time is an input."
    )

    input_duration: bool = Field(
        default=False,
        description="If True, the phase duration is an input."
    )

    initial_val: float | list | np.ndarray = Field(
        default=0.0,
        description="The default value of the initial time."
    )

    initial_bounds: tuple | None = Field(
        default=None,
        description="The bounds on the initial time value as (lower, upper)."
    )

    initial_scaler: float | None = Field(
        default=None,
        description="The scaler for the initial time."
    )

    initial_adder: float | None = Field(
        default=None,
        description="The adder for the initial time."
    )

    initial_ref0: float | None = Field(
        default=None,
        description="The zero-reference value for the initial time."
    )

    initial_ref: float | None = Field(
        default=None,
        description="The unit-reference value for the initial time."
    )

    duration_val: float | list | np.ndarray = Field(
        default=1.0,
        description="The default value of the phase duration."
    )

    duration_bounds: tuple | None = Field(
        default=None,
        description="The bounds on the phase duration as (lower, upper)."
    )

    duration_scaler: float | None = Field(
        default=None,
        description="The scaler for the phase duration."
    )

    duration_adder: float | None = Field(
        default=None,
        description="The adder for the phase duration."
    )

    duration_ref0: float | None = Field(
        default=None,
        description="The zero-reference value for the phase duration."
    )

    duration_ref: float | None = Field(
        default=None,
        description="The unit-reference value for the phase duration."
    )

    targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE for time variable."
    )

    time_phase_targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE for time_phase variable."
    )

    t_initial_targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE for t_initial variable."
    )

    t_duration_targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE for t_duration variable."
    )

    dt_dstau_targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE for dt_dstau variable."
    )

    @field_serializer('initial_val', 'duration_val', 'initial_bounds', 'duration_bounds')
    def serialize_arrays(self, value, _info):
        """
        Convert numpy arrays to lists for JSON serialization.
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return list(value)
        return value
