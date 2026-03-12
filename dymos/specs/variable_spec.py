"""
Variable specifications for dymos phases.

Defines specs for states, controls, parameters, and trajectory parameters.
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from pydantic import Field, field_serializer, field_validator

from dymos.specs.base_spec import DymosBaseSpec

from .base_spec import DymosVariableSpec


class StateSpec(DymosVariableSpec):
    """
    Specification for a state variable in a dymos phase.

    States are variables whose values are computed by integrating their rates
    as defined by the ODE system.
    """

    rate_source: str | None = Field(
        default=None,
        description="The source of the state rate in the ODE (e.g., 'ode.rate_of_state')."
    )

    source: str | None = Field(
        default=None,
        description="The source of the state value (for analytic solutions)."
    )

    targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE to which the state is connected."
    )

    fix_initial: bool = Field(
        default=False,
        description="If True, the initial state value is fixed."
    )

    fix_final: bool = Field(
        default=False,
        description="If True, the final state value is fixed."
    )

    initial_bounds: tuple | None = Field(
        default=None,
        description="Bounds on the initial state value as (lower, upper)."
    )

    final_bounds: tuple | None = Field(
        default=None,
        description="Bounds on the final state value as (lower, upper)."
    )

    defect_scaler: float | None = Field(
        default=None,
        description="The scaler for state defects in pseudospectral methods."
    )

    defect_ref: float | None = Field(
        default=None,
        description="The reference value for state defects instead of scaler."
    )

    continuity: bool | dict = Field(
        default=True,
        description="If True, enforce continuity at segment boundaries."
    )

    continuity_scaler: float | None = Field(
        default=None,
        description="The scaler for continuity constraints at segment boundaries."
    )

    continuity_ref: float | None = Field(
        default=None,
        description="The reference value for continuity constraints instead of scaler."
    )

    solve_segments: bool | Literal['forward', 'backward'] = Field(
        default=False,
        description="If True, use solver-based segment convergence."
    )

    input_initial: bool = Field(
        default=False,
        description="If True, the initial state value is an input."
    )

    input_final: bool = Field(
        default=False,
        description="If True, the final state value is an input."
    )


class ControlSpec(DymosVariableSpec):
    """
    Specification for a control variable in a dymos phase.

    Controls are design variables whose values are optimized during trajectory
    optimization. They can be full or polynomial controls.
    """

    control_type: Literal['full', 'polynomial'] = Field(
        default='full',
        description="The type of control: 'full' or 'polynomial'."
    )

    order: int | None = Field(
        default=None,
        description="Polynomial order when control is a reduced-order polynomial control."
    )

    targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE to which the control is connected."
    )

    rate_targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE for the control rate."
    )

    rate2_targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE for the control second derivative."
    )

    fix_initial: bool = Field(
        default=False,
        description="If True, the initial control value is fixed."
    )

    fix_final: bool = Field(
        default=False,
        description="If True, the final control value is fixed."
    )

    continuity: bool | dict = Field(
        default=True,
        description="If True, enforce continuity of control at segment boundaries."
    )

    continuity_scaler: float | None = Field(
        default=None,
        description="Scaler for control continuity constraints."
    )

    continuity_ref: float | None = Field(
        default=None,
        description="Reference value for control continuity constraints instead of scaler."
    )

    rate_continuity: bool | dict = Field(
        default=True,
        description="If True, enforce continuity of control rate at segment boundaries."
    )

    rate_continuity_scaler: float | None = Field(
        default=None,
        description="Scaler for control rate continuity constraints."
    )

    rate_continuity_ref: float | None = Field(
        default=None,
        description="Reference value for control rate continuity constraints instead of scaler."
    )

    rate2_continuity: bool | dict = Field(
        default=False,
        description="If True, enforce continuity of control second derivative at boundaries."
    )

    rate2_continuity_scaler: float | None = Field(
        default=None,
        description="Scaler for control rate2 continuity constraints."
    )

    rate2_continuity_ref: float | None = Field(
        default=None,
        description="Reference value for control rate2 continuity constraints instead of scaler."
    )

    # @field_validator('order')
    # @classmethod
    # def validate_order(cls, v, info):
    #     """Order is only valid for polynomial controls."""
    #     if v is not None and info.data.get('control_type') != 'polynomial':
    #         raise ValueError("'order' is only valid when control_type is 'polynomial'")
    #     return v


class ParameterSpec(DymosVariableSpec):
    """
    Specification for a parameter in a dymos phase.

    Parameters are design variables that are constant across a phase but can
    vary between phases in a trajectory.
    """

    targets: list[str] | None = Field(
        default=None,
        description="Targets in the ODE to which the parameter is connected."
    )

    static_targets: bool | list[str] | None = Field(
        default=None,
        description="If True/False, all targets are static/dynamic. If list, specifies static targets."
    )

    include_timeseries: bool = Field(
        default=True,
        description="If True, include this parameter in timeseries outputs."
    )


class TrajParameterSpec(ParameterSpec):
    """
    Specification for a parameter at the trajectory level.

    Trajectory parameters can have different targets in different phases.
    """

    targets: dict[str, list[str]] | None = Field(
        default=None,
        description="Mapping from phase name to list of targets in that phase's ODE."
    )

    custom_targets: dict[str, list[str]] | None = Field(
        default=None,
        description="Custom targets for phases not specified in targets dict."
    )


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
