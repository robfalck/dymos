from typing import Optional, Any
from pydantic import BaseModel, Field


class TimeOptions(BaseModel):
    units: Optional[str] = Field(default=None, description='Units for time')
    targets: list[str] = Field(
        default_factory=list, description='ODE inputs that should be connected to time'
    )
    fix_initial: bool = Field(
        default=False, description='If True, fix the initial time'
    )
    fix_duration: bool = Field(default=False, description='If True, fix the duration')
    initial_val: float = Field(default=0.0, description='Initial value for time')
    duration_val: float = Field(default=1.0, description='Initial value for duration')


class StateOptions(BaseModel):
    name: str = Field(..., description='The name of the state variable')
    units: Optional[str] = Field(default=None, description='Units for the state')
    rate_source: str = Field(
        ..., description='The ODE output that provides the rate of this state'
    )
    targets: list[str] = Field(
        default_factory=list,
        description='ODE inputs that should be connected to this state',
    )
    fix_initial: bool = Field(
        default=False, description='If True, fix the initial state value'
    )
    fix_final: bool = Field(
        default=False, description='If True, fix the final state value'
    )
    initial_val: Any = Field(
        default=0.0, description='Initial value/guess for the state'
    )
    shape: tuple[int, ...] = Field(
        default=(),
        description='The shape of the state variable at a single point in time',
    )


class ControlOptions(BaseModel):
    name: str = Field(..., description='The name of the control variable')
    units: Optional[str] = Field(default=None, description='Units for the control')
    targets: list[str] = Field(
        default_factory=list,
        description='ODE inputs that should be connected to this control',
    )
    opt: bool = Field(
        default=True, description='If True, the control is a design variable'
    )
    val: Any = Field(default=0.0, description='Initial value/guess for the control')
    shape: tuple[int, ...] = Field(
        default=(),
        description='The shape of the control variable at a single point in time',
    )


class ParameterOptions(BaseModel):
    name: str = Field(..., description='The name of the parameter')
    units: Optional[str] = Field(default=None, description='Units for the parameter')
    targets: list[str] = Field(
        default_factory=list,
        description='ODE inputs that should be connected to this parameter',
    )
    opt: bool = Field(
        default=False, description='If True, the parameter is a design variable'
    )
    val: Any = Field(default=0.0, description='Initial value/guess for the parameter')
    shape: tuple[int, ...] = Field(default=(), description='The shape of the parameter')
    static_target: bool = Field(default=False, description='Static vs time series')
