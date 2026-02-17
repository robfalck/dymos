"""Pydantic model for the complete phase specification."""
from typing import List
from pydantic import BaseModel, Field

from .state_spec import StateSpec
from .control_spec import ControlSpec
from .parameter_spec import ParameterSpec
from .time_spec import TimeSpec
from .grid_spec import GridSpec


class PhaseSpec(BaseModel):
    """Complete specification for an optimal control phase.

    This is a backend-agnostic specification that contains all static
    configuration for a phase. It can be serialized to JSON/YAML for
    reproducibility.

    Examples
    --------
    >>> spec = PhaseSpec(
    ...     states=[
    ...         StateSpec(name='x', fix_initial=True, fix_final=True),
    ...         StateSpec(name='y', fix_initial=True, fix_final=True),
    ...         StateSpec(name='v', fix_initial=True),
    ...     ],
    ...     controls=[ControlSpec(name='theta')],
    ...     parameters=[ParameterSpec(name='g', value=9.80665)],
    ...     time=TimeSpec(fix_initial=True, fix_duration=False,
    ...                   duration_bounds=(0.5, 10.0)),
    ...     grid=GridSpec(num_segments=3, order=3),
    ... )
    """

    # Problem structure
    states: List[StateSpec] = Field(..., description="State variables")
    controls: List[ControlSpec] = Field(
        default_factory=list,
        description="Control variables"
    )
    parameters: List[ParameterSpec] = Field(
        default_factory=list,
        description="Parameters (static or design variables)"
    )

    # Time and grid
    time: TimeSpec = Field(..., description="Time configuration")
    grid: GridSpec = Field(..., description="Grid discretization")

    # Objective
    objective: str = Field(
        'time',
        description="What to minimize: 'time' or custom"
    )

    model_config = {'validate_assignment': True}

    def get_state_names(self) -> List[str]:
        """Get list of state names."""
        return [s.name for s in self.states]

    def get_control_names(self) -> List[str]:
        """Get list of control names."""
        return [c.name for c in self.controls]

    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return [p.name for p in self.parameters]
