"""
Phase specification for dymos.

Defines the complete specification for a dymos Phase.
"""
from __future__ import annotations

import numpy as np
from pydantic import Field, field_serializer, field_validator

from .base_spec import DymosBaseSpec
from .time_spec import TimeSpec
from .variable_spec import StateSpec, ControlSpec, ParameterSpec
from .constraint_spec import BoundaryConstraintSpec, PathConstraintSpec
from .objective_spec import ObjectiveSpec
from .transcription_spec import TranscriptionSpec

# Import OpenMDAO spec types
try:
    from openmdao.specs.system_spec import SystemSpec
    from openmdao.specs.component_spec import OMExplicitComponentSpec
except ImportError:
    SystemSpec = None
    OMExplicitComponentSpec = None


class TimeseriesOutputSpec(DymosBaseSpec):
    """
    Specification for a timeseries output variable.
    """

    name: str = Field(
        ...,
        description="Name of the output variable in the phase."
    )

    output_name: str | None = Field(
        default=None,
        description="Name of the variable in the timeseries output (defaults to name)."
    )

    units: str | None = Field(
        default=None,
        description="Units of the output."
    )

    shape: tuple | list | None = Field(
        default=None,
        description="Shape of the output."
    )

    timeseries: str | None = Field(
        default=None,
        description="Name of the timeseries to add this output to."
    )

    @field_serializer('shape')
    def serialize_arrays(self, value, _info):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return list(value)
        return value


class GridRefinementSpec(DymosBaseSpec):
    """
    Specification for grid refinement options.
    """

    refine: bool = Field(
        default=False,
        description="If True, enable adaptive grid refinement."
    )

    tolerance: float = Field(
        default=1.0e-6,
        description="Tolerance for grid refinement."
    )

    min_order: int = Field(
        default=3,
        description="Minimum polynomial order for refined segments."
    )

    max_order: int = Field(
        default=12,
        description="Maximum polynomial order for refined segments."
    )

    smoothness_factor: float = Field(
        default=1.0,
        description="Smoothness factor for order changes between segments."
    )


class SimulateSpec(DymosBaseSpec):
    """
    Specification for simulation options.
    """

    method: str = Field(
        default='RK45',
        description="Integration method for simulation."
    )

    atol: float = Field(
        default=1.0e-9,
        description="Absolute tolerance for simulation."
    )

    rtol: float = Field(
        default=1.0e-6,
        description="Relative tolerance for simulation."
    )

    first_step: float | None = Field(
        default=None,
        description="First step size."
    )

    max_step: float | None = Field(
        default=None,
        description="Maximum step size."
    )

    times_per_seg: int = Field(
        default=10,
        description="Number of time steps per segment."
    )


class PhaseSpec(DymosBaseSpec):
    """
    Complete specification for a dymos Phase.

    Contains all configuration needed to recreate a phase through instantiation.
    """

    name: str = Field(
        ...,
        description="Name of the phase."
    )

    ode_spec: OMExplicitComponentSpec | dict = Field(
        ...,
        description=(
            "ODE component specification. Can be either:\n"
            "1. OMExplicitComponentSpec: Complete spec with inputs, outputs, and options. "
            "   Use shape_by_conn=True for all variables - shapes will be inferred from "
            "   connections during setup. Do NOT include 'num_nodes' in options.\n"
            "2. dict (deprecated): Legacy dict format with 'path' and 'init_kwargs' keys.\n"
            "NOTE: num_nodes is computed automatically by dymos based on the "
            "transcription grid structure and is injected during to_group_spec()."
        )
    )

    transcription: TranscriptionSpec = Field(
        ...,
        description="Transcription method specification."
    )

    time_options: TimeSpec = Field(
        default_factory=TimeSpec,
        description="Time options for the phase."
    )

    states: list[StateSpec] = Field(
        default_factory=list,
        description="State variables in the phase."
    )

    controls: list[ControlSpec] = Field(
        default_factory=list,
        description="Control variables in the phase."
    )

    parameters: list[ParameterSpec] = Field(
        default_factory=list,
        description="Parameter variables in the phase."
    )

    boundary_constraints: list[BoundaryConstraintSpec] = Field(
        default_factory=list,
        description="Boundary constraints (initial and final)."
    )

    path_constraints: list[PathConstraintSpec] = Field(
        default_factory=list,
        description="Path constraints (at all nodes)."
    )

    objectives: list[ObjectiveSpec] = Field(
        default_factory=list,
        description="Objectives for optimization."
    )

    timeseries_outputs: list[TimeseriesOutputSpec] = Field(
        default_factory=list,
        description="Variables to include in timeseries outputs."
    )

    refine_options: GridRefinementSpec | None = Field(
        default=None,
        description="Grid refinement options."
    )

    simulate_options: SimulateSpec | None = Field(
        default=None,
        description="Simulation options."
    )

    auto_solvers: bool = Field(
        default=True,
        description="If True, automatically configure solvers."
    )

    @field_validator('ode_spec')
    @classmethod
    def validate_ode_spec(cls, v):
        """Validate ode_spec and ensure num_nodes is not included."""
        # Handle OMExplicitComponentSpec
        if hasattr(v, 'options'):  # OMExplicitComponentSpec
            if 'num_nodes' in v.options:
                raise ValueError(
                    "ode_spec.options should not contain 'num_nodes'. "
                    "This value is computed automatically by dymos based on the "
                    "transcription grid structure and is injected during to_group_spec()."
                )
            return v

        # Handle dict (legacy format)
        if isinstance(v, dict):
            # Check for num_nodes in init_kwargs
            init_kwargs = v.get('init_kwargs', {})
            if isinstance(init_kwargs, dict) and 'num_nodes' in init_kwargs:
                raise ValueError(
                    "ode_spec['init_kwargs'] should not contain 'num_nodes'. "
                    "This value is computed automatically by dymos based on the "
                    "transcription grid structure (num_segments, order, etc.). "
                    "Only ODE-specific parameters (e.g., 'static_gravity') should be in init_kwargs."
                )
            return v

        raise ValueError(
            f"ode_spec must be either OMExplicitComponentSpec or dict, got {type(v)}"
        )

    @field_validator('states')
    @classmethod
    def validate_states(cls, v, info):
        """Validate that states have rate_source (unless Analytic transcription)."""
        transcription_type = info.data.get('transcription', {}).get('transcription_type')
        if transcription_type and transcription_type != 'analytic':
            for state in v:
                if state.rate_source is None:
                    raise ValueError(
                        f"State '{state.name}' must have rate_source specified "
                        f"for non-analytic transcriptions"
                    )
        return v
