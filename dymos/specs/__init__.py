"""
Dymos Specifications System

This package provides Pydantic-based specifications for dymos Phases and Trajectories.
It enables JSON serialization, bidirectional conversion, and declarative phase/trajectory definition.

The specs follow the same pattern as OpenMDAO's specifications system, providing:
- Type-safe schema definitions with validation
- JSON serialization and deserialization
- Conversion between specs and dymos objects (spec ↔ system)
- Registry patterns for extensibility

Basic Usage
-----------

Convert a Phase to a spec::

    from dymos.specs import phase_to_spec
    import json

    spec = phase_to_spec(my_phase)
    json_str = spec.model_dump_json(exclude_none=True)

Create a Phase from a spec::

    from dymos.specs import instantiate_phase_from_spec, PhaseSpec

    spec = PhaseSpec.model_validate_json(json_str)
    phase = instantiate_phase_from_spec(spec)

Create a Trajectory from a spec::

    from dymos.specs import instantiate_trajectory_from_spec, TrajectorySpec

    spec = TrajectorySpec(
        phases=[phase_spec1, phase_spec2],
        linkages=[linkage_spec],
    )
    trajectory = instantiate_trajectory_from_spec(spec)
"""

# Base spec classes
from .base_spec import DymosBaseSpec, DymosVariableSpec

# Time specification
from .variable_spec import TimeSpec

# Variable specifications
from .variable_spec import (
    StateSpec,
    ControlSpec,
    ParameterSpec,
    TrajParameterSpec,
)

# Constraint and objective specifications
from .constraint_spec import ConstraintSpec, BoundaryConstraintSpec, PathConstraintSpec
from .objective_spec import ObjectiveSpec

# Transcription specifications
from .transcription_spec import (
    GaussLobattoSpec,
    RadauSpec,
    RadauNewSpec,
    BirkhoffSpec,
    ExplicitShootingSpec,
    PicardShootingSpec,
    AnalyticSpec,
    TranscriptionSpec,
)

# Phase and trajectory specifications
from .phase_spec import (
    PhaseSpec,
    TimeseriesOutputSpec,
    GridRefinementSpec,
    SimulateSpec,
)
from .linkage_spec import LinkageSpec
from .trajectory_spec import TrajectorySpec

# Conversion functions (spec ← system)
from .conversion import (
    transcription_to_spec,
    phase_to_spec,
    trajectory_to_spec,
)

# Instantiation functions (system ← spec)
from .instantiation import (
    instantiate_transcription_from_spec,
    instantiate_phase_from_spec,
    instantiate_trajectory_from_spec,
)

# Registry functions
from .registries import (
    get_transcription_spec_class,
    get_transcription_class,
    get_all_transcription_types,
)

# Validation utilities
from . import validation

__all__ = [
    # Base spec classes
    'DymosBaseSpec',
    'DymosVariableSpec',

    # Time specification
    'TimeSpec',

    # Variable specifications
    'StateSpec',
    'ControlSpec',
    'ParameterSpec',
    'TrajParameterSpec',

    # Constraint and objective specifications
    'ConstraintSpec',
    'BoundaryConstraintSpec',
    'PathConstraintSpec',
    'ObjectiveSpec',

    # Transcription specifications
    'TranscriptionBase',
    'GaussLobattoSpec',
    'RadauSpec',
    'RadauNewSpec',
    'BirkhoffSpec',
    'ExplicitShootingSpec',
    'PicardShootingSpec',
    'AnalyticSpec',
    'TranscriptionSpec',

    # Phase and trajectory specifications
    'PhaseSpec',
    'TimeseriesOutputSpec',
    'GridRefinementSpec',
    'SimulateSpec',
    'LinkageSpec',
    'TrajectorySpec',

    # Conversion functions
    'transcription_to_spec',
    'phase_to_spec',
    'trajectory_to_spec',

    # Instantiation functions
    'instantiate_transcription_from_spec',
    'instantiate_phase_from_spec',
    'instantiate_trajectory_from_spec',

    # Registry functions
    'get_transcription_spec_class',
    'get_transcription_class',
    'get_all_transcription_types',

    # Validation utilities
    'validation',
]
