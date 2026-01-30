"""
Constraint specifications for dymos phases.

Defines specs for boundary and path constraints.
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from pydantic import Field, field_validator, field_serializer

from .base_spec import DymosBaseSpec


class ConstraintSpec(DymosBaseSpec):
    """
    Base specification for constraints in a dymos phase.
    """

    name: str = Field(
        ...,
        description="The name of the constraint."
    )

    constraint_name: str | None = Field(
        default=None,
        description="The name of the constraint in the problem (if different from name)."
    )

    constraint_path: str | None = Field(
        default=None,
        description="The path to the constraint in the OpenMDAO model."
    )

    lower: float | list | np.ndarray | None = Field(
        default=None,
        description="The lower bound on the constraint."
    )

    upper: float | list | np.ndarray | None = Field(
        default=None,
        description="The upper bound on the constraint."
    )

    equals: float | list | np.ndarray | None = Field(
        default=None,
        description="The equality value for the constraint (mutually exclusive with lower/upper)."
    )

    scaler: float | list | np.ndarray | None = Field(
        default=None,
        description="The scaler for the constraint."
    )

    adder: float | list | np.ndarray | None = Field(
        default=None,
        description="The adder for the constraint."
    )

    ref0: float | list | np.ndarray | None = Field(
        default=None,
        description="The zero-reference value for the constraint."
    )

    ref: float | list | np.ndarray | None = Field(
        default=None,
        description="The unit-reference value for the constraint."
    )

    indices: list[int] | np.ndarray | None = Field(
        default=None,
        description="Indices to apply if the constraint is vector-valued."
    )

    shape: tuple | list | None = Field(
        default=None,
        description="The shape of the constraint output."
    )

    units: str | None = Field(
        default=None,
        description="The units of the constraint."
    )

    linear: bool = Field(
        default=False,
        description="If True, the constraint is linear."
    )

    flat_indices: bool = Field(
        default=False,
        description="If True, indices are applied in flattened form."
    )

    @field_validator('equals')
    @classmethod
    def validate_equals(cls, v, info):
        """equals is mutually exclusive with lower and upper."""
        if v is not None:
            if info.data.get('lower') is not None or info.data.get('upper') is not None:
                raise ValueError("'equals' is mutually exclusive with 'lower' and 'upper'")
        return v

    @field_validator('lower', 'upper', 'equals', 'scaler', 'adder', 'ref0', 'ref', 'indices', 'shape')
    @classmethod
    def check_has_constraint_value(cls, v, info):
        """At least one of lower, upper, or equals must be specified."""
        # This check happens in model_validator
        return v

    @field_serializer('lower', 'upper', 'equals', 'scaler', 'adder', 'ref0', 'ref', 'indices', 'shape')
    def serialize_arrays(self, value, _info):
        """
        Convert numpy arrays to lists for JSON serialization.
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return list(value)
        return value


class BoundaryConstraintSpec(ConstraintSpec):
    """
    Specification for a boundary constraint in a dymos phase.

    Boundary constraints are applied at the initial or final time of a phase.
    """

    loc: Literal['initial', 'final'] = Field(
        ...,
        description="Location of the boundary constraint: 'initial' or 'final'."
    )


class PathConstraintSpec(ConstraintSpec):
    """
    Specification for a path constraint in a dymos phase.

    Path constraints are applied at all nodes in the discretized phase.
    """

    loc: Literal['path'] = Field(
        default='path',
        description="Location of the path constraint."
    )
