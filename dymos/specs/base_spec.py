"""
Base specification classes for dymos specs.

This module defines the foundational Pydantic models that all dymos specs inherit from.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, field_serializer, ConfigDict


class DymosBaseSpec(BaseModel):
    """
    Base specification class for all dymos specs.

    Provides common Pydantic configuration and utilities for dymos specifications.
    """

    model_config = ConfigDict(
        extra='forbid',              # Reject extra fields
        validate_assignment=True,    # Validate on assignment
        arbitrary_types_allowed=True,  # Allow numpy types, etc.
    )


class DymosVariableSpec(DymosBaseSpec):
    """
    Base specification for dymos variables (states, controls, parameters).

    Contains fields common to all variable types.
    """

    name: str = Field(
        ...,
        description="The name of the variable."
    )

    units: str | None = Field(
        default=None,
        description="The units in which the variable is defined."
    )

    val: float | list | np.ndarray = Field(
        default=0.0,
        description="The default value of the variable."
    )

    shape: tuple | list | None = Field(
        default=None,
        description="The shape of the variable."
    )

    desc: str = Field(
        default="",
        description="A description of the variable."
    )

    opt: bool = Field(
        default=True,
        description="If True, the variable is a design variable for optimization."
    )

    lower: float | list | np.ndarray | None = Field(
        default=None,
        description="The lower bound of the variable."
    )

    upper: float | list | np.ndarray | None = Field(
        default=None,
        description="The upper bound of the variable."
    )

    ref0: float | list | np.ndarray | None = Field(
        default=None,
        description="The zero-reference value for the variable."
    )

    ref: float | list | np.ndarray | None = Field(
        default=None,
        description="The unit-reference value for the variable."
    )

    scaler: float | list | np.ndarray | None = Field(
        default=None,
        description="The scaler for the variable."
    )

    adder: float | list | np.ndarray | None = Field(
        default=None,
        description="The adder for the variable."
    )

    @field_serializer('val', 'lower', 'upper', 'ref0', 'ref', 'scaler', 'adder', 'shape')
    def serialize_arrays(self, value, _info):
        """
        Convert numpy arrays to lists for JSON serialization.
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return list(value)
        return value
