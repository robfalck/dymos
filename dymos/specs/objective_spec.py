"""
Objective specification for dymos phases.
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from pydantic import Field, field_serializer

from .base_spec import DymosBaseSpec


class ObjectiveSpec(DymosBaseSpec):
    """
    Specification for an objective in a dymos phase.

    Objectives are variables to be minimized or maximized during optimization.
    """

    name: str = Field(
        ...,
        description="The name of the objective."
    )

    loc: Literal['initial', 'final', 'path'] = Field(
        ...,
        description="Location of the objective: 'initial', 'final', or 'path'."
    )

    index: int | None = Field(
        default=None,
        description="Index if the objective is a scalar within a vector output."
    )

    scaler: float | None = Field(
        default=None,
        description="The scaler for the objective."
    )

    adder: float | None = Field(
        default=None,
        description="The adder for the objective."
    )

    ref0: float | None = Field(
        default=None,
        description="The zero-reference value for the objective."
    )

    ref: float | None = Field(
        default=None,
        description="The unit-reference value for the objective."
    )

    units: str | None = Field(
        default=None,
        description="The units of the objective."
    )

    @field_serializer('scaler', 'adder', 'ref0', 'ref')
    def serialize_scalars(self, value, _info):
        """
        Handle numpy scalars for JSON serialization.
        """
        if isinstance(value, np.generic):
            return float(value)
        return value
