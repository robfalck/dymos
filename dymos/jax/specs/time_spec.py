"""Pydantic model for time configuration specification."""
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field


class TimeSpec(BaseModel):
    """Specification for time configuration."""

    # Fixed values
    fix_initial: bool = Field(True, description="Fix initial time")
    fix_duration: bool = Field(False, description="Fix phase duration")

    # Values
    initial_value: float = Field(0.0, description="Initial time value")
    duration_value: Optional[float] = Field(
        None,
        description="Duration value (required if fix_duration=True)"
    )

    # Bounds
    initial_bounds: Tuple[float, float] = Field(
        (0.0, 0.0),
        description="Bounds on initial time"
    )
    duration_bounds: Optional[Tuple[float, float]] = Field(
        None,
        description="Bounds on duration (required if fix_duration=False)"
    )

    # Optional metadata
    units: str = Field('s', description="Time units")
    targets: Optional[List[str]] = Field(
        None,
        description="ODE inputs for time value. Defaults to ['time']"
    )

    def get_targets(self) -> List[str]:
        """Get targets with default convention."""
        return self.targets or ['time']
