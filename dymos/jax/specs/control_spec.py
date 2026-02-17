"""Pydantic model for control variable specification."""
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field


class ControlSpec(BaseModel):
    """Specification for a control variable."""

    name: str = Field(..., description="Control variable name")
    targets: Optional[List[str]] = Field(
        None,
        description="ODE input keys for control value. Defaults to [name]"
    )
    order: int = Field(
        1,
        description="Control order (1=piecewise, higher for polynomial)"
    )

    # Optional metadata
    units: Optional[str] = Field(None, description="Control units")
    shape: Tuple[int, ...] = Field((), description="Control shape (default scalar)")

    # Bounds
    lower: Optional[float] = Field(None, description="Lower bound")
    upper: Optional[float] = Field(None, description="Upper bound")

    def get_targets(self) -> List[str]:
        """Get targets with default convention."""
        return self.targets or [self.name]
