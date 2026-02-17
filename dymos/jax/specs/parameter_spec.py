"""Pydantic model for parameter specification."""
from typing import Optional, List
from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    """Specification for a parameter (constant or design variable)."""

    name: str = Field(..., description="Parameter name")
    value: float = Field(..., description="Parameter value")
    static: bool = Field(
        True,
        description="If True, doesn't vary across collocation nodes"
    )
    targets: Optional[List[str]] = Field(
        None,
        description="ODE input keys for parameter. Defaults to [name]"
    )

    # Optional metadata
    units: Optional[str] = Field(None, description="Parameter units")
    opt: bool = Field(
        False,
        description="If True, this parameter is a design variable"
    )
    lower: Optional[float] = Field(None, description="Lower bound (if opt=True)")
    upper: Optional[float] = Field(None, description="Upper bound (if opt=True)")

    def get_targets(self) -> List[str]:
        """Get targets with default convention."""
        return self.targets or [self.name]
