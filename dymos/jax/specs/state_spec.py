"""Pydantic model for state variable specification."""
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field


class StateSpec(BaseModel):
    """Specification for a state variable."""

    name: str = Field(..., description="State variable name")
    rate_source: Optional[str] = Field(
        None,
        description="ODE output key for state rate. Defaults to '{name}_dot'"
    )
    targets: Optional[List[str]] = Field(
        None,
        description="ODE input keys where state value is needed. Defaults to [name]"
    )

    # Boundary conditions
    fix_initial: bool = Field(False, description="Fix initial value")
    fix_final: bool = Field(False, description="Fix final value")
    initial_bounds: Optional[Tuple[float, float]] = Field(
        None, description="Bounds on initial value (lower, upper)"
    )
    final_bounds: Optional[Tuple[float, float]] = Field(
        None, description="Bounds on final value (lower, upper)"
    )

    # Optional metadata
    units: Optional[str] = Field(None, description="State units")
    shape: Tuple[int, ...] = Field((), description="State shape (default scalar)")

    def get_rate_source(self) -> str:
        """Get rate source with default convention."""
        return self.rate_source or f"{self.name}_dot"

    def get_targets(self) -> List[str]:
        """Get targets with default convention."""
        return self.targets or [self.name]
