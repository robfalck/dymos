"""Pydantic model for grid discretization specification."""
from typing import Optional, List
from pydantic import BaseModel, Field


class GridSpec(BaseModel):
    """Specification for grid discretization."""

    num_segments: int = Field(..., ge=1, description="Number of segments")
    order: int = Field(3, ge=1, description="Polynomial order per segment")
    transcription: str = Field(
        'radau',
        description="Transcription type (radau, gauss-lobatto, etc.)"
    )

    # Optional: segment-specific configuration
    segment_ends: Optional[List[float]] = Field(
        None,
        description=(
            "Custom segment boundaries in phase tau [-1, 1]. "
            "If None, use uniform spacing."
        )
    )
