"""
Transcription specifications for dymos phases.

Defines specs for all transcription methods: pseudospectral (GL, Radau, etc.),
shooting methods, and analytic.
"""
from __future__ import annotations

from typing import Literal, Annotated, TYPE_CHECKING
import numpy as np
from pydantic import Field, field_serializer, model_validator

from .base_spec import DymosBaseSpec

if TYPE_CHECKING:
    from dymos.transcriptions.grid_data import GridData
    from dymos.specs.phase_spec import PhaseSpec
    from dymos.specs.group_structure import GroupStructure


class TranscriptionBase(DymosBaseSpec):
    """
    Base specification for all transcription methods.

    Each transcription spec computes and stores its GridData during validation.
    """

    transcription_type: str = Field(
        ...,
        frozen=True,
        description="Type identifier for this transcription."
    )

    grid_data: 'GridData' | None = Field(
        default=None,
        exclude=True,
        description="Computed grid data for this transcription (not JSON serialized)."
    )

    def compute_grid_data(self) -> 'GridData':
        """
        Compute and return GridData for this transcription.

        Must be implemented by each transcription subclass.

        Returns
        -------
        GridData
            Grid data containing node locations, weights, and subset information.
        """
        raise NotImplementedError(f"compute_grid_data() not implemented for {self.__class__.__name__}")

    @model_validator(mode='after')
    def _compute_grid_data(self):
        """Compute grid_data during validation."""
        if self.grid_data is None:
            self.grid_data = self.compute_grid_data()
        return self

    def get_group_structure(self, phase_spec: 'PhaseSpec') -> 'GroupStructure':
        """
        Generate OpenMDAO group structure for this transcription.

        Must be implemented by each transcription subclass.

        Parameters
        ----------
        phase_spec : PhaseSpec
            The complete phase specification.

        Returns
        -------
        GroupStructure
            Contains subsystems and connections for the phase group.
        """
        raise NotImplementedError(f"get_group_structure() not implemented for {self.__class__.__name__}")


class GaussLobattoSpec(TranscriptionBase):
    """
    Specification for Gauss-Lobatto pseudospectral transcription.

    High-order collocation method using Gauss-Lobatto quadrature.
    """

    transcription_type: Literal['gauss-lobatto'] = 'gauss-lobatto'

    num_segments: int = Field(
        default=10,
        description="Number of segments for the discretization."
    )

    segment_ends: list | np.ndarray | None = Field(
        default=None,
        description="Custom segment end points in normalized time [0, 1]."
    )

    order: int | list | np.ndarray = Field(
        default=3,
        description="Polynomial order (must be odd). Can be scalar or per-segment."
    )

    compressed: bool = Field(
        default=False,
        description="If True, use compressed transcription for memory efficiency."
    )

    solve_segments: bool | Literal['forward', 'backward'] = Field(
        default=False,
        description="If True or specified, use solver-based segment convergence."
    )

    def compute_grid_data(self) -> 'GridData':
        """Compute GridData for Gauss-Lobatto transcription."""
        from dymos.transcriptions.grid_data import GaussLobattoGrid
        return GaussLobattoGrid(
            num_segments=self.num_segments,
            nodes_per_seg=self.order,
            segment_ends=self.segment_ends,
            compressed=self.compressed
        )

    @field_serializer('segment_ends', 'order')
    def serialize_arrays(self, value, _info):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return list(value)
        return value


class RadauSpec(TranscriptionBase):
    """
    Specification for Radau pseudospectral transcription.

    High-order collocation method using Radau quadrature.
    """

    transcription_type: Literal['radau'] = 'radau'

    num_segments: int = Field(
        default=10,
        description="Number of segments for the discretization."
    )

    segment_ends: list | np.ndarray | None = Field(
        default=None,
        description="Custom segment end points in normalized time [0, 1]."
    )

    order: int | list | np.ndarray = Field(
        default=3,
        description="Polynomial order (must be odd). Can be scalar or per-segment."
    )

    compressed: bool = Field(
        default=False,
        description="If True, use compressed transcription for memory efficiency."
    )

    def compute_grid_data(self) -> 'GridData':
        """Compute GridData for Radau pseudospectral transcription."""
        from dymos.transcriptions.grid_data import RadauGrid
        return RadauGrid(
            num_segments=self.num_segments,
            nodes_per_seg=np.asarray(self.order, dtype=int) + 1,
            segment_ends=self.segment_ends,
            compressed=self.compressed
        )

    @field_serializer('segment_ends', 'order')
    def serialize_arrays(self, value, _info):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return list(value)
        return value


class RadauNewSpec(TranscriptionBase):
    """
    Specification for new Radau pseudospectral transcription.

    Alternative Radau implementation with solver-based segment convergence.
    """

    transcription_type: Literal['radau-new'] = 'radau-new'

    num_segments: int = Field(
        default=10,
        description="Number of segments for the discretization."
    )

    segment_ends: list | np.ndarray | None = Field(
        default=None,
        description="Custom segment end points in normalized time [0, 1]."
    )

    order: int | list | np.ndarray = Field(
        default=3,
        description="Polynomial order (must be odd). Can be scalar or per-segment."
    )

    compressed: bool = Field(
        default=False,
        description="If True, use compressed transcription for memory efficiency."
    )

    solve_segments: bool | Literal['forward', 'backward'] = Field(
        default=False,
        description="If True or specified, use solver-based segment convergence."
    )

    def compute_grid_data(self) -> 'GridData':
        """Compute GridData for Radau pseudospectral transcription (new variant)."""
        from dymos.transcriptions.grid_data import RadauGrid
        return RadauGrid(
            num_segments=self.num_segments,
            nodes_per_seg=np.asarray(self.order, dtype=int) + 1,
            segment_ends=self.segment_ends,
            compressed=self.compressed
        )

    @field_serializer('segment_ends', 'order')
    def serialize_arrays(self, value, _info):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return list(value)
        return value


class BirkhoffSpec(TranscriptionBase):
    """
    Specification for Birkhoff pseudospectral transcription.

    High-order collocation method using Birkhoff polynomial basis.
    """

    transcription_type: Literal['birkhoff'] = 'birkhoff'

    num_nodes: int = Field(
        default=25,
        description="Total number of nodes (not segments)."
    )

    grid_type: Literal['cgl', 'lgl'] = Field(
        default='cgl',
        description="Grid type: 'cgl' (Chebyshev-Gauss-Lobatto) or 'lgl' (Legendre-Gauss-Lobatto)."
    )

    def compute_grid_data(self) -> 'GridData':
        """Compute GridData for Birkhoff pseudospectral transcription."""
        from dymos.transcriptions.grid_data import BirkhoffGrid
        return BirkhoffGrid(
            num_nodes=self.num_nodes,
            grid_type=self.grid_type
        )


class ExplicitShootingSpec(TranscriptionBase):
    """
    Specification for explicit shooting transcription.

    Uses external ODE integrator for segment propagation with multiple shooting.
    """

    transcription_type: Literal['explicit-shooting'] = 'explicit-shooting'

    num_segments: int = Field(
        default=10,
        description="Number of segments for shooting."
    )

    method: str = Field(
        default='RK45',
        description="Integration method: 'DOP853', 'RK45', 'RK23', 'BDF', 'Radau', 'LSODA'."
    )

    atol: float = Field(
        default=1.0e-9,
        description="Absolute tolerance for integration."
    )

    rtol: float = Field(
        default=1.0e-6,
        description="Relative tolerance for integration."
    )

    first_step: float | None = Field(
        default=None,
        description="First step size for integration."
    )

    max_step: float | None = Field(
        default=None,
        description="Maximum step size for integration."
    )

    propagate_derivs: bool = Field(
        default=True,
        description="If True, propagate analytical derivatives through segments."
    )

    grid: str = Field(
        default='uniform',
        description="Input grid type."
    )

    output_grid: str | None = Field(
        default=None,
        description="Output grid type (can differ from input grid)."
    )

    control_interp: str = Field(
        default='cubic',
        description="Control interpolation method: 'cubic', 'vandermonde', 'barycentric'."
    )

    times_per_seg: int | None = Field(
        default=None,
        description="Number of times per segment for dense output."
    )

    def compute_grid_data(self) -> 'GridData':
        """Compute GridData for explicit shooting transcription."""
        from dymos.transcriptions.grid_data import ShootingGrid
        return ShootingGrid(
            num_segments=self.num_segments,
            grid_type=self.grid,
            output_grid=self.output_grid
        )


class PicardShootingSpec(TranscriptionBase):
    """
    Specification for Picard shooting transcription.

    Uses Picard iteration-based shooting method.
    """

    transcription_type: Literal['picard-shooting'] = 'picard-shooting'

    num_segments: int = Field(
        default=10,
        description="Number of segments for shooting."
    )

    method: str = Field(
        default='RK45',
        description="Integration method: 'DOP853', 'RK45', 'RK23', 'BDF', 'Radau', 'LSODA'."
    )

    atol: float = Field(
        default=1.0e-9,
        description="Absolute tolerance for integration."
    )

    rtol: float = Field(
        default=1.0e-6,
        description="Relative tolerance for integration."
    )

    first_step: float | None = Field(
        default=None,
        description="First step size for integration."
    )

    max_step: float | None = Field(
        default=None,
        description="Maximum step size for integration."
    )

    propagate_derivs: bool = Field(
        default=True,
        description="If True, propagate analytical derivatives through segments."
    )

    grid: str = Field(
        default='uniform',
        description="Input grid type."
    )

    output_grid: str | None = Field(
        default=None,
        description="Output grid type (can differ from input grid)."
    )

    control_interp: str = Field(
        default='cubic',
        description="Control interpolation method: 'cubic', 'vandermonde', 'barycentric'."
    )

    times_per_seg: int | None = Field(
        default=None,
        description="Number of times per segment for dense output."
    )

    def compute_grid_data(self) -> 'GridData':
        """Compute GridData for Picard shooting transcription."""
        from dymos.transcriptions.grid_data import ShootingGrid
        return ShootingGrid(
            num_segments=self.num_segments,
            grid_type=self.grid,
            output_grid=self.output_grid
        )


class AnalyticSpec(TranscriptionBase):
    """
    Specification for analytic transcription.

    Used for phases with analytically-known solutions.
    """

    transcription_type: Literal['analytic'] = 'analytic'

    num_segments: int = Field(
        default=10,
        description="Number of segments for the discretization."
    )

    segment_ends: list | np.ndarray | None = Field(
        default=None,
        description="Custom segment end points in normalized time [0, 1]."
    )

    order: int | list | np.ndarray = Field(
        default=3,
        description="Polynomial order (must be odd). Can be scalar or per-segment."
    )

    compressed: bool = Field(
        default=False,
        description="If True, use compressed transcription for memory efficiency."
    )

    def compute_grid_data(self) -> 'GridData':
        """Compute GridData for analytic transcription."""
        from dymos.transcriptions.grid_data import GaussLobattoGrid
        return GaussLobattoGrid(
            num_segments=self.num_segments,
            nodes_per_seg=self.order,
            segment_ends=self.segment_ends,
            compressed=self.compressed
        )

    @field_serializer('segment_ends', 'order')
    def serialize_arrays(self, value, _info):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return list(value)
        return value


# Discriminated union for all transcription specs
TranscriptionSpec = Annotated[
    Union[
        GaussLobattoSpec,
        RadauSpec,
        RadauNewSpec,
        BirkhoffSpec,
        ExplicitShootingSpec,
        PicardShootingSpec,
        AnalyticSpec,
    ],
    Field(discriminator='transcription_type')
]
