"""
Transcription specifications for dymos phases.

Defines specs for all transcription methods: pseudospectral (GL, Radau, etc.),
shooting methods, and analytic.
"""
from __future__ import annotations

from typing import Literal, Annotated, TYPE_CHECKING, Union
import numpy as np
from pydantic import Field, field_serializer, field_validator

from dymos.specs.phase_spec import PhaseSpec

from om4.core.group import Group
from .base_spec import DymosBaseSpec


if TYPE_CHECKING:
    from dymos.transcriptions.grid_data import GridData
    from dymos.specs.phase_spec import PhaseSpec
    # from dymos.specs.group_structure import GroupStructure


class TranscriptionSpecBase(DymosBaseSpec):
    """
    Base specification for all transcription methods.

    Each transcription spec computes and stores its GridData during validation.
    """

    # type: str = Field(
    #     ...,
    #     frozen=True,
    #     description="Type identifier for this transcription."
    # )

    num_segments: int = Field(
        default=...,
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

    nodes_per_seg: int | list | np.ndarray | None = Field(
        default=3,
        ge=3,
        description="Number of nodes in each segment."
    )

    compressed: bool = Field(
        default=False,
        description="If True, use compressed transcription for memory efficiency."
    )

    solve_segments: bool | Literal['forward', 'backward'] = Field(
        default=False,
        description="If True or specified, use solver-based segment convergence."
    )

    # def compute_grid_data(self) -> 'GridData':
    #     """
    #     Compute and return GridData for this transcription.

    #     Must be implemented by each transcription subclass.

    #     Returns
    #     -------
    #     GridData
    #         Grid data containing node locations, weights, and subset information.
    #     """
    #     raise NotImplementedError(f"compute_grid_data() not implemented for {self.__class__.__name__}")

    # @model_validator(mode='after')
    # def _compute_grid_data(self):
    #     """Compute grid_data during validation."""
    #     if self.grid_data is None:
    #         self.grid_data = self.compute_grid_data()
    #     return self

    # def get_group_structure(self, phase_spec: 'PhaseSpec') -> 'GroupStructure':
    #     """
    #     Generate OpenMDAO group structure for this transcription.

    #     Must be implemented by each transcription subclass.

    #     Parameters
    #     ----------
    #     phase_spec : PhaseSpec
    #         The complete phase specification.

    #     Returns
    #     -------
    #     GroupStructure
    #         Contains subsystems and connections for the phase group.
    #     """
    #     raise NotImplementedError(f"get_group_structure() not implemented for {self.__class__.__name__}")

    def build_system(self, phase_spec: 'PhaseSpec'):
        raise NotImplementedError(f'Transcription {self.__class__} has not implemented build_system.')


class GaussLobattoSpec(TranscriptionSpecBase):
    """
    Specification for Gauss-Lobatto pseudospectral transcription.

    High-order collocation method using Gauss-Lobatto quadrature.
    """

    type: Literal['gauss-lobatto'] = 'gauss-lobatto'

    @field_validator('nodes_per_seg')
    @classmethod
    def check_if_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError('ensure this value is an odd number')
        return v

    # def compute_grid_data(self) -> 'GridData':
    #     """Compute GridData for Gauss-Lobatto transcription."""
    #     from dymos.transcriptions.grid_data import GaussLobattoGrid
    #     return GaussLobattoGrid(
    #         num_segments=self.num_segments,
    #         nodes_per_seg=self.order,
    #         segment_ends=self.segment_ends,
    #         compressed=self.compressed
    #     )

    # @field_serializer('segment_ends', 'order')
    # def serialize_arrays(self, value, _info):
    #     """Convert numpy arrays to lists for JSON serialization."""
    #     if isinstance(value, np.ndarray):
    #         return value.tolist()
    #     elif isinstance(value, (list, tuple)):
    #         return list(value)
    #     return value


class RadauSpec(TranscriptionSpecBase):
    """
    Specification for Radau pseudospectral transcription.

    High-order collocation method using Radau quadrature.
    """

    type: Literal['radau'] = 'radau'

    def build_system(self, phase_spec: PhaseSpec):
        subsystems = ['grid', 'time', 'controls', 'ode_iter_group', 'boundary_vals']
        ode_iter_group = Group(subsystems=[''])
        phase_group = Group(promotes_inputs=[], promotes_outputs=[], connections=[],
                            subsystems=[], input_defaults=[], nonlinear_solver=None,
                            linear_solver=None, auto_order=True)


class RadauNewSpec(TranscriptionSpecBase):
    """
    Specification for new Radau pseudospectral transcription.

    Alternative Radau implementation with solver-based segment convergence.
    """

    type: Literal['radau-new'] = 'radau-new'


class BirkhoffSpec(TranscriptionSpecBase):
    """
    Specification for Birkhoff pseudospectral transcription.

    High-order collocation method using Birkhoff polynomial basis.
    """

    type: Literal['birkhoff'] = 'birkhoff'


class ExplicitShootingSpec(TranscriptionSpecBase):
    """
    Specification for explicit shooting transcription.

    Uses external ODE integrator for segment propagation with multiple shooting.
    """

    type: Literal['explicit-shooting'] = 'explicit-shooting'

    # num_segments: int = Field(
    #     default=10,
    #     description="Number of segments for shooting."
    # )

    # method: str = Field(
    #     default='RK45',
    #     description="Integration method: 'DOP853', 'RK45', 'RK23', 'BDF', 'Radau', 'LSODA'."
    # )

    # atol: float = Field(
    #     default=1.0e-9,
    #     description="Absolute tolerance for integration."
    # )

    # rtol: float = Field(
    #     default=1.0e-6,
    #     description="Relative tolerance for integration."
    # )

    # first_step: float | None = Field(
    #     default=None,
    #     description="First step size for integration."
    # )

    # max_step: float | None = Field(
    #     default=None,
    #     description="Maximum step size for integration."
    # )

    # propagate_derivs: bool = Field(
    #     default=True,
    #     description="If True, propagate analytical derivatives through segments."
    # )

    # grid: str = Field(
    #     default='uniform',
    #     description="Input grid type."
    # )

    # output_grid: str | None = Field(
    #     default=None,
    #     description="Output grid type (can differ from input grid)."
    # )

    # control_interp: str = Field(
    #     default='cubic',
    #     description="Control interpolation method: 'cubic', 'vandermonde', 'barycentric'."
    # )

    # times_per_seg: int | None = Field(
    #     default=None,
    #     description="Number of times per segment for dense output."
    # )

    # def compute_grid_data(self) -> 'GridData':
    #     """Compute GridData for explicit shooting transcription."""
    #     from dymos.transcriptions.grid_data import ShootingGrid
    #     return ShootingGrid(
    #         num_segments=self.num_segments,
    #         grid_type=self.grid,
    #         output_grid=self.output_grid
    #     )


class PicardShootingSpec(TranscriptionSpecBase):
    """
    Specification for Picard shooting transcription.

    Uses Picard iteration-based shooting method.
    """

    type: Literal['picard-shooting'] = 'picard-shooting'

    # num_segments: int = Field(
    #     default=10,
    #     description="Number of segments for shooting."
    # )

    # method: str = Field(
    #     default='RK45',
    #     description="Integration method: 'DOP853', 'RK45', 'RK23', 'BDF', 'Radau', 'LSODA'."
    # )

    # atol: float = Field(
    #     default=1.0e-9,
    #     description="Absolute tolerance for integration."
    # )

    # rtol: float = Field(
    #     default=1.0e-6,
    #     description="Relative tolerance for integration."
    # )

    # first_step: float | None = Field(
    #     default=None,
    #     description="First step size for integration."
    # )

    # max_step: float | None = Field(
    #     default=None,
    #     description="Maximum step size for integration."
    # )

    # propagate_derivs: bool = Field(
    #     default=True,
    #     description="If True, propagate analytical derivatives through segments."
    # )

    # grid: str = Field(
    #     default='uniform',
    #     description="Input grid type."
    # )

    # output_grid: str | None = Field(
    #     default=None,
    #     description="Output grid type (can differ from input grid)."
    # )

    # control_interp: str = Field(
    #     default='cubic',
    #     description="Control interpolation method: 'cubic', 'vandermonde', 'barycentric'."
    # )

    # times_per_seg: int | None = Field(
    #     default=None,
    #     description="Number of times per segment for dense output."
    # )

    # def compute_grid_data(self) -> 'GridData':
    #     """Compute GridData for Picard shooting transcription."""
    #     from dymos.transcriptions.grid_data import ShootingGrid
    #     return ShootingGrid(
    #         num_segments=self.num_segments,
    #         grid_type=self.grid,
    #         output_grid=self.output_grid
    #     )


class AnalyticSpec(TranscriptionSpecBase):
    """
    Specification for analytic transcription.

    Used for phases with analytically-known solutions.
    """

    type: Literal['analytic'] = 'analytic'

    # num_segments: int = Field(
    #     default=10,
    #     description="Number of segments for the discretization."
    # )

    # segment_ends: list | np.ndarray | None = Field(
    #     default=None,
    #     description="Custom segment end points in normalized time [0, 1]."
    # )

    # order: int | list | np.ndarray = Field(
    #     default=3,
    #     description="Polynomial order (must be odd). Can be scalar or per-segment."
    # )

    # compressed: bool = Field(
    #     default=False,
    #     description="If True, use compressed transcription for memory efficiency."
    # )

    # def compute_grid_data(self) -> 'GridData':
    #     """Compute GridData for analytic transcription."""
    #     from dymos.transcriptions.grid_data import GaussLobattoGrid
    #     return GaussLobattoGrid(
    #         num_segments=self.num_segments,
    #         nodes_per_seg=self.order,
    #         segment_ends=self.segment_ends,
    #         compressed=self.compressed
    #     )

    # @field_serializer('segment_ends', 'order')
    # def serialize_arrays(self, value, _info):
    #     """Convert numpy arrays to lists for JSON serialization."""
    #     if isinstance(value, np.ndarray):
    #         return value.tolist()
    #     elif isinstance(value, (list, tuple)):
    #         return list(value)
    #     return value


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
    Field(discriminator='type')
]
