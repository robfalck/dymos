"""
Builder functions for creating OpenMDAO component specs for dymos components.

These builders create complete OMExplicitComponentSpec objects for each dymos
component type, including proper input/output specifications with shapes, units,
and other metadata.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from dymos.transcriptions.grid_data import GridData
    from dymos.specs.variable_spec import StateSpec, ControlSpec, ParameterSpec
    from dymos.specs.phase_spec import PhaseSpec
    from dymos.specs.time_spec import TimeSpec
    from openmdao.specs.component_spec import OMExplicitComponentSpec


def build_time_comp_spec(grid_data: GridData, time_options: TimeSpec) -> OMExplicitComponentSpec:
    """
    Build spec for TimeComp.

    TimeComp computes time and time_phase from t_initial and t_duration.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    time_options : TimeSpec
        Time options from the phase spec.

    Returns
    -------
    OMExplicitComponentSpec
        Component spec for TimeComp.
    """
    from openmdao.specs.component_spec import OMExplicitComponentSpec, OMInputSpec, OMOutputSpec

    # Get number of time nodes (all nodes for Gauss-Lobatto)
    num_nodes = grid_data.subset_num_nodes['all']

    spec = OMExplicitComponentSpec(
        name='TimeComp',
        class_name='dymos.phases.components.time_comp.TimeComp',
        options={
            'num_segments': grid_data.num_segments,
            'node_ptau': grid_data.node_ptau,
            't_initial_targets': time_options.t_initial_targets,
            't_duration_targets': time_options.t_duration_targets,
        },
        inputs=[
            OMInputSpec(name='t_initial', shape=(1,), units=time_options.units),
            OMInputSpec(name='t_duration', shape=(1,), units=time_options.units),
        ],
        outputs=[
            OMOutputSpec(name='time', shape=(num_nodes,), units=time_options.units),
            OMOutputSpec(name='time_phase', shape=(num_nodes,), units=time_options.units),
        ]
    )
    return spec


def build_parameter_comp_spec(parameters: list[ParameterSpec] | None = None) -> OMExplicitComponentSpec:
    """
    Build spec for ParameterComp.

    ParameterComp provides design parameters that are constant across all nodes.

    Parameters
    ----------
    parameters : list[ParameterSpec], optional
        List of parameter specs.

    Returns
    -------
    OMExplicitComponentSpec
        Component spec for ParameterComp.
    """
    from openmdao.specs.component_spec import OMExplicitComponentSpec, OMInputSpec

    inputs = []
    if parameters:
        for param in parameters:
            shape = param.shape if param.shape else (1,)
            inputs.append(
                OMInputSpec(
                    name=param.name,
                    shape=shape,
                    units=param.units,
                    val=param.val if param.val is not None else 0.0
                )
            )

    spec = OMExplicitComponentSpec(
        name='ParameterComp',
        class_name='dymos.phases.components.parameter_comp.ParameterComp',
        inputs=inputs
    )
    return spec


def build_control_interpolation_comp_spec(
    grid_data: GridData,
    controls: list[ControlSpec] | None = None
) -> OMExplicitComponentSpec:
    """
    Build spec for ControlInterpComp.

    ControlInterpComp interpolates control values from nodes to collocation nodes.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    controls : list[ControlSpec], optional
        List of control specs.

    Returns
    -------
    OMExplicitComponentSpec
        Component spec for ControlInterpComp.
    """
    from openmdao.specs.component_spec import OMExplicitComponentSpec, OMInputSpec, OMOutputSpec

    inputs = []
    outputs = []

    if controls:
        for control in controls:
            # Determine number of control nodes
            num_control_nodes = grid_data.subset_num_nodes.get('control_nodes', grid_data.subset_num_nodes['all'])
            num_col_nodes = grid_data.subset_num_nodes.get('col', grid_data.subset_num_nodes['all'])

            shape = control.shape if control.shape else (1,)

            # Input: control values at control nodes
            inputs.append(
                OMInputSpec(
                    name=f'controls:{control.name}',
                    shape=(num_control_nodes,) if shape == (1,) else (num_control_nodes, *shape),
                    units=control.units
                )
            )

            # Output: interpolated control values at collocation nodes
            outputs.append(
                OMOutputSpec(
                    name=f'control_interp:{control.name}',
                    shape=(num_col_nodes,) if shape == (1,) else (num_col_nodes, *shape),
                    units=control.units
                )
            )

    spec = OMExplicitComponentSpec(
        name='ControlInterpComp',
        class_name='dymos.phases.components.control_interp_comp.ControlInterpComp',
        options={
            'control_interp': 'cubic',  # Default, can be overridden
        },
        inputs=inputs,
        outputs=outputs
    )
    return spec


def build_state_independents_spec(
    grid_data: GridData,
    states: list[StateSpec] | None = None
) -> OMExplicitComponentSpec:
    """
    Build spec for IndepVarComp with state initial values.

    This component provides the independent state variables that the phase
    optimizes.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    states : list[StateSpec], optional
        List of state specs.

    Returns
    -------
    OMExplicitComponentSpec
        Component spec for IndepVarComp (states).
    """
    from openmdao.specs.component_spec import OMExplicitComponentSpec, OMOutputSpec

    outputs = []

    if states:
        for state in states:
            # Number of state nodes (typically all nodes)
            num_state_nodes = grid_data.subset_num_nodes['all']

            shape = state.shape if state.shape else (1,)
            outputs.append(
                OMOutputSpec(
                    name=f'states:{state.name}',
                    shape=(num_state_nodes,) if shape == (1,) else (num_state_nodes, *shape),
                    units=state.units,
                    val=state.val if state.val is not None else 0.0
                )
            )

    spec = OMExplicitComponentSpec(
        name='IndepVarComp',
        class_name='openmdao.components.indep_var_comp.IndepVarComp',
        outputs=outputs
    )
    return spec


def build_state_interpolation_comp_spec(
    grid_data: GridData,
    states: list[StateSpec] | None = None
) -> OMExplicitComponentSpec:
    """
    Build spec for StateInterpolationComp.

    StateInterpolationComp interpolates state values from state nodes to
    collocation nodes using Lagrange interpolation.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    states : list[StateSpec], optional
        List of state specs.

    Returns
    -------
    OMExplicitComponentSpec
        Component spec for StateInterpolationComp.
    """
    from openmdao.specs.component_spec import OMExplicitComponentSpec, OMInputSpec, OMOutputSpec

    inputs = []
    outputs = []

    if states:
        for state in states:
            # Number of state nodes and collocation nodes
            num_state_nodes = grid_data.subset_num_nodes['all']
            num_col_nodes = grid_data.subset_num_nodes.get('col', num_state_nodes)

            shape = state.shape if state.shape else (1,)

            # Input: state values at state nodes
            inputs.append(
                OMInputSpec(
                    name=f'states:{state.name}',
                    shape=(num_state_nodes,) if shape == (1,) else (num_state_nodes, *shape),
                    units=state.units
                )
            )

            # Output: interpolated state values at collocation nodes
            outputs.append(
                OMOutputSpec(
                    name=f'state_interp:{state.name}',
                    shape=(num_col_nodes,) if shape == (1,) else (num_col_nodes, *shape),
                    units=state.units
                )
            )

    spec = OMExplicitComponentSpec(
        name='StateInterpolationComp',
        class_name='dymos.phases.components.state_interp_comp.StateInterpolationComp',
        inputs=inputs,
        outputs=outputs
    )
    return spec


def build_ode_component_spec(
    ode_spec: OMExplicitComponentSpec,
    num_nodes: int
) -> OMExplicitComponentSpec:
    """
    Build ODE component spec with injected num_nodes option.

    Takes the base ode_spec (with shape_by_conn=True for all variables) and creates
    a copy with num_nodes injected into options.

    Parameters
    ----------
    ode_spec : OMExplicitComponentSpec
        Base ODE spec from PhaseSpec with inputs/outputs and ODE-specific options.
        All variables should use shape_by_conn=True for shapes to be inferred from
        connections during setup.
    num_nodes : int
        Number of nodes computed from transcription grid (to be injected into options).

    Returns
    -------
    OMExplicitComponentSpec
        ODE spec with num_nodes injected into options for instantiation.

    Notes
    -----
    The returned spec is a modified copy of ode_spec with num_nodes added to options.
    This allows the ODE component to be instantiated with the correct node count
    determined by the transcription grid structure.
    """
    # Create a copy with num_nodes injected into options
    return ode_spec.model_copy(
        update={
            'options': {**ode_spec.options, 'num_nodes': num_nodes}
        }
    )


def build_collocation_constraint_comp_spec(
    grid_data: GridData,
    states: list[StateSpec] | None = None
) -> OMExplicitComponentSpec:
    """
    Build spec for CollocationConstraintComp.

    CollocationConstraintComp enforces the collocation constraint that the
    state time-derivatives match the ODE rate equations at collocation nodes.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    states : list[StateSpec], optional
        List of state specs.

    Returns
    -------
    OMExplicitComponentSpec
        Component spec for CollocationConstraintComp.
    """
    from openmdao.specs.component_spec import OMExplicitComponentSpec, OMInputSpec, OMOutputSpec

    inputs = []
    outputs = []

    if states:
        for state in states:
            num_col_nodes = grid_data.subset_num_nodes.get('col', grid_data.subset_num_nodes['all'])
            shape = state.shape if state.shape else (1,)

            # Inputs: computed and approximate state rates
            inputs.append(
                OMInputSpec(
                    name=f'f_computed:{state.name}',
                    shape=(num_col_nodes,) if shape == (1,) else (num_col_nodes, *shape),
                    units=f'{state.units}/s'  # Rough approximation
                )
            )
            inputs.append(
                OMInputSpec(
                    name=f'f_approx:{state.name}',
                    shape=(num_col_nodes,) if shape == (1,) else (num_col_nodes, *shape),
                    units=f'{state.units}/s'
                )
            )

            # Output: defect (residual of constraint)
            outputs.append(
                OMOutputSpec(
                    name=f'defect:{state.name}',
                    shape=(num_col_nodes,) if shape == (1,) else (num_col_nodes, *shape),
                    units=f'{state.units}/s'
                )
            )

    spec = OMExplicitComponentSpec(
        name='CollocationConstraintComp',
        class_name='dymos.phases.components.collocation_constraint_comp.CollocationConstraintComp',
        inputs=inputs,
        outputs=outputs
    )
    return spec


def build_continuity_constraint_comp_spec(
    grid_data: GridData,
    states: list[StateSpec] | None = None,
    controls: list[ControlSpec] | None = None
) -> OMExplicitComponentSpec:
    """
    Build spec for ContinuityConstraintComp.

    ContinuityConstraintComp enforces continuity of states and controls
    across segment boundaries.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    states : list[StateSpec], optional
        List of state specs.
    controls : list[ControlSpec], optional
        List of control specs.

    Returns
    -------
    OMExplicitComponentSpec
        Component spec for ContinuityConstraintComp.
    """
    from openmdao.specs.component_spec import OMExplicitComponentSpec, OMInputSpec, OMOutputSpec

    inputs = []
    outputs = []

    if states:
        for state in states:
            shape = state.shape if state.shape else (1,)
            num_segments = grid_data.num_segments

            # Inputs: state values at segment boundaries
            inputs.append(
                OMInputSpec(
                    name=f'states:{state.name}',
                    shape=(num_segments,) if shape == (1,) else (num_segments, *shape),
                    units=state.units
                )
            )

            # Output: continuity constraint (residual)
            outputs.append(
                OMOutputSpec(
                    name=f'continuity:{state.name}',
                    shape=(num_segments - 1,) if shape == (1,) else (num_segments - 1, *shape),
                    units=state.units
                )
            )

    if controls:
        for control in controls:
            shape = control.shape if control.shape else (1,)
            num_segments = grid_data.num_segments

            # Similar for controls if they have continuity constraints
            if control.continuity:
                inputs.append(
                    OMInputSpec(
                        name=f'controls:{control.name}',
                        shape=(num_segments,) if shape == (1,) else (num_segments, *shape),
                        units=control.units
                    )
                )
                outputs.append(
                    OMOutputSpec(
                        name=f'continuity:{control.name}',
                        shape=(num_segments - 1,) if shape == (1,) else (num_segments - 1, *shape),
                        units=control.units
                    )
                )

    spec = OMExplicitComponentSpec(
        name='ContinuityConstraintComp',
        class_name='dymos.phases.components.continuity_constraint_comp.ContinuityConstraintComp',
        inputs=inputs,
        outputs=outputs
    )
    return spec


def build_timeseries_output_comp_spec(
    grid_data: GridData,
    timeseries_outputs: list | None = None
) -> OMExplicitComponentSpec:
    """
    Build spec for TimeseriesOutputComp.

    TimeseriesOutputComp collects specified variables into timeseries outputs
    for post-processing.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    timeseries_outputs : list, optional
        List of output variable names to include in timeseries.

    Returns
    -------
    OMExplicitComponentSpec
        Component spec for TimeseriesOutputComp.
    """
    from openmdao.specs.component_spec import OMExplicitComponentSpec

    spec = OMExplicitComponentSpec(
        name='TimeseriesOutputComp',
        class_name='dymos.phases.components.timeseries_output_comp.TimeseriesOutputComp',
        options={
            'timeseries_outputs': timeseries_outputs or []
        }
    )
    return spec
