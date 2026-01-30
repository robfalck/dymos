"""
Helper functions for building connection specs between dymos components.

These builders create lists of ConnectionSpec objects that define how variables
flow between components in a dymos phase group.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dymos.transcriptions.grid_data import GridData
    from dymos.specs.variable_spec import StateSpec, ControlSpec, ParameterSpec
    from dymos.specs.phase_spec import PhaseSpec
    from dymos.specs.time_spec import TimeSpec
    from openmdao.specs.connection_spec import ConnectionSpec


def build_time_connections(
    grid_data: GridData,
    time_comp_name: str = 'time_comp',
    rhs_comp_name: str = 'rhs_disc'
) -> list[ConnectionSpec]:
    """
    Build connections from TimeComp to ODE component.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    time_comp_name : str
        Name of the time component.
    rhs_comp_name : str
        Name of the ODE component.

    Returns
    -------
    list[ConnectionSpec]
        Connections for time and time_phase from TimeComp to ODE.
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = [
        ConnectionSpec(src=f'{time_comp_name}.time', tgt=f'{rhs_comp_name}.t_all'),
        ConnectionSpec(src=f'{time_comp_name}.time_phase', tgt=f'{rhs_comp_name}.t_phase'),
    ]
    return connections


def build_parameter_connections(
    parameters: list[ParameterSpec] | None = None,
    param_comp_name: str = 'parameter_comp',
    rhs_comp_name: str = 'rhs_disc'
) -> list[ConnectionSpec]:
    """
    Build connections from ParameterComp to ODE component.

    Parameters
    ----------
    parameters : list[ParameterSpec], optional
        List of parameter specs.
    param_comp_name : str
        Name of the parameter component.
    rhs_comp_name : str
        Name of the ODE component.

    Returns
    -------
    list[ConnectionSpec]
        Connections for parameters from ParameterComp to ODE.
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = []
    if parameters:
        for param in parameters:
            # Parameters connect directly (no interpolation needed)
            connections.append(
                ConnectionSpec(
                    src=f'{param_comp_name}.{param.name}',
                    tgt=f'{rhs_comp_name}.{param.name}'
                )
            )
    return connections


def build_state_rate_source_connections(
    states: list[StateSpec] | None = None,
    rhs_comp_name: str = 'rhs_disc'
) -> list[ConnectionSpec]:
    """
    Build connections from ODE component outputs to state rate sources.

    The ODE provides time-derivatives (rates) for each state.

    Parameters
    ----------
    states : list[StateSpec], optional
        List of state specs.
    rhs_comp_name : str
        Name of the ODE component.

    Returns
    -------
    list[ConnectionSpec]
        Connections from ODE rates to state rate sources.
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = []
    if states:
        for state in states:
            if state.rate_source:
                # rate_source can include 'ode.' prefix or not
                rate_src = state.rate_source
                if not rate_src.startswith('ode.'):
                    # If no prefix, assume it's in the ODE component
                    rate_src = f'ode.{rate_src}'

                # Extract just the output name (remove 'ode.' prefix)
                rate_output = rate_src.replace('ode.', '')

                connections.append(
                    ConnectionSpec(
                        src=f'{rhs_comp_name}.{rate_output}',
                        tgt=f'rate_sources:{state.name}'
                    )
                )
    return connections


def build_control_connections(
    controls: list[ControlSpec] | None = None,
    control_interp_comp_name: str = 'control_interp_comp',
    rhs_comp_name: str = 'rhs_disc'
) -> list[ConnectionSpec]:
    """
    Build connections from ControlInterpComp to ODE component.

    Parameters
    ----------
    controls : list[ControlSpec], optional
        List of control specs.
    control_interp_comp_name : str
        Name of the control interpolation component.
    rhs_comp_name : str
        Name of the ODE component.

    Returns
    -------
    list[ConnectionSpec]
        Connections for interpolated controls to ODE.
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = []
    if controls:
        for control in controls:
            if control.targets:
                # control.targets is a list of target names
                for target in control.targets:
                    connections.append(
                        ConnectionSpec(
                            src=f'{control_interp_comp_name}.control_interp:{control.name}',
                            tgt=f'{rhs_comp_name}.{target}'
                        )
                    )
    return connections


def build_state_interpolation_connections(
    states: list[StateSpec] | None = None,
    state_interp_comp_name: str = 'state_interp_comp',
    rhs_comp_name: str = 'rhs_col'
) -> list[ConnectionSpec]:
    """
    Build connections from StateInterpolationComp to collocation ODE component.

    Interpolated states are needed at collocation nodes for evaluation.

    Parameters
    ----------
    states : list[StateSpec], optional
        List of state specs.
    state_interp_comp_name : str
        Name of the state interpolation component.
    rhs_comp_name : str
        Name of the collocation ODE component.

    Returns
    -------
    list[ConnectionSpec]
        Connections for interpolated states to collocation ODE.
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = []
    if states:
        for state in states:
            if state.targets:
                # state.targets is a list of target names in the ODE
                for target in state.targets:
                    connections.append(
                        ConnectionSpec(
                            src=f'{state_interp_comp_name}.state_interp:{state.name}',
                            tgt=f'{rhs_comp_name}.{target}'
                        )
                    )
    return connections


def build_collocation_defect_connections(
    states: list[StateSpec] | None = None,
    collocation_comp_name: str = 'collocation_constraint_comp',
    rhs_col_comp_name: str = 'rhs_col'
) -> list[ConnectionSpec]:
    """
    Build connections to enforce collocation constraints.

    Collocation constraints check that state derivatives from interpolation
    match ODE rates at collocation nodes.

    Parameters
    ----------
    states : list[StateSpec], optional
        List of state specs.
    collocation_comp_name : str
        Name of the collocation constraint component.
    rhs_col_comp_name : str
        Name of the collocation ODE component.

    Returns
    -------
    list[ConnectionSpec]
        Connections for collocation constraints.
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = []
    if states:
        for state in states:
            if state.rate_source:
                # Get the rate output name from the ODE
                rate_src = state.rate_source
                if not rate_src.startswith('ode.'):
                    rate_src = f'ode.{rate_src}'
                rate_output = rate_src.replace('ode.', '')

                # Connect ODE rate to collocation constraint
                connections.append(
                    ConnectionSpec(
                        src=f'{rhs_col_comp_name}.{rate_output}',
                        tgt=f'{collocation_comp_name}.f_computed:{state.name}'
                    )
                )
    return connections


def build_continuity_connections(
    states: list[StateSpec] | None = None,
    controls: list[ControlSpec] | None = None,
    continuity_comp_name: str = 'continuity_constraint_comp'
) -> list[ConnectionSpec]:
    """
    Build connections for continuity constraints.

    Continuity constraints ensure states and controls are continuous
    across segment boundaries.

    Parameters
    ----------
    states : list[StateSpec], optional
        List of state specs.
    controls : list[ControlSpec], optional
        List of control specs.
    continuity_comp_name : str
        Name of the continuity constraint component.

    Returns
    -------
    list[ConnectionSpec]
        Connections for continuity constraints.
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = []

    # State continuity
    if states:
        for state in states:
            connections.append(
                ConnectionSpec(
                    src=f'indep_states.states:{state.name}',
                    tgt=f'{continuity_comp_name}.states:{state.name}'
                )
            )

    # Control continuity
    if controls:
        for control in controls:
            if control.continuity:
                connections.append(
                    ConnectionSpec(
                        src=f'indep_controls.controls:{control.name}',
                        tgt=f'{continuity_comp_name}.controls:{control.name}'
                    )
                )

    return connections


def build_timeseries_connections(
    grid_data: GridData,
    timeseries_outputs: list | None = None,
    timeseries_comp_name: str = 'timeseries_outputs',
    rhs_comp_name: str = 'rhs_all'
) -> list[ConnectionSpec]:
    """
    Build connections for timeseries output collection.

    Timeseries components gather specified variables for post-processing output.

    Parameters
    ----------
    grid_data : GridData
        Grid data for the phase.
    timeseries_outputs : list, optional
        List of output specifications.
    timeseries_comp_name : str
        Name of the timeseries output component.
    rhs_comp_name : str
        Name of the ODE component (for source of variables).

    Returns
    -------
    list[ConnectionSpec]
        Connections to timeseries output component.
    """
    from openmdao.specs.connection_spec import ConnectionSpec

    connections = []
    if timeseries_outputs:
        for output_spec in timeseries_outputs:
            # Determine source of the timeseries output
            # This could be from ODE, states, controls, or parameters
            source = output_spec.get('source', f'{rhs_comp_name}.{output_spec["name"]}')
            output_name = output_spec.get('output_name', output_spec['name'])

            connections.append(
                ConnectionSpec(
                    src=source,
                    tgt=f'{timeseries_comp_name}.{output_name}'
                )
            )

    return connections
