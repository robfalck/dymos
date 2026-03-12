"""
Conversion functions from dymos objects to specs.

Functions to convert Phase and Trajectory objects to their spec representations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import importlib
import numpy as np

from dymos.utils.misc import _unspecified

if TYPE_CHECKING:
    from dymos.phase.phase import Phase
    from dymos.trajectory.trajectory import Trajectory
    from dymos.transcriptions.transcription_base import TranscriptionBase

from .transcription_spec import (
    GaussLobattoSpec, RadauSpec, RadauNewSpec, BirkhoffSpec,
    ExplicitShootingSpec, PicardShootingSpec, AnalyticSpec,
    TranscriptionSpec
)
from .variable_spec import TimeSpec
from .variable_spec import StateSpec, ControlSpec, ParameterSpec, TrajParameterSpec
from .constraint_spec import BoundaryConstraintSpec, PathConstraintSpec
from .objective_spec import ObjectiveSpec
from .phase_spec import PhaseSpec, TimeseriesOutputSpec, GridRefinementSpec, SimulateSpec
from .trajectory_spec import TrajectorySpec, LinkageSpec


def _convert_none_or_unspecified(value: any) -> any:
    """Convert _unspecified or None to None for specs."""
    if value is None or value is _unspecified:
        return None
    return value


def _array_to_list_or_scalar(value: any) -> any:
    """Convert numpy arrays to lists, keep scalars as-is."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        return value.tolist()
    elif isinstance(value, (list, tuple)):
        return list(value)
    return value


def transcription_to_spec(transcription: TranscriptionBase) -> TranscriptionSpec:
    """
    Convert a dymos transcription object to a TranscriptionSpec.

    Parameters
    ----------
    transcription : TranscriptionBase
        The transcription object to convert.

    Returns
    -------
    TranscriptionSpec
        The corresponding specification.

    Raises
    ------
    ValueError
        If transcription type is not recognized.
    """
    class_name = transcription.__class__.__name__

    # Get transcription options
    options = transcription.options

    # Convert based on class name
    if class_name == 'GaussLobatto':
        return GaussLobattoSpec(
            num_segments=options['num_segments'],
            segment_ends=_convert_none_or_unspecified(options.get('segment_ends')),
            order=_array_to_list_or_scalar(options.get('order', 3)),
            compressed=options.get('compressed', False),
            solve_segments=options.get('solve_segments', False),
        )

    elif class_name == 'Radau':
        return RadauSpec(
            num_segments=options['num_segments'],
            segment_ends=_convert_none_or_unspecified(options.get('segment_ends')),
            order=_array_to_list_or_scalar(options.get('order', 3)),
            compressed=options.get('compressed', False),
        )

    elif class_name == 'RadauNew':
        return RadauNewSpec(
            num_segments=options['num_segments'],
            segment_ends=_convert_none_or_unspecified(options.get('segment_ends')),
            order=_array_to_list_or_scalar(options.get('order', 3)),
            compressed=options.get('compressed', False),
            solve_segments=options.get('solve_segments', False),
        )

    elif class_name == 'Birkhoff':
        return BirkhoffSpec(
            num_nodes=options.get('num_nodes', 25),
            grid_type=options.get('grid_type', 'cgl'),
        )

    elif class_name == 'ExplicitShooting':
        return ExplicitShootingSpec(
            num_segments=options['num_segments'],
            method=options.get('method', 'RK45'),
            atol=options.get('atol', 1.0e-9),
            rtol=options.get('rtol', 1.0e-6),
            first_step=_convert_none_or_unspecified(options.get('first_step')),
            max_step=_convert_none_or_unspecified(options.get('max_step')),
            propagate_derivs=options.get('propagate_derivs', True),
            grid=options.get('grid', 'uniform'),
            output_grid=_convert_none_or_unspecified(options.get('output_grid')),
            control_interp=options.get('control_interp', 'cubic'),
            times_per_seg=options.get('times_per_seg'),
        )

    elif class_name == 'PicardShooting':
        return PicardShootingSpec(
            num_segments=options['num_segments'],
            method=options.get('method', 'RK45'),
            atol=options.get('atol', 1.0e-9),
            rtol=options.get('rtol', 1.0e-6),
            first_step=_convert_none_or_unspecified(options.get('first_step')),
            max_step=_convert_none_or_unspecified(options.get('max_step')),
            propagate_derivs=options.get('propagate_derivs', True),
            grid=options.get('grid', 'uniform'),
            output_grid=_convert_none_or_unspecified(options.get('output_grid')),
            control_interp=options.get('control_interp', 'cubic'),
            times_per_seg=options.get('times_per_seg'),
        )

    elif class_name == 'Analytic':
        return AnalyticSpec(
            num_segments=options['num_segments'],
            segment_ends=_convert_none_or_unspecified(options.get('segment_ends')),
            order=_array_to_list_or_scalar(options.get('order', 3)),
            compressed=options.get('compressed', False),
        )

    else:
        raise ValueError(f"Unknown transcription type: {class_name}")


def _state_options_to_spec(name: str, state_opts: dict) -> StateSpec:
    """Convert state options dict to StateSpec."""
    return StateSpec(
        name=name,
        units=_convert_none_or_unspecified(state_opts.get('units')),
        val=_array_to_list_or_scalar(state_opts.get('val', 0.0)),
        shape=_array_to_list_or_scalar(state_opts.get('shape')),
        desc=state_opts.get('desc', ''),
        opt=state_opts.get('opt', True),
        lower=_convert_none_or_unspecified(_array_to_list_or_scalar(state_opts.get('lower'))),
        upper=_convert_none_or_unspecified(_array_to_list_or_scalar(state_opts.get('upper'))),
        ref0=_convert_none_or_unspecified(_array_to_list_or_scalar(state_opts.get('ref0'))),
        ref=_convert_none_or_unspecified(_array_to_list_or_scalar(state_opts.get('ref'))),
        scaler=_convert_none_or_unspecified(_array_to_list_or_scalar(state_opts.get('scaler'))),
        adder=_convert_none_or_unspecified(_array_to_list_or_scalar(state_opts.get('adder'))),
        rate_source=state_opts.get('rate_source'),
        source=_convert_none_or_unspecified(state_opts.get('source')),
        targets=_convert_none_or_unspecified(state_opts.get('targets')),
        fix_initial=state_opts.get('fix_initial', False),
        fix_final=state_opts.get('fix_final', False),
        initial_bounds=_convert_none_or_unspecified(state_opts.get('initial_bounds')),
        final_bounds=_convert_none_or_unspecified(state_opts.get('final_bounds')),
        defect_scaler=_convert_none_or_unspecified(state_opts.get('defect_scaler')),
        defect_ref=_convert_none_or_unspecified(state_opts.get('defect_ref')),
        continuity=state_opts.get('continuity', True),
        continuity_scaler=_convert_none_or_unspecified(state_opts.get('continuity_scaler')),
        continuity_ref=_convert_none_or_unspecified(state_opts.get('continuity_ref')),
        solve_segments=state_opts.get('solve_segments', False),
        input_initial=state_opts.get('input_initial', False),
        input_final=state_opts.get('input_final', False),
    )


def _control_options_to_spec(name: str, control_opts: dict) -> ControlSpec:
    """Convert control options dict to ControlSpec."""
    return ControlSpec(
        name=name,
        units=_convert_none_or_unspecified(control_opts.get('units')),
        val=_array_to_list_or_scalar(control_opts.get('val', np.zeros(1))),
        shape=_array_to_list_or_scalar(control_opts.get('shape')),
        desc=control_opts.get('desc', ''),
        opt=control_opts.get('opt', True),
        lower=_convert_none_or_unspecified(_array_to_list_or_scalar(control_opts.get('lower'))),
        upper=_convert_none_or_unspecified(_array_to_list_or_scalar(control_opts.get('upper'))),
        ref0=_convert_none_or_unspecified(_array_to_list_or_scalar(control_opts.get('ref0'))),
        ref=_convert_none_or_unspecified(_array_to_list_or_scalar(control_opts.get('ref'))),
        scaler=_convert_none_or_unspecified(_array_to_list_or_scalar(control_opts.get('scaler'))),
        adder=_convert_none_or_unspecified(_array_to_list_or_scalar(control_opts.get('adder'))),
        control_type=control_opts.get('control_type', 'full'),
        order=control_opts.get('order'),
        targets=_convert_none_or_unspecified(control_opts.get('targets')),
        rate_targets=_convert_none_or_unspecified(control_opts.get('rate_targets')),
        rate2_targets=_convert_none_or_unspecified(control_opts.get('rate2_targets')),
        fix_initial=control_opts.get('fix_initial', False),
        fix_final=control_opts.get('fix_final', False),
        continuity=control_opts.get('continuity', True),
        continuity_scaler=_convert_none_or_unspecified(control_opts.get('continuity_scaler')),
        continuity_ref=_convert_none_or_unspecified(control_opts.get('continuity_ref')),
        rate_continuity=control_opts.get('rate_continuity', True),
        rate_continuity_scaler=_convert_none_or_unspecified(control_opts.get('rate_continuity_scaler')),
        rate_continuity_ref=_convert_none_or_unspecified(control_opts.get('rate_continuity_ref')),
        rate2_continuity=control_opts.get('rate2_continuity', False),
        rate2_continuity_scaler=_convert_none_or_unspecified(control_opts.get('rate2_continuity_scaler')),
        rate2_continuity_ref=_convert_none_or_unspecified(control_opts.get('rate2_continuity_ref')),
    )


def _parameter_options_to_spec(name: str, param_opts: dict) -> ParameterSpec:
    """Convert parameter options dict to ParameterSpec."""
    return ParameterSpec(
        name=name,
        units=_convert_none_or_unspecified(param_opts.get('units')),
        val=_array_to_list_or_scalar(param_opts.get('val', 0.0)),
        shape=_array_to_list_or_scalar(param_opts.get('shape')),
        desc=param_opts.get('desc', ''),
        opt=param_opts.get('opt', True),
        lower=_convert_none_or_unspecified(_array_to_list_or_scalar(param_opts.get('lower'))),
        upper=_convert_none_or_unspecified(_array_to_list_or_scalar(param_opts.get('upper'))),
        ref0=_convert_none_or_unspecified(_array_to_list_or_scalar(param_opts.get('ref0'))),
        ref=_convert_none_or_unspecified(_array_to_list_or_scalar(param_opts.get('ref'))),
        scaler=_convert_none_or_unspecified(_array_to_list_or_scalar(param_opts.get('scaler'))),
        adder=_convert_none_or_unspecified(_array_to_list_or_scalar(param_opts.get('adder'))),
        targets=_convert_none_or_unspecified(param_opts.get('targets')),
        static_targets=_convert_none_or_unspecified(param_opts.get('static_targets')),
        include_timeseries=param_opts.get('include_timeseries', True),
    )


def _ode_to_spec(ode_class, ode_init_kwargs: dict) -> dict | object:
    """
    Convert ODE class to spec format.

    Can return either a dict (legacy format) or OMExplicitComponentSpec (new format).
    For now, this returns a dict, but it's designed to be extended to return
    OMExplicitComponentSpec in the future via introspection.

    Parameters
    ----------
    ode_class : type
        The ODE component class.
    ode_init_kwargs : dict
        ODE-specific initialization parameters (filtered to exclude num_nodes).

    Returns
    -------
    dict or OMExplicitComponentSpec
        ODE specification (currently returns dict, future: OMExplicitComponentSpec).

    Notes
    -----
    Future enhancement: Introspect the ODE component's setup() method to
    extract inputs, outputs, and create a full OMExplicitComponentSpec with
    shape_by_conn=True for all variables.
    """
    ode_class_path = f"{ode_class.__module__}.{ode_class.__name__}"

    # Filter out num_nodes (defensive - should never be there)
    ode_init_kwargs = {k: v for k, v in ode_init_kwargs.items() if k != 'num_nodes'}

    # Return legacy dict format (future: would return OMExplicitComponentSpec)
    return {
        'path': ode_class_path,
        'init_kwargs': ode_init_kwargs,
    }


def phase_to_spec(phase: Phase) -> PhaseSpec:
    """
    Convert a dymos Phase to a PhaseSpec.

    Parameters
    ----------
    phase : dymos.Phase
        The phase to convert.

    Returns
    -------
    PhaseSpec
        The phase specification.
    """
    # Get ODE spec using helper function
    ode_class = phase.options['ode_class']
    ode_init_kwargs = phase.options.get('ode_init_kwargs', {})
    ode_spec = _ode_to_spec(ode_class, ode_init_kwargs)

    # Convert transcription
    transcription_spec = transcription_to_spec(phase.options['transcription'])

    # Convert time options
    time_opts = phase.time_options
    time_spec = TimeSpec(
        name=time_opts.get('name', 'time'),
        units=_convert_none_or_unspecified(time_opts.get('units')),
        fix_initial=time_opts.get('fix_initial', False),
        fix_duration=time_opts.get('fix_duration', False),
        input_initial=time_opts.get('input_initial', False),
        input_duration=time_opts.get('input_duration', False),
        initial_val=_array_to_list_or_scalar(time_opts.get('initial_val', 0.0)),
        initial_bounds=_convert_none_or_unspecified(time_opts.get('initial_bounds')),
        initial_scaler=_convert_none_or_unspecified(time_opts.get('initial_scaler')),
        initial_adder=_convert_none_or_unspecified(time_opts.get('initial_adder')),
        initial_ref0=_convert_none_or_unspecified(time_opts.get('initial_ref0')),
        initial_ref=_convert_none_or_unspecified(time_opts.get('initial_ref')),
        duration_val=_array_to_list_or_scalar(time_opts.get('duration_val', 1.0)),
        duration_bounds=_convert_none_or_unspecified(time_opts.get('duration_bounds')),
        duration_scaler=_convert_none_or_unspecified(time_opts.get('duration_scaler')),
        duration_adder=_convert_none_or_unspecified(time_opts.get('duration_adder')),
        duration_ref0=_convert_none_or_unspecified(time_opts.get('duration_ref0')),
        duration_ref=_convert_none_or_unspecified(time_opts.get('duration_ref')),
        targets=_convert_none_or_unspecified(time_opts.get('targets')),
        time_phase_targets=_convert_none_or_unspecified(time_opts.get('time_phase_targets')),
        t_initial_targets=_convert_none_or_unspecified(time_opts.get('t_initial_targets')),
        t_duration_targets=_convert_none_or_unspecified(time_opts.get('t_duration_targets')),
        dt_dstau_targets=_convert_none_or_unspecified(time_opts.get('dt_dstau_targets')),
    )

    # Convert state, control, and parameter options
    states = [_state_options_to_spec(name, opts) for name, opts in phase.state_options.items()]
    controls = [_control_options_to_spec(name, opts) for name, opts in phase.control_options.items()]
    parameters = [_parameter_options_to_spec(name, opts) for name, opts in phase.parameter_options.items()]

    # Convert constraints (TODO: Extract from phase internals)
    boundary_constraints = []
    path_constraints = []

    # Convert objectives (TODO: Extract from phase internals)
    objectives = []

    # Convert timeseries outputs (TODO: Extract from phase internals)
    timeseries_outputs = []

    return PhaseSpec(
        name=phase.name,
        ode_spec=ode_spec,
        transcription=transcription_spec,
        time_options=time_spec,
        states=states,
        controls=controls,
        parameters=parameters,
        boundary_constraints=boundary_constraints,
        path_constraints=path_constraints,
        objectives=objectives,
        timeseries_outputs=timeseries_outputs,
        auto_solvers=phase.options.get('auto_solvers', True),
    )


def trajectory_to_spec(trajectory: Trajectory) -> TrajectorySpec:
    """
    Convert a dymos Trajectory to a TrajectorySpec.

    Parameters
    ----------
    trajectory : dymos.Trajectory
        The trajectory to convert.

    Returns
    -------
    TrajectorySpec
        The trajectory specification.
    """
    # Convert phases
    phases = [phase_to_spec(phase) for phase in trajectory._phases.values()]

    # Convert trajectory parameters (TODO)
    parameters = []

    # Convert linkages (TODO)
    linkages = []

    return TrajectorySpec(
        name=trajectory.name if hasattr(trajectory, 'name') else 'traj',
        phases=phases,
        parameters=parameters,
        linkages=linkages,
    )
