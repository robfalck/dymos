"""
Instantiation functions to create dymos objects from specs.

Functions to create Phase and Trajectory objects from their spec representations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import importlib

import dymos as dm
from dymos.transcriptions.transcription_base import TranscriptionBase

if TYPE_CHECKING:
    from .transcription_spec import TranscriptionSpec
    from .phase_spec import PhaseSpec
    from .trajectory_spec import TrajectorySpec

from .registries import get_transcription_class


def instantiate_transcription_from_spec(spec: TranscriptionSpec) -> TranscriptionBase:
    """
    Create a transcription object from a TranscriptionSpec.

    Parameters
    ----------
    spec : TranscriptionSpec
        The transcription specification.

    Returns
    -------
    TranscriptionBase
        The instantiated transcription object.

    Raises
    ------
    ValueError
        If transcription type is not recognized.
    """
    transcription_type = spec.transcription_type

    # Get the transcription class
    tx_class = get_transcription_class(transcription_type)
    if tx_class is None:
        raise ValueError(f"Unknown transcription type: {transcription_type}")

    # Extract kwargs from spec (exclude discriminator)
    kwargs = spec.model_dump(exclude={'transcription_type'}, exclude_none=True)

    # Instantiate and return
    return tx_class(**kwargs)


def instantiate_phase_from_spec(spec: PhaseSpec) -> dm.Phase:
    """
    Create a dymos Phase from a PhaseSpec.

    Parameters
    ----------
    spec : PhaseSpec
        The phase specification.

    Returns
    -------
    dymos.Phase
        The instantiated phase.

    Raises
    ------
    ImportError
        If ODE class cannot be imported.
    """
    # Extract ODE class from spec
    # For now, ode_spec contains 'path' and 'init_kwargs' for compatibility
    if isinstance(spec.ode_spec, dict):
        ode_class_path = spec.ode_spec.get('path')
        ode_init_kwargs = spec.ode_spec.get('init_kwargs', {})

        if not ode_class_path:
            raise ValueError("ode_spec must contain 'path' field with the ODE class path")

        # Import ODE class dynamically
        module_path, class_name = ode_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        ode_class = getattr(module, class_name)
    else:
        raise ValueError(f"ode_spec must be a dict, got {type(spec.ode_spec)}")

    # Instantiate transcription
    transcription = instantiate_transcription_from_spec(spec.transcription)

    # Create Phase
    phase = dm.Phase(
        ode_class=ode_class,
        transcription=transcription,
        auto_solvers=spec.auto_solvers,
        **ode_init_kwargs
    )

    # Set time options
    time_opts = spec.time_options
    phase.set_time_options(
        name=time_opts.name,
        units=time_opts.units,
        fix_initial=time_opts.fix_initial,
        fix_duration=time_opts.fix_duration,
        input_initial=time_opts.input_initial,
        input_duration=time_opts.input_duration,
        initial_val=time_opts.initial_val,
        initial_bounds=time_opts.initial_bounds,
        initial_scaler=time_opts.initial_scaler,
        initial_adder=time_opts.initial_adder,
        initial_ref0=time_opts.initial_ref0,
        initial_ref=time_opts.initial_ref,
        duration_val=time_opts.duration_val,
        duration_bounds=time_opts.duration_bounds,
        duration_scaler=time_opts.duration_scaler,
        duration_adder=time_opts.duration_adder,
        duration_ref0=time_opts.duration_ref0,
        duration_ref=time_opts.duration_ref,
        targets=time_opts.targets,
        time_phase_targets=time_opts.time_phase_targets,
        t_initial_targets=time_opts.t_initial_targets,
        t_duration_targets=time_opts.t_duration_targets,
        dt_dstau_targets=time_opts.dt_dstau_targets,
    )

    # Add states
    for state_spec in spec.states:
        phase.add_state(
            name=state_spec.name,
            rate_source=state_spec.rate_source,
            source=state_spec.source,
            units=state_spec.units,
            shape=state_spec.shape,
            val=state_spec.val,
            targets=state_spec.targets,
            opt=state_spec.opt,
            fix_initial=state_spec.fix_initial,
            fix_final=state_spec.fix_final,
            initial_bounds=state_spec.initial_bounds,
            final_bounds=state_spec.final_bounds,
            lower=state_spec.lower,
            upper=state_spec.upper,
            scaler=state_spec.scaler,
            adder=state_spec.adder,
            ref0=state_spec.ref0,
            ref=state_spec.ref,
            defect_scaler=state_spec.defect_scaler,
            defect_ref=state_spec.defect_ref,
            continuity=state_spec.continuity,
            continuity_scaler=state_spec.continuity_scaler,
            continuity_ref=state_spec.continuity_ref,
            solve_segments=state_spec.solve_segments,
            input_initial=state_spec.input_initial,
            input_final=state_spec.input_final,
        )

    # Add controls
    for control_spec in spec.controls:
        if control_spec.control_type == 'polynomial':
            phase.add_polynomial_control(
                name=control_spec.name,
                order=control_spec.order,
                units=control_spec.units,
                shape=control_spec.shape,
                val=control_spec.val,
                targets=control_spec.targets,
                opt=control_spec.opt,
                fix_initial=control_spec.fix_initial,
                fix_final=control_spec.fix_final,
                lower=control_spec.lower,
                upper=control_spec.upper,
                scaler=control_spec.scaler,
                adder=control_spec.adder,
                ref0=control_spec.ref0,
                ref=control_spec.ref,
            )
        else:  # full control
            phase.add_control(
                name=control_spec.name,
                units=control_spec.units,
                shape=control_spec.shape,
                val=control_spec.val,
                targets=control_spec.targets,
                rate_targets=control_spec.rate_targets,
                rate2_targets=control_spec.rate2_targets,
                opt=control_spec.opt,
                fix_initial=control_spec.fix_initial,
                fix_final=control_spec.fix_final,
                lower=control_spec.lower,
                upper=control_spec.upper,
                scaler=control_spec.scaler,
                adder=control_spec.adder,
                ref0=control_spec.ref0,
                ref=control_spec.ref,
                continuity=control_spec.continuity,
                continuity_scaler=control_spec.continuity_scaler,
                continuity_ref=control_spec.continuity_ref,
                rate_continuity=control_spec.rate_continuity,
                rate_continuity_scaler=control_spec.rate_continuity_scaler,
                rate_continuity_ref=control_spec.rate_continuity_ref,
                rate2_continuity=control_spec.rate2_continuity,
                rate2_continuity_scaler=control_spec.rate2_continuity_scaler,
                rate2_continuity_ref=control_spec.rate2_continuity_ref,
            )

    # Add parameters
    for parameter_spec in spec.parameters:
        phase.add_parameter(
            name=parameter_spec.name,
            units=parameter_spec.units,
            val=parameter_spec.val,
            targets=parameter_spec.targets,
            opt=parameter_spec.opt,
            lower=parameter_spec.lower,
            upper=parameter_spec.upper,
            scaler=parameter_spec.scaler,
            adder=parameter_spec.adder,
            ref0=parameter_spec.ref0,
            ref=parameter_spec.ref,
            static_targets=parameter_spec.static_targets,
            include_timeseries=parameter_spec.include_timeseries,
        )

    # Add boundary constraints (TODO: from spec)
    for bc_spec in spec.boundary_constraints:
        # phase.add_boundary_constraint(...)
        pass

    # Add path constraints (TODO: from spec)
    for pc_spec in spec.path_constraints:
        # phase.add_path_constraint(...)
        pass

    # Add objectives (TODO: from spec)
    for obj_spec in spec.objectives:
        # phase.add_objective(...)
        pass

    # Add timeseries outputs (TODO: from spec)
    for ts_spec in spec.timeseries_outputs:
        # phase.add_timeseries_output(...)
        pass

    return phase


def instantiate_trajectory_from_spec(spec: TrajectorySpec) -> dm.Trajectory:
    """
    Create a dymos Trajectory from a TrajectorySpec.

    Parameters
    ----------
    spec : TrajectorySpec
        The trajectory specification.

    Returns
    -------
    dymos.Trajectory
        The instantiated trajectory.
    """
    # Create Trajectory
    traj = dm.Trajectory(
        parallel_phases=spec.parallel_phases,
        auto_solvers=spec.auto_solvers,
    )

    # Instantiate and add phases
    for phase_spec in spec.phases:
        phase = instantiate_phase_from_spec(phase_spec)
        traj.add_phase(phase_spec.name, phase)

    # Add trajectory parameters (TODO: from spec)
    for param_spec in spec.parameters:
        # traj.add_parameter(...)
        pass

    # Add phase linkages (TODO: from spec)
    for linkage_spec in spec.linkages:
        # traj.link_phases(...)
        pass

    return traj
