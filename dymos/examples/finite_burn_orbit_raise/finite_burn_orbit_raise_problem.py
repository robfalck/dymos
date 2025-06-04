import numpy as np

import openmdao.api as om

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE


def make_traj(transcription='gauss-lobatto', transcription_order=3, compressed=False,
              connected=False, default_nonlinear_solver=None, default_linear_solver=None):
    """
    Build a trajectory for the finite burn orbit raise problem.

    Parameters
    ----------
    transcription : str
        One of 'radau' or 'gauss-lobatto'.
    transcription_order : int
        The order of the state transcription polynomials.
    compressed : bool
        If True, use a compressed transcription.
    connected : bool
        If True, connect the phases together, otherwise enforce continuity via constraints.

    Returns
    -------
    traj : dm.Trajectory
        The Trajectory object.
    """
    t = {'gauss-lobatto': dm.GaussLobatto(num_segments=5, order=transcription_order, compressed=compressed),
         'radau': dm.Radau(num_segments=5, order=transcription_order, compressed=compressed)}

    traj = dm.Trajectory()

    if default_nonlinear_solver is not None:
        traj.phases.nonlinear_solver = default_nonlinear_solver

    if default_linear_solver is not None:
        traj.phases.linear_solver = default_linear_solver

    traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                       targets={'burn1': ['c'], 'burn2': ['c']})

    # First Phase (burn)

    burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

    burn1 = traj.add_phase('burn1', burn1)

    burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
    burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=1.0,
                    rate_source='r_dot', units='DU')
    burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=1.0E-3,
                    rate_source='theta_dot', units='rad')
    burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=1.0,
                    rate_source='vr_dot', units='DU/TU')
    burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=1.0,
                    rate_source='vt_dot', units='DU/TU')
    burn1.add_state('accel', fix_initial=True, fix_final=False,
                    rate_source='at_dot', units='DU/TU**2')
    burn1.add_state('deltav', fix_initial=True, fix_final=False,
                    rate_source='deltav_dot', units='DU/TU')
    burn1.add_control('u1', rate_continuity=False, rate2_continuity=False, units='deg', scaler=0.01,
                      rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                      lower=-30, upper=30)
    # Second Phase (Coast)
    coast = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

    coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 50), duration_ref=50, units='TU')
    coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=1.0,
                    rate_source='r_dot', units='DU')
    coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=1.0E-3,
                    rate_source='theta_dot', units='rad')
    coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=1.0,
                    rate_source='vr_dot', units='DU/TU')
    coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=1.0,
                    rate_source='vt_dot', units='DU/TU')
    coast.add_state('accel', fix_initial=True, fix_final=True,
                    rate_source='at_dot', units='DU/TU**2')
    coast.add_state('deltav', fix_initial=False, fix_final=False,
                    rate_source='deltav_dot', units='DU/TU')

    coast.add_parameter('u1', opt=False, val=0.0, units='deg')
    coast.add_parameter('c', val=1.0, units='DU/TU')

    # Third Phase (burn)
    burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

    traj.add_phase('coast', coast)
    traj.add_phase('burn2', burn2)

    if connected:

        burn2.set_time_options(initial_bounds=(1.0, 60), duration_bounds=(-10.0, -0.5),
                               initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=True, fix_final=False, defect_scaler=1.0E-2,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=1000.0,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=1.0E-2,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=1.0,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_scaler=1.0,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0,
                        rate_source='deltav_dot', units='DU/TU')

        burn2.add_objective('deltav', loc='initial', scaler=100.0)

        burn2.add_control('u1', rate_continuity=False, rate2_continuity=False, units='deg',
                          scaler=0.01, lower=-180, upper=180)
    else:

        burn2.set_time_options(initial_bounds=(0.5, 100), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True, defect_scaler=1.0E-2,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=1000.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True, defect_scaler=1.0E-2,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True, defect_scaler=1.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_scaler=1.0,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False, lower=0.0, defect_scaler=1.0,
                        rate_source='deltav_dot', units='DU/TU')

        burn2.add_objective('deltav', loc='final', scaler=100.0)

        burn2.add_control('u1', rate_continuity=False, rate2_continuity=False, units='deg',
                          scaler=0.01, lower=-180, upper=180, targets=['u1'])

    burn1.add_timeseries_output('pos_x')
    coast.add_timeseries_output('pos_x')
    burn2.add_timeseries_output('pos_x')

    burn1.add_timeseries_output('pos_y')
    coast.add_timeseries_output('pos_y')
    burn2.add_timeseries_output('pos_y')

    # Link Phases
    if connected:
        traj.link_phases(phases=['burn1', 'coast'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'],
                         connected=True)

        # No direct connections to the end of a phase.
        traj.link_phases(phases=['burn2', 'coast'],
                         vars=['r', 'theta', 'vr', 'vt', 'deltav'],
                         locs=('final', 'final'))
        traj.link_phases(phases=['burn2', 'coast'],
                         vars=['time'], locs=('final', 'final'))

        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'],
                         locs=('final', 'final'))

    else:
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

    return traj


def two_burn_orbit_raise_problem(transcription='gauss-lobatto', optimizer='SLSQP', r_target=3.0,
                                 transcription_order=3, compressed=False, run_driver=True,
                                 max_iter=300, simulate=True, show_output=True, connected=False, restart=None,
                                 solution_record_file='dymos_solution.db', simulation_record_file='dymos_simulation.db',
                                 default_nonlinear_solver=None, default_linear_solver=None):
    """
    Build and run the finite burn orbit raise problem.

    Parameters
    ----------
    transcription : str
        One of 'radau' or 'gauss-lobatto'.
    optimizer : str
        The pyoptsparse optimizer to use.
    r_target : float
        The final radius target, in AU.
    transcription_order : int
        The order of the state transcription.
    compressed : bool
        If True, use a compressed transcription.
    run_driver : bool
        If True, run the optimization driver.
    max_iter : int
        The maximum allowable number of optimizer iterations.
    simulate : bool
        If True, run simulate on the solution.
    show_output : bool
        If True, display the optimizer output.
    connected : bool
        If True, connect phases for continuity.  Otherwise, enforce continuity via linkage constraints.
    restart : str or None
        The restart file to use, if available.

    Returns
    -------
    p : om.Problem
        The OpenMDAO Problem instance.
    """
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.options['invalid_desvar_behavior'] = 'raise'
    p.driver.declare_coloring()
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = max_iter
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        if show_output:
            p.driver.opt_settings['iSumm'] = 6
    elif optimizer == 'IPOPT':
        p.driver.opt_settings['mu_init'] = 1e-3
        p.driver.opt_settings['max_iter'] = max_iter
        p.driver.opt_settings['acceptable_tol'] = 1e-3
        p.driver.opt_settings['constr_viol_tol'] = 1e-3
        p.driver.opt_settings['compl_inf_tol'] = 1e-3
        p.driver.opt_settings['acceptable_iter'] = 0
        p.driver.opt_settings['tol'] = 1e-3
        p.driver.opt_settings['print_level'] = 5 if show_output else 0
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        p.driver.opt_settings['derivative_test'] = 'first-order'

    traj = make_traj(transcription=transcription, transcription_order=transcription_order,
                     compressed=compressed, connected=connected,
                     default_nonlinear_solver=default_nonlinear_solver,
                     default_linear_solver=default_linear_solver)
    p.model.add_subsystem('traj', subsys=traj)

    # Needed to move the direct solver down into the phases for use with MPI.
    #  - After moving down, used fewer iterations (about 30 less)

    p.setup(check=False)

    # Set Initial Guesses
    traj.set_parameter_val('c', val=1.5, units='DU/TU')

    burn1 = p.model._get_subsystem('traj.phases.burn1')
    burn2 = p.model._get_subsystem('traj.phases.burn2')
    coast = p.model._get_subsystem('traj.phases.coast')

    if burn1 in p.model.traj.phases._subsystems_myproc:
        p.set_val('traj.burn1.t_initial', 0.0)
        p.set_val('traj.burn1.t_duration', 2.25)

        p.set_val('traj.burn1.initial_states:r', 1.0)
        p.set_val('traj.burn1.initial_states:theta', 0.0)
        p.set_val('traj.burn1.initial_states:vr', 0.0)
        p.set_val('traj.burn1.initial_states:vt', 1.0)
        p.set_val('traj.burn1.initial_states:accel', 0.1)
        p.set_val('traj.burn1.initial_states:deltav', 0.0)

        p.set_val('traj.burn1.final_states:r', 1.5)
        p.set_val('traj.burn1.final_states:theta', 1.7)
        p.set_val('traj.burn1.final_states:vr', 0.0)
        p.set_val('traj.burn1.final_states:vt', 1.0)
        p.set_val('traj.burn1.final_states:accel', 0.0)
        p.set_val('traj.burn1.final_states:deltav', 0.1)

        p.set_val('traj.burn1.states:r', burn1.interp('r', ys=[1., 1.5]))
        p.set_val('traj.burn1.states:theta', burn1.interp('theta', ys=[0., 1.7]))
        p.set_val('traj.burn1.states:vr', burn1.interp('vr', ys=[0., 1.]))
        p.set_val('traj.burn1.states:vt', burn1.interp('vt', ys=[1., 1.]))
        p.set_val('traj.burn1.states:accel', burn1.interp('accel', ys=[0, 0]))
        p.set_val('traj.burn1.states:deltav', burn1.interp('deltav', ys=[0, 0.1]))

        p.set_val('traj.burn1.controls:u1', burn1.interp('u1', ys=[0, 0]))

    if coast in p.model.traj.phases._subsystems_myproc:
        p.set_val('traj.coast.t_initial', 2.25)
        p.set_val('traj.coast.t_duration', 3.0)

        p.set_val('traj.coast.initial_states:r', 1.3)
        p.set_val('traj.coast.initial_states:theta', 2.1767)
        p.set_val('traj.coast.initial_states:vr', 0.3285)
        p.set_val('traj.coast.initial_states:vt', 0.97)
        p.set_val('traj.coast.initial_states:accel', 0.0)

        p.set_val('traj.coast.final_states:r', 1.5)
        p.set_val('traj.coast.final_states:theta', 1.7)
        p.set_val('traj.coast.final_states:vr', 0.0)
        p.set_val('traj.coast.final_states:vt', 1.0)
        p.set_val('traj.coast.final_states:accel', 0.0)
        p.set_val('traj.coast.final_states:deltav', 0.0)

        p.set_val('traj.coast.states:r', coast.interp('r', ys=[1.3, 1.5]))
        p.set_val('traj.coast.states:theta', coast.interp('theta', ys=[2.1767, 1.7]))
        p.set_val('traj.coast.states:vr', coast.interp('vr', ys=[0.3285, 0.0]))
        p.set_val('traj.coast.states:vt', coast.interp('vt', ys=[0.97, 1.]))
        p.set_val('traj.coast.states:accel', coast.interp('accel', ys=[0, 0]))

        p.set_val('traj.coast.parameters:u1', 0.0)


    if burn2 in p.model.traj.phases._subsystems_myproc:
        if connected:
            # burn2.set_time_val(initial=7.0, duration=-1.75)
            # p.set_val('traj.burn2.t_initial', 2.25)
            # burn2.set_state_val('r', [r_target, 1])
            # burn2.set_state_val('theta', [4, 0])
            # burn2.set_state_val('vr', [0, 0])
            # burn2.set_state_val('vt', [np.sqrt(1 / r_target), 1])
            # burn2.set_state_val('accel', [0.0, 0.1])
            # burn2.set_state_val('deltav', [0.2, 0.1])
            # burn2.set_control_val('u1', [0, 0])

            p.set_val('traj.burn1.t_initial', 7.0)
            p.set_val('traj.burn1.t_duration', -1.75)

            p.set_val('traj.burn1.initial_states:r', r_target)
            p.set_val('traj.burn1.initial_states:theta', 4.0)
            p.set_val('traj.burn1.initial_states:vr', 0.0)
            p.set_val('traj.burn1.initial_states:vt', np.sqrt(1 / r_target))
            p.set_val('traj.burn1.initial_states:accel', 0.0)
            p.set_val('traj.burn1.initial_states:deltav', 0.2)

            p.set_val('traj.burn1.final_states:r', 1.)
            p.set_val('traj.burn1.final_states:theta', 0.0)
            p.set_val('traj.burn1.final_states:vr', 0.0)
            p.set_val('traj.burn1.final_states:vt', 1.0)
            p.set_val('traj.burn1.final_states:accel', 0.1)
            p.set_val('traj.burn1.final_states:deltav', 0.1)

            p.set_val('traj.burn1.states:r', burn1.interp('r', ys=[r_target, 1.]))
            p.set_val('traj.burn1.states:theta', burn1.interp('theta', ys=[4., 0.]))
            p.set_val('traj.burn1.states:vr', burn1.interp('vr', ys=[0., 0.]))
            p.set_val('traj.burn1.states:vt', burn1.interp('vt', ys=[np.sqrt(1 / r_target), 1]))
            p.set_val('traj.burn1.states:accel', burn1.interp('accel', ys=[0, 0.1]))
            p.set_val('traj.burn1.states:deltav', burn1.interp('deltav', ys=[0.2, 0.1]))

            p.set_val('traj.burn1.controls:u1', burn1.interp('u1', ys=[0, 0]))

        else:
            # burn2.set_time_val(initial=5.25, duration=1.75)
            # burn2.set_val('t_initial', 5.25)
            # burn2.set_state_val('r', [1, r_target])
            # burn2.set_state_val('theta', [0, 4])
            # burn2.set_state_val('vr', [0, 0])
            # burn2.set_state_val('vt', [1, np.sqrt(1 / r_target)])
            # burn2.set_state_val('accel', [0.1, 0.0])
            # burn2.set_state_val('deltav', [0.1, 0.2])
            # burn2.set_control_val('u1', [0, 0])
            p.set_val('traj.burn1.t_initial', 5.25)
            p.set_val('traj.burn1.t_duration', 1.75)

            p.set_val('traj.burn1.initial_states:r', 1.5)
            p.set_val('traj.burn1.initial_states:theta', 0.)
            p.set_val('traj.burn1.initial_states:vr', 0.)
            p.set_val('traj.burn1.initial_states:vt', 1.)
            p.set_val('traj.burn1.initial_states:accel', 0.1)
            p.set_val('traj.burn1.initial_states:deltav', 0.1)

            p.set_val('traj.burn1.final_states:r', r_target)
            p.set_val('traj.burn1.final_states:theta', 4.0)
            p.set_val('traj.burn1.final_states:vr', 0.0)
            p.set_val('traj.burn1.final_states:vt', np.sqrt(1 / r_target))
            p.set_val('traj.burn1.final_states:accel', 0.)
            p.set_val('traj.burn1.final_states:deltav', 0.2)

            p.set_val('traj.burn1.states:r', burn1.interp('r', ys=[1.5, r_target]))
            p.set_val('traj.burn1.states:theta', burn1.interp('theta', ys=[0., 4.]))
            p.set_val('traj.burn1.states:vr', burn1.interp('vr', ys=[0., 0.]))
            p.set_val('traj.burn1.states:vt', burn1.interp('vt', ys=[1., np.sqrt(1 / r_target)]))
            p.set_val('traj.burn1.states:accel', burn1.interp('accel', ys=[0.1, 0.]))
            p.set_val('traj.burn1.states:deltav', burn1.interp('deltav', ys=[0.1, 0.3]))

            p.set_val('traj.burn1.controls:u1', burn1.interp('u1', ys=[0, 0]))


    # p.final_setup()
    # if burn2 in p.model.traj.phases._subsystems_myproc:
    #     print(burn2.get_val('t_initial'))
    #     print(burn2.get_val('t_duration'))

    # p.list_driver_vars(print_arrays=True)
    # exit(0)

    if run_driver or simulate:
        dm.run_problem(p, run_driver=run_driver, simulate=simulate, restart=restart, make_plots=True,
                       solution_record_file=solution_record_file, simulation_record_file=simulation_record_file)

    print(p.get_reports_dir())
    return p


if __name__ == '__main__':  # pragma: no cover

    import openmdao.api as om
    from openmdao.utils.assert_utils import assert_near_equal

    optimizer = 'IPOPT'

    CONNECTED = False

    p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                     compressed=False, optimizer=optimizer, simulate=True,
                                     connected=CONNECTED, show_output=False)

    sol_db = p.get_outputs_dir() / 'dymos_solution.db'
    sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

    sol_case = om.CaseReader(sol_db).get_case('final')
    sim_case = om.CaseReader(sim_db).get_case('final')

    # The last phase in this case is run in reverse time if CONNECTED=True,
    # so grab the correct index to test the resulting delta-V.
    end_idx = 0 if CONNECTED else -1

    assert_near_equal(sol_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)
    assert_near_equal(sim_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)
