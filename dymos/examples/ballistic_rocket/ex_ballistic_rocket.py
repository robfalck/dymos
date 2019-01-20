from __future__ import print_function, division, absolute_import

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase, Trajectory
from dymos.examples.ballistic_rocket.ballistic_rocket_ode import BallisticRocketUnguidedODE, \
    BallisticRocketGuidedODE

SHOW_PLOTS = True


def ballistic_rocket_max_range(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                               run_driver=True, top_level_jacobian='csc', compressed=True,
                               sim_record='ballistic_rocket_sim.db', optimizer='SLSQP',
                               dynamic_simul_derivs=True):
    p = Problem(model=Group())

    if optimizer == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['iSumm'] = 6
    else:
        p.driver = ScipyOptimizeDriver()

    p.driver.options['dynamic_simul_derivs'] = dynamic_simul_derivs

    #
    # Phase 0: Vertical Boost
    #

    boost_phase = Phase(transcription,
                  ode_class=BallisticRocketUnguidedODE,
                  num_segments=num_segments,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('boost', boost_phase)

    boost_phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    boost_phase.set_state_options('x', fix_initial=True, fix_final=False)
    boost_phase.set_state_options('y', fix_initial=True, fix_final=True)
    boost_phase.set_state_options('vx', fix_initial=True, fix_final=False)
    boost_phase.set_state_options('vy', fix_initial=True, fix_final=False)
    boost_phase.set_state_options('mprop', fix_initial=True, fix_final=False)

    boost_phase.add_design_parameter('thrust', units='N', opt=False, val=2000.0)
    boost_phase.add_design_parameter('theta', units='deg', opt=False, val=90.0)
    boost_phase.add_design_parameter('mstruct', units='kg', opt=False, val=100.0)
    boost_phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)
    boost_phase.add_design_parameter('Isp', units='s', opt=False, val=300.0)

    boost_phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.options['assembled_jac_type'] = 'csc'
    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup(check=True)

    p['boost.t_initial'] = 0.0
    p['boost.t_duration'] = 2.0

    p['boost.states:x'] = boost_phase.interpolate(ys=[0, 10], nodes='state_input')
    p['boost.states:y'] = boost_phase.interpolate(ys=[0, 100], nodes='state_input')
    p['boost.states:vx'] = boost_phase.interpolate(ys=[0, 0], nodes='state_input')
    p['boost.states:vy'] = boost_phase.interpolate(ys=[0, 500], nodes='state_input')

    p['boost.design_parameters:g'] = 9.80665
    p['boost.design_parameters:theta'] = 90.0
    p['boost.design_parameters:mstruct'] = 100

    p.run_model()

    exp_out = boost_phase.simulate()

    t_imp = boost_phase.get_values('time')
    y_imp = boost_phase.get_values('y')

    t_exp = exp_out.get_values('time')
    y_exp = exp_out.get_values('y')

    import matplotlib.pyplot as plt
    plt.plot(t_imp, y_imp, 'ro', label='implicit')
    plt.plot(t_exp, y_exp, 'b-', label='explicit')
    plt.show()
    exit(0)

    # #
    # # Phase 1: Pitchover
    # #
    #
    # # Minimize time at the end of the phase
    # boost_phase.add_objective('time_phase', loc='final', scaler=10)
    #
    # p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
    # p.model.linear_solver = DirectSolver(assemble_jac=True)
    # p.setup(check=True)
    #
    # p['phase0.t_initial'] = 0.0
    # p['phase0.t_duration'] = 2.0
    #
    # p['phase0.states:x'] = boost_phase.interpolate(ys=[0, 10], nodes='state_input')
    # p['phase0.states:y'] = boost_phase.interpolate(ys=[10, 5], nodes='state_input')
    # p['phase0.states:v'] = boost_phase.interpolate(ys=[0, 9.9], nodes='state_input')
    # p['phase0.controls:theta'] = boost_phase.interpolate(ys=[5, 100], nodes='control_input')
    # p['phase0.design_parameters:g'] = 9.80665

    # p.run_model()
    # if run_driver:
    #     p.run_driver()

    # Plot results
    if SHOW_PLOTS:
        # exp_out = boost_phase.simulate(times=50, record_file=sim_record)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = boost_phase.get_values('x', nodes='all')
        y_imp = boost_phase.get_values('y', nodes='all')

        x_exp = exp_out.get_values('x')
        y_exp = exp_out.get_values('y')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = boost_phase.get_values('time_phase', nodes='all')
        y_imp = boost_phase.get_values('theta_rate2', nodes='all')

        x_exp = exp_out.get_values('time_phase')
        y_exp = exp_out.get_values('theta_rate2')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('theta rate2 (rad/s**2)')
        ax.grid(True)
        ax.legend(loc='lower right')

        plt.show()

    return p


if __name__ == '__main__':
    ballistic_rocket_max_range(transcription='radau-ps', num_segments=10, run_driver=True,
                               transcription_order=3, compressed=True,
                               optimizer='SNOPT')
