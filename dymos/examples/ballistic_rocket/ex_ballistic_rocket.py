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
                               run_driver=True, top_level_jacobian='csc', compressed=False,
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
    # The Trajectory Group
    #
    traj = Trajectory()

    #
    # Phase 0: Vertical Boost
    #

    boost_phase = Phase(transcription,
                  ode_class=BallisticRocketUnguidedODE,
                  num_segments=num_segments,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('boost', boost_phase)

    boost_phase.set_time_options(fix_initial=True, duration_bounds=(.5, 30))

    boost_phase.set_state_options('x', fix_initial=True, fix_final=False, defect_ref=1e3)
    boost_phase.set_state_options('y', fix_initial=True, fix_final=True)
    boost_phase.set_state_options('vx', fix_initial=True, fix_final=False)
    boost_phase.set_state_options('vy', fix_initial=True, fix_final=False)
    boost_phase.set_state_options('mprop', fix_initial=True, fix_final=False)

    boost_phase.add_design_parameter('thrust', units='N', opt=False, val=2000.0)
    boost_phase.add_design_parameter('theta', units='deg', opt=False, val=90.0)
    boost_phase.add_design_parameter('mstruct', units='kg', opt=False, val=100.0)
    boost_phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)
    boost_phase.add_design_parameter('Isp', units='s', opt=False, val=300.0)

    # boost_phase.add_objective('mprop', loc='final', scaler=10)
    
    traj.add_phase('boost', boost_phase)

    #
    # Phase 1: Pitchover 
    #

    pitchover_phase = Phase(transcription,
                            ode_class=BallisticRocketGuidedODE,
                            num_segments=num_segments,
                            transcription_order=transcription_order,
                            compressed=compressed)

    p.model.add_subsystem('pitchover', pitchover_phase)

    pitchover_phase.set_time_options(fix_initial=False, initial_bounds=(1,20), duration_bounds=(2, 100))

    pitchover_phase.set_state_options('x', fix_initial=False, fix_final=False)
    pitchover_phase.set_state_options('y', fix_initial=False, fix_final=False)
    pitchover_phase.set_state_options('vx', fix_initial=False, fix_final=False)
    pitchover_phase.set_state_options('vy', fix_initial=False, fix_final=False)
    pitchover_phase.set_state_options('mprop', fix_initial=False, fix_final=True)

    pitchover_phase.add_design_parameter('thrust', units='N', opt=False, val=2000.0)
    pitchover_phase.add_design_parameter('mstruct', units='kg', opt=False, val=100.0)
    pitchover_phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)
    pitchover_phase.add_design_parameter('Isp', units='s', opt=False, val=300.0)
    pitchover_phase.add_design_parameter('theta_0', units='deg', opt=False, val=90.0)
    pitchover_phase.add_design_parameter('theta_f', units='deg', opt=False, val=45.0, lower=40., upper=90)

    pitchover_phase.add_objective('time', loc='final', scaler=10)
    
    traj.add_phase('pitchover', pitchover_phase)

    traj.link_phases(phases=['boost', 'pitchover'], vars=['time', 'x', 'y', 'vx', 'vy', 'mprop'])

    p.model = Group()
    p.model.add_subsystem('traj', traj)

    #
    # Setup and set values
    #
    # p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup(check=True)

    p['traj.boost.t_initial'] = 0.0
    p['traj.boost.t_duration'] = 10

    p['traj.boost.states:x'] = boost_phase.interpolate(ys=[0, 0], nodes='state_input')
    p['traj.boost.states:y'] = boost_phase.interpolate(ys=[0, 100], nodes='state_input')
    p['traj.boost.states:vx'] = boost_phase.interpolate(ys=[0, 0], nodes='state_input')
    p['traj.boost.states:vy'] = boost_phase.interpolate(ys=[0, 500], nodes='state_input')
    p['traj.boost.states:mprop'] = boost_phase.interpolate(ys=[20, 10.], nodes='state_input')

    p['traj.boost.design_parameters:g'] = 9.80665
    p['traj.boost.design_parameters:theta'] = 90.0
    p['traj.boost.design_parameters:mstruct'] = 100

    p['traj.pitchover.t_initial'] = 2.0
    p['traj.pitchover.t_duration'] = 2.0

    p['traj.pitchover.states:x'] = boost_phase.interpolate(ys=[0, 10], nodes='state_input')
    p['traj.pitchover.states:y'] = boost_phase.interpolate(ys=[20, 100], nodes='state_input')
    p['traj.pitchover.states:vx'] = boost_phase.interpolate(ys=[0, 10], nodes='state_input')
    p['traj.pitchover.states:vy'] = boost_phase.interpolate(ys=[50, 100], nodes='state_input')
    p['traj.pitchover.states:mprop'] = boost_phase.interpolate(ys=[16, 0], nodes='state_input')

    p['traj.pitchover.design_parameters:g'] = 9.80665
    p['traj.pitchover.design_parameters:theta_0'] = 90.0
    p['traj.pitchover.design_parameters:theta_f'] = 80.0
    p['traj.pitchover.design_parameters:mstruct'] = 100

   

    return p


if __name__ == '__main__':
    p = ballistic_rocket_max_range(transcription='radau-ps', num_segments=10, run_driver=True,
                                   transcription_order=3, compressed=False,
                                   optimizer='SNOPT')

    # p.run_model()
    p.run_driver()



    # p.run_model()
    # if run_driver:
    #     p.run_driver()

    # Plot results
    if SHOW_PLOTS:
        exp_out = p.model.traj.simulate()

        fig, axes = plt.subplots(ncols=2)
        fig.suptitle('Ballistic Rocket Solution')

        boost_phase = p.model.traj.phases.boost
        x_imp = boost_phase.get_values('x', nodes='all')
        y_imp = boost_phase.get_values('y', nodes='all')
        t_imp = boost_phase.get_values('time', nodes='all')
        
        # vals = boost_phase.get_values('mprop', nodes='all')
        # print(vals[-1])
        # vals = boost_phase.get_values('vx', nodes='all')
        # print(vals[-1])
        # vals = boost_phase.get_values('vy', nodes='all')
        # print(vals[-1])

        x_exp = exp_out.get_values('x', phases='boost', flat=True)
        y_exp = exp_out.get_values('y', phases='boost', flat=True)
        t_exp = exp_out.get_values('time', phases='boost', flat=True)

        ax = axes[0]
        ax.plot(t_imp, y_imp, 'ro', label='i: boost')
        ax.plot(t_exp, y_exp, 'b-', label='sim: boost')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        ax = axes[1]
        ax.plot(x_imp, y_imp, 'ro', label='i: boost')
        ax.plot(x_exp, y_exp, 'b-', label='sim: boost')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(-1,1)
        ax.grid(True)

        plt.show()
