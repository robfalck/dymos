import unittest

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal

from dymos.utils.misc import om_version


SHOW_PLOTS = True


# @use_tempdirs
class TestBalancedFieldLengthForDocs(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    def test_balanced_field_length_solved(self):
        import openmdao.api as om
        from openmdao.utils.general_utils import set_pyoptsparse_opt
        import dymos as dm
        from dymos.examples.balanced_field.balanced_field_ode import BalancedFieldODEComp

        p = om.Problem()

        # First Phase: Brake release to V1 - both engines operable
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=11)
        br_to_v1 = dm.Phase(ode_class=BalancedFieldODEComp, transcription=tx,
                            ode_init_kwargs={'mode': 'runway',
                                             'attitude_input': 'pitch',
                                             'control': 'gam_rate'})
        br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000))
        br_to_v1.add_state('r', fix_initial=True, lower=0)
        br_to_v1.add_state('v', fix_initial=True, lower=0)
        br_to_v1.add_parameter('pitch', val=0.0, opt=False, units='deg')
        br_to_v1.add_parameter('v1', val=150.0, opt=False, units='kn')
        br_to_v1.add_calc_expr('v_to_go = v - v1',
                               v={'shape': (1,), 'units': 'kn'},
                               v1={'shape': (1,), 'units': 'kn'},
                               v_to_go={'shape': (1,), 'units': 'kn'})
        br_to_v1.add_boundary_balance(param='t_duration', name='v_to_go', tgt_val=0.0, loc='final')
        br_to_v1.add_timeseries_output('*')

        # br_to_v1.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        # br_to_v1.linear_solver = om.DirectSolver()

        # Second Phase: Rejected takeoff at V1 - no engines operable
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=11)
        rto = dm.Phase(ode_class=BalancedFieldODEComp, transcription=tx,
                       ode_init_kwargs={'mode': 'runway',
                                        'attitude_input': 'pitch',
                                        'control': 'gam_rate'})
        rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        rto.add_state('r', fix_initial=False, lower=0)
        rto.add_state('v', fix_initial=False, lower=0)
        rto.add_parameter('pitch', val=0.0, opt=False, units='deg')
        rto.add_boundary_balance(param='t_duration', name='v', tgt_val=0.0, loc='final')
        rto.add_timeseries_output('*')

        # rto.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        # rto.linear_solver = om.DirectSolver()

        # Third Phase: V1 to Vr - single engine operable
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=11)
        v1_to_vr = dm.Phase(ode_class=BalancedFieldODEComp, transcription=tx,
                            ode_init_kwargs={'mode': 'runway',
                                             'attitude_input': 'pitch',
                                             'control': 'gam_rate'})
        v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        v1_to_vr.add_state('r', fix_initial=False, lower=0)
        v1_to_vr.add_state('v', fix_initial=False, lower=0)
        v1_to_vr.add_parameter('pitch', val=0.0, opt=False, units='deg')
        v1_to_vr.add_boundary_balance(param='t_duration', name='v_over_v_stall', tgt_val=1.2)
        v1_to_vr.add_timeseries_output('*')

        # v1_to_vr.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        # v1_to_vr.linear_solver = om.DirectSolver()

        # Fourth Phase: Rotate - single engine operable
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=11)
        rotate = dm.Phase(ode_class=BalancedFieldODEComp, transcription=tx,
                          ode_init_kwargs={'mode': 'runway',
                                           'attitude_input': 'pitch',
                                           'control': 'gam_rate'})
        rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 10.), duration_ref=1.0)
        rotate.add_state('r', fix_initial=False, lower=0)
        rotate.add_state('v', fix_initial=False, lower=0)
        rotate.add_state('pitch', rate_source='pitch_rate', fix_initial=True, lower=0, upper=15, units='deg')
        rotate.add_parameter('pitch_rate', opt=False, units='deg/s')
        rotate.add_boundary_balance(param='t_duration', name='F_r', tgt_val=0.0, )
        # rotate.add_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10, ref=10, val=[0, 10],
        #                    control_type='polynomial')
        rotate.add_timeseries_output('*')
        rotate.add_timeseries_output('alpha', units='deg')

        # Fifth Phase: Liftoff of ground and continue rotation until desired climb gradient is hit.
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=9)
        liftoff_to_pitch_limit = dm.Phase(ode_class=BalancedFieldODEComp, transcription=tx,
                                          ode_init_kwargs={'mode': 'runway',
                                                           'attitude_input': 'pitch',
                                                           'control': 'attitude'})
        liftoff_to_pitch_limit.set_time_options(fix_initial=True, fix_duration=True, duration_ref=1.0, duration_bounds=(0.1, 4.0))
        liftoff_to_pitch_limit.set_state_options('r', fix_initial=True, lower=0)
        liftoff_to_pitch_limit.set_state_options('h', fix_initial=True, rate_source='h_dot', lower=0)
        liftoff_to_pitch_limit.set_state_options('v', fix_initial=True, lower=0)
        liftoff_to_pitch_limit.set_state_options('gam', fix_initial=True, rate_source='gam_dot', lower=0)
        liftoff_to_pitch_limit.set_state_options('pitch', fix_initial=True, rate_source='pitch_rate', opt=False, units='deg')
        liftoff_to_pitch_limit.add_parameter('pitch_rate', opt=False, units='deg/s')
        liftoff_to_pitch_limit.add_parameter('mu_r', opt=False, val=0.0, units=None)
        liftoff_to_pitch_limit.add_boundary_balance(param='t_duration', name='climb_gradient', tgt_val=0.04, loc='final', lower=0, upper=4)
        liftoff_to_pitch_limit.add_timeseries_output('alpha', units='deg')
        liftoff_to_pitch_limit.add_timeseries_output('h', units='ft')
        liftoff_to_pitch_limit.add_timeseries_output('*')

        # Sixth Phase: Assume constant flight path angle until 35ft altitude is reached.
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=9)
        climb_to_obstacle_clearance = dm.Phase(ode_class=BalancedFieldODEComp, transcription=tx,
                                               ode_init_kwargs={'mode': 'runway',
                                                                'attitude_input': 'pitch',
                                                                'control': 'gam_rate'})
        climb_to_obstacle_clearance.set_time_options(fix_initial=True, fix_duration=True, duration_ref=1.0, duration_bounds=(0.01, 100.0))
        climb_to_obstacle_clearance.set_state_options('r', fix_initial=True, lower=0)
        climb_to_obstacle_clearance.set_state_options('h', fix_initial=True, rate_source='h_dot', lower=0)
        climb_to_obstacle_clearance.set_state_options('v', fix_initial=True, lower=0)
        climb_to_obstacle_clearance.set_state_options('gam', fix_initial=True, rate_source='gam_dot', lower=0)
        climb_to_obstacle_clearance.set_state_options('pitch', fix_initial=True, rate_source='pitch_rate', opt=False, units='deg')
        climb_to_obstacle_clearance.add_parameter('pitch_rate', opt=False, units='deg/s')
        climb_to_obstacle_clearance.add_parameter('gam_rate', opt=False, units='deg/s')
        climb_to_obstacle_clearance.add_parameter('mu_r', opt=False, val=0.0, units=None)
        climb_to_obstacle_clearance.add_boundary_balance(param='t_duration', name='h', tgt_val=35, eq_units='ft', loc='final', lower=0.01, upper=100)
        climb_to_obstacle_clearance.add_timeseries_output('alpha', units='deg')
        climb_to_obstacle_clearance.add_timeseries_output('h', units='ft')
        climb_to_obstacle_clearance.add_timeseries_output('*')

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory(parallel_phases=False)
        p.model.add_subsystem('traj', traj)
        traj.add_phase('br_to_v1', br_to_v1)
        traj.add_phase('rto', rto)
        traj.add_phase('v1_to_vr', v1_to_vr)
        traj.add_phase('rotate', rotate)
        traj.add_phase('liftoff_to_pitch_limit', liftoff_to_pitch_limit)
        traj.add_phase('climb_to_obstacle_clearance', climb_to_obstacle_clearance)
        # traj.add_phase('climb', climb)

        # traj.phases.nonlinear_solver = om.NonlinearBlockJac()
        # traj.phases.linear_solver = om.DirectSolver()

        traj.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, maxiter=100)
        traj.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        traj.linear_solver = om.DirectSolver()

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter('m', val=174200., opt=False, units='lbm',
                           desc='aircraft mass',
                           targets={phase: ['m'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})
                                    #'v1_to_vr': ['m'], 'rto': ['m'],
                                    # 'rotate': ['m'], 'climb': ['m']})

        traj.add_parameter('T_nominal', val=27000 * 2, opt=False, units='lbf', static_target=True,
                           desc='nominal aircraft thrust',
                           targets={'br_to_v1': ['T']})

        traj.add_parameter('T_engine_out', val=27000, opt=False, units='lbf', static_target=True,
                           desc='thrust under a single engine',
                           targets={phase: ['T'] for phase in ['v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})
                           #, 'rotate': ['T'], 'climb': ['T']})

        traj.add_parameter('T_shutdown', val=0.0, opt=False, units='lbf', static_target=True,
                           desc='thrust when engines are shut down for rejected takeoff',
                           targets={'rto': ['T']})

        traj.add_parameter('mu_r_nominal', val=0.03, opt=False, units=None, static_target=True,
                           desc='nominal runway friction coefficient',
                           targets={phase: ['mu_r'] for phase in ['br_to_v1', 'v1_to_vr', 'rotate']})
                        #    targets={'br_to_v1': ['mu_r'], 'v1_to_vr': ['mu_r']})#,  'rotate': ['mu_r']})

        traj.add_parameter('mu_r_braking', val=0.3, opt=False, units=None, static_target=True,
                           desc='runway friction coefficient under braking',
                           targets={'rto': ['mu_r']})

        traj.add_parameter('h_runway', val=0., opt=False, units='ft',
                           desc='runway altitude',
                           targets={phase: ['h'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate']})
                                    # 'v1_to_vr': ['h'], 'rto': ['h'],
                                    # 'rotate': ['h']})

        traj.add_parameter('rho', val=1.225, opt=False, units='kg/m**3', static_target=True,
                           desc='atmospheric density',
                           targets={phase: ['rho'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})
                                    # 'v1_to_vr': ['rho'], 'rto': ['rho'],
                                    # 'rotate': ['rho']})

        traj.add_parameter('S', val=124.7, opt=False, units='m**2', static_target=True,
                           desc='aerodynamic reference area',
                           targets={f'{phase}': ['S'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        traj.add_parameter('CD0', val=0.03, opt=False, units=None, static_target=True,
                           desc='zero-lift drag coefficient',
                           targets={f'{phase}': ['CD0'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        traj.add_parameter('AR', val=9.45, opt=False, units=None, static_target=True,
                           desc='wing aspect ratio',
                           targets={f'{phase}': ['AR'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        traj.add_parameter('e', val=801, opt=False, units=None, static_target=True,
                           desc='Oswald span efficiency factor',
                           targets={f'{phase}': ['e'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        traj.add_parameter('span', val=35.7, opt=False, units='m', static_target=True,
                           desc='wingspan',
                           targets={f'{phase}': ['span'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        traj.add_parameter('h_w', val=1.0, opt=False, units='m', static_target=True,
                           desc='height of wing above CG',
                           targets={f'{phase}': ['h_w'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        traj.add_parameter('CL0', val=0.5, opt=False, units=None, static_target=True,
                           desc='zero-alpha lift coefficient',
                           targets={f'{phase}': ['CL0'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        traj.add_parameter('CL_max', val=2.0, opt=False, units=None, static_target=True,
                           desc='maximum lift coefficient for linear fit',
                           targets={f'{phase}': ['CL_max'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        traj.add_parameter('alpha_max', val=10.0, opt=False, units='deg', static_target=True,
                           desc='angle of attack at maximum lift',
                           targets={f'{phase}': ['alpha_max'] for phase in ['br_to_v1', 'rto', 'v1_to_vr', 'rotate', 'liftoff_to_pitch_limit', 'climb_to_obstacle_clearance']})#, 'v1_to_vr',
                                                                      #'rto', 'rotate', 'climb']})

        # Standard "end of first phase to beginning of second phase" linkages
        # Alpha changes from being a parameter in v1_to_vr to a polynomial control
        # in rotate, to a dynamic control in `climb`.
        # traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
        # traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
        # traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'], connected=True)
        traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'], connected=True)
        traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v'], connected=True)
        traj.link_phases(['rotate', 'liftoff_to_pitch_limit'], vars=['time', 'r', 'v', 'pitch'], connected=True)
        traj.link_phases(['liftoff_to_pitch_limit', 'climb_to_obstacle_clearance'],
                         vars=['time', 'h', 'gam', 'r', 'v', 'pitch'],
                         connected=True)

        # Less common "final value of r must be the match at ends of two phases".
        # traj.add_linkage_constraint(phase_a='rto', var_a='r', loc_a='final',
        #                             phase_b='climb', var_b='r', loc_b='final',
        #                             ref=1000)

        # # Define the constraints and objective for the optimal control problem
        # v1_to_vr.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=100)

        # rto.add_boundary_constraint('v', loc='final', equals=0., ref=100, linear=True)

        # rotate.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

        # climb.add_boundary_constraint('h', loc='final', equals=35, ref=35, units='ft', linear=True)
        # climb.add_boundary_constraint('gam', loc='final', equals=5, ref=5, units='deg', linear=True)
        # climb.add_path_constraint('gam', lower=0, upper=5, ref=5, units='deg')
        # climb.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.25, ref=1.25)

        # rto.add_objective('r', loc='final', ref=1.0)

        # for phase_name, phase in traj._phases.items():
        #     if 'T_nominal' in phase.parameter_options:
        #         phase.add_timeseries_output('T_nominal', output_name='T')
        #     if 'T_engine_out' in phase.parameter_options:
        #         phase.add_timeseries_output('T_engine_out', output_name='T')
        #     if 'T_shutdown' in phase.parameter_options:
        #         phase.add_timeseries_output('T_shutdown', output_name='T')
        #     phase.add_timeseries_output('alpha')

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        br_to_v1.set_time_val(initial=0.0, duration=35.0)
        br_to_v1.set_state_val('r', [0, 2500.0])
        br_to_v1.set_state_val('v', [0.0, 100.0])
        br_to_v1.set_parameter_val('pitch', 0.0, units='deg')

        # v1_to_vr.set_time_val(initial=35.0, duration=35.0)
        # v1_to_vr.set_state_val('r', [2500, 300.0])
        # v1_to_vr.set_state_val('v', [100, 110.0])
        # v1_to_vr.set_parameter_val('alpha', 0.0, units='deg')

        rto.set_time_val(initial=30.0, duration=35.0)
        rto.set_state_val('r', [2500, 5000.0])
        rto.set_state_val('v', [100, 0.0], units='kn')
        rto.set_parameter_val('pitch', 0.0, units='deg')

        rotate.set_time_val(initial=30.0, duration=5.0)
        rotate.set_state_val('r', [1750, 1800.0])
        rotate.set_state_val('v', [80, 85.0])
        rotate.set_state_val('pitch', [0.0, 10], units='deg')
        rotate.set_parameter_val('pitch_rate', val=1.0, units='deg/s')

        liftoff_to_pitch_limit.set_time_val(initial=35.0, duration=4.1)
        liftoff_to_pitch_limit.set_state_val('r', [1800, 2000.0], units='ft')
        liftoff_to_pitch_limit.set_state_val('v', [160, 170.0], units='kn')
        liftoff_to_pitch_limit.set_state_val('h', [0.0, 35.0], units='ft')
        liftoff_to_pitch_limit.set_state_val('gam', [0.0, 5.0], units='deg')
        liftoff_to_pitch_limit.set_state_val('pitch', [5.0, 15.0], units='deg')
        liftoff_to_pitch_limit.set_parameter_val('pitch_rate', 0.8, units='deg/s')
        liftoff_to_pitch_limit.set_parameter_val('mu_r', 0.0, units=None)

        climb_to_obstacle_clearance.set_time_val(initial=40.0, duration=2)
        climb_to_obstacle_clearance.set_state_val('r', [1800, 2000.0], units='ft')
        climb_to_obstacle_clearance.set_state_val('v', [160, 170.0], units='kn')
        climb_to_obstacle_clearance.set_state_val('h', [25.0, 35.0], units='ft')
        climb_to_obstacle_clearance.set_state_val('gam', [0.0, 5.0], units='deg')
        climb_to_obstacle_clearance.set_state_val('pitch', [5.0, 15.0], units='deg')
        climb_to_obstacle_clearance.set_parameter_val('pitch_rate', 0.0, units='deg/s')
        climb_to_obstacle_clearance.set_parameter_val('gam_rate', 0.00, units='deg/s')
        climb_to_obstacle_clearance.set_parameter_val('mu_r', 0.0, units=None)

        # climb.set_time_val(initial=75.0, duration=15.0)
        # climb.set_state_val('r', [5000, 5500.0], units='ft')
        # climb.set_state_val('v', [160, 170.0], units='kn')
        # climb.set_state_val('h', [0.0, 35.0], units='ft')
        # climb.set_state_val('gam', [0.0, 5.0], units='deg')
        # climb.set_control_val('alpha', 5.0, units='deg')

        p.final_setup()
        # om.n2(p)

        dm.run_problem(p, run_driver=False, simulate=False, make_plots=True)

        print(p.get_reports_dir())

        # om.n2(p)

        # sol_db = 'dymos_solution.db'
        # sim_db = 'dymos_simulation.db'
        # if om_version()[0] > (3, 34, 2):
        #     sol_db = p.get_outputs_dir() / sol_db
        #     sim_db = traj.sim_prob.get_outputs_dir() / sim_db

        # sol = om.CaseReader(sol_db).get_case('final')
        # sim = om.CaseReader(sim_db).get_case('final')

        # sol_r_f_climb = sol.get_val('traj.climb.timeseries.r')[-1, ...]
        # sol_r_f_rto = sol.get_val('traj.rto.timeseries.r')[-1, ...]
        # sim_r_f_climb = sim.get_val('traj.climb.timeseries.r')[-1, ...]
        # sim_r_f_rto = sim.get_val('traj.rto.timeseries.r')[-1, ...]

        # assert_near_equal(2114.387, sol_r_f_climb, tolerance=0.01)
        # assert_near_equal(2114.387, sol_r_f_rto, tolerance=0.01)
        # assert_near_equal(2114.387, sim_r_f_climb, tolerance=0.01)
        # assert_near_equal(2114.387, sim_r_f_rto, tolerance=0.01)
