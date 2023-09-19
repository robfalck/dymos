import matplotlib
import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import require_pyoptsparse

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


SHOW_PLOTS = True
# matplotlib.use('Agg')


@require_pyoptsparse(optimizer='SLSQP')
def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=True, optimizer='SLSQP', run_driver=True, force_alloc_complex=False,
                             solve_segments=False):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Major iterations limit'] = 200
    p.driver.declare_coloring(tol=1.0E-12)

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)
    elif transcription == 'shooting-gauss-lobatto':
        grid = dm.GaussLobattoGrid(num_segments=num_segments,
                                   nodes_per_seg=transcription_order,
                                   compressed=compressed)
        t = dm.ExplicitShooting(grid=grid)
    elif transcription == 'shooting-radau':
        grid = dm.RadauGrid(num_segments=num_segments,
                            nodes_per_seg=transcription_order + 1,
                            compressed=compressed)
        t = dm.ExplicitShooting(grid=grid)
    elif transcription == 'birkhoff':
        grid = dm.BirkhoffGrid(num_segments=num_segments,
                               nodes_per_seg=transcription_order + 1,
                               compressed=compressed, grid_type='cgl')
        t = dm.Birkhoff(grid=grid)
        # phase = dm.ImplicitPhase(ode_class=BrachistochroneODE, num_nodes=11)

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
    p.model.add_subsystem('traj0', traj)
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10))

    phase.add_state('x', fix_initial=False, fix_final=False, solve_segments=solve_segments)
    phase.add_state('y', fix_initial=False, fix_final=False, solve_segments=solve_segments)

    # Note that by omitting the targets here Dymos will automatically attempt to connect
    # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
    phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=solve_segments)

    phase.add_control('theta',
                      continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', targets=['g'], units='m/s**2')

    # phase.add_timeseries('timeseries2',
    #                      transcription=dm.Radau(num_segments=num_segments*5,
    #                                             order=transcription_order,
    #                                             compressed=compressed),
    #                      subset='control_input')

    phase.add_boundary_constraint('x', loc='initial', equals=0)
    phase.add_boundary_constraint('y', loc='initial', equals=10)

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    # phase.add_path_constraint('y', lower=0, upper=20)
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    phase.add_timeseries_output('check')

    p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

    phase.set_simulate_options(method='RK23')

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 1.8016

    if transcription.startswith('shooting'):
        p['traj0.phase0.initial_states:x'] = 0
        p['traj0.phase0.initial_states:y'] = 10
        p['traj0.phase0.initial_states:v'] = 0
    else:
        p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100.5])

        # p.set_val('traj0.phase0.states:x',
        #           [0.00000000e+00, 1.42250145e-03, 4.31023165e-02, 2.79830093e-01,
        #            8.48081379e-01, 1.56677729e+00, 1.99368459e+00, 1.99368459e+00,
        #            2.47686966e+00, 3.68121972e+00, 5.54133199e+00, 7.61206815e+00,
        #            9.24652275e+00, 1.00000000e+01])
        #
        # p.set_val('traj0.phase0.state_rates:x',
        #           np.asarray([0.          ,0.05604133,  0.53923264,  1.83190579,  3.63378098,  5.23521981,
        #            5.94581767  ,5.94581767,  6.63901393,  8.07546112,  9.42455534, 10.04657996,
        #            9.95258199  ,9.73726393]) * 0.4504)
        #
        # p.set_val('traj0.phase0.states:y',
        #           [10.,          9.97138214,  9.72441314,  9.06746221,  8.1348229,   7.32485517,
        #            6.94241351,  6.94241351,  6.57041326,  5.85051447,  5.1574541,   4.84062147,
        #            4.88908317,  5.])
        #
        # p.set_val('traj0.phase0.state_rates:y',
        #           np.asarray([0., -0.74693993, -2.26173074, -3.8612525, -4.82928417, -4.99333353,
        #            -4.94390406, -4.94390406, -4.80125032, -4.01427054, -2.4698582, -0.50315392,
        #            1.08742993, 1.80338364]) * 0.4504)
        #
        # p.set_val('traj0.phase0.states:v',
        #           [0.,          0.74919451,  2.32490085,  4.27669824,  6.04832824,  7.2435088,
        #            7.7439887,   7.7439887,   8.20155613,  9.02136839,  9.74568227, 10.05944468,
        #            10.01209018,  9.90285312])
        #
        # p.set_val('traj0.phase0.state_rates:v',
        #           np.asarray([9.80664981, 9.77916426, 9.53928033, 8.86007219, 7.83610596, 6.76847194,
        #            6.26986963, 6.26986963, 5.74673961, 4.36524639, 2.48604075, 0.49052294,
        #            -1.06514625, -1.78586432]) * 0.4504)
        #
        # p.set_val('traj0.phase0.controls:theta',
        #           [1.11374860e-02, 4.29074274e+00, 1.34098851e+01, 2.53812082e+01,
        #            3.69595266e+01, 4.63546862e+01, 5.02567239e+01, 5.02567239e+01,
        #            5.41260252e+01, 6.35682635e+01, 7.53149467e+01, 8.71329020e+01,
        #            9.62354647e+01, 1.00492539e+02])

        if transcription == 'birkhoff':
            segment_start_idxs = grid.segment_indices[:, 0]
            segment_end_idxs = grid.segment_indices[:, 1] - 1
            p['traj0.phase0.initial_states:x'] = phase.interp('x', [0, 10], nodes='all')[segment_start_idxs, ...]
            p['traj0.phase0.final_states:x'] = phase.interp('x', [0, 10], nodes='all')[segment_end_idxs, ...]
            p['traj0.phase0.initial_states:y'] = phase.interp('y', [10, 5], nodes='all')[segment_start_idxs, ...]
            p['traj0.phase0.final_states:y'] = phase.interp('y', [10, 5], nodes='all')[segment_end_idxs, ...]
            p['traj0.phase0.initial_states:v'] = phase.interp('v', [0, 9.9], nodes='all')[segment_start_idxs, ...]
            p['traj0.phase0.final_states:v'] = phase.interp('v', [0, 9.9], nodes='all')[segment_end_idxs, ...]
            # p.set_val('traj0.phase0.initial_states:x', [0.0, 1.99368459e+00])
            # p.set_val('traj0.phase0.initial_states:y', [10.0, 6.94241351])
            # p.set_val('traj0.phase0.initial_states:v', [0.0, 7.7439887])
            #
            # p.set_val('traj0.phase0.final_states:x', [1.99368459e+00, 10.0])
            # p.set_val('traj0.phase0.final_states:y', [6.94241351, 5.0])
            # p.set_val('traj0.phase0.final_states:v', [7.7439887, 9.90285312])



    # p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
    p['traj0.phase0.parameters:g'] = 9.80665

    p.run_model()

    dm.run_problem(p, run_driver=run_driver, simulate=False, make_plots=True,
                   simulate_kwargs={'times_per_seg': 100})

    # p.model.list_inputs(print_arrays=True)
    # p.model.list_outputs(print_arrays=True)

    return p


if __name__ == '__main__':

    with dm.options.temporary(include_check_partials=True):
        p = brachistochrone_min_time(transcription='birkhoff', num_segments=2, run_driver=True,
                                     transcription_order=18, compressed=False, optimizer='SNOPT',
                                     solve_segments=False, force_alloc_complex=True)

        J = p.compute_totals(return_format='array')
        print(np.linalg.cond(J))

        # print('x')
        # print(p.get_val('traj0.phase0.timeseries.x').T)
        # print('y')
        # print(p.get_val('traj0.phase0.timeseries.y').T)
        # print('v')
        # print(p.get_val('traj0.phase0.timeseries.v').T)
        # print('theta')
        # print(p.get_val('traj0.phase0.timeseries.theta').T)
        # p.check_totals(method='cs', compact_print=True)

        # p.list_problem_vars(print_arrays=True)

        # p.model.list_outputs(print_arrays=True, includes=['*ode_all*', '*collocation_comp*'])
        # p.model.list_inputs(print_arrays=True, includes=['*ode_all*', '*collocation_comp*'])

        # with np.printoptions(linewidth=1024):
        #     print(p.get_val('traj0.phase0.ode_all.xdot').T)
        #     print(p.get_val('traj0.phase0.state_rates:x').T)
        #     print(p.get_val('traj0.phase0.f_computed:x').T)
        #
        #     print(p.get_val('traj0.phase0.state_rate_defects:x').T)
        #     print(p.get_val('traj0.phase0.state_rate_defects:y').T)
        #     print(p.get_val('traj0.phase0.state_rate_defects:v').T)
        #
        #     print(p.get_val('traj0.phase0.state_defects:x').T)
        #     print(p.get_val('traj0.phase0.state_defects:y').T)
        #     print(p.get_val('traj0.phase0.state_defects:v').T)
        #
        #     p.check_partials(method='cs', compact_print=True)

        # p.check_partials(method='cs', compact_print=True)
        # om.n2(p.model)