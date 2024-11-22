import numpy as np
import openmdao.api as om

import dymos as dm


class SimpleCostateODE(om.ExplicitComponent):
    """
    A simple ODE for testing costate estimation.

    This example is taken from Example 1 of `Direct Trajectory Optimization and Costate
    Estimation of Finite-Horizon and Infinite-Horizon Optimal Control Problems
    Using a Radau Pseudospectral Method` by Garg, et.al.

    https://www.anilvrao.com/Publications/JournalPublications/COAP818.pdf
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cite = ('@article{garg2011direct\n'
                     '  title={Direct trajectory optimization and costate estimation of finite-horizon '
                     '         and infinite-horizon optimal control problems using a Radau pseudospectral method},\n'
                     '  author={Garg, Divya and Patterson, Michael A and Francolin, Camila and Darby,\n'
                     '          Christopher L and Huntington, Geoffrey T and Hager, William W and Rao, Anil V},\n'
                     '  journal={Computational Optimization and Applications},\n'
                     '  volume={49},\n'
                     '  pages={335--358},\n'
                     '  year={2011},\n'
                     '  publisher={Springer}\n'
                     '}')

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # time varying inputs
        self.add_input('y', units='unitless', shape=nn)
        self.add_input('u', units='unitless', shape=nn)

        # state rates
        self.add_output('y_dot', shape=nn, units='unitless/s',
                        tags=['dymos.state_rate_source:y', 'dymos.state_units:unitless'])
        self.add_output('J_dot', shape=nn, units='unitless/s',
                        tags=['dymos.state_rate_source:J', 'dymos.state_units:unitless'])

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='y_dot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='y_dot', wrt='u', rows=ar, cols=ar)

        self.declare_partials(of='J_dot', wrt='y', rows=ar, cols=ar, val=0.5)
        self.declare_partials(of='J_dot', wrt='u', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        y, u = inputs.values()

        outputs['y_dot'] = 2 * y + 2 * u * np.sqrt(y)
        outputs['J_dot'] = 0.5 * (y + u ** 2)

    def compute_partials(self, inputs, partials):
        y, u = inputs.values()
        sqrt_y = np.sqrt(y)

        partials['y_dot', 'y'] = 2 + 2 * u / sqrt_y
        partials['y_dot', 'u'] = 2 * sqrt_y

        partials['J_dot', 'u'] = y * u


class FahrooRossOrbitCostateODE(om.ExplicitComponent):
    """
    An example from Fahroo and Ross for the demonstration of costate estimation.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cite = ('@article{doi:10.2514/2.4709,\n'
                     '  author = {Fahroo, Fariba and Ross, I. Michael},\n'
                     '  title = {Costate Estimation by a Legendre Pseudospectral Method},\n'
                     '  journal = {Journal of Guidance, Control, and Dynamics},\n'
                     '  volume = {24},\n'
                     '  number = {2},\n'
                     '  pages = {270-277},\n'
                     '  year = {2001},\n'
                     '  doi = {10.2514/2.4709},\n'
                     '}')

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # time varying inputs
        self.add_input('r', units='unitless', shape=nn)
        self.add_input('u', units='unitless', shape=nn)
        self.add_input('v', units='unitless', shape=nn)
        self.add_input('epsilon', units='unitless', shape=nn)

        # state rates
        self.add_output('r_dot', shape=nn, units='unitless/s',
                        tags=['dymos.state_rate_source:r', 'dymos.state_units:unitless'])

        self.add_output('theta_dot', shape=nn, units='unitless/s',
                        tags=['dymos.state_rate_source:theta', 'dymos.state_units:unitless'])

        self.add_output('u_dot', shape=nn, units='unitless/s',
                        tags=['dymos.state_rate_source:u', 'dymos.state_units:unitless'])

        self.add_output('v_dot', shape=nn, units='unitless/s',
                        tags=['dymos.state_rate_source:v', 'dymos.state_units:unitless'])

        self.add_output('E', shape=nn, units='unitless')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='r_dot', wrt='u', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='theta_dot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='theta_dot', wrt='r', rows=ar, cols=ar)

        self.declare_partials(of='u_dot', wrt='r', rows=ar, cols=ar)
        # self.declare_partials(of='u_dot', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='u_dot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='u_dot', wrt='epsilon', rows=ar, cols=ar)

        self.declare_partials(of='v_dot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='epsilon', rows=ar, cols=ar)

        self.declare_partials(of='E', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='E', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='E', wrt='r', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        r, u, v, epsilon = inputs.values()

        outputs['r_dot'] = u
        outputs['theta_dot'] = v / r
        outputs['u_dot'] = v**2 / r - 1 / r ** 2 + 0.01 * np.sin(epsilon)
        outputs['v_dot'] = - u * v / r + 0.01 * np.cos(epsilon)

        outputs['E'] = (0.5 * (u**2 + v**2) - 1 / r)

    def compute_partials(self, inputs, partials):
        r, u, v, epsilon = inputs.values()

        partials['theta_dot', 'v'] = 1 / r
        partials['theta_dot', 'r'] = - v / r**2

        partials['u_dot', 'r'] = -v**2 / r**2 + 2 / r**3
        partials['u_dot', 'v'] = 2 * v / r
        partials['u_dot', 'epsilon'] = 0.01 * np.cos(epsilon)

        partials['v_dot', 'r'] = u * v / r ** 2
        partials['v_dot', 'u'] = - v / r
        partials['v_dot', 'v'] = - u / r
        partials['v_dot', 'epsilon'] = -0.01 * np.sin(epsilon)

        partials['E', 'r'] = 1 / r ** 2
        partials['E', 'u'] = u
        partials['E', 'v'] = v


def costate_estimation_example_fahroo():

    p = om.Problem()

    p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['mu_init'] = 1e-3
    # p.driver.opt_settings['max_iter'] = 500
    # p.driver.opt_settings['acceptable_tol'] = 1e-5
    # p.driver.opt_settings['constr_viol_tol'] = 1e-6
    # p.driver.opt_settings['compl_inf_tol'] = 1e-6
    p.driver.opt_settings['tol'] = 1e-8
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
    p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
    p.driver.opt_settings['mu_strategy'] = 'monotone'
    p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'

    # p.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    # p.driver.opt_settings['iSumm'] = 6

    p.driver.declare_coloring()

    tx = dm.Radau(num_segments=30, order=3)
    # tx = dm.GaussLobatto(num_segments=50, order=3)
    # tx = dm.Birkhoff(num_nodes=60)

    traj = dm.Trajectory()
    phase = traj.add_phase('phase', phase=dm.Phase(ode_class=FahrooRossOrbitCostateODE,
                                                   transcription=tx))

    p.model.add_subsystem('traj', traj)

    phase.set_time_options(fix_initial=True, fix_duration=True)
    phase.set_state_options('r', fix_initial=True, fix_final=False)
    phase.set_state_options('theta', fix_initial=True, fix_final=False)
    phase.set_state_options('u', fix_initial=True, fix_final=False)
    phase.set_state_options('v', fix_initial=True, fix_final=False)
    phase.add_control('epsilon', units='unitless')#, rate_continuity_ref=1E3, continuity_ref=1E2)

    phase.add_objective('E', ref=-1.0)
    phase.add_timeseries_output('E')

    p.setup(force_alloc_complex=True)

    phase.simulate_options.set(times_per_seg=100)
    phase.set_time_val(initial=0, duration=50)

    phase.set_state_val('r', [1.1, 3])
    phase.set_state_val('theta', [0, 12])
    phase.set_state_val('u', [0, 0.15])
    phase.set_state_val('v', [1/np.sqrt(1.1), 0.5])
    phase.set_control_val('epsilon', [0.001, 0.001])

    # Start the problem by simulating the initial control guess (per the reference)
    dm.run_problem(p, run_driver=False, simulate=True, make_plots=False)

    # Now load the simulated trjaectory guess as the restart.
    dm.run_problem(p, run_driver=True, simulate=True, make_plots=True,
                   restart=traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db')

    t = phase.get_val('timeseries.time')
    r = phase.get_val('timeseries.r')
    theta = phase.get_val('timeseries.theta')
    u = phase.get_val('timeseries.u')
    v = phase.get_val('timeseries.v')
    epsilon = phase.get_val('timeseries.epsilon')
    E = phase.get_val('timeseries.E')

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    lam_dict = {}


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x, y, '-', label='x-y')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-3, 4)
    ax.set_ylim(-3, 5)
    ax.legend()
    fig.savefig(f'/Users/rfalck/Desktop/{tx.__class__.__name__.lower()}_fahroo_x_y.png')


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(t, epsilon, '-', label=r'$\epsilon$')
    ax.set_xlabel('t')
    ax.set_ylabel('Thrust angle (rad)')
    ax.set_ylim(-0.05, 0.35)
    ax.legend()
    fig.savefig(f'/Users/rfalck/Desktop/{tx.__class__.__name__.lower()}_fahroo_epsilon.png')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(t, u, '-', label='u')
    ax.plot(t, v, '-', label='v')
    ax.set_xlabel('t')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(f'/Users/rfalck/Desktop/{tx.__class__.__name__.lower()}_fahroo_u_v.png')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(t, E, '-', label='E')
    ax.set_xlabel('t')
    ax.legend()
    fig.savefig(f'/Users/rfalck/Desktop/{tx.__class__.__name__.lower()}_fahroo_E.png')

    # plt.show()

    p.list_driver_vars()

    # return

    gd = tx.grid_data
    lambdas = p.driver.pyopt_solution.lambdaStar
    plot_costates = True
    with np.printoptions(linewidth=10000):
        for con_name, lamb in lambdas.items():
            if isinstance(tx, (dm.Radau, dm.GaussLobatto)):
                if 'collocation_constraint' in con_name:
                    state_name = con_name.split(":")[-1]
                    # print(f'{state_name} values')
                    # print(p.get_val(f'traj.phase.timeseries.{state_name}').ravel())
                    # print(f'\u03BB {state_name}')
                    # print(lamb / gd.node_weight[gd.subset_node_indices['col']])
                    lam_dict[state_name] = lamb / gd.node_weight[gd.subset_node_indices['col']]
            elif isinstance(tx, dm.Birkhoff):
                om.issue_warning('Birkhoff does not implement covector mapping yet.')
                plot_costates = False
            else:
                om.issue_warning(f'Transcirption {tx.__class__.__name__} does not support covector mapping.')

    if plot_costates:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(t[gd.subset_node_indices['col']], lam_dict['r'], '-', label=r'$\lambda_r$')
        ax.plot(t[gd.subset_node_indices['col']], lam_dict['u'], '-', label=r'$\lambda_u$')
        ax.plot(t[gd.subset_node_indices['col']], lam_dict['v'], '-', label=r'$\lambda_v$')
        ax.set_xlabel('t')
        ax.set_ylim(-0.8, 0.3)
        ax.legend()
        fig.savefig(f'/Users/rfalck/Desktop/{tx.__class__.__name__.lower()}_fahroo_costates.png')

    plt.show()



if __name__ == '__main__':

    p = costate_estimation_example_fahroo()
