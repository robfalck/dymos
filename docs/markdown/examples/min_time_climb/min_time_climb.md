```python
# tags: active-ipynb, remove-input, remove-output
# This cell is mandatory in all Dymos documentation notebooks.
missing_packages = []
try:
    import openmdao.api as om  # noqa: F401
except ImportError:
    if 'google.colab' in str(get_ipython()):
        !python -m pip install openmdao[notebooks]
    else:
        missing_packages.append('openmdao')
try:
    import dymos as dm  # noqa: F401
except ImportError:
    if 'google.colab' in str(get_ipython()):
        !python -m pip install dymos
    else:
        missing_packages.append('dymos')
try:
    import pyoptsparse  # noqa: F401
except ImportError:
    if 'google.colab' in str(get_ipython()):
        !pip install -q condacolab
        import condacolab
        condacolab.install_miniconda()
        !conda install -c conda-forge pyoptsparse
    else:
        missing_packages.append('pyoptsparse')
if missing_packages:
    raise EnvironmentError('This notebook requires the following packages '
                           'please install them and restart this notebook\'s runtime: {",".join(missing_packages)}')
```

(examples:minimum_time_climb)=
# Supersonic Interceptor Minimum Time Climb

This example is based on the _A/C Min Time to Climb_ example given in
chapter 4 of Bryson {cite}`bryson1999dynamic`. It finds the
angle-of-attack history required to accelerate a supersonic interceptor
from near ground level, Mach 0.4 to an altitude of 20 km and Mach 1.0.

![The free-body-diagram of the min-time-climb problem.](min_time_climb_fbd.png)

The vehicle dynamics are given by

\begin{align}
  \frac{dv}{dt} &= \frac{T}{m} \cos \alpha - \frac{D}{m} - g \sin \gamma \\
  \frac{d\gamma}{dt} &= \frac{T}{m v} \sin \alpha + \frac{L}{m v} - \frac{g \cos \gamma}{v} \\
  \frac{dh}{dt} &= v \sin \gamma \\
  \frac{dr}{dt} &= v \cos \gamma \\
  \frac{dm}{dt} &= - \frac{T}{g I_{sp}}
\end{align}

The initial conditions are

\begin{align}
  r_0 &= 0 \rm{\,m} \\
  h_0 &= 100 \rm{\,m} \\
  v_0 &= 135.964 \rm{\,m/s} \\
  \gamma_0 &= 0 \rm{\,deg} \\
  m_0 &= 19030.468 \rm{\,kg}
\end{align}

and the final conditions are

\begin{align}
  h_f &= 20000 \rm{\,m} \\
  M_f &= 1.0 \\
  \gamma_0 &= 0 \rm{\,deg}
\end{align}

## The ODE System: min_time_climb_ode.py

The top level ODE definition is a _Group_ that connects several subsystems.

```python
# tags: hide-output, remove-input
om.display_source("dymos.examples.min_time_climb.doc.aero_partial_coloring")
```

```python
import openmdao.api as om
from dymos.models.eom import FlightPathEOM2D
from dymos.examples.min_time_climb.prop import PropGroup
from dymos.models.atmosphere import USatm1976Comp
from dymos.examples.min_time_climb.doc.aero_partial_coloring import AeroGroup


class MinTimeClimbODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('fd', types=bool, default=False, desc='If True, use fd for partials')
        self.options.declare('partial_coloring', types=bool, default=False,
                             desc='If True and fd is True, color the approximated partials')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn, h_def='geodetic'),
                           promotes_inputs=['h'])

        self.add_subsystem(name='aero',
                           subsys=AeroGroup(num_nodes=nn,
                                            fd=self.options['fd'],
                                            partial_coloring=self.options['partial_coloring']),
                           promotes_inputs=['v', 'alpha', 'S'])

        self.connect('atmos.sos', 'aero.sos')
        self.connect('atmos.rho', 'aero.rho')

        self.add_subsystem(name='prop',
                           subsys=PropGroup(num_nodes=nn),
                           promotes_inputs=['h', 'Isp', 'throttle'])

        self.connect('aero.mach', 'prop.mach')

        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightPathEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'gam', 'alpha'])

        self.connect('aero.f_drag', 'flight_dynamics.D')
        self.connect('aero.f_lift', 'flight_dynamics.L')
        self.connect('prop.thrust', 'flight_dynamics.T')

```

## Building and running the problem

In the following code we follow the following process to solve the
problem:

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
# tags: output_scroll
import matplotlib.pyplot as plt

import openmdao.api as om

import dymos as dm
from dymos.examples.plotting import plot_results

#
# Instantiate the problem and configure the optimization driver
#
p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.declare_coloring()

#
# Instantiate the trajectory and phase
#
traj = dm.Trajectory()

phase = dm.Phase(ode_class=MinTimeClimbODE,
                 transcription=dm.GaussLobatto(num_segments=15, compressed=False))

traj.add_phase('phase0', phase)

p.model.add_subsystem('traj', traj)

#
# Set the options on the optimization variables
# Note the use of explicit state units here since much of the ODE uses imperial units
# and we prefer to solve this problem using metric units.
#
phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                       duration_ref=100.0)

phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6, units='m',
                ref=1.0E3, defect_ref=1.0E3,
                rate_source='flight_dynamics.r_dot')

phase.add_state('h', fix_initial=True, lower=0, upper=20000.0, units='m',
                ref=1.0E2, defect_ref=1.0E2,
                rate_source='flight_dynamics.h_dot')

phase.add_state('v', fix_initial=True, lower=10.0, units='m/s',
                ref=1.0E2, defect_ref=1.0E2,
                rate_source='flight_dynamics.v_dot')

phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5, units='rad',
                ref=1.0, defect_ref=1.0,
                rate_source='flight_dynamics.gam_dot')

phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5, units='kg',
                ref=1.0E3, defect_ref=1.0E3,
                rate_source='prop.m_dot')

phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                  rate_continuity=True, rate_continuity_scaler=100.0,
                  rate2_continuity=False)

phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

#
# Setup the boundary and path constraints
#
phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3)
phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
phase.add_boundary_constraint('gam', loc='final', equals=0.0)

phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

# Minimize time at the end of the phase
phase.add_objective('time', loc='final', ref=1.0)

p.model.linear_solver = om.DirectSolver()

#
# Setup the problem and set the initial guess
#
p.setup(check=True)

phase.set_time_val(initial=0.0, duration=350)
phase.set_state_val('r', [0.0, 50000.0])
phase.set_state_val('h', [100.0, 20000.0])
phase.set_state_val('v', [135.964, 283.159])
phase.set_state_val('gam', [0.0, 0.0])
phase.set_state_val('m', [19030.468, 10000.])
phase.set_control_val('alpha', [0.0, 0.0])

#
# Solve for the optimal trajectory
#
dm.run_problem(p, simulate=True)


```

```python
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.h',
               'time (s)', 'altitude (m)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.alpha',
               'time (s)', 'alpha (deg)')],
             title='Supersonic Minimum Time-to-Climb Solution',
             p_sol=sol, p_sim=sim)

plt.show()
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(p.get_val('traj.phase0.t_duration'), 321.0, tolerance=1.0E-1)
```

## References

```{bibliography}
:filter: docname in docnames
```
