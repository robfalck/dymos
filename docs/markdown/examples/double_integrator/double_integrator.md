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

# Double Integrator

In the double integrator problem, we seek to maximize the distance
traveled by a block (that starts and ends at rest) sliding without
friction along a horizontal surface, with acceleration as the control.

We minimize the final time, $t_f$, by varying the dynamic control,
$u$, subject to the dynamics:

\begin{align}
  \frac{dx}{dt} &= v \\
  \frac{dv}{dt} &= u
\end{align}

The initial conditions are

\begin{align}
  x_0 &= 0 \\
  v_0 &= 0
\end{align}

and the final conditions are

\begin{align}
  x_f &= \rm{free} \\
  v_f &= 0
\end{align}

The control $u$ is constrained to fall between -1 and 1. Due to the fact
that the control appears linearly in the equations of motion, we should
expect _bang-bang_ behavior in the control (alternation between its extreme values).

## The ODE System: double\_integrator\_ode.py

This problem is unique in that we do not actually have to calculate
anything in the Dymos formulation of the ODE. We create an
_ExplicitComponent_ and provide it with the _num\_nodes_
option, but it has no inputs and no outputs. The rates for the states
are entirely provided by the other states and controls.

```python
class DoubleIntegratorODE(om.ExplicitComponent):
    """
    The double integrator is a special case where the state rates are all set to other states
    or parameters.  Since we aren't computing any other outputs, the ODE doesn't actually
    need to compute anything.  OpenMDAO will warn us that the component has no outputs, but
    Dymos will solve the problem just fine.

    Note we still have to declare the num_nodes option in initialize so that Dymos can instantiate
    the ODE.

    Also note that neither time, states, nor parameters have targets, since there are no inputs
    in the ODE system.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
```

## Building and running the problem

In order to facilitate the bang-bang behavior in the control, we disable
continuity and rate continuity in the control value.

```python
# tags: remove-input, hide-output
om.display_source("dymos.examples.double_integrator.double_integrator_ode")
```

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

# Initialize the problem and assign the driver
p = om.Problem(model=om.Group())
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.declare_coloring()

# Setup the trajectory and its phase
traj = p.model.add_subsystem('traj', dm.Trajectory())

transcription = dm.Radau(num_segments=30, order=3, compressed=False)

phase = traj.add_phase('phase0',
                       dm.Phase(ode_class=DoubleIntegratorODE, transcription=transcription))

#
# Set the options for our variables.
#
phase.set_time_options(fix_initial=True, fix_duration=True, units='s')
phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')
phase.add_state('x', fix_initial=True, rate_source='v', units='m')

phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                  rate2_continuity=False, shape=(1, ), lower=-1.0, upper=1.0)

#
# Maximize distance travelled.
#
phase.add_objective('x', loc='final', scaler=-1)

p.model.linear_solver = om.DirectSolver()

#
# Setup the problem and set our initial values.
#
p.setup(check=True)

phase.set_time_val(initial=0.0, duration=1.0)

phase.set_state_val('x', [0, 0.25])
phase.set_state_val('v', [0, 0])
phase.set_control_val('u', [1, -1])

#
# Solve the problem.
#
dm.run_problem(p, simulate=True)
```

```python
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.x',
               'time (s)', 'x $(m)$'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.v',
               'time (s)', 'v $(m/s)$'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.u',
               'time (s)', 'u $(m/s^2)$')],
             title='Double Integrator Solution\nRadau Pseudospectral Method',
             p_sol=sol, p_sim=sim)

plt.show()
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

x = p.get_val('traj.phase0.timeseries.x')
v = p.get_val('traj.phase0.timeseries.v')

assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)
```
