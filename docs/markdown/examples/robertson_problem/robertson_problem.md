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

(examples:the_robertson_problem)=

# The Robertson Problem

The [Robertson Problem](https://en.wikipedia.org/w/index.php?title=Stiff_equation&oldid=1017928453#Motivating_example) is a famous example for a [stiff ODE](https://en.wikipedia.org/w/index.php?title=Stiff_equation&oldid=1017928453). Solving stiff ODEs with [explicit integration methods](https://en.wikipedia.org/w/index.php?title=Explicit_and_implicit_methods&oldid=1036001392) leads to unstable behaviour unless an extremly small step size is choosen. [Implicit methods](https://en.wikipedia.org/w/index.php?title=Explicit_and_implicit_methods&oldid=1036001392) such as the [Radau](https://en.wikipedia.org/w/index.php?title=Runge%E2%80%93Kutta_methods&oldid=1052924118#Implicit_Runge%E2%80%93Kutta_methods), [BDF](https://en.wikipedia.org/w/index.php?title=Backward_differentiation_formula&oldid=1027943694) and LSODA methods can help solve such problems. The following example shows how to solve the Robertson Problem using [SciPys LSODA method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp) and Dymos. 

## The ODE system

The ODE of the Robertson Problem is

\begin{align}
    \dot x = &        - 0.04 x + 10^4 y \cdot z &                          \\
    \dot y = & \;\;\;\: 0.04 x - 10^4 y \cdot z &        - 3\cdot 10^7 y^2 \\
    \dot z = &                                  & \;\;\;\: 3\cdot 10^7 y^2 \\
\end{align}

where $x$, $y$ and $z$ are arbitrary states. The initial conditions are

\begin{align}
    x_0 &= 1 \\
    y_0 &= z_0 = 0.
\end{align}

The problem is solved for the time interval $t\in[0,40)$. There are no controls and constraints.

```python
import numpy as np
import openmdao.api as om


class RobertsonODE(om.ExplicitComponent):
    """example for a stiff ODE from Robertson.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # input: state
        self.add_input('x', shape=nn, desc="state x", units=None)
        self.add_input('y', shape=nn, desc="state y", units=None)
        self.add_input('z', shape=nn, desc="state z", units=None)
        
        # output: derivative of state
        self.add_output('xdot', shape=nn, desc='derivative of x', units="1/s")
        self.add_output('ydot', shape=nn, desc='derivative of y', units="1/s")
        self.add_output('zdot', shape=nn, desc='derivative of z', units="1/s")

        r = np.arange(0, nn)
        self.declare_partials(of='*', wrt='*', method='exact',  rows=r, cols=r)

    def compute(self, inputs, outputs):

        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        xdot = -0.04 * x + 1e4 * y * z
        zdot = 3e7 * y ** 2
        ydot = - (xdot + zdot)
        
        outputs['xdot'] = xdot
        outputs['ydot'] = ydot
        outputs['zdot'] = zdot
        
    def compute_partials(self, inputs, jacobian):

        # x = inputs['x']  # x is not needed to compute partials
        y = inputs['y']
        z = inputs['z']

        xdot_y = 1e4 * z
        xdot_z = 1e4 * y

        zdot_y = 6e7 * y

        jacobian['xdot', 'x'] = -0.04
        jacobian['xdot', 'y'] = xdot_y
        jacobian['xdot', 'z'] = xdot_z

        jacobian['ydot', 'x'] = 0.04
        jacobian['ydot', 'y'] = - (xdot_y + zdot_y)
        jacobian['ydot', 'z'] = - xdot_z

        jacobian['zdot', 'x'] = 0.0
        jacobian['zdot', 'y'] = zdot_y
        jacobian['zdot', 'z'] = 0.0


```

```python
# tags: remove-input, remove-output
num_nodes = 3

p = om.Problem(model=om.Group())

p.model.add_subsystem('ode', RobertsonODE(num_nodes=num_nodes), promotes=['*'])

p.setup(force_alloc_complex=True)

p.set_val('x', [10., 100., 1000.])
p.set_val('y', [1, 0.1, 0.01])
p.set_val('z', [1., 2., 3.])

p.run_model()
cpd = p.check_partials(method='cs', compact_print=True)
```

```python
# tags: remove-input, remove-output
from dymos.utils.testing_utils import assert_check_partials

assert_check_partials(cpd)
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(p.get_val('xdot'), [9999.6, 1996., 260.])
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(p.get_val('ydot'), [-3.00099996E7, -3.01996E5, -3.26E3])
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(p.get_val('zdot'), [3.0E7, 3.0E5, 3.0E3])
```

## Building and running the problem

Here we're using the ExplicitShooting transcription in Dymos.
The ExplicitShooting transcription explicit integrates the given ODE using the `solve_ivp` method of [scipy](https://scipy.org/).

Since this is purely an integration with no controls to be determined, a single call to `run_model` will propagate the solution for us. There's no need to run a driver. Even the typical follow-up call to `traj.simulate` is unnecessary.

Technically, we could even solve this using a single segment since segment spacing in the explicit shooting transcription determines the density of the control nodes, and there are no controls for this simulation.
Providing more segments in this case (or a higher segment order) increases the number of nodes at which the outputs are provided.

```python
import openmdao.api as om
import dymos as dm


def robertson_problem(t_final=1.):
    #
    # Initialize the Problem
    #
    p = om.Problem(model=om.Group())
    
    #
    # Create a trajectory and add a phase to it
    #
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    
    tx = dm.ExplicitShooting(num_segments=10, method='LSODA')

    phase = traj.add_phase('phase0',
                           dm.Phase(ode_class=RobertsonODE, transcription=tx))

    #
    # Set the variables
    #
    phase.set_time_options(fix_initial=True, fix_duration=True)

    phase.add_state('x0', fix_initial=True, fix_final=False, rate_source='xdot', targets='x')
    phase.add_state('y0', fix_initial=True, fix_final=False, rate_source='ydot', targets='y')
    phase.add_state('z0', fix_initial=True, fix_final=False, rate_source='zdot', targets='z')

    #
    # Setup the Problem
    #
    p.setup(check=True)

    #
    # Set the initial values
    #

    phase.set_time_val(initial=0.0, duration=t_final)
    phase.set_state_val('x0', [1.0, 0.7])
    phase.set_state_val('y0', [0.0, 1e-5])
    phase.set_state_val('z0', [0.0, 0.3])

    return p

```

```python
# tags: output_scroll
# just set up the problem, test it elsewhere
p = robertson_problem(t_final=40)

p.run_model()
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(p.get_val('traj.phase0.timeseries.x0')[-1], 0.71583161, tolerance=1E-4)
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(p.get_val('traj.phase0.timeseries.y0')[-1], 9.18571144e-06, tolerance=1E-4)
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(p.get_val('traj.phase0.timeseries.z0')[-1], 0.2841592, tolerance=1E-4)
```

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
import matplotlib.pyplot as plt

t = p.get_val('traj.phase0.timeseries.time')

states = ['x0', 'y0', 'z0']
fig, axes = plt.subplots(len(states), 1)
for i, state in enumerate(states):
    axes[i].plot(t, p.get_val(f'traj.phase0.timeseries.{state}'), 'o')
    axes[i].set_ylabel(state[0])
axes[-1].set_xlabel('time (s)')
plt.tight_layout()
plt.show()
```
