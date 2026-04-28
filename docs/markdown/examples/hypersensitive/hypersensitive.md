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

# Hyper-Sensitive Problem

This example is based on the Hyper-Sensitive problem given in
Patterson {cite}`patterson2015ph`. In this problem we seek to minimize both
the distance traveled when moving between fixed boundary conditions and
also to minimize the control $u$ used. The cost function to be minimized is:

\begin{align}
    J &= \frac{1}{2} \int_{0}^{t_f} (x^2 + u^2) dt
\end{align}

The system is subject to the dynamic constraints:

\begin{align}
    \frac{dx}{dt} &= -x + u
\end{align}

The boundary conditions are:

\begin{align}
    x(t_0) &= 1.5 \\
    x(t_f) &= 1
\end{align}

The control $u$ is unconstrained while the final time $t_f$ is fixed.

Due to the nature of dynamics, for sufficiently large values of $t_f$,
the problem exhibits a _dive_, _cruise_, and _resurface_ type
structure, where the all interesting behavior occurs at the beginning and
end while remaining relatively constant in the middle.

This problem has a known analytic optimal solution:

\begin{align}
    x^*(t) &= c_1 e^{\sqrt{2} t} + c_2 e^{-\sqrt{2} t} \\
      u^*(t) &= \dot{x}^*(t) + x^*(t)
\end{align}

where:

\begin{align}
    c_1 &= \frac{1.5 e^{-\sqrt{2} t_f} - 1}{e^{-\sqrt{2} t_f} - e^{\sqrt{2} t_f}} \\
    c_2 &= \frac{1 - 1.5 e^{\sqrt{2} t_f}}{e^{-\sqrt{2} t_f} - e^{\sqrt{2} t_f}}
\end{align}

## The ODE System: hyper\_sensitive\_ode.py

```python
import numpy as np
import openmdao.api as om


class HyperSensitiveODE(om.ExplicitComponent):
    states = {'x': {'rate_source': 'x_dot'},
              'xL': {'rate_source': 'L'}}

    parameters = {'u': {'targets': 'u'}}

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs
        self.add_input('x', val=np.zeros(nn), desc='state')
        self.add_input('xL', val=np.zeros(nn), desc='cost_state')

        self.add_input('u', val=np.zeros(nn), desc='control')

        self.add_output('x_dot', val=np.zeros(nn), desc='state rate', units='1/s')
        self.add_output('L', val=np.zeros(nn), desc='Lagrangian', units='1/s')

        # Setup partials
        self.declare_partials(of='x_dot', wrt='x', rows=np.arange(nn), cols=np.arange(nn), val=-1)
        self.declare_partials(of='x_dot', wrt='u', rows=np.arange(nn), cols=np.arange(nn), val=1)

        self.declare_partials(of='L', wrt='x', rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(of='L', wrt='u', rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs):
        x = inputs['x']
        u = inputs['u']

        outputs['x_dot'] = -x + u
        outputs['L'] = (x ** 2 + u ** 2) / 2

    def compute_partials(self, inputs, jacobian):
        x = inputs['x']
        u = inputs['u']

        jacobian['L', 'x'] = x
        jacobian['L', 'u'] = u
```

## Building and running the problem with grid refinement

The hypersenstive problem is notorious for being stiff near the endpoints, where the stiffness is more and more pronounced as the time duration is increased.


The accuracy of a pseudospectral method can suffer when the state interpolating polynomials don't have a high enough degree to accurately capture rapidly changing states in these stiff regions.
To counter this, grid refinement algorithms are used to assess error in the interpolating polynomials and change the grid (the number of interpolating segments, the polynomial order of the segments, and their positions in time).

In the example below, we tell Dymos to use, at most, ten passes of its default grid refinement algorithm by providing `run_problem` with the `refine_iteration_limit` option.


```python
# tags: remove-input, remove-output
tf = 10.0


def solution():
    sqrt_two = np.sqrt(2)
    val = sqrt_two * tf
    c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
    c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

    ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
    uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
    J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
               (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)
    return ui, uf, J
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

transcription = dm.Radau(num_segments=10, order=3, compressed=False)

phase = traj.add_phase('phase0',
                       dm.Phase(ode_class=HyperSensitiveODE, transcription=transcription))

phase.set_time_options(fix_initial=True, fix_duration=True)
phase.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
phase.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
phase.add_control('u', opt=True, targets=['u'])

phase.add_boundary_constraint('x', loc='final', equals=1)

phase.add_objective('xL', loc='final')

p.setup(check=True)

phase.set_time_val(initial=0, duration=tf)
phase.set_state_val('x', [1.5, 1])
phase.set_state_val('xL', [0, 1])
phase.set_control_val('u', [-0.6, 2.4])

#
# Solve the problem.
#

phase.set_refine_options(tol=1.0E-7)
dm.run_problem(p, simulate=True, refine_iteration_limit=10)
```

```python
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.x',
               'time (s)', 'x $(m)$'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.u',
               'time (s)', 'u $(m/s^2)$')],
             title='Hyper Sensitive Problem Solution\nRadau Pseudospectral Method',
             p_sol=sol, p_sim=sim)

plt.show()
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

ui, uf, J = solution()

assert_near_equal(p.get_val('traj.phase0.timeseries.u')[0],
                  ui,
                  tolerance=1.5e-2)

assert_near_equal(p.get_val('traj.phase0.timeseries.u')[-1],
                  uf,
                  tolerance=1.5e-2)

assert_near_equal(p.get_val('traj.phase0.timeseries.xL')[-1],
                  J,
                  tolerance=1e-2)
```

## References

```{bibliography}
:filter: docname in docnames
```
