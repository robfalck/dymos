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

# How do I run two phases parallel-in-time?

Complex models sometimes encounter state variables which are best simulated on different time scales, with some state variables changing quickly (fast variables) and some evolving slowly (slow variables).  

For instance, an aircraft trajectory optimization which includes vehicle component temperatures might see relatively gradual changes in altitude over the course of a two hour flight while temperatures of some components seem to exhibit step-function-like behavior on the same scale.

To accommodate both fast and slow variables in the same ODE, one would typically need to use a _dense_ grid (with many segments/higher order segments).
This can be unnecessarily burdensome when there are many slow variables or evaluating their rates is particularly expensive.

As a solution, Dymos allows the user to run two phases over the same range of times, where one phase may have a more sparse grid to accommodate the slow variables, and one has a more dense grid for the fast variables.

To connect the two phases, state variable values are passed from the first (slow) phase to the second (fast) phase as non-optimal dynamic control variables.
These values are then used to evaluate the rates of the fast variables.
Since outputs from the first phase in general will not fall on the appropriate grid points to be used by the second phase, interpolation is necessary.  
This is one application of the interpolating timeseries component.

In the following example, we solve the brachistochrone problem but do so to minimize the arclength of the resulting wire instead of the time required for the bead to travel along the wire.  
This is a trivial solution which should find a straight line from the starting point to the ending point.

There are two phases involved, the first utilizes the standard ODE for the brachistochrone problem.
The second integrates the arclength (𝑆) of the wire using the equation:

\begin{align}
    S = \int v \sin \theta  \sqrt{1 + \frac{1}{\tan^2 \theta}} dt
\end{align}

## The ODE for the wire arclength

```python
# tags: remove-input
om.display_source('dymos.examples.brachistochrone.doc.test_doc_brachistochrone_tandem_phases.BrachistochroneArclengthODE')
```

The trick is that the bead velocity ($v$) is a state variable solved for in the first phase,
and the wire angle ($\theta$) is a control variable "owned" by the first phase.  In the
second phase they are used as control variables with option ``opt=False`` so that their values are
expected as inputs for the second phase.  We need to connect their values from the first phase
to the second phase, at the `control_input` node subset of the second phase.

In the following example, we instantiate two phases and add an interpolating timeseries to the first phase
which provides outputs at the `control_input` nodes of the second phase.  Those values are
then connected and the entire problem run. The result is that the position and velocity variables
are solved on a relatively coarse grid while the arclength of the wire is solved on a much denser grid.

```python
# tags: remove-input, remove-output
class BrachistochroneArclengthODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')
        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')
        self.add_output('Sdot', val=np.zeros(nn), desc='rate of change of arclength', units='m/s')

        # Setup partials
        arange = np.arange(nn)

        self.declare_partials(of='Sdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='Sdot', wrt='theta', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        v = inputs['v']
        outputs['Sdot'] = np.sqrt(1.0 + (1.0/np.tan(theta))**2) * v * np.sin(theta)

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        v = inputs['v']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        tan_theta = np.tan(theta)
        cot_theta = 1.0 / tan_theta
        csc_theta = 1.0 / sin_theta

        jacobian['Sdot', 'v'] = sin_theta * np.sqrt(1.0 + cot_theta**2)
        jacobian['Sdot', 'theta'] = v * (cos_theta * (cot_theta**2 + 1) - cot_theta * csc_theta) / \
            (np.sqrt(1 + cot_theta**2))
```

```python
# tags: remove-input, hide-output
om.display_source('dymos.examples.brachistochrone.brachistochrone_ode')
```

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
# tags: output_scroll
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
import dymos as dm

p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.declare_coloring()

# The transcription of the first phase
tx0 = dm.GaussLobatto(num_segments=10, order=3, compressed=False)

# The transcription for the second phase (and the secondary timeseries outputs from the first phase)
tx1 = dm.Radau(num_segments=20, order=9, compressed=False)

#
# First Phase: Integrate the standard brachistochrone ODE
#
phase0 = dm.Phase(ode_class=BrachistochroneODE, transcription=tx0)

p.model.add_subsystem('phase0', phase0)

phase0.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

phase0.add_state('x', fix_initial=True, fix_final=False)

phase0.add_state('y', fix_initial=True, fix_final=False)

phase0.add_state('v', fix_initial=True, fix_final=False)

phase0.add_control('theta', continuity=True, rate_continuity=True,
                   units='deg', lower=0.01, upper=179.9)

phase0.add_parameter('g', units='m/s**2', val=9.80665)

phase0.add_boundary_constraint('x', loc='final', equals=10)
phase0.add_boundary_constraint('y', loc='final', equals=5)

# Add alternative timeseries output to provide control inputs for the next phase
phase0.add_timeseries('timeseries2', transcription=tx1, subset='control_input')

#
# Second Phase: Integration of ArcLength
#
phase1 = dm.Phase(ode_class=BrachistochroneArclengthODE, transcription=tx1)

p.model.add_subsystem('phase1', phase1)

phase1.set_time_options(fix_initial=True, input_duration=True)

phase1.add_state('S', fix_initial=True, fix_final=False,
                 rate_source='Sdot', units='m')

phase1.add_control('theta', opt=False, units='deg', targets='theta')
phase1.add_control('v', opt=False, units='m/s', targets='v')

#
# Connect the two phases
#
p.model.connect('phase0.t_duration_val', 'phase1.t_duration')

p.model.connect('phase0.timeseries2.theta', 'phase1.controls:theta')
p.model.connect('phase0.timeseries2.v', 'phase1.controls:v')

# Minimize arclength at the end of the second phase
phase1.add_objective('S', loc='final', ref=1)

p.model.linear_solver = om.DirectSolver()
p.setup(check=True)

phase0.set_time_val(0.0, 2.0)

phase0.set_state_val('x', [0, 10])
phase0.set_state_val('y', [10, 5])
phase0.set_state_val('v', [0, 9.9])
phase0.set_control_val('theta', [5, 100])
phase0.set_parameter_val('g', 9.80665)

phase1.set_state_val('S', 0.0)

dm.run_problem(p)

fig, (ax0, ax1) = plt.subplots(2, 1)
fig.tight_layout()
ax0.plot(p.get_val('phase0.timeseries.x'), p.get_val('phase0.timeseries.y'), '.')
ax0.set_xlabel('x (m)')
ax0.set_ylabel('y (m)')
ax1.plot(p.get_val('phase1.timeseries.time'), p.get_val('phase1.timeseries.S'), '+')
ax1.set_xlabel('t (s)')
ax1.set_ylabel('S (m)')
plt.show()
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

expected = np.sqrt((10-0)**2 + (10 - 5)**2)
assert_near_equal(p.get_val('phase1.timeseries.S')[-1], expected, tolerance=1.0E-3)
```
