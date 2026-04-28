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

(examples:brachistochrone_tandem_phases)=
# Brachistochrone with tandem phases

```{admonition} Things you'll learn through this example
- How to run two phases with different ODE's and different grids simultaneously in time.
```

This is a contrived example but it demonstrates a useful feature of Dymos we call **tandem phases**.
Tandem phases are two phases that occur simultaneously in time (having the same start time and duration) but with different dynamics.
In practice, this can be useful when some of your dynamics are quite expensive and you can tolerate evaluating them on fewer nodes.
Or perhaps one phase has relatively rapid dynamics compared to the other one.  For instance, thermal responses tend to happen vary rapidly in an electric aircraft compared to changes in the flight dynamics state of the vehicle.

In this example we'll evaulate the standard brachistochrone problem, but limit the arclength of the wire along which the bead travels.
The arclength is integrated as a state variable, and can be done so along with the typical states _x_, _y_, and _v_, but for the purposes of this contrive example we'll perform this integration in a separate phase that occurs at the same time.

## The first phase to integrate the standard brachistochrone ODE

```{admonition} Things to note about this phase
- The transcriptions for the two phases are delcared up front so that `tx1` may be used as both the transcription of the second phase, and for outputting the states of the first phase to the control input nodes of the second phase.
```

This secondary timeseries is the key to making this sort of formulation work.

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import openmdao.api as om
import dymos as dm


p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.options['print_results'] = False
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

```

## The ODE for integrating the arc-length of the wire

```python
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

## The second phase to integrate the arclength of the wire.

```{admonition} Things to note about this phase
- Initial time and duration are input from those of the first phase (note the connections on line 16).
- _theta_ and _v_ are time-varying but determined by the first phase. Note the setting of `opt=False` and the connections on lines 18 and 19.
```

```python
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

# Minimize time
phase1.add_objective('time', loc='final', ref=1)

# Constraint the length of the wire.
phase1.add_boundary_constraint('S', loc='final', upper=11.5)
```

## Setup and run

```python
p.model.linear_solver = om.DirectSolver()
p.setup()

phase0.set_time_val(initial=0.0, duration=2.0)
phase0.set_state_val('x', [0, 10])
phase0.set_state_val('y', [10, 5])
phase0.set_state_val('v', [0, 9.9])
phase0.set_control_val('theta', [5, 100.5])
phase0.set_parameter_val('g', 9.80665)

phase1.set_state_val('S', 0.0)

res = dm.run_problem(p)
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert not res
assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.85266, tolerance=1.0E-3)
```

## Plots

The following plots show the trajectory of the x, y, and v states (the top plot) and the trajectory of the arclength state (the bottom plot).  Note that these plots are linked, but use different grid spacings - the arclength is integrated on a significantly more dense grid.  This is enabled by the secondary timeseries output `timeseries2` in the first phase.

```python
# tags: hide-input
import pathlib

from bokeh.plotting import figure, output_file, save

from bokeh.palettes import d3
from bokeh.models import Legend

from IPython.display import HTML

c = d3['Category10'][10]
i = np.array(0)
legend_contents = []

print(p.get_outputs_dir())
import os
print(os.listdir(os.getcwd()))

sol_case = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')

sol_x = sol_case.get_val('phase0.timeseries.x')
sol_y = sol_case.get_val('phase0.timeseries.y')
sol_v = sol_case.get_val('phase0.timeseries.v')
sol_t0 = sol_case.get_val('phase0.timeseries.time')
sol_t1 = sol_case.get_val('phase1.timeseries.time')
sol_s = sol_case.get_val('phase1.timeseries.S')


def add_plot(p, x, y, label, i):
    circle = p.circle(x.ravel(), y.ravel(), color=c[i], size=5)
    line = p.line(x.ravel(), y.ravel(), color=c[i])
    legend_contents.append((label, [circle, line]))
    i += 1


p1 = figure(width=800, height=300)
add_plot(p1, sol_t0, sol_x, 'x (m)', i)
add_plot(p1, sol_t0, sol_y, 'y (m)', i)
add_plot(p1, sol_t0, sol_v, 'v', i)
add_plot(p1, sol_t1, sol_s, 'arclength', i)

p1.add_layout(Legend(items=legend_contents), 'right')

p1.xaxis.axis_label = 'time (s)'
p1.yaxis.axis_label = 'state value'
p1.legend.location = 'bottom_right'

output_file('plot.html', mode='inline')
plot_file = save(p1)

# Define the path to the HTML file
html_file_path = pathlib.Path('plot.html')
html_content = html_file_path.read_text()

# Inject CSS to control the output cell height and avoid scrollbars
html_with_custom_height = f"""
<div style="height: 320px; overflow: auto;">
    {html_content}
</div>
"""

HTML(html_with_custom_height)
```
