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

# Moon Landing Problem

The Moon landing problem is a version of the soft landing problem presented in {cite}`Meditch1964`. The problem is simplified to have one degree-of-freedom and normalized such that the Moon's gravity is unity. The goal is to minimize the amount of fuel consumed or, stated differently, maximize the final mass, while bringing the lander down to the surface for a soft landing.

## State and control variables

This system has three state variables, the altitude ($h$), velocity ($v$), and mass ($m$) of the lander.

This system has one control variable, ($T$), the thrust applied to the vehicle.

The dynamics of the system are given by

\begin{align}
  \dot{h} &= v \\
  \dot{v} &= -1 + \frac{T}{m} \\
  \dot{m} &= -\frac{T}{2.349}
\end{align}

## Problem Definition

We seek to maximize the final mass of the vehicle while bringing it to a soft landing.

\begin{align}
  \mathrm{Minimize} \, J &= m_f
\end{align}

The initial conditions are
\begin{align}
  h_0 &= 1 \\
  v_0 &= -0.783 \\
  m_0 &= 1
\end{align}
and the terminal constraints are
\begin{align}
  h_f &= 0 \\
  v_f &= 0
\end{align}

Additionally, the thrust is constrained to be positive but remain under 1.227.

\begin{align}
  0 \le T \le 1.227 
\end{align}

## Defining the ODE

The following implements the dynamics of the Moon landing problem described above.

```python
import numpy as np
import openmdao.api as om


class MoonLandingProblemODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs
        self.add_input('h', val=np.ones(nn), units=None, desc='Altitude')
        self.add_input('v', val=np.ones(nn), units='1/s', desc='Velocity')
        self.add_input('m', val=np.ones(nn), units=None, desc='Mass')
        self.add_input('T', val=np.ones(nn), units=None, desc='Thrust')

        # outputs
        self.add_output('h_dot', val=np.ones(nn), units='1/s', desc='Rate of change of Altitude')
        self.add_output('v_dot', val=np.ones(nn), units='1/s**2', desc='Rate of change of Velocity')
        self.add_output('m_dot', val=np.ones(nn), units='1/s', desc='Rate of change of Mass')

        # partials
        ar = np.arange(nn)
        self.declare_partials(of='h_dot', wrt='v', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='v_dot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='T', rows=ar, cols=ar)
        self.declare_partials(of='m_dot', wrt='T', rows=ar, cols=ar, val=-1/2.349)
        self.declare_partials(of='m_dot', wrt='T', rows=ar, cols=ar, val=-1/2.349)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        v = inputs['v']
        m = inputs['m']
        T = inputs['T']

        outputs['h_dot'] = v
        outputs['v_dot'] = -1 + T/m
        outputs['m_dot'] = -T/2.349

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m = inputs['m']
        T = inputs['T']

        partials['v_dot', 'T'] = 1/m
        partials['v_dot', 'm'] = -T/m**2
```

## Solving the Moon landing problem with Dymos

The optimal solution to this problem is known to have _bang-bang_ control. That is, the control has a "jump" that render it discontinuous in time. Capturing this behavior accurately requires the use of grid refinement for the Gauss-Lobatto and Radau pseudospectral transcriptions but the Birkhoff pseudospectral transcription can be used to handle this behavior without the use of any grid refinement. The following code shows the use of the Birkhoff pseudospectral transcription to solve the problem.

```python
import dymos as dm

p = om.Problem(model=om.Group())
p.driver = om.pyOptSparseDriver()
p.driver.declare_coloring()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
p.driver.opt_settings['print_level'] = 0
p.driver.opt_settings['linear_solver'] = 'mumps'
p.driver.declare_coloring()

t = dm.Birkhoff(num_nodes=20)

traj = p.model.add_subsystem('traj', dm.Trajectory())
phase = dm.Phase(ode_class=MoonLandingProblemODE, transcription=t)

phase.set_time_options(fix_initial=True, fix_duration=False)
phase.add_state('h', fix_initial=True, rate_source='h_dot')
phase.add_state('v', fix_initial=True, rate_source='v_dot')
phase.add_state('m', fix_initial=True, lower=1e-3, rate_source='m_dot')
phase.add_control('T', lower=0.0, upper=1.227)

phase.add_boundary_constraint('h', loc='final', equals=0.0)
phase.add_boundary_constraint('v', loc='final', equals=0.0)

phase.add_objective('m', scaler=-1)
phase.set_simulate_options(atol=1.0E-1, rtol=1.0E-2)

traj.add_phase('phase', phase)

p.setup(check=True, force_alloc_complex=True)

phase.set_time_val(initial=0.0, duration=1.0)
phase.set_state_val('h', [1.0, 0.0])
phase.set_state_val('v', [-0.783, 0.0])
phase.set_state_val('m', [1.0, 0.2])
phase.set_control_val('T', [0.0, 1.227])
dm.run_problem(p, simulate=False, simulate_kwargs={'times_per_seg': 100}, make_plots=True)
```

```python
from IPython.display import HTML

# Define the path to the HTML file
html_file_path = p.get_reports_dir() / 'traj_results_report.html'
html_content = html_file_path.read_text()

# Inject CSS to control the output cell height and avoid scrollbars
html_with_custom_height = f"""
<div style="height: 800px; overflow: auto;">
    {html_content}
</div>
"""

HTML(html_with_custom_height)
```

### Notes on the solution

We can see that the collocation solution accurately captures the jump in the thrust. The oscillatory behavior observed is a result of interpolation performed post solution rather than a property of the solution itself.

## References

```{bibliography}
:filter: docname in docnames
```
