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

# Calculation Expressions

Sometimes a user may wish to calculate some additional output that is not inherently provided by the ODE system. In cases where they have access to the model, it might be trivial to add it to the ODE's outputs. In other cases, they might not want to modify a model provided by someone else.

Calculation expressions provide the user with a way of computing additional outputs that is extremely general.
Older version of dymos supported expressions for use as constraints, objectives, or timeseries outputs.
Now dymos just computes these expressions along-side the ODE, making them available to basically all of dymos (constraints, objectives, timeseries outputs, state rates, etc.)

The most general way to add a calculated expression to an ODE output is to use the `phase.add_calc_expr` method.

```{eval-rst}
    .. automethod:: dymos.Phase.add_calc_expr
        :noindex:
```

Providing the calculated expression as the name in `add_boundary_constraint`, `add_path_constraint`, `add_objective`, or `add_timeseries_output` will still work for backward compatibility.
In general, using `add_calc_expr` will just always work, and makes that output variable available across the phase.

## Units

In general, a given calculation that a user might apply assumes specific units for each of the input variables.
In dymos, introspection for units happens _after_ the ODE is completely defined.
Because the ExecComp that computes the calcuation expressions is itself part of the ODE, we cannot wait for this introspection step.
Units for inputs **will default to None** if not otherwise specified.

If you're not certain of the units of an input variable upon which your calculation relies, you should explicitly provide it via keyword arguments to the `add_calc_expr` call (or one of the associated calls that accepts calculation expressions, mentioned above).

## Variable Name Promotion

By default, the inputs and outputs of the calculation expressions will be promoted to the top level
of your ODE. In some cases, names of your ODE variables may include colons and thus not be valid variable names for an ExecComp.

To connect such ODE variable names to your calculation expression, you can promote them by providing a `promote_as` field in the the keyword arguments. Note that this is different from the way things are handled in OpenMDAO.

In the example below, we promote the valid `ExecComp` variable `k` to a variable name that may not be used in an ExecComp due to the use of a colon: `brachistochrone:check`.

## Example - Adding an additional calculation to the brachistochrone

Let's assume that we have a brachistochrone ODE that was provided to us, but for configuration control reasons, we might not want to edit the source code.

Here we're adding a calculation expression for the additional output k, which is the ratio

\begin{align}
    k &= \frac{sin(\theta)}{v}
\end{align}

In the brachistochrone solution, $k$ is approximately constant, and its value is dependent upon the start and end point of the trajectory, as well as the gravitational parameter. In practice, it's value is somewhat large at the start since we're starting with nearly zero velocity.

```python
import openmdao.api as om
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

p = om.Problem(model=om.Group())

p.driver = om.ScipyOptimizeDriver()

p.driver.declare_coloring()

# Create the transcription so we can get the number of nodes for the downstream analysis
tx = dm.Radau(num_segments=20, order=3, compressed=False)

traj = dm.Trajectory()
phase = dm.Phase(transcription=tx, ode_class=BrachistochroneODE)
traj.add_phase('phase0', phase)

p.model.add_subsystem('traj', traj)

phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

phase.add_state('x', units='m', rate_source='xdot', fix_initial=True, fix_final=True)
phase.add_state('y', units='m', rate_source='ydot', fix_initial=True, fix_final=True)
phase.add_state('v', units='m/s', rate_source='vdot', fix_initial=True, fix_final=False)

phase.add_control('theta', units='deg', lower=0.01, upper=179.9,
                  continuity=True, rate_continuity=True)

phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

phase.add_calc_expr('k = sin(theta) / v', theta={'units': 'rad'}, v={'units': 'm/s'},
                    k={'promote_as': 'brachistochrone:check'})

# Minimize time at the end of the phase
phase.add_objective('time', loc='final', scaler=1)

p.setup(check=True)

phase.set_time_val(initial=0.0, duration=2.0)
phase.set_state_val('x', [0, 10])
phase.set_state_val('y', [10, 5])
phase.set_state_val('v', [1.0E-1, 9.9])
phase.set_control_val('theta', [5, 100])
phase.set_parameter_val('g', 9.80665)

dm.run_problem(p, simulate=True, make_plots=True)
```

```python
# tags: hide-input
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
