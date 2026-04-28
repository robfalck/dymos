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

# The Brachistochrone with Externally-Sourced initial state values

```{admonition} Things you'll learn through this example
    - How to link phase state boundary values to an externally provided value.
```

This is another modification of the brachistochrone in which the target initial value of a state is provided by an external source (an IndepVarComp in this case).

Rather than the external value being directly connected to the phase, the values are "linked" via constraint.
This is exactly how phase linkages in trajectories work as well, but the trajectory hides some of the implementation.

The following script fully defines the brachistochrone problem with Dymos and solves it. A new `IndepVarComp` is added before the trajectory which provides `x0`. An ExecComp then computes the error between `x0_target` (taken from the IndepVarComp) and `x0_actual` (taken from the phase timeseries output). The result of this calculation (`x0_error`) is then constrained as a normal OpenMDAO constraint.

```python
# tags: remove-input, hide-output
om.display_source("dymos.examples.brachistochrone.doc.brachistochrone_ode")
```

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
import openmdao.api as om
import dymos as dm

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

#
# Define the OpenMDAO problem
#
p = om.Problem(model=om.Group())

# Instantiate the transcription so we can get the number of nodes from it while
# building the problem.
tx = dm.GaussLobatto(num_segments=10, order=3)

# Add an indep var comp to provide the external control values
ivc = p.model.add_subsystem('states_ivc', om.IndepVarComp(), promotes_outputs=['*'])

# Add the output to provide the values of theta at the control input nodes of the transcription.
ivc.add_output('x0', shape=(1,), units='m')

# Connect x0 to the state error component so we can constrain the given value of x0
# to be equal to the value chosen in the phase.
p.model.connect('x0', 'state_error_comp.x0_target')
p.model.connect('traj.phase0.timeseries.x', 'state_error_comp.x0_actual', src_indices=[0])

#
# Define a Trajectory object
#
traj = dm.Trajectory()

p.model.add_subsystem('traj', subsys=traj)

p.model.add_subsystem('state_error_comp',
                      om.ExecComp('x0_error = x0_target - x0_actual',
                                  x0_error={'units': 'm'},
                                  x0_target={'units': 'm'},
                                  x0_actual={'units': 'm'}))

p.model.add_constraint('state_error_comp.x0_error', equals=0.0)

#
# Define a Dymos Phase object with GaussLobatto Transcription
#
phase = dm.Phase(ode_class=BrachistochroneODE,
                 transcription=tx)

traj.add_phase(name='phase0', phase=phase)

#
# Set the time options
# Time has no targets in our ODE.
# We fix the initial time so that the it is not a design variable in the optimization.
# The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
#
phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

#
# Set the time options
# Initial values of positions and velocity are all fixed.
# The final value of position are fixed, but the final velocity is a free variable.
# The equations of motion are not functions of position, so 'x' and 'y' have no targets.
# The rate source points to the output in the ODE which provides the time derivative of the
# given state.
phase.add_state('x', fix_initial=False, fix_final=True, units='m', rate_source='xdot')
phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='ydot')
phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                rate_source='vdot', targets=['v'])

# Define theta as a control.
# Use opt=False to allow it to be connected to an external source.
# Arguments lower and upper are no longer valid for an input control.
phase.add_control(name='theta', units='rad', targets=['theta'])

# Minimize final time.
phase.add_objective('time', loc='final')

# Set the driver.
p.driver = om.ScipyOptimizeDriver()

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup(check=True)

# Now that the OpenMDAO problem is setup, we can set the values of the states.
p.set_val('x0', 0.0, units='m')

# Here we're intentially setting the intiial x value to something other than zero, just
# to demonstrate that the optimizer brings it back in line with the value of x0 set above.
phase.set_state_val('x', [1, 10],
                   units='m')

phase.set_state_val('y', [10, 5],
                   units='m')

phase.set_state_val('v', [0, 5],
                   units='m/s')

phase.set_control_val('theta',
                      [90, 90],
                     units='deg')

# Run the driver to solve the problem
dm.run_problem(p, make_plots=True)

print(p.get_val('traj.phase0.timeseries.time'))

# Check the validity of our results by using scipy.integrate.solve_ivp to
# integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

axes[0].plot(p.get_val('traj.phase0.timeseries.x'),
             p.get_val('traj.phase0.timeseries.y'),
             'ro', label='solution')

axes[0].plot(sim_out.get_val('traj.phase0.timeseries.x'),
             sim_out.get_val('traj.phase0.timeseries.y'),
             'b-', label='simulation')

axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m/s)')
axes[0].legend()
axes[0].grid()

axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.theta', units='deg'),
             'ro', label='solution')

axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.theta', units='deg'),
             'b-', label='simulation')

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel(r'$\theta$ (deg)')
axes[1].legend()
axes[1].grid()

plt.show()

```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)
```
