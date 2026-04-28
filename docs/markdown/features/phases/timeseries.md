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

# Timeseries Outputs

Different optimal control transcriptions work in different ways.
The Radau Pseudospectral transcription keeps a contiguous vector of state values at all nodes.
The Gauss Lobatto transcription keeps two separate continuous vectors; one at the discretization nodes and the other at the collocation nodes.
Retrieving a timeseries values of output values is thus transcription dependent.

In order to make obtaining the timeseries output of a phase easier, each phase provides a timeseries component which collects and outputs the appropriate timeseries data.
For the pseudospectral transcriptions, timeseries outputs are provided at all nodes.
By default, the timeseries output will include the following variables for every problem.

## Paths to timeseries outputs in Dymos

|Path                                                          | Description                                         |
|--------------------------------------------------------------|-----------------------------------------------------|
|``<phase path>.timeseries.time``                              |Current time value                                   |
|``<phase path>.timeseries.time_phase``                        |Current phase elapsed time                           |
|``<phase path>.timeseries.<x>``                        |Value of state variable named x                      |
|``<phase path>.timeseries.<u>``                      |Value of control variable named u                    |
|``<phase path>.timeseries.<u>_rate``            |Time derivative of control named u                   |
|``<phase path>.timeseries.<u>_rate2``           |Second time derivative of control named u            |
|``<phase path>.timeseries.<p>``           |Value of polynomial control variable named u         |
|``<phase path>.timeseries.<p>_rate`` |Time derivative of polynomial control named u        |
|``<phase path>.timeseries.<p>_rate2``|Second time derivative of polynomial control named u |
|``<phase path>.timeseries.<d>``                    |Value of parameter named d                           |

## Adding additional timeseries outputs

In addition to these default values, any output of the ODE can be added to the timeseries output
using the ``add_timeseries_output`` method on Phase.  These outputs are available as
``<phase path>.timeseries.<output name>``.  A glob pattern can be used with ``add_timeseries_output``
to add multiple outputs to the timeseries simultaneously.  For instance, just passing '*' as the variable
name will add all dynamic outputs of the ODE to the timeseries.

Dymos will ignore any ODE outputs that are not sized such that the first dimension is the same as the
number of nodes in the ODE.  That is, if the output variable doesn't appear to be dynamic, it will not
be included in the timeseries outputs.

```{eval-rst}
    .. automethod:: dymos.Phase.add_timeseries_output
        :noindex:
```

## Interpolated Timeseries Outputs

Sometimes a user may want to interpolate the results of a phase onto a different grid.  This is particularly
useful in the context of [tandem phases](examples:brachistochrone_tandem_phases).  Additional timeseries may be added to a phase using the ``add_timeseries`` method.  By default all timeseries will provide times, states, controls, and parameters on the specified output grid.  Adding other variables is accomplished using the
``timeseries`` argument in the ``add_timeseries_output`` method.

```{eval-rst}
    .. automethod:: dymos.Phase.add_timeseries
        :noindex:
```

# Interpolating Timeseries Outputs

Outputs from the timeseries are provided at the node resolution used to solve the problem.
Sometimes it is useful to visualize the data at a different resolution - for instance to see oscillatory behavior in the solution that may not be noticeable by only viewing the solution nodes.

The Phase.interp method allows nodes to be provided as a sequence in the "tau space" (nondimensional time) of the phase. In phase tau space,
the time-like variable is -1 at the start of the phase and +1 at the end of the phase.

For instance, let's say we want to solve the minimum time climb example and plot the control solution at 100 nodes equally spaced throughout the phase.

```python
import numpy as np
import matplotlib.pyplot as plt

from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.options['print_results'] = 'minimal'
p.driver.declare_coloring()

p.driver.opt_settings['tol'] = 1.0E-5
p.driver.opt_settings['print_level'] = 0
p.driver.opt_settings['mu_strategy'] = 'monotone'
p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
p.driver.opt_settings['mu_init'] = 0.01

tx = dm.Radau(num_segments=8, order=3)

traj = dm.Trajectory()

phase = dm.Phase(ode_class=MinTimeClimbODE, transcription=tx)
traj.add_phase('phase0', phase)

p.model.add_subsystem('traj', traj)

phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                       duration_ref=100.0)

phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6,
                ref=1.0E3, defect_ref=1.0E3, units='m',
                rate_source='flight_dynamics.r_dot')

phase.add_state('h', fix_initial=True, lower=0, upper=20000.0,
                ref=20_000, defect_ref=20_000, units='m',
                rate_source='flight_dynamics.h_dot', targets=['h'])

phase.add_state('v', fix_initial=True, lower=10.0,
                ref=1.0E2, defect_ref=1.0E2, units='m/s',
                rate_source='flight_dynamics.v_dot', targets=['v'])

phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5,
                ref=1.0, defect_ref=1.0, units='rad',
                rate_source='flight_dynamics.gam_dot', targets=['gam'])

phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5,
                ref=10_000, defect_ref=10_000, units='kg',
                rate_source='prop.m_dot', targets=['m'])

phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                    rate_continuity=True, rate_continuity_scaler=100.0,
                    rate2_continuity=False, targets=['alpha'])

phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3)
phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
phase.add_boundary_constraint('gam', loc='final', equals=0.0)

phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

# Minimize time at the end of the phase
phase.add_objective('time', loc='final', ref=1.0)

# test mixing wildcard ODE variable expansion and unit overrides
phase.add_timeseries_output(['aero.*', 'prop.thrust', 'prop.m_dot'],
                            units={'aero.f_lift': 'lbf', 'prop.thrust': 'lbf'})

p.model.linear_solver = om.DirectSolver()

p.setup(check=True)

p['traj.phase0.t_initial'] = 0.0
p['traj.phase0.t_duration'] = 350.0

phase.set_time_val(initial=0.0, duration=350.0)
phase.set_state_val('r', [0.0, 111319.54])
phase.set_state_val('h', [100.0, 20000.0])
phase.set_state_val('v', [135.964, 283.159])
phase.set_state_val('gam', [0.0, 0.0])
phase.set_state_val('m', [19030.468, 16841.431])
phase.set_control_val('alpha', [0.0, 0.0])

dm.run_problem(p)
```

Now we pull the values of time and alpha from the timeseries.
Remember that the timeseries provides values at the nodes only. The fitting polynomials may exhibit oscillations between these nodes.

We use cubic interpolation to interpolate time and alpha onto 100 evenly distributed nodes across the phase, and then plot the results.

```python


t = p.get_val('traj.phase0.timeseries.time')
alpha = p.get_val('traj.phase0.timeseries.alpha', units='deg')

t_100 = phase.interp(xs=t, ys=t,
                     nodes=np.linspace(-1, 1, 100),
                     kind='cubic')
alpha_100 = phase.interp(xs=t, ys=alpha,
                         nodes=np.linspace(-1, 1, 100),
                         kind='cubic')

# Plot the solution nodes
%matplotlib inline
plt.plot(t, alpha, 'o')
plt.plot(t_100, alpha_100, '-')
plt.xlabel('time (s)')
plt.ylabel('angle of attack (deg)')
plt.grid()
plt.show()
```

Both the `simulate` method and the `interp` method can provide more dense output. The difference is that simulate is effectively finding a different solution for the states, by starting from the initial values and propagating the ODE forward in time using the interpolated control splines as inputs. The `interp` method is making some assumptions about the interpolation (for instance, cubic interpolation in this case), so it may not be a perfectly accurate representation of the underlying interpolating polynomials.
