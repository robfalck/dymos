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

(content:examples:race_car)=
# Race car Lap Simulation

```{admonition} Things you'll learn through this example
- Optimizing trajectories for cyclic events (such as race car laps)
    - Linking a phase with itself so it is ensured that start and end states are identical
- Using an independent variable other than time
```

```{admonition} The race track problem 
_The race track problem is different from other dymos example problems in that it is a cyclic problem.
The solution must be such that the start and end states are identical. The problem for this example
is given a race track shape, what is the optimum values for the controls such that the time elapsed to complete
the track is a minimum._
```

## Background of the problem

This example is based on the work of [Pieter de Buck](http://www.formulae.one/contact)
The goal of his problem was to find an optimal trajectory for the race car with active aerodynamic and
four wheel drive controls.

Pieter posted an [issue](https://github.com/OpenMDAO/dymos/issues/369) to the Dymos repository asking
how to use dymos for cyclic events such as this. His solution involved using an iterative solution, which worked but
dymos includes a feature that allows for simple and more efficient solution.
This example shows how to handle this situation using simple dymos calls without iteration.

If you are interested, Pieter and Adeline Shin have created an
[interactive visualization](http://www.formulae.one/lapsimulation) using his version of the code.

While Pieter's code includes several actual race tracks shapes, for simplicity and speed of solution, this dymos
example uses an oval track. If you
are interested in trying other race tracks, they are defined in the
[tracks.py](https://github.com/OpenMDAO/dymos/tree/master/dymos/examples/racecar/tracks.py) file.

## Optimizing trajectories for cyclic events

For trajectory optimization of cyclic events (such as race car laps), there needs to be
a way that ensure that the start and end states are identical.

In this race car example, the states and controls such as velocity, thrust, and steering angle need
to be the same at the start and finish.

Linking the controls is not really needed since it generally follows from the linked states.

There is a simple way to link the states of cyclic events in dymos.
You can use the `Trajectory.link_phases` functionality to "link" a
phase with itself. In this example this is accomplished with the following code which links all 7 state
variables.

```python
traj.link_phases(phases=['phase0', 'phase0'], vars=['V','n','alpha','omega','lambda','ax','ay'], locs=('final', 'initial'))
```

This code causes the value of `V` at the end of `phase0` ('final') to be linked via constraint
to the value of `V` at the beginning ('initial') of `phase0`.
This call does the same for the other 6 state variables as well.

See the [Trajectory API page](https://openmdao.github.io/dymos/api/trajectory_api.html)
for more information about the `link_phases` method.

## Using an independent variable other than time

In all the other dymos examples, time is the independent variable.
In this race car example, the independent variable is the arc length along the track.
Integration is done from the start of the track to the end.
The “typical” equations of motion are computed, so the rate of change of x wrt time (**$\frac{dx}{dt}$**).

By dividing the rates of change w.r.t. time by (**$\frac{ds}{dt}$**),
the rates of change of the states w.r.t. track arclength traversed are computed.

## Definition of the problem

Here are some figures which help define the problem.

This table describes the controls, states, and constraints. The symbols match up with the
variable names in the code shown at the bottom of this page.

### Table 1. Optimization problem formulation

|                     |                               | Symbol    | Lower | Upper | Units      |
|---------------------|-------------------------------|-----------|-------|-------|------------|
| **Minimize**        | Time                          |         t |       |       |          s |
| **With respect to** |                               |           |       |       |            |
| **Controls**        | Front left thrust             | $T_{fl}$  |       |       |            |
|                     | Front right thrust            | $T_{fr}$  |       |       |            |
|                     | Rear left thrust              | $T_{rl}$  |       |       |            |
|                     | Rear right thrust             | $T_{rr}$  |       |       |            |
|                     | Rear wing flap angle          | $\gamma$  |     0 |    50 |        deg |
|                     | Steering angle                |  $\delta$ |       |       |        rad |
| **States**          | Speed                         |         V |       |       |  $ms^{-1}$ |
|                     | Longitudinal acceleration     |     $a_x$ |       |       | $ms^{-2}$  |
|                     | Lateral acceleration          |     $a_y$ |       |       | $ms^{-2}$  |
|                     | Normal distance to centerline |         n |    -4 |     4 |          m |
|                     | Angle relative to centerline  |  $\alpha$ |       |       |        rad |
|                     | Slip angle                    | $\lambda$ |       |       |        rad |
|                     | Yaw rate                      | $\Omega$  |       |       | $rad^{-1}$ |
| **Subject to**      | Front left adherence          | $c_{fl}$  |     0 |     1 |            |
|                     | Front right adherence         | $c_{fr}$  |     0 |     1 |            |
|                     | Rear left adherence           | $c_{rl}$  |     0 |     1 |            |
|                     | Rear right adherence          | $c_{rr}$  |     0 |     1 |            |
|                     | Front left power              | $P_{fl}$  |       |    75 |         kW |
|                     | Front right power             | $P_{fr}$  |       |    75 |         kW |
|                     | Rear left power               | $P_{rl}$  |       |    75 |         kW |
|                     | Rear right power              | $P_{rr}$  |       |    75 |         kW |
|                     | Periodic state constraints    |           |       |       |            |
|                     | System dynamics               |           |       |       |            |
|                     | Track Layout                  |           |       |       |            |

The next figure defines the vehicle parameters. Again, the symbols match up with the
variable names in the code.

### Table 2. Vehicle parameters

| Parameter      | Value                                | Units        | Description                                |
|----------------|--------------------------------------|--------------|--------------------------------------------|
| $M$            | 1184                                 | $kg$         | Vehicle mass                               |
| $a$            | 1.404                                | $m$          | CoG to front axle distance                 |
| $b$            | 1.356                                | $m$          | CoG to rear axle distance                  |
| $t_w$          | 0.807                                | $m$          | Half track width                           |
| $h$            | 0.4                                  | $m$          | CoG height                                 |
| $I_z$          | 1775                                 | $kg\ m^{2}$  | Yaw inertia                                |
| $\beta$        | 0.62                                 | -            | Brake balance                              |
| $\chi$         | 0.5                                  | -            | Roll stiffness                             |
| $\rho$         | 1.2                                  | $kg\ m^{-3}$ | Air density                                |
| $\mu_{0}^{x}$  | 1.68                                 | -            | Longitudinal base friction coefficient     |
| $\mu_{0}^{y}$  | 1.68                                 | -            | Lateral base friction coefficient          |
| $K_{\mu}$      | -0.5                                 | -            | Tire load sensitivity                      |
| $K_{\lambda}$  | 44                                   | -            | Tire lateral stiffness                     |
| $\tau_{a_{x}}$ | 0.2                                  | $s$          | Longitudinal load transfer time constant   |
| $\tau_{a_{y}}$ | 0.2                                  | $s$          | Lateral load transfer time constant        |
| $S_{w}$        | 0.8                                  | $m^{2}$      | Wing planform area                         |
| $CoP$          | 1.404                                | $m$          | Center of pressure to front axle distance  |

There is also a figure which describes the states that govern the vehicle position
relative to the track.

![Vehicle Position States](vehicle_position_states.png)

Here is the code to solve this problem:

```python
# tags: remove-input, hide-output
om.display_source("dymos.examples.racecar.combinedODE")
```

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
# tags: output_scroll
import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import matplotlib as mpl

from dymos.examples.racecar.combinedODE import CombinedODE
from dymos.examples.racecar.spline import (get_spline, get_track_points, get_gate_normals,
                                           reverse_transform_gates, set_gate_displacements,
                                           transform_gates, )
from dymos.examples.racecar.linewidthhelper import linewidth_from_data_units
from dymos.examples.racecar.tracks import ovaltrack  # track curvature imports

# change track here and in curvature.py. Tracks are defined in tracks.py
track = ovaltrack

# generate nodes along the centerline for curvature calculation (different
# than collocation nodes)
points = get_track_points(track)

# fit the centerline spline.
finespline, gates, gatesd, curv, slope = get_spline(points, s=0.0)
# by default 10000 points
s_final = track.get_total_length()

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=CombinedODE,
                 transcription=dm.GaussLobatto(num_segments=50, order=3, compressed=True))

traj.add_phase(name='phase0', phase=phase)

# Set the time options, in this problem we perform a change of variables. So 'time' is
# actually 's' (distance along the track centerline)
# This is done to fix the collocation nodes in space, which saves us the calculation of
# the rate of change of curvature.
# The state equations are written with respect to time, the variable change occurs in
# timeODE.py
# Here we're using set_integ_var_options, which is just an alias for set_time_options.
phase.set_integ_var_options(fix_initial=True, fix_duration=True, duration_val=s_final,
                            targets=['curv.s'], units='m', duration_ref=s_final,
                            duration_ref0=10, name='s')

# Define states
phase.add_state('t', fix_initial=True, fix_final=False, units='s', lower=0,
                rate_source='dt_ds', ref=100)  # time
phase.add_state('n', fix_initial=False, fix_final=False, units='m', upper=4.0, lower=-4.0,
                rate_source='dn_ds', targets=['n'],
                ref=4.0)  # normal distance to centerline. The bounds on n define the
# width of the track
phase.add_state('V', fix_initial=False, fix_final=False, units='m/s', ref=40, ref0=5,
                rate_source='dV_ds', targets=['V'])  # velocity
phase.add_state('alpha', fix_initial=False, fix_final=False, units='rad',
                rate_source='dalpha_ds', targets=['alpha'],
                ref=0.15)  # vehicle heading angle with respect to centerline
phase.add_state('lambda', fix_initial=False, fix_final=False, units='rad',
                rate_source='dlambda_ds', targets=['lambda'],
                ref=0.01)  # vehicle slip angle, or angle between the axis of the vehicle
# and velocity vector (all cars drift a little)
phase.add_state('omega', fix_initial=False, fix_final=False, units='rad/s',
                rate_source='domega_ds', targets=['omega'], ref=0.3)  # yaw rate
phase.add_state('ax', fix_initial=False, fix_final=False, units='m/s**2',
                rate_source='dax_ds', targets=['ax'], ref=8)  # longitudinal acceleration
phase.add_state('ay', fix_initial=False, fix_final=False, units='m/s**2',
                rate_source='day_ds', targets=['ay'], ref=8)  # lateral acceleration

# Define Controls
phase.add_control(name='delta', units='rad', lower=None, upper=None, fix_initial=False,
                  fix_final=False, ref=0.04, rate_continuity=True)  # steering angle
phase.add_control(name='thrust', units=None, fix_initial=False, fix_final=False, rate_continuity=True)
# the thrust controls the longitudinal force of the rear tires and is positive while accelerating, negative while braking

# Performance Constraints
pmax = 960000  # W
phase.add_path_constraint('power', upper=pmax, ref=100000)  # engine power limit

# The following four constraints are the tire friction limits, with 'rr' designating the
# rear right wheel etc. This limit is computed in tireConstraintODE.py
phase.add_path_constraint('c_rr', upper=1)
phase.add_path_constraint('c_rl', upper=1)
phase.add_path_constraint('c_fr', upper=1)
phase.add_path_constraint('c_fl', upper=1)

# Some of the vehicle design parameters are available to set here. Other parameters can
# be found in their respective ODE files.
phase.add_parameter('M', val=800.0, units='kg', opt=False,
                    targets=['car.M', 'tire.M', 'tireconstraint.M', 'normal.M'],
                    static_target=True)  # vehicle mass
phase.add_parameter('beta', val=0.62, units=None, opt=False, targets=['tire.beta'],
                    static_target=True)  # brake bias
phase.add_parameter('CoP', val=1.6, units='m', opt=False, targets=['normal.CoP'],
                    static_target=True)  # center of pressure location
phase.add_parameter('h', val=0.3, units='m', opt=False, targets=['normal.h'],
                    static_target=True)  # center of gravity height
phase.add_parameter('chi', val=0.5, units=None, opt=False, targets=['normal.chi'],
                    static_target=True)  # roll stiffness
phase.add_parameter('ClA', val=4.0, units='m**2', opt=False, targets=['normal.ClA'],
                    static_target=True)  # downforce coefficient*area
phase.add_parameter('CdA', val=2.0, units='m**2', opt=False, targets=['car.CdA'],
                    static_target=True)  # drag coefficient*area

# Minimize final time.
# note that we use the 'state' time instead of Dymos 'time'
phase.add_objective('t', loc='final')

# Add output timeseries
phase.add_timeseries_output('*')
phase.add_timeseries_output('t', output_name='time')

# Link the states at the start and end of the phase in order to ensure a continous lap
traj.link_phases(phases=['phase0', 'phase0'],
                 vars=['V', 'n', 'alpha', 'omega', 'lambda', 'ax', 'ay'],
                 locs=('final', 'initial'))

# Set the driver. IPOPT or SNOPT are recommended but SLSQP might work.
p.driver = om.pyOptSparseDriver(optimizer='IPOPT')

p.driver.opt_settings['mu_init'] = 1e-3
p.driver.opt_settings['max_iter'] = 500
p.driver.opt_settings['acceptable_tol'] = 1e-3
p.driver.opt_settings['constr_viol_tol'] = 1e-3
p.driver.opt_settings['compl_inf_tol'] = 1e-3
p.driver.opt_settings['acceptable_iter'] = 0
p.driver.opt_settings['tol'] = 1e-3
p.driver.opt_settings['print_level'] = 0
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
p.driver.opt_settings['mu_strategy'] = 'monotone'
p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
p.driver.options['print_results'] = False

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup(check=True)  # force_alloc_complex=True
# Now that the OpenMDAO problem is setup, we can set the values of the states.

# States
# Nonzero velocity to avoid division by zero errors
phase.set_state_val('V', 20.0, units='m/s')
# All other states start at 0
phase.set_state_val('lambda', 0.0, units='rad')
phase.set_state_val('omega', 0.0, units='rad/s')
phase.set_state_val('alpha', 0.0, units='rad')
phase.set_state_val('ax', 0.0, units='m/s**2')
phase.set_state_val('ay', 0.0, units='m/s**2')
phase.set_state_val('n', 0.0, units='m')
# initial guess for what the final time should be
phase.set_state_val('t', [0.0, 100.0], units='s')

# Controls
# a small amount of thrust can speed up convergence
phase.set_control_val('delta', 0.0, units='rad')
phase.set_control_val('thrust', 0.1, units=None)

dm.run_problem(p, run_driver=True)
print('Optimization finished')

# Get optimized time series
n = p.get_val('traj.phase0.timeseries.n')
s = p.get_val('traj.phase0.timeseries.s')
V = p.get_val('traj.phase0.timeseries.V')
thrust = p.get_val('traj.phase0.timeseries.thrust')
delta = p.get_val('traj.phase0.timeseries.delta')
power = p.get_val('traj.phase0.timeseries.power', units='W')


```

```python
# Get optimized time series
n = p.get_val('traj.phase0.timeseries.n')
s = p.get_val('traj.phase0.timeseries.s')
V = p.get_val('traj.phase0.timeseries.V')
thrust = p.get_val('traj.phase0.timeseries.thrust')
delta = p.get_val('traj.phase0.timeseries.delta')
power = p.get_val('traj.phase0.timeseries.power', units='W')

# We know the optimal distance from the centerline (n). To transform this into the racing
# line we fit a spline to the displaced points. This will let us plot the racing line in
# x/y coordinates
normals = get_gate_normals(finespline, slope)
newgates = []
newnormals = []
newn = []
for i in range(len(n)):
    index = ((s[i] / s_final) * np.array(finespline).shape[1]).astype(
        int)  # interpolation to find the appropriate index
    if index[0] == np.array(finespline).shape[1]:
        index[0] = np.array(finespline).shape[1] - 1
    if i > 0 and s[i] == s[i - 1]:
        continue
    else:
        newgates.append([finespline[0][index[0]], finespline[1][index[0]]])
        newnormals.append(normals[index[0]])
        newn.append(n[i][0])

newgates = reverse_transform_gates(newgates)
displaced_gates = set_gate_displacements(newn, newgates, newnormals)
displaced_gates = np.array((transform_gates(displaced_gates)))

npoints = 1000
# fit the racing line spline to npoints
displaced_spline, gates, gatesd, curv, slope = get_spline(displaced_gates, 1 / npoints, 0)

plt.rcParams.update({'font.size': 12})


def plot_track_with_data(state, s):
    # this function plots the track
    state = np.array(state)[:, 0]
    s = np.array(s)[:, 0]
    s_new = np.linspace(0, s_final, npoints)

    # Colormap and norm of the track plot
    cmap = mpl.colormaps['viridis']
    norm = mpl.colors.Normalize(vmin=np.amin(state), vmax=np.amax(state))

    _, ax = plt.subplots(figsize=(15, 6))
    # establishes the figure axis limits needed for plotting the track below
    plt.plot(displaced_spline[0], displaced_spline[1], linewidth=0.1, solid_capstyle="butt")

    plt.axis('equal')
    # the linewidth is set in order to match the width of the track
    plt.plot(finespline[0], finespline[1], 'k',
             linewidth=linewidth_from_data_units(8.5, ax),
             solid_capstyle="butt")
    plt.plot(finespline[0], finespline[1], 'w', linewidth=linewidth_from_data_units(8, ax),
             solid_capstyle="butt")  # 8 is the width, and the 8.5 wide line draws 'kerbs'
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # plot spline with color
    for i in range(1, len(displaced_spline[0])):
        s_spline = s_new[i]
        index_greater = np.argwhere(s >= s_spline)[0][0]
        index_less = np.argwhere(s < s_spline)[-1][0]

        x = s_spline
        xp = np.array([s[index_less], s[index_greater]])
        fp = np.array([state[index_less], state[index_greater]])
        interp_state = np.interp(x, xp,
                                 fp)  # interpolate the given state to calculate the color

        # calculate the appropriate color
        state_color = norm(interp_state)
        color = cmap(state_color)
        color = mpl.colors.to_hex(color)

        # the track plot consists of thousands of tiny lines:
        point = [displaced_spline[0][i], displaced_spline[1][i]]
        prevpoint = [displaced_spline[0][i - 1], displaced_spline[1][i - 1]]
        if i <= 5 or i == len(displaced_spline[0]) - 1:
            plt.plot([point[0], prevpoint[0]], [point[1], prevpoint[1]], color,
                     linewidth=linewidth_from_data_units(1.5, ax), solid_capstyle="butt",
                     antialiased=True)
        else:
            plt.plot([point[0], prevpoint[0]], [point[1], prevpoint[1]], color,
                     linewidth=linewidth_from_data_units(1.5, ax),
                     solid_capstyle="projecting", antialiased=True)

    clb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), fraction=0.02,
                       ax=ax, pad=0.04)  # add colorbar

    if np.array_equal(state, V[:, 0]):
        clb.set_label('Velocity (m/s)')
    elif np.array_equal(state, thrust[:, 0]):
        clb.set_label('Thrust')
    elif np.array_equal(state, delta[:, 0]):
        clb.set_label('Delta')

    plt.tight_layout()
    plt.grid()


# Create the plots
plot_track_with_data(V, s)
plot_track_with_data(thrust, s)
plot_track_with_data(delta, s)

# Plot the main vehicle telemetry
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 8))

# Velocity vs s
axes[0].plot(s,
             p.get_val('traj.phase0.timeseries.V'), label='solution')

axes[0].set_xlabel('s (m)')
axes[0].set_ylabel('V (m/s)')
axes[0].grid()
axes[0].set_xlim(0, s_final)

# n vs s
axes[1].plot(s,
             p.get_val('traj.phase0.timeseries.n', units='m'), label='solution')

axes[1].set_xlabel('s (m)')
axes[1].set_ylabel('n (m)')
axes[1].grid()
axes[1].set_xlim(0, s_final)

# throttle vs s
axes[2].plot(s, thrust)

axes[2].set_xlabel('s (m)')
axes[2].set_ylabel('thrust')
axes[2].grid()
axes[2].set_xlim(0, s_final)

# delta vs s
axes[3].plot(s,
             p.get_val('traj.phase0.timeseries.delta', units=None),
             label='solution')

axes[3].set_xlabel('s (m)')
axes[3].set_ylabel('delta')
axes[3].grid()
axes[3].set_xlim(0, s_final)

plt.tight_layout()

# Performance constraint plot. Tire friction and power constraints
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 4))
plt.subplots_adjust(right=0.82, bottom=0.14, top=0.97, left=0.07)

axes.plot(s,
          p.get_val('traj.phase0.timeseries.c_fl', units=None), label='c_fl')
axes.plot(s,
          p.get_val('traj.phase0.timeseries.c_fr', units=None), label='c_fr')
axes.plot(s,
          p.get_val('traj.phase0.timeseries.c_rl', units=None), label='c_rl')
axes.plot(s,
          p.get_val('traj.phase0.timeseries.c_rr', units=None), label='c_rr')

axes.plot(s, power / pmax, label='Power')

axes.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
axes.set_xlabel('s (m)')
axes.set_ylabel('Performance constraints')
axes.grid()
axes.set_xlim(0, s_final)
plt.show()
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

# Test this example in Dymos' continuous integration process
assert_near_equal(p.get_val('traj.phase0.timeseries.t')[-1], 22.2657, tolerance=0.01)
```

```python
print(s)
```
