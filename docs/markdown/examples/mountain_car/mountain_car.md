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

(examples:the_mountain_car_problem)=
# The Mountain Car Problem

The mountain car problem proposes a vehicle stuck in a "well."  It lacks the power to directly climb out of the well, but instead must accelerate repeatedly forwards and backwards until it has achieved the energy necessary to exit the well.

The problem is a popular machine learning test case, though the methods in Dymos are capable of solving it.
It first appeared in the PhD thesis of Andrew Moore in 1990. {cite}`moore1990efficient`.
The implementation here is based on that given by Melnikov, Makmal, and Briegel {cite}`melnikov2014projective`.

## State and control variables

This system has two state variables, the position ($x$) and velocity ($v$) of the car. 

This system has a single control variable ($u$), the effort put into moving.  This control is contrained to the range $[-1 \, 1]$.

The dynamics of the system are governed by

\begin{align}
  \dot{x} &= v \\
  \dot{v} &= 0.001 * u - 0.0025 * \cos(3 x)
\end{align}



## Problem Definition

We seek to minimize the time required to exit the well in the positive direction.

\begin{align}
    \mathrm{Minimize} \, J &= t_f
\end{align}

Subject to the initial conditions

\begin{align}
    x_0 &= -0.5 \\
    v_0 &= 0.0
\end{align}

the control constraints

\begin{align}
    |u| \le 1
\end{align}

and the terminal constraints

\begin{align}
    x_f &= 0.5 \\
    v_f &\ge 0.0
\end{align}

## Defining the ODE

The following code implements the equations of motion for the mountain car problem.

A few things to note:

1. By providing the tag `dymos.state_rate_source:{name}`, we're letting Dymos know what states need to be integrated, there's no need to specify a rate source when using this ODE in our Phase.
2. Pairing the above tag with `dymos.state_units:{units}` means we don't have to specify units when setting properties for the state in our run script.
3. We only use compute_partials to override the values of $\frac{\partial \dot{v}}{\partial x}$ because $\frac{\partial \dot{v}}{\partial u}$ and $\frac{\partial \dot{x}}{\partial v}$ are constant and their value is specified during `setup`.

```python
import numpy as np
import openmdao.api as om


class MountainCarODE(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        nn = self.options['num_nodes']
        
        self.add_input('x', shape=(nn,), units='m')
        self.add_input('v', shape=(nn,), units='m/s')
        self.add_input('u', shape=(nn,), units='unitless')
        
        self.add_output('x_dot', shape=(nn,), units='m/s',
                        tags=['dymos.state_rate_source:x', 'dymos.state_units:m'])
        self.add_output('v_dot', shape=(nn,), units='m/s**2',
                        tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s'])
        
        ar = np.arange(nn, dtype=int)
        
        self.declare_partials(of='x_dot', wrt='v', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='v_dot', wrt='u', rows=ar, cols=ar, val=0.001)
        self.declare_partials(of='v_dot', wrt='x', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        x = inputs['x']
        v = inputs['v']
        u = inputs['u']
        outputs['x_dot'] = v
        outputs['v_dot'] = 0.001 * u - 0.0025 * np.cos(3*x)

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        partials['v_dot', 'x'] = 3 * 0.0025 * np.sin(3 * x)
        
```

## Solving the minimum-time mountain car problem with Dymos

The following script solves the minimum-time mountain car problem with Dymos.
Note that this example requires the IPOPT optimizer via the `pyoptsparse` package.
Scipy's SLSQP optimizer is generally not capable of solving this problem.

To begin, import the packages we require:

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
import dymos as dm
import matplotlib.pyplot as plt
from matplotlib import animation
```

Next, we set two constants.
`U_MAX` is the maximum allowable magnitude of the acceleration.
The references show this problem being solved with $-1 \le u \le 1$.

Variable `NUM_SEG` is the number of equally spaced polynomial segments into which time is being divided.
Within each of these segments, the time-history of each state and control is being treated as a polynomial (we're using the default order of 3).

```python
# The maximum absolute value of the acceleration authority of the car
U_MAX = 1.0

# The number of segments into which the problem is discretized
NUM_SEG = 30
```

We then instantiate an OpenMDAO problem and set the optimizer and its options.

For IPOPT, setting option `nlp_scaling_method` to `'gradient-based'` can substantially improve the convergence of the optimizer without the need for us to set all of the scaling manually.

The call to `declare_coloring` tells the optimizer to attempt to find a sparsity pattern that minimizes the work required to compute the derivatives across the model.

```python
#
# Initialize the Problem and the optimization driver
#
p = om.Problem()
               
p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
p.driver.opt_settings['print_level'] = 0
p.driver.opt_settings['max_iter'] = 500
p.driver.opt_settings['mu_strategy'] = 'adaptive'
p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
p.driver.opt_settings['tol'] = 1.0E-8
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence

p.driver.declare_coloring()
```

Next, we add a Dymos Trajectory group to the problem's model and add a phase to it.

In this case we're using the Radau pseudospectral transcription to solve the problem.

```python
#
# Create a trajectory and add a phase to it
#
traj = p.model.add_subsystem('traj', dm.Trajectory())
tx = transcription = dm.Radau(num_segments=NUM_SEG)
phase = traj.add_phase('phase0', dm.Phase(ode_class=MountainCarODE, transcription=tx))
```

At this point, we set the options on the main variables used in a Dymos phase.  

In addition to `time`, we have two states (`x` and `v`) and a single control (`u`).  

There are no parameters and no polynomial controls.
We could have tried to use a polynomial control here, but as we will see the solution contains large discontinuities in the control value, which make it ill-suited for a polynomial control.  Polynomial controls are modeled as a single (typically low-order) polynomial across the entire phase.

We're fixing the initial time and states to whatever values we provide before executing the problem.  We will constrain the final values with nonlinear constraints in the next step.

The scaler values (`ref`) are all set to 1 here.  We're using IPOPT's `gradient-based` scaling option and will let it work the scaling out for us.

Bounds on time duration are guesses, and the bounds on the states and controls come from the implementation in the references.

Also, we don't need to specify targets for any of the variables here because their names _are_ the targets in the top-level of the model.
The rate source and units for the states are obtained from the tags in the ODE component we previously defined.

```python
#
# Set the variables
#
phase.set_time_options(fix_initial=True, duration_bounds=(.05, 10000), duration_ref=1)

phase.add_state('x', fix_initial=True, fix_final=False, lower=-1.2, upper=0.5, ref=1, defect_ref=1)
phase.add_state('v', fix_initial=True, fix_final=False, lower=-0.07, upper=0.07, ref=1, defect_ref=1)
phase.add_control('u', lower=-U_MAX, upper=U_MAX, ref=1, continuity=True, rate_continuity=False)
```

Next we define the optimal control problem by specifying the objective, boundary constraints, and path constraints.

**Why do we have a path constraint on the control `u` when we've already specified its bounds?**

Excellent question!
In the `Radau` transcription, the $n^{th}$ order control polynomial is governed by design variables provided at $n$ points in the segment that **do not contain the right-most endpoint**.
Instead, this value is interpolated based on the values of the first $(n-1)$.
Since this value is not a design variable, it is necessary to constrain its value separately.
We could forgo specifying any bounds on `u` since it's completely covered by the path constraint, but specifying the bounds on the design variable values can sometimes help by telling the optimizer, "Don't even bother trying values outside of this range.".

Note that sometimes the opposite is true, and giving the optimizer the freedom to explore a larger design space, only to eventually be "reined-in" by the path constraint can sometimes be valuable.

The purpose of this interactive documentation is to let the user experiment.
If you remove the path constraint, you might notice some outlying control values in the solution below.

```python
#
# Minimize time at the end of the phase
#
phase.add_objective('time', loc='final', ref=1000)

phase.add_boundary_constraint('x', loc='final', lower=0.5)
phase.add_boundary_constraint('v', loc='final', lower=0.0)
phase.add_path_constraint('u', lower=-U_MAX, upper=U_MAX)

#
# Setup the Problem
#
p.setup()
```

We then set the initial guesses for the variables in the problem and solve it.

Since `fix_initial=True` is set for time and the states, those values are not design variables and will remain at the values given below throughout the solution process.

We're using the phase `interp` method to provide initial guesses for the states and controls.
In this case, by giving it two values, it is linearly interpolating from the first value to the second value, and then returning the interpolated value at the input nodes for the given variable.

Finally, we use the `dymos.run_problem` method to execute the problem.
This interface allows us to do some things that the standard OpenMDAO `problem.run_driver` interface does not.
It will automatically record the final solution achieved by the optimizer in case named `'final'` in a file called `dymos_solution.db`.
By specifying `simulate=True`, it will automatically follow the solution with an explicit integration using `scipy.solve_ivp`.
The results of the simulation are stored in a case named `final` in the file `dymos_simulation.db`.
This explicit simulation demonstrates how the system evolved with the given controls, and serves as a check that we're using a dense enough grid (enough segments and segments of sufficient order) to accurately represent the solution.

If those two solution didn't agree reasonably well, we could rerun the problem with a more dense grid.
Instead, we're asking Dymos to automatically change the grid if necessary by specifying `refine_method='ph'`.
This will attempt to repeatedly solve the problem and change the number of segments and segment orders until the solution is in reasonable agreement.

```python
# tags: hide-output
#
# Set the initial values
#
phase.set_time_val(initial=0.0, duration=500.0)

phase.set_state_val('x', [-0.5, 0.5])
phase.set_state_val('v', [0, 0.07])
phase.set_control_val('u', [0, np.sin(1.0)])

#
# Solve for the optimal trajectory
#
dm.run_problem(p, run_driver=True, simulate=True, refine_method='ph', refine_iteration_limit=5)

```

## Plotting the solution

The recommended practice is to obtain values from the recorded cases.
While the problem object can also be queried for values, building plotting scripts that use the case recorder files as the data source means that the problem doesn't need to be solved just to change a plot.
Here we load values of various variables from the solution and simulation for use in the animation to follow.

```python
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

t = sol.get_val('traj.phase0.timeseries.time')
x = sol.get_val('traj.phase0.timeseries.x')
v = sol.get_val('traj.phase0.timeseries.v')
u = sol.get_val('traj.phase0.timeseries.u')
h = np.sin(3 * x) / 3

t_sim = sim.get_val('traj.phase0.timeseries.time')
x_sim = sim.get_val('traj.phase0.timeseries.x')
v_sim = sim.get_val('traj.phase0.timeseries.v')
u_sim = sim.get_val('traj.phase0.timeseries.u')
h_sim = np.sin(3 * x_sim) / 3
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(t[-1, 0], 102.479, tolerance=5.0E-3)
assert_near_equal(x[-1, 0], 0.5, tolerance=5.0E-3)
```

## Animating the Solution

The collapsed code cell below contains the code used to produce an animation of the mountain car solution using Matplotlib.

The green area represents the hilly terrain the car is traversing.  The black circle is the center of the car, and the orange arrow is the applied control.

The applied control _generally_ has the same sign as the velocity and is 'bang-bang', that is, it wants to be at its maximum possible magnitude.  Interestingly, the sign of the control flips shortly before the sign of the velocity changes.

```python
# tags: hide-input
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
gs = fig.add_gridspec(3, 2)

anim_ax = fig.add_subplot(gs[:, 0])
anim_ax.set_aspect('equal')

x_ax = fig.add_subplot(gs[0, 1:])
v_ax = fig.add_subplot(gs[1, 1:])
u_ax = fig.add_subplot(gs[2, 1:])

x_ax.set_ylabel('x')
v_ax.set_ylabel('v')
u_ax.set_ylabel('u')
u_ax.set_xlabel('t')

# set up the subplots as needed
anim_ax.set_xlim((-1.75, 0.75))          
anim_ax.set_ylim((-1.25, 1.25))
anim_ax.set_xlabel('x')
anim_ax.set_ylabel('h')

x_sol_line, = x_ax.plot(t, x, 'o', ms=1, label='solution')
v_ax.plot(t, v, 'o', ms=1)
u_ax.plot(t, u, 'o', ms=1)

x_sim_line, = x_ax.plot([], [], '-', linewidth=3, label='simulation')
v_sim_line, = v_ax.plot([], [], '-', linewidth=3)
u_sim_line, = u_ax.plot([], [], '-', linewidth=3)

plt.figlegend(ncol=2, handles=[x_sol_line, x_sim_line], loc='upper center',
              bbox_to_anchor=(0.78, 0.98))

x_ax.grid(alpha=0.2)
txt_x = x_ax.text(0.8, 0.1, f'x = {x_sim[0, 0]:6.3f}', horizontalalignment='left',
                  verticalalignment='center', transform=x_ax.transAxes)

v_ax.grid(alpha=0.2)
txt_v = v_ax.text(0.8, 0.1, f'v = {v_sim[0, 0]:6.3f}', horizontalalignment='left',
                  verticalalignment='center', transform=v_ax.transAxes)

u_ax.grid(alpha=0.2)
txt_u = u_ax.text(0.8, 0.1, f'u = {u_sim[0, 0]:6.3f}', horizontalalignment='left',
                  verticalalignment='center', transform=u_ax.transAxes)

x_terrain = np.linspace(-1.75, 0.75, 100)
h_terrain = np.sin(3 * x_terrain) / 3
terrain_line, = anim_ax.plot(x_terrain, h_terrain, '-', color='tab:gray', lw=2)
terrain = anim_ax.fill_between(x_terrain, h_terrain, -1.25*np.ones_like(x_terrain), color='tab:green')
car, = anim_ax.plot([], [], 'ko', ms=12)
u_vec = anim_ax.quiver(x_sim[0] + 0.005, h_sim[0] + 0.005, u_sim[0], [0], scale=10, angles='xy', color='tab:orange')

# See https://brushingupscience.com/2019/08/01/elaborate-matplotlib-animations/ for quiver animation

ANIM_DURATION = 5
PAUSE_DURATION = 2
ANIM_FPS = 20

num_points = t_sim.size
num_frames = ANIM_DURATION * ANIM_FPS
pause_frames = PAUSE_DURATION * ANIM_FPS

idx_from_frame_num = np.linspace(0, num_points-1, num_frames, dtype=int)


def drawframe(n):

    if n >= idx_from_frame_num.size:
        idx = num_points - 1
    else:
        idx = idx_from_frame_num[n]

    x = x_sim[idx]
    v = v_sim[idx]
    u = u_sim[idx]
    h = np.sin(3 * x) / 3 + 0.025
    car.set_data(x, h)
    
    dh_dx = np.cos(3 * x)

    u_vec.set_offsets(np.atleast_2d(np.asarray([x + 0.005, h + 0.005]).T))
    u_vec.set_UVC(u * np.cos(dh_dx), u * np.sin(dh_dx))

    x_sim_line.set_data(t_sim[:idx], x_sim[:idx])
    v_sim_line.set_data(t_sim[:idx], v_sim[:idx])
    u_sim_line.set_data(t_sim[:idx], u_sim[:idx])

    txt_x.set_text(f'x = {x[0]:6.3f}')
    txt_v.set_text(f'v = {v[0]:6.3f}')
    txt_u.set_text(f'u = {u[0]:6.3f}')
    
    return car, u_vec, x_sim_line, v_sim_line, u_sim_line


# # blit=True re-draws only the parts that have changed.
# # repeat_delay has no effect when using to_jshtml, so pad drawframe to show the final frame for PAUSE_FRAMES extra frames.
anim = animation.FuncAnimation(fig, drawframe, frames=num_frames + pause_frames, interval=1000/ANIM_FPS, blit=True)
plt.close()  # Don't let jupyter display the un-animated plot

from IPython.display import HTML
js = anim.to_jshtml()
with open('mountain_car_anim.html', 'w') as f:
    print(js, file=f)
HTML(filename='mountain_car_anim.html')
```

## References

```{bibliography}
:filter: docname in docnames
```
