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

(examples:hull)=
# The Hull Problem

The Hull problem is a 1-DOF optimal control problem {cite}`hull2003oct`. It can be stated as:

Minimize the control effort required to move a frictionless sliding block from some initial position such that the final displacement from a pre-specified point is minimized.

## State and control variables

This system has one state variables, the position ($x$) of the sliding block. 

This system has a single control variable ($u$), the velocity of the block.

The dynamics of the system are governed by

\begin{align}
  \dot{x} &= u
\end{align}



## Problem Definition

We seek to minimize the control effort required and minimize the displacement from the origin.

\begin{align}
    \mathrm{Minimize} \, J &= 2.5x_f^2 \, + \, 0.5 \int_0^1 u^2 dt
\end{align}

Subject to the initial conditions

\begin{align}
    t_0 &= 0.0 \\
    x_0 &= 1.5
\end{align}

and the terminal constraints

\begin{align}
    t_f &= 10.0
\end{align}



## Dealing with combined terminal and integral costs in Dymos

In classic optimal control, the objective is often broken into the terminal component (the Mayer term) and the integral component (the Lagrange term).
Dymos does not distinguish between the two.
In this case, since the objective $J$ consists of both a terminal cost and an integrated cost (Bolza form), we add a term to the ODE to account for the integrated quantity


\begin{align}
  \dot{x_L} &= L \\
  L &= 0.5 u^2
\end{align}

where $x_L$ is a state added to account for the Lagrange term.

Dymos supports the definition of simple mathematical expressions as the cost, so the final value of $x_L$ can be added to the final value of $2.5x^2$.

## Defining the ODE

The following code implements the equations of motion for the Hull problem.
Since the rate of $x$ is given by a control ($u$), there is no need to compute its rate in the ODE.
Dymos can pull their values from those other states and controls.
The ODE, therefore, only needs to compute the rate of change of $x_L$ ($L$).

A few things to note:

1. By providing the tag `dymos.state_rate_source:{name}`, we're letting Dymos know what states need to be integrated, there's no need to specify a rate source when using this ODE in our Phase.
2. Pairing the above tag with `dymos.state_units:{units}` means we don't have to specify units when setting properties for the state in our run script.

```python
import numpy as np
import openmdao.api as om


class HullProblemODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('u', val=np.zeros(nn), desc='control')

        self.add_output('L', val=np.zeros(nn), desc='Lagrangian', units='1/s')

        # Setup partials
        self.declare_partials(of='L', wrt='u', rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        u = inputs['u']

        outputs['L'] = 0.5 * u ** 2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        u = inputs['u']
        partials['L', 'u'] = u

```

## Solving the Hull problem with Dymos

The following script solves the Hull problem with Dymos.

To begin, import the packages we require:

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
import dymos as dm
import matplotlib.pyplot as plt
```

We then instantiate an OpenMDAO problem and set the optimizer and its options.

The call to `declare_coloring` tells the optimizer to attempt to find a sparsity pattern that minimizes the work required to compute the derivatives across the model.


```python
#
# Initialize the Problem and the optimization driver
#
p = om.Problem()
               
p.driver = om.pyOptSparseDriver()
p.driver.declare_coloring()
```

Next, we add a Dymos Trajectory group to the problem's model and add a phase to it.

In this case we're using the Radau pseudospectral transcription to solve the problem.

```python
#
# Create a trajectory and add a phase to it
#
traj = p.model.add_subsystem('traj', dm.Trajectory())
tx = transcription = dm.Radau(num_segments=24)
phase = traj.add_phase('phase0', dm.Phase(ode_class=HullProblemODE, transcription=tx))
```

At this point, we set the options on the main variables used in a Dymos phase.  

In addition to `time`, we have two states (`x`, and `x_L`) and a single control (`u`).  

Here we use bounds on the states themselves to constrain the initial values of `x` and1 `x_L`.
From an optimization perspective, this means that we are removing the first and last values in the state histories of $x$ and $x_L$ from the vector of design variables.
Their initial values will remain unchanged throughout the optimization process.

On the other hand, we could specify `fix_initial=False, fix_final=False` for these values, and Dymos would be free to change them.
We would then need to put a boundary constraint in place to enforce their final values.
Feel free to experiment with different ways of enforcing the boundary constraints on this problem and see how it affects performance.

The scaler values (`ref`) are all set to 1 here.

Also, we don't need to specify targets for any of the variables here because their names _are_ the targets in the top-level of the model.
The rate source and units for the states are obtained from the tags in the ODE component we previously defined.


```python
#
# Set the variables
#
phase.set_time_options(fix_initial=True, fix_duration=True)

phase.set_time_options(fix_initial=True, fix_duration=True)
phase.add_state('x', fix_initial=True, fix_final=False, rate_source='u')
phase.add_state('xL', fix_initial=True, fix_final=False, rate_source='L')
phase.add_parameter('u', opt=True, targets=['u'], units='1/s')
phase.add_timeseries_output('u')

```

```python
#
# Minimize time at the end of the phase
#
phase.add_objective('J = 2.5*x**2 + xL')

#
# Setup the Problem
#
p.setup()
```

We then set the initial guesses for the variables in the problem and solve it.

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
phase.set_state_val('x', [1.5, 1])
phase.set_state_val('xL', [0, 1])
phase.set_time_val(initial=0.0, duration=1.0)
phase.set_parameter_val('u', 0.7)

#
# Solve for the optimal trajectory
#
dm.run_problem(p, run_driver=True, simulate=True)

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
xL = sol.get_val('traj.phase0.timeseries.xL')
u = sol.get_val('traj.phase0.timeseries.u')

t_sim = sim.get_val('traj.phase0.timeseries.time')
x_sim = sim.get_val('traj.phase0.timeseries.x')
xL_sim = sim.get_val('traj.phase0.timeseries.xL')
u_sim = sim.get_val('traj.phase0.timeseries.u')

fig = plt.figure(constrained_layout=True, figsize=(12, 4))
gs = fig.add_gridspec(3, 1)

x_ax = fig.add_subplot(gs[0, 0])
xL_ax = fig.add_subplot(gs[1, 0])
u_ax = fig.add_subplot(gs[2, 0])

x_ax.set_ylabel('x ($m$)')
xL_ax.set_ylabel('xL ($m/s$)')
u_ax.set_ylabel('u ($m/s^2$)')
xL_ax.set_xlabel('t (s)')
u_ax.set_xlabel('t (s)')

x_sol_handle, = x_ax.plot(t, x, 'o', ms=1)
xL_ax.plot(t, xL, 'o', ms=1)
u_ax.plot(t, u, 'o', ms=1)

x_sim_handle, = x_ax.plot(t_sim, x_sim, '-', ms=1)
xL_ax.plot(t_sim, xL_sim, '-', ms=1)
u_ax.plot(t_sim, u_sim, '-', ms=1)

for ax in [x_ax, xL_ax, u_ax]:
    ax.grid(True, alpha=0.2)
    
plt.figlegend([x_sol_handle, x_sim_handle], ['solution', 'simulation'], ncol=2, loc='lower center');

```

## References

```{bibliography}
:filter: docname in docnames
```
