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

# Multi-Phase Cannonball

Maximizing the range of a cannonball in a vacuum is a typical
introductory problem for optimal control. In this example we are going
to demonstrate a more multidisciplinary take on the problem. We will
assume a density of the metal from which the cannonball is constructed,
and a cannon that can fire any diameter cannonball but is limited to a
maximum muzzle energy. If we make the cannonball large it will be heavy
and the cannon will not be capable of propelling it very far. If we make
the cannonball too small, it will have a low ballistic coefficient and
not be able to sustain its momentum in the presence of atmospheric drag.
Somewhere between these two extremes is the cannonball radius which
allows for maximum range flight.

The presence of atmospheric drag also means that we typically want to
launch the cannonball with more horizontal velocity, and thus use a
launch angle less than 45 degrees.

The goal of our optimization is to find the optimal design for the cannonball (its
radius) and the optimal flight profile (its launch angle)
simultaneously.

## Using two phases to capture an intermediate boundary constraint

This problem demonstrates the use of two phases to capture the state of
the system at an event in the trajectory. Here, we have the first phase
(ascent) terminate when the flight path angle reaches zero (apogee). The
descent phase follows until the cannonball impacts the ground.

The dynamics are given by

\begin{align}
  \frac{dv}{dt} &= \frac{D}{m} - g \sin \gamma \\
  \frac{d\gamma}{dt} &= - \frac{g \cos \gamma}{v} \\
  \frac{dh}{dt} &= v \sin \gamma \\
  \frac{dr}{dt} &= v \cos \gamma \\
\end{align}

The initial conditions are

\begin{align}
  r_0 &= 0 \rm{\,m} \\
  h_0 &= 100 \rm{\,m} \\
  v_0 &= \rm{free} \\
  \gamma_0 &= \rm{free}
\end{align}

and the final conditions are

\begin{align}
  h_f &= 0 \rm{\,m}
\end{align}

## Designing a cannonball for maximum range

This problem demonstrates a very simple vehicle design capability that
is run before the trajectory.

We assume our cannon can shoot a cannonball with some fixed kinetic
energy and that our cannonball is made of solid iron. The volume (and
mass) of the cannonball is proportional to its radius cubed, while the
cross-sectional area is proportional to its radius squared. If we
increase the size of the cannonball, the ballistic coefficient

\begin{align}
  BC &= \frac{m}{C_D A}
\end{align}

will increase, meaning the cannonball overcome air resistance more
easily and thus carry more distance.

However, making the cannonball larger also increases its mass. Our
cannon can impart the cannonball with, at most, 400 kJ of kinetic
energy. So making the cannonball larger will decrease the initial
velocity, and thus negatively impact its range.

We therefore have a design that affects the objective in competing ways.
We cannot make the cannonball too large, as it will be too heavy to
shoot. We also cannot make the cannonball too small, as it will be more
susceptible to air resistance. Somewhere in between is the sweet spot
that provides the maximum range cannonball.

## The cannonball sizing component

This compoennt computes the area (needed to compute drag) and mass (needed to compute energy) of a cannonball of a given radius and density.

This component sits upstream of the trajectory model and feeds its outputs to the trajectory as parameters.

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
# tags: output_scroll
import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

import dymos as dm
from dymos.models.atmosphere.atmos_1976 import USatm1976Data


#############################################
# Component for the design part of the model
#############################################
class CannonballSizeComp(om.ExplicitComponent):
    """
    Compute the area and mass of a cannonball with a given radius and density.

    Notes
    -----
    This component is not vectorized with 'num_nodes' as is the usual way
    with Dymos, but is instead intended to compute a scalar mass and reference
    area from scalar radius and density inputs. This component does not reside
    in the ODE but instead its outputs are connected to the trajectory via
    input design parameters.
    """
    def setup(self):
        self.add_input(name='radius', val=1.0, desc='cannonball radius', units='m')
        self.add_input(name='dens', val=7870., desc='cannonball density', units='kg/m**3')

        self.add_output(name='mass', shape=(1,), desc='cannonball mass', units='kg')
        self.add_output(name='S', shape=(1,), desc='aerodynamic reference area', units='m**2')

        self.declare_partials(of='mass', wrt='dens')
        self.declare_partials(of='mass', wrt='radius')

        self.declare_partials(of='S', wrt='radius')

    def compute(self, inputs, outputs):
        radius = inputs['radius']
        dens = inputs['dens']

        outputs['mass'] = (4/3.) * dens * np.pi * radius ** 3
        outputs['S'] = np.pi * radius ** 2

    def compute_partials(self, inputs, partials):
        radius = inputs['radius']
        dens = inputs['dens']

        partials['mass', 'dens'] = (4/3.) * np.pi * radius ** 3
        partials['mass', 'radius'] = 4. * dens * np.pi * radius ** 2

        partials['S', 'radius'] = 2 * np.pi * radius

```

## The cannonball ODE component

This component computes the state rates and the kinetic energy of the cannonball.
By calling the `declare_coloring` method wrt all inputs and using method `'cs'`, we're telling OpenMDAO to automatically determine the sparsity pattern of the outputs with respect to the inputs, **and** to automatically compute those outputs using complex-step approximation.

```python
class CannonballODE(om.ExplicitComponent):
    """
    Cannonball ODE assuming flat earth and accounting for air resistance
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # static parameters
        self.add_input('m', units='kg')
        self.add_input('S', units='m**2')
        # 0.5 good assumption for a sphere
        self.add_input('CD', 0.5)

        # time varying inputs
        self.add_input('h', units='m', shape=nn)
        self.add_input('v', units='m/s', shape=nn)
        self.add_input('gam', units='rad', shape=nn)

        # state rates
        self.add_output('v_dot', shape=nn, units='m/s**2', tags=['dymos.state_rate_source:v'])
        self.add_output('gam_dot', shape=nn, units='rad/s', tags=['dymos.state_rate_source:gam'])
        self.add_output('h_dot', shape=nn, units='m/s', tags=['dymos.state_rate_source:h'])
        self.add_output('r_dot', shape=nn, units='m/s', tags=['dymos.state_rate_source:r'])
        self.add_output('ke', shape=nn, units='J')

        # Ask OpenMDAO to compute the partial derivatives using complex-step
        # with a partial coloring algorithm for improved performance, and use
        # a graph coloring algorithm to automatically detect the sparsity pattern.
        self.declare_coloring(wrt='*', method='cs')

        alt_data = USatm1976Data.alt * om.unit_conversion('ft', 'm')[0]
        rho_data = USatm1976Data.rho * om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
        self.rho_interp = InterpND(points=np.array(alt_data),
                                   values=np.array(rho_data),
                                   method='slinear').interpolate

    def compute(self, inputs, outputs):

        gam = inputs['gam']
        v = inputs['v']
        h = inputs['h']
        m = inputs['m']
        S = inputs['S']
        CD = inputs['CD']

        GRAVITY = 9.80665  # m/s**2

        # handle complex-step gracefully from the interpolant
        if np.iscomplexobj(h):
            rho = self.rho_interp(inputs['h'])
        else:
            rho = self.rho_interp(inputs['h']).real

        q = 0.5*rho*inputs['v']**2
        qS = q * S
        D = qS * CD
        cgam = np.cos(gam)
        sgam = np.sin(gam)
        outputs['v_dot'] = - D/m-GRAVITY*sgam
        outputs['gam_dot'] = -(GRAVITY/v)*cgam
        outputs['h_dot'] = v*sgam
        outputs['r_dot'] = v*cgam
        outputs['ke'] = 0.5*m*v**2
```

## Building and running the problem

The following code defines the components for the physical
cannonball calculations and ODE problem, sets up trajectory using two phases,
and links them accordingly. The initial flight path angle is free, since
45 degrees is not necessarily optimal once air resistance is taken into
account.

```python
p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.declare_coloring()

p.model.add_subsystem('size_comp', CannonballSizeComp(),
                      promotes_inputs=['radius', 'dens'])
p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
p.model.add_design_var('radius', lower=0.01, upper=0.10,
                       ref0=0.01, ref=0.10, units='m')

traj = p.model.add_subsystem('traj', dm.Trajectory())

transcription = dm.Radau(num_segments=5, order=3, compressed=True)
ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

ascent = traj.add_phase('ascent', ascent)

# All initial states except flight path angle are fixed
# Final flight path angle is fixed (we will set it to zero
# so that the phase ends at apogee).
# The output of the ODE which provides the rate source for each state
# is obtained from the tags used on those outputs in the ODE.
# The units of the states are automatically inferred by multiplying the units
# of those rates by the time units.
ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100),
                        duration_ref=100, units='s')
ascent.set_state_options('r', fix_initial=True, fix_final=False)
ascent.set_state_options('h', fix_initial=True, fix_final=False)
ascent.set_state_options('gam', fix_initial=False, fix_final=True)
ascent.set_state_options('v', fix_initial=False, fix_final=False)

ascent.add_parameter('S', units='m**2', static_target=True)
ascent.add_parameter('m', units='kg', static_target=True)

# Limit the muzzle energy
ascent.add_boundary_constraint('ke', loc='initial',
                               upper=400000, lower=0, ref=100000)

# Second Phase (descent)
transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
# descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)
descent = ascent.duplicate(transcription=transcription)

traj.add_phase('descent', descent)

# Because we copied the descent phase
# - The 'fix_initial' option for time was set to False
# - All state 'fix_initial' and 'fix_final' options are set to False.
# - We only need to fix the final value of altitude so the descent phase ends at ground impact.
descent.set_state_options('h', fix_final=True)
descent.add_objective('r', loc='final', scaler=-1.0)

# Add internally-managed design parameters to the trajectory.
traj.add_parameter('CD',
                   targets={'ascent': ['CD'], 'descent': ['CD']},
                   val=0.5, units=None, opt=False, static_target=True)

# Add externally-provided design parameters to the trajectory.
# In this case, we connect 'm' to pre-existing input parameters
# named 'mass' in each phase.
traj.add_parameter('m', units='kg', val=1.0,
                   targets={'ascent': 'm', 'descent': 'm'},
                   static_target=True)

# In this case, by omitting targets, we're connecting these
# parameters to parameters with the same name in each phase.
traj.add_parameter('S', units='m**2', val=0.005, static_target=True)

# Link Phases (link time and all state variables)
traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

# Issue Connections
p.model.connect('size_comp.mass', 'traj.parameters:m')
p.model.connect('size_comp.S', 'traj.parameters:S')

# A linear solver at the top level can improve performance.
p.model.linear_solver = om.DirectSolver()

# Finish Problem Setup
p.setup()

#############################################
# Set constants and initial guesses
#############################################
p.set_val('radius', 0.05, units='m')
p.set_val('dens', 7.87, units='g/cm**3')

traj.set_parameter_val('CD', 0.5)


ascent.set_time_val(initial=0.0, duration=10.0)

ascent.set_state_val('r', [0, 100])
ascent.set_state_val('h', [0, 100])
ascent.set_state_val('v', [200, 150])
ascent.set_state_val('gam', [25, 0], units='deg')

descent.set_time_val(initial=10.0, duration=10.0)

descent.set_state_val('r', [100, 200])
descent.set_state_val('h', [100, 0])
descent.set_state_val('v', [150, 200])
descent.set_state_val('gam', [0, -45], units='deg')

#####################################################
# Run the optimization and final explicit simulation
#####################################################
dm.run_problem(p, simulate=True)
```

## Plotting the results

```python
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

#############################################
# Plot the results
#############################################
rad = p.get_val('radius', units='m')[0]
print(f'optimal radius: {rad} m ')
mass = p.get_val('size_comp.mass', units='kg')[0]
print(f'cannonball mass: {mass} kg ')
area = p.get_val('size_comp.S', units='cm**2')[0]
print(f'cannonball aerodynamic reference area: {area} cm**2 ')
angle = p.get_val('traj.ascent.timeseries.gam', units='deg')[0, 0]
print(f'launch angle: {angle} deg')
max_range = p.get_val('traj.descent.timeseries.r')[-1, 0]
print(f'maximum range: {max_range} m')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

time_imp = {'ascent': p.get_val('traj.ascent.timeseries.time'),
            'descent': p.get_val('traj.descent.timeseries.time')}

time_exp = {'ascent': sim.get_val('traj.ascent.timeseries.time'),
            'descent': sim.get_val('traj.descent.timeseries.time')}

r_imp = {'ascent': p.get_val('traj.ascent.timeseries.r'),
         'descent': p.get_val('traj.descent.timeseries.r')}

r_exp = {'ascent': sim.get_val('traj.ascent.timeseries.r'),
         'descent': sim.get_val('traj.descent.timeseries.r')}

h_imp = {'ascent': p.get_val('traj.ascent.timeseries.h'),
         'descent': p.get_val('traj.descent.timeseries.h')}

h_exp = {'ascent': sim.get_val('traj.ascent.timeseries.h'),
         'descent': sim.get_val('traj.descent.timeseries.h')}

axes.plot(r_imp['ascent'], h_imp['ascent'], 'bo')

axes.plot(r_imp['descent'], h_imp['descent'], 'ro')

axes.plot(r_exp['ascent'], h_exp['ascent'], 'b--')

axes.plot(r_exp['descent'], h_exp['descent'], 'r--')

axes.set_xlabel('range (m)')
axes.set_ylabel('altitude (m)')
axes.grid(True)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
states = ['r', 'h', 'v', 'gam']
for i, state in enumerate(states):
    x_imp = {'ascent': sol.get_val(f'traj.ascent.timeseries.{state}'),
             'descent': sol.get_val(f'traj.descent.timeseries.{state}')}

    x_exp = {'ascent': sim.get_val(f'traj.ascent.timeseries.{state}'),
             'descent': sim.get_val(f'traj.descent.timeseries.{state}')}

    axes[i].set_ylabel(state)
    axes[i].grid(True)

    axes[i].plot(time_imp['ascent'], x_imp['ascent'], 'bo')
    axes[i].plot(time_imp['descent'], x_imp['descent'], 'ro')
    axes[i].plot(time_exp['ascent'], x_exp['ascent'], 'b--')
    axes[i].plot(time_exp['descent'], x_exp['descent'], 'r--')

plt.show()
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(sol.get_val('traj.descent.states:r')[-1],
                  3183.25, tolerance=1.0E-2)
```
