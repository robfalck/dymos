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

# Water Rocket

Author: Bernardo Bahia Monteiro (bbahia@umich.edu)

In this example, we will optimize a water rocket for range and height at the apogee, using design variables that are easily modifiable just before launch: the empty mass, the initial water volume and the launch angle.
This example builds on [multi-phase cannonball](../multi_phase_cannonball/multi_phase_cannonball.ipynb) ane is adapted from _Optimization of a Water Rocket in OpenMDAO/Dymos_ {cite}`bbahia_2020`.

## Nomenclature

| Symbol               | definition                             |
|----------------------|----------------------------------------|
| $v_\text{out}$       | water exit speed at the nozzle         |
| $A_\text{out}$       | nozzle area                            |
| $V_w$                | water volume in the rocket             |
| $p$                  | pressure in the rocket                 |
| $p_a$                | ambient pressure                       |
| $\dot{\,}$           | time derivative                        |
| $k$                  | polytropic constant                    |
| $V_b$                | internal volume of the rocket          |
| $\rho_w$             | water density                          |
| $T$                  | thrust                                 |
| $q$                  | dynamic pressure                       |
| $S$                  | cross sectional area                   |
| $(\cdot)_0$          | value of $(\cdot)$ at $t=0$            |
| $t$                  | time                                   |

## Problem Formulation

A natural objective function for a water rocket is the maximum height achieved by the rocket during flight, or the horizontal distance it travels, i.e. its range.
The design of a water rocket is somewhat constrained by the soda bottle used as its engine.
This means that the volume available for water and air is fixed, the initial launch pressure is limited by the bottle's strength (since the pressure is directly related to the energy available for the rocket, it is easy to see that it should be as high as possible) and the nozzle throat area is also fixed.
Given these manufacturing constraints, the design variables we are left with are the empty mass (it can be easily changed through adding ballast), the water volume at the launch, and the launch angle.
With this considerations in mind, a natural formulation for the water rocket problem is

\begin{align}
    \text{maximize}   &\quad \text{range or height} \\
    \text{w.r.t.}     &\quad \text{empty mass, initial water volume, launch angle, trajectory} \\
    \text{subject to} &\quad \text{flight dynamics} \\
                      &\quad \text{fluid dynamics inside the rocket} \\
                      &\quad 0 < \text{initial water volume} < \text{volume of bottle} \\
                      &\quad 0^\circ < \text{launch angle} < 90^\circ \\
                      &\quad 0 < \text{empty mass}
\end{align}

##  Model

The water rocket model is divided into three basic components: a *water engine*, responsible for modelling the fluid dynamics inside the rocket and returning its thrust;  the *aerodynamics*, responsible for calculating the atmospheric drag of the rocket; and the *equations of motion*, responsible for propagating the rocket's trajectory in time, using Newton's laws and the forces provided by the other two components.

In order to integrate these three basic components, some additional interfacing components are necessary: an atmospheric model to provide values of ambient pressure for the water engine and air density to the calculation of the dynamic pressure for the aerodynamic model, and a component that calculates the instantaneous mass of the rocket by summing the water mass with the rocket's empty mass.
The high level layout of this model is shown in below.

<figure>
  <img src="figures/water_rocket_overview.svg"/>
  <figcaption>N2 diagram for the water rocket model</figcaption>
</figure>

`atmos`, `dynamic_pressure`, `aero` and `eom` are the same models used in [multi-phase cannonball](../multi_phase_cannonball/multi_phase_cannonball.ipynb).
The remaining components are discussed below.

```{Warning}
The `eom` component has a singularity in the flight path angle derivative when the flight speed is zero.
This happens because the rotational dynamics are not modelled.
This can cause convergence problems if the launch velocity is set to zero or the launch angle is set to $90^\circ$
```

```{Note}
Since the range of altitudes achieved by the water rocket is very small (100m), the air density and pressure are practically constant, thus the use of an atmospheric model is not necessary. However, using it makes it easier to reuse code from [multi-phase cannonball](../multi_phase_cannonball/multi_phase_cannonball.ipynb).
```

### Water engine

The water engine is modelled by assuming that the air expansion in the rocket
follows an adiabatic process and the water flow is incompressible and inviscid,
i.e.  it follows Bernoulli's equation. We also make the following simplifying
assumptions:

1. The thrust developed after the water is depleted is negligible
2. The area inside the bottle is much smaller than the nozzle area
3. The inertial forces do not affect the fluid dynamics inside the bottle

This simplified modelling can be found in Prusa[@Prusa2000].
A more rigorous formulation, which drops all these simplifying assumptions can be found in Wheeler[@Wheeler2002], Gommes[@Gommes2010], and Barria-Perotti[@BarrioPerotti2010].

The first assumption leads to an underestimation of the rocket performance, since the air left in the bottle after it is out of water is known to generate appreciable thrust[@Thorncroft2009].
This simplified model, however, produces physically meaningful results.

There are two states in this dynamical model, the water volume in the rocket $V_w$ and the gauge pressure inside the rocket $p$.
The constitutive equations and the N2 diagram showing the model organization are shown below.

### Constitutive equations of the water engine model
| Component              | Equation                                                    |
| -----------------------|-------------------------------------------------------------|
| water_exhaust_speed    | $v_\text{out} = \sqrt{2(p-p_a)/\rho_w}$                     |
| water_flow_rate        | $\dot{V}_w = -v_\text{out} A_\text{out}$                    |
| pressure_rate          | $\dot{p} = kp\frac{\dot{V_w}}{(V_b-V_w)}$                   |
| water_thrust           | $T = (\rho_w v_\text{out})(v_\text{out}A_\text{out})$       |

<figure>
  <img src="figures/water_rocket_waterengine.svg"/>
  <figcaption>N2 diagram for the water engine group</figcaption>
</figure>

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import openmdao.api as om
import dymos as dm
```

```python
# tags: remove-input, remove-output
from openmdao.utils.notebook_utils import display_source
```

```python
# tags: remove-input
display_source('dymos.examples.water_rocket.water_engine_comp.WaterEngine')
```

```python
# tags: remove-input
display_source('dymos.examples.water_rocket.water_engine_comp._WaterExhaustSpeed')
```

```python
# tags: remove-input
display_source('dymos.examples.water_rocket.water_engine_comp._PressureRate')
```

```python
# tags: remove-input
display_source('dymos.examples.water_rocket.water_engine_comp._WaterFlowRate')
```

The `_MassAdder` component calculates the rocket's instantaneous mass by
summing the water mass with the rockets empty mass, i.e.

\begin{align}
    m = m_\text{empty}+\rho_w V_w
\end{align}

```python
# tags: remove-input
display_source('dymos.examples.water_rocket.water_propulsion_ode._MassAdder')
```

Now these components are joined in a single group

```python
display_source('dymos.examples.water_rocket.water_propulsion_ode.WaterPropulsionODE')
```

## Phases

The flight of the water rocket is split in three distinct phases: propelled ascent, ballistic ascent and ballistic descent.
If the simplification of no thrust without water were lifted, there would be an extra "air propelled ascent" phase between the propelled ascent and ballistic ascent phases.

**Propelled ascent:** is the flight phase where the rocket still has water inside, and hence it is producing thrust.
The thrust is given by the water engine model, and fed into the flight dynamic equations.
It starts at launch and finishes when the water is depleted, i.e. $V_w=0$.

**Ballistic ascent:** is the flight phase where the rocket is ascending ($\gamma>0$) but produces no thrust.
This phase begins at the end of thepropelled ascent phase and ends at the apogee, defined by $\gamma=0$.

**Descent:** is the phase where the rocket is descending without thrust.
It begins at the end of the ballistic ascent phase and ends with ground impact, i.e. $h=0$.

```python
# tags: remove-input
display_source('dymos.examples.water_rocket.phases.new_propelled_ascent_phase')
```

```python
# tags: remove-input
display_source('dymos.examples.water_rocket.phases.new_ballistic_ascent_phase')
```

```python
# tags: remove-input
display_source('dymos.examples.water_rocket.phases.new_descent_phase')
```

##  Model parameters

The model requires a few constant parameters.
The values used are shown in the following table.

Values for parameters in the water rocket model

|   Parameter        | Value                | Unit         | Reference                                           |
|--------------------|----------------------|--------------|-----------------------------------------------------|
| $C_D$              | 0.3450               | -            | {cite}`BarrioPerotti2009`                           |
| $S$                | $\pi 106^2/4$        | $mm^2$       | {cite}`BarrioPerotti2009`                           |
| $k$                | 1.2                  | -            | {cite}`Thorncroft2009` {cite}`Fischer2020` {cite}`Romanelli2013`   |
| $A_\text{out}$     | $\pi22^2/4$          | $mm^2$       | {cite}`aircommand_nozzle`                                |
| $V_b$              | 2                    | L            |                                                     |
| $\rho_w$           | 1000                 | $kg/m^3$     |                                                     |
| $p_0$              | 6.5                  | bar          |                                                     |
| $v_0$              | 0.1                  | $m/s$        |                                                     |
| $h_0$              | 0                    | $m$          |                                                     |
| $r_0$              | 0                    | $m$          |                                                     |

Values for the bottle volume $V_b$, its cross-sectional area $S$ and the nozzle area $A_\text{out}$ are determined by the soda bottle that makes the rocket primary structure, and thus are not easily modifiable by the designer.
The polytropic coefficient $k$ is a function of the moist air characteristics inside the rocket.
The initial speed $v_0$ must be set to a value higher than zero, otherwise the flight dynamic equations become singular.
This issue arises from the angular dynamics of the rocket not being modelled.
The drag coefficient $C_D$ is sensitive to the aerodynamic design, but can be optimized by a single discipline analysis.
The initial pressure $p_0$ should be maximized in order to obtain the maximum range or height for the rocket.
It is limited by the structural properties of the bottle, which are modifiable by the designer, since the bottle needs to be available commercially.
Finally, the starting point of the rocket is set to the origin.

## Putting it all together

The different phases must be combined in a single trajectory, and linked in a sequence.
Here we also define the design variables.

```python
display_source('dymos.examples.water_rocket.phases.new_water_rocket_trajectory')
```

## Helper Functions to Access the Results

```python
from collections import namedtuple


def summarize_results(water_rocket_problem):
    p = water_rocket_problem
    Entry = namedtuple('Entry', 'value unit')
    summary = {
        'Launch angle': Entry(p.get_val('traj.propelled_ascent.timeseries.gam',  units='deg')[0, 0], 'deg'),
        'Flight angle at end of propulsion': Entry(p.get_val('traj.propelled_ascent.timeseries.gam',
                                                   units='deg')[-1, 0], 'deg'),
        'Empty mass': Entry(p.get_val('traj.parameters:m_empty', units='kg')[0], 'kg'),
        'Water volume': Entry(p.get_val('traj.propelled_ascent.timeseries.V_w', 'L')[0, 0], 'L'),
        'Maximum range': Entry(p.get_val('traj.descent.timeseries.r', units='m')[-1, 0], 'm'),
        'Maximum height': Entry(p.get_val('traj.ballistic_ascent.timeseries.h', units='m')[-1, 0], 'm'),
        'Maximum velocity': Entry(p.get_val('traj.propelled_ascent.timeseries.v', units='m/s')[-1, 0], 'm/s'),
    }

    return summary

```

```python
colors = {'pa': 'tab:blue', 'ba': 'tab:orange', 'd': 'tab:green'}


def plot_propelled_ascent(p, exp_out):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(12, 6))
    t_imp = p.get_val('traj.propelled_ascent.timeseries.time', 's')
    t_exp = exp_out.get_val('traj.propelled_ascent.timeseries.time', 's')
    
    c = colors['pa']

    ax[0, 0].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.p', 'bar'), '.', color=c)
    ax[0, 0].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.p', 'bar'), '-', color=c)
    ax[0, 0].set_ylabel('p (bar)')
    ax[0, 0].set_ylim(bottom=0)

    ax[1, 0].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.V_w', 'L'), '.', color=c)
    ax[1, 0].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.V_w', 'L'), '-', color=c)
    ax[1, 0].set_ylabel('$V_w$ (L)')

    ax[0, 1].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.T', 'N'), '.', color=c)
    ax[0, 1].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.T', 'N'), '-', color=c)
    ax[0, 1].set_ylabel('T (N)')
    ax[0, 1].set_ylim(bottom=0)

    ax[1, 1].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.v', 'm/s'), '.', color=c)
    ax[1, 1].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.v', 'm/s'), '-', color=c)
    ax[1, 1].set_ylabel('v (m/s)')
    ax[1, 1].set_ylim(bottom=0)

    ax[1, 0].set_xlabel('t (s)')
    ax[1, 1].set_xlabel('t (s)')
    
    for i in range(4):
        ax.ravel()[i].grid(True, alpha=0.2)

    fig.tight_layout()
```

```python
def plot_states(p, exp_out, legend_loc='right', legend_ncol=3):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4), sharex=True)

    states = ['r', 'h', 'v', 'gam']
    units = ['m', 'm', 'm/s', 'deg']
    phases = ['propelled_ascent', 'ballistic_ascent', 'descent']

    time_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.time'),
                'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.time'),
                'descent': p.get_val('traj.descent.timeseries.time')}

    time_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.time'),
                'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                'descent': exp_out.get_val('traj.descent.timeseries.time')}

    x_imp = {phase: {state: p.get_val(f"traj.{phase}.timeseries.{state}", unit)
                     for state, unit in zip(states, units)
                     }
             for phase in phases
             }

    x_exp = {phase: {state: exp_out.get_val(f"traj.{phase}.timeseries.{state}", unit)
                     for state, unit in zip(states, units)
                     }
             for phase in phases
             }

    for i, (state, unit) in enumerate(zip(states, units)):
        axes.ravel()[i].set_ylabel(f"{state} ({unit})" if state != 'gam' else f'$\gamma$ ({unit})')

        axes.ravel()[i].plot(time_imp['propelled_ascent'], x_imp['propelled_ascent'][state], '.', color=colors['pa'])
        axes.ravel()[i].plot(time_imp['ballistic_ascent'], x_imp['ballistic_ascent'][state], '.', color=colors['ba'])
        axes.ravel()[i].plot(time_imp['descent'], x_imp['descent'][state], '.', color=colors['d'])
        h1, = axes.ravel()[i].plot(time_exp['propelled_ascent'], x_exp['propelled_ascent'][state], '-', color=colors['pa'], label='Propelled Ascent')
        h2, = axes.ravel()[i].plot(time_exp['ballistic_ascent'], x_exp['ballistic_ascent'][state], '-', color=colors['ba'], label='Ballistic Ascent')
        h3, = axes.ravel()[i].plot(time_exp['descent'], x_exp['descent'][state], '-', color=colors['d'], label='Descent')

        if state == 'gam':
            axes.ravel()[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', steps=[1, 1.5, 3, 4.5, 6, 9, 10]))
            axes.ravel()[i].set_yticks(np.arange(-90, 91, 45))
        
        axes.ravel()[i].grid(True, alpha=0.2)

    axes[1, 0].set_xlabel('t (s)')
    axes[1, 1].set_xlabel('t (s)')
    
    plt.figlegend(handles=[h1, h2, h3], loc=legend_loc, ncol=legend_ncol)

    fig.tight_layout()
```

```python
def plot_trajectory(p, exp_out, legend_loc='center right'):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    r_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.r'),
             'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.r'),
             'descent': p.get_val('traj.descent.timeseries.r')}

    r_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.r'),
             'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.r'),
             'descent': exp_out.get_val('traj.descent.timeseries.r')}

    h_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.h'),
             'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.h'),
             'descent': p.get_val('traj.descent.timeseries.h')}

    h_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.h'),
             'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.h'),
             'descent': exp_out.get_val('traj.descent.timeseries.h')}

    axes.plot(r_imp['propelled_ascent'], h_imp['propelled_ascent'], 'o', color=colors['pa'])
    axes.plot(r_imp['ballistic_ascent'], h_imp['ballistic_ascent'], 'o', color=colors['ba'])
    axes.plot(r_imp['descent'], h_imp['descent'], 'o', color=colors['d'])

    h1, = axes.plot(r_exp['propelled_ascent'], h_exp['propelled_ascent'], '-', color=colors['pa'], label='Propelled Ascent')
    h2, = axes.plot(r_exp['ballistic_ascent'], h_exp['ballistic_ascent'], '-', color=colors['ba'], label='Ballistic Ascent')
    h3, = axes.plot(r_exp['descent'], h_exp['descent'], '-', color=colors['d'], label='Descent')

    axes.set_xlabel('r (m)')
    axes.set_ylabel('h (m)')
    axes.set_aspect('equal', 'box')
    plt.figlegend(handles=[h1, h2, h3], loc=legend_loc)
    axes.grid(alpha=0.2)

    fig.tight_layout()
```

## Optimizing for Height

```python
from dymos.examples.water_rocket.phases import new_water_rocket_trajectory, set_sane_initial_guesses

p = om.Problem(model=om.Group())

traj, phases = new_water_rocket_trajectory(objective='height')
traj = p.model.add_subsystem('traj', traj)

p.driver = om.pyOptSparseDriver(optimizer='IPOPT', print_results=False)
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['max_iter'] = 1000
p.driver.opt_settings['mu_strategy'] = 'adaptive'
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
p.driver.declare_coloring(tol=1.0E-12)

# Finish Problem Setup
p.model.linear_solver = om.DirectSolver()

p.setup()
set_sane_initial_guesses(phases)

dm.run_problem(p, run_driver=True, simulate=True)

summary = summarize_results(p)
for key, entry in summary.items():
    print(f'{key}: {entry.value:6.4f} {entry.unit}')
    
sol_out = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim_out = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')
```

### Maximum Height Solution: Propulsive Phase

```python
plot_propelled_ascent(sol_out, sim_out)
```

## Maximum Height Solution: Height vs. Range

Note that the equations of motion used here are singular in vertical flight, so the launch angle (the initial flight path angle) was limited to 85 degrees.

```python
plot_trajectory(sol_out, sim_out, legend_loc='center right')
```

## Maximum Height Solution: State History

```python
plot_states(sol_out, sim_out, legend_loc='lower center', legend_ncol=3)
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

assert_near_equal(summary['Launch angle'].value, 85, 0.01)
assert_near_equal(summary['Empty mass'].value, 0.144, 0.01)
assert_near_equal(summary['Water volume'].value, 0.98, 0.01)
assert_near_equal(summary['Maximum height'].value, 53.5, 0.01)
```

# Optimizing for Range

```python
# tags: output_scroll
from dymos.examples.water_rocket.phases import new_water_rocket_trajectory, set_sane_initial_guesses

p = om.Problem(model=om.Group())

traj, phases = new_water_rocket_trajectory(objective='range')
traj = p.model.add_subsystem('traj', traj)

p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['max_iter'] = 1000
p.driver.opt_settings['mu_strategy'] = 'adaptive'
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
p.driver.declare_coloring(tol=1.0E-12)

# Finish Problem Setup
p.model.linear_solver = om.DirectSolver()

p.setup()
set_sane_initial_guesses(phases)

dm.run_problem(p, run_driver=True, simulate=True)

summary = summarize_results(p)
for key, entry in summary.items():
    print(f'{key}: {entry.value:6.4f} {entry.unit}')

sol_out = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim_out = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')
```

## Maximum Range Solution: Propulsive Phase

```python
plot_propelled_ascent(sol_out, sim_out)
```

## Maximum Range Solution: Height vs. Range

```python
plot_trajectory(sol_out, sim_out, legend_loc='center')
```

## Maximum Range Solution: State History

```python
plot_states(sol_out, sim_out, legend_loc='lower center')
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

# Check results (tolerance is relative unless value is zero)
assert_near_equal(summary['Launch angle'].value, 46, 0.02)
assert_near_equal(summary['Flight angle at end of propulsion'].value, 38, 0.02)
assert_near_equal(summary['Empty mass'].value, 0.189, 1e-2)
assert_near_equal(summary['Water volume'].value, 1.026, 1e-2)
assert_near_equal(summary['Maximum range'].value, 85.11, 1e-2)
assert_near_equal(summary['Maximum height'].value, 23.08, 1e-2)
assert_near_equal(summary['Maximum velocity'].value, 41.31, 1e-2)
```

## References

```{bibliography}
:filter: docname in docnames
```
