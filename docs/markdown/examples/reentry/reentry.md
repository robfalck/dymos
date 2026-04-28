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

# Single-Phase Space Shuttle Reentry

The problem of the space shuttle reentering Earth's atmosphere is an
optimal control problem governed by six equations of motion and limited
by the aerodynamic heating rate. For a detailed layout of this problem
and other optimal control problems see Betts {cite}`betts2010practical`.
The governing equations of motion for this problem are:

\begin{align}
  \frac{dh}{dt} &= v \sin \gamma \\
  \frac{d\phi}{dt} &= \frac{v}{r} \cos \gamma \frac{\sin \psi}{\cos \theta} \\
  \frac{d\theta}{dt} &= \frac{v}{r} \cos \gamma \cos \psi  \\
  \frac{dv}{dt} &= - \frac{D}{m} - g \sin \gamma \\
  \frac{d\gamma}{dt} &= \frac{L}{mv} \cos \beta + \cos \gamma (\frac{v}{r} - \frac{g}{v}) \\
  \frac{d\psi}{dt} &= \frac{L \sin \beta}{mv \cos \gamma} + \frac{v}{r \cos \theta} \cos \gamma \sin \psi \sin \theta
\end{align}

where $v$ $[ft/s]$ is airspeed, $\gamma$ $[rad]$ is flight path angle,
$r$ $[ft]$ is distance from the center of the Earth, $\psi$ $[rad]$ is
azimuth, $\theta$ $[rad]$ is latitude, $D$ $[lb]$ is drag, $m$ $[sl]$ is
mass, $g$ $[\frac{ft}{s^2}]$ is the local gravitational acceleration, $L$
$[lb]$ is lift, $\beta$ $[rad]$ is bank angle, $h$ $[ft]$ is altitude,
and $\phi$ $[rad]$ is longitude. Mass is considered to be a constant for
this case, because the model spans the time from when the shuttle begins
reentry to the time right before the shuttle starts its engines. The
engines are not actually running at any time during the model, so there
is no thrust and thus no mass lost. The goal is to maximize the
crossrange (latitude) that the shuttle can cover before reaching the
final altitude, without exceding a maximum heat rate at the leading
edges. This heat rate is constrained by $q \leq 70$ where q
$[\frac{btu}{ft^2s}]$ is the heating rate.

The initial conditions are

\begin{align}
  h_0 &= 26000 \\
  v_0 &= 25600 \\
  \phi_0 &= 0 \\
  \gamma_0 &= -0.01745 \\
  \theta_0 &= 0 \\
  \psi_0 &= \frac{\pi}{2}
\end{align}

and the final conditions are

\begin{align}
  h_0 &= 80000 \\
  v_0 &= 2500 \\
  \gamma_0 &= -0.08727 \\
  \theta &= \rm{free} \\
  \psi &= \rm{free}
\end{align}

Notice that no final condition appears for $\phi$. This is because none
of the equations of motion actually depend on $\phi$, and as a result,
while $\phi$ exists in the dymos model (last code block below) as a
state variable, it does not exist as either an input or output in the
ode (ShuttleODE group, second to last code block below).

This model uses four explicit OpenMDAO components. The first component
computes the local atmospheric condition at the shuttle's altitude. The
second component computes the aerodynamic forces of lift and drag on the
shuttle. The third component is where the heating rate on the leading
edge of the shuttles wings is computed. The heating rate is given by
$q = q_a q_r$ where

\begin{align}
  q_a &= c_0 + c_1\alpha + c_2 \alpha^2 + c_3 \alpha^3
\end{align}

and

\begin{align}
  q_r &= 17700 \rho^.5 (.0001v)^{3.07}
\end{align}

where $c_0, c_1, c_2,$ and $c_3$ are constants, $\alpha$ $[deg]$ is the
angle of attack, $\rho$ $[\frac{sl}{ft^3}]$ is local atmospheric density,
and $v$ $[\frac{ft}{s}]$ is velocity. The final component is where the
equations of motion are implemented. These four components are put
together in the ShuttleODE group, which is the top level ode that the
dymos model sees.

## Component Models

Below is the code for the atmospheric component:

```python
import numpy as np
import openmdao.api as om


class Atmosphere(om.ExplicitComponent):
    """
    Defines the logarithmic atmosphere model for the shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 248, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('h', val=np.ones(nn), desc='altitude', units='ft')
        self.add_output('rho', val=np.ones(nn), desc='local density', units='slug/ft**3')
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('rho', 'h', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        h = inputs['h']
        h_r = 23800
        rho_0 = .002378
        outputs['rho'] = rho_0 * np.exp(-h / h_r)

    def compute_partials(self, inputs, partials):
        h = inputs['h']
        h_r = 23800
        rho_0 = .002378
        partials['rho', 'h'] = -1 / h_r * rho_0 * np.exp(-h / h_r)

```

Below is the code for the aerodynamics component:

```python
import openmdao.api as om


class Aerodynamics(om.ExplicitComponent):
    """
    Defines the aerodynamics for the shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 248, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', val=np.ones(nn), desc='angle of attack', units='deg')
        self.add_input('v', val=np.ones(nn), desc='velocity of shuttle', units='ft/s')
        self.add_input('rho', val=np.ones(nn), desc='local atmospheric density',
                       units='slug/ft**3')

        self.add_output('drag', val=np.ones(nn), desc='drag on shuttle', units='lb')
        self.add_output('lift', val=np.ones(nn), desc='lift on shuttle', units='lb')

        partial_range = np.arange(nn, dtype=int)

        self.declare_partials('drag', 'alpha', rows=partial_range, cols=partial_range)
        self.declare_partials('drag', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('drag', 'rho', rows=partial_range, cols=partial_range)
        self.declare_partials('lift', 'alpha', rows=partial_range, cols=partial_range)
        self.declare_partials('lift', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('lift', 'rho', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        a_0 = -.20704
        a_1 = .029244
        b_0 = .07854
        b_1 = -.61592e-2
        b_2 = .621408e-3
        S = 2690
        alpha = inputs['alpha']
        v = inputs['v']
        rho = inputs['rho']
        c_L = a_0 + a_1 * alpha
        c_D = b_0 + b_1 * alpha + b_2 * alpha ** 2

        outputs['drag'] = .5 * c_D * S * rho * v ** 2
        outputs['lift'] = .5 * c_L * S * rho * v ** 2

    def compute_partials(self, inputs, J):
        alpha = inputs['alpha']
        v = inputs['v']
        rho = inputs['rho']
        a_0 = -.20704
        a_1 = .029244
        b_0 = .07854
        b_1 = -.61592e-2
        b_2 = .621408e-3
        S = 2690
        c_L = a_0 + a_1 * alpha
        c_D = b_0 + b_1 * alpha + b_2 * alpha ** 2

        dD_dCD = .5 * S * rho * v ** 2
        dCD_dalpha = b_1 + 2 * alpha * b_2
        dL_dCL = dD_dCD
        dCL_dalpha = a_1

        J['drag', 'alpha'] = dD_dCD * dCD_dalpha
        J['drag', 'v'] = c_D * S * rho * v
        J['drag', 'rho'] = .5 * c_D * S * v ** 2
        J['lift', 'alpha'] = dL_dCL * dCL_dalpha
        J['lift', 'v'] = c_L * S * rho * v
        J['lift', 'rho'] = .5 * c_L * S * v ** 2

```

Below is the code for the heating component:

```python
import openmdao.api as om


class AerodynamicHeating(om.ExplicitComponent):
    """
    Defines the Aerodynamic heating equations for the shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 248, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('rho', val=np.ones(nn), desc='local density', units='slug/ft**3')
        self.add_input('v', val=np.ones(nn), desc='velocity of shuttle', units='ft/s')
        self.add_input('alpha', val=np.ones(nn), desc='angle of attack of shuttle',
                       units='deg')

        self.add_output('q', val=np.ones(nn),
                        desc='aerodynamic heating on leading edge of shuttle',
                        units='Btu/ft**2/s')

        partial_range = np.arange(nn)

        self.declare_partials('q', 'rho', rows=partial_range, cols=partial_range)
        self.declare_partials('q', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('q', 'alpha', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        rho = inputs['rho']
        v = inputs['v']
        alpha = inputs['alpha']
        c_0 = 1.0672181
        c_1 = -0.19213774e-1
        c_2 = 0.21286289e-3
        c_3 = -0.10117249e-5

        q_r = 17700.0 * np.sqrt(rho) * (0.0001 * v) ** 3.07
        q_a = c_0 + c_1 * alpha + c_2 * alpha ** 2 + c_3 * alpha ** 3

        outputs['q'] = q_r * q_a

    def compute_partials(self, inputs, partials):
        rho = inputs['rho']
        v = inputs['v']
        alpha = inputs['alpha']
        c_0 = 1.0672181
        c_1 = -.19213774e-1
        c_2 = .21286289e-3
        c_3 = -.10117249e-5

        sqrt_rho = np.sqrt(rho)

        q_r = 17700 * sqrt_rho * (.0001 * v) ** 3.07
        q_a = c_0 + c_1 * alpha + c_2 * alpha ** 2 + c_3 * alpha ** 3

        dqr_drho = 0.5 * q_r / rho
        dqr_dv = 17700 * sqrt_rho * 0.0001 * 3.07 * (0.0001 * v) ** 2.07

        dqa_dalpha = c_1 + 2 * c_2 * alpha + 3 * c_3 * alpha ** 2

        partials['q', 'rho'] = dqr_drho * q_a
        partials['q', 'v'] = dqr_dv * q_a
        partials['q', 'alpha'] = dqa_dalpha * q_r
```

Below is the code for the component containing the equations of motion:

```python
from openmdao.api import ExplicitComponent


class FlightDynamics(ExplicitComponent):
    """
    Defines the flight dynamics for the shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 247, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('beta', val=np.ones(nn), desc='bank angle', units='rad')
        self.add_input('gamma', val=np.ones(nn), desc='flight path angle', units='rad')
        self.add_input('h', val=np.ones(nn), desc='altitude of shuttle', units='ft')
        self.add_input('psi', val=np.ones(nn), desc='azimuthal angle', units='rad')
        self.add_input('theta', val=np.ones(nn), desc='latitude', units='rad')
        self.add_input('v', val=np.ones(nn), desc='velocity of shuttle', units='ft/s')
        self.add_input('lift', val=np.ones(nn), desc='lift on shuttle', units='lb')
        self.add_input('drag', val=np.ones(nn), desc='drag on shuttle', units='lb')

        self.add_output('hdot', val=np.ones(nn), desc='rate of change of altitude',
                        units='ft/s')
        self.add_output('gammadot', val=np.ones(nn),
                        desc='rate of change of flight path angle', units='rad/s')
        self.add_output('phidot', val=np.ones(nn), desc='rate of change of longitude',
                        units='rad/s')
        self.add_output('psidot', val=np.ones(nn), desc='rate of change of azimuthal angle',
                        units='rad/s')
        self.add_output('thetadot', val=np.ones(nn), desc='rate of change of latitude',
                        units='rad/s')
        self.add_output('vdot', val=np.ones(nn), desc='rate of change of velocity',
                        units='ft/s**2')

        partial_range = np.arange(nn, dtype=int)

        self.declare_partials('hdot', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('hdot', 'gamma', rows=partial_range, cols=partial_range)

        self.declare_partials('gammadot', 'lift', rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'h', rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'beta', rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'v', rows=partial_range, cols=partial_range)

        self.declare_partials('phidot', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'h', rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'psi', rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'theta', rows=partial_range, cols=partial_range)

        self.declare_partials('psidot', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'h', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'beta', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'theta', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'psi', rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'lift', rows=partial_range, cols=partial_range)

        self.declare_partials('thetadot', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'h', rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'psi', rows=partial_range, cols=partial_range)

        self.declare_partials('vdot', 'drag', rows=partial_range, cols=partial_range)
        self.declare_partials('vdot', 'gamma', rows=partial_range, cols=partial_range)
        self.declare_partials('vdot', 'h', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        v = inputs['v']
        gamma = inputs['gamma']
        theta = inputs['theta']
        lift = inputs['lift']
        drag = inputs['drag']
        h = inputs['h']
        beta = inputs['beta']
        psi = inputs['psi']
        g_0 = 32.174
        w = 203000
        R_e = 20902900
        mu = .14076539e17
        s_beta = np.sin(beta)
        c_beta = np.cos(beta)
        s_gamma = np.sin(gamma)
        c_gamma = np.cos(gamma)
        s_psi = np.sin(psi)
        c_psi = np.cos(psi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        r = R_e + h
        m = w / g_0
        g = mu / r ** 2

        outputs['hdot'] = v * s_gamma
        outputs['gammadot'] = lift / (m * v) * c_beta + c_gamma * (v / r - g / v)
        outputs['phidot'] = v / r * c_gamma * s_psi / c_theta
        outputs['psidot'] = lift * s_beta / (m * v * c_gamma) + \
            v * c_gamma * s_psi * s_theta / (r * c_theta)
        outputs['thetadot'] = c_gamma * c_psi * v / r
        outputs['vdot'] = -drag / m - g * s_gamma

    def compute_partials(self, inputs, J):
        v = inputs['v']
        gamma = inputs['gamma']
        theta = inputs['theta']
        lift = inputs['lift']
        h = inputs['h']
        beta = inputs['beta']
        psi = inputs['psi']
        g_0 = 32.174
        w = 203000
        R_e = 20902900
        mu = .14076539e17
        s_beta = np.sin(beta)
        c_beta = np.cos(beta)
        s_gamma = np.sin(gamma)
        c_gamma = np.cos(gamma)
        s_psi = np.sin(psi)
        c_psi = np.cos(psi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        r = R_e + h
        m = w / g_0
        g = mu / r ** 2

        J['hdot', 'v'] = s_gamma
        J['hdot', 'gamma'] = v * c_gamma

        J['gammadot', 'lift'] = c_beta / (m * v)
        J['gammadot', 'h'] = c_gamma * (-v / r ** 2 + 2 * mu / (r ** 3 * v))
        J['gammadot', 'beta'] = -lift / (m * v) * s_beta
        J['gammadot', 'gamma'] = -s_gamma * (v / r - g / v)
        J['gammadot', 'v'] = -lift / (m * v ** 2) * c_beta + c_gamma * (1 / r + g / v ** 2)

        J['phidot', 'v'] = c_gamma * s_psi / (c_theta * r)
        J['phidot', 'h'] = -v / r ** 2 * c_gamma * s_psi / c_theta
        J['phidot', 'gamma'] = -v / r * s_gamma * s_psi / c_theta
        J['phidot', 'psi'] = v / r * c_gamma * c_psi / c_theta
        J['phidot', 'theta'] = v / r * c_gamma * s_psi / (c_theta ** 2) * s_theta

        J['psidot', 'v'] = -lift * s_beta / (m * c_gamma * v ** 2) + \
            c_gamma * s_psi * s_theta / (r * c_theta)
        J['psidot', 'gamma'] = lift * s_beta / (m * v * c_gamma ** 2) * s_gamma - \
            v * s_gamma * s_psi * s_theta / (r * c_theta)
        J['psidot', 'h'] = -v * c_gamma * s_psi * s_theta / (c_theta * r ** 2)
        J['psidot', 'beta'] = lift * c_beta / (m * v * c_gamma)
        J['psidot', 'theta'] = v * c_gamma * s_psi / (r * c_theta ** 2)
        J['psidot', 'psi'] = v * c_gamma * c_psi * s_theta / (r * c_theta)
        J['psidot', 'lift'] = s_beta / (m * v * c_gamma)

        J['thetadot', 'v'] = c_gamma * c_psi / r
        J['thetadot', 'h'] = -v / r ** 2 * c_gamma * c_psi
        J['thetadot', 'gamma'] = -v / r * s_gamma * c_psi
        J['thetadot', 'psi'] = -v / r * c_gamma * s_psi

        J['vdot', 'h'] = 2 * s_gamma * mu / r ** 3
        J['vdot', 'drag'] = -1 / m
        J['vdot', 'gamma'] = -g * c_gamma

```

## Defining the ODE

Below is the code for the top level ode group that will be fed to dymos:

```python
from openmdao.api import Group


class ShuttleODE(Group):
    """
    The ODE for the Shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 248, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('atmosphere', subsys=Atmosphere(num_nodes=nn),
                           promotes_inputs=['h'], promotes_outputs=['rho'])
        self.add_subsystem('aerodynamics', subsys=Aerodynamics(num_nodes=nn),
                           promotes_inputs=['alpha', 'v', 'rho'],
                           promotes_outputs=['lift', 'drag'])
        self.add_subsystem('heating', subsys=AerodynamicHeating(num_nodes=nn),
                           promotes_inputs=['rho', 'v', 'alpha'], promotes_outputs=['q'])
        self.add_subsystem('eom', subsys=FlightDynamics(num_nodes=nn),
                           promotes_inputs=['beta', 'gamma', 'h', 'psi', 'theta', 'v', 'lift',
                                            'drag'],
                           promotes_outputs=['hdot', 'gammadot', 'phidot', 'psidot', 'thetadot',
                                             'vdot'])

```

## Building and running the problem

The following code is the dymos implementation of the model. As the code
shows, there are six states, two controls, and one constraint in the
model. The states are $h, v, \phi, \gamma, \theta,$ and $\psi$.
The two controls are $\alpha$ and $\beta$, and the constraint is $q$.

```python
# tags: active-ipynb, remove-input, remove-output
%matplotlib inline
```

```python
# tags: output_scroll
import openmdao.api as om
import dymos as dm

from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt

# Instantiate the problem, add the driver, and allow it to use coloring
p = om.Problem(model=om.Group())
p.driver = om.pyOptSparseDriver()
p.driver.declare_coloring()
p.driver.options['optimizer'] = 'SLSQP'

# Instantiate the trajectory and add a phase to it
traj = p.model.add_subsystem('traj', dm.Trajectory())
phase0 = traj.add_phase('phase0',
                        dm.Phase(ode_class=ShuttleODE,
                                 transcription=dm.Radau(num_segments=15, order=3)))

phase0.set_time_options(fix_initial=True, units='s', duration_ref=200)
phase0.add_state('h', fix_initial=True, fix_final=True, units='ft', rate_source='hdot',
                 lower=0, ref0=75000, ref=300000, defect_ref=1000)
phase0.add_state('gamma', fix_initial=True, fix_final=True, units='rad',
                 rate_source='gammadot',
                 lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
phase0.add_state('phi', fix_initial=True, fix_final=False, units='rad',
                 rate_source='phidot', lower=0, upper=89. * np.pi / 180)
phase0.add_state('psi', fix_initial=True, fix_final=False, units='rad',
                 rate_source='psidot', lower=0, upper=90. * np.pi / 180)
phase0.add_state('theta', fix_initial=True, fix_final=False, units='rad',
                 rate_source='thetadot',
                 lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
phase0.add_state('v', fix_initial=True, fix_final=True, units='ft/s',
                 rate_source='vdot', lower=0, ref0=2500, ref=25000)
phase0.add_control('alpha', units='rad', opt=True, lower=-np.pi / 2, upper=np.pi / 2, )
phase0.add_control('beta', units='rad', opt=True, lower=-89 * np.pi / 180, upper=1 * np.pi / 180, )

# The original implementation by Betts includes a heating rate path constraint.
# This will work with the SNOPT optimizer but SLSQP has difficulty converging the solution.
# phase0.add_path_constraint('q', lower=0, upper=70, ref=70)
phase0.add_timeseries_output('q', shape=(1,))

phase0.add_objective('theta', loc='final', ref=-0.01)

p.setup(check=True)

phase0.set_time_val(initial=0, duration=2000, units='s')

phase0.set_state_val('h', [260000, 80000], units='ft')
phase0.set_state_val('gamma', [-1, -5], units='deg')
phase0.set_state_val('phi', [0, 75], units='deg')
phase0.set_state_val('psi', [90, 10], units='deg')
phase0.set_state_val('theta', [0, 25], units='deg')
phase0.set_state_val('v', [25600, 2500], units='ft/s')

phase0.set_control_val('alpha', 17.4, units='deg')
phase0.set_control_val('beta', [-75, 0], units='deg')

# Run the driver
dm.run_problem(p, simulate=True)
```

```python
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.alpha',
               'time (s)', 'alpha (rad)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.beta',
               'time (s)', 'beta (rad)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.theta',
               'time (s)', 'theta (rad)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.q',
               'time (s)', 'q (Btu/ft**2/s)')], title='Reentry Solution', p_sol=sol,
             p_sim=sim)

plt.show()
```

```python
# tags: remove-input, remove-output
from openmdao.utils.assert_utils import assert_near_equal

# Check the validity of the solution
assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 2008.59,
                  tolerance=1e-3)
assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                  34.1412, tolerance=1e-3)
```

## References

```{bibliography}
:filter: docname in docnames
```
