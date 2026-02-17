"""Dict I/O version of the brachistochrone ODE for spec-based systems.

This module provides the brachistochrone ODE with dict-based inputs/outputs,
suitable for use with the spec-based backend (create_jax_radau_phase).

States:
    x : horizontal position
    y : vertical position (positive downward)
    v : velocity magnitude

Control:
    theta : angle of the wire from horizontal (radians)

Parameter:
    g : gravitational acceleration (default 9.80665 m/s^2)
"""
import jax.numpy as jnp


def brachistochrone_ode_dict(states, controls, parameters):
    """Brachistochrone ODE with dict-based I/O.

    Parameters
    ----------
    states : dict
        State values at each collocation node:
        - 'x': array(num_nodes,) - horizontal position
        - 'y': array(num_nodes,) - vertical position (positive downward)
        - 'v': array(num_nodes,) - velocity magnitude
    controls : dict
        Control values at each collocation node:
        - 'theta': array(num_nodes,) - wire angle from horizontal (radians)
    parameters : dict
        Static parameters:
        - 'g': float - gravitational acceleration (m/s^2)

    Returns
    -------
    rates : dict
        State time derivatives:
        - 'x_dot': array(num_nodes,) - dx/dt
        - 'y_dot': array(num_nodes,) - dy/dt
        - 'v_dot': array(num_nodes,) - dv/dt

    Notes
    -----
    This is a pure JAX function suitable for use with jax.jit, jax.grad, etc.

    The equations of motion are:
        dx/dt = v * sin(theta)
        dy/dt = v * cos(theta)
        dv/dt = g * cos(theta)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> states = {'x': jnp.zeros(5), 'y': jnp.zeros(5), 'v': jnp.ones(5)}
    >>> controls = {'theta': jnp.ones(5) * jnp.pi / 4}
    >>> parameters = {'g': 9.80665}
    >>> rates = brachistochrone_ode_dict(states, controls, parameters)
    >>> # rates['x_dot'], rates['y_dot'], rates['v_dot'] each have shape (5,)
    """
    v = states['v']
    theta = controls['theta']
    g = parameters['g']

    x_dot = v * jnp.sin(theta)
    y_dot = v * jnp.cos(theta)
    v_dot = g * jnp.cos(theta)

    return {
        'x_dot': x_dot,
        'y_dot': y_dot,
        'v_dot': v_dot,
    }
