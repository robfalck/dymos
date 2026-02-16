"""JAX implementation of the Brachistochrone ODE.

The brachistochrone problem seeks the path of fastest descent for a bead
sliding along a wire under gravity between two points.

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


def brachistochrone_ode(x, y, v, theta, g=9.80665):
    """
    Compute state derivatives for the brachistochrone problem.

    The brachistochrone problem finds the curve of fastest descent between
    two points under gravity. A bead slides along a frictionless wire, and
    we seek to minimize transit time.

    Parameters
    ----------
    x : ArrayLike
        Horizontal position at each node.
        Shape: (num_nodes,) or scalar
    y : ArrayLike
        Vertical position at each node (positive downward).
        Shape: (num_nodes,) or scalar
    v : ArrayLike
        Velocity magnitude at each node.
        Shape: (num_nodes,) or scalar
    theta : ArrayLike
        Angle of wire from horizontal at each node (radians).
        Shape: (num_nodes,) or scalar
        Positive theta means descending to the right.
    g : float, optional
        Gravitational acceleration. Default: 9.80665 m/s^2

    Returns
    -------
    x_dot : ArrayLike
        Time derivative of horizontal position (dx/dt).
        Shape: same as inputs
    y_dot : ArrayLike
        Time derivative of vertical position (dy/dt).
        Shape: same as inputs
    v_dot : ArrayLike
        Time derivative of velocity (dv/dt).
        Shape: same as inputs

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not enforced - caller must ensure consistency.

    The equations of motion are:
        dx/dt = v * sin(theta)
        dy/dt = v * cos(theta)
        dv/dt = g * cos(theta)

    These come from the constraint that the bead moves along the wire at angle
    theta, with gravitational acceleration g acting downward.

    Examples
    --------
    >>> # Single evaluation
    >>> x, y, v = 0.0, 0.0, 0.0
    >>> theta = jnp.pi / 4  # 45 degrees
    >>> g = 9.80665
    >>> x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta, g)

    >>> # Vectorized over nodes
    >>> import jax.numpy as jnp
    >>> num_nodes = 10
    >>> x = jnp.linspace(0, 10, num_nodes)
    >>> y = jnp.linspace(0, -10, num_nodes)
    >>> v = jnp.linspace(0, 14, num_nodes)
    >>> theta = jnp.ones(num_nodes) * jnp.pi / 4
    >>> x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta)
    """
    # Compute state derivatives
    # Horizontal velocity component
    x_dot = v * jnp.sin(theta)

    # Vertical velocity component (positive downward)
    y_dot = v * jnp.cos(theta)

    # Acceleration along the wire due to gravity
    v_dot = g * jnp.cos(theta)

    return x_dot, y_dot, v_dot


def brachistochrone_ode_vectorized(states, controls, params=None):
    """
    Vectorized version of brachistochrone ODE for use in trajectory optimization.

    This function provides a convenient interface that accepts states and controls
    as separate arguments, useful for integration with optimization frameworks.

    Parameters
    ----------
    states : dict or tuple
        State variables. If dict, keys are 'x', 'y', 'v'.
        If tuple, order is (x, y, v).
        Each value has shape: (num_nodes,)
    controls : dict or tuple
        Control variables. If dict, key is 'theta'.
        If tuple, single element is (theta,).
        Shape: (num_nodes,)
    params : dict, optional
        Parameters. If dict, key is 'g' for gravity.
        If None, uses default g=9.80665 m/s^2.

    Returns
    -------
    state_rates : tuple
        Time derivatives (x_dot, y_dot, v_dot).
        Each has shape: (num_nodes,)

    Examples
    --------
    >>> # Using dictionaries
    >>> states = {'x': x_vals, 'y': y_vals, 'v': v_vals}
    >>> controls = {'theta': theta_vals}
    >>> params = {'g': 9.80665}
    >>> x_dot, y_dot, v_dot = brachistochrone_ode_vectorized(states, controls, params)

    >>> # Using tuples
    >>> states = (x_vals, y_vals, v_vals)
    >>> controls = (theta_vals,)
    >>> x_dot, y_dot, v_dot = brachistochrone_ode_vectorized(states, controls)
    """
    # Extract states
    if isinstance(states, dict):
        x = states['x']
        y = states['y']
        v = states['v']
    else:
        x, y, v = states

    # Extract controls
    if isinstance(controls, dict):
        theta = controls['theta']
    else:
        theta = controls[0] if isinstance(controls, (tuple, list)) else controls

    # Extract parameters
    if params is not None and isinstance(params, dict):
        g = params.get('g', 9.80665)
    else:
        g = 9.80665

    return brachistochrone_ode(x, y, v, theta, g)
