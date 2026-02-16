"""JAX function for timeseries output interpolation."""
import jax.numpy as jnp


def timeseries_interp(values, L=None, D=None, dt_dstau=None, is_rate=False):
    """
    Interpolate values from input grid to output grid using Lagrange interpolation.

    This function interpolates variable values from one grid to another, optionally
    computing time derivatives. Used for creating timeseries outputs at arbitrary
    time points from values defined at collocation/discretization nodes.

    Parameters
    ----------
    values : ArrayLike
        Values at input grid nodes.
        Shape: (num_input_nodes, *var_shape)
    L : ArrayLike, optional
        Lagrange interpolation matrix mapping input nodes to output nodes.
        Shape: (num_output_nodes, num_input_nodes)
        If None, assumes no interpolation needed (output = input).
    D : ArrayLike, optional
        Lagrange differentiation matrix for computing derivatives.
        Shape: (num_output_nodes, num_input_nodes)
        Required if is_rate=True.
    dt_dstau : ArrayLike, optional
        Ratio of time derivative to segment tau derivative at output nodes.
        Shape: (num_output_nodes,)
        Required if is_rate=True.
    is_rate : bool, optional
        If True, compute time derivative of the variable. Default: False

    Returns
    -------
    output : ArrayLike
        Interpolated values (or derivatives) at output grid nodes.
        Shape: (num_output_nodes, *var_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    For value interpolation:
        output = L @ values

    For rate (derivative) interpolation:
        output = (D @ values) / dt_dstau

    where dt_dstau is broadcast appropriately across variable dimensions.

    The matrices L and D are typically constructed using Lagrange polynomial
    interpolation based on the input and output grid node locations.
    """
    # If no interpolation matrix provided, pass through directly
    if L is None and not is_rate:
        return values

    # Flatten variable dimensions for matrix multiplication
    num_input_nodes = values.shape[0]
    var_shape = values.shape[1:]

    if len(var_shape) > 0:
        # Reshape to (num_input_nodes, size)
        values_flat = jnp.reshape(values, (num_input_nodes, -1))
    else:
        values_flat = values

    # Perform interpolation or differentiation
    if is_rate:
        # Compute derivative using differentiation matrix
        if D is None:
            raise ValueError("Differentiation matrix D required when is_rate=True")
        if dt_dstau is None:
            raise ValueError("dt_dstau required when is_rate=True")

        # D @ values gives derivative wrt segment tau
        output_flat = D @ values_flat

        # Scale by dt_dstau to get derivative wrt time
        # dt_dstau has shape (num_output_nodes,), need to broadcast
        dt_dstau_expanded = dt_dstau[:, jnp.newaxis]
        output_flat = output_flat / dt_dstau_expanded

    else:
        # Standard interpolation using interpolation matrix
        if L is None:
            # No interpolation needed
            output_flat = values_flat
        else:
            output_flat = L @ values_flat

    # Reshape back to variable shape
    num_output_nodes = output_flat.shape[0]
    if len(var_shape) > 0:
        output = jnp.reshape(output_flat, (num_output_nodes,) + var_shape)
    else:
        output = output_flat

    return output


def timeseries_value_interp(values, L=None):
    """
    Interpolate variable values to output grid.

    Convenience function for value interpolation (is_rate=False).

    Parameters
    ----------
    values : ArrayLike
        Values at input grid nodes.
        Shape: (num_input_nodes, *var_shape)
    L : ArrayLike, optional
        Lagrange interpolation matrix.
        Shape: (num_output_nodes, num_input_nodes)

    Returns
    -------
    output : ArrayLike
        Interpolated values at output grid nodes.
        Shape: (num_output_nodes, *var_shape)

    Notes
    -----
    This is equivalent to:
        timeseries_interp(values, L=L, is_rate=False)
    """
    return timeseries_interp(values, L=L, is_rate=False)


def timeseries_rate_interp(values, D, dt_dstau):
    """
    Interpolate variable time derivatives to output grid.

    Convenience function for rate interpolation (is_rate=True).

    Parameters
    ----------
    values : ArrayLike
        Values at input grid nodes.
        Shape: (num_input_nodes, *var_shape)
    D : ArrayLike
        Lagrange differentiation matrix.
        Shape: (num_output_nodes, num_input_nodes)
    dt_dstau : ArrayLike
        Ratio dt/dstau at output nodes.
        Shape: (num_output_nodes,)

    Returns
    -------
    output : ArrayLike
        Time derivative at output grid nodes.
        Shape: (num_output_nodes, *var_shape)

    Notes
    -----
    This is equivalent to:
        timeseries_interp(values, D=D, dt_dstau=dt_dstau, is_rate=True)
    """
    return timeseries_interp(values, D=D, dt_dstau=dt_dstau, is_rate=True)
