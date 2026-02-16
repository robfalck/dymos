"""JAX function for Gauss-Lobatto node interleaving."""
import jax.numpy as jnp


def gauss_lobatto_interleave(disc_values, col_values, disc_indices, col_indices, num_nodes):
    """
    Interleave discretization and collocation node values into a contiguous array at all nodes.

    In Gauss-Lobatto transcription, some quantities are computed separately at state
    discretization nodes and collocation nodes. This function combines them into a
    single array with values at all nodes in the correct order.

    Parameters
    ----------
    disc_values : ArrayLike
        Values at state discretization nodes.
        Shape: (num_disc_nodes, *value_shape)
    col_values : ArrayLike
        Values at collocation nodes.
        Shape: (num_col_nodes, *value_shape)
    disc_indices : ArrayLike
        Indices indicating where discretization values should be placed in the output.
        Shape: (num_disc_nodes,)
    col_indices : ArrayLike
        Indices indicating where collocation values should be placed in the output.
        Shape: (num_col_nodes,)
    num_nodes : int
        Total number of nodes in the output array.

    Returns
    -------
    all_values : ArrayLike
        Values at all nodes, with discretization and collocation values interleaved.
        Shape: (num_nodes, *value_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The operation is essentially:
        all_values[disc_indices] = disc_values
        all_values[col_indices] = col_values
    """
    # Get the value shape from the discretization values
    value_shape = disc_values.shape[1:]

    # Initialize output array with zeros
    all_values = jnp.zeros((num_nodes,) + value_shape, dtype=disc_values.dtype)

    # Place discretization node values at their indices
    all_values = all_values.at[disc_indices].set(disc_values)

    # Place collocation node values at their indices
    all_values = all_values.at[col_indices].set(col_values)

    return all_values
