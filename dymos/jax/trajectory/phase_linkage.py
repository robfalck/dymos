"""JAX function for phase linkage computation."""
import jax.numpy as jnp


def phase_linkage(var_a, var_b, loc_a='final', loc_b='initial',
                 mult_a=1.0, mult_b=-1.0, conv_a=1.0, conv_b=1.0,
                 offset_a=0.0, offset_b=0.0):
    """
    Compute phase linkage constraint value between two phases.

    This function computes a linkage equation that can be constrained to enforce
    continuity or other relationships between variables in two different phases.
    Typically used to link the final state of one phase to the initial state of another.

    Parameters
    ----------
    var_a : ArrayLike
        Variable values from phase A. Should include both initial and final values.
        Shape: (2, *var_shape) where index 0 is initial, index 1 is final
    var_b : ArrayLike
        Variable values from phase B. Should include both initial and final values.
        Shape: (2, *var_shape) where index 0 is initial, index 1 is final
    loc_a : str, optional
        Location in phase A to use: 'initial' or 'final'. Default: 'final'
    loc_b : str, optional
        Location in phase B to use: 'initial' or 'final'. Default: 'initial'
    mult_a : float or ArrayLike, optional
        Multiplier for variable from phase A. Default: 1.0
    mult_b : float or ArrayLike, optional
        Multiplier for variable from phase B. Default: -1.0
    conv_a : float, optional
        Unit conversion factor for phase A variable. Default: 1.0
    conv_b : float, optional
        Unit conversion factor for phase B variable. Default: 1.0
    offset_a : float, optional
        Offset for phase A variable (applied before conversion). Default: 0.0
    offset_b : float, optional
        Offset for phase B variable (applied before conversion). Default: 0.0

    Returns
    -------
    linkage : ArrayLike
        Linkage constraint value.
        Shape: (*var_shape,)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller (via conv_a and conv_b).

    The linkage equation is:
        linkage = mult_a * (var_a[loc_a] + offset_a) * conv_a +
                  mult_b * (var_b[loc_b] + offset_b) * conv_b

    With default values (mult_a=1, mult_b=-1, no offsets/conversions), this gives:
        linkage = var_a[final] - var_b[initial]

    Constraining this to zero enforces: var_a[final] = var_b[initial]

    Examples
    --------
    >>> # Link final state of phase A to initial state of phase B
    >>> var_a = jnp.array([[1.0], [5.0]])  # [initial, final]
    >>> var_b = jnp.array([[5.0], [10.0]])  # [initial, final]
    >>> linkage = phase_linkage(var_a, var_b)
    >>> # linkage = 5.0 - 5.0 = 0.0 (continuous)

    >>> # Link with unit conversion (e.g., meters to feet)
    >>> var_a_m = jnp.array([[0.0], [1.0]])  # meters
    >>> var_b_ft = jnp.array([[3.28084], [6.56168]])  # feet
    >>> linkage = phase_linkage(var_a_m, var_b_ft, conv_a=3.28084, conv_b=1.0)
    >>> # linkage ≈ 0.0 (1 meter ≈ 3.28 feet)
    """
    # Extract values at specified locations
    idx_a = 0 if loc_a == 'initial' else -1
    idx_b = 0 if loc_b == 'initial' else -1

    val_a = var_a[idx_a, ...]
    val_b = var_b[idx_b, ...]

    # Apply offsets and unit conversions
    a_contribution = mult_a * (val_a + offset_a) * conv_a
    b_contribution = mult_b * (val_b + offset_b) * conv_b

    # Compute linkage
    linkage = a_contribution + b_contribution

    return linkage
