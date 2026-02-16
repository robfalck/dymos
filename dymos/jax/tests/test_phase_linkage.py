"""Test JAX phase_linkage function."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.trajectory.phase_linkage import phase_linkage


class TestPhaseLinkageJax(unittest.TestCase):
    """Test JAX phase_linkage function."""

    def test_basic_linkage(self):
        """Test basic phase linkage."""
        # Phase A: initial=0, final=5
        var_a = jnp.array([[0.0], [5.0]])
        # Phase B: initial=5, final=10
        var_b = jnp.array([[5.0], [10.0]])

        # Default linkage: final of A to initial of B
        linkage = phase_linkage(var_a, var_b)

        # Expected: 5.0 - 5.0 = 0.0
        assert_allclose(linkage, 0.0, atol=1e-14,
                       err_msg="Basic linkage should be zero for continuous phases")

    def test_discontinuous_linkage(self):
        """Test linkage with discontinuity."""
        var_a = jnp.array([[0.0], [5.0]])
        var_b = jnp.array([[6.0], [10.0]])  # Discontinuity

        linkage = phase_linkage(var_a, var_b)

        # Expected: 5.0 - 6.0 = -1.0
        assert_allclose(linkage, -1.0, rtol=1e-14,
                       err_msg="Linkage defect incorrect")

    def test_custom_multipliers(self):
        """Test with custom multipliers."""
        var_a = jnp.array([[0.0], [2.0]])
        var_b = jnp.array([[3.0], [5.0]])

        # Custom linkage: 2*a_final + 3*b_initial
        linkage = phase_linkage(var_a, var_b, mult_a=2.0, mult_b=3.0)

        # Expected: 2*2.0 + 3*3.0 = 4 + 9 = 13
        assert_allclose(linkage, 13.0, rtol=1e-14,
                       err_msg="Custom multipliers incorrect")

    def test_unit_conversion(self):
        """Test with unit conversion."""
        # Phase A in meters
        var_a_m = jnp.array([[0.0], [1.0]])
        # Phase B in feet
        var_b_ft = jnp.array([[3.28084], [6.56168]])

        # Convert meters to feet: 1 m = 3.28084 ft
        linkage = phase_linkage(var_a_m, var_b_ft, conv_a=3.28084, conv_b=1.0)

        # Expected: 1.0*3.28084 - 3.28084 ≈ 0
        assert_allclose(linkage, 0.0, atol=1e-5,
                       err_msg="Unit conversion linkage incorrect")

    def test_different_locations(self):
        """Test linking different locations."""
        var_a = jnp.array([[1.0], [2.0]])
        var_b = jnp.array([[3.0], [4.0]])

        # Link initial of A to final of B
        linkage = phase_linkage(var_a, var_b, loc_a='initial', loc_b='final')

        # Expected: 1.0 - 4.0 = -3.0
        assert_allclose(linkage, -3.0, rtol=1e-14,
                       err_msg="Different locations linkage incorrect")

    def test_vector_variable(self):
        """Test with vector-valued variables."""
        var_a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var_b = jnp.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        linkage = phase_linkage(var_a, var_b)

        # Expected: [4, 5, 6] - [4, 5, 6] = [0, 0, 0]
        assert_allclose(linkage, jnp.zeros(3), atol=1e-14,
                       err_msg="Vector variable linkage incorrect")

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        var_a = jnp.array([[0.0], [1.0]])
        var_b = jnp.array([[1.0], [2.0]])

        # JIT compile
        linkage_jitted = jax.jit(phase_linkage)

        # First call
        result1 = linkage_jitted(var_a, var_b)

        # Second call
        result2 = linkage_jitted(var_a, var_b)

        # Should be identical
        assert_allclose(result1, result2, rtol=1e-14)

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff works."""
        var_a = jnp.array([[0.0], [2.0]])
        var_b = jnp.array([[3.0], [5.0]])

        # Define objective
        def objective(a):
            return jnp.sum(phase_linkage(a, var_b)**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(var_a)

        # Verify gradient is not all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")


if __name__ == '__main__':
    unittest.main()
