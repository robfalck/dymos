"""Test JAX timeseries_interp functions."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.common.timeseries_output import (
    timeseries_interp, timeseries_value_interp, timeseries_rate_interp
)


class TestTimeseriesOutputJax(unittest.TestCase):
    """Test JAX timeseries interpolation functions."""

    def test_no_interpolation(self):
        """Test pass-through when no interpolation needed."""
        values = jnp.array([[1.0], [2.0], [3.0]])

        output = timeseries_interp(values, L=None, is_rate=False)

        # Should pass through unchanged
        assert_allclose(output, values, rtol=1e-14,
                       err_msg="Pass-through should return input unchanged")

    def test_basic_interpolation(self):
        """Test basic Lagrange interpolation."""
        # Input nodes at tau = [-1, 0, 1]
        values = jnp.array([[1.0], [2.0], [3.0]])

        # Interpolate to mid-points: tau = [-0.5, 0.5]
        # Create simple interpolation matrix
        # At tau=-0.5, linear interp between nodes 0 and 1: 0.75*val[0] + 0.25*val[1]
        # At tau=0.5, linear interp between nodes 1 and 2: 0.25*val[1] + 0.75*val[2]
        L = jnp.array([
            [0.75, 0.25, 0.0],
            [0.0, 0.25, 0.75]
        ])

        output = timeseries_interp(values, L=L)

        # Expected values
        expected = jnp.array([
            [0.75*1.0 + 0.25*2.0],  # 1.25
            [0.25*2.0 + 0.75*3.0]   # 2.75
        ])

        assert_allclose(output, expected, rtol=1e-14,
                       err_msg="Basic interpolation incorrect")

    def test_rate_interpolation(self):
        """Test rate (derivative) computation."""
        # Values at 3 nodes
        values = jnp.array([[0.0], [1.0], [4.0]])

        # Simple differentiation matrix (finite differences)
        # d/dtau at midpoints
        D = jnp.array([
            [-1.0, 1.0, 0.0],   # (val[1] - val[0])
            [0.0, -1.0, 1.0]    # (val[2] - val[1])
        ])

        dt_dstau = jnp.array([2.0, 2.0])

        output = timeseries_interp(values, D=D, dt_dstau=dt_dstau, is_rate=True)

        # Expected: D @ values / dt_dstau
        # Row 0: (1.0 - 0.0) / 2.0 = 0.5
        # Row 1: (4.0 - 1.0) / 2.0 = 1.5
        expected = jnp.array([[0.5], [1.5]])

        assert_allclose(output, expected, rtol=1e-14,
                       err_msg="Rate interpolation incorrect")

    def test_vector_variable(self):
        """Test with vector-valued variables."""
        # 3 input nodes, shape (2,)
        values = jnp.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])

        # Simple interpolation: average of first two nodes
        L = jnp.array([[0.5, 0.5, 0.0]])

        output = timeseries_interp(values, L=L)

        # Expected: 0.5*[1,2] + 0.5*[3,4] = [2,3]
        expected = jnp.array([[2.0, 3.0]])

        assert_allclose(output, expected, rtol=1e-14,
                       err_msg="Vector variable interpolation incorrect")

    def test_convenience_functions(self):
        """Test convenience functions."""
        values = jnp.array([[1.0], [2.0], [3.0]])
        L = jnp.eye(3)
        D = jnp.eye(3)
        dt_dstau = jnp.ones(3)

        # Value interpolation
        val_output = timeseries_value_interp(values, L)
        expected_val = timeseries_interp(values, L=L, is_rate=False)
        assert_allclose(val_output, expected_val, rtol=1e-14)

        # Rate interpolation
        rate_output = timeseries_rate_interp(values, D, dt_dstau)
        expected_rate = timeseries_interp(values, D=D, dt_dstau=dt_dstau, is_rate=True)
        assert_allclose(rate_output, expected_rate, rtol=1e-14)

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        values = jnp.ones((3, 1))
        L = jnp.eye(3)

        # JIT compile
        interp_jitted = jax.jit(timeseries_interp)

        # First call
        result1 = interp_jitted(values, L)

        # Second call
        result2 = interp_jitted(values, L)

        # Should be identical
        assert_allclose(result1, result2, rtol=1e-14)

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff works."""
        values = jnp.array([[1.0], [2.0], [3.0]])
        L = jnp.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])

        # Define objective
        def objective(vals):
            output = timeseries_interp(vals, L)
            return jnp.sum(output**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(values)

        # Verify gradient is not all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

    def test_vmap_compatibility(self):
        """Test vectorization with jax.vmap."""
        batch_size = 2
        values_batch = jnp.array(np.random.rand(batch_size, 3, 1))
        L = jnp.eye(3)

        # Vectorize over batch
        interp_batched = jax.vmap(
            lambda v: timeseries_interp(v, L),
            in_axes=0
        )

        results = interp_batched(values_batch)

        # Verify shape
        self.assertEqual(results.shape, (batch_size, 3, 1))


if __name__ == '__main__':
    unittest.main()
