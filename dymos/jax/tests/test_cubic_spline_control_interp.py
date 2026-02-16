"""Test JAX cubic_spline_control_interp function."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.explicit_shooting.cubic_spline_control_interp import cubic_spline_control_interp


class TestCubicSplineControlInterpJax(unittest.TestCase):
    """Test JAX cubic_spline_control_interp function."""

    def test_basic_interpolation(self):
        """Test basic interpolation functionality."""
        # Create simple input data
        num_input_nodes = 5
        u_input = jnp.linspace(0, 1, num_input_nodes).reshape(-1, 1)

        # Phase tau grid
        ptau_grid = jnp.linspace(-1, 1, 10)
        input_node_indices = jnp.array([0, 2, 5, 7, 9])  # Some subset

        # Evaluation points
        ptau = jnp.array([-0.5, 0.0, 0.5])
        t_duration = 2.0

        u, u_dot, u_ddot = cubic_spline_control_interp(
            u_input, ptau, ptau_grid, input_node_indices, t_duration
        )

        # Verify shapes
        self.assertEqual(u.shape, (3, 1))
        self.assertEqual(u_dot.shape, (3, 1))
        self.assertEqual(u_ddot.shape, (3, 1))

        # Values should be between min and max of input
        self.assertTrue(jnp.all(u >= jnp.min(u_input)))
        self.assertTrue(jnp.all(u <= jnp.max(u_input)))

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        num_input_nodes = 4
        u_input = jnp.ones((num_input_nodes, 1))
        ptau_grid = jnp.linspace(-1, 1, 8)
        input_node_indices = jnp.array([0, 2, 5, 7])
        ptau = jnp.array([0.0])
        t_duration = 1.0

        # JIT compile
        interp_jitted = jax.jit(cubic_spline_control_interp)

        # First call
        result1 = interp_jitted(u_input, ptau, ptau_grid, input_node_indices, t_duration)

        # Second call
        result2 = interp_jitted(u_input, ptau, ptau_grid, input_node_indices, t_duration)

        # Should be identical
        assert_allclose(result1[0], result2[0], rtol=1e-14)

    def test_constant_input(self):
        """Test that constant input gives constant output."""
        num_input_nodes = 5
        constant_value = 3.5
        u_input = jnp.ones((num_input_nodes, 1)) * constant_value

        ptau_grid = jnp.linspace(-1, 1, 10)
        input_node_indices = jnp.arange(0, 10, 2)  # Every other node
        ptau = jnp.linspace(-0.8, 0.8, 7)
        t_duration = 2.0

        u, u_dot, u_ddot = cubic_spline_control_interp(
            u_input, ptau, ptau_grid, input_node_indices, t_duration
        )

        # Constant input should give constant output
        assert_allclose(u, constant_value, rtol=1e-6,
                       err_msg="Constant input should give constant output")

    def test_vector_control(self):
        """Test with vector-valued controls."""
        num_input_nodes = 4
        control_shape = (3,)
        u_input = jnp.array(np.random.rand(num_input_nodes, *control_shape))

        ptau_grid = jnp.linspace(-1, 1, 8)
        input_node_indices = jnp.array([0, 2, 5, 7])
        ptau = jnp.array([-0.5, 0.0, 0.5])
        t_duration = 2.0

        u, u_dot, u_ddot = cubic_spline_control_interp(
            u_input, ptau, ptau_grid, input_node_indices, t_duration
        )

        # Verify shapes
        self.assertEqual(u.shape, (3, *control_shape))
        self.assertEqual(u_dot.shape, (3, *control_shape))
        self.assertEqual(u_ddot.shape, (3, *control_shape))

    def test_vmap_compatibility(self):
        """Test vectorization with jax.vmap."""
        num_input_nodes = 4
        batch_size = 3
        u_batch = jnp.array(np.random.rand(batch_size, num_input_nodes, 1))

        ptau_grid = jnp.linspace(-1, 1, 8)
        input_node_indices = jnp.array([0, 2, 5, 7])
        ptau = jnp.array([0.0])
        t_duration = 1.0

        # Vectorize over batch
        interp_batched = jax.vmap(
            lambda u: cubic_spline_control_interp(
                u, ptau, ptau_grid, input_node_indices, t_duration
            ),
            in_axes=0
        )

        results = interp_batched(u_batch)
        u_results, _, _ = results

        # Verify shape
        self.assertEqual(u_results.shape, (batch_size, 1, 1))


if __name__ == '__main__':
    unittest.main()
