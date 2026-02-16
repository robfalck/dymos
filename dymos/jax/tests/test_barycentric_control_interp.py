"""Test JAX barycentric_control_interp function."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.explicit_shooting.barycentric_control_interp import (
    barycentric_control_interp, _compute_lagrange_basis
)


class TestBarycentricControlInterpJax(unittest.TestCase):
    """Test JAX barycentric_control_interp function."""

    def test_lagrange_basis_simple(self):
        """Test Lagrange basis computation for simple case."""
        # For 3 nodes at tau = [-1, 0, 1], evaluate basis at tau = 0
        taus = jnp.array([-1.0, 0.0, 1.0])
        tau = jnp.array([0.0])

        l, dl_dtau, d2l_dtau2 = _compute_lagrange_basis(tau, taus)

        # At tau = 0, the middle basis function should be 1, others 0
        expected_l = jnp.array([[0.0, 1.0, 0.0]])
        assert_allclose(l, expected_l, rtol=1e-10,
                       err_msg="Lagrange basis values incorrect")

    def test_basic_interpolation(self):
        """Test basic barycentric interpolation."""
        # Simple setup with 3 nodes
        n = 3
        u_input = jnp.array([[1.0], [2.0], [3.0]])

        stau = jnp.array([0.0])
        dstau_dt = 1.0

        input_to_disc_map = jnp.array([0, 1, 2])
        disc_node_indices = jnp.array([0, 1, 2])
        taus_seg = jnp.array([-1.0, 0.0, 1.0])

        # Simple identity weights
        w_b = jnp.eye(n)

        u, u_dot, u_ddot = barycentric_control_interp(
            u_input, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, taus_seg, w_b
        )

        # Verify shapes
        self.assertEqual(u.shape, (1, 1))
        self.assertEqual(u_dot.shape, (1, 1))
        self.assertEqual(u_ddot.shape, (1, 1))

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        n = 3
        u_input = jnp.ones((n, 1))
        stau = jnp.array([0.5])
        dstau_dt = 1.0

        input_to_disc_map = jnp.array([0, 1, 2])
        disc_node_indices = jnp.array([0, 1, 2])
        taus_seg = jnp.array([-1.0, 0.0, 1.0])
        w_b = jnp.eye(n)

        # JIT compile
        interp_jitted = jax.jit(barycentric_control_interp)

        # First call
        result1 = interp_jitted(
            u_input, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, taus_seg, w_b
        )

        # Second call
        result2 = interp_jitted(
            u_input, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, taus_seg, w_b
        )

        # Should be identical
        assert_allclose(result1[0], result2[0], rtol=1e-14)

    def test_constant_input(self):
        """Test that constant input gives constant output."""
        n = 4
        constant_value = 2.5
        u_input = jnp.ones((n, 1)) * constant_value

        stau = jnp.array([-0.5, 0.0, 0.5])
        dstau_dt = 1.0

        input_to_disc_map = jnp.arange(n)
        disc_node_indices = jnp.arange(n)
        taus_seg = jnp.linspace(-1, 1, n)
        w_b = jnp.eye(n)

        u, u_dot, u_ddot = barycentric_control_interp(
            u_input, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, taus_seg, w_b
        )

        # Constant input should give constant output
        assert_allclose(u, constant_value, rtol=1e-5,
                       err_msg="Constant input should give constant output")

    def test_vector_control(self):
        """Test with vector-valued controls."""
        n = 3
        control_shape = (2,)
        u_input = jnp.array(np.random.rand(n, *control_shape))

        stau = jnp.array([0.0, 0.5])
        dstau_dt = 2.0

        input_to_disc_map = jnp.arange(n)
        disc_node_indices = jnp.arange(n)
        taus_seg = jnp.array([-1.0, 0.0, 1.0])
        w_b = jnp.eye(n)

        u, u_dot, u_ddot = barycentric_control_interp(
            u_input, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, taus_seg, w_b
        )

        # Verify shapes
        self.assertEqual(u.shape, (2, *control_shape))
        self.assertEqual(u_dot.shape, (2, *control_shape))
        self.assertEqual(u_ddot.shape, (2, *control_shape))

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff works."""
        n = 3
        u_input = jnp.array([[1.0], [2.0], [1.5]])
        stau = jnp.array([0.3])
        dstau_dt = 1.0

        input_to_disc_map = jnp.arange(n)
        disc_node_indices = jnp.arange(n)
        taus_seg = jnp.array([-1.0, 0.0, 1.0])
        w_b = jnp.eye(n)

        # Define objective
        def objective(u_nodes):
            u, _, _ = barycentric_control_interp(
                u_nodes, stau, dstau_dt, input_to_disc_map,
                disc_node_indices, taus_seg, w_b
            )
            return jnp.sum(u**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(u_input)

        # Verify gradient is not all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")


if __name__ == '__main__':
    unittest.main()
