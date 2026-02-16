"""Test JAX vandermonde_control_interp function."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.explicit_shooting.vandermonde_control_interp import vandermonde_control_interp


class TestVandermondeControlInterpJax(unittest.TestCase):
    """Test JAX vandermonde_control_interp function."""

    def test_basic_interpolation(self):
        """Test basic Vandermonde interpolation with known polynomial."""
        # For a quadratic polynomial u(tau) = a0 + a1*tau + a2*tau^2
        # If we have 3 nodes and know the polynomial coefficients,
        # we can verify the interpolation works correctly.

        # Use 3 nodes (quadratic polynomial)
        order = 3
        disc_node_tau = jnp.array([-1.0, 0.0, 1.0])

        # Create inverse Vandermonde matrix for these nodes
        V_disc = jnp.vander(disc_node_tau, N=order, increasing=True)
        V_hat_inv = jnp.linalg.inv(V_disc)

        # Test polynomial: u(tau) = 1 + 2*tau + 3*tau^2
        # At nodes: u(-1) = 1-2+3=2, u(0) = 1, u(1) = 1+2+3=6
        u_at_nodes = jnp.array([[2.0], [1.0], [6.0]])

        # Simple identity mapping (input nodes = disc nodes)
        input_to_disc_map = jnp.array([0, 1, 2])
        disc_node_indices = jnp.array([0, 1, 2])

        # Evaluate at tau = 0.5
        # Expected: u(0.5) = 1 + 2*0.5 + 3*0.25 = 1 + 1 + 0.75 = 2.75
        stau = jnp.array([0.5])
        dstau_dt = 1.0

        u, u_dot, u_ddot = vandermonde_control_interp(
            u_at_nodes, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, V_hat_inv
        )

        # Check interpolated value
        expected_u = 1 + 2*0.5 + 3*0.5**2
        assert_allclose(u[0, 0], expected_u, rtol=1e-10,
                       err_msg="Interpolated value incorrect")

        # Check first derivative: du/dtau = 2 + 6*tau
        # At tau=0.5: du/dtau = 2 + 3 = 5
        expected_du = 2 + 6*0.5
        assert_allclose(u_dot[0, 0], expected_du, rtol=1e-10,
                       err_msg="First derivative incorrect")

        # Check second derivative: d2u/dtau2 = 6
        expected_d2u = 6.0
        assert_allclose(u_ddot[0, 0], expected_d2u, rtol=1e-10,
                       err_msg="Second derivative incorrect")

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        order = 3
        disc_node_tau = jnp.array([-1.0, 0.0, 1.0])
        V_disc = jnp.vander(disc_node_tau, N=order, increasing=True)
        V_hat_inv = jnp.linalg.inv(V_disc)

        u_at_nodes = jnp.ones((3, 1))
        input_to_disc_map = jnp.array([0, 1, 2])
        disc_node_indices = jnp.array([0, 1, 2])
        stau = jnp.array([0.0, 0.5])
        dstau_dt = 1.0

        # JIT compile
        interp_jitted = jax.jit(vandermonde_control_interp)

        # First call
        result1 = interp_jitted(
            u_at_nodes, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, V_hat_inv
        )

        # Second call
        result2 = interp_jitted(
            u_at_nodes, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, V_hat_inv
        )

        # Should be identical
        assert_allclose(result1[0], result2[0], rtol=1e-14)
        assert_allclose(result1[1], result2[1], rtol=1e-14)
        assert_allclose(result1[2], result2[2], rtol=1e-14)

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff works on the interpolation function."""
        order = 3
        disc_node_tau = jnp.array([-1.0, 0.0, 1.0])
        V_disc = jnp.vander(disc_node_tau, N=order, increasing=True)
        V_hat_inv = jnp.linalg.inv(V_disc)

        u_at_nodes = jnp.array([[1.0], [2.0], [3.0]])
        input_to_disc_map = jnp.array([0, 1, 2])
        disc_node_indices = jnp.array([0, 1, 2])
        stau = jnp.array([0.3])
        dstau_dt = 1.0

        # Define objective as sum of interpolated values
        def objective(u_nodes):
            u, _, _ = vandermonde_control_interp(
                u_nodes, stau, dstau_dt, input_to_disc_map,
                disc_node_indices, V_hat_inv
            )
            return jnp.sum(u**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(u_at_nodes)

        # Verify gradient is not all zeros (shows autodiff is working)
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

    def test_vmap_compatibility(self):
        """Test vectorization with jax.vmap."""
        order = 3
        disc_node_tau = jnp.array([-1.0, 0.0, 1.0])
        V_disc = jnp.vander(disc_node_tau, N=order, increasing=True)
        V_hat_inv = jnp.linalg.inv(V_disc)

        # Batch of control inputs
        batch_size = 4
        u_batch = jnp.array(np.random.rand(batch_size, 3, 1))

        input_to_disc_map = jnp.array([0, 1, 2])
        disc_node_indices = jnp.array([0, 1, 2])
        stau = jnp.array([0.5])
        dstau_dt = 1.0

        # Vectorize over batch
        interp_batched = jax.vmap(
            lambda u: vandermonde_control_interp(
                u, stau, dstau_dt, input_to_disc_map,
                disc_node_indices, V_hat_inv
            ),
            in_axes=0
        )

        results = interp_batched(u_batch)
        u_results, u_dot_results, u_ddot_results = results

        # Verify shapes
        self.assertEqual(u_results.shape, (batch_size, 1, 1))
        self.assertEqual(u_dot_results.shape, (batch_size, 1, 1))
        self.assertEqual(u_ddot_results.shape, (batch_size, 1, 1))

    def test_vector_control(self):
        """Test with vector-valued controls."""
        order = 3
        disc_node_tau = jnp.array([-1.0, 0.0, 1.0])
        V_disc = jnp.vander(disc_node_tau, N=order, increasing=True)
        V_hat_inv = jnp.linalg.inv(V_disc)

        # Control with shape (3,) at each node
        control_shape = (3,)
        u_at_nodes = jnp.array(np.random.rand(3, *control_shape))

        input_to_disc_map = jnp.array([0, 1, 2])
        disc_node_indices = jnp.array([0, 1, 2])
        stau = jnp.array([0.0, 0.5, 1.0])
        dstau_dt = 2.0

        u, u_dot, u_ddot = vandermonde_control_interp(
            u_at_nodes, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, V_hat_inv
        )

        # Verify output shapes
        self.assertEqual(u.shape, (3, *control_shape))
        self.assertEqual(u_dot.shape, (3, *control_shape))
        self.assertEqual(u_ddot.shape, (3, *control_shape))

    def test_constant_polynomial(self):
        """Test that constant polynomial is interpolated exactly."""
        order = 3
        disc_node_tau = jnp.array([-1.0, 0.0, 1.0])
        V_disc = jnp.vander(disc_node_tau, N=order, increasing=True)
        V_hat_inv = jnp.linalg.inv(V_disc)

        # Constant value at all nodes
        constant_value = 5.0
        u_at_nodes = jnp.ones((3, 1)) * constant_value

        input_to_disc_map = jnp.array([0, 1, 2])
        disc_node_indices = jnp.array([0, 1, 2])
        stau = jnp.linspace(-1, 1, 10)
        dstau_dt = 1.0

        u, u_dot, u_ddot = vandermonde_control_interp(
            u_at_nodes, stau, dstau_dt, input_to_disc_map,
            disc_node_indices, V_hat_inv
        )

        # Constant polynomial should give constant value everywhere
        assert_allclose(u, constant_value, rtol=1e-10,
                       err_msg="Constant polynomial not preserved")

        # Derivatives should be zero
        assert_allclose(u_dot, 0.0, atol=1e-10,
                       err_msg="First derivative of constant should be zero")
        assert_allclose(u_ddot, 0.0, atol=1e-10,
                       err_msg="Second derivative of constant should be zero")


if __name__ == '__main__':
    unittest.main()
