"""Test JAX control_interp functions."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.common.control_interp import (
    control_interp_polynomial,
    control_interp_full
)


class TestControlInterpJax(unittest.TestCase):
    """Test JAX control interpolation functions."""

    def test_polynomial_basic(self):
        """Test polynomial control interpolation with simple example."""
        # Linear control: u(tau) = 1 + tau, tau in [-1, 1]
        # At LGL nodes for order 2: tau = [-1, 0, 1]
        # u = [0, 1, 2]
        u_input = jnp.array([[0.0], [1.0], [2.0]])
        t_duration = 10.0

        # Simple Lagrange matrices (order 2)
        # Evaluate at tau = [-1, -0.5, 0, 0.5, 1]
        num_output = 5
        order = 2

        # Identity interpolation matrix for testing (diagonal blocks)
        L = jnp.array([
            [1.0, 0.0, 0.0],  # tau = -1
            [0.375, 0.75, -0.125],  # tau = -0.5
            [0.0, 1.0, 0.0],  # tau = 0
            [-0.125, 0.75, 0.375],  # tau = 0.5
            [0.0, 0.0, 1.0],  # tau = 1
        ])

        # First derivative matrix (for linear function, derivative = 1)
        D = jnp.array([
            [-0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ])

        # Second derivative matrix (for linear function, second deriv = 0)
        D2 = jnp.zeros((num_output, order + 1))

        val, rate, rate2, bval, brate, brate2 = control_interp_polynomial(
            u_input, L, D, D2, t_duration
        )

        # Check shapes
        self.assertEqual(val.shape, (5, 1))
        self.assertEqual(rate.shape, (5, 1))
        self.assertEqual(rate2.shape, (5, 1))
        self.assertEqual(bval.shape, (2, 1))
        self.assertEqual(brate.shape, (2, 1))
        self.assertEqual(brate2.shape, (2, 1))

        # Boundary values should match first and last
        assert_allclose(bval, jnp.array([[val[0, 0]], [val[-1, 0]]]), rtol=1e-14)

    def test_polynomial_vector_control(self):
        """Test polynomial interpolation with vector-valued control."""
        u_input = jnp.array([
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0]
        ])
        t_duration = 5.0

        # Simple matrices
        L = jnp.eye(3)
        D = jnp.array([
            [-0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ])
        D2 = jnp.zeros((3, 3))

        val, rate, rate2, bval, brate, brate2 = control_interp_polynomial(
            u_input, L, D, D2, t_duration
        )

        # Check shapes
        self.assertEqual(val.shape, (3, 2))
        self.assertEqual(bval.shape, (2, 2))

        # Boundary values
        assert_allclose(bval[0], val[0], rtol=1e-14)
        assert_allclose(bval[1], val[-1], rtol=1e-14)

    def test_full_basic(self):
        """Test full control interpolation with simple example."""
        # 4 input nodes, 6 output nodes
        u_input = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        dt_dstau = jnp.ones(6) * 2.0  # Constant dt_dstau

        # Simple interpolation (identity + extra nodes)
        L = jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # First derivative (finite differences)
        D = jnp.array([
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ])

        D2 = jnp.zeros((6, 4))

        val, rate, rate2, bval, brate, brate2, vc, rc, r2c = control_interp_full(
            u_input, L, D, D2, dt_dstau, S=None
        )

        # Check shapes
        self.assertEqual(val.shape, (6, 1))
        self.assertEqual(rate.shape, (6, 1))
        self.assertEqual(bval.shape, (2, 1))

        # Continuity defects should be None (S=None)
        self.assertIsNone(vc)
        self.assertIsNone(rc)
        self.assertIsNone(r2c)

        # Boundary values
        assert_allclose(bval, jnp.array([[1.0], [4.0]]), rtol=1e-14)

    def test_full_with_continuity_defects(self):
        """Test full control with continuity defect computation."""
        # 2 segments with 3 nodes each: [0, 1, 2] [3, 4, 5]
        u_input = jnp.array([[0.0], [1.0], [2.0], [3.0]])
        dt_dstau = jnp.ones(6)

        L = jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])

        D = jnp.ones((6, 4)) * 0.5  # Simplified
        D2 = jnp.zeros((6, 4))

        # Selection matrix for continuity:
        # Defect = val[3] - val[2] (start of seg 1 - end of seg 0)
        S = jnp.array([[0.0, 0.0, -1.0, 1.0, 0.0, 0.0]])

        val, rate, rate2, bval, brate, brate2, vc, rc, r2c = control_interp_full(
            u_input, L, D, D2, dt_dstau, S=S
        )

        # Check continuity defects are computed
        self.assertIsNotNone(vc)
        self.assertIsNotNone(rc)
        self.assertIsNotNone(r2c)

        # Check shapes
        self.assertEqual(vc.shape, (1, 1))
        self.assertEqual(rc.shape, (1, 1))
        self.assertEqual(r2c.shape, (1, 1))

    def test_full_vector_control(self):
        """Test full interpolation with vector-valued control."""
        u_input = jnp.array([
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ])
        dt_dstau = jnp.ones(3) * 1.5

        L = jnp.eye(3)
        D = jnp.array([
            [-0.5, 0.5, 0.0],
            [0.0, -0.5, 0.5],
            [-0.5, 0.0, 0.5],
        ])
        D2 = jnp.zeros((3, 3))

        val, rate, rate2, bval, brate, brate2, vc, rc, r2c = control_interp_full(
            u_input, L, D, D2, dt_dstau, S=None
        )

        # Check shapes
        self.assertEqual(val.shape, (3, 2))
        self.assertEqual(rate.shape, (3, 2))
        self.assertEqual(bval.shape, (2, 2))

    def test_jit_compilation_polynomial(self):
        """Verify polynomial function works with JAX JIT compilation."""
        u_input = jnp.ones((3, 1))
        L = jnp.eye(3)
        D = jnp.ones((3, 3)) * 0.5
        D2 = jnp.zeros((3, 3))
        t_duration = 10.0

        # JIT compile
        interp_jitted = jax.jit(control_interp_polynomial)

        # First call
        result1 = interp_jitted(u_input, L, D, D2, t_duration)

        # Second call
        result2 = interp_jitted(u_input, L, D, D2, t_duration)

        # Should be identical
        for r1, r2 in zip(result1, result2):
            assert_allclose(r1, r2, rtol=1e-14)

    def test_jit_compilation_full(self):
        """Verify full function works with JAX JIT compilation."""
        u_input = jnp.ones((4, 1))
        L = jnp.eye(4)
        D = jnp.ones((4, 4)) * 0.5
        D2 = jnp.zeros((4, 4))
        dt_dstau = jnp.ones(4) * 2.0

        # JIT compile
        interp_jitted = jax.jit(control_interp_full, static_argnames=['S'])

        # First call (without S)
        result1 = interp_jitted(u_input, L, D, D2, dt_dstau, S=None)

        # Second call
        result2 = interp_jitted(u_input, L, D, D2, dt_dstau, S=None)

        # Should be identical
        for r1, r2 in zip(result1[:6], result2[:6]):  # First 6 outputs
            assert_allclose(r1, r2, rtol=1e-14)

    def test_derivatives_polynomial(self):
        """Verify JAX autodiff works for polynomial interpolation."""
        u_input = jnp.array([[1.0], [2.0], [3.0]])
        L = jnp.eye(3)
        D = jnp.ones((3, 3)) * 0.5
        D2 = jnp.zeros((3, 3))
        t_duration = 10.0

        # Define objective
        def objective(u):
            val, _, _, _, _, _ = control_interp_polynomial(u, L, D, D2, t_duration)
            return jnp.sum(val**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(u_input)

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

    def test_derivatives_full(self):
        """Verify JAX autodiff works for full interpolation."""
        u_input = jnp.array([[1.0], [2.0], [3.0]])
        L = jnp.eye(3)
        D = jnp.ones((3, 3)) * 0.5
        D2 = jnp.zeros((3, 3))
        dt_dstau = jnp.ones(3) * 2.0

        # Define objective
        def objective(u):
            val, _, _, _, _, _, _, _, _ = control_interp_full(u, L, D, D2, dt_dstau, S=None)
            return jnp.sum(val**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(u_input)

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

    def test_vmap_compatibility_polynomial(self):
        """Test vectorization with jax.vmap for polynomial."""
        batch_size = 2
        u_batch = jnp.array(np.random.rand(batch_size, 3, 1))
        L = jnp.eye(3)
        D = jnp.ones((3, 3)) * 0.5
        D2 = jnp.zeros((3, 3))
        t_duration = 10.0

        # Vectorize over batch
        interp_batched = jax.vmap(
            lambda u: control_interp_polynomial(u, L, D, D2, t_duration),
            in_axes=0
        )

        results = interp_batched(u_batch)

        # Verify shape of first output (val)
        self.assertEqual(results[0].shape, (batch_size, 3, 1))

    def test_vmap_compatibility_full(self):
        """Test vectorization with jax.vmap for full."""
        batch_size = 2
        u_batch = jnp.array(np.random.rand(batch_size, 3, 1))
        L = jnp.eye(3)
        D = jnp.ones((3, 3)) * 0.5
        D2 = jnp.zeros((3, 3))
        dt_dstau = jnp.ones(3) * 2.0

        # Vectorize over batch
        interp_batched = jax.vmap(
            lambda u: control_interp_full(u, L, D, D2, dt_dstau, S=None),
            in_axes=0
        )

        results = interp_batched(u_batch)

        # Verify shape of first output (val)
        self.assertEqual(results[0].shape, (batch_size, 3, 1))

    def test_rate_scaling_polynomial(self):
        """Test that rate scaling is correct for polynomial controls."""
        # Constant control
        u_input = jnp.ones((3, 1)) * 5.0
        L = jnp.eye(3)
        D = jnp.zeros((3, 3))  # Derivative of constant is zero
        D2 = jnp.zeros((3, 3))
        t_duration = 10.0

        val, rate, rate2, _, _, _ = control_interp_polynomial(
            u_input, L, D, D2, t_duration
        )

        # Rate should be zero (derivative of constant)
        assert_allclose(rate, 0.0, atol=1e-14)

    def test_rate_scaling_full(self):
        """Test that rate scaling with dt_dstau is correct."""
        # Linear increase
        u_input = jnp.array([[0.0], [1.0], [2.0]])
        L = jnp.eye(3)

        # Derivative matrix (forward difference)
        D = jnp.array([
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
        ])

        D2 = jnp.zeros((3, 3))
        dt_dstau = jnp.ones(3) * 2.0  # Constant scaling

        val, rate, rate2, _, _, _, _, _, _ = control_interp_full(
            u_input, L, D, D2, dt_dstau, S=None
        )

        # Rate should be approximately (1 - 0) / 2 = 0.5 at node 0
        # (derivative in tau space is 1, divided by dt_dstau = 2)
        self.assertTrue(jnp.abs(rate[0, 0] - 0.5) < 0.1,
                       f"Expected rate ~0.5, got {rate[0, 0]}")


if __name__ == '__main__':
    unittest.main()
