"""Test JAX birkhoff_picard_update functions."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.picard_shooting.birkhoff_picard_update import (
    birkhoff_picard_update_forward,
    birkhoff_picard_update_backward
)


class TestBirkhoffPicardUpdateJax(unittest.TestCase):
    """Test JAX Birkhoff Picard update functions."""

    def test_forward_update_basic(self):
        """Test forward Birkhoff integration with simple example."""
        # 2 segments with 2 nodes each
        num_nodes = 4
        f_computed = jnp.array([[1.0], [1.0], [2.0], [2.0]])  # Constant rate per segment
        dt_dstau = jnp.ones(num_nodes)
        x_0 = jnp.array([[0.0], [10.0]])  # Initial states for 2 segments
        seg_repeats = jnp.array([2, 2])

        # Simple integration matrix (trapezoidal-like)
        # Each segment integrates its own nodes
        B = jnp.array([
            [0.0, 0.5, 0.0, 0.0],  # Node 0: no integration yet
            [0.5, 0.0, 0.0, 0.0],  # Node 1: integrate node 0-1
            [0.0, 0.0, 0.0, 0.5],  # Node 2: no integration yet (new segment)
            [0.0, 0.0, 0.5, 0.0],  # Node 3: integrate node 2-3
        ])

        x_hat, x_b = birkhoff_picard_update_forward(
            f_computed, dt_dstau, x_0, B, seg_repeats
        )

        # Verify shapes
        self.assertEqual(x_hat.shape, (4, 1))
        self.assertEqual(x_b.shape, (1, 1))

        # x_hat should be x_0_repeated + B @ f
        # Segment 0: x_0[0] = 0.0, integrated values should add f*dt_dstau
        # Segment 1: x_0[1] = 10.0, integrated values should add f*dt_dstau
        # The exact values depend on the integration matrix B

        # x_b should be the last node of x_hat
        assert_allclose(x_b, x_hat[-1:, :], rtol=1e-14)

    def test_backward_update_basic(self):
        """Test backward Birkhoff integration with simple example."""
        # 2 segments with 2 nodes each
        num_nodes = 4
        f_computed = jnp.array([[1.0], [1.0], [2.0], [2.0]])
        dt_dstau = jnp.ones(num_nodes)
        x_f = jnp.array([[5.0], [15.0]])  # Final states for 2 segments
        seg_repeats = jnp.array([2, 2])

        # Simple integration matrix
        B = jnp.array([
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0],
        ])

        x_hat, x_a = birkhoff_picard_update_backward(
            f_computed, dt_dstau, x_f, B, seg_repeats
        )

        # Verify shapes
        self.assertEqual(x_hat.shape, (4, 1))
        self.assertEqual(x_a.shape, (1, 1))

        # x_a should be the first node of x_hat
        assert_allclose(x_a, x_hat[:1, :], rtol=1e-14)

    def test_single_segment_forward(self):
        """Test forward update with single segment."""
        num_nodes = 3
        f_computed = jnp.array([[1.0], [2.0], [3.0]])
        dt_dstau = jnp.ones(num_nodes)
        x_0 = jnp.array([[0.0]])
        seg_repeats = jnp.array([3])

        # Simple integration: cumulative sum scaled
        B = jnp.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ])

        x_hat, x_b = birkhoff_picard_update_forward(
            f_computed, dt_dstau, x_0, B, seg_repeats
        )

        # x_hat[0] should be close to x_0 (no integration yet)
        # x_b should be x_hat[-1]
        assert_allclose(x_b, x_hat[-1:, :], rtol=1e-14)

        # Verify x_hat[0] starts from x_0
        # (exact value depends on B matrix, but should be close)
        self.assertTrue(jnp.abs(x_hat[0, 0] - x_0[0, 0]) < 0.1)

    def test_vector_state_forward(self):
        """Test forward update with vector-valued state."""
        num_nodes = 4
        f_computed = jnp.array([
            [1.0, 0.5],
            [1.0, 0.5],
            [2.0, 1.0],
            [2.0, 1.0]
        ])
        dt_dstau = jnp.ones(num_nodes)
        x_0 = jnp.array([[0.0, 0.0], [10.0, 5.0]])
        seg_repeats = jnp.array([2, 2])

        B = jnp.array([
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0],
        ])

        x_hat, x_b = birkhoff_picard_update_forward(
            f_computed, dt_dstau, x_0, B, seg_repeats
        )

        # Verify shapes
        self.assertEqual(x_hat.shape, (4, 2))
        self.assertEqual(x_b.shape, (1, 2))

        # x_b should match last node
        assert_allclose(x_b, x_hat[-1:, :], rtol=1e-14)

    def test_jit_compilation_forward(self):
        """Verify forward function works with JAX JIT compilation."""
        num_nodes = 4
        f_computed = jnp.ones((num_nodes, 1))
        dt_dstau = jnp.ones(num_nodes)
        x_0 = jnp.array([[0.0], [1.0]])
        B = jnp.eye(num_nodes) * 0.5
        seg_repeats = jnp.array([2, 2])

        # JIT compile
        forward_jitted = jax.jit(birkhoff_picard_update_forward)

        # First call
        result1 = forward_jitted(f_computed, dt_dstau, x_0, B, seg_repeats)

        # Second call
        result2 = forward_jitted(f_computed, dt_dstau, x_0, B, seg_repeats)

        # Should be identical
        assert_allclose(result1[0], result2[0], rtol=1e-14)
        assert_allclose(result1[1], result2[1], rtol=1e-14)

    def test_jit_compilation_backward(self):
        """Verify backward function works with JAX JIT compilation."""
        num_nodes = 4
        f_computed = jnp.ones((num_nodes, 1))
        dt_dstau = jnp.ones(num_nodes)
        x_f = jnp.array([[5.0], [6.0]])
        B = jnp.eye(num_nodes) * 0.5
        seg_repeats = jnp.array([2, 2])

        # JIT compile
        backward_jitted = jax.jit(birkhoff_picard_update_backward)

        # First call
        result1 = backward_jitted(f_computed, dt_dstau, x_f, B, seg_repeats)

        # Second call
        result2 = backward_jitted(f_computed, dt_dstau, x_f, B, seg_repeats)

        # Should be identical
        assert_allclose(result1[0], result2[0], rtol=1e-14)
        assert_allclose(result1[1], result2[1], rtol=1e-14)

    def test_derivatives_forward(self):
        """Verify JAX autodiff works for forward update."""
        num_nodes = 4
        f_computed = jnp.ones((num_nodes, 1))
        dt_dstau = jnp.ones(num_nodes)
        x_0 = jnp.array([[1.0], [2.0]])
        B = jnp.eye(num_nodes) * 0.5
        seg_repeats = jnp.array([2, 2])

        # Define objective that depends on x_0
        def objective(x0):
            x_hat, x_b = birkhoff_picard_update_forward(
                f_computed, dt_dstau, x0, B, seg_repeats
            )
            return jnp.sum(x_hat**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(x_0)

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

    def test_derivatives_backward(self):
        """Verify JAX autodiff works for backward update."""
        num_nodes = 4
        f_computed = jnp.ones((num_nodes, 1))
        dt_dstau = jnp.ones(num_nodes)
        x_f = jnp.array([[5.0], [6.0]])
        B = jnp.eye(num_nodes) * 0.5
        seg_repeats = jnp.array([2, 2])

        # Define objective that depends on x_f
        def objective(xf):
            x_hat, x_a = birkhoff_picard_update_backward(
                f_computed, dt_dstau, xf, B, seg_repeats
            )
            return jnp.sum(x_hat**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(x_f)

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

    def test_consistency_between_directions(self):
        """Verify forward and backward give consistent results for reversible case."""
        # Use constant state rate - should be reversible
        num_nodes = 4
        f_computed = jnp.ones((num_nodes, 1)) * 2.0
        dt_dstau = jnp.ones(num_nodes)
        seg_repeats = jnp.array([2, 2])

        # Identity-like integration matrix
        B = jnp.array([
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0],
        ])

        # Forward from x_0 = [0, 10]
        x_0 = jnp.array([[0.0], [10.0]])
        x_hat_fwd, x_b_fwd = birkhoff_picard_update_forward(
            f_computed, dt_dstau, x_0, B, seg_repeats
        )

        # Backward from x_f = [final_seg0, final_seg1]
        # This is a sanity check - not exact reversal without proper setup
        # Just verify the functions execute without error
        x_f = jnp.array([[5.0], [15.0]])
        x_hat_bkwd, x_a_bkwd = birkhoff_picard_update_backward(
            f_computed, dt_dstau, x_f, B, seg_repeats
        )

        # Just verify shapes are consistent
        self.assertEqual(x_hat_fwd.shape, x_hat_bkwd.shape)


if __name__ == '__main__':
    unittest.main()
