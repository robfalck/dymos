"""Test JAX node_dptau_dstau function."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.grid.node_dptau_dstau import node_dptau_dstau


class TestNodeDptauDstauJax(unittest.TestCase):
    """Test JAX node_dptau_dstau function."""

    def test_equal_segments(self):
        """Test with two equal segments."""
        # Two segments, each spanning half of ptau space [-1, 1]
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])

        result = node_dptau_dstau(segment_ends, nodes_per_segment)

        # Each segment has length 1.0 in ptau space
        # dptau_dstau = 1.0 / 2.0 = 0.5 for both segments
        expected = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Equal segments should have equal dptau_dstau")

    def test_unequal_segments(self):
        """Test with two unequal segments."""
        # First segment: 75% of phase, second: 25%
        segment_ends = jnp.array([-1.0, 0.5, 1.0])
        nodes_per_segment = jnp.array([4, 2])

        result = node_dptau_dstau(segment_ends, nodes_per_segment)

        # Segment 0: (0.5 - (-1.0)) / 2.0 = 0.75
        # Segment 1: (1.0 - 0.5) / 2.0 = 0.25
        expected = jnp.array([0.75, 0.75, 0.75, 0.75, 0.25, 0.25])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Unequal segments should have different dptau_dstau")

    def test_three_segments(self):
        """Test with three segments of varying sizes."""
        segment_ends = jnp.array([-1.0, -0.5, 0.0, 1.0])
        nodes_per_segment = jnp.array([2, 3, 4])

        result = node_dptau_dstau(segment_ends, nodes_per_segment)

        # Segment 0: (-0.5 - (-1.0)) / 2.0 = 0.25
        # Segment 1: (0.0 - (-0.5)) / 2.0 = 0.25
        # Segment 2: (1.0 - 0.0) / 2.0 = 0.5
        expected = jnp.array([
            0.25, 0.25,           # Segment 0
            0.25, 0.25, 0.25,     # Segment 1
            0.5, 0.5, 0.5, 0.5    # Segment 2
        ])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Three segments incorrect")

    def test_single_segment(self):
        """Test with a single segment."""
        segment_ends = jnp.array([-1.0, 1.0])
        nodes_per_segment = jnp.array([5])

        result = node_dptau_dstau(segment_ends, nodes_per_segment)

        # Single segment spanning entire phase: (1.0 - (-1.0)) / 2.0 = 1.0
        expected = jnp.ones(5)

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Single segment should have dptau_dstau = 1.0")

    def test_different_node_counts(self):
        """Test with segments having very different node counts."""
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([10, 2])  # First segment has many nodes

        result = node_dptau_dstau(segment_ends, nodes_per_segment)

        # Both segments span 1.0, so dptau_dstau = 0.5 for both
        expected_seg0 = jnp.ones(10) * 0.5
        expected_seg1 = jnp.ones(2) * 0.5
        expected = jnp.concatenate([expected_seg0, expected_seg1])

        assert_allclose(result, expected, rtol=1e-14)

    def test_non_symmetric_phase(self):
        """Test with phase not centered at zero."""
        # Phase from 0 to 2
        segment_ends = jnp.array([0.0, 1.0, 2.0])
        nodes_per_segment = jnp.array([3, 3])

        result = node_dptau_dstau(segment_ends, nodes_per_segment)

        # Each segment has length 1.0, dptau_dstau = 0.5
        expected = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        assert_allclose(result, expected, rtol=1e-14)

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])
        total_nodes = 6  # Static argument for JIT

        # JIT compile with total_nodes as static argument
        jitted_fn = jax.jit(node_dptau_dstau, static_argnames=['total_nodes'])

        # First call
        result1 = jitted_fn(segment_ends, nodes_per_segment, total_nodes)

        # Second call
        result2 = jitted_fn(segment_ends, nodes_per_segment, total_nodes)

        # Should be identical
        assert_allclose(result1, result2, rtol=1e-14)

    def test_derivatives_wrt_segment_ends(self):
        """Verify JAX autodiff works with respect to segment endpoints."""
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])

        # Define objective that depends on segment_ends
        def objective(seg_ends):
            dptau_dstau_vals = node_dptau_dstau(seg_ends, nodes_per_segment)
            return jnp.sum(dptau_dstau_vals**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(segment_ends)

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

        # Manually verify gradient for middle segment endpoint
        # If we perturb segment_ends[1], it affects both segments
        eps = 1e-6
        seg_ends_plus = segment_ends.at[1].set(segment_ends[1] + eps)
        seg_ends_minus = segment_ends.at[1].set(segment_ends[1] - eps)

        obj_plus = objective(seg_ends_plus)
        obj_minus = objective(seg_ends_minus)

        grad_fd = (obj_plus - obj_minus) / (2 * eps)

        # Compare finite difference to autodiff gradient at middle point
        # Use absolute tolerance since both values are very small (near zero)
        assert_allclose(grad_jax[1], grad_fd, rtol=1e-5, atol=1e-9,
                       err_msg="Autodiff gradient doesn't match finite difference")

    def test_vmap_compatibility(self):
        """Test vectorization with jax.vmap."""
        # Batch of different segment configurations
        batch_size = 3
        segment_ends_batch = jnp.array([
            [-1.0, 0.0, 1.0],
            [-1.0, -0.5, 1.0],
            [-1.0, 0.25, 1.0],
        ])
        nodes_per_segment = jnp.array([3, 3])  # Same for all

        # Vectorize over batch
        batched_fn = jax.vmap(
            lambda seg_ends: node_dptau_dstau(seg_ends, nodes_per_segment),
            in_axes=0
        )

        results = batched_fn(segment_ends_batch)

        # Verify shape
        self.assertEqual(results.shape, (batch_size, 6))

        # Verify first batch element manually
        expected_0 = node_dptau_dstau(segment_ends_batch[0], nodes_per_segment)
        assert_allclose(results[0], expected_0, rtol=1e-14)

    def test_optimization_example(self):
        """Demonstrate grid optimization capability."""
        # Suppose we want to find segment spacing that minimizes some objective
        nodes_per_segment = jnp.array([3, 3])

        # Objective: minimize sum of squared dptau_dstau
        # (contrived example, but demonstrates differentiability)
        def objective(middle_point):
            # Constrain middle_point to be between -1 and 1
            segment_ends = jnp.array([-1.0, middle_point, 1.0])
            dptau_dstau_vals = node_dptau_dstau(segment_ends, nodes_per_segment)
            return jnp.sum(dptau_dstau_vals**2)

        # Compute gradient at middle_point = 0.0
        grad_at_zero = jax.grad(objective)(0.0)

        # Gradient should be zero at equal spacing (symmetric minimum)
        assert_allclose(grad_at_zero, 0.0, atol=1e-10,
                       err_msg="Gradient should be zero at symmetric spacing")

        # Compute gradient at asymmetric point
        grad_at_quarter = jax.grad(objective)(0.25)

        # Gradient should be non-zero
        self.assertTrue(abs(grad_at_quarter) > 1e-6,
                       "Gradient should be non-zero at asymmetric spacing")

    def test_consistency_with_formula(self):
        """Verify the formula dptau_dstau = (ptau_end - ptau_start) / 2."""
        segment_ends = jnp.array([-1.0, -0.3, 0.4, 1.0])
        nodes_per_segment = jnp.array([2, 3, 4])

        result = node_dptau_dstau(segment_ends, nodes_per_segment)

        # Manually compute expected values
        seg0_dptau_dstau = (-0.3 - (-1.0)) / 2.0  # 0.35
        seg1_dptau_dstau = (0.4 - (-0.3)) / 2.0   # 0.35
        seg2_dptau_dstau = (1.0 - 0.4) / 2.0      # 0.30

        expected = jnp.array([
            seg0_dptau_dstau, seg0_dptau_dstau,                    # Segment 0
            seg1_dptau_dstau, seg1_dptau_dstau, seg1_dptau_dstau,  # Segment 1
            seg2_dptau_dstau, seg2_dptau_dstau, seg2_dptau_dstau, seg2_dptau_dstau  # Segment 2
        ])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Results don't match manual formula calculation")

    def test_total_num_nodes(self):
        """Verify output length matches sum of nodes_per_segment."""
        segment_ends = jnp.array([-1.0, 0.0, 0.5, 1.0])
        nodes_per_segment = jnp.array([5, 7, 3])

        result = node_dptau_dstau(segment_ends, nodes_per_segment)

        expected_length = 5 + 7 + 3
        self.assertEqual(len(result), expected_length,
                        "Output length should equal sum of nodes_per_segment")


if __name__ == '__main__':
    unittest.main()
