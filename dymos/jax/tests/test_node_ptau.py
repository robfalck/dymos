"""Test JAX node_ptau function."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.grid.node_ptau import node_ptau


class TestNodePtauJax(unittest.TestCase):
    """Test JAX node_ptau function."""

    def test_equal_segments_uniform_nodes(self):
        """Test with two equal segments and uniform node spacing."""
        # Two segments: [-1, 0] and [0, 1]
        # Each segment has 3 nodes at stau = [-1, 0, 1]
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        node_stau = jnp.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # Segment 0: maps stau [-1, 0, 1] to ptau [-1.0, -0.5, 0.0]
        # Segment 1: maps stau [-1, 0, 1] to ptau [0.0, 0.5, 1.0]
        expected = jnp.array([-1.0, -0.5, 0.0, 0.0, 0.5, 1.0])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Equal segments with uniform nodes incorrect")

    def test_unequal_segments(self):
        """Test with two unequal segments."""
        # First segment: 75% of phase ([-1, 0.5])
        # Second segment: 25% of phase ([0.5, 1])
        segment_ends = jnp.array([-1.0, 0.5, 1.0])
        node_stau = jnp.array([-1.0, 1.0, -1.0, 1.0])
        nodes_per_segment = jnp.array([2, 2])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # Segment 0: stau=-1 maps to -1.0, stau=1 maps to 0.5
        # Segment 1: stau=-1 maps to 0.5, stau=1 maps to 1.0
        expected = jnp.array([-1.0, 0.5, 0.5, 1.0])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Unequal segments incorrect")

    def test_three_segments(self):
        """Test with three segments of varying sizes."""
        segment_ends = jnp.array([-1.0, -0.5, 0.0, 1.0])
        # Each segment has 2 nodes at stau = [-1, 1]
        node_stau = jnp.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        nodes_per_segment = jnp.array([2, 2, 2])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # Segment 0: [-1, -0.5] -> stau [-1, 1] maps to [-1.0, -0.5]
        # Segment 1: [-0.5, 0.0] -> stau [-1, 1] maps to [-0.5, 0.0]
        # Segment 2: [0.0, 1.0] -> stau [-1, 1] maps to [0.0, 1.0]
        expected = jnp.array([-1.0, -0.5, -0.5, 0.0, 0.0, 1.0])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Three segments incorrect")

    def test_single_segment(self):
        """Test with a single segment spanning entire phase."""
        segment_ends = jnp.array([-1.0, 1.0])
        node_stau = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        nodes_per_segment = jnp.array([5])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # Single segment: stau directly maps to ptau (both span [-1, 1])
        expected = node_stau  # Direct mapping

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Single segment should map stau directly to ptau")

    def test_non_uniform_nodes(self):
        """Test with non-uniformly spaced nodes within segments."""
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        # Segment 0: nodes at stau = [-1, -0.5, 1]
        # Segment 1: nodes at stau = [-1, 0, 0.5, 1]
        node_stau = jnp.array([-1.0, -0.5, 1.0, -1.0, 0.0, 0.5, 1.0])
        nodes_per_segment = jnp.array([3, 4])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # Segment 0: [-1, 0] in ptau
        # stau=-1 -> ptau=-1, stau=-0.5 -> ptau=-0.75, stau=1 -> ptau=0
        # Segment 1: [0, 1] in ptau
        # stau=-1 -> ptau=0, stau=0 -> ptau=0.5, stau=0.5 -> ptau=0.75, stau=1 -> ptau=1
        expected_seg0 = jnp.array([-1.0, -0.75, 0.0])
        expected_seg1 = jnp.array([0.0, 0.5, 0.75, 1.0])
        expected = jnp.concatenate([expected_seg0, expected_seg1])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Non-uniform nodes incorrect")

    def test_non_symmetric_phase(self):
        """Test with phase not centered at zero."""
        # Phase from 0 to 2
        segment_ends = jnp.array([0.0, 1.0, 2.0])
        node_stau = jnp.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # Segment 0: [0, 1] -> stau [-1, 0, 1] maps to [0.0, 0.5, 1.0]
        # Segment 1: [1, 2] -> stau [-1, 0, 1] maps to [1.0, 1.5, 2.0]
        expected = jnp.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])

        assert_allclose(result, expected, rtol=1e-14)

    def test_mapping_formula(self):
        """Verify the mapping formula: ptau = ptau_start + (stau+1)*(ptau_end-ptau_start)/2."""
        segment_ends = jnp.array([-1.0, 0.3, 1.0])
        node_stau = jnp.array([-0.8, 0.2, -0.6, 0.4])
        nodes_per_segment = jnp.array([2, 2])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # Manually compute expected values
        # Segment 0: ptau_start=-1.0, ptau_end=0.3
        ptau_0_node0 = -1.0 + (-0.8 + 1.0) * (0.3 - (-1.0)) / 2.0  # stau=-0.8
        ptau_0_node1 = -1.0 + (0.2 + 1.0) * (0.3 - (-1.0)) / 2.0   # stau=0.2

        # Segment 1: ptau_start=0.3, ptau_end=1.0
        ptau_1_node0 = 0.3 + (-0.6 + 1.0) * (1.0 - 0.3) / 2.0  # stau=-0.6
        ptau_1_node1 = 0.3 + (0.4 + 1.0) * (1.0 - 0.3) / 2.0   # stau=0.4

        expected = jnp.array([ptau_0_node0, ptau_0_node1, ptau_1_node0, ptau_1_node1])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Mapping formula verification failed")

    def test_boundary_nodes(self):
        """Test that segment boundaries map correctly."""
        segment_ends = jnp.array([-1.0, -0.2, 0.5, 1.0])
        # Put nodes at segment boundaries (stau = -1 and +1)
        node_stau = jnp.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        nodes_per_segment = jnp.array([2, 2, 2])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # Nodes at segment boundaries should equal segment_ends values
        expected = jnp.array([
            -1.0, -0.2,  # Segment 0 boundaries
            -0.2, 0.5,   # Segment 1 boundaries
            0.5, 1.0     # Segment 2 boundaries
        ])

        assert_allclose(result, expected, rtol=1e-14,
                       err_msg="Boundary nodes should match segment_ends")

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        node_stau = jnp.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])
        total_nodes = 6

        # JIT compile with total_nodes as static argument
        jitted_fn = jax.jit(node_ptau, static_argnames=['total_nodes'])

        # First call
        result1 = jitted_fn(segment_ends, node_stau, nodes_per_segment, total_nodes)

        # Second call
        result2 = jitted_fn(segment_ends, node_stau, nodes_per_segment, total_nodes)

        # Should be identical
        assert_allclose(result1, result2, rtol=1e-14)

    def test_derivatives_wrt_segment_ends(self):
        """Verify JAX autodiff works with respect to segment endpoints."""
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        node_stau = jnp.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])

        # Define objective that depends on segment_ends
        def objective(seg_ends):
            ptau_vals = node_ptau(seg_ends, node_stau, nodes_per_segment)
            return jnp.sum(ptau_vals**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(segment_ends)

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

        # Verify gradient manually for middle segment endpoint
        eps = 1e-6
        seg_ends_plus = segment_ends.at[1].set(segment_ends[1] + eps)
        seg_ends_minus = segment_ends.at[1].set(segment_ends[1] - eps)

        obj_plus = objective(seg_ends_plus)
        obj_minus = objective(seg_ends_minus)
        grad_fd = (obj_plus - obj_minus) / (2 * eps)

        # Compare finite difference to autodiff gradient at middle point
        assert_allclose(grad_jax[1], grad_fd, rtol=1e-5,
                       err_msg="Autodiff gradient doesn't match finite difference")

    def test_derivatives_wrt_node_stau(self):
        """Verify JAX autodiff works with respect to node positions."""
        segment_ends = jnp.array([-1.0, 0.0, 1.0])
        node_stau = jnp.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])

        # Define objective that depends on node_stau
        def objective(stau):
            ptau_vals = node_ptau(segment_ends, stau, nodes_per_segment)
            return jnp.sum(ptau_vals**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(node_stau)

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

    def test_vmap_compatibility(self):
        """Test vectorization with jax.vmap."""
        # Batch of different segment configurations
        batch_size = 3
        segment_ends_batch = jnp.array([
            [-1.0, 0.0, 1.0],
            [-1.0, -0.5, 1.0],
            [-1.0, 0.25, 1.0],
        ])
        node_stau = jnp.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])

        # Vectorize over batch
        batched_fn = jax.vmap(
            lambda seg_ends: node_ptau(seg_ends, node_stau, nodes_per_segment),
            in_axes=0
        )

        results = batched_fn(segment_ends_batch)

        # Verify shape
        self.assertEqual(results.shape, (batch_size, 6))

        # Verify first batch element manually
        expected_0 = node_ptau(segment_ends_batch[0], node_stau, nodes_per_segment)
        assert_allclose(results[0], expected_0, rtol=1e-14)

    def test_optimization_example(self):
        """Demonstrate grid optimization capability."""
        node_stau = jnp.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        nodes_per_segment = jnp.array([3, 3])

        # Objective: find segment spacing that places nodes at specific targets
        # Use asymmetric target to ensure gradient is non-zero
        target_ptau = jnp.array([-1.0, -0.7, -0.3, 0.1, 0.5, 1.0])

        def objective(middle_point):
            segment_ends = jnp.array([-1.0, middle_point, 1.0])
            ptau_vals = node_ptau(segment_ends, node_stau, nodes_per_segment)
            return jnp.sum((ptau_vals - target_ptau)**2)

        # Compute gradient at middle_point = 0.0
        grad_at_zero = jax.grad(objective)(0.0)

        # Gradient should be non-zero for asymmetric target
        self.assertTrue(abs(grad_at_zero) > 1e-6,
                       "Gradient should be non-zero for asymmetric target")

        # Find better middle point
        from scipy.optimize import minimize_scalar
        result_opt = minimize_scalar(lambda x: float(objective(x)),
                                     bounds=(-0.8, 0.8), method='bounded')

        # Optimal middle point should reduce objective
        obj_initial = objective(0.0)
        obj_optimal = objective(result_opt.x)
        self.assertTrue(obj_optimal < obj_initial,
                       "Optimization should reduce objective")

    def test_consistency_across_segments(self):
        """Verify adjacent segment boundaries produce same ptau value."""
        segment_ends = jnp.array([-1.0, -0.3, 0.4, 1.0])
        # Put nodes at segment boundaries
        node_stau = jnp.array([1.0, -1.0, 1.0, -1.0])  # End of seg, start of next
        nodes_per_segment = jnp.array([1, 1, 1, 1])

        result = node_ptau(segment_ends, node_stau, nodes_per_segment)

        # End of segment i should equal start of segment i+1
        # Seg 0 end (stau=1) should be ptau=-0.3
        # Seg 1 start (stau=-1) should be ptau=-0.3
        assert_allclose(result[0], -0.3, rtol=1e-14)
        assert_allclose(result[1], -0.3, rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
