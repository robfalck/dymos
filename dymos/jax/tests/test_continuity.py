"""Test JAX continuity defect functions."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.common.continuity import (
    continuity_defect, state_continuity_defect,
    control_continuity_defect, control_rate_continuity_defect,
    control_rate2_continuity_defect
)


class TestContinuityJax(unittest.TestCase):
    """Test JAX continuity defect functions."""

    def test_basic_continuity_defect(self):
        """Test basic continuity defect computation."""
        # Create segment endpoint values: [seg0_start, seg0_end, seg1_start, seg1_end, seg2_start, seg2_end]
        # For 3 segments, we have 6 endpoint nodes
        # Discontinuous: seg0_end != seg1_start, seg1_end != seg2_start
        segment_values = jnp.array([
            [0.0],  # seg0_start
            [1.0],  # seg0_end
            [1.5],  # seg1_start - jump of 0.5
            [2.5],  # seg1_end
            [3.0],  # seg2_start - jump of 0.5
            [4.0],  # seg2_end
        ])

        defect = continuity_defect(segment_values)

        # Expected defects: [seg1_start - seg0_end, seg2_start - seg1_end]
        expected = jnp.array([[0.5], [0.5]])

        assert_allclose(defect, expected, rtol=1e-14,
                       err_msg="Basic continuity defect incorrect")

    def test_continuous_values(self):
        """Test that continuous values give zero defect."""
        # Continuous values: seg_end[i] == seg_start[i+1]
        segment_values = jnp.array([
            [0.0],  # seg0_start
            [1.0],  # seg0_end
            [1.0],  # seg1_start - continuous
            [2.0],  # seg1_end
            [2.0],  # seg2_start - continuous
            [3.0],  # seg2_end
        ])

        defect = continuity_defect(segment_values)

        # All defects should be zero
        assert_allclose(defect, 0.0, atol=1e-14,
                       err_msg="Continuous values should have zero defect")

    def test_derivative_continuity_scaling(self):
        """Test derivative continuity with scaling."""
        # Same values, but testing rate continuity (deriv_order=1)
        segment_values = jnp.array([
            [0.0],  # seg0_start
            [1.0],  # seg0_end
            [1.5],  # seg1_start - jump of 0.5
            [2.5],  # seg1_end
        ])

        t_duration = 2.0
        dt_dptau = t_duration / 2.0  # = 1.0

        defect = continuity_defect(segment_values, dt_dptau=dt_dptau, deriv_order=1)

        # Expected defect: (seg1_start - seg0_end) * dt_dptau^1
        expected = jnp.array([[0.5]])

        assert_allclose(defect, expected, rtol=1e-14,
                       err_msg="First derivative continuity scaling incorrect")

        # Test second derivative (deriv_order=2)
        defect2 = continuity_defect(segment_values, dt_dptau=dt_dptau, deriv_order=2)

        # Expected defect: (seg1_start - seg0_end) * dt_dptau^2
        expected2 = jnp.array([[0.5]])

        assert_allclose(defect2, expected2, rtol=1e-14,
                       err_msg="Second derivative continuity scaling incorrect")

    def test_vector_state(self):
        """Test with vector-valued states."""
        # State with shape (3,)
        segment_values = jnp.array([
            [1.0, 2.0, 3.0],  # seg0_start
            [2.0, 3.0, 4.0],  # seg0_end
            [2.5, 3.5, 4.5],  # seg1_start
            [3.5, 4.5, 5.5],  # seg1_end
        ])

        defect = continuity_defect(segment_values)

        # Expected: seg1_start - seg0_end
        expected = jnp.array([[0.5, 0.5, 0.5]])

        assert_allclose(defect, expected, rtol=1e-14,
                       err_msg="Vector state continuity incorrect")

    def test_convenience_functions(self):
        """Test convenience functions."""
        values = jnp.array([
            [0.0],
            [1.0],
            [1.2],
            [2.0],
        ])

        # State continuity
        state_defect = state_continuity_defect(values)
        expected = continuity_defect(values, deriv_order=0)
        assert_allclose(state_defect, expected, rtol=1e-14)

        # Control continuity
        control_defect = control_continuity_defect(values)
        assert_allclose(control_defect, expected, rtol=1e-14)

        # Control rate continuity
        t_duration = 2.0
        rate_defect = control_rate_continuity_defect(values, t_duration)
        expected_rate = continuity_defect(values, dt_dptau=t_duration/2.0, deriv_order=1)
        assert_allclose(rate_defect, expected_rate, rtol=1e-14)

        # Control rate2 continuity
        rate2_defect = control_rate2_continuity_defect(values, t_duration)
        expected_rate2 = continuity_defect(values, dt_dptau=t_duration/2.0, deriv_order=2)
        assert_allclose(rate2_defect, expected_rate2, rtol=1e-14)

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        values = jnp.ones((6, 1))

        # JIT compile
        defect_jitted = jax.jit(continuity_defect)

        # First call
        result1 = defect_jitted(values)

        # Second call
        result2 = defect_jitted(values)

        # Should be identical
        assert_allclose(result1, result2, rtol=1e-14)

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff works."""
        values = jnp.array([[1.0], [2.0], [2.5], [3.0]])

        # Define objective
        def objective(vals):
            defect = continuity_defect(vals)
            return jnp.sum(defect**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(values)

        # Verify gradient is not all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient should not be all zeros")

    def test_vmap_compatibility(self):
        """Test vectorization with jax.vmap."""
        batch_size = 3
        values_batch = jnp.array(np.random.rand(batch_size, 6, 1))

        # Vectorize over batch
        defect_batched = jax.vmap(continuity_defect, in_axes=0)

        results = defect_batched(values_batch)

        # Verify shape
        self.assertEqual(results.shape, (batch_size, 2, 1))


if __name__ == '__main__':
    unittest.main()
