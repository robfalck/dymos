"""Test JAX multiple_shooting_update functions."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.picard_shooting.multiple_shooting_update import (
    multiple_shooting_update_forward,
    multiple_shooting_update_backward
)


class TestMultipleShootingUpdateJax(unittest.TestCase):
    """Test JAX multiple shooting update functions."""

    def test_forward_update_basic(self):
        """Test forward shooting update with basic example."""
        # 3 segments with 2 nodes each: [0, 1] [2, 3] [4, 5]
        # Segment end indices are [1, 3] (odd indices)
        x = jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        x_a = jnp.array([[10.0]])  # Phase initial condition
        seg_end_indices = jnp.array([1, 3])

        x_0 = multiple_shooting_update_forward(x, x_a, seg_end_indices)

        # Expected:
        # Segment 0: x_a = 10.0
        # Segment 1: x[1] = 1.0
        # Segment 2: x[3] = 3.0
        expected = jnp.array([[10.0], [1.0], [3.0]])

        assert_allclose(x_0, expected, rtol=1e-14,
                       err_msg="Forward update incorrect")

    def test_backward_update_basic(self):
        """Test backward shooting update with basic example."""
        # 3 segments with 2 nodes each: [0, 1] [2, 3] [4, 5]
        # Segment start indices for segments 1, 2 are [2, 4] (even indices)
        x = jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        x_b = jnp.array([[20.0]])  # Phase final condition
        seg_start_indices = jnp.array([2, 4])

        x_f = multiple_shooting_update_backward(x, x_b, seg_start_indices)

        # Expected:
        # Segment 0: x[2] = 2.0 (start of next segment)
        # Segment 1: x[4] = 4.0 (start of next segment)
        # Segment 2: x_b = 20.0
        expected = jnp.array([[2.0], [4.0], [20.0]])

        assert_allclose(x_f, expected, rtol=1e-14,
                       err_msg="Backward update incorrect")

    def test_single_segment_forward(self):
        """Test forward update with single segment."""
        x = jnp.array([[0.0], [1.0], [2.0]])
        x_a = jnp.array([[5.0]])
        seg_end_indices = jnp.array([], dtype=jnp.int32)  # No intermediate segments

        x_0 = multiple_shooting_update_forward(x, x_a, seg_end_indices)

        # Only one segment, should just be x_a
        expected = jnp.array([[5.0]])
        assert_allclose(x_0, expected, rtol=1e-14)

    def test_single_segment_backward(self):
        """Test backward update with single segment."""
        x = jnp.array([[0.0], [1.0], [2.0]])
        x_b = jnp.array([[7.0]])
        seg_start_indices = jnp.array([], dtype=jnp.int32)

        x_f = multiple_shooting_update_backward(x, x_b, seg_start_indices)

        # Only one segment, should just be x_b
        expected = jnp.array([[7.0]])
        assert_allclose(x_f, expected, rtol=1e-14)

    def test_vector_state_forward(self):
        """Test forward update with vector-valued state."""
        # State shape (2,), 2 segments
        x = jnp.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ])
        x_a = jnp.array([[10.0, 11.0]])
        seg_end_indices = jnp.array([1])

        x_0 = multiple_shooting_update_forward(x, x_a, seg_end_indices)

        # Segment 0: x_a
        # Segment 1: x[1]
        expected = jnp.array([[10.0, 11.0], [1.0, 1.1]])
        assert_allclose(x_0, expected, rtol=1e-14)

    def test_vector_state_backward(self):
        """Test backward update with vector-valued state."""
        # State shape (2,), 2 segments
        x = jnp.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ])
        x_b = jnp.array([[20.0, 21.0]])
        seg_start_indices = jnp.array([2])

        x_f = multiple_shooting_update_backward(x, x_b, seg_start_indices)

        # Segment 0: x[2]
        # Segment 1: x_b
        expected = jnp.array([[2.0, 2.1], [20.0, 21.0]])
        assert_allclose(x_f, expected, rtol=1e-14)

    def test_jit_compilation_forward(self):
        """Verify forward function works with JAX JIT compilation."""
        x = jnp.ones((6, 1))
        x_a = jnp.array([[5.0]])
        seg_end_indices = jnp.array([1, 3])

        # JIT compile
        forward_jitted = jax.jit(multiple_shooting_update_forward)

        # First call
        result1 = forward_jitted(x, x_a, seg_end_indices)

        # Second call
        result2 = forward_jitted(x, x_a, seg_end_indices)

        # Should be identical
        assert_allclose(result1, result2, rtol=1e-14)

    def test_jit_compilation_backward(self):
        """Verify backward function works with JAX JIT compilation."""
        x = jnp.ones((6, 1))
        x_b = jnp.array([[7.0]])
        seg_start_indices = jnp.array([2, 4])

        # JIT compile
        backward_jitted = jax.jit(multiple_shooting_update_backward)

        # First call
        result1 = backward_jitted(x, x_b, seg_start_indices)

        # Second call
        result2 = backward_jitted(x, x_b, seg_start_indices)

        # Should be identical
        assert_allclose(result1, result2, rtol=1e-14)

    def test_derivatives_forward(self):
        """Verify JAX autodiff works for forward update."""
        x = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        x_a = jnp.array([[10.0]])
        seg_end_indices = jnp.array([1])

        # Define objective
        def objective(xa):
            x_0 = multiple_shooting_update_forward(x, xa, seg_end_indices)
            return jnp.sum(x_0**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(x_a)

        # Manually compute expected gradient
        # x_0 = [xa, x[1]] = [xa, 2.0]
        # objective = xa^2 + 4.0
        # d(objective)/d(xa) = 2*xa = 2*10 = 20
        expected_grad = jnp.array([[2 * 10.0]])

        assert_allclose(grad_jax, expected_grad, rtol=1e-12)

    def test_derivatives_backward(self):
        """Verify JAX autodiff works for backward update."""
        x = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        x_b = jnp.array([[20.0]])
        seg_start_indices = jnp.array([2])

        # Define objective
        def objective(xb):
            x_f = multiple_shooting_update_backward(x, xb, seg_start_indices)
            return jnp.sum(x_f**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(x_b)

        # Manually compute expected gradient
        # x_f = [x[2], xb] = [3.0, xb]
        # objective = 9.0 + xb^2
        # d(objective)/d(xb) = 2*xb = 2*20 = 40
        expected_grad = jnp.array([[2 * 20.0]])

        assert_allclose(grad_jax, expected_grad, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
