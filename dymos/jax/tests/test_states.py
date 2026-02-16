"""Test JAX states_passthrough function."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.picard_shooting.states import states_passthrough


class TestStatesJax(unittest.TestCase):
    """Test JAX states_passthrough function."""

    def test_identity_operation(self):
        """Verify function returns input unchanged."""
        # Scalar state
        states_scalar = jnp.array([[1.0], [2.0], [3.0]])
        output = states_passthrough(states_scalar)
        assert_allclose(output, states_scalar, rtol=1e-14,
                       err_msg="Scalar state should pass through unchanged")

        # Vector state
        states_vector = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = states_passthrough(states_vector)
        assert_allclose(output, states_vector, rtol=1e-14,
                       err_msg="Vector state should pass through unchanged")

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        states = jnp.ones((5, 2))

        # JIT compile
        passthrough_jitted = jax.jit(states_passthrough)

        # First call
        result1 = passthrough_jitted(states)

        # Second call
        result2 = passthrough_jitted(states)

        # Should be identical
        assert_allclose(result1, result2, rtol=1e-14)
        assert_allclose(result1, states, rtol=1e-14)

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff works (should be identity Jacobian)."""
        states = jnp.array([[1.0], [2.0], [3.0]])

        # Define objective
        def objective(s):
            output = states_passthrough(s)
            return jnp.sum(output**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(states)

        # Gradient should be 2 * states (from sum(s^2))
        expected_grad = 2 * states
        assert_allclose(grad_jax, expected_grad, rtol=1e-14,
                       err_msg="Gradient should be 2*states")

    def test_vmap_compatibility(self):
        """Test vectorization with jax.vmap."""
        batch_size = 3
        states_batch = jnp.array(np.random.rand(batch_size, 5, 2))

        # Vectorize over batch
        passthrough_batched = jax.vmap(states_passthrough, in_axes=0)

        results = passthrough_batched(states_batch)

        # Verify shape and values
        self.assertEqual(results.shape, (batch_size, 5, 2))
        assert_allclose(results, states_batch, rtol=1e-14)

    def test_preserves_dtype(self):
        """Verify function preserves input dtype."""
        # Float32
        states_f32 = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
        output_f32 = states_passthrough(states_f32)
        self.assertEqual(output_f32.dtype, jnp.float32)

        # Float64
        states_f64 = jnp.array([[1.0], [2.0]], dtype=jnp.float64)
        output_f64 = states_passthrough(states_f64)
        self.assertEqual(output_f64.dtype, jnp.float64)


if __name__ == '__main__':
    unittest.main()
