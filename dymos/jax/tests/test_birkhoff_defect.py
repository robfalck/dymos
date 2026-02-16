"""Test JAX birkhoff_defect function against original OpenMDAO component."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.pseudospectral.components.birkhoff_defect import birkhoff_defect


class TestBirkhoffDefectJax(unittest.TestCase):
    """Test JAX birkhoff_defect function."""

    def test_basic_computation(self):
        """Test basic defect computation with simple matrices."""
        # Simple test case with known dimensions
        num_col_nodes = 4
        state_shape = (1,)

        # Create test data
        X = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        V = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        f_computed = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        dt_dstau = jnp.ones(num_col_nodes)

        # Create simple Birkhoff matrices (identity-like for testing)
        # A maps the stacked XV to state defects
        A = jnp.eye(num_col_nodes, 2 * num_col_nodes)
        # C maps boundary conditions
        C = jnp.zeros((num_col_nodes, 2))

        # xv_indices - in this simple case, just range
        xv_indices = jnp.arange(2 * num_col_nodes)

        x_initial = jnp.zeros((1, *state_shape))
        x_final = jnp.ones((1, *state_shape))

        # Compute defects
        state_defect, state_rate_defect, cnty_defect = birkhoff_defect(
            X, V, f_computed, dt_dstau, A, C, xv_indices,
            x_initial, x_final, num_segments=1
        )

        # Verify shapes
        self.assertEqual(state_defect.shape, (num_col_nodes, *state_shape))
        self.assertEqual(state_rate_defect.shape, (num_col_nodes, *state_shape))
        self.assertIsNone(cnty_defect)  # Single segment, no continuity defect

    def test_state_rate_defect(self):
        """Test that state rate defect is computed correctly."""
        num_col_nodes = 5
        state_shape = (1,)

        X = jnp.ones((num_col_nodes, *state_shape))
        V = jnp.ones((num_col_nodes, *state_shape)) * 2.0
        f_computed = jnp.ones((num_col_nodes, *state_shape)) * 1.5
        dt_dstau = jnp.ones(num_col_nodes) * 0.5

        # Simple matrices (won't affect state_rate_defect)
        A = jnp.eye(num_col_nodes, 2 * num_col_nodes)
        C = jnp.zeros((num_col_nodes, 2))
        xv_indices = jnp.arange(2 * num_col_nodes)

        x_initial = jnp.zeros((1, *state_shape))
        x_final = jnp.ones((1, *state_shape))

        _, state_rate_defect, _ = birkhoff_defect(
            X, V, f_computed, dt_dstau, A, C, xv_indices,
            x_initial, x_final, num_segments=1
        )

        # State rate defect should be: V - f_computed * dt_dstau
        # = 2.0 - 1.5 * 0.5 = 2.0 - 0.75 = 1.25
        expected = V - f_computed * dt_dstau[:, None]

        assert_allclose(state_rate_defect, expected, rtol=1e-14,
                       err_msg="State rate defect computation is incorrect")

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff derivatives work correctly."""
        num_col_nodes = 4
        state_shape = (1,)

        # Create test data
        X = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        V = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        f_computed = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        dt_dstau = jnp.array(np.random.rand(num_col_nodes))

        A = jnp.array(np.random.rand(num_col_nodes, 2 * num_col_nodes))
        C = jnp.array(np.random.rand(num_col_nodes, 2))
        xv_indices = jnp.arange(2 * num_col_nodes)

        x_initial = jnp.array(np.random.rand(1, *state_shape))
        x_final = jnp.array(np.random.rand(1, *state_shape))

        # Define a scalar objective based on defects
        def objective(X_input):
            state_def, rate_def, _ = birkhoff_defect(
                X_input, V, f_computed, dt_dstau, A, C, xv_indices,
                x_initial, x_final, num_segments=1
            )
            return jnp.sum(state_def**2 + rate_def**2)

        # Compute gradient with JAX
        grad_jax = jax.grad(objective)(X)

        # Compute gradient with finite differences
        eps = 1e-7
        grad_fd = np.zeros_like(X)
        for i in range(num_col_nodes):
            X_plus = X.at[i, 0].add(eps)
            X_minus = X.at[i, 0].add(-eps)

            obj_plus = objective(X_plus)
            obj_minus = objective(X_minus)

            grad_fd[i, 0] = (obj_plus - obj_minus) / (2 * eps)

        # Compare gradients
        assert_allclose(grad_jax, grad_fd, rtol=1e-5, atol=1e-8,
                       err_msg="JAX gradient doesn't match finite difference")

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        num_col_nodes = 4
        state_shape = (1,)

        X = jnp.ones((num_col_nodes, *state_shape))
        V = jnp.ones((num_col_nodes, *state_shape))
        f_computed = jnp.ones((num_col_nodes, *state_shape))
        dt_dstau = jnp.ones(num_col_nodes)

        A = jnp.eye(num_col_nodes, 2 * num_col_nodes)
        C = jnp.zeros((num_col_nodes, 2))
        xv_indices = jnp.arange(2 * num_col_nodes)

        x_initial = jnp.zeros((1, *state_shape))
        x_final = jnp.ones((1, *state_shape))

        # JIT compile the function
        defect_jitted = jax.jit(
            birkhoff_defect,
            static_argnames=('num_segments',)
        )

        # First call (compilation)
        result1 = defect_jitted(
            X, V, f_computed, dt_dstau, A, C, xv_indices,
            x_initial, x_final, num_segments=1
        )

        # Second call (should use cached compilation)
        result2 = defect_jitted(
            X, V, f_computed, dt_dstau, A, C, xv_indices,
            x_initial, x_final, num_segments=1
        )

        # Results should be identical
        assert_allclose(result1[0], result2[0], rtol=1e-14)
        assert_allclose(result1[1], result2[1], rtol=1e-14)

    def test_vmap_compatibility(self):
        """Test that function works with jax.vmap for batch processing."""
        num_col_nodes = 4
        state_shape = (1,)
        batch_size = 3

        # Create batch of inputs (different trajectories)
        X_batch = jnp.array(np.random.rand(batch_size, num_col_nodes, *state_shape))
        V_batch = jnp.array(np.random.rand(batch_size, num_col_nodes, *state_shape))
        f_computed_batch = jnp.array(np.random.rand(batch_size, num_col_nodes, *state_shape))

        # Same matrices for all
        dt_dstau = jnp.ones(num_col_nodes)
        A = jnp.eye(num_col_nodes, 2 * num_col_nodes)
        C = jnp.zeros((num_col_nodes, 2))
        xv_indices = jnp.arange(2 * num_col_nodes)
        x_initial = jnp.zeros((1, *state_shape))
        x_final = jnp.ones((1, *state_shape))

        # Vectorize over batches
        defect_batched = jax.vmap(
            lambda X, V, f: birkhoff_defect(
                X, V, f, dt_dstau, A, C, xv_indices,
                x_initial, x_final, num_segments=1
            ),
            in_axes=(0, 0, 0)
        )

        results = defect_batched(X_batch, V_batch, f_computed_batch)
        state_defects, rate_defects, _ = results

        # Verify shapes
        self.assertEqual(state_defects.shape, (batch_size, num_col_nodes, *state_shape))
        self.assertEqual(rate_defects.shape, (batch_size, num_col_nodes, *state_shape))

        # Verify each batch result matches individual computation
        for i in range(batch_size):
            state_def, rate_def, _ = birkhoff_defect(
                X_batch[i], V_batch[i], f_computed_batch[i],
                dt_dstau, A, C, xv_indices,
                x_initial, x_final, num_segments=1
            )
            assert_allclose(state_defects[i], state_def, rtol=1e-14)
            assert_allclose(rate_defects[i], rate_def, rtol=1e-14)

    def test_vector_state(self):
        """Test with vector-valued state (shape > 1)."""
        num_col_nodes = 4
        state_shape = (3,)

        # Create test data with vector state
        X = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        V = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        f_computed = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        dt_dstau = jnp.array(np.random.rand(num_col_nodes))

        A = jnp.array(np.random.rand(num_col_nodes, 2 * num_col_nodes))
        C = jnp.array(np.random.rand(num_col_nodes, 2))
        xv_indices = jnp.arange(2 * num_col_nodes)

        x_initial = jnp.array(np.random.rand(1, *state_shape))
        x_final = jnp.array(np.random.rand(1, *state_shape))

        # Compute defects
        state_defect, state_rate_defect, _ = birkhoff_defect(
            X, V, f_computed, dt_dstau, A, C, xv_indices,
            x_initial, x_final, num_segments=1
        )

        # Verify shapes
        self.assertEqual(state_defect.shape, (num_col_nodes, *state_shape))
        self.assertEqual(state_rate_defect.shape, (num_col_nodes, *state_shape))

        # Verify state rate defect computation for vector states
        dt_dstau_expanded = dt_dstau[:, None]
        expected_rate_defect = V - f_computed * dt_dstau_expanded

        assert_allclose(state_rate_defect, expected_rate_defect, rtol=1e-14)

    def test_xv_reordering(self):
        """Test that XV reordering works correctly."""
        num_col_nodes = 4
        state_shape = (1,)

        X = jnp.arange(num_col_nodes, dtype=float).reshape(-1, 1)
        V = jnp.arange(num_col_nodes, num_col_nodes * 2, dtype=float).reshape(-1, 1)
        f_computed = jnp.zeros((num_col_nodes, *state_shape))
        dt_dstau = jnp.ones(num_col_nodes)

        # Test different reordering
        # Normal order: [X0, X1, X2, X3, V0, V1, V2, V3]
        # Reordered: [X0, V0, X1, V1, X2, V2, X3, V3]
        xv_indices = jnp.array([0, 4, 1, 5, 2, 6, 3, 7])

        # Create A matrix that extracts the reordered values
        A = jnp.eye(2 * num_col_nodes)
        C = jnp.zeros((2 * num_col_nodes, 2))

        x_initial = jnp.zeros((1, *state_shape))
        x_final = jnp.ones((1, *state_shape))

        state_defect, _, _ = birkhoff_defect(
            X, V, f_computed, dt_dstau, A, C, xv_indices,
            x_initial, x_final, num_segments=1
        )

        # After reordering, the state defect should have alternating X and V values
        # state_defect = A @ XV_reordered - C @ x_ab
        # Since A is identity and C is zero, state_defect = XV_reordered
        expected_order = jnp.array([0., 4., 1., 5., 2., 6., 3., 7.]).reshape(-1, 1)

        assert_allclose(state_defect, expected_order, rtol=1e-14,
                       err_msg="XV reordering not working correctly")


if __name__ == '__main__':
    unittest.main()
