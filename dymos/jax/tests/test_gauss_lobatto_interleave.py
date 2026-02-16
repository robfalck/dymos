"""Test JAX gauss_lobatto_interleave function against original OpenMDAO component."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import openmdao.api as om
from numpy.testing import assert_allclose

from dymos.jax.pseudospectral.components.gauss_lobatto_interleave import gauss_lobatto_interleave
from dymos.transcriptions.pseudospectral.components.gauss_lobatto_interleave_comp import GaussLobattoInterleaveComp
from dymos.transcriptions.grid_data import GridData


class TestGaussLobattoInterleaveJax(unittest.TestCase):
    """Test JAX gauss_lobatto_interleave function against original component."""

    def setUp(self):
        """Set up grid data for tests."""
        # Create a simple Gauss-Lobatto grid with 2 segments, 3 nodes per segment
        self.gd = GridData(
            num_segments=2,
            transcription='gauss-lobatto',
            transcription_order=3,
            segment_ends=np.array([0.0, 0.5, 1.0]),
            compressed=False
        )

    def test_outputs_match_original_scalar(self):
        """Verify JAX function outputs match original component for scalar values."""
        # Setup test data
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']
        num_nodes = self.gd.subset_num_nodes['all']

        disc_values = np.random.rand(num_disc_nodes)
        col_values = np.random.rand(num_col_nodes)

        # Get outputs from original OpenMDAO component
        prob = om.Problem()
        comp = GaussLobattoInterleaveComp(grid_data=self.gd)
        prob.model.add_subsystem('interleave', comp)
        comp.add_var('x', shape=(1,), units=None, disc_src='disc', col_src='col')
        prob.setup()

        prob.set_val('interleave.disc_values:x', disc_values.reshape(-1, 1))
        prob.set_val('interleave.col_values:x', col_values.reshape(-1, 1))
        prob.run_model()

        all_values_orig = prob.get_val('interleave.all_values:x').flatten()

        # Get outputs from JAX function
        disc_indices = self.gd.subset_node_indices['state_disc']
        col_indices = self.gd.subset_node_indices['col']

        all_values_jax = gauss_lobatto_interleave(
            disc_values.reshape(-1, 1),
            col_values.reshape(-1, 1),
            disc_indices,
            col_indices,
            num_nodes
        ).flatten()

        # Compare outputs (should match exactly)
        assert_allclose(all_values_jax, all_values_orig, rtol=1e-14,
                       err_msg="Interleaved values don't match")

    def test_outputs_match_original_vector(self):
        """Verify JAX function outputs match original component for vector values."""
        # Setup test data with shape (3,)
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']
        num_nodes = self.gd.subset_num_nodes['all']
        shape = (3,)

        disc_values = np.random.rand(num_disc_nodes, *shape)
        col_values = np.random.rand(num_col_nodes, *shape)

        # Get outputs from original OpenMDAO component
        prob = om.Problem()
        comp = GaussLobattoInterleaveComp(grid_data=self.gd)
        prob.model.add_subsystem('interleave', comp)
        comp.add_var('x', shape=shape, units=None, disc_src='disc', col_src='col')
        prob.setup()

        prob.set_val('interleave.disc_values:x', disc_values)
        prob.set_val('interleave.col_values:x', col_values)
        prob.run_model()

        all_values_orig = prob.get_val('interleave.all_values:x')

        # Get outputs from JAX function
        disc_indices = self.gd.subset_node_indices['state_disc']
        col_indices = self.gd.subset_node_indices['col']

        all_values_jax = gauss_lobatto_interleave(
            disc_values,
            col_values,
            disc_indices,
            col_indices,
            num_nodes
        )

        # Compare outputs (should match exactly)
        assert_allclose(all_values_jax, all_values_orig, rtol=1e-14,
                       err_msg="Interleaved vector values don't match")

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff derivatives match expected values."""
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']
        num_nodes = self.gd.subset_num_nodes['all']
        disc_indices = self.gd.subset_node_indices['state_disc']
        col_indices = self.gd.subset_node_indices['col']

        # Create appropriately sized arrays
        disc_values = jnp.arange(1.0, num_disc_nodes + 1.0).reshape(-1, 1)
        col_values = jnp.arange(1.5, num_col_nodes + 1.5).reshape(-1, 1)

        # The interleave operation is linear, so derivatives should be 0 or 1
        # d(all_values[i])/d(disc_values[j]) = 1 if i == disc_indices[j], else 0
        # d(all_values[i])/d(col_values[j]) = 1 if i == col_indices[j], else 0

        # Test gradient with respect to disc_values
        def output_sum_disc(dv):
            result = gauss_lobatto_interleave(dv, col_values, disc_indices, col_indices, num_nodes)
            return jnp.sum(result)

        grad_disc = jax.grad(output_sum_disc)(disc_values)

        # Since we sum all outputs, and each disc_value appears exactly once,
        # gradient should be all ones
        expected_grad_disc = jnp.ones_like(disc_values)
        assert_allclose(grad_disc, expected_grad_disc, rtol=1e-12,
                       err_msg="JAX gradient wrt disc_values doesn't match expected")

        # Test gradient with respect to col_values
        def output_sum_col(cv):
            result = gauss_lobatto_interleave(disc_values, cv, disc_indices, col_indices, num_nodes)
            return jnp.sum(result)

        grad_col = jax.grad(output_sum_col)(col_values)

        # Same reasoning - gradient should be all ones
        expected_grad_col = jnp.ones_like(col_values)
        assert_allclose(grad_col, expected_grad_col, rtol=1e-12,
                       err_msg="JAX gradient wrt col_values doesn't match expected")

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']
        num_nodes = self.gd.subset_num_nodes['all']
        disc_indices = self.gd.subset_node_indices['state_disc']
        col_indices = self.gd.subset_node_indices['col']

        disc_values = jnp.ones((num_disc_nodes, 1))
        col_values = jnp.ones((num_col_nodes, 1)) * 2.0

        # JIT compile the function
        interleave_jitted = jax.jit(
            gauss_lobatto_interleave,
            static_argnames=('num_nodes',)
        )

        # First call (compilation)
        result1 = interleave_jitted(disc_values, col_values, disc_indices, col_indices, num_nodes)

        # Second call (should use cached compilation)
        result2 = interleave_jitted(disc_values, col_values, disc_indices, col_indices, num_nodes)

        # Results should be identical
        assert_allclose(result1, result2, rtol=1e-14)

        # Check that values are correctly interleaved
        assert_allclose(result1[disc_indices], disc_values, rtol=1e-14)
        assert_allclose(result1[col_indices], col_values, rtol=1e-14)

    def test_vmap_compatibility(self):
        """Test that function works with jax.vmap for batch processing."""
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']
        num_nodes = self.gd.subset_num_nodes['all']
        disc_indices = self.gd.subset_node_indices['state_disc']
        col_indices = self.gd.subset_node_indices['col']

        # Create batch of inputs
        batch_size = 3
        disc_values_batch = jnp.ones((batch_size, num_disc_nodes, 1))
        col_values_batch = jnp.ones((batch_size, num_col_nodes, 1)) * 2.0

        # Vectorize over batches
        interleave_batched = jax.vmap(
            lambda dv, cv: gauss_lobatto_interleave(dv, cv, disc_indices, col_indices, num_nodes),
            in_axes=(0, 0)
        )

        results = interleave_batched(disc_values_batch, col_values_batch)

        # Verify shapes
        self.assertEqual(results.shape, (batch_size, num_nodes, 1))

        # Verify each batch result
        for i in range(batch_size):
            assert_allclose(results[i][disc_indices], disc_values_batch[i], rtol=1e-14)
            assert_allclose(results[i][col_indices], col_values_batch[i], rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
