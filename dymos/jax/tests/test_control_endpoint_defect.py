"""Test JAX control_endpoint_defect function against original OpenMDAO component."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import openmdao.api as om
from numpy.testing import assert_allclose

from dymos.jax.pseudospectral.components.control_endpoint_defect import control_endpoint_defect
from dymos.transcriptions.pseudospectral.components.control_endpoint_defect_comp import ControlEndpointDefectComp
from dymos.transcriptions.grid_data import GridData


class TestControlEndpointDefectJax(unittest.TestCase):
    """Test JAX control_endpoint_defect function against original component."""

    def setUp(self):
        """Set up grid data for tests."""
        # Create a Radau pseudospectral grid
        self.gd = GridData(
            num_segments=2,
            transcription='radau-ps',
            transcription_order=3,
            segment_ends=np.array([0.0, 0.5, 1.0]),
            compressed=False
        )

    def test_outputs_match_original_scalar(self):
        """Verify JAX function outputs match original component for scalar control."""
        # Setup control options
        control_options = {
            'u': {
                'shape': (1,),
                'units': None,
                'opt': True,
                'fix_initial': False,
                'fix_final': False,
                'targets': ['u'],
                'rate_targets': [],
                'rate2_targets': []
            }
        }

        num_nodes = self.gd.subset_num_nodes['all']
        u_values = np.random.rand(num_nodes, 1)

        # Get outputs from original OpenMDAO component
        prob = om.Problem()
        prob.model.add_subsystem(
            'endpoint_defect',
            ControlEndpointDefectComp(
                grid_data=self.gd,
                control_options=control_options
            )
        )
        prob.setup()
        prob.set_val('endpoint_defect.controls:u', u_values)
        prob.run_model()

        defect_orig = prob.get_val('endpoint_defect.control_endpoint_defects:u')

        # Get outputs from JAX function
        # Extract the Lagrange matrix from the component
        comp = prob.model.endpoint_defect
        L = comp._L
        num_disc_end_segment = comp._num_disc_end_segment
        col_indices = self.gd.subset_node_indices['col']

        defect_jax = control_endpoint_defect(
            u_values,
            L,
            col_indices,
            num_disc_end_segment
        )

        # Compare outputs (should match very closely)
        assert_allclose(defect_jax, defect_orig, rtol=1e-12,
                       err_msg="Control endpoint defects don't match")

    def test_outputs_match_original_vector(self):
        """Verify JAX function outputs match original component for vector control."""
        # Setup control options with shape (3,)
        control_options = {
            'u': {
                'shape': (3,),
                'units': None,
                'opt': True,
                'fix_initial': False,
                'fix_final': False,
                'targets': ['u'],
                'rate_targets': [],
                'rate2_targets': []
            }
        }

        num_nodes = self.gd.subset_num_nodes['all']
        u_values = np.random.rand(num_nodes, 3)

        # Get outputs from original OpenMDAO component
        prob = om.Problem()
        prob.model.add_subsystem(
            'endpoint_defect',
            ControlEndpointDefectComp(
                grid_data=self.gd,
                control_options=control_options
            )
        )
        prob.setup()
        prob.set_val('endpoint_defect.controls:u', u_values)
        prob.run_model()

        defect_orig = prob.get_val('endpoint_defect.control_endpoint_defects:u')

        # Get outputs from JAX function
        comp = prob.model.endpoint_defect
        L = comp._L
        num_disc_end_segment = comp._num_disc_end_segment
        col_indices = self.gd.subset_node_indices['col']

        defect_jax = control_endpoint_defect(
            u_values,
            L,
            col_indices,
            num_disc_end_segment
        )

        # Compare outputs
        assert_allclose(defect_jax, defect_orig, rtol=1e-12,
                       err_msg="Vector control endpoint defects don't match")

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff derivatives match finite difference."""
        # Simple test setup
        num_nodes = self.gd.subset_num_nodes['all']
        col_indices = self.gd.subset_node_indices['col']

        # Create simple Lagrange matrix and control values
        num_disc_end_segment = self.gd.subset_num_nodes_per_segment['col'][-1]
        L = np.random.rand(1, num_disc_end_segment)
        u_values = jnp.array(np.random.rand(num_nodes, 1))

        # Define a scalar function for gradient computation
        def defect_sum(u):
            defect = control_endpoint_defect(u, L, col_indices, num_disc_end_segment)
            return jnp.sum(defect**2)

        # Compute gradient with JAX
        grad_jax = jax.grad(defect_sum)(u_values)

        # Compute gradient with finite differences
        eps = 1e-7
        grad_fd = np.zeros_like(u_values)
        for i in range(num_nodes):
            u_plus = u_values.at[i, 0].add(eps)
            u_minus = u_values.at[i, 0].add(-eps)

            defect_plus = defect_sum(u_plus)
            defect_minus = defect_sum(u_minus)

            grad_fd[i, 0] = (defect_plus - defect_minus) / (2 * eps)

        # Compare gradients
        assert_allclose(grad_jax, grad_fd, rtol=1e-5, atol=1e-8,
                       err_msg="JAX gradient doesn't match finite difference")

    def test_defect_is_zero_for_perfect_interpolation(self):
        """Test that defect is zero when endpoint matches interpolation exactly."""
        # Create a scenario where the endpoint value is exactly the interpolated value
        num_nodes = self.gd.subset_num_nodes['all']
        num_col_nodes = self.gd.subset_num_nodes['col']
        col_indices = self.gd.subset_node_indices['col']
        num_disc_end_segment = self.gd.subset_num_nodes_per_segment['col'][-1]

        # Create Lagrange matrix that averages the last few nodes
        L = np.ones((1, num_disc_end_segment)) / num_disc_end_segment

        # Create control values where the last value is the average of the last segment
        u_all = np.random.rand(num_nodes, 1)
        u_col = u_all[col_indices]
        u_col_end = u_col[-num_disc_end_segment:]

        # Set the endpoint to be exactly the interpolated value
        u_all[-1, 0] = np.mean(u_col_end[:, 0])

        # Compute defect
        defect = control_endpoint_defect(u_all, L, col_indices, num_disc_end_segment)

        # Defect should be very close to zero
        assert_allclose(defect, 0.0, atol=1e-12,
                       err_msg="Defect should be zero for perfect interpolation")

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        num_nodes = self.gd.subset_num_nodes['all']
        col_indices = self.gd.subset_node_indices['col']
        num_disc_end_segment = self.gd.subset_num_nodes_per_segment['col'][-1]

        L = np.random.rand(1, num_disc_end_segment)
        u_values = jnp.ones((num_nodes, 1))

        # JIT compile the function
        defect_jitted = jax.jit(
            control_endpoint_defect,
            static_argnames=('num_disc_end_segment',)
        )

        # First call (compilation)
        defect1 = defect_jitted(u_values, L, col_indices, num_disc_end_segment)

        # Second call (should use cached compilation)
        defect2 = defect_jitted(u_values, L, col_indices, num_disc_end_segment)

        # Results should be identical
        assert_allclose(defect1, defect2, rtol=1e-14)

    def test_vmap_compatibility(self):
        """Test that function works with jax.vmap for batch processing."""
        num_nodes = self.gd.subset_num_nodes['all']
        col_indices = self.gd.subset_node_indices['col']
        num_disc_end_segment = self.gd.subset_num_nodes_per_segment['col'][-1]

        # Create batch of inputs (different control trajectories)
        batch_size = 4
        u_batch = jnp.array(np.random.rand(batch_size, num_nodes, 1))
        L = np.random.rand(1, num_disc_end_segment)

        # Vectorize over batches
        defect_batched = jax.vmap(
            lambda u: control_endpoint_defect(u, L, col_indices, num_disc_end_segment),
            in_axes=0
        )

        defects = defect_batched(u_batch)

        # Verify shapes
        self.assertEqual(defects.shape, (batch_size, 1, 1))

        # Verify each batch result matches individual computation
        for i in range(batch_size):
            defect_single = control_endpoint_defect(
                u_batch[i], L, col_indices, num_disc_end_segment
            )
            assert_allclose(defects[i], defect_single, rtol=1e-14)

    def test_various_grid_configurations(self):
        """Test function with different grid configurations."""
        test_cases = [
            (1, 3),  # 1 segment, order 3
            (3, 5),  # 3 segments, order 5
            (2, 7),  # 2 segments, order 7
        ]

        for num_segments, order in test_cases:
            with self.subTest(num_segments=num_segments, order=order):
                gd = GridData(
                    num_segments=num_segments,
                    transcription='radau-ps',
                    transcription_order=order,
                    segment_ends=np.linspace(0, 1, num_segments + 1),
                    compressed=False
                )

                control_options = {
                    'u': {
                        'shape': (1,),
                        'units': None,
                        'opt': True,
                        'fix_initial': False,
                        'fix_final': False,
                        'targets': ['u'],
                        'rate_targets': [],
                        'rate2_targets': []
                    }
                }

                num_nodes = gd.subset_num_nodes['all']
                u_values = np.random.rand(num_nodes, 1)

                # Get reference from OpenMDAO
                prob = om.Problem()
                prob.model.add_subsystem(
                    'endpoint_defect',
                    ControlEndpointDefectComp(
                        grid_data=gd,
                        control_options=control_options
                    )
                )
                prob.setup()
                prob.set_val('endpoint_defect.controls:u', u_values)
                prob.run_model()

                defect_orig = prob.get_val('endpoint_defect.control_endpoint_defects:u')

                # Compute with JAX
                comp = prob.model.endpoint_defect
                L = comp._L
                num_disc_end_segment = comp._num_disc_end_segment
                col_indices = gd.subset_node_indices['col']

                defect_jax = control_endpoint_defect(
                    u_values, L, col_indices, num_disc_end_segment
                )

                # Compare
                assert_allclose(defect_jax, defect_orig, rtol=1e-12,
                               err_msg=f"Defects don't match for {num_segments} segments, order {order}")


if __name__ == '__main__':
    unittest.main()
