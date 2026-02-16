"""Test JAX radau_defect function against original OpenMDAO component."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import openmdao.api as om
from numpy.testing import assert_allclose

from dymos.jax.pseudospectral.components.radau_defect import radau_defect
from dymos.transcriptions.pseudospectral.components.radau_defect_comp import RadauDefectComp
from dymos.transcriptions.grid_data import GridData
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestRadauDefectJax(unittest.TestCase):
    """Test JAX radau_defect function against original component."""

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
        """Verify JAX function outputs match original component for scalar state."""
        # Create a simple scalar ODE for testing
        from openmdao.test_suite.components.paraboloid import Paraboloid

        class SimpleODE(om.ExplicitComponent):
            """Simple ODE for testing: x_dot = -x"""
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']
                self.add_input('x', shape=(nn,))
                self.add_output('x_dot', shape=(nn,))

            def compute(self, inputs, outputs):
                outputs['x_dot'] = -inputs['x']

        # Setup state options
        state_options = {
            'x': {
                'shape': (1,),
                'units': None,
                'rate_source': 'x_dot',
                'targets': ['x'],
                'defect_scaler': None,
                'defect_ref': None,
                'solve_segments': False,
                'connected_initial': False,
            }
        }

        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']
        num_nodes = self.gd.subset_num_nodes['all']

        # Create test data
        x_values = np.random.rand(num_disc_nodes, 1)
        x_initial = np.random.rand(1, 1)
        x_final = np.random.rand(1, 1)
        f_ode_values = np.random.rand(num_col_nodes, 1)
        dt_dstau = np.random.rand(num_col_nodes)

        # Get outputs from original OpenMDAO component
        prob = om.Problem()

        # Add ODE
        ode_class = SimpleODE
        prob.model.add_subsystem('ode', ode_class(num_nodes=num_col_nodes))

        # Add defect comp
        defect_comp = RadauDefectComp(
            grid_data=self.gd,
            state_options=state_options,
            time_units='s'
        )
        prob.model.add_subsystem('defect', defect_comp)

        # Configure
        class DummyPhase:
            def classify_var(self, name):
                return 'ode'

        defect_comp.configure_io(DummyPhase())
        prob.setup()

        # Set values
        prob.set_val('defect.states:x', x_values)
        prob.set_val('defect.initial_states:x', x_initial)
        prob.set_val('defect.final_states:x', x_final)
        prob.set_val('defect.f_ode:x', f_ode_values)
        prob.set_val('defect.dt_dstau', dt_dstau)

        prob.run_model()

        # Get original outputs
        rate_defect_orig = prob.get_val('defect.state_rate_defects:x')
        initial_defect_orig = prob.get_val('defect.initial_state_defects:x')
        final_defect_orig = prob.get_val('defect.final_state_defects:x')

        # Get differentiation matrix from component
        D = prob.model.defect._D.toarray()  # Convert sparse to dense

        # Get segment end indices
        segment_end_indices = self.gd.subset_node_indices['segment_ends']

        # Compute with JAX function
        rate_defect_jax, initial_defect_jax, final_defect_jax, cnty_defect_jax = radau_defect(
            x_values, f_ode_values, dt_dstau, D, x_initial, x_final, segment_end_indices
        )

        # Compare outputs
        assert_allclose(rate_defect_jax, rate_defect_orig, rtol=1e-12,
                       err_msg="Rate defects don't match")
        assert_allclose(initial_defect_jax, initial_defect_orig, rtol=1e-12,
                       err_msg="Initial defects don't match")
        assert_allclose(final_defect_jax, final_defect_orig, rtol=1e-12,
                       err_msg="Final defects don't match")

    def test_defect_relationships(self):
        """Test mathematical relationships in defect computation."""
        # Simple test with known values
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']

        # Create simple linear state: x(tau) = tau
        # Then dx/dtau = 1
        x_values = jnp.linspace(0, 1, num_disc_nodes).reshape(-1, 1)
        x_initial = jnp.array([[0.0]])
        x_final = jnp.array([[1.0]])

        # For a linear function, any differentiation should give constant derivative
        # Create a simple differentiation matrix
        D = jnp.ones((num_col_nodes, num_disc_nodes)) / num_disc_nodes

        dt_dstau = jnp.ones(num_col_nodes)

        # f_ode should equal dx/dtau scaled by dt_dstau
        # For linear x, dx/dtau = constant
        f_ode = jnp.ones((num_col_nodes, 1))

        rate_defect, initial_defect, final_defect, _ = radau_defect(
            x_values, f_ode, dt_dstau, D, x_initial, x_final, None
        )

        # Initial defect should be zero since x[0] = x_initial
        assert_allclose(initial_defect, 0.0, atol=1e-12,
                       err_msg="Initial defect should be zero")

        # Final defect should be zero since x[-1] = x_final
        assert_allclose(final_defect, 0.0, atol=1e-12,
                       err_msg="Final defect should be zero")

    def test_derivatives_with_jax_grad(self):
        """Verify JAX autodiff derivatives work correctly."""
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']

        # Create test data
        x_values = jnp.array(np.random.rand(num_disc_nodes, 1))
        f_ode = jnp.array(np.random.rand(num_col_nodes, 1))
        dt_dstau = jnp.array(np.random.rand(num_col_nodes))
        D = jnp.array(np.random.rand(num_col_nodes, num_disc_nodes))
        x_initial = jnp.array(np.random.rand(1, 1))
        x_final = jnp.array(np.random.rand(1, 1))

        # Define a scalar objective based on defects
        def objective(x):
            rate_def, init_def, final_def, _ = radau_defect(
                x, f_ode, dt_dstau, D, x_initial, x_final, None
            )
            return jnp.sum(rate_def**2 + init_def**2 + final_def**2)

        # Compute gradient with JAX
        grad_jax = jax.grad(objective)(x_values)

        # Compute gradient with finite differences
        eps = 1e-7
        grad_fd = np.zeros_like(x_values)
        for i in range(num_disc_nodes):
            x_plus = x_values.at[i, 0].add(eps)
            x_minus = x_values.at[i, 0].add(-eps)

            obj_plus = objective(x_plus)
            obj_minus = objective(x_minus)

            grad_fd[i, 0] = (obj_plus - obj_minus) / (2 * eps)

        # Compare gradients
        assert_allclose(grad_jax, grad_fd, rtol=1e-5, atol=1e-8,
                       err_msg="JAX gradient doesn't match finite difference")

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']

        x_values = jnp.ones((num_disc_nodes, 1))
        f_ode = jnp.ones((num_col_nodes, 1))
        dt_dstau = jnp.ones(num_col_nodes)
        D = jnp.eye(num_col_nodes, num_disc_nodes)
        x_initial = jnp.ones((1, 1))
        x_final = jnp.ones((1, 1))

        # JIT compile the function
        defect_jitted = jax.jit(radau_defect, static_argnames=())

        # First call (compilation)
        result1 = defect_jitted(x_values, f_ode, dt_dstau, D, x_initial, x_final, None)

        # Second call (should use cached compilation)
        result2 = defect_jitted(x_values, f_ode, dt_dstau, D, x_initial, x_final, None)

        # Results should be identical
        assert_allclose(result1[0], result2[0], rtol=1e-14)
        assert_allclose(result1[1], result2[1], rtol=1e-14)
        assert_allclose(result1[2], result2[2], rtol=1e-14)

    def test_vmap_compatibility(self):
        """Test that function works with jax.vmap for batch processing."""
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']

        # Create batch of inputs (different trajectories)
        batch_size = 3
        x_batch = jnp.array(np.random.rand(batch_size, num_disc_nodes, 1))
        f_ode_batch = jnp.array(np.random.rand(batch_size, num_col_nodes, 1))

        # Same D matrix for all
        dt_dstau = jnp.ones(num_col_nodes)
        D = jnp.eye(num_col_nodes, num_disc_nodes)
        x_initial = jnp.zeros((1, 1))
        x_final = jnp.ones((1, 1))

        # Vectorize over batches
        defect_batched = jax.vmap(
            lambda x, f: radau_defect(x, f, dt_dstau, D, x_initial, x_final, None),
            in_axes=(0, 0)
        )

        results = defect_batched(x_batch, f_ode_batch)
        rate_defects, initial_defects, final_defects, _ = results

        # Verify shapes
        self.assertEqual(rate_defects.shape, (batch_size, num_col_nodes, 1))
        self.assertEqual(initial_defects.shape, (batch_size, 1, 1))
        self.assertEqual(final_defects.shape, (batch_size, 1, 1))

        # Verify each batch result matches individual computation
        for i in range(batch_size):
            rate_def, init_def, final_def, _ = radau_defect(
                x_batch[i], f_ode_batch[i], dt_dstau, D, x_initial, x_final, None
            )
            assert_allclose(rate_defects[i], rate_def, rtol=1e-14)
            assert_allclose(initial_defects[i], init_def, rtol=1e-14)
            assert_allclose(final_defects[i], final_def, rtol=1e-14)

    def test_vector_state(self):
        """Test with vector-valued state (shape > 1)."""
        num_disc_nodes = self.gd.subset_num_nodes['state_disc']
        num_col_nodes = self.gd.subset_num_nodes['col']
        state_shape = (3,)

        # Create test data with vector state
        x_values = jnp.array(np.random.rand(num_disc_nodes, *state_shape))
        f_ode = jnp.array(np.random.rand(num_col_nodes, *state_shape))
        dt_dstau = jnp.array(np.random.rand(num_col_nodes))
        D = jnp.array(np.random.rand(num_col_nodes, num_disc_nodes))
        x_initial = jnp.array(np.random.rand(1, *state_shape))
        x_final = jnp.array(np.random.rand(1, *state_shape))

        # Compute defects
        rate_defect, initial_defect, final_defect, _ = radau_defect(
            x_values, f_ode, dt_dstau, D, x_initial, x_final, None
        )

        # Verify shapes
        self.assertEqual(rate_defect.shape, (num_col_nodes, *state_shape))
        self.assertEqual(initial_defect.shape, (1, *state_shape))
        self.assertEqual(final_defect.shape, (1, *state_shape))

        # Verify initial and final defects
        expected_initial = x_initial - x_values[0:1, :]
        expected_final = x_final - x_values[-1:, :]

        assert_allclose(initial_defect, expected_initial, rtol=1e-14)
        assert_allclose(final_defect, expected_final, rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
