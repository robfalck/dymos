"""Test JAX Radau brachistochrone phase assembly."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.examples.radau_brachistochrone_phase import (
    radau_brachistochrone_phase,
    create_radau_grid_data
)


class TestRadauBrachistochronePhase(unittest.TestCase):
    """Test Radau pseudospectral phase assembly for brachistochrone."""

    def setUp(self):
        """Set up test grid and design variables."""
        # Create a simple 2-segment grid
        self.num_segments = 2
        self.order = 3
        self.grid_data = create_radau_grid_data(self.num_segments, self.order)

        num_disc_nodes = self.grid_data['num_disc_nodes']
        num_all_nodes = self.grid_data['num_all_nodes']

        # Create simple design variables
        self.design_vars = {
            'states:x': jnp.linspace(0, 10, num_disc_nodes),
            'states:y': jnp.linspace(0, -10, num_disc_nodes),
            'states:v': jnp.linspace(0, 14, num_disc_nodes),
            'controls:theta': jnp.ones(num_all_nodes) * jnp.pi / 4,
            't_initial': 0.0,
            't_duration': 1.8,
            'x_initial': 0.0,
            'y_initial': 0.0,
            'v_initial': 0.0,
            'x_final': 10.0,
            'y_final': -10.0,
        }

        self.options = {
            'g': 9.80665,
            'enforce_initial': {'x': True, 'y': True, 'v': True},
            'enforce_final': {'x': True, 'y': True},
        }

    def test_basic_phase_assembly(self):
        """Test basic phase assembly runs without errors."""
        residuals = radau_brachistochrone_phase(
            self.design_vars, self.grid_data, self.options
        )

        # Check that all expected residuals are present
        self.assertIn('defect:x', residuals)
        self.assertIn('defect:y', residuals)
        self.assertIn('defect:v', residuals)
        self.assertIn('objective', residuals)

    def test_defect_shapes(self):
        """Test that defect arrays have correct shapes."""
        residuals = radau_brachistochrone_phase(
            self.design_vars, self.grid_data, self.options
        )

        num_col_nodes = self.grid_data['num_col_nodes']

        # Defects should have shape (num_col_nodes,)
        self.assertEqual(residuals['defect:x'].shape, (num_col_nodes,))
        self.assertEqual(residuals['defect:y'].shape, (num_col_nodes,))
        self.assertEqual(residuals['defect:v'].shape, (num_col_nodes,))

    def test_continuity_defects(self):
        """Test that continuity defects have correct shapes."""
        residuals = radau_brachistochrone_phase(
            self.design_vars, self.grid_data, self.options
        )

        # For 2 segments, should have 1 continuity constraint
        num_segments = self.num_segments
        expected_continuity_size = num_segments - 1

        if residuals['continuity:x'].size > 0:
            self.assertEqual(residuals['continuity:x'].shape[0], expected_continuity_size)
            self.assertEqual(residuals['continuity:y'].shape[0], expected_continuity_size)
            self.assertEqual(residuals['continuity:v'].shape[0], expected_continuity_size)

    def test_boundary_constraints(self):
        """Test that boundary constraints are created when enforced."""
        residuals = radau_brachistochrone_phase(
            self.design_vars, self.grid_data, self.options
        )

        # Initial constraints (enforced)
        self.assertIn('initial:x', residuals)
        self.assertIn('initial:y', residuals)
        self.assertIn('initial:v', residuals)

        # Final constraints (x and y enforced, v not)
        self.assertIn('final:x', residuals)
        self.assertIn('final:y', residuals)
        self.assertNotIn('final:v', residuals)

        # Check that initial constraints are approximately satisfied
        # (depends on design_vars setup)
        assert_allclose(residuals['initial:x'], 0.0, atol=0.1)
        assert_allclose(residuals['initial:y'], 0.0, atol=0.1)

    def test_objective_value(self):
        """Test that objective is the time duration."""
        residuals = radau_brachistochrone_phase(
            self.design_vars, self.grid_data, self.options
        )

        # Objective should equal t_duration
        assert_allclose(residuals['objective'], self.design_vars['t_duration'],
                       rtol=1e-14)

    def test_jit_compilation(self):
        """Verify phase function works with JAX JIT compilation."""
        # Need to mark static arguments for JIT
        # For now, just verify JIT works at all
        phase_jitted = jax.jit(
            lambda dv: radau_brachistochrone_phase(dv, self.grid_data, self.options)
        )

        # First call
        result1 = phase_jitted(self.design_vars)

        # Second call
        result2 = phase_jitted(self.design_vars)

        # Should be identical
        for key in result1:
            if isinstance(result1[key], jnp.ndarray) and result1[key].size > 0:
                assert_allclose(result1[key], result2[key], rtol=1e-14,
                               err_msg=f"Mismatch in {key}")

    def test_derivatives_wrt_states(self):
        """Verify JAX autodiff works for state variables."""
        # Define objective that depends on state variables
        def objective(x_disc):
            design_vars = self.design_vars.copy()
            design_vars['states:x'] = x_disc
            residuals = radau_brachistochrone_phase(
                design_vars, self.grid_data, self.options
            )
            # Sum of squared defects
            return jnp.sum(residuals['defect:x']**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(self.design_vars['states:x'])

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient wrt states should not be all zeros")

    def test_derivatives_wrt_controls(self):
        """Verify JAX autodiff works for control variables."""
        # Define objective that depends on controls
        def objective(theta):
            design_vars = self.design_vars.copy()
            design_vars['controls:theta'] = theta
            residuals = radau_brachistochrone_phase(
                design_vars, self.grid_data, self.options
            )
            # Sum of squared defects
            return jnp.sum(residuals['defect:v']**2)

        # Compute gradient
        grad_jax = jax.grad(objective)(self.design_vars['controls:theta'])

        # Gradient should not be all zeros
        self.assertTrue(jnp.any(grad_jax != 0),
                       "Gradient wrt controls should not be all zeros")

    def test_derivatives_wrt_time(self):
        """Verify JAX autodiff works for time variables."""
        # Define objective that depends on time duration
        def objective(t_dur):
            design_vars = self.design_vars.copy()
            design_vars['t_duration'] = t_dur
            residuals = radau_brachistochrone_phase(
                design_vars, self.grid_data, self.options
            )
            # Objective is time
            return residuals['objective']

        # Compute gradient
        grad_jax = jax.grad(objective)(self.design_vars['t_duration'])

        # Gradient should be 1.0 (objective = t_duration)
        assert_allclose(grad_jax, 1.0, rtol=1e-10)

    def test_grid_data_helper(self):
        """Test grid data creation helper function."""
        grid_data = create_radau_grid_data(num_segments=3, order=4)

        # Check expected keys
        required_keys = [
            'segment_ends', 'nodes_per_segment', 'node_stau',
            'disc_indices', 'col_indices', 'seg_end_indices',
            'D_matrix', 'A_matrix'
        ]
        for key in required_keys:
            self.assertIn(key, grid_data)

        # Check sizes
        self.assertEqual(len(grid_data['segment_ends']), 4)  # 3 segments + 1
        self.assertEqual(len(grid_data['nodes_per_segment']), 3)

        # Check matrix shapes
        num_col = grid_data['num_col_nodes']
        num_disc = grid_data['num_disc_nodes']
        self.assertEqual(grid_data['A_matrix'].shape, (num_col, num_disc))
        self.assertEqual(grid_data['D_matrix'].shape, (num_col, num_disc))

    def test_single_segment(self):
        """Test with single segment."""
        grid_data = create_radau_grid_data(num_segments=1, order=3)

        num_disc_nodes = grid_data['num_disc_nodes']
        num_all_nodes = grid_data['num_all_nodes']

        design_vars = {
            'states:x': jnp.linspace(0, 10, num_disc_nodes),
            'states:y': jnp.linspace(0, -10, num_disc_nodes),
            'states:v': jnp.linspace(0, 14, num_disc_nodes),
            'controls:theta': jnp.ones(num_all_nodes) * jnp.pi / 6,
            't_initial': 0.0,
            't_duration': 2.0,
        }

        options = {'g': 9.80665, 'enforce_initial': {}, 'enforce_final': {}}

        residuals = radau_brachistochrone_phase(design_vars, grid_data, options)

        # Should have defects
        self.assertEqual(residuals['defect:x'].shape, (grid_data['num_col_nodes'],))

        # Should have no continuity defects (single segment)
        self.assertEqual(residuals['continuity:x'].size, 0)

    def test_different_gravity(self):
        """Test with different gravitational acceleration."""
        options_moon = {
            'g': 9.80665 / 6.0,  # Moon gravity
            'enforce_initial': {},
            'enforce_final': {},
        }

        residuals_earth = radau_brachistochrone_phase(
            self.design_vars, self.grid_data, self.options
        )
        residuals_moon = radau_brachistochrone_phase(
            self.design_vars, self.grid_data, options_moon
        )

        # v defects should be different (affected by gravity)
        self.assertFalse(jnp.allclose(residuals_earth['defect:v'],
                                      residuals_moon['defect:v']),
                        "Gravity should affect velocity defects")

        # x and y defects might also differ due to coupling

    def test_zero_initial_velocity(self):
        """Test case starting from rest."""
        design_vars = self.design_vars.copy()
        design_vars['states:v'] = jnp.linspace(0, 14, self.grid_data['num_disc_nodes'])
        design_vars['v_initial'] = 0.0

        options = self.options.copy()
        options['enforce_initial']['v'] = True

        residuals = radau_brachistochrone_phase(design_vars, self.grid_data, options)

        # Initial velocity constraint should be satisfied
        assert_allclose(residuals['initial:v'], 0.0, atol=0.1)

    def test_phase_consistency(self):
        """Test that phase assembly is self-consistent."""
        residuals = radau_brachistochrone_phase(
            self.design_vars, self.grid_data, self.options
        )

        # All defects should be arrays
        self.assertIsInstance(residuals['defect:x'], jnp.ndarray)
        self.assertIsInstance(residuals['defect:y'], jnp.ndarray)
        self.assertIsInstance(residuals['defect:v'], jnp.ndarray)

        # Objective should be scalar
        self.assertTrue(jnp.isscalar(residuals['objective']) or
                       residuals['objective'].shape == ())


if __name__ == '__main__':
    unittest.main()
