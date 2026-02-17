"""Tests for the JAX Radau pseudospectral backend."""
import json
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from dymos.jax.specs import (
    PhaseSpec, StateSpec, ControlSpec, ParameterSpec, TimeSpec, GridSpec
)
from dymos.jax.backend import create_jax_radau_phase
from dymos.jax.backend.radau_backend import _validate_spec
from dymos.jax.examples.brachistochrone_ode_dict import brachistochrone_ode_dict
from dymos.jax.examples.radau_brachistochrone_phase import (
    radau_brachistochrone_phase, create_radau_grid_data
)


def _make_brachistochrone_spec(num_segments=2, order=3):
    """Create a brachistochrone PhaseSpec for testing."""
    return PhaseSpec(
        states=[
            StateSpec(name='x', fix_initial=True, fix_final=True),
            StateSpec(name='y', fix_initial=True, fix_final=True),
            StateSpec(name='v', fix_initial=True),
        ],
        controls=[ControlSpec(name='theta')],
        parameters=[ParameterSpec(name='g', value=9.80665)],
        time=TimeSpec(
            fix_initial=True,
            fix_duration=False,
            initial_value=0.0,
            duration_bounds=(0.5, 10.0),
        ),
        grid=GridSpec(num_segments=num_segments, order=order, transcription='radau'),
    )


def _make_design_vars(grid_data, t_initial=0.0, t_duration=1.8):
    """Build spec-based design variables."""
    num_disc_nodes = grid_data['num_disc_nodes']
    num_all_nodes = grid_data['num_all_nodes']
    return {
        'states': {
            'x': jnp.linspace(0.0, 10.0, num_disc_nodes),
            'y': jnp.linspace(0.0, -10.0, num_disc_nodes),
            'v': jnp.linspace(0.0, 14.0, num_disc_nodes),
        },
        'controls': {
            'theta': jnp.ones(num_all_nodes) * jnp.pi / 4,
        },
        'time': {
            'initial': t_initial,
            'duration': t_duration,
        },
        'x_initial': 0.0,
        'y_initial': 0.0,
        'v_initial': 0.0,
        'x_final': 10.0,
        'y_final': -10.0,
    }


def _make_legacy_design_vars(grid_data, t_initial=0.0, t_duration=1.8):
    """Build legacy flat design variables for comparison."""
    num_disc_nodes = grid_data['num_disc_nodes']
    num_all_nodes = grid_data['num_all_nodes']
    return {
        'states:x': jnp.linspace(0.0, 10.0, num_disc_nodes),
        'states:y': jnp.linspace(0.0, -10.0, num_disc_nodes),
        'states:v': jnp.linspace(0.0, 14.0, num_disc_nodes),
        'controls:theta': jnp.ones(num_all_nodes) * jnp.pi / 4,
        't_initial': t_initial,
        't_duration': t_duration,
        'x_initial': 0.0,
        'y_initial': 0.0,
        'v_initial': 0.0,
        'x_final': 10.0,
        'y_final': -10.0,
    }


class TestValidateSpec(unittest.TestCase):
    """Test _validate_spec helper."""

    def test_valid_spec_passes(self):
        spec = _make_brachistochrone_spec()
        _validate_spec(spec)  # Should not raise

    def test_no_states_raises(self):
        spec = PhaseSpec(
            states=[],
            time=TimeSpec(fix_initial=True, duration_bounds=(0.5, 10.0)),
            grid=GridSpec(num_segments=2),
        )
        with self.assertRaises(ValueError, msg="PhaseSpec must have at least one state"):
            _validate_spec(spec)

    def test_fix_duration_no_value_raises(self):
        spec = PhaseSpec(
            states=[StateSpec(name='x')],
            time=TimeSpec(fix_initial=True, fix_duration=True, duration_value=None),
            grid=GridSpec(num_segments=2),
        )
        with self.assertRaises(ValueError):
            _validate_spec(spec)

    def test_free_duration_no_bounds_raises(self):
        spec = PhaseSpec(
            states=[StateSpec(name='x')],
            time=TimeSpec(fix_initial=True, fix_duration=False, duration_bounds=None),
            grid=GridSpec(num_segments=2),
        )
        with self.assertRaises(ValueError):
            _validate_spec(spec)


class TestCreateJaxRadauPhase(unittest.TestCase):
    """Test create_jax_radau_phase factory function."""

    def setUp(self):
        self.num_segments = 2
        self.order = 3
        self.spec = _make_brachistochrone_spec(self.num_segments, self.order)
        self.grid_data = create_radau_grid_data(self.num_segments, self.order)
        self.design_vars = _make_design_vars(self.grid_data)
        self.phase_fn = create_jax_radau_phase(self.spec, brachistochrone_ode_dict)

    def test_returns_callable(self):
        self.assertTrue(callable(self.phase_fn))

    def test_basic_evaluation(self):
        residuals = self.phase_fn(self.design_vars)
        self.assertIsInstance(residuals, dict)

    def test_residual_keys(self):
        residuals = self.phase_fn(self.design_vars)
        for state_name in ['x', 'y', 'v']:
            self.assertIn(f'defect:{state_name}', residuals)
            self.assertIn(f'continuity:{state_name}', residuals)
        self.assertIn('objective', residuals)

    def test_boundary_constraint_keys(self):
        residuals = self.phase_fn(self.design_vars)
        # fix_initial=True for x, y, v
        self.assertIn('initial:x', residuals)
        self.assertIn('initial:y', residuals)
        self.assertIn('initial:v', residuals)
        # fix_final=True for x, y only
        self.assertIn('final:x', residuals)
        self.assertIn('final:y', residuals)
        # v has fix_final=False
        self.assertNotIn('final:v', residuals)

    def test_defect_shapes(self):
        residuals = self.phase_fn(self.design_vars)
        num_col_nodes = self.grid_data['num_col_nodes']
        for state_name in ['x', 'y', 'v']:
            self.assertEqual(residuals[f'defect:{state_name}'].shape, (num_col_nodes,))

    def test_continuity_shapes(self):
        residuals = self.phase_fn(self.design_vars)
        expected_size = self.num_segments - 1  # 1 continuity per interior boundary
        for state_name in ['x', 'y', 'v']:
            key = f'continuity:{state_name}'
            if residuals[key].size > 0:
                self.assertEqual(residuals[key].shape[0], expected_size)

    def test_objective_equals_duration(self):
        residuals = self.phase_fn(self.design_vars)
        t_duration = self.design_vars['time']['duration']
        assert_allclose(float(residuals['objective']), t_duration, rtol=1e-14)

    def test_initial_boundary_near_target(self):
        residuals = self.phase_fn(self.design_vars)
        # x_disc[0] = 0.0, x_initial target = 0.0 → residual = 0
        assert_allclose(float(residuals['initial:x']), 0.0, atol=1e-14)
        assert_allclose(float(residuals['initial:y']), 0.0, atol=1e-14)
        assert_allclose(float(residuals['initial:v']), 0.0, atol=1e-14)

    def test_defects_match_legacy(self):
        """Spec-based defects must match the legacy radau_brachistochrone_phase."""
        legacy_dv = _make_legacy_design_vars(self.grid_data)
        legacy_options = {
            'g': 9.80665,
            'enforce_initial': {'x': True, 'y': True, 'v': True},
            'enforce_final': {'x': True, 'y': True},
        }
        legacy_residuals = radau_brachistochrone_phase(
            legacy_dv, self.grid_data, legacy_options
        )

        new_residuals = self.phase_fn(self.design_vars)

        for state_name in ['x', 'y', 'v']:
            key = f'defect:{state_name}'
            assert_allclose(
                np.array(new_residuals[key]),
                np.array(legacy_residuals[key]),
                rtol=1e-10,
                err_msg=f"Mismatch in {key} between spec-based and legacy"
            )

    def test_continuity_matches_legacy(self):
        """Spec-based continuity defects must match the legacy implementation."""
        legacy_dv = _make_legacy_design_vars(self.grid_data)
        legacy_options = {
            'g': 9.80665,
            'enforce_initial': {},
            'enforce_final': {},
        }
        legacy_residuals = radau_brachistochrone_phase(
            legacy_dv, self.grid_data, legacy_options
        )
        new_residuals = self.phase_fn(self.design_vars)

        for state_name in ['x', 'y', 'v']:
            key = f'continuity:{state_name}'
            if new_residuals[key].size > 0 and legacy_residuals[key].size > 0:
                assert_allclose(
                    np.array(new_residuals[key]),
                    np.array(legacy_residuals[key]),
                    rtol=1e-10,
                    err_msg=f"Mismatch in {key} between spec-based and legacy"
                )


class TestJITCompilation(unittest.TestCase):
    """Test JIT compilation of spec-based phase function."""

    def setUp(self):
        spec = _make_brachistochrone_spec(2, 3)
        self.phase_fn = create_jax_radau_phase(spec, brachistochrone_ode_dict)
        grid_data = create_radau_grid_data(2, 3)
        self.design_vars = _make_design_vars(grid_data)

    def test_jit_compiles(self):
        phase_jitted = jax.jit(self.phase_fn)
        residuals = phase_jitted(self.design_vars)
        self.assertIn('objective', residuals)

    def test_jit_consistent_results(self):
        phase_jitted = jax.jit(self.phase_fn)
        r1 = phase_jitted(self.design_vars)
        r2 = phase_jitted(self.design_vars)
        for key in r1:
            if hasattr(r1[key], 'shape') and r1[key].size > 0:
                assert_allclose(r1[key], r2[key], rtol=1e-14,
                                err_msg=f"JIT inconsistency in {key}")


class TestGradients(unittest.TestCase):
    """Test gradient computation through spec-based phase function."""

    def setUp(self):
        spec = _make_brachistochrone_spec(2, 3)
        self.phase_fn = create_jax_radau_phase(spec, brachistochrone_ode_dict)
        grid_data = create_radau_grid_data(2, 3)
        self.design_vars = _make_design_vars(grid_data)

    def test_grad_wrt_duration(self):
        def objective(dv):
            return self.phase_fn(dv)['objective']

        grad_fn = jax.grad(objective)
        grads = grad_fn(self.design_vars)
        # d(t_duration)/d(t_duration) = 1
        assert_allclose(float(grads['time']['duration']), 1.0, rtol=1e-10)

    def test_grad_structure_matches_design_vars(self):
        def objective(dv):
            return jnp.sum(self.phase_fn(dv)['defect:x'] ** 2)

        grad_fn = jax.grad(objective)
        grads = grad_fn(self.design_vars)
        # Gradient dict should have same top-level keys
        self.assertEqual(set(grads.keys()), set(self.design_vars.keys()))

    def test_grad_wrt_states_nonzero(self):
        def objective(dv):
            return jnp.sum(self.phase_fn(dv)['defect:v'] ** 2)

        grad_fn = jax.grad(objective)
        grads = grad_fn(self.design_vars)
        # At least some state gradients should be non-zero
        any_nonzero = any(
            jnp.any(grads['states'][name] != 0)
            for name in ['x', 'y', 'v']
        )
        self.assertTrue(any_nonzero, "Gradients wrt states should not all be zero")

    def test_grad_wrt_controls_nonzero(self):
        def objective(dv):
            return jnp.sum(self.phase_fn(dv)['defect:x'] ** 2)

        grad_fn = jax.grad(objective)
        grads = grad_fn(self.design_vars)
        self.assertTrue(
            jnp.any(grads['controls']['theta'] != 0),
            "Gradients wrt controls should not all be zero"
        )


class TestDifferentGridConfigs(unittest.TestCase):
    """Test backend with various grid configurations."""

    def _run_phase(self, num_segments, order):
        spec = _make_brachistochrone_spec(num_segments, order)
        phase_fn = create_jax_radau_phase(spec, brachistochrone_ode_dict)
        grid_data = create_radau_grid_data(num_segments, order)
        dv = _make_design_vars(grid_data)
        return phase_fn(dv), grid_data

    def test_single_segment(self):
        residuals, grid_data = self._run_phase(1, 3)
        self.assertEqual(residuals['defect:x'].shape, (grid_data['num_col_nodes'],))
        self.assertEqual(residuals['continuity:x'].size, 0)

    def test_three_segments(self):
        residuals, grid_data = self._run_phase(3, 3)
        self.assertEqual(residuals['defect:x'].shape, (grid_data['num_col_nodes'],))

    def test_order_2(self):
        residuals, grid_data = self._run_phase(2, 2)
        self.assertEqual(residuals['defect:x'].shape, (grid_data['num_col_nodes'],))

    def test_order_4(self):
        residuals, grid_data = self._run_phase(2, 4)
        self.assertEqual(residuals['defect:x'].shape, (grid_data['num_col_nodes'],))


class TestSerialization(unittest.TestCase):
    """Test that spec serialization produces identical phase functions."""

    def test_json_roundtrip_same_residuals(self):
        spec = _make_brachistochrone_spec(2, 3)
        grid_data = create_radau_grid_data(2, 3)
        dv = _make_design_vars(grid_data)

        phase_fn_orig = create_jax_radau_phase(spec, brachistochrone_ode_dict)
        residuals_orig = phase_fn_orig(dv)

        # Serialize and reload
        spec_dict = spec.model_dump()
        spec_loaded = PhaseSpec(**json.loads(json.dumps(spec_dict)))
        phase_fn_loaded = create_jax_radau_phase(spec_loaded, brachistochrone_ode_dict)
        residuals_loaded = phase_fn_loaded(dv)

        assert_allclose(
            float(residuals_orig['objective']),
            float(residuals_loaded['objective']),
            rtol=1e-14
        )
        for state_name in ['x', 'y', 'v']:
            key = f'defect:{state_name}'
            assert_allclose(
                np.array(residuals_orig[key]),
                np.array(residuals_loaded[key]),
                rtol=1e-14,
                err_msg=f"Mismatch in {key} after JSON roundtrip"
            )


class TestParameterHandling(unittest.TestCase):
    """Test that parameters are correctly passed to the ODE."""

    def test_gravity_affects_v_defect(self):
        """Different gravity values should produce different v defects."""
        def make_phase(g_value):
            spec = PhaseSpec(
                states=[
                    StateSpec(name='x'),
                    StateSpec(name='y'),
                    StateSpec(name='v'),
                ],
                controls=[ControlSpec(name='theta')],
                parameters=[ParameterSpec(name='g', value=g_value)],
                time=TimeSpec(fix_initial=True, fix_duration=False,
                              duration_bounds=(0.5, 10.0)),
                grid=GridSpec(num_segments=2, order=3),
            )
            return create_jax_radau_phase(spec, brachistochrone_ode_dict)

        grid_data = create_radau_grid_data(2, 3)
        dv = _make_design_vars(grid_data)

        phase_earth = make_phase(9.80665)
        phase_moon = make_phase(9.80665 / 6.0)

        r_earth = phase_earth(dv)
        r_moon = phase_moon(dv)

        self.assertFalse(
            jnp.allclose(r_earth['defect:v'], r_moon['defect:v']),
            "Different gravity should produce different v defects"
        )


if __name__ == '__main__':
    unittest.main()
