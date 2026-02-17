"""Tests for Pydantic phase specification models."""
import json
import unittest

from dymos.jax.specs import (
    PhaseSpec, StateSpec, ControlSpec, ParameterSpec, TimeSpec, GridSpec
)


class TestStateSpec(unittest.TestCase):
    """Test StateSpec model."""

    def test_basic_creation(self):
        s = StateSpec(name='x')
        self.assertEqual(s.name, 'x')
        self.assertFalse(s.fix_initial)
        self.assertFalse(s.fix_final)

    def test_rate_source_default(self):
        s = StateSpec(name='x')
        self.assertEqual(s.get_rate_source(), 'x_dot')

    def test_rate_source_explicit(self):
        s = StateSpec(name='x', rate_source='xdot')
        self.assertEqual(s.get_rate_source(), 'xdot')

    def test_targets_default(self):
        s = StateSpec(name='x')
        self.assertEqual(s.get_targets(), ['x'])

    def test_targets_explicit(self):
        s = StateSpec(name='x', targets=['x_val', 'x_ref'])
        self.assertEqual(s.get_targets(), ['x_val', 'x_ref'])

    def test_fix_flags(self):
        s = StateSpec(name='v', fix_initial=True, fix_final=False)
        self.assertTrue(s.fix_initial)
        self.assertFalse(s.fix_final)

    def test_units_optional(self):
        s = StateSpec(name='x', units='m')
        self.assertEqual(s.units, 'm')

    def test_invalid_name_raises(self):
        with self.assertRaises(Exception):
            StateSpec()  # name is required

    def test_json_roundtrip(self):
        s = StateSpec(name='x', fix_initial=True, units='m')
        d = s.model_dump()
        s2 = StateSpec(**d)
        self.assertEqual(s.name, s2.name)
        self.assertEqual(s.fix_initial, s2.fix_initial)
        self.assertEqual(s.units, s2.units)


class TestControlSpec(unittest.TestCase):
    """Test ControlSpec model."""

    def test_basic_creation(self):
        c = ControlSpec(name='theta')
        self.assertEqual(c.name, 'theta')

    def test_targets_default(self):
        c = ControlSpec(name='theta')
        self.assertEqual(c.get_targets(), ['theta'])

    def test_targets_explicit(self):
        c = ControlSpec(name='theta', targets=['theta_val'])
        self.assertEqual(c.get_targets(), ['theta_val'])

    def test_bounds(self):
        c = ControlSpec(name='theta', lower=-1.0, upper=1.0)
        self.assertEqual(c.lower, -1.0)
        self.assertEqual(c.upper, 1.0)

    def test_json_roundtrip(self):
        c = ControlSpec(name='theta', lower=-1.0, upper=1.0)
        c2 = ControlSpec(**c.model_dump())
        self.assertEqual(c.name, c2.name)
        self.assertEqual(c.lower, c2.lower)


class TestParameterSpec(unittest.TestCase):
    """Test ParameterSpec model."""

    def test_basic_creation(self):
        p = ParameterSpec(name='g', value=9.80665)
        self.assertEqual(p.name, 'g')
        self.assertAlmostEqual(p.value, 9.80665)
        self.assertTrue(p.static)

    def test_targets_default(self):
        p = ParameterSpec(name='g', value=9.80665)
        self.assertEqual(p.get_targets(), ['g'])

    def test_opt_parameter(self):
        p = ParameterSpec(name='mass', value=1.0, opt=True, lower=0.1, upper=10.0)
        self.assertTrue(p.opt)
        self.assertEqual(p.lower, 0.1)

    def test_json_roundtrip(self):
        p = ParameterSpec(name='g', value=9.80665, static=True)
        p2 = ParameterSpec(**p.model_dump())
        self.assertEqual(p.name, p2.name)
        self.assertAlmostEqual(p.value, p2.value)


class TestTimeSpec(unittest.TestCase):
    """Test TimeSpec model."""

    def test_basic_creation(self):
        t = TimeSpec(fix_initial=True, fix_duration=False, duration_bounds=(0.5, 10.0))
        self.assertTrue(t.fix_initial)
        self.assertFalse(t.fix_duration)

    def test_targets_default(self):
        t = TimeSpec(fix_initial=True, duration_bounds=(0.5, 10.0))
        self.assertEqual(t.get_targets(), ['time'])

    def test_targets_explicit(self):
        t = TimeSpec(fix_initial=True, duration_bounds=(0.5, 10.0), targets=['t'])
        self.assertEqual(t.get_targets(), ['t'])

    def test_fixed_duration(self):
        t = TimeSpec(fix_initial=True, fix_duration=True, duration_value=2.0)
        self.assertTrue(t.fix_duration)
        self.assertEqual(t.duration_value, 2.0)

    def test_json_roundtrip(self):
        t = TimeSpec(fix_initial=True, fix_duration=False, duration_bounds=(0.5, 10.0))
        t2 = TimeSpec(**t.model_dump())
        self.assertEqual(t.fix_initial, t2.fix_initial)
        self.assertEqual(t.duration_bounds, t2.duration_bounds)


class TestGridSpec(unittest.TestCase):
    """Test GridSpec model."""

    def test_basic_creation(self):
        g = GridSpec(num_segments=3, order=3)
        self.assertEqual(g.num_segments, 3)
        self.assertEqual(g.order, 3)
        self.assertEqual(g.transcription, 'radau')

    def test_num_segments_validation(self):
        with self.assertRaises(Exception):
            GridSpec(num_segments=0, order=3)  # ge=1

    def test_order_validation(self):
        with self.assertRaises(Exception):
            GridSpec(num_segments=3, order=0)  # ge=1

    def test_json_roundtrip(self):
        g = GridSpec(num_segments=5, order=4, transcription='radau')
        g2 = GridSpec(**g.model_dump())
        self.assertEqual(g.num_segments, g2.num_segments)
        self.assertEqual(g.order, g2.order)


class TestPhaseSpec(unittest.TestCase):
    """Test PhaseSpec model."""

    def _make_spec(self, num_segments=2, order=3):
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
                duration_bounds=(0.5, 10.0),
            ),
            grid=GridSpec(num_segments=num_segments, order=order),
        )

    def test_basic_creation(self):
        spec = self._make_spec()
        self.assertEqual(len(spec.states), 3)
        self.assertEqual(len(spec.controls), 1)
        self.assertEqual(len(spec.parameters), 1)

    def test_get_names(self):
        spec = self._make_spec()
        self.assertEqual(spec.get_state_names(), ['x', 'y', 'v'])
        self.assertEqual(spec.get_control_names(), ['theta'])
        self.assertEqual(spec.get_parameter_names(), ['g'])

    def test_no_controls(self):
        spec = PhaseSpec(
            states=[StateSpec(name='x')],
            time=TimeSpec(fix_initial=True, duration_bounds=(0.5, 10.0)),
            grid=GridSpec(num_segments=2),
        )
        self.assertEqual(spec.get_control_names(), [])

    def test_no_parameters(self):
        spec = PhaseSpec(
            states=[StateSpec(name='x')],
            time=TimeSpec(fix_initial=True, duration_bounds=(0.5, 10.0)),
            grid=GridSpec(num_segments=2),
        )
        self.assertEqual(spec.get_parameter_names(), [])

    def test_states_required(self):
        with self.assertRaises(Exception):
            PhaseSpec(
                time=TimeSpec(fix_initial=True, duration_bounds=(0.5, 10.0)),
                grid=GridSpec(num_segments=2),
            )

    def test_time_required(self):
        with self.assertRaises(Exception):
            PhaseSpec(
                states=[StateSpec(name='x')],
                grid=GridSpec(num_segments=2),
            )

    def test_grid_required(self):
        with self.assertRaises(Exception):
            PhaseSpec(
                states=[StateSpec(name='x')],
                time=TimeSpec(fix_initial=True, duration_bounds=(0.5, 10.0)),
            )

    def test_json_roundtrip(self):
        spec = self._make_spec()
        spec_dict = spec.model_dump()
        spec_json = json.dumps(spec_dict)
        spec_loaded = PhaseSpec(**json.loads(spec_json))

        self.assertEqual(spec.get_state_names(), spec_loaded.get_state_names())
        self.assertEqual(spec.get_control_names(), spec_loaded.get_control_names())
        self.assertEqual(spec.get_parameter_names(), spec_loaded.get_parameter_names())
        self.assertEqual(spec.grid.num_segments, spec_loaded.grid.num_segments)
        self.assertEqual(spec.grid.order, spec_loaded.grid.order)

    def test_state_rate_sources(self):
        spec = self._make_spec()
        rate_sources = [s.get_rate_source() for s in spec.states]
        self.assertEqual(rate_sources, ['x_dot', 'y_dot', 'v_dot'])

    def test_objective_default(self):
        spec = self._make_spec()
        self.assertEqual(spec.objective, 'time')

    def test_different_grid_configs(self):
        for num_seg, order in [(1, 2), (3, 3), (5, 4), (10, 2)]:
            spec = self._make_spec(num_segments=num_seg, order=order)
            self.assertEqual(spec.grid.num_segments, num_seg)
            self.assertEqual(spec.grid.order, order)


if __name__ == '__main__':
    unittest.main()
