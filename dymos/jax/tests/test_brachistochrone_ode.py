"""Test JAX brachistochrone ODE function."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dymos.jax.examples.brachistochrone_ode import (
    brachistochrone_ode,
    brachistochrone_ode_vectorized
)


class TestBrachistochroneODE(unittest.TestCase):
    """Test JAX brachistochrone ODE functions."""

    def test_basic_evaluation(self):
        """Test basic ODE evaluation at a single point."""
        x = 0.0
        y = 0.0
        v = 5.0
        theta = jnp.pi / 4  # 45 degrees
        g = 9.80665

        x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta, g)

        # At 45 degrees with v=5:
        # x_dot = 5 * sin(45°) ≈ 3.536
        # y_dot = 5 * cos(45°) ≈ 3.536
        # v_dot = 9.80665 * cos(45°) ≈ 6.933
        expected_x_dot = 5.0 * jnp.sin(jnp.pi / 4)
        expected_y_dot = 5.0 * jnp.cos(jnp.pi / 4)
        expected_v_dot = 9.80665 * jnp.cos(jnp.pi / 4)

        assert_allclose(x_dot, expected_x_dot, rtol=1e-10)
        assert_allclose(y_dot, expected_y_dot, rtol=1e-10)
        assert_allclose(v_dot, expected_v_dot, rtol=1e-10)

    def test_zero_velocity(self):
        """Test at rest (zero velocity)."""
        x = 0.0
        y = 0.0
        v = 0.0
        theta = jnp.pi / 6
        g = 9.80665

        x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta, g)

        # With v=0, position derivatives should be zero
        assert_allclose(x_dot, 0.0, atol=1e-14)
        assert_allclose(y_dot, 0.0, atol=1e-14)

        # Acceleration should still be non-zero
        expected_v_dot = g * jnp.cos(theta)
        assert_allclose(v_dot, expected_v_dot, rtol=1e-10)

    def test_horizontal_motion(self):
        """Test horizontal motion (theta = 0)."""
        x = 1.0
        y = 2.0
        v = 3.0
        theta = 0.0  # Horizontal
        g = 9.80665

        x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta, g)

        # At theta=0 (horizontal):
        # x_dot = v * sin(0) = 0
        # y_dot = v * cos(0) = v
        # v_dot = g * cos(0) = g
        assert_allclose(x_dot, 0.0, atol=1e-14)
        assert_allclose(y_dot, v, rtol=1e-10)
        assert_allclose(v_dot, g, rtol=1e-10)

    def test_vertical_motion(self):
        """Test vertical motion (theta = 90 degrees)."""
        x = 1.0
        y = 2.0
        v = 3.0
        theta = jnp.pi / 2  # Vertical
        g = 9.80665

        x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta, g)

        # At theta=90° (vertical):
        # x_dot = v * sin(90°) = v
        # y_dot = v * cos(90°) = 0
        # v_dot = g * cos(90°) = 0
        assert_allclose(x_dot, v, rtol=1e-10)
        assert_allclose(y_dot, 0.0, atol=1e-10)
        assert_allclose(v_dot, 0.0, atol=1e-10)

    def test_vectorized_evaluation(self):
        """Test vectorized evaluation over multiple nodes."""
        num_nodes = 10
        x = jnp.linspace(0, 10, num_nodes)
        y = jnp.linspace(0, -10, num_nodes)
        v = jnp.linspace(0, 14, num_nodes)
        theta = jnp.ones(num_nodes) * jnp.pi / 4

        x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta)

        # Check shapes
        self.assertEqual(x_dot.shape, (num_nodes,))
        self.assertEqual(y_dot.shape, (num_nodes,))
        self.assertEqual(v_dot.shape, (num_nodes,))

        # Check first point manually
        expected_x_dot_0 = v[0] * jnp.sin(theta[0])
        expected_y_dot_0 = v[0] * jnp.cos(theta[0])
        expected_v_dot_0 = 9.80665 * jnp.cos(theta[0])

        assert_allclose(x_dot[0], expected_x_dot_0, rtol=1e-10)
        assert_allclose(y_dot[0], expected_y_dot_0, rtol=1e-10)
        assert_allclose(v_dot[0], expected_v_dot_0, rtol=1e-10)

    def test_vectorized_interface_dict(self):
        """Test vectorized interface with dictionary inputs."""
        num_nodes = 5
        states = {
            'x': jnp.linspace(0, 5, num_nodes),
            'y': jnp.linspace(0, -5, num_nodes),
            'v': jnp.linspace(1, 10, num_nodes)
        }
        controls = {'theta': jnp.ones(num_nodes) * jnp.pi / 6}
        params = {'g': 9.80665}

        x_dot, y_dot, v_dot = brachistochrone_ode_vectorized(states, controls, params)

        # Verify shapes
        self.assertEqual(x_dot.shape, (num_nodes,))
        self.assertEqual(y_dot.shape, (num_nodes,))
        self.assertEqual(v_dot.shape, (num_nodes,))

        # Verify against direct call
        x_dot_direct, y_dot_direct, v_dot_direct = brachistochrone_ode(
            states['x'], states['y'], states['v'], controls['theta'], params['g']
        )

        assert_allclose(x_dot, x_dot_direct, rtol=1e-14)
        assert_allclose(y_dot, y_dot_direct, rtol=1e-14)
        assert_allclose(v_dot, v_dot_direct, rtol=1e-14)

    def test_vectorized_interface_tuple(self):
        """Test vectorized interface with tuple inputs."""
        num_nodes = 5
        x = jnp.linspace(0, 5, num_nodes)
        y = jnp.linspace(0, -5, num_nodes)
        v = jnp.linspace(1, 10, num_nodes)
        theta = jnp.ones(num_nodes) * jnp.pi / 3

        states = (x, y, v)
        controls = (theta,)

        x_dot, y_dot, v_dot = brachistochrone_ode_vectorized(states, controls)

        # Verify against direct call
        x_dot_direct, y_dot_direct, v_dot_direct = brachistochrone_ode(x, y, v, theta)

        assert_allclose(x_dot, x_dot_direct, rtol=1e-14)
        assert_allclose(y_dot, y_dot_direct, rtol=1e-14)
        assert_allclose(v_dot, v_dot_direct, rtol=1e-14)

    def test_jit_compilation(self):
        """Verify function works with JAX JIT compilation."""
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, -1.0, -2.0])
        v = jnp.array([0.0, 5.0, 10.0])
        theta = jnp.array([0.5, 0.6, 0.7])

        # JIT compile
        ode_jitted = jax.jit(brachistochrone_ode)

        # First call
        result1 = ode_jitted(x, y, v, theta)

        # Second call
        result2 = ode_jitted(x, y, v, theta)

        # Should be identical
        for r1, r2 in zip(result1, result2):
            assert_allclose(r1, r2, rtol=1e-14)

    def test_derivatives_wrt_states(self):
        """Verify JAX autodiff works for state derivatives."""
        x = 1.0
        y = 2.0
        v = 5.0
        theta = jnp.pi / 4

        # Define objective that depends on states
        def objective(v_val):
            x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v_val, theta)
            return x_dot**2 + y_dot**2 + v_dot**2

        # Compute gradient
        grad_jax = jax.grad(objective)(v)

        # Gradient should not be zero
        self.assertTrue(abs(grad_jax) > 1e-10,
                       "Gradient wrt velocity should be non-zero")

    def test_derivatives_wrt_controls(self):
        """Verify JAX autodiff works for control derivatives."""
        x = 1.0
        y = 2.0
        v = 5.0
        theta = jnp.pi / 4

        # Define objective that depends on control
        def objective(theta_val):
            x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta_val)
            return x_dot**2 + y_dot**2 + v_dot**2

        # Compute gradient
        grad_jax = jax.grad(objective)(theta)

        # Gradient should not be zero
        self.assertTrue(abs(grad_jax) > 1e-10,
                       "Gradient wrt theta should be non-zero")

    def test_vmap_compatibility(self):
        """Test vectorization with jax.vmap."""
        batch_size = 3
        x_batch = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        y_batch = jnp.array([[0.0, -1.0], [-2.0, -3.0], [-4.0, -5.0]])
        v_batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        theta_batch = jnp.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])

        # Vectorize over batch dimension
        ode_batched = jax.vmap(
            lambda x, y, v, theta: brachistochrone_ode(x, y, v, theta),
            in_axes=(0, 0, 0, 0)
        )

        results = ode_batched(x_batch, y_batch, v_batch, theta_batch)

        # Verify shapes
        self.assertEqual(results[0].shape, (batch_size, 2))
        self.assertEqual(results[1].shape, (batch_size, 2))
        self.assertEqual(results[2].shape, (batch_size, 2))

    def test_energy_conservation_check(self):
        """Verify energy relationship (not conserved, but check formulation)."""
        # For brachistochrone, mechanical energy is E = (1/2)*v^2 - g*y
        # dE/dt = v*v_dot - g*y_dot
        #       = v*(g*cos(theta)) - g*(v*cos(theta))
        #       = 0 (energy is conserved!)

        x = 1.0
        y = -3.0  # Below start
        v = 7.0
        theta = jnp.pi / 3
        g = 9.80665

        x_dot, y_dot, v_dot = brachistochrone_ode(x, y, v, theta, g)

        # Compute dE/dt
        dE_dt = v * v_dot - g * y_dot

        # Should be approximately zero (energy conservation)
        assert_allclose(dE_dt, 0.0, atol=1e-10,
                       err_msg="Energy should be conserved in brachistochrone")

    def test_different_gravity(self):
        """Test with different gravitational acceleration."""
        x = 0.0
        y = 0.0
        v = 1.0
        theta = jnp.pi / 4

        # Earth gravity
        g_earth = 9.80665
        x_dot_earth, y_dot_earth, v_dot_earth = brachistochrone_ode(x, y, v, theta, g_earth)

        # Moon gravity (approx 1/6 of Earth)
        g_moon = 9.80665 / 6.0
        x_dot_moon, y_dot_moon, v_dot_moon = brachistochrone_ode(x, y, v, theta, g_moon)

        # Position derivatives should be same (depend only on v and theta)
        assert_allclose(x_dot_earth, x_dot_moon, rtol=1e-14)
        assert_allclose(y_dot_earth, y_dot_moon, rtol=1e-14)

        # Acceleration should scale with gravity
        assert_allclose(v_dot_moon, v_dot_earth / 6.0, rtol=1e-10)

    def test_position_independence(self):
        """Verify that derivatives don't depend on absolute position."""
        # The brachistochrone ODE doesn't depend on x or y values
        v = 5.0
        theta = jnp.pi / 4

        # Different positions
        x_dot_1, y_dot_1, v_dot_1 = brachistochrone_ode(0.0, 0.0, v, theta)
        x_dot_2, y_dot_2, v_dot_2 = brachistochrone_ode(10.0, -5.0, v, theta)

        # Derivatives should be identical
        assert_allclose(x_dot_1, x_dot_2, rtol=1e-14)
        assert_allclose(y_dot_1, y_dot_2, rtol=1e-14)
        assert_allclose(v_dot_1, v_dot_2, rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
