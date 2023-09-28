import time
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
import dymos as dm

from dymos.transcriptions.explicit_shooting.vandermonde_control_interp_comp import VandermondeControlInterpComp
from dymos.transcriptions.explicit_shooting.barycentric_lagrange_interp_comp import BarycentricLagrangeInterpComp
from dymos.transcriptions.common.control_group import ControlGroup

_TOL = 1.0E-8


class TestGroup(om.Group):

    def __init__(self, grid_data, control_options, time_units, **kwargs):
        self._control_options = control_options
        self._grid_data = grid_data
        self._time_units = time_units
        super().__init__(**kwargs)

    def setup(self):

        self.add_subsystem('control_group',
                           ControlGroup(grid_data=self._grid_data,
                                        control_options=self._control_options,
                                        time_units=self._time_units),
                           promotes=['*'])

        self.add_subsystem('barycentric_interp_comp',
                           BarycentricLagrangeInterpComp(grid_data=self._grid_data),
                           promotes=['*'])

    def configure(self):
        self._get_subsystem('control_group').configure_io()
        interp_comp = self._get_subsystem('barycentric_interp_comp')

        for control_name, options in self._control_options.items():
            interp_comp.add_interp(control_name,
                                   input_name=f'control_values:{control_name}',
                                   output_name=control_name,
                                   shape=options['shape'],
                                   units=options['units'])

            interp_comp.add_interp(f'{control_name}_rate',
                                   input_name=f'control_rates:{control_name}_rate',
                                   output_name=f'{control_name}_rate',
                                   shape=options['shape'],
                                   units=options['units'] + '/s')


            interp_comp.add_interp(f'{control_name}_rate2',
                                   input_name=f'control_rates:{control_name}_rate2',
                                   output_name=f'{control_name}_rate2',
                                   shape=options['shape'],
                                   units=options['units'] + '/s**2')

        interp_comp.configure_io()



class TestBarycentricLagrangeInterpComp(unittest.TestCase):

    def test_scalar_interp(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()

        g = p.model.add_subsystem('test_group',
                                  TestGroup(grid_data=grid_data,
                                            control_options=control_options,
                                            time_units='s'))

        p.setup(force_alloc_complex=True)

        p.set_val('test_group.controls:u1',
                  val=np.asarray([[9.26636810e-02, 2.57652082e+01, 4.95555398e+01,
                                   5.82728007e+01, 7.57893300e+01, 9.18440884e+01, 1.01007525e+02]]),
                  units='deg')

        interp_comp = g._get_subsystem('barycentric_interp_comp')

        ptau_results = []
        u1_results = []
        u1_rate_results = []
        u1_rate2_results = []

        for seg_idx in range(2):
            interp_comp.set_segment_index(seg_idx)
            for stau in np.linspace(-1, 1, 1000):
                if seg_idx == 0:
                    ptau_results.append((stau - 1.0) / 2.)
                elif seg_idx == 1:
                    ptau_results.append((stau + 1.0) / 2.)
                p.set_val('test_group.stau', stau)
                p.run_model()
                u1_results.extend(p.get_val('test_group.u1')[0])
                u1_rate_results.extend(p.get_val('test_group.u1_rate')[0])
                u1_rate2_results.extend(p.get_val('test_group.u1_rate2')[0])

        p.set_val('test_group.stau', 0.5356)
        p.run_model()

        cpd = p.check_partials(compact_print=False, method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_scalar_interp_large_num_nodes(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[50, 200], compressed=False)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'unitless'

        p = om.Problem()

        g = p.model.add_subsystem('test_group',
                                  TestGroup(grid_data=grid_data,
                                            control_options=control_options,
                                            time_units='s'))

        p.setup(force_alloc_complex=True)

        # om.n2(p)

        t = 2 * np.pi * (grid_data.node_ptau + 1)
        x = np.sin(t)

        p.set_val('test_group.dt_dstau', np.pi)

        p.set_val('test_group.controls:u1',
                  val=x,
                  units='unitless')

        interp_comp = g._get_subsystem('barycentric_interp_comp')

        ptau_results = []
        u1_results = []
        u1_rate_results = []
        u1_rate2_results = []

        t0 = time.time_ns()
        for seg_idx in range(2):
            interp_comp.set_segment_index(seg_idx)
            for stau in np.linspace(-1, 1, 1000):
                if seg_idx == 0:
                    ptau_results.append((stau - 1.0) / 2.)
                elif seg_idx == 1:
                    ptau_results.append((stau + 1.0) / 2.)
                p.set_val('test_group.stau', stau)
                p.run_model()
                u1_results.extend(p.get_val('test_group.u1')[0])
                u1_rate_results.extend(p.get_val('test_group.u1_rate')[0])
                u1_rate2_results.extend(p.get_val('test_group.u1_rate2')[0])
        tf = time.time_ns()
        print((tf - t0) / 1.0E9)

        # import matplotlib.pyplot as plt
        # plt.plot(grid_data.node_ptau, p.get_val('test_group.control_values:u1'), 'o', ms=3)
        # plt.plot(ptau_results, u1_results, '-')
        #
        # plt.plot(grid_data.node_ptau, p.get_val('test_group.control_rates:u1_rate'), 'o', ms=3)
        # plt.plot(ptau_results, u1_rate_results, '-')
        #
        # plt.plot(grid_data.node_ptau, p.get_val('test_group.control_rates:u1_rate2'), 'o', ms=3)
        # plt.plot(ptau_results, u1_rate2_results, '-')
        # plt.show()
        #
        # p.set_val('test_group.stau', 0.5356)
        # p.run_model()

        assert_near_equal(u1_results, np.sin(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-12)
        assert_near_equal(u1_rate_results, np.cos(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-9)
        assert_near_equal(u1_rate2_results, -np.sin(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-9)

        t0 = time.time_ns()
        cpd = p.check_partials(compact_print=False, method='cs', out_stream=None)
        tf = time.time_ns()
        print((tf - t0) / 1.0E9)

        assert_check_partials(cpd)


    def test_temp(self):
        n = 4
        print(np.prod([0.]))
        import itertools
        idxs = list(itertools.combinations(range(n)[::-1], n - 1))
        print(idxs)
        for tup in idxs:
            idxs2 = list(itertools.combinations(tup, n - 2))
            print(idxs2)
#
#         interp_comp.options['segment_index'] = 1
#
#         p.set_val('interp.controls:u1', [[0.0, 3.0, 0.0, 4.0, 3.0, 4.0, 3.0]])
#
#         p.set_val('interp.stau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), np.zeros((1, 1)), tolerance=_TOL)
#
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#         p.set_val('interp.stau', 0.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=True, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#         p.set_val('interp.stau', 1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=True, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#         interp_comp.options['segment_index'] = 0
#
#         p.set_val('interp.stau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=True, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#         p.set_val('interp.stau', 0.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=True, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#         p.set_val('interp.stau', 1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#         p.set_val('interp.stau', 0.54262)
#         p.run_model()
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#     def test_eval_control_radau_compressed(self):
#         grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
#                                                          transcription_order=[3, 5], compressed=True)
#
#         time_options = dm.phase.options.TimeOptionsDictionary()
#
#         time_options['units'] = 's'
#
#         control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}
#
#         control_options['u1']['shape'] = (1,)
#         control_options['u1']['units'] = 'rad'
#
#         p = om.Problem()
#         interp_comp = p.model.add_subsystem('interp',
#                                             VandermondeControlInterpComp(grid_data=grid_data,
#                                                                          control_options=control_options,
#                                                                          standalone_mode=True,
#                                                                          time_units='s'))
#         p.setup(force_alloc_complex=True)
#
#         interp_comp.options['segment_index'] = 1
#         p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])
#
#         p.set_val('interp.stau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.stau', -0.72048)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', -0.167181)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', 0.446314)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', 0.885792)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)
#
#         interp_comp.options['segment_index'] = 0
#
#         p.set_val('interp.stau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', -0.28989795)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', 0.68989795)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[1.5]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', 0.54262)
#         p.run_model()
#
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd)
#
#     def test_eval_control_gl_uncompressed(self):
#         grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
#                                                          transcription_order=[3, 5], compressed=False)
#
#         time_options = dm.phase.options.TimeOptionsDictionary()
#
#         time_options['units'] = 's'
#
#         control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}
#
#         control_options['u1']['shape'] = (1,)
#         control_options['u1']['units'] = 'rad'
#
#         p = om.Problem()
#         interp_comp = p.model.add_subsystem('interp',
#                                             VandermondeControlInterpComp(grid_data=grid_data,
#                                                                          control_options=control_options,
#                                                                          standalone_mode=True,
#                                                                          time_units='s'))
#         p.setup(force_alloc_complex=True)
#
#         interp_comp.options['segment_index'] = 1
#         p.set_val('interp.controls:u1', [0.0, 3.0, 0.0, 0.0, 4.0, 3.0, 4.0, 3.0])
#
#         p.set_val('interp.stau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.stau', 0.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
#
#         p.set_val('interp.stau', 1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
#
#         interp_comp.options['segment_index'] = 0
#
#         p.set_val('interp.stau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.stau', 0.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
#
#         p.set_val('interp.stau', 1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.stau', 0.54262)
#         p.run_model()
#
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#     def test_eval_control_radau_uncompressed(self):
#         grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
#                                                          transcription_order=[3, 5], compressed=False)
#
#         time_options = dm.phase.options.TimeOptionsDictionary()
#
#         time_options['units'] = 's'
#
#         control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}
#
#         control_options['u1']['shape'] = (1,)
#         control_options['u1']['units'] = 'rad'
#
#         p = om.Problem()
#         interp_comp = p.model.add_subsystem('interp',
#                                             VandermondeControlInterpComp(grid_data=grid_data,
#                                                                          control_options=control_options,
#                                                                          standalone_mode=True,
#                                                                          time_units='s'))
#         p.setup(force_alloc_complex=True)
#
#         interp_comp.options['segment_index'] = 1
#         p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])
#
#         p.set_val('interp.stau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.stau', -0.72048)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', -0.167181)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', 0.446314)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', 0.885792)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)
#
#         interp_comp.options['segment_index'] = 0
#
#         p.set_val('interp.stau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', -0.28989795)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', 0.68989795)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.control_values:u1'), [[1.5]], tolerance=1.0E-5)
#
#         p.set_val('interp.stau', 0.54262)
#         p.run_model()
#
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#     def test_eval_control_radau_uncompressed_vectorized(self):
#         grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
#                                                          transcription_order=[3, 5], compressed=False)
#
#         time_options = dm.phase.options.TimeOptionsDictionary()
#
#         time_options['units'] = 's'
#
#         control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}
#
#         control_options['u1']['shape'] = (1,)
#         control_options['u1']['units'] = 'rad'
#
#         p = om.Problem()
#         interp_comp = p.model.add_subsystem('interp',
#                                             VandermondeControlInterpComp(grid_data=grid_data,
#                                                                          control_options=control_options,
#                                                                          vec_size=5,
#                                                                          standalone_mode=True,
#                                                                          time_units='s'))
#         p.setup(force_alloc_complex=True)
#
#         interp_comp.options['segment_index'] = 1
#         p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])
#
#         p.set_val('interp.stau', [-1.0, -0.72048, -0.167181, 0.446314, 0.885792])
#
#         p.run_model()
#
#         expected = np.array([[0.0, 4.0, 3.0, 4.0, 3.0]]).T
#
#         assert_near_equal(p.get_val('interp.control_values:u1'), expected, tolerance=1.0E-6)
#
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#
# class TestPolynomialControlInterpolation(unittest.TestCase):
#
#     def test_eval_polycontrol_gl_compressed(self):
#         grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
#                                                          transcription_order=[3, 5], compressed=True)
#
#         time_options = dm.phase.options.TimeOptionsDictionary()
#
#         time_options['units'] = 's'
#
#         pc_options = {'u1': dm.phase.options.PolynomialControlOptionsDictionary()}
#
#         pc_options['u1']['shape'] = (1,)
#         pc_options['u1']['units'] = 'rad'
#         pc_options['u1']['order'] = 6
#
#         p = om.Problem()
#         interp_comp = p.model.add_subsystem('interp',
#                                             VandermondeControlInterpComp(grid_data=grid_data,
#                                                                          polynomial_control_options=pc_options,
#                                                                          standalone_mode=True,
#                                                                          time_units='s'))
#         p.setup(force_alloc_complex=True)
#
#         interp_comp.options['segment_index'] = 1
#         p.set_val('interp.t_duration', 12.2352)
#         p.set_val('interp.dstau_dt', .3526)
#         p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])
#
#         p.set_val('interp.ptau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 0.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)
#
#         interp_comp.options['segment_index'] = 0
#
#         p.set_val('interp.ptau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 0.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 0.54262)
#         p.run_model()
#
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#     def test_eval_polycontrol_gl_compressed_vectorized(self):
#         grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
#                                                          transcription_order=[3, 5], compressed=True)
#
#         time_options = dm.phase.options.TimeOptionsDictionary()
#
#         time_options['units'] = 's'
#
#         pc_options = {'u1': dm.phase.options.PolynomialControlOptionsDictionary()}
#
#         pc_options['u1']['shape'] = (1,)
#         pc_options['u1']['units'] = 'rad'
#         pc_options['u1']['order'] = 6
#
#         p = om.Problem()
#         interp_comp = p.model.add_subsystem('interp',
#                                             VandermondeControlInterpComp(grid_data=grid_data,
#                                                                          vec_size=6,
#                                                                          polynomial_control_options=pc_options,
#                                                                          standalone_mode=True,
#                                                                          time_units='s'))
#         p.setup(force_alloc_complex=True)
#
#         interp_comp.options['segment_index'] = 1
#         p.set_val('interp.t_duration', 12.2352)
#         p.set_val('interp.dstau_dt', .3526)
#         p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])
#
#         p.set_val('interp.ptau', -1.0)
#         p.run_model()
#
#         ptau = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
#
#         p.set_val('interp.ptau', ptau)
#         p.run_model()
#
#         expected = np.array([[0.0, 1.5, 4.0, 0.0, 1.5, 4.0]]).T
#
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), expected, tolerance=_TOL)
#
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)
#
#     def test_eval_polycontrol_radau_compressed(self):
#         grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
#                                                          transcription_order=[3, 5], compressed=True)
#
#         time_options = dm.phase.options.TimeOptionsDictionary()
#
#         time_options['units'] = 's'
#
#         pc_options = {'u1': dm.phase.options.PolynomialControlOptionsDictionary()}
#
#         pc_options['u1']['shape'] = (1,)
#         pc_options['u1']['units'] = 'rad'
#         pc_options['u1']['order'] = 6
#
#         p = om.Problem()
#         interp_comp = p.model.add_subsystem('interp',
#                                             VandermondeControlInterpComp(grid_data=grid_data,
#                                                                          polynomial_control_options=pc_options,
#                                                                          standalone_mode=True,
#                                                                          time_units='s'))
#         p.setup(force_alloc_complex=True)
#
#         p.set_val('interp.t_duration', 12.252)
#         interp_comp.options['segment_index'] = 1
#         p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])
#
#         p.set_val('interp.ptau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 0.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)
#
#         interp_comp.options['segment_index'] = 0
#
#         p.set_val('interp.ptau', -1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 0.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 1.0)
#         p.run_model()
#         assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)
#
#         p.set_val('interp.ptau', 0.54262)
#         p.run_model()
#
#         with np.printoptions(linewidth=1024):
#             cpd = p.check_partials(compact_print=False, method='cs')
#             assert_check_partials(cpd, atol=_TOL, rtol=_TOL)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
