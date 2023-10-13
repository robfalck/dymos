import time
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
import dymos as dm

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

        # self.add_subsystem('control_group',
        #                    ControlGroup(grid_data=self._grid_data,
        #                                 control_options=self._control_options,
        #                                 time_units=self._time_units),
        #                    promotes=['*'])

        self.add_subsystem('barycentric_interp_comp',
                           BarycentricLagrangeInterpComp(grid_data=self._grid_data),
                           promotes=['*'])

    def configure(self):
        interp_comp = self._get_subsystem('barycentric_interp_comp')

        for control_name, options in self._control_options.items():
            interp_comp.add_control_interp(control_name,
                                           shape=options['shape'],
                                           units=options['units'])

        interp_comp.configure_io()



class TestBarycentricLagrangeInterpComp(unittest.TestCase):

    def test_scalar_interp(self):
        for transcription in ['radau-ps', 'gauss-lobatto']:
            for compressed in [True, False]:
                with self.subTest(msg=f'tx={transcription}; compressed={compressed}'):
                    grid_data = dm.transcriptions.grid_data.GridData(num_segments=2,
                                                                     transcription=transcription,
                                                                     transcription_order=[3, 5],
                                                                     compressed=compressed)

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

                    t = 2 * np.pi * (grid_data.node_ptau + 1)
                    x = np.sin(t)

                    p.set_val('test_group.t_duration', 2 * np.pi)

                    p.set_val('test_group.controls:u1',
                              val=x[grid_data.subset_node_indices['control_input']],
                              units='unitless')

                    interp_comp = g._get_subsystem('barycentric_interp_comp')

                    ptau_results = []
                    u1_results = []
                    u1_rate_results = []
                    u1_rate2_results = []

                    t0 = time.time_ns()
                    for seg_idx in range(2):
                        interp_comp.set_segment_index(seg_idx)
                        for stau in np.linspace(-1, 1, 100):
                            if seg_idx == 0:
                                ptau_results.append((stau - 1.0) / 2.)
                            elif seg_idx == 1:
                                ptau_results.append((stau + 1.0) / 2.)
                            p.set_val('test_group.stau', stau)
                            p.run_model()
                            u1_results.extend(p.get_val('test_group.control_values:u1')[0])
                            u1_rate_results.extend(p.get_val('test_group.control_rates:u1_rate')[0])
                            u1_rate2_results.extend(p.get_val('test_group.control_rates:u1_rate2')[0])
                    tf = time.time_ns()
                    print((tf - t0) / 1.0E9)

                    # import matplotlib.pyplot as plt
                    # t_plot = (np.asarray(ptau_results) + 1) * 2 * np.pi
                    # plt.plot(t_plot, u1_results, '-', label='u1')
                    # plt.plot(t_plot, u1_rate_results, '-', label='u1_rate')
                    # plt.plot(t_plot, u1_rate2_results, '-', label='u1_rate2')
                    # plt.legend()
                    # plt.show()
                    #
                    p.set_val('test_group.stau', 0.5356)
                    p.run_model()

                    # assert_near_equal(u1_results, np.sin(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-12)
                    # assert_near_equal(u1_rate_results, np.cos(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-9)
                    # assert_near_equal(u1_rate2_results, -np.sin(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-9)

                    t0 = time.time_ns()
                    with np.printoptions(linewidth=1024, edgeitems=1024):
                        cpd = p.check_partials(compact_print=False, method='cs')
                    tf = time.time_ns()
                    print((tf - t0) / 1.0E9)

                    assert_check_partials(cpd)

    def test_vector_interp(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=False)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (2,)
        control_options['u1']['units'] = 'unitless'

        p = om.Problem()

        g = p.model.add_subsystem('test_group',
                                  TestGroup(grid_data=grid_data,
                                            control_options=control_options,
                                            time_units='s'))

        p.setup(force_alloc_complex=True)

        t = 2 * np.pi * (grid_data.node_ptau + 1)
        x = np.sin(t)
        y = np.cos(t)
        val = np.stack((x, y)).T

        p.set_val('test_group.t_duration', 2 * np.pi)
        p.set_val('test_group.controls:u1', val=val)

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
                u1_results.append(p.get_val('test_group.control_values:u1').tolist())
                u1_rate_results.append(p.get_val('test_group.control_rates:u1_rate').tolist())
                u1_rate2_results.append(p.get_val('test_group.control_rates:u1_rate2').tolist())

        u1_results = np.asarray(u1_results)
        u1_rate_results = np.asarray(u1_rate_results)
        u1_rate2_results = np.asarray(u1_rate2_results)

        # import matplotlib.pyplot as plt
        # plt.plot(grid_data.node_ptau, p.get_val('test_group.control_values:u1')[:, 0], 'o', ms=3)
        # plt.plot(ptau_results, u1_results[:, 0, 0], '-')
        # plt.plot(grid_data.node_ptau, p.get_val('test_group.control_values:u1')[:, 1], 'o', ms=3)
        # plt.plot(ptau_results, u1_results[:, 0, 1], '-')
        # # #
        # plt.plot(grid_data.node_ptau, p.get_val('test_group.control_rates:u1_rate')[:, 0], 'o', ms=3)
        # plt.plot(ptau_results, u1_rate_results[:, 0, 0], '-')
        # plt.plot(grid_data.node_ptau, p.get_val('test_group.control_rates:u1_rate')[:, 1], 'o', ms=3)
        # plt.plot(ptau_results, u1_rate_results[:, 0, 1], '-')
        # #
        # # plt.plot(grid_data.node_ptau, p.get_val('test_group.control_rates:u1_rate2'), 'o', ms=3)
        # # plt.plot(ptau_results, u1_rate2_results, '-')
        # plt.show()
        #
        p.set_val('test_group.stau', 0.5356)
        p.run_model()

        # assert_near_equal(u1_results[:, 0, 0], np.sin(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-4)
        # assert_near_equal(u1_results[:, 0, 1], np.cos(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-4)
        #
        # assert_near_equal(u1_rate_results[:, 0, 0], np.cos(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-3)
        # assert_near_equal(u1_rate_results[:, 0, 1], -np.sin(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=1.0E-3)
        #
        # assert_near_equal(u1_rate2_results[:, 0, 0], -np.sin(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=5.0E-3)
        # assert_near_equal(u1_rate2_results[:, 0, 1], -np.cos(2 * np.pi * (np.asarray(ptau_results) + 1)), tolerance=5.0E-3)

        with np.printoptions(linewidth=100_000, edgeitems=100_000):
            cpd = p.check_partials(compact_print=False, method='cs')
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
