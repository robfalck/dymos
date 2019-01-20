from __future__ import print_function, division

import numpy as np

from openmdao.api import ExplicitComponent


class BallisticRocketGuidanceComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('time_phase',
                       val=np.zeros(nn),
                       desc='current elapsed time of the current phase',
                       units='s')

        self.add_input('t_duration',
                       val=np.zeros(nn),
                       desc='total duration of the current phase',
                       units='s')

        self.add_input('theta_0',
                       val=np.zeros(nn),
                       desc='thrust pitch angle at the start of pitchover',
                       units='s')

        self.add_input('theta_f',
                       val=np.zeros(nn),
                       desc='thrust pitch angle at the end of pitchover',
                       units='s')

        self.add_output('theta',
                        val=np.zeros(nn),
                        desc='pitch angle',
                        units='rad')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='theta', wrt='time_phase', rows=arange, cols=arange)
        self.declare_partials(of='theta', wrt='t_duration', rows=arange, cols=arange)
        self.declare_partials(of='theta', wrt='theta_0', rows=arange, cols=arange)
        self.declare_partials(of='theta', wrt='theta_f', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        time_phase = inputs['time_phase']
        t_duration = inputs['t_duration']
        theta_f = inputs['theta_f']
        theta_0 = inputs['theta_0']
        outputs['theta'] = theta_0 - (time_phase / t_duration) * (theta_f - theta_0)

    def compute_partials(self, inputs, partials):
        time_phase = inputs['time_phase']
        t_duration = inputs['t_duration']
        theta_f = inputs['theta_f']
        theta_0 = inputs['theta_0']

        partials['theta', 'time_phase'] = -(theta_f - theta_0) / t_duration
        partials['theta', 't_duration'] = time_phase * (theta_f - theta_0) / t_duration**2
        partials['theta', 'theta_0'] = 1.0 + 1.0 / (time_phase / t_duration)
        partials['theta', 'theta_f'] = -(time_phase / t_duration)
