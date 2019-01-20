from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent


class BallisticRocketEOMComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('x',
                       val=np.zeros(nn),
                       desc='horizontal position',
                       units='m')

        self.add_input('y',
                       val=np.zeros(nn),
                       desc='vertical position',
                       units='m')

        self.add_input('vx',
                       val=np.zeros(nn),
                       desc='horizontal velocity',
                       units='m/s')

        self.add_input('vy',
                       val=np.zeros(nn),
                       desc='vertical velocity',
                       units='m/s')

        self.add_input('g',
                       val=9.80665 * np.ones(nn),
                       desc='gravitational acceleration',
                       units='m/s**2')

        self.add_input('theta',
                       val=np.zeros(nn),
                       desc='pitch angle',
                       units='rad')

        self.add_input('thrust',
                       val=np.zeros(nn),
                       desc='thrust magnitude',
                       units='rad')

        self.add_output('x_dot',
                        val=np.zeros(nn),
                        desc='horizontal position rate of change',
                        units='m/s')

        self.add_output('v_dot',
                        val=np.zeros(nn),
                        desc='vertical position rate of change',
                        units='m/s')

        self.add_output('vx_dot',
                        val=np.zeros(nn),
                        desc='horizontal acceleration magnitude',
                        units='m/s**2')

        self.add_output('vy_dot',
                        val=np.zeros(nn),
                        desc='vertical acceleration magnitude',
                        units='m/s**2')

        self.add_output('m_dot',
                        val=np.zeros(nn),
                        desc='vehicle mass rate of change',
                        units='kg/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='x_dot', wrt='vx', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='y_dot', wrt='vy', rows=arange, cols=arange, val=1.0)

        self.declare_partials(of='vx_dot', wrt='thrust', rows=arange, cols=arange)
        self.declare_partials(of='vx_dot', wrt='theta', rows=arange, cols=arange)
        self.declare_partials(of='vx_dot', wrt='mprop', rows=arange, cols=arange)
        self.declare_partials(of='vx_dot', wrt='mstruct', rows=arange, cols=arange)

        self.declare_partials(of='vy_dot', wrt='thrust', rows=arange, cols=arange)
        self.declare_partials(of='vy_dot', wrt='mprop', rows=arange, cols=arange)
        self.declare_partials(of='vy_dot', wrt='mstruct', rows=arange, cols=arange)
        self.declare_partials(of='vy_dot', wrt='theta', rows=arange, cols=arange)
        self.declare_partials(of='vy_dot', wrt='g', rows=arange, cols=arange, val=-1.0)

        self.declare_partials(of='m', wrt='Isp', rows=arange, cols=arange)
        self.declare_partials(of='m', wrt='thrust', rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        theta = inputs['theta']
        thrust = inputs['thrust']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        mprop = inputs['mprop']
        mstruct = inputs['mstruct']
        mtotal = mstruct + mprop

        outputs['vx_dot'] = (thrust / mtotal) * cos_theta
        outputs['vy_dot'] = (thrust / mtotal) * sin_theta - g
        outputs['x_dot'] = inputs['vx']
        outputs['y_dot'] = inputs['vy']
        outputs['m_dot'] = inputs['thrust'] / (9.80665 * inputs['Isp'])

    def compute_partials(self, inputs, partials):
        theta = inputs['theta']
        thrust = inputs['thrust']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        mprop = inputs['mprop']
        mstruct = inputs['mstruct']
        mtotal = mstruct + mprop

        -thrust * sin_theta / (mstruct + mprop)

        partials['vx_dot', 'thrust'] = cos_theta / mtotal
        partials['vx_dot', 'theta'] = -(thrust / mtotal) * sin_theta
        partials['vx_dot', 'mstruct'] = thrust * sin_theta / mtotal**2
        partials['vx_dot', 'mprop'] = thrust * sin_theta / mtotal**2

        partials['vy_dot', 'thrust'] = sin_theta
        partials['vy_dot', 'theta'] = (thrust / mtotal) * cos_theta
        partials['vy_dot', 'mstruct'] = -(thrust / mtotal**2) * sin_theta
        partials['vy_dot', 'mprop'] = -(thrust / mtotal**2) * sin_theta

        partials['m_dot', 'thrust'] = 1 / (9.80665 * inputs['Isp'])
        partials['m_dot', 'Isp'] = -thrust / (9.80665 * inputs['Isp']**2)
