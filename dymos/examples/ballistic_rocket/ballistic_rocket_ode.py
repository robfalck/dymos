from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import Group
from dymos import declare_time, declare_state, declare_parameter

from .ballistic_rocket_eom_comp import BallisticRocketEOMComp
from .ballistic_rocket_guidance_comp import BallisticRocketGuidanceComp

@declare_time(units='s')
@declare_state('x', rate_source='eom.x_dot', units='m')
@declare_state('y', rate_source='eom.y_dot', units='m')
@declare_state('vx', rate_source='eom.vx_dot', targets=['eom.vx'], units='m/s')
@declare_state('vy', rate_source='eom.vy_dot', targets=['eom.vy'], units='m/s')
@declare_state('mprop', rate_source='eom.mprop_dot', targets=['eom.mprop'], units='kg')
@declare_parameter('mstruct', targets=['eom.mstruct'], units='kg')
@declare_parameter('theta', targets=['eom.theta'], units='rad')
@declare_parameter('thrust', targets=['eom.thrust'], units='N')
@declare_parameter('Isp', targets=['eom.Isp'], units='s')
@declare_parameter('g', units='m/s**2', targets=['eom.g'])
class BallisticRocketUnguidedODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem(name='eom', subsys=BallisticRocketEOMComp(num_nodes=nn))


@declare_time(units='s', time_phase_targets=['guidance.time_phase'],
              t_duration_targets=['guidance.t_duration'])
@declare_state('x', rate_source='eom.x_dot', units='m')
@declare_state('y', rate_source='eom.y_dot', units='m')
@declare_state('vx', rate_source='eom.vx_dot', targets=['eom.vx'], units='m/s')
@declare_state('vy', rate_source='eom.vy_dot', targets=['eom.vy'], units='m/s')
@declare_state('mprop', rate_source='eom.mprop_dot', targets=['eom.mprop'], units='kg')
@declare_parameter('mstruct', targets=['eom.mstruct'], units='kg')
@declare_parameter('thrust', targets=['eom.thrust'], units='N')
@declare_parameter('Isp', targets=['eom.Isp'], units='s')
@declare_parameter('g', units='m/s**2', targets=['eom.g'])
@declare_parameter('theta_0', units='rad', targets=['guidance.theta_0'])
@declare_parameter('theta_f', units='rad', targets=['guidance.theta_f'])
class BallisticRocketGuidedODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem(name='guidance', subsys=BallisticRocketGuidanceComp(num_nodes=nn))
        self.add_subsystem(name='eom', subsys=BallisticRocketEOMComp(num_nodes=nn))

        self.connect('guidance.theta', 'eom.theta')