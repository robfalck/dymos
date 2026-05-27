import sys
import os

sys.path.insert(0, os.path.expanduser('~/Codes/om4.git'))
sys.path.insert(0, '.')

import importlib.util

spec = importlib.util.spec_from_file_location(
    'brachistochrone_ode', 'dymos/examples/brachistochrone/om4/brachistochrone_ode.py'
)
brach_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brach_mod)

from om4_dymos.options import StateOptions, TimeOptions
from om4_dymos.transcriptions.radau import RadauTranscription
from om4_dymos.phase import Phase

# 1. Create an abstract phase template (No Transcription!)
my_phase = Phase(
    type='om4_dymos.phase.Phase',
    ode_class=brach_mod.BrachistochroneODE,
    state_options={
        'x': StateOptions(name='x', rate_source='xdot', targets=['x']),
        'y': StateOptions(name='y', rate_source='ydot', targets=['y']),
        'v': StateOptions(name='v', rate_source='vdot', targets=['v']),
    },
)

print('--- INITIAL ABSTRACT PHASE ---')
print(f'Transcription state: {my_phase.transcription}')
print(f'Subsystems count: {len(my_phase.subsystems)}')

print('\n--- ASSIGNING TRANSCRIPTION ---')
my_phase.transcription = RadauTranscription(num_segments=3, order=3, compressed=False)


def get_ode_nodes(phase):
    if isinstance(phase.subsystems, dict):
        return phase.subsystems['ode'].num_nodes
    elif isinstance(phase.subsystems, list):
        for s in phase.subsystems:
            n = s.get('name') if isinstance(s, dict) else getattr(s, 'name', '')
            if n == 'ode':
                return s.get('num_nodes') if isinstance(s, dict) else s.num_nodes


print(f'ODE component num_nodes: {get_ode_nodes(my_phase)}')

print('\n--- MODIFYING TRANSCRIPTION PROPERTY ---')
my_phase.transcription = RadauTranscription(num_segments=10, order=5, compressed=False)

print(f'New ODE component num_nodes: {get_ode_nodes(my_phase)}')

print('\n--- CLEARING TRANSCRIPTION ---')
my_phase.transcription = '__UNSET__'
print(f'Subsystems count: {len(my_phase.subsystems)}')
