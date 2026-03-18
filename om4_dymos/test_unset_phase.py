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
from om4_dymos.phase import Phase, UNSET

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
print(f'Is transcription UNSET? {my_phase.transcription is UNSET}')

print('\n--- ASSIGNING TRANSCRIPTION ---')
my_phase.transcription = RadauTranscription(num_segments=3, order=3, compressed=False)
print(f'Transcription state: {type(my_phase.transcription).__name__}')
print(f'Is transcription UNSET? {my_phase.transcription is UNSET}')

print('\n--- CLEARING TRANSCRIPTION ---')
my_phase.transcription = UNSET
print(f'Transcription state: {my_phase.transcription}')
print(f'Is transcription UNSET? {my_phase.transcription is UNSET}')
print(f'Subsystems count: {len(my_phase.subsystems)}')
