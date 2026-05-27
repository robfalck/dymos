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
abstract_phase = Phase(
    type='om4_dymos.phase.Phase',
    ode_class=brach_mod.BrachistochroneODE,
    state_options={
        'x': StateOptions(name='x', rate_source='xdot', targets=['x']),
        'y': StateOptions(name='y', rate_source='ydot', targets=['y']),
        'v': StateOptions(name='v', rate_source='vdot', targets=['v']),
    },
)

print('--- ABSTRACT PHASE ---')
print(f'Transcription state: {abstract_phase.transcription}')
print(f'Subsystems generated: {abstract_phase.subsystems}')

# 2. Assign the transcription directly
abstract_phase.transcription = RadauTranscription(
    num_segments=4, order=3, compressed=False
)

# 3. Manually compile the phase into a concrete OM4 Group
compiled_phase = abstract_phase.compile()

print('\n--- COMPILED PHASE ---')
print(f'Transcription state: {type(compiled_phase.transcription).__name__}')

if isinstance(compiled_phase.subsystems, dict):
    print(f'Subsystems generated: {list(compiled_phase.subsystems.keys())}')
    print(f'ODE component num_nodes: {compiled_phase.subsystems["ode"].num_nodes}')
elif isinstance(compiled_phase.subsystems, list):
    print(
        f'Subsystems generated: {[s.get("name") if isinstance(s, dict) else getattr(s, "name", "") for s in compiled_phase.subsystems]}'
    )

    def get_comp(name):
        for s in compiled_phase.subsystems:
            n = s.get('name') if isinstance(s, dict) else getattr(s, 'name', '')
            if n == name:
                return s

    ode_comp = get_comp('ode')
    if isinstance(ode_comp, dict):
        print(f'ODE component num_nodes: {ode_comp.get("num_nodes")}')
    else:
        print(f'ODE component num_nodes: {ode_comp.num_nodes}')
