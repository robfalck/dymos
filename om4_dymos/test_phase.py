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
from om4_dymos.transcription import StubRadauTranscription
from om4_dymos.phase import Phase

phase = Phase(
    type='om4_dymos.phase.Phase',
    ode_class=brach_mod.BrachistochroneODE,
    transcription=StubRadauTranscription(num_segments=5, order=3),
    state_options={
        'x': StateOptions(name='x', rate_source='xdot', targets=['x']),
        'y': StateOptions(name='y', rate_source='ydot', targets=['y']),
        'v': StateOptions(name='v', rate_source='vdot', targets=['v']),
    },
)

if isinstance(phase.subsystems, dict):
    print(f'Phase Subsystems: {list(phase.subsystems.keys())}')
    print(f'ODE component type: {type(phase.subsystems["ode"]).__name__}')
elif isinstance(phase.subsystems, list):
    print(
        f'Phase Subsystems: {[s.get("name") if isinstance(s, dict) else getattr(s, "name", "") for s in phase.subsystems]}'
    )
    # In a real scenario we might have converted them to dicts, but let's just show it parsed

print('Connections:')
if phase.connections:
    for conn in phase.connections:
        # Check if they are Connection objects or raw dicts
        src = getattr(conn, 'src', None) or (
            conn.get('src') if isinstance(conn, dict) else str(conn)
        )
        tgt = getattr(conn, 'tgt', None) or (
            conn.get('tgt') if isinstance(conn, dict) else str(conn)
        )
        print(f'  {src} -> {tgt}')
else:
    print('  None')
