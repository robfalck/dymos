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

phase = Phase(
    type='om4_dymos.phase.Phase',
    ode_class=brach_mod.BrachistochroneODE,
    time_options=TimeOptions(
        units='s', targets=['t']
    ),  # Adding a fake 't' target to ODE for test verification
    state_options={
        'x': StateOptions(name='x', rate_source='xdot', targets=['x']),
        'y': StateOptions(name='y', rate_source='ydot', targets=['y']),
        'v': StateOptions(name='v', rate_source='vdot', targets=['v']),
    },
)

phase.transcription = RadauTranscription(num_segments=3, order=3, compressed=False)

if isinstance(phase.subsystems, dict):
    print(f'Phase Subsystems: {list(phase.subsystems.keys())}')
elif isinstance(phase.subsystems, list):
    print(
        f'Phase Subsystems: {[s.get("name") if isinstance(s, dict) else getattr(s, "name", "") for s in phase.subsystems]}'
    )

print('\nConnections:')
if phase.connections:
    for conn in phase.connections:
        src = getattr(conn, 'src', None) or (
            conn.get('src') if isinstance(conn, dict) else str(conn)
        )
        tgt = getattr(conn, 'tgt', None) or (
            conn.get('tgt') if isinstance(conn, dict) else str(conn)
        )
        # Print connections involving time
        if 'time' in src or 'time' in tgt:
            print(f'  {src} -> {tgt}')
