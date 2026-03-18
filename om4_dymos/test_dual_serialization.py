import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.expanduser('~/Codes/om4.git'))
sys.path.insert(0, '.')

import importlib.util

spec = importlib.util.spec_from_file_location(
    'brachistochrone_ode', 'dymos/examples/brachistochrone/om4/brachistochrone_ode.py'
)
brach_mod = importlib.util.module_from_spec(spec)
sys.modules['brachistochrone_ode'] = brach_mod  # THIS WAS THE MISSING KEY FOR IMPORTLIB
spec.loader.exec_module(brach_mod)

from om4_dymos.options import (
    StateOptions,
    TimeOptions,
    ControlOptions,
    ParameterOptions,
)
from om4_dymos.transcriptions.radau_new import RadauNew
from om4_dymos.phase import Phase

from om4.specs.spec_manager import SpecManager

SpecManager.MODELS['om4_dymos.phase.Phase'] = Phase
SpecManager.MODELS['brachistochrone_ode.BrachistochroneODE'] = (
    brach_mod.BrachistochroneODE
)
SpecManager.MODELS['om4_dymos.components.time_comp.TimeComp'] = importlib.import_module(
    'om4_dymos.components.time_comp'
).TimeComp
SpecManager.MODELS[
    'om4_dymos.transcriptions.components.state_interp_comp.StateInterpComp'
] = importlib.import_module(
    'om4_dymos.transcriptions.components.state_interp_comp'
).StateInterpComp
SpecManager.MODELS[
    'om4_dymos.transcriptions.components.collocation_comp.CollocationComp'
] = importlib.import_module(
    'om4_dymos.transcriptions.components.collocation_comp'
).CollocationComp

phase = Phase(
    type='om4_dymos.phase.Phase',
    ode_class=brach_mod.BrachistochroneODE,
    time_options=TimeOptions(units='s', targets=[]),
    state_options={
        'x': StateOptions(
            name='x',
            rate_source='xdot',
            targets=[],
            fix_initial=True,
            initial_val=0.0,
            shape=(),
        ),
    },
    transcription=RadauNew(num_segments=3, order=3, compressed=False),
)

print('--- 1. STANDARD DYMOS SERIALIZATION ---')
dymos_json = phase.model_dump_json(indent=2)
print("Contains 'state_options'? ", 'state_options' in dymos_json)
print('type:', json.loads(dymos_json)['type'])

print('\n--- 2. OM4 GROUP SERIALIZATION ---')
om4_json = phase.model_dump_json(indent=2, context={'as_om4_group': True})
print("Contains 'state_options'? ", 'state_options' in om4_json)
print('type:', json.loads(om4_json)['type'])

# Can we validate the pure OM4 Group directly without any Dymos dependencies?
from om4.core.group import Group

reloaded_group = Group.model_validate_json(om4_json)
print('\nReloaded purely as om4.Group successfully!')
print(f'Reloaded Subsystems: {list(reloaded_group.subsystems.keys())}')
