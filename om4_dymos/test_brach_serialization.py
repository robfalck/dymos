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
spec.loader.exec_module(brach_mod)

from om4_dymos.options import (
    StateOptions,
    TimeOptions,
    ControlOptions,
    ParameterOptions,
)
from om4_dymos.transcriptions.radau_new import RadauNew
from om4_dymos.phase import Phase
from om4.core.concrete_model import ConcreteModel
from om4.specs.spec_manager import SpecManager
from om4.core.group import Group

# Ensure our local classes are registered with SpecManager
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
SpecManager.MODELS['om4_dymos.components.control_interp_comp.ControlInterpComp'] = (
    importlib.import_module(
        'om4_dymos.components.control_interp_comp'
    ).ControlInterpComp
)

# 1. Define the Phase
phase = Phase(
    type='om4_dymos.phase.Phase',
    ode_class=brach_mod.BrachistochroneODE,
    time_options=TimeOptions(units='s', targets=[]),
    state_options={
        'x': StateOptions(
            name='x', rate_source='xdot', targets=[], fix_initial=True, initial_val=0.0
        ),
        'y': StateOptions(
            name='y', rate_source='ydot', targets=[], fix_initial=True, initial_val=10.0
        ),
        'v': StateOptions(
            name='v',
            rate_source='vdot',
            targets=['v'],
            fix_initial=True,
            initial_val=0.0,
        ),
    },
    control_options={
        'theta': ControlOptions(name='theta', units='rad', targets=['theta'])
    },
    parameter_options={
        'g': ParameterOptions(name='g', units='m/s**2', targets=['g'], val=9.80665)
    },
    transcription=RadauNew(num_segments=3, order=3, compressed=False),
)

from pprint import pprint

pprint(phase.model_dump(), indent=2)

# # 3. Embed inside a ConcreteModel (OM4 execution wrapper)
# model = ConcreteModel(
#     type="om4.core.concrete_model.ConcreteModel",
#     group=Group(
#         type="om4.core.group.Group",
#         subsystems={"brach_phase": phase}
#     )
# )

# # 4. Attempt to dump the full model and reload it
# model_json = model.model_dump_json(indent=2)
# print("--- FULL MODEL SERIALIZED SUCCESSFULLY ---")
# print("First 20 lines of JSON:")
# print("\n".join(model_json.split("\n")[:20]))

# # Can we reload it?
# loaded_model = ConcreteModel.model_validate_json(model_json)
# print("\n--- RELOADED FROM JSON SUCCESSFULLY ---")
# print(f"Loaded phases: {list(loaded_model.group.subsystems.keys())}")
