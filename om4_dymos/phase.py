from typing import Dict, Any, Literal
from pydantic import (
    Field,
    model_validator,
    ConfigDict,
    field_serializer,
    model_serializer,
)
from om4.core.group import Group, Connection
from om4.specs.deferrable import Deferrable

from om4_dymos.options import (
    TimeOptions,
    StateOptions,
    ControlOptions,
    ParameterOptions,
)
from om4_dymos.transcription import TranscriptionBase

UNSET: Literal['__UNSET__'] = '__UNSET__'


class Phase(Group):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    ode_class: Any = Field(description='The OM4 component class for the ODE')
    transcription: Deferrable[TranscriptionBase] = Field(
        default=UNSET, description='The transcription factory'
    )

    time_options: TimeOptions = Field(default_factory=TimeOptions)
    state_options: Dict[str, StateOptions] = Field(default_factory=dict)
    control_options: Dict[str, ControlOptions] = Field(default_factory=dict)
    parameter_options: Dict[str, ParameterOptions] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def forbid_manual_group_fields(cls, data: dict) -> dict:
        if 'type' not in data:
            for field in {'subsystems', 'connections', 'input_defaults'}:
                data.pop(field, None)
        return data

    @field_serializer('ode_class')
    def serialize_ode_class(self, val):
        if hasattr(val, '__name__'):
            return f'{val.__module__}.{val.__name__}'
        return str(val)

    @model_serializer(mode='wrap')
    def serialize_phase(self, handler, info):
        data = handler(self)
        if info.context and info.context.get('as_om4_group'):
            # Ensure we absolutely strip ALL Dymos data
            pure_group_data = {}
            for k in ['name', 'subsystems', 'connections', 'input_defaults']:
                if k in data:
                    pure_group_data[k] = data[k]
            pure_group_data['type'] = 'om4.core.group.Group'
            return pure_group_data
        return data

    @model_validator(mode='after')
    def build_phase_structure(self) -> 'Phase':
        if self.transcription is UNSET:
            if isinstance(self.subsystems, dict):
                self.subsystems.clear()
            self.__dict__['connections'] = []
            return self

        subs, conns = self.transcription.build_spec(self)

        for name, sub in subs.items():
            sub.__dict__['_name'] = name

        if self.subsystems is None or isinstance(self.subsystems, list):
            self.__dict__['subsystems'] = {}

        self.subsystems.clear()
        self.subsystems.update(subs)
        self.__dict__['connections'] = [
            Connection(**c) if isinstance(c, dict) else c for c in conns
        ]

        self.__dict__['_resolver'] = None
        self.__dict__['_owned_conns'] = None
        self.__dict__['_contained_conns'] = None

        return self
