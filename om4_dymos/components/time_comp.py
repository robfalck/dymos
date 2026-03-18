import numpy as np
from om4.core.explicitcomponent import ExplicitComponent
from om4.specs.varspec import ContinuousVar
from pydantic import Field, model_validator, field_serializer
from typing import Optional, Any


class TimeComp(ExplicitComponent):
    num_nodes: int = Field(...)
    node_ptau: Any = Field(...)
    node_dptau_dstau: Any = Field(...)
    units: Optional[str] = Field(default=None)

    default_shape: Optional[tuple[int, ...]] = Field(default=None)
    default_units: Optional[str] = Field(default=None)

    @field_serializer('node_ptau', 'node_dptau_dstau')
    def serialize_array(self, val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    @model_validator(mode='after')
    def setup_vars(self) -> 'TimeComp':
        nn = self.num_nodes
        units = self.units
        self.inputs = {
            't_initial': ContinuousVar(val=0.0, units=units),
            't_duration': ContinuousVar(val=1.0, units=units),
        }
        self.outputs = {
            't': ContinuousVar(shape=(nn,), units=units, tags={'dymos.time_output'}),
            't_phase': ContinuousVar(
                shape=(nn,), units=units, tags={'dymos.time_output'}
            ),
            'dt_dstau': ContinuousVar(
                shape=(nn,), units=units, tags={'dymos.time_output'}
            ),
        }
        return self

    def __call__(self, t_initial, t_duration):
        return tuple()
