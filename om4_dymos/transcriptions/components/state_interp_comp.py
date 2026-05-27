from typing import Dict, Any, Optional
import numpy as np

from om4.core.explicitcomponent import ExplicitComponent
from om4.specs.varspec import ContinuousVar, NpFloatArray
from pydantic import Field, model_validator, field_serializer


class StateInterpComp(ExplicitComponent):
    num_disc_nodes: int = Field(...)
    num_col_nodes: int = Field(...)
    state_options: Dict[str, Any] = Field(default_factory=dict)
    Ad: NpFloatArray = Field(...)
    Li: NpFloatArray = Field(...)

    default_shape: Optional[tuple[int, ...]] = Field(default=None)
    default_units: Optional[str] = Field(default=None)

    @field_serializer('Ad', 'Li')
    def serialize_array(self, val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    @model_validator(mode='after')
    def setup_vars(self) -> 'StateInterpComp':
        self.inputs = {
            'dt_dstau': ContinuousVar(shape=(self.num_col_nodes,), units='s')
        }
        self.outputs = {}
        for name, options in self.state_options.items():
            opt_shape = (
                options.get('shape', ()) if isinstance(options, dict) else options.shape
            )
            opt_units = (
                options.get('units') if isinstance(options, dict) else options.units
            )

            shape = (
                tuple(opt_shape)
                if isinstance(opt_shape, (list, tuple))
                else (opt_shape,)
            )
            if not shape:
                shape = ()
            in_shape = (self.num_disc_nodes,) + shape
            out_shape = (self.num_col_nodes,) + shape

            self.inputs[f'state_disc:{name}'] = ContinuousVar(
                shape=in_shape, units=opt_units
            )
            self.outputs[f'state_col:{name}'] = ContinuousVar(
                shape=out_shape, units=opt_units
            )
            rate_units = None if opt_units is None else f'{opt_units}/s'
            self.outputs[f'staterate_col:{name}'] = ContinuousVar(
                shape=out_shape, units=rate_units, tags={'dymos.state_rate_interp'}
            )
        return self

    def __call__(self, dt_dstau, **kwargs):
        return tuple()
