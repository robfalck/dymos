import numpy as np
from om4.core.explicitcomponent import ExplicitComponent
from om4.specs.varspec import ContinuousVar, NpFloatArray
from pydantic import Field, model_validator, field_serializer
from typing import Dict, Optional
from om4_dymos.options import TimeOptions, ControlOptions


class ControlInterpComp(ExplicitComponent):
    num_input_nodes: int = Field(...)
    num_output_nodes: int = Field(...)
    time_options: TimeOptions = Field(default_factory=TimeOptions)
    control_options: Dict[str, ControlOptions] = Field(default_factory=dict)

    L: NpFloatArray = Field(...)
    D: NpFloatArray = Field(...)
    D2: NpFloatArray = Field(...)

    default_shape: Optional[tuple[int, ...]] = Field(default=None)
    default_units: Optional[str] = Field(default=None)

    @field_serializer('L', 'D', 'D2')
    def serialize_array(self, val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    @model_validator(mode='after')
    def setup_vars(self) -> 'ControlInterpComp':
        self.inputs = {
            'dt_dstau': ContinuousVar(
                shape=(self.num_output_nodes,), units=self.time_options.units
            )
        }
        self.outputs = {}
        for name, options in self.control_options.items():
            # Unpack options if it degraded to a dict during JSON deserialization
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

            rate_units = None if opt_units is None else f'{opt_units}/s'
            rate2_units = None if opt_units is None else f'{opt_units}/s**2'

            out_shape = (self.num_output_nodes,) + shape
            in_shape = (self.num_input_nodes,) + shape

            self.inputs[f'controls:{name}'] = ContinuousVar(
                shape=in_shape, units=opt_units
            )
            self.outputs[f'control_values:{name}'] = ContinuousVar(
                shape=out_shape, units=opt_units
            )
            self.outputs[f'control_rates:{name}'] = ContinuousVar(
                shape=out_shape, units=rate_units
            )
            self.outputs[f'control_rates2:{name}'] = ContinuousVar(
                shape=out_shape, units=rate2_units
            )
        return self

    def __call__(self, dt_dstau, **kwargs):
        return tuple()
