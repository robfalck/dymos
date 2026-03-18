from typing import Dict, Any, Optional

from om4.core.explicitcomponent import ExplicitComponent
from om4.specs.varspec import ContinuousVar
from pydantic import Field, model_validator


class CollocationComp(ExplicitComponent):
    num_col_nodes: int = Field(...)
    state_options: Dict[str, Any] = Field(default_factory=dict)

    default_shape: Optional[tuple[int, ...]] = Field(default=None)
    default_units: Optional[str] = Field(default=None)

    @model_validator(mode='after')
    def setup_vars(self) -> 'CollocationComp':
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
            out_shape = (self.num_col_nodes,) + shape
            rate_units = None if opt_units is None else f'{opt_units}/s'

            self.inputs[f'f_approx:{name}'] = ContinuousVar(
                shape=out_shape, units=rate_units
            )
            self.inputs[f'f_computed:{name}'] = ContinuousVar(
                shape=out_shape, units=rate_units
            )
            self.outputs[f'defects:{name}'] = ContinuousVar(
                shape=out_shape, units=opt_units, tags={'dymos.defect'}
            )
        return self

    def __call__(self, dt_dstau, **kwargs):
        return tuple()
