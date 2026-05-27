from typing import Dict, Any

from om4.core.explicitcomponent import ExplicitComponent
from om4.specs.varspec import ContinuousVar
from pydantic import Field, model_validator


class TimeseriesComp(ExplicitComponent):
    """
    A standard OM4 Timeseries component.
    """

    num_nodes: int = Field(..., description='Number of timeseries nodes')
    state_options: Dict[str, Any] = Field(
        default_factory=dict, description='State options dictionary'
    )

    @model_validator(mode='after')
    def setup_vars(self) -> 'TimeseriesComp':
        self.inputs = {}
        self.outputs = {}

        for name, options in self.state_options.items():
            shape = options.shape

            self.inputs[name] = ContinuousVar(
                shape=(self.num_nodes, *shape), units=options.units
            )

            self.outputs[name] = ContinuousVar(
                shape=(self.num_nodes, *shape),
                units=options.units,
                tags={'dymos.timeseries'},
            )

        return self

    def __call__(self, **kwargs):
        # Timeseries is basically a passthrough right now
        return tuple(kwargs.values())
