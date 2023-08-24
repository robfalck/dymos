"""Definition of the Passthru Component for States."""


import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from ...._options import options as dymos_options


class BirkhoffStatesComp(ExplicitComponent):
    """
    A component which simply passes a state value from an input to an equivalent output.

    Parameters
    ----------
    state_options : StateOptionsDictionary
        The state options of the parent phase.
    grid_data : GridData
        The GridData of the parent phase.
    """

    def __init__(self, state_options, grid_data):
        """
        Instantiate MuxComp and populate private members.
        """
        super().__init__()
        self._state_options = state_options
        self._grid_data = grid_data

        self._no_check_partials = not dymos_options['include_check_partials']

    def configure_io(self):
        """
        A configure method called during configure of the owning phase.
        """
        gd = self._grid_data
        nn = gd.subset_num_nodes['all']
        ar = np.arange(nn, dtype=int)

        for state_name, options in self._state_options.items():
            shape = (nn,) + options['shape']
            self.add_input(name=f'states:{state_name}', units=options['units'], shape=shape)
            self.add_output(name=f'state_vals:{state_name}', units=options['units'], shape=shape)
            self.declare_partials(of=f'state_vals:{state_name}', wrt=f'states:{state_name}', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Populate the outputs from the inputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        discrete_inputs : Vector
            Discrete input variables read via discrete_inputs[key].
        discrete_outputs : Vector
            Discrete output variables read via discrete_outputs[key].
        """
        outputs.set_val(inputs.asarray())
