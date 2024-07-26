import numpy as np
import openmdao.api as om


class Atmosphere(om.ExplicitComponent):
    """
    Define the logarithmic atmosphere model for the shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 248, 2010.
    """

    def initialize(self):
        """
        Declare otpions for the Atmosphere component.
        """
        self.options.declare('num_nodes', types=int)

    def setup(self):
        """
        Add I/O for the Atmosphere component.
        """
        nn = self.options['num_nodes']
        self.add_input('h', val=np.ones(nn), desc='altitude', units='ft')
        self.add_output('rho', val=np.ones(nn), desc='local density', units='slug/ft**3')
        partial_range = np.arange(nn, dtype=int)
        self.declare_partials('rho', 'h', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        """
        Compute the outputs of the Atmosphere component.

        Parameters
        ----------
        inputs : Vector
            Inputs.
        outputs : Vector
            Outputs.
        """
        h = inputs['h']
        h_r = 23800
        rho_0 = .002378
        outputs['rho'] = rho_0 * np.exp(-h / h_r)

    def compute_partials(self, inputs, partials):
        """
        Compute the partial derivatives of the atmosphere component.

        Parameters
        ----------
        inputs : Vector
            Inputs
        partials : dict[tuple[str, str]: ArrayLike]
            Partial derivatives.
        """
        h = inputs['h']
        h_r = 23800
        rho_0 = .002378
        partials['rho', 'h'] = -1 / h_r * rho_0 * np.exp(-h / h_r)
