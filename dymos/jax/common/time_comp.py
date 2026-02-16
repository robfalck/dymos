"""JAX version of TimeComp."""
import jax.numpy as jnp
import numpy as np  # Only for setup/initialization
import openmdao.api as om

from dymos._options import options as dymos_options


class TimeComp(om.JaxExplicitComponent):
    """
    JAX implementation of TimeComp.

    Computes time values at each node given initial time and duration.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    Notes
    -----
    This JAX implementation uses automatic differentiation instead of
    manual partial derivative computation. Units are ignored in this version.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        # Required
        self.options.declare('num_nodes', types=int,
                             desc='The total number of points at which times are required in the'
                                  'phase.')

        self.options.declare('node_ptau', types=(np.ndarray,),
                             desc='The locations of all nodes in non-dimensional phase tau space.')

        self.options.declare('node_dptau_dstau', types=(np.ndarray,),
                             desc='For each node, the ratio of the total phase length to the length'
                                  ' of the nodes containing segment.')

        # Optional
        self.options.declare('initial_val', default=0.0, types=(int, float),
                             desc='default value of initial time')

        self.options.declare('duration_val', default=1.0, types=(int, float),
                             desc='default value of duration')

    def setup(self):
        """
        I/O creation.
        """
        num_nodes = self.options['num_nodes']

        # Inputs - without units for JAX version
        self.add_input('t_initial', val=self.options['initial_val'])
        self.add_input('t_duration', val=self.options['duration_val'])

        # Outputs - without units for JAX version
        self.add_output('t', val=np.ones(num_nodes))
        self.add_output('t_phase', val=np.ones(num_nodes))
        self.add_output('dt_dstau', val=np.ones(num_nodes))

    def get_self_statics(self):
        """
        Return static variables needed by compute_primal.

        Returns
        -------
        tuple
            (node_ptau, node_dptau_dstau) arrays from options
        """
        return (
            self.options['node_ptau'],
            self.options['node_dptau_dstau']
        )

    def compute_primal(self, t_initial, t_duration, node_ptau=None, node_dptau_dstau=None):
        """
        Compute time values using JAX.

        Parameters
        ----------
        t_initial : float
            Initial time value
        t_duration : float
            Duration of the phase
        node_ptau : array
            Locations of nodes in non-dimensional phase tau space (from get_self_statics)
        node_dptau_dstau : array
            Ratio of total phase length to segment length at each node (from get_self_statics)

        Returns
        -------
        tuple
            (t, t_phase, dt_dstau) in order of add_output declarations
        """
        # Compute time at each node
        t = t_initial + 0.5 * (node_ptau + 1) * t_duration

        # Compute phase time (time since start of phase)
        t_phase = 0.5 * (node_ptau + 1) * t_duration

        # Compute dt/dstau
        dt_dstau = 0.5 * t_duration * node_dptau_dstau

        return t, t_phase, dt_dstau
