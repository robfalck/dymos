import numpy as np
import openmdao.api as om
import scipy.sparse as sp

from dymos._options import options as dymos_options
from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units


class GaussLobattoInterpComp(om.ExplicitComponent):
    r"""
    Provide interpolated state values and rates for the Gauss Lobatto transcription.

    When the transcription is *gauss-lobatto* it accepts the state values and derivatives
    at discretization nodes and computes the interpolated state values and derivatives
    at the collocation nodes, using a Hermite interpolation scheme.

    .. math:: x_c = \left[ A_i \right] x_d + \frac{dt}{d\tau_s} \left[ B_i \right] f_d
    .. math:: \dot{x}_c = \frac{d\tau_s}{dt} \left[ A_d \right] x_d + \left[ B_d \right] f_d

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """Declare component options."""
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of the integration variable')

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units.
        """
        time_units = self.options['time_units']

        num_disc_nodes = self.options['grid_data'].subset_num_nodes['state_disc']
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']

        state_options = self.options['state_options']

        self.add_input(name='dt_dstau', shape=(num_col_nodes,), units=time_units,
                       desc='For each node, the duration of its '
                            'segment in the integration variable')

        self.xd_str = {}
        self.fd_str = {}
        self.xc_str = {}
        self.xdotc_str = {}

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.add_input(
                name=f'state_disc:{state_name}',
                shape=(num_disc_nodes,) + shape,
                desc=f'Values of state {state_name} at discretization nodes',
                units=units)

            self.add_input(
                name=f'staterate_disc:{state_name}',
                shape=(num_disc_nodes,) + shape,
                units=rate_units,
                desc=f'EOM time derivative of state {state_name} at discretization nodes')

            self.add_output(
                name=f'state_col:{state_name}',
                shape=(num_col_nodes,) + shape, units=units,
                desc=f'Interpolated values of state {state_name} at collocation nodes')

            self.add_output(
                name=f'staterate_col:{state_name}',
                shape=(num_col_nodes,) + shape,
                units=rate_units,
                desc=f'Interpolated rate of state {state_name} at collocation nodes')

            self.xd_str[state_name] = f'state_disc:{state_name}'
            self.fd_str[state_name] = f'staterate_disc:{state_name}'
            self.xc_str[state_name] = f'state_col:{state_name}'
            self.xdotc_str[state_name] = f'staterate_col:{state_name}'

        Ai, Bi, Ad, Bd = self.options['grid_data'].phase_hermite_matrices('state_disc', 'col', sparse=True)

        self._matrices = {'Ai': Ai, 'Bi': Bi, 'Ad': Ad, 'Bd': Bd}
        self._jacs = {'Ai': {}, 'Bi': {}, 'Ad': {}, 'Bd': {}}
        self._sizes = {}

        for name, options in state_options.items():
            shape = options['shape']

            size = np.prod(shape)
            self._sizes[name] = size

            for key in self._jacs:
                # Each jacobian matrix has a form that is defined by the Kronecker product
                # of the interpolation matrix eye(size).
                self._jacs[key][name] = sp.kron(sp.csr_matrix(self._matrices[key]),
                                                sp.eye(size, format='csr'),
                                                format='csr')

            self._sizes[name] = size

            #
            # Partial of xdotc wrt dt_dstau
            #
            rs = np.arange(num_col_nodes * size, dtype=int)
            cs = np.repeat(np.arange(num_col_nodes, dtype=int), size)

            self.declare_partials(of=self.xdotc_str[name], wrt='dt_dstau',
                                  rows=rs, cols=cs)

            self.declare_partials(of=self.xc_str[name], wrt='dt_dstau',
                                  rows=rs, cols=cs)

            Ai_rows, Ai_cols, data = sp.find(self._jacs['Ai'][name])
            self.declare_partials(of=self.xc_str[name], wrt=self.xd_str[name],
                                  rows=Ai_rows, cols=Ai_cols, val=data)

            Bi_rows, Bi_cols = self._jacs['Bi'][name].nonzero()
            self.declare_partials(of=self.xc_str[name], wrt=self.fd_str[name],
                                  rows=Bi_rows, cols=Bi_cols)

            Bd_rows, Bd_cols, data = sp.find(self._jacs['Bd'][name])
            self.declare_partials(of=self.xdotc_str[name], wrt=self.fd_str[name],
                                  rows=Bd_rows, cols=Bd_cols, val=data)

            Ad_rows, Ad_cols = self._jacs['Ad'][name].nonzero()
            self.declare_partials(of=self.xdotc_str[name], wrt=self.xd_str[name],
                                  rows=Ad_rows, cols=Ad_cols)

    def compute(self, inputs, outputs):
        """
        Compute the outputs of the GaussLobattoInterpComp.

        Parameters
        ----------
        inputs : dict[str, ArrayLike]
            Inputs used in the calculation.
        outputs : dict[str, ArrayLike]
            Outputs of the calculation.
        """
        state_options = self.options['state_options']

        dt_dstau = inputs['dt_dstau'][:, np.newaxis]

        num_disc_nodes = self.options['grid_data'].subset_num_nodes['state_disc']
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']

        Ai = self._matrices['Ai']
        Bi = self._matrices['Bi']
        Ad = self._matrices['Ad']
        Bd = self._matrices['Bd']

        for name in state_options:
            shape = state_options[name]['shape']
            size = np.prod(shape)
            xc_str = self.xc_str[name]
            xdotc_str = self.xdotc_str[name]
            xd_str = self.xd_str[name]
            fd_str = self.fd_str[name]

            xd_flat = np.reshape(inputs[xd_str], newshape=(num_disc_nodes, size))
            fd_flat = np.reshape(inputs[fd_str], newshape=(num_disc_nodes, size))

            col_val = Bi.dot(fd_flat) * dt_dstau + Ai.dot(xd_flat)

            outputs[xc_str] = np.reshape(col_val, (num_col_nodes,) + shape)

            col_rate = Ad.dot(xd_flat) / dt_dstau + Bd.dot(fd_flat)

            outputs[xdotc_str] = np.reshape(col_rate, (num_col_nodes,) + shape)

    def compute_partials(self, inputs, partials):
        """
        Compute the partial derivatives of the GaussLobattoInterpComp.

        Parameters
        ----------
        inputs : dict[str, ArrayLike]
            Inputs used in the calculation.
        partials : dict[tuple[str, str], ArrayLike]
            Partials of outputs with respect to inputs.
        """
        ndn = self.options['grid_data'].subset_num_nodes['state_disc']

        Ad = self._matrices['Ad']
        Bi = self._matrices['Bi']

        dstau_dt = np.reciprocal(inputs['dt_dstau'])
        dstau_dt2 = dstau_dt ** 2

        for name in self.options['state_options']:
            size = self._sizes[name]

            xdotc_name = self.xdotc_str[name]
            xd_name = self.xd_str[name]

            xc_name = self.xc_str[name]
            fd_name = self.fd_str[name]

            # Unroll matrix-shaped states into an array at each node
            xd = np.reshape(inputs[xd_name], (ndn, size))
            fd = np.reshape(inputs[fd_name], (ndn, size))

            dt_dstau_x_size = np.repeat(inputs['dt_dstau'], size)[:, np.newaxis]

            partials[xc_name, 'dt_dstau'] = Bi.dot(fd).ravel()

            partials[xdotc_name, 'dt_dstau'] = (-Ad.dot(xd) * dstau_dt2[:, np.newaxis]).ravel()

            dxc_dfd = self._jacs['Bi'][name].multiply(dt_dstau_x_size)
            partials[xc_name, fd_name] = dxc_dfd.data

            dxdotc_dxd = self._jacs['Ad'][name].multiply(np.reciprocal(dt_dstau_x_size))
            partials[xdotc_name, xd_name] = dxdotc_dxd.data
