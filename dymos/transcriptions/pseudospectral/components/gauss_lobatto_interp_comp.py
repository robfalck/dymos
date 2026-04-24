"""Define the GaussLobattoInterpComp class."""
import numpy as np
import scipy.sparse as sp
import openmdao.api as om

from ...grid_data import GridData
from ....utils.misc import get_rate_units
from ...._options import options as dymos_options


class GaussLobattoInterpComp(om.ExplicitComponent):
    r"""
    Hermite interpolation for GaussLobattoNew transcription.

    Accepts state values and rates at discretization nodes and:
    1. Computes interpolated state values at collocation nodes via Hermite interpolation.
    2. Assembles all-nodes state array (disc passthrough + col interpolated) for ODE feedback.
    3. Computes interpolated state rates at collocation nodes (f_approx for defects).

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
        gd = self.options['grid_data']
        state_options = self.options['state_options']
        time_units = self.options['time_units']

        n_disc = gd.subset_num_nodes['state_disc']
        n_col = gd.subset_num_nodes['col']
        n_all = gd.num_nodes
        disc_idxs = gd.subset_node_indices['state_disc']
        col_idxs = gd.subset_node_indices['col']

        self._disc_idxs = disc_idxs
        self._col_idxs = col_idxs
        self._n_disc = n_disc
        self._n_col = n_col
        self._n_all = n_all

        # Hermite interpolation matrices (sparse)
        Ai, Bi, Ad, Bd = gd.phase_hermite_matrices('state_disc', 'col', sparse=True)
        self._matrices = {'Ai': Ai, 'Bi': Bi, 'Ad': Ad, 'Bd': Bd}
        self._jacs = {}

        self.add_input('dt_dstau', shape=(n_col,), units=time_units,
                       desc='For each collocation node, dt/dstau (time-step scaling)')

        for name, options in state_options.items():
            shape = options['shape']
            units = options['units']
            size = int(np.prod(shape))
            rate_units = get_rate_units(units, time_units)

            self.add_input(f'state_disc:{name}',
                           shape=(n_disc,) + shape, units=units,
                           desc=f'State {name} values at discretization nodes')

            self.add_input(f'staterate_disc:{name}',
                           shape=(n_disc,) + shape, units=rate_units,
                           desc=f'State rate {name} at discretization nodes (from ODE)')

            self.add_output(f'states_all:{name}',
                            shape=(n_all,) + shape, units=units,
                            desc=f'State {name} at all nodes (disc passthrough + col interpolated)')

            self.add_output(f'staterate_col:{name}',
                            shape=(n_col,) + shape, units=rate_units,
                            desc=f'Interpolated rate of {name} at collocation nodes (f_approx)')

            # Build Kronecker product jacobian matrices
            Ai_kron = sp.kron(sp.csr_matrix(Ai), sp.eye(size, format='csr'), format='csr')
            Bi_kron = sp.kron(sp.csr_matrix(Bi), sp.eye(size, format='csr'), format='csr')
            Ad_kron = sp.kron(sp.csr_matrix(Ad), sp.eye(size, format='csr'), format='csr')
            Bd_kron = sp.kron(sp.csr_matrix(Bd), sp.eye(size, format='csr'), format='csr')

            self._jacs[name] = {'Ai': Ai_kron, 'Bi': Bi_kron, 'Ad': Ad_kron, 'Bd': Bd_kron}

            # Flat index arrays for all-nodes output assembly
            # disc_all_flat[k] = flat index in states_all for k-th disc element
            disc_all_flat = (np.repeat(disc_idxs, size) * size +
                             np.tile(np.arange(size), n_disc)).astype(int)
            # col_all_flat[k] = flat index in states_all for k-th col element
            col_all_flat = (np.repeat(col_idxs, size) * size +
                            np.tile(np.arange(size), n_col)).astype(int)

            self._disc_all_flat = disc_all_flat
            self._col_all_flat = col_all_flat

            disc_in_flat = np.arange(n_disc * size, dtype=int)

            # --- Partials for states_all ---

            # states_all wrt state_disc:
            #   - disc rows: identity passthrough
            #   - col rows: Ai interpolation matrix
            Ai_nz_rows, Ai_nz_cols = Ai_kron.nonzero()
            Ai_nz_vals = np.asarray(Ai_kron[Ai_nz_rows, Ai_nz_cols]).ravel()
            sa_xd_rows = np.concatenate([disc_all_flat, col_all_flat[Ai_nz_rows]])
            sa_xd_cols = np.concatenate([disc_in_flat, Ai_nz_cols])
            sa_xd_vals = np.concatenate([np.ones(n_disc * size), Ai_nz_vals])
            self.declare_partials(f'states_all:{name}', f'state_disc:{name}',
                                  rows=sa_xd_rows, cols=sa_xd_cols, val=sa_xd_vals)

            # states_all wrt staterate_disc:
            #   - disc rows: zero (disc passthrough doesn't depend on rates)
            #   - col rows: Bi * dt_dstau (dynamic, set in compute_partials)
            Bi_nz_rows, Bi_nz_cols = Bi_kron.nonzero()
            self.declare_partials(f'states_all:{name}', f'staterate_disc:{name}',
                                  rows=col_all_flat[Bi_nz_rows], cols=Bi_nz_cols)

            # states_all wrt dt_dstau:
            #   - only col rows: d(Bi @ fd * dt_dstau[i])/d(dt_dstau[i]) = (Bi @ fd)[i, :]
            sa_dt_rows = col_all_flat  # n_col * size elements
            sa_dt_cols = np.repeat(np.arange(n_col), size).astype(int)
            self.declare_partials(f'states_all:{name}', 'dt_dstau',
                                  rows=sa_dt_rows, cols=sa_dt_cols)

            # --- Partials for staterate_col ---

            # staterate_col wrt state_disc: Ad / dt_dstau (dynamic)
            Ad_nz_rows, Ad_nz_cols = Ad_kron.nonzero()
            self.declare_partials(f'staterate_col:{name}', f'state_disc:{name}',
                                  rows=Ad_nz_rows, cols=Ad_nz_cols)

            # staterate_col wrt staterate_disc: Bd (static)
            Bd_nz_rows, Bd_nz_cols = Bd_kron.nonzero()
            Bd_nz_vals = np.asarray(Bd_kron[Bd_nz_rows, Bd_nz_cols]).ravel()
            self.declare_partials(f'staterate_col:{name}', f'staterate_disc:{name}',
                                  rows=Bd_nz_rows, cols=Bd_nz_cols, val=Bd_nz_vals)

            # staterate_col wrt dt_dstau: -Ad @ xd / dt_dstau^2 (dynamic)
            r = np.arange(n_col * size, dtype=int)
            c = np.repeat(np.arange(n_col), size).astype(int)
            self.declare_partials(f'staterate_col:{name}', 'dt_dstau', rows=r, cols=c)

    def compute(self, inputs, outputs):
        """
        Compute interpolated state values and rates at collocation nodes.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables.
        outputs : Vector
            Unscaled, dimensional output variables.
        """
        gd = self.options['grid_data']
        state_options = self.options['state_options']

        n_disc = self._n_disc
        n_col = self._n_col
        disc_idxs = self._disc_idxs
        col_idxs = self._col_idxs

        Ai = self._matrices['Ai']
        Bi = self._matrices['Bi']
        Ad = self._matrices['Ad']
        Bd = self._matrices['Bd']

        dt_dstau = inputs['dt_dstau'][:, np.newaxis]

        for name, options in state_options.items():
            shape = options['shape']
            size = int(np.prod(shape))

            xd_flat = inputs[f'state_disc:{name}'].reshape(n_disc, size)
            fd_flat = inputs[f'staterate_disc:{name}'].reshape(n_disc, size)

            # Hermite interpolation at collocation nodes
            col_val = Bi.dot(fd_flat) * dt_dstau + Ai.dot(xd_flat)
            col_rate = Ad.dot(xd_flat) / dt_dstau + Bd.dot(fd_flat)

            # Assemble states at all nodes: disc passthrough + col interpolated.
            # Use xd_flat.dtype so the intermediate array is complex during complex-step.
            states_all = np.empty((gd.num_nodes,) + shape, dtype=xd_flat.dtype)
            states_all[disc_idxs] = xd_flat.reshape((n_disc,) + shape)
            states_all[col_idxs] = col_val.reshape((n_col,) + shape)

            outputs[f'states_all:{name}'] = states_all
            outputs[f'staterate_col:{name}'] = col_rate.reshape((n_col,) + shape)

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables.
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        state_options = self.options['state_options']

        n_disc = self._n_disc
        n_col = self._n_col

        Bi = self._matrices['Bi']
        Ad = self._matrices['Ad']

        dt_dstau = inputs['dt_dstau']
        dstau_dt = 1.0 / dt_dstau
        dstau_dt2 = dstau_dt ** 2

        for name, options in state_options.items():
            shape = options['shape']
            size = int(np.prod(shape))

            xd_flat = inputs[f'state_disc:{name}'].reshape(n_disc, size)
            fd_flat = inputs[f'staterate_disc:{name}'].reshape(n_disc, size)

            dt_x_size = np.repeat(dt_dstau, size)
            dstau_x_size = np.repeat(dstau_dt, size)

            Bi_kron = self._jacs[name]['Bi']
            Ad_kron = self._jacs[name]['Ad']

            # states_all wrt staterate_disc: Bi * dt_dstau (scale each kron row by dt)
            partials[f'states_all:{name}', f'staterate_disc:{name}'] = \
                Bi_kron.multiply(dt_x_size[:, np.newaxis]).data

            # states_all wrt dt_dstau: Bi @ fd at col positions
            col_val_from_Bi = Bi.dot(fd_flat)  # shape (n_col, size)
            partials[f'states_all:{name}', 'dt_dstau'] = col_val_from_Bi.ravel()

            # staterate_col wrt state_disc: Ad / dt_dstau
            partials[f'staterate_col:{name}', f'state_disc:{name}'] = \
                Ad_kron.multiply(dstau_x_size[:, np.newaxis]).data

            # staterate_col wrt dt_dstau: -Ad @ xd * (dstau_dt)^2
            partials[f'staterate_col:{name}', 'dt_dstau'] = \
                (-Ad.dot(xd_flat) * dstau_dt2[:, np.newaxis]).ravel()
