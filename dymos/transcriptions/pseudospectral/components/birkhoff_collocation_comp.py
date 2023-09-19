import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos._options import options as dymos_options
from dymos.utils.lgl import lgl
from dymos.utils.lgr import lgr
from dymos.utils.cgl import cgl
from dymos.utils.birkhoff import birkhoff_matrix


class BirkhoffCollocationComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')

    def configure_io(self, phase):
        """
        I/O creation is delayed until configure so we can determine shape and units.
        """
        gd = self.options['grid_data']
        num_segs = gd.num_segments
        num_nodes = gd.subset_num_nodes['col']
        time_units = self.options['time_units']
        state_options = self.options['state_options']

        self.add_input('dt_dstau', units=time_units, shape=(gd.subset_num_nodes['col'],))

        self.var_names = var_names = {}
        for state_name, options in state_options.items():
            var_names[state_name] = {
                'f_value': f'state_rates:{state_name}',
                'f_computed': f'f_computed:{state_name}',
                'state_value': f'states:{state_name}',
                'state_initial_value': f'initial_states:{state_name}',
                'state_final_value': f'final_states:{state_name}',
                'state_defect': f'state_defects:{state_name}',
                'state_rate_defect': f'state_rate_defects:{state_name}',
                'final_state_defect': f'final_state_defects:{state_name}',
                'state_continuity_defect': f'state_cnty_defects:{state_name}',
                'state_rate_continuity_defect': f'state_rate_cnty_defects:{state_name}',
            }

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)
            var_names = self.var_names[state_name]

            rate_source = options['rate_source']
            rate_source_type = phase.classify_var(rate_source)

            self.add_input(
                name=var_names['f_value'],
                shape=(num_nodes,) + shape,
                desc=f'Estimated derivative of state {state_name} at the polynomial nodes',
                units=units)

            if rate_source_type == 'state':
                var_names['f_computed'] = f'states:{rate_source}'
            else:
                self.add_input(
                    name=var_names['f_computed'],
                    shape=(num_nodes,) + shape,
                    desc=f'Computed derivative of state {state_name} at the polynomial nodes',
                    units=rate_units)

            self.add_input(
                name=var_names['state_value'],
                shape=(num_nodes,) + shape,
                desc=f'Value of the state {state_name} at the polynomial nodes',
                units=units
            )

            self.add_input(
                name=var_names['state_initial_value'],
                shape=(num_segs,) + shape,
                desc=f'Desired initial value of state {state_name}',
                units=units
            )

            self.add_input(
                name=var_names['state_final_value'],
                shape=(num_segs,) + shape,
                desc=f'Desired final value of state {state_name}',
                units=units
            )

            self.add_output(
                name=var_names['state_defect'],
                shape=(num_nodes + num_segs,) + shape,
                units=units
            )

            # self.declare_partials(of=var_names['state_defect'], wrt='*', method='fd')

            self.add_output(
                name=var_names['state_rate_defect'],
                shape=(num_nodes,) + shape,
                units=units
            )

            # self.declare_partials(of=var_names['state_rate_defect'], wrt='*', method='fd')

            # self.add_output(
            #     name=var_names['final_state_defect'],
            #     shape=(num_segs,) + shape,
            #     units=units
            # )

            if num_segs > 1:
                self.add_output(
                    name=var_names['state_continuity_defect'],
                    shape=(num_segs-1,) + shape,
                    units=units
                )

                rs = cs = np.arange(num_segs - 1, dtype=int)

                self.declare_partials(of=var_names['state_continuity_defect'],
                                      wrt=var_names['state_initial_value'],
                                      rows=rs,
                                      cols=cs + 1,
                                      val=-1)

                self.declare_partials(of=var_names['state_continuity_defect'],
                                      wrt=var_names['state_final_value'],
                                      rows=rs,
                                      cols=cs,
                                      val=1)

                self.add_output(
                    name=var_names['state_rate_continuity_defect'],
                    shape=(num_segs-1,) + shape,
                    units=units
                )

                # _xv_idxs is a set of indices that arranges the stacked
                # [[X^T],[V^T]] arrays into a segment-by-segment ordering
                # instead of all of the state values followed by all of the state rates.
                self._xv_idxs = []

                # _ab_idxs is a set of indices that arranges the stacked [x_a, x_b] arrays
                # into segment-by-segment ordering
                self._ab_idxs = []
                idx0 = 0
                for i in range(num_segs):
                    nnps = gd.subset_num_nodes_per_segment['all'][i]
                    ar_nnps = np.arange(nnps, dtype=int)
                    self._xv_idxs.extend(idx0 + ar_nnps)
                    self._xv_idxs.extend(idx0 + ar_nnps + num_nodes)
                    idx0 += nnps
                    self._ab_idxs.extend([i, i + num_segs])
            else:
                self._xv_idxs = np.arange(2 * num_nodes, dtype=int)
                self._ab_idxs = np.arange(2 * num_segs, dtype=int)

            if 'defect_ref' in options and options['defect_ref'] is not None:
                defect_ref = options['defect_ref']
            elif 'defect_scaler' in options and options['defect_scaler'] is not None:
                defect_ref = np.divide(1.0, options['defect_scaler'])
            else:
                if 'ref' in options and options['ref'] is not None:
                    defect_ref = options['ref']
                elif 'scaler' in options and options['scaler'] is not None:
                    defect_ref = np.divide(1.0, options['scaler'])
                else:
                    defect_ref = 1.0

            if not np.isscalar(defect_ref):
                defect_ref = np.asarray(defect_ref)
                if defect_ref.shape == shape:
                    defect_ref = np.tile(defect_ref.flatten(), num_nodes)
                else:
                    raise ValueError('array-valued scaler/ref must length equal to state-size')

            if not options['solve_segments']:
                self.add_constraint(name=var_names['state_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                self.add_constraint(name=var_names['state_rate_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                # self.add_constraint(name=var_names['final_state_defect'],
                #                     equals=0.0,
                #                     ref=defect_ref)

                if gd.num_segments > 1:
                    self.add_constraint(name=var_names['state_continuity_defect'],
                                        equals=0.0,
                                        ref=defect_ref)

        if gd.grid_type == 'lgl':
            tau, w = lgl(num_nodes)
        elif gd.grid_type == 'lgr':
            tau, w = lgr(num_nodes, include_endpoint=False)
        elif gd.grid_type == 'cgl':
            tau, w = cgl(num_nodes)
        else:
            raise ValueError('invalid grid type')
        
        node_funcs = {'lgl': lgl, 'cgl': cgl}

        A_blocks = []
        B_blocks = []
        C_blocks = []
        for i in range(num_segs):
            num_nodes_i = gd.subset_num_nodes_per_segment['all'][i]
            tau_i, w_i = node_funcs[gd.grid_type](num_nodes_i)
            B_i = birkhoff_matrix(tau_i, w_i, grid_type=gd.grid_type)

            A_i = np.zeros((num_nodes_i + 1, 2 * num_nodes_i))
            A_i[:num_nodes_i, :num_nodes_i] = np.eye(num_nodes_i)
            A_i[:num_nodes_i, num_nodes_i:] = -B_i
            A_i[-1, num_nodes_i:] = B_i[-1, :]

            C_i = np.zeros((num_nodes_i + 1, 2))
            C_i[:-1, 0] = 1.
            C_i[-1, :] = [-1, 1]

            A_blocks.append(A_i)
            B_blocks.append(B_i)
            C_blocks.append(C_i)

        # self._A = sp.csr_matrix(block_diag(*A_blocks))
        # self._B = sp.csr_matrix(block_diag(*B_blocks))
        # self._C = sp.csr_matrix(block_diag(*C_blocks))

        self._A = block_diag(*A_blocks)
        self._B = block_diag(*B_blocks)
        self._C = block_diag(*C_blocks)
            
        # B = birkhoff_matrix(tau, w, grid_type=gd.grid_type)

        # self._A = np.zeros((num_nodes + 1, 2 * num_nodes))
        # self._A[:num_nodes, :num_nodes] = np.eye(num_nodes)
        # self._A[:num_nodes, num_nodes:] = -B
        # self._A[-1, num_nodes:] = B[-1, :]

        # self._C = np.zeros((num_nodes + 1, 2))
        # self._C[:-1, 0] = 1.
        # self._C[-1, :] = [-1, 1]

        # Setup partials

        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)

            ar1 = np.arange(num_nodes * size)
            ar2 = np.arange(size)
            c1 = np.repeat(np.arange(num_nodes), size)

            rB = np.zeros(num_nodes*num_nodes*size)
            for j in range(num_nodes):
                rB[j*num_nodes*size:(j+1)*num_nodes*size] = np.tile(np.arange(size), num_nodes) + j*size
            cB = np.tile(np.arange(num_nodes*size), num_nodes)

            var_names = self.var_names[state_name]

            d_state_defect_dXV = self._A
            dXV_dX = np.vstack((np.eye(num_nodes), np.zeros((num_nodes, num_nodes))))[self._xv_idxs]
            dXV_dV = np.vstack((np.zeros((num_nodes, num_nodes)), np.eye(num_nodes)))[self._xv_idxs]
            d_state_defect_dX = np.dot(d_state_defect_dXV, dXV_dX)
            d_state_defect_dV = np.dot(d_state_defect_dXV, dXV_dV)
            rs_dX, cs_dX = sp.csr_matrix(d_state_defect_dX).nonzero()
            rs_dV, cs_dV = sp.csr_matrix(d_state_defect_dV).nonzero()
            val_dV = sp.csr_matrix(d_state_defect_dV).data

            # with np.printoptions(linewidth=1024, edgeitems=1000):
            #     print(d_state_defect_dX)
            #     print(d_state_defect_dX.shape)
            #     rs, cs = sp.csr_matrix(d_state_defect_dX).nonzero()
            #     print(rs)
            #     print(cs)
            #     exit(0)

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['state_value'],
                                  rows=rs_dX, cols=cs_dX, val=1.0)
            #
            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['f_value'],
                                  rows=rs_dV, cols=cs_dV, val=val_dV)
                                  # method='fd')
                                  # rows=rB,
                                  # cols=cB,
                                  # val=np.repeat(-self._B, size))

            dstate_defect_dXAB = -self._C
            dXAB_dxa = np.vstack((np.eye(num_segs), np.zeros((num_segs, num_segs))))[self._ab_idxs]
            dXAB_dxb = np.vstack((np.zeros((num_segs, num_segs)), np.eye(num_segs)))[self._ab_idxs]
            d_state_defect_dxa = np.dot(dstate_defect_dXAB, dXAB_dxa)
            d_state_defect_dxb = np.dot(dstate_defect_dXAB, dXAB_dxb)

            rs_dxa, cs_dxa = sp.csr_matrix(d_state_defect_dxa).nonzero()
            val_dxa = sp.csr_matrix(d_state_defect_dxa).data

            rs_dxb, cs_dxb = sp.csr_matrix(d_state_defect_dxb).nonzero()
            val_dxb = sp.csr_matrix(d_state_defect_dxb).data

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['state_initial_value'],
                                  rows=rs_dxa, cols=cs_dxa, val=val_dxa)

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['state_final_value'],
                                  rows=rs_dxb, cols=cs_dxb, val=val_dxb)

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt=var_names['f_value'],
                                  rows=ar1, cols=ar1, val=1.0)

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt=var_names['f_computed'],
                                  rows=ar1, cols=ar1)

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt='dt_dstau',
                                  rows=ar1, cols=c1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dt_dstau = np.atleast_2d(inputs['dt_dstau']).T
        num_segs = self.options['grid_data'].num_segments

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]

            x_a = inputs[var_names['state_initial_value']]
            x_b = inputs[var_names['state_final_value']]
            X = inputs[var_names['state_value']]
            V = inputs[var_names['f_value']]
            f = inputs[var_names['f_computed']]

            # Stack initial/final segment state values on a segment-by-segment basis.
            X_AB = np.vstack((x_a, x_b))[self._ab_idxs, ...]

            # Stack state value/rate values on a segment-by-segment basis.
            XV = np.vstack((X, V))[self._xv_idxs, ...]

            outputs[var_names['state_defect']] = np.dot(self._A, XV) - np.dot(self._C, X_AB)
            outputs[var_names['state_rate_defect']] = (V - np.einsum('i...,i...->i...', f, dt_dstau))

            if num_segs > 1:
                outputs[var_names['state_continuity_defect']] = x_b[:-1, ...] - x_a[1:, ...]

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        dt_dstau = inputs['dt_dstau']

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]
            shape = options['shape']
            size = np.prod(shape)
            f = inputs[var_names['f_computed']]

            partials[var_names['state_rate_defect'], var_names['f_computed']] = np.repeat(-dt_dstau, size)
            partials[var_names['state_rate_defect'], 'dt_dstau'] = -f.ravel()
