"""Define the GaussLobattoDefectComp class."""
import numpy as np

import openmdao.api as om

from ...grid_data import GridData
from ....utils.misc import get_rate_units
from ...._options import options as dymos_options


class GaussLobattoDefectComp(om.ExplicitComponent):
    r"""
    Compute all GaussLobattoNew collocation and boundary-state defects.

    Collocation defects (at collocation nodes):

    .. math::

        \\text{defects}_{name} = (f_{\\text{approx}} - f_{\\text{computed}}) \\cdot dt/d\\tau_s

    Boundary state defects:

    .. math::

        \\text{initial\\_state\\_defects}_{name} =
            \\text{initial\\_states}_{name} - \\text{states\\_all}_{name}[0]

        \\text{final\\_state\\_defects}_{name} =
            \\text{final\\_states}_{name} - \\text{states\\_all}_{name}[-1]

    For non-solve_segments states all three defects are constrained to zero.

    This is the GaussLobattoNew analogue of ``RadauDefectComp``.

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
        self.options.declare('grid_data', types=GridData,
                             desc='Container object for grid info')
        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of the integration variable')

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units.
        """
        gd = self.options['grid_data']
        state_options = self.options['state_options']
        time_units = self.options['time_units']

        n_all = gd.num_nodes
        n_col = gd.subset_num_nodes['col']
        n_segs = gd.num_segments

        self.add_input('dt_dstau', units=time_units, shape=(n_col,))

        self.var_names = {}
        for name in state_options:
            self.var_names[name] = {
                'f_approx': f'f_approx:{name}',
                'f_computed': f'f_computed:{name}',
                'defect': f'defects:{name}',
                'initial_val': f'initial_states:{name}',
                'final_val': f'final_states:{name}',
                'states_all': f'states_all:{name}',
                'initial_defect': f'initial_state_defects:{name}',
                'final_defect': f'final_state_defects:{name}',
            }

        se = gd.subset_node_indices['segment_ends']

        for name, options in state_options.items():
            shape = options['shape']
            units = options['units']
            size = int(np.prod(shape))
            solve_segs = options['solve_segments']
            rate_units = get_rate_units(units, time_units)

            vn = self.var_names[name]

            # --- Collocation defect inputs ---
            self.add_input(vn['f_approx'],
                           shape=(n_col,) + shape, units=rate_units,
                           desc=f'Hermite-interpolated rate of {name} at collocation nodes')
            self.add_input(vn['f_computed'],
                           shape=(n_col,) + shape, units=rate_units,
                           desc=f'ODE-computed rate of {name} at collocation nodes')

            # --- Collocation defect output ---
            self.add_output(vn['defect'],
                            shape=(n_col,) + shape, units=units,
                            desc=f'Collocation defect of state {name}')

            # --- Boundary state inputs ---
            self.add_input(vn['initial_val'],
                           shape=(1,) + shape, units=units,
                           desc=f'Initial boundary value of state {name}')
            self.add_input(vn['final_val'],
                           shape=(1,) + shape, units=units,
                           desc=f'Final boundary value of state {name}')
            self.add_input(vn['states_all'],
                           shape=(n_all,) + shape, units=units,
                           desc=f'State {name} at all nodes')

            # --- Boundary state defect outputs ---
            self.add_output(vn['initial_defect'],
                            shape=(1,) + shape, units=units,
                            desc='Initial state defect: initial_states - states_all[0]')
            self.add_output(vn['final_defect'],
                            shape=(1,) + shape, units=units,
                            desc='Final state defect: final_states - states_all[-1]')

            # --- Determine defect scaling reference ---
            if 'defect_ref' in options and options['defect_ref'] is not None:
                defect_ref = np.atleast_1d(options['defect_ref'])
            elif 'defect_scaler' in options and options['defect_scaler'] is not None:
                defect_ref = np.divide(1.0, np.atleast_1d(options['defect_scaler']))
            elif 'ref' in options and options['ref'] is not None:
                defect_ref = np.atleast_1d(options['ref'])
            elif 'scaler' in options and options['scaler'] is not None:
                defect_ref = np.divide(1.0, np.atleast_1d(options['scaler']))
            else:
                defect_ref = 1.0

            if np.isscalar(defect_ref):
                defect_ref = defect_ref * np.ones(shape)

            if defect_ref.shape != shape:
                raise ValueError(
                    f'array-valued scaler/ref/defect_ref for state {name} must be the same '
                    f'shape as the state')

            col_defect_ref = np.tile(defect_ref, (n_col, 1))

            # --- Add constraints for non-solve_segments states ---
            if not solve_segs:
                self.add_constraint(vn['defect'], equals=0.0, ref=col_defect_ref)
                # Boundary conditions at initial/final nodes are enforced by excluding those
                # nodes from the states:{name} design variable in _configure_desvars.
                # No boundary defect constraints are needed.

            # --- Declare partials ---
            r = np.arange(n_col * size)
            c = np.repeat(np.arange(n_col), size)

            # defect wrt f_approx and f_computed (dynamic, set in compute_partials)
            self.declare_partials(vn['defect'], vn['f_approx'], rows=r, cols=r)
            self.declare_partials(vn['defect'], vn['f_computed'], rows=r, cols=r)
            self.declare_partials(vn['defect'], 'dt_dstau', rows=r, cols=c)

            ar = np.arange(size, dtype=int)

            # initial_defect wrt initial_states: +I
            self.declare_partials(vn['initial_defect'], vn['initial_val'],
                                  rows=ar, cols=ar, val=1.0)
            # initial_defect wrt states_all: -I at first node (flat cols 0..size-1)
            self.declare_partials(vn['initial_defect'], vn['states_all'],
                                  rows=ar, cols=ar, val=-1.0)

            # final_defect wrt final_states: +I
            self.declare_partials(vn['final_defect'], vn['final_val'],
                                  rows=ar, cols=ar, val=1.0)
            # final_defect wrt states_all: -I at last node
            last_node_cols = (n_all - 1) * size + ar
            self.declare_partials(vn['final_defect'], vn['states_all'],
                                  rows=ar, cols=last_node_cols, val=-1.0)

            # --- Segment continuity defects (uncompressed multi-segment only) ---
            # cnty_defect[i] = states_all[seg_start[i+1]] - states_all[seg_end[i]]
            # where seg_start and seg_end are the disc nodes at segment boundaries.
            if n_segs > 1 and not gd.compressed:
                ns_m1 = n_segs - 1
                vn['cnty_defect'] = f'state_cnty_defects:{name}'

                self.add_output(vn['cnty_defect'],
                                shape=(ns_m1,) + shape, units=units,
                                desc=f'Segment continuity defect of state {name}')

                if not solve_segs:
                    cnty_ref = np.tile(defect_ref, (ns_m1, 1))
                    self.add_constraint(vn['cnty_defect'], equals=0.0, ref=cnty_ref)

                # se = [seg0_start, seg0_end, seg1_start, seg1_end, ...]
                pos_start = se[2::2]    # start node of each subsequent segment
                neg_end = se[1:-2:2]    # end node of each preceding segment

                out_r = np.arange(ns_m1 * size)
                pos_c = np.tile(np.arange(size), ns_m1) + np.repeat(pos_start * size, size)
                neg_c = np.tile(np.arange(size), ns_m1) + np.repeat(neg_end * size, size)

                self.declare_partials(vn['cnty_defect'], vn['states_all'],
                                      rows=np.concatenate([out_r, out_r]),
                                      cols=np.concatenate([pos_c, neg_c]),
                                      val=np.concatenate([np.ones(ns_m1 * size),
                                                          -np.ones(ns_m1 * size)]))

    def compute(self, inputs, outputs):
        """
        Compute collocation and boundary-state defects.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables.
        outputs : Vector
            Unscaled, dimensional output variables.
        """
        dt_dstau = inputs['dt_dstau']

        gd = self.options['grid_data']
        se = gd.subset_node_indices['segment_ends']

        for name, vn in self.var_names.items():
            f_approx = inputs[vn['f_approx']]
            f_computed = inputs[vn['f_computed']]
            outputs[vn['defect']] = ((f_approx - f_computed).T * dt_dstau).T

            x0 = inputs[vn['initial_val']]     # (1,)+shape
            xf = inputs[vn['final_val']]       # (1,)+shape
            x = inputs[vn['states_all']]       # (n_all,)+shape
            outputs[vn['initial_defect']] = x0 - x[[0], ...]
            outputs[vn['final_defect']] = xf - x[[-1], ...]

            if 'cnty_defect' in vn:
                outputs[vn['cnty_defect']] = x[se[2::2]] - x[se[1:-2:2]]

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables.
        partials : Jacobian
            Sub-jacobian components.
        """
        dt_dstau = inputs['dt_dstau']

        for name, options in self.options['state_options'].items():
            size = int(np.prod(options['shape']))
            vn = self.var_names[name]

            f_approx = inputs[vn['f_approx']]
            f_computed = inputs[vn['f_computed']]

            k = np.repeat(dt_dstau, size)
            partials[vn['defect'], vn['f_approx']] = k
            partials[vn['defect'], vn['f_computed']] = -k
            partials[vn['defect'], 'dt_dstau'] = (f_approx - f_computed).ravel()
            # boundary defect partials are all constant (declared with val=), no update needed
