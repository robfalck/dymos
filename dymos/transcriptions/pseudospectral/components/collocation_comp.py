"""Define the CollocationComp class."""
import numpy as np

import openmdao.api as om

from ...grid_data import GridData
from ....utils.misc import get_rate_units
from ...._options import options as dymos_options


class CollocationComp(om.ExplicitComponent):
    """
    Class definiton for the Collocationcomp.

    CollocationComp computes the generalized defects for implicit collocation.
    The defects is the interpolated state derivative at the collocation nodes minus
    the computed state derivative at the collocation nodes, in dimensionless
    "segment tau" time.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
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
        num_nodes = gd.subset_num_nodes['all']
        num_col_nodes = gd.subset_num_nodes['col']
        num_segs = gd.num_segments
        compressed = gd.compressed
        time_units = self.options['time_units']
        state_options = self.options['state_options']

        self.add_input('dt_dstau', units=time_units, shape=(num_col_nodes,))

        self.var_names = var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'f_approx': f'f_approx:{state_name}',
                'f_computed': f'f_computed:{state_name}',
                'initial_val': f'initial_states:{state_name}',
                'final_val': f'final_states:{state_name}',
                'val': f'states:{state_name}',
                'defect': f'state_rate_defects:{state_name}',
                'cnty_defect': f'state_cnty_defects:{state_name}',
                'initial_defect': f'initial_state_defects:{state_name}',
                'final_defect': f'final_state_defects:{state_name}'
            }

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            var_names = self.var_names[state_name]

            self.add_input(name=var_names['initial_val'],
                           shape=(1,) + shape,
                           units=units,
                           desc='Initial value of the state at the start of the phase.')

            self.add_input(name=var_names['final_val'],
                           shape=(1,) + shape,
                           units=units,
                           desc='Final value of the state at the end of the phase.')

            self.add_input(var_names['val'],
                           shape=(num_nodes,) + shape,
                           units=units,
                           desc='state value at all nodes within the phase')

            self.add_input(
                name=var_names['f_approx'],
                shape=(num_col_nodes,) + shape,
                desc=f'Estimated derivative of state {state_name} at the collocation nodes',
                units=rate_units)

            self.add_input(
                name=var_names['f_computed'],
                shape=(num_col_nodes,) + shape,
                desc=f'Computed derivative of state {state_name} at the collocation nodes',
                units=rate_units)

            self.add_output(
                name=var_names['defect'],
                shape=(num_col_nodes,) + shape,
                desc=f'Interior defects of state {state_name}',
                units=units)

            self.add_output(
                name=var_names['initial_defect'],
                shape=(1,) + shape,
                desc=f'Initial value defect of state {state_name}',
                units=units)

            self.add_output(
                name=var_names['final_defect'],
                shape=(1,) + shape,
                desc=f'Final value defect of state {state_name}',
                units=units)

            if not compressed:
                self.add_output(
                    name=var_names['cnty_defect'],
                    shape=(num_segs-1,) + shape,
                    desc=f'Segment continuity defect of state {state_name}',
                    units=units)

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
                    defect_ref = np.tile(defect_ref.flatten(), num_col_nodes)
                else:
                    raise ValueError('array-valued scaler/ref must length equal to state-size')

            if not options['solve_segments']:
                self.add_constraint(name=var_names['defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                if not compressed:
                    self.add_constraint(name=var_names['cnty_defect'],
                                        equals=0.0,
                                        ref=defect_ref)

        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)

            r = np.arange(num_col_nodes * size)

            var_names = self.var_names[state_name]

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_approx'],
                                  rows=r, cols=r)

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_computed'],
                                  rows=r, cols=r)

            c = np.repeat(np.arange(num_col_nodes), size)
            self.declare_partials(of=var_names['defect'],
                                  wrt='dt_dstau',
                                  rows=r, cols=c)

            ar_size = np.arange(size, dtype=int)
            ar_last_node = np.arange(size * num_nodes, dtype=int)[-size:]

            self.declare_partials(of=var_names['initial_defect'],
                                  wrt=var_names['initial_val'],
                                  rows=ar_size, cols=ar_size, val=1.0)

            self.declare_partials(of=var_names['initial_defect'],
                                  wrt=var_names['val'],
                                  rows=ar_size, cols=ar_size, val=-1.0)

            self.declare_partials(of=var_names['final_defect'],
                                  wrt=var_names['final_val'],
                                  rows=ar_size, cols=ar_size, val=1.0)

            self.declare_partials(of=var_names['final_defect'],
                                  wrt=var_names['val'],
                                  rows=ar_size, cols=ar_last_node, val=-1.0)

    def compute(self, inputs, outputs):
        """
        Compute collocation defects.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        state_options = self.options['state_options']
        dt_dstau = inputs['dt_dstau']

        for state_name in state_options:
            var_names = self.var_names[state_name]

            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            val = inputs[var_names['val']]
            initial_val = inputs[var_names['initial_val']]
            final_val = inputs[var_names['final_val']]

            outputs[var_names['defect']] = ((f_approx - f_computed).T * dt_dstau).T

            outputs[var_names['initial_defect']] = (initial_val - val[0, ...])
            outputs[var_names['final_defect']] = (final_val - val[-1, ...])

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
            size = np.prod(options['shape'])
            var_names = self.var_names[state_name]
            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            k = np.repeat(dt_dstau, size)

            partials[var_names['defect'], var_names['f_approx']] = k
            partials[var_names['defect'], var_names['f_computed']] = -k
            partials[var_names['defect'], 'dt_dstau'] = (f_approx - f_computed).ravel()
