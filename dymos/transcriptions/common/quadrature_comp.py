import functools

import numpy as np
import openmdao.api as om
from openmdao.utils.units import simplify_unit

from ..._options import options as dymos_options


class QuadratureComp(om.ExplicitComponent):
    r"""
    Class definition for the QuadratureComp.

    Compute an integrated value at the ends of the phase via quadrature.
    Unlike states, quadrature values are computed explicitly.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    Notes
    -----
    .. math::

        q_f = q_0 + \sum_{i=0}^{n} \omega_i f_i

        q_i = q_f - \sum_{i=0}^{n} \omega_i f_i

    where
    :math:`q_i` is the initial quadrature value when direction='backward',
    :math:`q_f` is the final quadrature value when direction='forward',
    :math:`q_0` is the initial quadrature value,
    :math:`\omega_i` are the polynomial weights of each node,
    :math:`\f_i` are the integrand values at each node,
    :math:`n` is the number of polynomial nodes in the phase,
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

        self._initial_names = {}
        self._final_names = {}
        self._input_names = {}
        self._output_names = {}

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare(
            'quadrature_options', types=dict,
            desc='Dictionary of options for the quadrature variables')

    def setup(self):
        """
        Perform setup procedure for the QuadratureComp.

        All IO is added during phase configuration.
        """
        pass

    def _shape_func(self, name, shapes):
        in_name = self._input_names[name]
        in_shape = shapes[in_name]
        if len(in_shape) == 1:
            # Input is just an n vector
            return (1,)
        return in_shape[1:]

    def configure_io(self, phase):
        quad_options = self.options['quadrature_options']
        time_units = phase.time_options['units']

        self.add_input('t_duration', units=time_units)

        for name, options in quad_options.items():

            self._initial_names[name] = f'initial_quadratures:{name}'
            self._final_names[name] = f'final_quadratures:{name}'
            self._input_names[name] = f'input_values:{name}'
            self._output_names[name] = f'{name}'

            input_name = self._input_names[name]

            units = options['units']
            out_units = simplify_unit(f'{units}*{time_units}')

            self.add_input(input_name, shape_by_conn=True, units=units)

            if options['direction'] == 'forward':
                self.add_input(self._initial_names[name], val=0.0, shape_by_conn=True,
                               copy_shape=self._output_names[name])
            else:
                self.add_input(self._final_names[name], val=0.0, shape_by_conn=True,
                               copy_shape=self._output_names[name])

            shape_func = functools.partial(self._shape_func, name=name)

            self.add_output(self._output_names[name], units=out_units,
                            compute_shape=shape_func)

    def setup_partials(self):
        gd = self.options['grid_data']
        w = gd.node_weight

        quad_options = self.options['quadrature_options']

        for name, options in quad_options.items():
            in_shape = self._get_var_meta(self._input_names[name], 'shape')
            ndim = max(2, len(in_shape) - 1)
            if options['direction'] == 'forward':
                k = 1.0
                self.declare_partials(of=self._output_val_names[name],
                                      wrt=self._initial_names[name], val=1.0)
            else:
                k = -1.0
                self.declare_partials(of=self._output_val_names[name],
                                      wrt=self._final_names[name], val=1.0)
            self.declare_partials(of=self._output_val_names[name],
                                  wrt=self._input_names[name],
                                  val=k * w.reshape(-1, *[1] * (ndim - 1)))

    def compute(self, inputs, outputs):
        """
        Compute interpolated control values and rates.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        gd = self.options['grid_data']
        quadrature_options = self.options['quadrature_options']

        w = gd.node_weight

        for name, options in quadrature_options.items():
            _inp = inputs[self._input_names[name]]
            quad = np.tensordot(w, _inp, axes=(0, 0))
            if options['direction'] == 'forward':
                v0 = inputs[self._initial_names[name]]
                k = 1.0
            else:
                v0 = inputs[self._final_names[name]]
                k = -1.0
            outputs[name] = v0 + k * quad
