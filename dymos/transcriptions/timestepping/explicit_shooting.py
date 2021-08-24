from fnmatch import filter
import warnings

import numpy as np

import openmdao.api as om

from ..transcription_base import TranscriptionBase
from ..grid_data import GridData
from .euler_integration_comp import EulerIntegrationComp
from ..common import TimeComp
from ...utils.misc import get_rate_units, get_target_metadata, get_source_metadata, \
    _unspecified, CoerceDesvar
from ...utils.constants import INF_BOUND
from ...utils.introspection import get_targets
from ...utils.indexing import get_src_indices_by_row


class ExplicitShooting(TranscriptionBase):
    """
    The Transcription class for explicit shooting methods.

    Parameters
    ----------
    grid_data : GridData
        Grid data for this phases.
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super(ExplicitShooting, self).__init__(**kwargs)
        self._rhs_source = 'ode'

    def initialize(self):
        """
        Declare transcription options.
        """
        self.options.declare('grid', values=['radau-ps', 'gauss-lobatto'],
                             default='gauss-lobatto', desc='The type of transcription used to layout'
                             ' the segments and control discretization nodes.')
        self.options.declare('num_steps_per_segment', types=int,
                             default=10, desc='Number of integration steps in each segment')

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription=self.options['grid'],
                                  transcription_order=self.options['order'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

    def setup_time(self, phase):
        """
        Setup the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase.check_time_options()

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        print('ExplicitShooting: configure time')
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp.configure_time_io()

        time_options = phase.time_options

        if not (time_options['input_initial'] or time_options['fix_initial']):
            lb, ub = time_options['initial_bounds']
            lb = -INF_BOUND if lb is None else lb
            ub = INF_BOUND if ub is None else ub

            phase.add_design_var('t_initial',
                                 lower=lb,
                                 upper=ub,
                                 scaler=time_options['initial_scaler'],
                                 adder=time_options['initial_adder'],
                                 ref0=time_options['initial_ref0'],
                                 ref=time_options['initial_ref'])

        if not (time_options['input_duration'] or time_options['fix_duration']):
            lb, ub = time_options['duration_bounds']
            lb = -INF_BOUND if lb is None else lb
            ub = INF_BOUND if ub is None else ub

            phase.add_design_var('t_duration',
                                 lower=lb,
                                 upper=ub,
                                 scaler=time_options['duration_scaler'],
                                 adder=time_options['duration_adder'],
                                 ref0=time_options['duration_ref0'],
                                 ref=time_options['duration_ref'])

        # # The tuples here are (name, user_specified_targets, dynamic)
        # for name, usr_tgts, dynamic in [('time', options['targets'], True),
        #                                 ('time_phase', options['time_phase_targets'], True),
        #                                 ('t_initial', options['t_initial_targets'], False),
        #                                 ('t_duration', options['t_duration_targets'], False)]:
        #
        #     targets = get_targets(phase.ode, name=name, user_targets=usr_tgts)
        #     if targets:
        #         phase.connect(name, [f'ode.{t}' for t in targets])

        # super(ExplicitShooting, self).configure_time(phase)
        # num_seg = self.grid_data.num_segments
        # grid_data = self.grid_data
        # output_nodes_per_seg = self.options['output_nodes_per_seg']
        #
        # phase.time.configure_io()
        #
        # for i in range(num_seg):
        #     phase.connect('t_initial', f'segment_{i}.t_initial')
        #     phase.connect('t_duration', f'segment_{i}.t_duration')
        #     if output_nodes_per_seg is None:
        #         i1, i2 = grid_data.subset_segment_indices['all'][i, :]
        #         src_idxs = grid_data.subset_node_indices['all'][i1:i2]
        #     else:
        #         src_idxs = np.arange(i * output_nodes_per_seg, output_nodes_per_seg * (i + 1),
        #                              dtype=int)
        #     phase.connect('time', f'segment_{i}.time', src_indices=src_idxs)
        #     phase.connect('time_phase', f'segment_{i}.time_phase', src_indices=src_idxs)
        #
        # options = phase.time_options
        #
        # # The tuples here are (name, user_specified_targets, dynamic)
        # for name, usr_tgts, dynamic in [('time', options['targets'], True),
        #                                 ('time_phase', options['time_phase_targets'], True),
        #                                 ('t_initial', options['t_initial_targets'], False),
        #                                 ('t_duration', options['t_duration_targets'], False)]:
        #
        #     targets = get_targets(phase.ode, name=name, user_targets=usr_tgts)
        #     if targets:
        #         phase.connect(name, [f'ode.{t}' for t in targets])

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        # phase.add_subsystem('indep_states', om.IndepVarComp(),
        #                     promotes_outputs=['*'])
        pass

    def configure_states(self, phase):
        """
        Configure state connections post-introspection.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        print('ExplicitShooting: configure states')
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp.configure_states_io()

        # Add the appropriate design parameters
        for state_name, options in phase.state_options.items():
            if options['fix_final']:
                raise ValueError('fix_final is not a valid option for states when using the '
                                 'ExplicitShooting transcription.')
            if options['opt'] and not options['fix_initial']:
                phase.add_design_var(name=f'states:{state_name}',
                                     lower=options['lower'],
                                     upper=options['upper'],
                                     scaler=options['scaler'],
                                     adder=options['adder'],
                                     ref0=options['ref0'],
                                     ref=options['ref'])

    def _get_ode(self, phase):
        integrator = phase._get_subsystem('integrator')
        subprob = integrator._prob
        ode = subprob.model._get_subsystem('ode_eval.ode')
        return ode

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        integrator_comp = EulerIntegrationComp(ode_class=phase.options['ode_class'],
                                               time_options=phase.time_options,
                                               state_options=phase.state_options,
                                               parameter_options=phase.parameter_options,
                                               control_options=phase.control_options,
                                               polynomial_control_options=phase.polynomial_control_options,
                                               num_steps_per_segment=self.options['num_steps_per_segment'],
                                               grid_data=self.grid_data,
                                               ode_init_kwargs=phase.options['ode_init_kwargs'],
                                               standalone_mode=False,
                                               complex_step_mode=True)

        phase.add_subsystem(name='integrator', subsys=integrator_comp, promotes_inputs=['*'])

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        print('ExplicitShooting: configure ode')
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_storage()

    def setup_controls(self, phase):
        """
        Setup the control group.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._check_control_options()

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_controls(phase)
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp.configure_controls_io()

        # Add the appropriate design parameters
        ncin = self.grid_data.subset_num_nodes['control_input']
        for control_name, options in phase.control_options.items():
            if options['opt']:
                coerce_desvar_option = CoerceDesvar(num_input_nodes=ncin, options=options)

                phase.add_design_var(name=f'controls:{control_name}',
                                     lower=coerce_desvar_option('lower'),
                                     upper=coerce_desvar_option('upper'),
                                     scaler=coerce_desvar_option('scaler'),
                                     adder=coerce_desvar_option('adder'),
                                     ref0=coerce_desvar_option('ref0'),
                                     ref=coerce_desvar_option('ref'),
                                     indices=coerce_desvar_option.desvar_indices)

    def setup_polynomial_controls(self, phase):
        """
        Adds the polynomial control group to the model if any polynomial controls are present.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        print('ExplicitShooting: configure polynomial controls')
        super(ExplicitShooting, self).configure_polynomial_controls(phase)
        # # In transcription_base, we get the control units/shape from the target, and then call
        # # configure on the control_group.
        # super(SolveIVP, self).configure_polynomial_controls(phase)
        #
        # # Additional connections.
        # for name, options in phase.polynomial_control_options.items():
        #
        #     for iseg in range(self.grid_data.num_segments):
        #         phase.connect(src_name=f'polynomial_controls:{name}',
        #                       tgt_name=f'segment_{iseg}.polynomial_controls:{name}')
        #
        #     targets = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])
        #     if targets:
        #         phase.connect(f'polynomial_control_values:{name}', [f'ode.{t}' for t in targets])
        #
        #     targets = get_targets(ode=phase.ode, name=f'{name}_rate',
        #                           user_targets=options['rate_targets'])
        #     if targets:
        #         phase.connect(f'polynomial_control_rates:{name}_rate',
        #                       [f'ode.{t}' for t in targets])
        #
        #     targets = get_targets(ode=phase.ode, name=f'{name}_rate2',
        #                           user_targets=options['rate2_targets'])
        #     if targets:
        #         phase.connect(f'polynomial_control_rates:{name}_rate2',
        #                       [f'ode.{t}' for t in targets])

    def configure_parameters(self, phase):
        """
        Configure parameter promotion.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        print('ExplicitShooting: configure parameters')
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp.configure_parameters_io()

    def setup_defects(self, phase):
        """
        Not used in ExplicitShooting

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_defects(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_objective(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_objective(phase)

    def setup_path_constraints(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_path_constraints(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_boundary_constraints(self, loc, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        loc : str
            The kind of boundary constraints being setup.  Must be one of 'initial' or 'final'.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().setup_boundary_constraints(loc, phase)

    def configure_boundary_constraints(self, loc, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        loc : str
            The kind of boundary constraints being setup.  Must be one of 'initial' or 'final'.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_boundary_constraints(loc, phase)

    def setup_solvers(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_solvers(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_timeseries_outputs(self, phase):
        """
        Setup the timeseries for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass
        # gd = self.grid_data
        #
        # timeseries_comp = \
        #     SolveIVPTimeseriesOutputComp(input_grid_data=gd,
        #                                  output_nodes_per_seg=self.options['output_nodes_per_seg'])
        #
        # phase.add_subsystem('timeseries', subsys=timeseries_comp)

    def configure_timeseries_outputs(self, phase):
        """
        Create connections from time series to all post-introspection sources.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass
        # gd = self.grid_data
        # num_seg = gd.num_segments
        # output_nodes_per_seg = self.options['output_nodes_per_seg']
        # time_units = phase.time_options['units']
        #
        # timeseries_name = 'timeseries'
        # timeseries_comp = phase._get_subsystem(timeseries_name)
        #
        # timeseries_comp._add_output_configure('time', shape=(1,), units=time_units, desc='time')
        # timeseries_comp._add_output_configure('time_phase', shape=(1,), units=time_units,
        #                                       desc='elapsed phase time')
        #
        # phase.connect(src_name='time', tgt_name='timeseries.all_values:time')
        #
        # phase.connect(src_name='time_phase', tgt_name='timeseries.all_values:time_phase')
        #
        # for name, options in phase.state_options.items():
        #
        #     timeseries_comp._add_output_configure(f'states:{name}',
        #                                           shape=options['shape'],
        #                                           units=options['units'],
        #                                           desc=options['desc'])
        #
        #     timeseries_comp._add_output_configure(f'state_rates:{name}',
        #                                           shape=options['shape'],
        #                                           units=get_rate_units(options['units'],
        #                                                                time_units, deriv=1),
        #                                           desc=f'first time-derivative of state {name}')
        #
        #     phase.connect(src_name=f'state_mux_comp.states:{name}',
        #                   tgt_name=f'timeseries.all_values:states:{name}')
        #
        #     rate_src = phase.state_options[name]['rate_source']
        #     if rate_src in phase.parameter_options:
        #         nn = num_seg * output_nodes_per_seg
        #         shape = phase.parameter_options[rate_src]['shape']
        #         param_size = np.prod(shape)
        #         src_idxs = np.tile(np.arange(0, param_size, dtype=int), nn)
        #         src_idxs = np.reshape(src_idxs, (nn,) + shape)
        #         phase.promotes('timeseries', inputs=[(f'all_values:state_rates:{name}',
        #                                               f'parameters:{rate_src}')],
        #                        src_indices=src_idxs, src_shape=shape)
        #     else:
        #         phase.connect(src_name=self.get_rate_source_path(name, phase),
        #                       tgt_name=f'timeseries.all_values:state_rates:{name}')
        #
        # for name, options in phase.control_options.items():
        #     control_units = options['units']
        #
        #     timeseries_comp._add_output_configure(f'controls:{name}',
        #                                           shape=options['shape'],
        #                                           units=control_units,
        #                                           desc=options['desc'])
        #
        #     phase.connect(src_name=f'control_values:{name}',
        #                   tgt_name=f'timeseries.all_values:controls:{name}')
        #
        #     # Control rates
        #     timeseries_comp._add_output_configure(f'control_rates:{name}_rate',
        #                                           shape=options['shape'],
        #                                           units=get_rate_units(control_units, time_units,
        #                                                                deriv=1),
        #                                           desc=f'first time-derivative of control {name}')
        #
        #     phase.connect(src_name=f'control_rates:{name}_rate',
        #                   tgt_name=f'timeseries.all_values:control_rates:{name}_rate')
        #
        #     # Control second derivatives
        #     timeseries_comp._add_output_configure(f'control_rates:{name}_rate2',
        #                                           shape=options['shape'],
        #                                           units=get_rate_units(control_units, time_units,
        #                                                                deriv=2),
        #                                           desc=f'first time-derivative of control {name}')
        #
        #     phase.connect(src_name=f'control_rates:{name}_rate2',
        #                   tgt_name=f'timeseries.all_values:control_rates:{name}_rate2')
        #
        # for name, options in phase.polynomial_control_options.items():
        #     control_units = options['units']
        #     timeseries_comp._add_output_configure(f'polynomial_controls:{name}',
        #                                           shape=options['shape'],
        #                                           units=control_units,
        #                                           desc=options['desc'])
        #
        #     phase.connect(src_name=f'polynomial_control_values:{name}',
        #                   tgt_name=f'timeseries.all_values:polynomial_controls:{name}')
        #
        #     # Polynomial control rates
        #     timeseries_comp._add_output_configure(f'polynomial_control_rates:{name}_rate',
        #                                           shape=options['shape'],
        #                                           units=get_rate_units(control_units, time_units,
        #                                                                deriv=1),
        #                                           desc=f'first time-derivative of control {name}')
        #
        #     phase.connect(src_name=f'polynomial_control_rates:{name}_rate',
        #                   tgt_name=f'timeseries.all_values:polynomial_control_rates:{name}_rate')
        #
        #     # Polynomial control second derivatives
        #     timeseries_comp._add_output_configure(f'polynomial_control_rates:{name}_rate2',
        #                                           shape=options['shape'],
        #                                           units=get_rate_units(control_units, time_units,
        #                                                                deriv=2),
        #                                           desc=f'second time-derivative of control {name}')
        #
        #     phase.connect(src_name=f'polynomial_control_rates:{name}_rate2',
        #                   tgt_name=f'timeseries.all_values:polynomial_control_rates:{name}_rate2')
        #
        # for name, options in phase.parameter_options.items():
        #     prom_name = f'parameters:{name}'
        #
        #     if options['include_timeseries']:
        #         phase.timeseries._add_output_configure(prom_name,
        #                                                desc='',
        #                                                shape=options['shape'],
        #                                                units=options['units'])
        #
        #         if output_nodes_per_seg is None:
        #             src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
        #         else:
        #             src_idxs_raw = np.zeros(num_seg * output_nodes_per_seg, dtype=int)
        #         src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
        #
        #         tgt_name = f'all_values:parameters:{name}'
        #         phase.promotes('timeseries', inputs=[(tgt_name, prom_name)],
        #                        src_indices=src_idxs, flat_src_indices=True)
        #
        # for var, options in phase._timeseries['timeseries']['outputs'].items():
        #     output_name = options['output_name']
        #     units = options.get('units', None)
        #     wildcard_units = options.get('wildcard_units', None)
        #
        #     if '*' in var:  # match outputs from the ODE
        #         ode_outputs = {opts['prom_name']: opts for (k, opts) in
        #                        phase.ode.get_io_metadata(iotypes=('output',)).items()}
        #         matches = filter(list(ode_outputs.keys()), var)
        #     else:
        #         matches = [var]
        #
        #     for v in matches:
        #         if '*' in var:
        #             output_name = v.split('.')[-1]
        #             units = ode_outputs[v]['units']
        #             # check for wildcard_units override of ODE units
        #             if v in wildcard_units:
        #                 units = wildcard_units[v]
        #
        #         # Determine the path to the variable which we will be constraining
        #         # This is more complicated for path constraints since, for instance,
        #         # a single state variable has two sources which must be connected to
        #         # the path component.
        #         var_type = phase.classify_var(v)
        #
        #         # Ignore any variables that we've already added (states, times, controls, etc)
        #         if var_type != 'ode':
        #             continue
        #
        #         # Skip the timeseries output if it does not appear to be shaped as a dynamic variable
        #         # If the full shape does not start with num_nodes, skip this variable.
        #         if self.is_static_ode_output(v, phase, self.num_output_nodes):
        #             warnings.warn(f'Cannot add ODE output {v} to the timeseries output. It is '
        #                           f'sized such that its first dimension != num_nodes.')
        #             continue
        #
        #         shape, units = get_source_metadata(phase.ode, src=v, user_shape=options['shape'],
        #                                            user_units=units)
        #
        #         try:
        #             timeseries_comp._add_output_configure(output_name, shape=shape, units=units, desc='')
        #         except ValueError as e:  # OK if it already exists
        #             if 'already exists' in str(e):
        #                 continue
        #             else:
        #                 raise e
        #
        #         # Failed to find variable, assume it is in the RHS
        #         phase.connect(src_name=f'ode.{v}',
        #                       tgt_name=f'timeseries.all_values:{output_name}')

    def get_parameter_connections(self, name, phase):
        """
        Returns info about a parameter's target connections in the phase.

        Parameters
        ----------
        name : str
            Parameter name.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        connection_info = []
        # connection_info.append(([f'parameters:{name}'], None))
        return connection_info

    def _get_boundary_constraint_src(self, var, loc, phase):
        """
        Return the path to the variable that will be  constrained.

        Parameters
        ----------
        var : str
            Name of the state.
        loc : str
            The location of the boundary constraint ['intitial', 'final'].
        phase : dymos.Phase
            Phase object containing the rate source.

        Returns
        -------
        str
            Path to the source.
        shape
            Source shape.
        str
            Source units.
        bool
            True if the constraint is linear.
        """
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

        if var_type == 'time':
            shape = (1,)
            units = time_units
            linear = True
            if loc == 'initial':
                constraint_path = 't_initial'
            else:
                constraint_path = 'integrator.t_final'
        elif var_type == 'time_phase':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 'integrator.time_phase'
        elif var_type == 'state':
            shape = phase.state_options[var]['shape']
            units = phase.state_options[var]['units']
            linear = loc == 'initial'
            constraint_path = f'integrator.states_out:{var}'
        elif var_type in 'indep_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = True
            constraint_path = f'control_values:{var}'
        elif var_type == 'input_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = False
            constraint_path = f'control_values:{var}'
        elif var_type in 'indep_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = True
            constraint_path = f'polynomial_control_values:{var}'
        elif var_type == 'input_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = False
            constraint_path = f'polynomial_control_values:{var}'
        elif var_type == 'parameter':
            shape = phase.parameter_options[var]['shape']
            units = phase.parameter_options[var]['units']
            linear = True
            constraint_path = f'parameters:{var}'
        elif var_type in ('control_rate', 'control_rate2'):
            control_var = var[:-5] if var_type == 'control_rate' else var[:-6]
            shape = phase.control_options[control_var]['shape']
            control_units = phase.control_options[control_var]['units']
            d = 2 if var_type == 'control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
            control_var = var[:-5]
            shape = phase.polynomial_control_options[control_var]['shape']
            control_units = phase.polynomial_control_options[control_var]['units']
            d = 2 if var_type == 'polynomial_control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            constraint_path = f'polynomial_control_rates:{var}'
        else:
            # Failed to find variable, assume it is in the ODE. This requires introspection.
            raise NotImplementedError('cannot yet constrain/optimize an ODE output using explicit shooting')
            constraint_path = f'{self._rhs_source}.{var}'
            ode = phase._get_subsystem(self._rhs_source)
            shape, units = get_source_metadata(ode, var, user_units=None, user_shape=None)
            linear = False

        return constraint_path, shape, units, linear

    def get_rate_source_path(self, state_var, phase):
        """
        Return the rate source location for a given state name.

        Parameters
        ----------
        state_var : str
            Name of the state.
        phase : dymos.Phase
            Phase object containing the rate source.

        Returns
        -------
        str
            Path to the rate source.
        """
        var = phase.state_options[state_var]['rate_source']

        if var == 'time':
            rate_path = 'time'
        elif var == 'time_phase':
            rate_path = 'time_phase'
        elif phase.state_options is not None and var in phase.state_options:
            rate_path = f'state_mux_comp.states:{var}'
        elif phase.control_options is not None and var in phase.control_options:
            rate_path = f'control_values:{var}'
        elif phase.polynomial_control_options is not None and var in phase.polynomial_control_options:
            rate_path = f'polynomial_control_values:{var}'
        elif phase.parameter_options is not None and var in phase.parameter_options:
            rate_path = f'parameters:{var}'
        elif var.endswith('_rate') and phase.control_options is not None and \
                var[:-5] in phase.control_options:
            rate_path = f'control_rates:{var}'
        elif var.endswith('_rate2') and phase.control_options is not None and \
                var[:-6] in phase.control_options:
            rate_path = f'control_rates:{var}'
        elif var.endswith('_rate') and phase.polynomial_control_options is not None and \
                var[:-5] in phase.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
        elif var.endswith('_rate2') and phase.polynomial_control_options is not None and \
                var[:-6] in phase.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
        else:
            rate_path = f'ode.{var}'

        return rate_path
