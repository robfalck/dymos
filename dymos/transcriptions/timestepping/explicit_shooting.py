from fnmatch import filter
import warnings

import numpy as np

import openmdao.api as om

from ..transcription_base import TranscriptionBase
from ..grid_data import GridData
from .euler_integration_comp import EulerIntegrationComp
from ..common import TimeComp
from ...utils.misc import get_rate_units, get_target_metadata, get_source_metadata, \
    _unspecified
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
        self._grid_data = GridData(num_segments=self.options['num_segments'],
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
        pass
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
        pass
        # num_seg = self.grid_data.num_segments
        #
        # for state_name, options in phase.state_options.items():
        #     phase.indep_states.add_output(f'initial_states:{state_name}',
        #                                   val=np.ones(((1,) + options['shape'])),
        #                                   units=options['units'])
        #
        # for state_name, options in phase.state_options.items():
        #     # Connect the initial state to the first segment
        #     src_idxs = get_src_indices_by_row([0], options['shape'])
        #
        #     phase.connect(f'initial_states:{state_name}',
        #                   f'segment_0.initial_states:{state_name}',
        #                   src_indices=src_idxs, flat_src_indices=True)
        #
        #     phase.connect(f'segment_0.states:{state_name}',
        #                   f'state_mux_comp.segment_0_states:{state_name}')
        #
        #     targets = get_targets(ode=phase.ode, name=state_name, user_targets=options['targets'])
        #
        #     if targets:
        #         phase.connect(f'state_mux_comp.states:{state_name}',
        #                       [f'ode.{t}' for t in targets])
        #
        #     # Connect the final state in segment n to the initial state in segment n + 1
        #     for i in range(1, num_seg):
        #         if self.options['output_nodes_per_seg'] is None:
        #             nnps_i = self.grid_data.subset_num_nodes_per_segment['all'][i]
        #         else:
        #             nnps_i = self.options['output_nodes_per_seg']
        #
        #         src_idxs = get_src_indices_by_row([nnps_i-1], shape=options['shape'])
        #         phase.connect(f'segment_{i - 1}.states:{state_name}',
        #                       f'segment_{i}.initial_states:{state_name}',
        #                       src_indices=src_idxs, flat_src_indices=True)
        #
        #         phase.connect(f'segment_{i}.states:{state_name}',
        #                       f'state_mux_comp.segment_{i}_states:{state_name}')

    def _get_ode(self, phase):
        integrator = phase._get_subsystem('integrator')
        subprob = integrator._prob
        ode = subprob.model.get_subsystem('ode_eval.ode')
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
                                               mode='fwd',
                                               num_steps_per_segment=self.options['num_steps_per_segment'],
                                               grid_data=self._grid_data,
                                               ode_init_kwargs=phase.options['ode_init_kwargs'],
                                               standalone_mode=False,
                                               complex_step_mode=True)

        phase.add_subsystem(name='integrator', subsys=integrator_comp,
                            promotes_outputs=['*'], promotes_inputs=['*'])

        # for i in range(num_seg):
        #     seg_i_comp = SegmentSimulationComp(
        #         index=i,
        #         simulate_options=phase.simulate_options,
        #         grid_data=self.grid_data,
        #         ode_class=phase.options['ode_class'],
        #         ode_init_kwargs=phase.options['ode_init_kwargs'],
        #         time_options=phase.time_options,
        #         state_options=phase.state_options,
        #         control_options=phase.control_options,
        #         polynomial_control_options=phase.polynomial_control_options,
        #         parameter_options=phase.parameter_options,
        #         output_nodes_per_seg=self.options['output_nodes_per_seg'])
        #
        #     segments_group.add_subsystem(f'segment_{i}', subsys=seg_i_comp)
        #
        # # scipy.integrate.solve_ivp does not actually evaluate the ODE at the desired output points,
        # # but just returns the time and interpolated integrated state values there instead. We need
        # # to instantiate a second ODE group that will call the ODE at those points so that we can
        # # accurately obtain timeseries for ODE outputs.
        # phase.add_subsystem('state_mux_comp',
        #                     SegmentStateMuxComp(grid_data=gd, state_options=phase.state_options,
        #                                         output_nodes_per_seg=self.options['output_nodes_per_seg']))
        #
        # if self.options['output_nodes_per_seg'] is None:
        #     self.num_output_nodes = gd.subset_num_nodes['all']
        # else:
        #     self.num_output_nodes = num_seg * self.options['output_nodes_per_seg']
        #
        # phase.add_subsystem('ode', phase.options['ode_class'](num_nodes=self.num_output_nodes,
        #                                                       **phase.options['ode_init_kwargs']))

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._get_subsystem('integrator').configure_io

    def setup_controls(self, phase):
        """
        Setup the control group.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._check_control_options()

        # output_nodes_per_seg = self.options['output_nodes_per_seg']
        #
        # phase._check_control_options()
        #
        # if phase.control_options:
        #     control_group = SolveIVPControlGroup(control_options=phase.control_options,
        #                                          time_units=phase.time_options['units'],
        #                                          grid_data=self.grid_data,
        #                                          output_nodes_per_seg=output_nodes_per_seg)
        #
        #     phase.add_subsystem('control_group',
        #                         subsys=control_group,
        #                         promotes=['controls:*', 'control_values:*', 'control_values_all:*',
        #                                   'control_rates:*'])

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass
        # super().configure_controls(phase)
        #
        # # Interrogate shapes and units.
        # for name, options in phase.control_options.items():
        #
        #     shape, units, static_target = get_target_metadata(ode, name=name,
        #                                                       user_targets=options['targets'],
        #                                                       user_units=options['units'],
        #                                                       user_shape=options['shape'],
        #                                                       control_rate=True)
        #
        #     options['units'] = units
        #     options['shape'] = shape
        #
        #     if static_target:
        #         raise ValueError(f"Control '{name}' cannot be connected to its targets because one"
        #                          f"or more targets are tagged with 'dymos.static_target'.")
        #
        #     # Now check rate targets
        #     _, _, static_target = get_target_metadata(ode, name=name,
        #                                               user_targets=options['rate_targets'],
        #                                               user_units=options['units'],
        #                                               user_shape=options['shape'],
        #                                               control_rate=True)
        #     if static_target:
        #         raise ValueError(f"Control rate of '{name}' cannot be connected to its targets "
        #                          f"because one or more targets are tagged with 'dymos.static_target'.")
        #
        #     # Now check rate2 targets
        #     _, _, static_target = get_target_metadata(ode, name=name,
        #                                               user_targets=options['rate2_targets'],
        #                                               user_units=options['units'],
        #                                               user_shape=options['shape'],
        #                                               control_rate=True)
        #     if static_target:
        #         raise ValueError(f"Control rate2 of '{name}' cannot be connected to its targets "
        #                          f"because one or more targets are tagged with 'dymos.static_target'.")
        #
        # grid_data = self.grid_data
        #
        # if phase.control_options:
        #     phase.control_group.configure_io()
        #     phase.connect('dt_dstau', 'control_group.dt_dstau')
        #
        # for name, options in phase.control_options.items():
        #     for i in range(grid_data.num_segments):
        #         i1, i2 = grid_data.subset_segment_indices['control_disc'][i, :]
        #         seg_idxs = grid_data.subset_node_indices['control_disc'][i1:i2]
        #         src_idxs = get_src_indices_by_row(row_idxs=seg_idxs, shape=options['shape'])
        #         phase.connect(src_name=f'control_values_all:{name}',
        #                       tgt_name=f'segment_{i}.controls:{name}',
        #                       src_indices=src_idxs, flat_src_indices=True)
        #
        #     targets = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])
        #     if targets:
        #         phase.connect(f'control_values:{name}', [f'ode.{t}' for t in targets])
        #
        #     targets = get_targets(ode=phase.ode, name=f'{name}_rate',
        #                           user_targets=options['rate_targets'])
        #     if targets:
        #         phase.connect(f'control_rates:{name}_rate',
        #                       [f'ode.{t}' for t in targets])
        #
        #     targets = get_targets(ode=phase.ode, name=f'{name}_rate2',
        #                           user_targets=options['rate2_targets'])
        #     if targets:
        #         phase.connect(f'control_rates:{name}_rate2',
        #                       [f'ode.{t}' for t in targets])

    def setup_polynomial_controls(self, phase):
        """
        Adds the polynomial control group to the model if any polynomial controls are present.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass
        # if phase.polynomial_control_options:
        #     sys = SolveIVPPolynomialControlGroup(grid_data=self.grid_data,
        #                                          polynomial_control_options=phase.polynomial_control_options,
        #                                          time_units=phase.time_options['units'],
        #                                          output_nodes_per_seg=self.options['output_nodes_per_seg'])
        #     phase.add_subsystem('polynomial_control_group', subsys=sys,
        #                         promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
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
        super(ExplicitShooting, self).configure_parameters(phase)
        #
        # gd = self.grid_data
        #
        # # We also need to take care of the segments.
        # segs = phase._get_subsystem('segments')
        #
        # for name, options in phase.parameter_options.items():
        #     prom_name = f'parameters:{name}'
        #     shape, units, static_target = get_target_metadata(phase.ode, name=name,
        #                                                       user_targets=options['targets'],
        #                                                       user_shape=options['shape'],
        #                                                       user_units=options['units'])
        #     options['units'] = units
        #     options['shape'] = shape
        #
        #     for i in range(gd.num_segments):
        #         seg_comp = segs._get_subsystem(f'segment_{i}')
        #         seg_comp.add_input(name=prom_name, val=np.ones(shape), units=units,
        #                            desc=f'values of parameter {name}.')
        #         segs.promotes(f'segment_{i}', inputs=[prom_name])

    def setup_defects(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_defects(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_objective(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

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
        pass

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
        pass

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
        connection_info.append((f'integrator.parameters:{name}', None))
        return connection_info

    def _get_boundary_constraint_src(self, var, loc):
        pass

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