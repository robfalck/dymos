"""Define the GaussLobattoIterGroup class."""
import numpy as np
import openmdao.api as om
from openmdao.components.input_resids_comp import InputResidsComp

from .gauss_lobatto_defect_comp import GaussLobattoDefectComp
from .gauss_lobatto_interp_comp import GaussLobattoInterpComp

from dymos.transcriptions.grid_data import GridData
from dymos.phase.options import TimeOptionsDictionary
from dymos.utils.ode_utils import _make_ode_system
from dymos.utils.misc import broadcast_to_nodes, determine_ref_ref0


class OdeInterpGroup(om.Group):
    """Inner group for ODE + Hermite interpolation with initial guess."""

    def guess_nonlinear(self, inputs, outputs, residuals):
        """
        Provide initial guess for states_all by evaluating lgl_interp_comp.

        This ensures the ODE evaluates with proper state values on the first NLBGS iteration.
        """
        # Get the lgl_interp_comp subsystem
        lgl_interp_comp = self._get_subsystem('lgl_interp_comp')

        # Compute interpolated states_all from current state_disc and staterate_disc
        interp_inputs = {}
        interp_outputs = {}

        # Extract all state_disc and staterate_disc from inputs
        for key in inputs._names:
            if key.startswith('state_disc:') or key.startswith('staterate_disc:'):
                interp_inputs[key] = inputs[key]
        if 'dt_dstau' in inputs._names:
            interp_inputs['dt_dstau'] = inputs['dt_dstau']

        # Prepare output dict for all states_all
        for key in outputs._names:
            if key.startswith('states_all:'):
                interp_outputs[key] = outputs[key]

        # Call lgl_interp_comp's compute to populate initial guesses for states_all
        if interp_inputs and interp_outputs:
            try:
                lgl_interp_comp.compute(interp_inputs, interp_outputs)
            except Exception:
                # If anything goes wrong, just skip the guess
                pass


class GaussLobattoIterGroup(om.Group):
    """
    Class definition for the GaussLobattoIterGroup.

    Implements a Picard-style NLBGS algebraic loop for Gauss-Lobatto collocation:

      1. ``ode`` evaluates xdot at all nodes (disc + col) using current state estimates.
      2. ``lgl_interp_comp`` uses disc states + disc rates to update col state values via
         Hermite interpolation, producing ``states_all:{name}`` (disc passthrough + col
         interpolated).
      3. ``states_all:{name}`` feeds back to ``ode`` targets (NLBGS algebraic loop).
      4. ``defects`` (CollocationComp) computes collocation residuals at col nodes.

    NLBGS converges in ~2 iterations:
      - Iter 1: ode evaluates disc + col states → lgl_interp_comp updates col estimates.
      - Iter 2: ode re-evaluates with corrected col states → same disc rates → converged.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._implicit_outputs = set()

    def initialize(self):
        """Declare group options."""
        self.options.declare('state_options', types=dict,
                             desc='Dictionary of options for the states.')
        self.options.declare('time_options', types=TimeOptionsDictionary,
                             desc='Options for time in the phase.')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info.')
        self.options.declare('ode_class', default=None,
                             desc='Callable that instantiates the ODE system.',
                             recordable=False)
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('calc_exprs', types=dict, default={}, recordable=False,
                             desc='Calculation expressions of the parent phase.')
        self.options.declare('parameter_options', types=dict, default={},
                             desc='Parameter options for the phase.')

    def setup(self):
        """Define the structure of the GaussLobattoIterGroup."""
        gd = self.options['grid_data']
        nn = gd.num_nodes  # total nodes (disc + col)
        state_options = self.options['state_options']
        time_options = self.options['time_options']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']

        # --- 1. Inner group: ODE + Hermite interpolation (tight algebraic loop) ---
        # ode evaluates xdot at all nodes using current states_all estimates.
        # lgl_interp_comp uses disc states + disc rates to update states_all via Hermite.
        # The NLBGS on this group converges the algebraic loop in ~2 iterations.
        ode_interp_group = self.add_subsystem('ode_interp_group', OdeInterpGroup(), promotes_inputs=['*'], promotes_outputs=['*'])
        ode_interp_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=10, iprint=0)
        ode_interp_group.linear_solver = om.DirectSolver()

        # Store grid_data for use in guess_nonlinear
        self._gd = gd

        ode = _make_ode_system(ode_class=ode_class,
                               num_nodes=nn,
                               ode_init_kwargs=ode_init_kwargs,
                               calc_exprs=self.options['calc_exprs'],
                               parameter_options=self.options['parameter_options'])
        ode_interp_group.add_subsystem('ode', subsys=ode)

        ode_interp_group.add_subsystem('lgl_interp_comp',
                                       subsys=GaussLobattoInterpComp(
                                           grid_data=gd,
                                           state_options=state_options,
                                           time_units=time_options['units']))

        # --- 2. Collocation + boundary-state defects (outer level) ---
        # defects:{name}           = (f_approx - f_computed) * dt_dstau  (at col nodes)
        # initial_state_defects:{name} = initial_states - states_all[0]
        # final_state_defects:{name}   = final_states   - states_all[-1]
        # state_cnty_defects:{name}    = states_all[seg_start] - states_all[seg_end]
        #                               (uncompressed multi-segment only)
        # Constraints (== 0) added for non-solve_segments states.
        self.add_subsystem('defects',
                           subsys=GaussLobattoDefectComp(grid_data=gd,
                                                         state_options=state_options,
                                                         time_units=time_options['units']),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        # --- 3. InputResidsComp for solve_segments (optional) ---
        has_solve_segments = any(opts['solve_segments'] in ('forward', 'backward') for opts in state_options.values())
        if has_solve_segments:
            self.add_subsystem('states_resids_comp', subsys=InputResidsComp(),
                               promotes_inputs=['*'], promotes_outputs=['*'])

        # --- 4. Set up outer group solver ---
        # Use Newton solver if solve_segments is being used; otherwise use default solver.
        if has_solve_segments:
            self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=0)
            self.linear_solver = om.DirectSolver()

    def _configure_desvars(self, name, options):
        """
        Add design variables for a state, accounting for solve_segments mode.

        Parameters
        ----------
        name : str
            Name of the state variable.
        options : dict
            Options for the state variable.

        Returns
        -------
        set
            Set of variable names that are implicit outputs (driven by residuals).
        """
        state_name = f'states:{name}'
        initial_state_name = f'initial_states:{name}'
        final_state_name = f'final_states:{name}'
        state_rate_name = f'state_rates:{name}'

        solve_segs = options['solve_segments']
        opt = options['opt']

        num_input_nodes = self.options['grid_data'].subset_num_nodes['state_input']

        ib = (None, None) if options['initial_bounds'] is None else options['initial_bounds']
        fb = (None, None) if options['final_bounds'] is None else options['final_bounds']
        lower = options['lower']
        upper = options['upper']

        scaler = options['scaler']
        adder = options['adder']
        ref0 = options['ref0']
        ref = options['ref']

        ref0, ref = determine_ref_ref0(ref0, ref, adder, scaler)
        scaler = None
        adder = None

        fix_initial = options['fix_initial']
        fix_final = options['fix_final']
        input_initial = options['input_initial']
        input_final = options['input_final']
        shape = options['shape']

        if not isinstance(fix_initial, bool):
            raise ValueError(f'Option fix_initial for state {name} must be True or False. '
                             f'If fixing some indices of a non-scalar state, use initial '
                             f'boundary constraints.')
        if not isinstance(fix_final, bool):
            raise ValueError(f'Option fix_final for state {name} must be True or False. '
                             f'If fixing some indices of a non-scalar state, use '
                             f'final boundary constraints.')

        if solve_segs == 'forward' and fix_final:
            raise ValueError(f"Option fix_final on state {name} may not "
                             f"be used with `solve_segments='forward'`.\n Use "
                             f"a boundary constraint to constrain the final "
                             f"state value instead.")
        elif solve_segs == 'backward' and fix_initial:
            raise ValueError(f"Option fix_initial on state {name} may not "
                             f"be used with `solve_segments='backward'`.\n Use "
                             f"a boundary constraint to constrain the initial "
                             f"state value instead.")

        ref0_at_input_nodes = broadcast_to_nodes(ref0, shape, num_input_nodes)
        ref_at_input_nodes = broadcast_to_nodes(ref, shape, num_input_nodes)

        free_vars = {state_name, initial_state_name, final_state_name}

        if solve_segs == 'forward':
            implicit_outputs = {state_name, final_state_name}
        elif solve_segs == 'backward':
            implicit_outputs = {state_name, initial_state_name}
        else:
            implicit_outputs = set()

        free_vars = free_vars - implicit_outputs

        if fix_initial or input_initial:
            free_vars = free_vars - {initial_state_name}
        if fix_final or input_final:
            free_vars = free_vars - {final_state_name}

        ref0_at_1_node = broadcast_to_nodes(ref0, shape, 1)
        ref_at_1_node = broadcast_to_nodes(ref, shape, 1)

        if options['initial_bounds'] is None:
            initial_lb = options['lower']
            initial_ub = options['upper']
        else:
            initial_lb, initial_ub = options['initial_bounds']

        if options['final_bounds'] is None:
            final_lb = options['lower']
            final_ub = options['upper']
        else:
            final_lb, final_ub = options['final_bounds']

        if opt:
            if state_name in free_vars:
                self.add_design_var(name=state_name,
                                    lower=lower,
                                    upper=upper,
                                    ref0=ref0_at_input_nodes,
                                    ref=ref_at_input_nodes)

            if state_rate_name in free_vars:
                self.add_design_var(name=state_rate_name,
                                    ref0=ref0_at_input_nodes,
                                    ref=ref_at_input_nodes)

            if initial_state_name in free_vars:
                self.add_design_var(name=initial_state_name,
                                    lower=initial_lb,
                                    upper=initial_ub,
                                    ref0=ref0_at_1_node,
                                    ref=ref_at_1_node)

            if final_state_name in free_vars:
                self.add_design_var(name=final_state_name,
                                    lower=final_lb,
                                    upper=final_ub,
                                    ref0=ref0_at_1_node,
                                    ref=ref_at_1_node)

        return implicit_outputs

    def configure_io(self, phase):
        """
        I/O creation is delayed until configure so that we can determine shape and units for states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ode_interp_group = self._get_subsystem('ode_interp_group')
        lgl_interp_comp = ode_interp_group._get_subsystem('lgl_interp_comp')
        lgl_interp_comp.configure_io()

        defect_comp = self._get_subsystem('defects')
        defect_comp.configure_io()

        gd = self.options['grid_data']
        nn = gd.num_nodes
        nin = gd.subset_num_nodes['state_input']
        ncn = gd.subset_num_nodes['col']
        ns = gd.num_segments

        disc_idxs = gd.subset_node_indices['state_disc']
        col_idxs = gd.subset_node_indices['col']
        state_input_to_disc = gd.input_maps['state_input_to_disc']

        state_options = self.options['state_options']
        states_resids_comp = self._get_subsystem('states_resids_comp')

        # dt_dstau: slice col nodes for defects; route through ode_interp_group to lgl_interp_comp.
        self.promotes('defects', inputs=('dt_dstau',),
                      src_indices=om.slicer[col_idxs, ...],
                      src_shape=(nn,))
        ode_interp_group.promotes('lgl_interp_comp', inputs=['dt_dstau'])
        self.promotes('ode_interp_group', inputs=[('dt_dstau', 'dt_dstau')],
                      src_indices=om.slicer[col_idxs, ...],
                      src_shape=(nn,))

        for name, options in state_options.items():
            units = options['units']
            shape = options['shape']

            # Route states:{name} (n_input nodes, outer) → lgl_interp_comp.state_disc:{name}
            # (n_disc nodes, inner) via state_input_to_disc expansion.
            ode_interp_group.promotes('lgl_interp_comp',
                                      inputs=[(f'state_disc:{name}', f'state_disc:{name}')])
            self.promotes('ode_interp_group',
                          inputs=[(f'state_disc:{name}', f'states:{name}')],
                          src_indices=om.slicer[state_input_to_disc, ...])

            self._implicit_outputs |= self._configure_desvars(name, options)

            # Only set input defaults for non-solve_segments states; for solve_segments,
            # states:{name} is an implicit output driven by residuals, not an external input.
            if f'states:{name}' not in self._implicit_outputs:
                self.set_input_defaults(f'states:{name}', val=1.0, units=units,
                                        src_shape=(nin,) + shape)

            if f'states:{name}' in self._implicit_outputs:
                # For solve_segments, defects:{name} (GL collocation residuals at col nodes)
                # drive the implicit disc states, analogous to state_rate_defects in Radau.
                states_resids_comp.add_input(f'initial_state_defects:{name}',
                                             shape=(1,) + shape, units=units)
                states_resids_comp.add_input(f'final_state_defects:{name}',
                                             shape=(1,) + shape, units=units)
                states_resids_comp.add_input(f'defects:{name}',
                                             shape=(ncn,) + shape, units=units)

                if ns > 1 and not gd.compressed:
                    states_resids_comp.add_input(f'state_cnty_defects:{name}',
                                                 shape=(ns - 1,) + shape,
                                                 units=units)

                states_resids_comp.add_output(f'states:{name}',
                                              shape=(nin,) + shape,
                                              lower=options['lower'],
                                              upper=options['upper'],
                                              units=units)

            if options['initial_bounds'] is None:
                initial_lb = options['lower']
                initial_ub = options['upper']
            else:
                initial_lb, initial_ub = options['initial_bounds']

            if options['final_bounds'] is None:
                final_lb = options['lower']
                final_ub = options['upper']
            else:
                final_lb, final_ub = options['final_bounds']

            if f'initial_states:{name}' in self._implicit_outputs:
                states_resids_comp.add_output(f'initial_states:{name}',
                                              shape=(1,) + shape, units=units,
                                              lower=initial_lb, upper=initial_ub)

            if f'final_states:{name}' in self._implicit_outputs:
                states_resids_comp.add_output(f'final_states:{name}',
                                              shape=(1,) + shape, units=units,
                                              lower=final_lb, upper=final_ub)

            try:
                rate_source_var = options['rate_source']
            except RuntimeError:
                raise ValueError(f"state '{name}' in phase '{phase.name}' was not given a "
                                 "rate_source")

            var_type = phase.classify_var(rate_source_var)

            # Promote states_all:{name} from lgl_interp_comp through ode_interp_group to outer.
            ode_interp_group.promotes('lgl_interp_comp', outputs=[f'states_all:{name}'])
            self.promotes('ode_interp_group', outputs=[f'states_all:{name}'])

            # Algebraic loop (inside ode_interp_group): states_all:{name} → ode.{target}
            for tgt in options['targets']:
                ode_interp_group.connect(f'states_all:{name}', f'ode.{tgt}')

            # Hermite-interpolated rates at col nodes → defects f_approx:{name}
            self.connect(f'lgl_interp_comp.staterate_col:{name}',
                         f'f_approx:{name}')

            if var_type == 'ode':
                # ode_all[disc_idxs] → lgl_interp_comp staterate_disc (inside ode_interp_group)
                ode_interp_group.connect(f'ode.{rate_source_var}',
                                         f'lgl_interp_comp.staterate_disc:{name}',
                                         src_indices=om.slicer[disc_idxs, ...])

                # ode_all[col_idxs] → defects f_computed (outer level)
                self.connect(f'ode.{rate_source_var}',
                             f'f_computed:{name}',
                             src_indices=om.slicer[col_idxs, ...])
            # else: non-ODE rate sources connected by GaussLobattoNew.configure_defects.
