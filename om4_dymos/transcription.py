from typing import Dict, Any
from pydantic import BaseModel, Field


class TranscriptionBase(BaseModel):
    """
    Base class for transcriptions. In the om4 architecture, transcriptions are factories
    that dictate how a Phase (which is an om4 Group) should be structurally built.
    """

    num_segments: int = Field(default=10, description='Number of segments in the phase')
    order: int = Field(default=3, description='Order of the polynomial interpolation')
    compressed: bool = Field(
        default=False, description='Whether to use compressed transcriptions'
    )

    def build_spec(self, phase_options: Any) -> tuple[Dict[str, Any], list[dict]]:
        """
        Generate the OM4 subsystems and connections needed for the phase based on this transcription.

        Parameters
        ----------
        phase_options : The Phase instance containing user-specified options (ODE class, state_options, etc.)

        Returns
        -------
        subsystems : dict
            A dictionary of subsystem names mapping to OM4 System instances (Components/Groups)
        connections : list
            A list of dictionary definitions for connections (e.g. `{"src": "time.t", "tgt": "ode.t"}`)
        """
        raise NotImplementedError('Subclasses must implement build_spec')


class StubRadauTranscription(TranscriptionBase):
    """
    A stub implementation of a Radau transcription for demonstration purposes.
    It simply creates the ODE and a mock "indep_states" component.
    """

    def build_spec(self, phase_options: Any) -> tuple[Dict[str, Any], list[dict]]:
        from om4.components.indepvarcomp import IndepVarComp
        from om4.specs.varspec import ContinuousVar

        subs = {}
        conns = []

        # 1. Compute transcription-specific sizing (stub logic)
        num_nodes = self.num_segments * self.order

        # 2. Add an IndepVarComp to hold initial times, states, etc (stub)
        ivc_outputs = {}

        ivc_outputs['t_initial'] = ContinuousVar(
            val=phase_options.time_options.initial_val
        )
        ivc_outputs['t_duration'] = ContinuousVar(
            val=phase_options.time_options.duration_val
        )

        for state_name, options in phase_options.state_options.items():
            ivc_outputs[f'states:{state_name}'] = ContinuousVar(
                shape=(num_nodes, *options.shape), val=options.initial_val
            )

        subs['indep_vars'] = IndepVarComp(outputs=ivc_outputs)

        # 3. Instantiate the user"s ODE class with the correct number of nodes
        subs['ode'] = phase_options.ode_class(num_nodes=num_nodes)

        # 4. Create connections from independent variables to the ODE
        for state_name, options in phase_options.state_options.items():
            for target in options.targets:
                conns.append(
                    {'src': f'indep_vars.states:{state_name}', 'tgt': f'ode.{target}'}
                )

        return subs, conns
