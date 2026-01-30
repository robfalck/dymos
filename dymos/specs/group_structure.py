"""
GroupStructure dataclass for organizing transcription subsystems and connections.

This module defines the return type for TranscriptionSpec.get_group_structure(),
which generates the list of subsystems and connections that form the OpenMDAO
group structure for a dymos phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmdao.specs.subsystem_spec import SubsystemSpec
    from openmdao.specs.connection_spec import ConnectionSpec


@dataclass
class GroupStructure:
    """
    Specification of subsystems and connections for a dymos phase group.

    This is the return type for TranscriptionSpec.get_group_structure().
    Each transcription type implements this method to generate the complete
    structure needed to create the OpenMDAO Group for a phase.

    Attributes
    ----------
    subsystems : list[SubsystemSpec]
        List of subsystems that form the phase group. Each SubsystemSpec
        includes:
        - name: Name of the subsystem in the parent group
        - system: The SystemSpec (ComponentSpec or GroupSpec)
        - promotes/promotes_inputs/promotes_outputs: Promotion specs

    connections : list[ConnectionSpec]
        List of connections between subsystems. Each ConnectionSpec includes:
        - src: Source variable name (e.g., 'rhs_disc.vdot')
        - tgt: Target variable name (e.g., 'collocation_constraint.f_approx:v')
        - src_indices: Indices into source (for subsetting)
    """

    subsystems: list[SubsystemSpec] = field(default_factory=list)
    connections: list[ConnectionSpec] = field(default_factory=list)
