"""
Registry pattern implementation for dymos transcriptions.

Maps transcription type strings to their spec classes and transcription classes.
"""
from __future__ import annotations

from typing import Dict, Type, Optional


# These will be populated by _init_registries()
_TRANSCRIPTION_SPECS: Dict[str, Type] = {}
_TRANSCRIPTION_CLASSES: Dict[str, Type] = {}


def _init_registries() -> None:
    """
    Initialize the transcription registries.

    This function maps transcription type strings to their spec classes and
    actual transcription classes. Called at module initialization.
    """
    from .transcription_spec import (
        GaussLobattoSpec,
        RadauSpec,
        RadauNewSpec,
        BirkhoffSpec,
        ExplicitShootingSpec,
        PicardShootingSpec,
        AnalyticSpec,
    )

    # Import actual transcription classes
    from dymos.transcriptions.pseudospectral.gauss_lobatto import GaussLobatto
    from dymos.transcriptions.pseudospectral.radau_new import RadauNew as Radau
    from dymos.transcriptions.pseudospectral.birkhoff import Birkhoff
    from dymos.transcriptions.pseudospectral.radau_new import RadauNew
    from dymos.transcriptions.explicit_shooting import ExplicitShooting
    from dymos.transcriptions.picard_shooting.picard_shooting import PicardShooting
    from dymos.transcriptions.analytic.analytic import Analytic

    # Register specs
    _TRANSCRIPTION_SPECS['gauss-lobatto'] = GaussLobattoSpec
    _TRANSCRIPTION_SPECS['radau'] = RadauSpec
    _TRANSCRIPTION_SPECS['radau-new'] = RadauNewSpec
    _TRANSCRIPTION_SPECS['birkhoff'] = BirkhoffSpec
    _TRANSCRIPTION_SPECS['explicit-shooting'] = ExplicitShootingSpec
    _TRANSCRIPTION_SPECS['picard-shooting'] = PicardShootingSpec
    _TRANSCRIPTION_SPECS['analytic'] = AnalyticSpec

    # Register transcription classes
    _TRANSCRIPTION_CLASSES['gauss-lobatto'] = GaussLobatto
    _TRANSCRIPTION_CLASSES['radau'] = Radau
    _TRANSCRIPTION_CLASSES['radau-new'] = RadauNew
    _TRANSCRIPTION_CLASSES['birkhoff'] = Birkhoff
    _TRANSCRIPTION_CLASSES['explicit-shooting'] = ExplicitShooting
    _TRANSCRIPTION_CLASSES['picard-shooting'] = PicardShooting
    _TRANSCRIPTION_CLASSES['analytic'] = Analytic


def get_transcription_spec_class(transcription_type: str) -> Optional[Type]:
    """
    Get the spec class for a given transcription type.

    Parameters
    ----------
    transcription_type : str
        The transcription type identifier (e.g., 'gauss-lobatto').

    Returns
    -------
    Type or None
        The spec class, or None if not found.
    """
    if not _TRANSCRIPTION_SPECS:
        _init_registries()
    return _TRANSCRIPTION_SPECS.get(transcription_type)


def get_transcription_class(transcription_type: str) -> Optional[Type]:
    """
    Get the transcription class for a given transcription type.

    Parameters
    ----------
    transcription_type : str
        The transcription type identifier (e.g., 'gauss-lobatto').

    Returns
    -------
    Type or None
        The transcription class, or None if not found.
    """
    if not _TRANSCRIPTION_CLASSES:
        _init_registries()
    return _TRANSCRIPTION_CLASSES.get(transcription_type)


def get_all_transcription_types() -> list[str]:
    """
    Get all registered transcription type identifiers.

    Returns
    -------
    list[str]
        List of registered transcription type identifiers.
    """
    if not _TRANSCRIPTION_SPECS:
        _init_registries()
    return list(_TRANSCRIPTION_SPECS.keys())


# Initialize registries at module load time
_init_registries()
