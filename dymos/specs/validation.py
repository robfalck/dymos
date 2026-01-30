"""
Validation utilities for dymos specs.

Custom validators for specs to ensure consistency and catch errors early.
"""
import numpy as np


def validate_bounds(lower: any | None, upper: any | None) -> None:
    """
    Validate that lower bound is less than or equal to upper bound.

    Parameters
    ----------
    lower : Optional[Any]
        Lower bound value.
    upper : Optional[Any]
        Upper bound value.

    Raises
    ------
    ValueError
        If lower > upper.
    """
    if lower is not None and upper is not None:
        lower_val = np.asarray(lower).min()
        upper_val = np.asarray(upper).max()
        if lower_val > upper_val:
            raise ValueError(f"Lower bound ({lower_val}) must be <= upper bound ({upper_val})")


def validate_equals_exclusive(
    equals: any | None,
    lower: any | None,
    upper: any | None
) -> None:
    """
    Validate that equals is mutually exclusive with lower and upper bounds.

    Parameters
    ----------
    equals : Optional[Any]
        Equality constraint value.
    lower : Optional[Any]
        Lower bound.
    upper : Optional[Any]
        Upper bound.

    Raises
    ------
    ValueError
        If equals is specified with lower or upper.
    """
    if equals is not None:
        if lower is not None or upper is not None:
            raise ValueError("'equals' is mutually exclusive with 'lower' and 'upper'")


def validate_constraint_has_value(
    lower: any | None,
    upper: any | None,
    equals: any | None
) -> None:
    """
    Validate that at least one of lower, upper, or equals is specified.

    Parameters
    ----------
    lower : Optional[Any]
        Lower bound.
    upper : Optional[Any]
        Upper bound.
    equals : Optional[Any]
        Equality value.

    Raises
    ------
    ValueError
        If none of lower, upper, equals are specified.
    """
    if lower is None and upper is None and equals is None:
        raise ValueError("At least one of 'lower', 'upper', or 'equals' must be specified")


def validate_control_polynomial_order(
    control_type: str,
    order: Optional[int]
) -> None:
    """
    Validate that polynomial order is only specified for polynomial controls.

    Parameters
    ----------
    control_type : str
        The control type ('full' or 'polynomial').
    order : Optional[int]
        The polynomial order.

    Raises
    ------
    ValueError
        If order is specified for non-polynomial control.
    """
    if control_type != 'polynomial' and order is not None:
        raise ValueError("'order' is only valid when control_type is 'polynomial'")


def validate_opt_related_fields(
    opt: bool,
    field_name: str,
    field_value: any
) -> None:
    """
    Validate that opt-related fields are only specified when opt=True.

    Parameters
    ----------
    opt : bool
        Whether variable is optimized.
    field_name : str
        Name of the field being checked.
    field_value : Any
        Value of the field.

    Raises
    ------
    ValueError
        If opt-related field is specified when opt=False.
    """
    opt_related_fields = {
        'lower', 'upper', 'scaler', 'adder', 'ref0', 'ref',
        'fix_initial', 'fix_final', 'continuity', 'continuity_scaler',
        'continuity_ref', 'rate_continuity', 'rate_continuity_scaler',
        'rate_continuity_ref', 'rate2_continuity', 'rate2_continuity_scaler',
        'rate2_continuity_ref', 'initial_bounds', 'final_bounds'
    }

    if field_name in opt_related_fields and field_value is not None:
        if not opt and field_name in {'lower', 'upper', 'scaler', 'adder', 'ref0', 'ref'}:
            raise ValueError(f"'{field_name}' is only valid when opt=True")


def validate_state_rate_source(
    rate_source: str | None,
    transcription_type: str | None,
    source: str | None
) -> None:
    """
    Validate that states have rate_source for non-analytic transcriptions.

    Parameters
    ----------
    rate_source : Optional[str]
        The rate source.
    transcription_type : Optional[str]
        The transcription type.
    source : Optional[str]
        The source (for analytic).

    Raises
    ------
    ValueError
        If rate_source is missing for non-analytic transcriptions.
    """
    if transcription_type and transcription_type != 'analytic':
        if rate_source is None:
            raise ValueError(
                "State must have rate_source specified for non-analytic transcriptions"
            )
    elif transcription_type == 'analytic':
        if source is None and rate_source is None:
            raise ValueError("Analytic transcriptions require either 'source' or 'rate_source'")


def validate_transcription_order(
    transcription_type: str,
    order: int | list
) -> None:
    """
    Validate transcription-specific order requirements.

    Parameters
    ----------
    transcription_type : str
        The transcription type.
    order : Union[int, list]
        The polynomial order(s).

    Raises
    ------
    ValueError
        If order is invalid for the transcription type.
    """
    # For pseudospectral methods, order must be odd
    if transcription_type in {'gauss-lobatto', 'radau', 'radau-new', 'analytic'}:
        orders = [order] if isinstance(order, int) else order
        for o in orders:
            if isinstance(o, int) and o % 2 == 0:
                raise ValueError(
                    f"Transcription '{transcription_type}' requires odd polynomial order, "
                    f"got {o}"
                )


def validate_links_reference_phases(
    phase_from: str,
    phase_to: str,
    existing_phases: set
) -> None:
    """
    Validate that linkage phases exist in trajectory.

    Parameters
    ----------
    phase_from : str
        Source phase name.
    phase_to : str
        Target phase name.
    existing_phases : set
        Set of existing phase names.

    Raises
    ------
    ValueError
        If referenced phases don't exist.
    """
    if phase_from not in existing_phases:
        raise ValueError(f"Phase '{phase_from}' referenced in linkage does not exist")
    if phase_to not in existing_phases:
        raise ValueError(f"Phase '{phase_to}' referenced in linkage does not exist")
