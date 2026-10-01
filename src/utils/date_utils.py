"""
Date Utility Functions for AFNO Bay of Bengal Forecasting

This module provides utilities to convert between human-readable dates (dd-mm-yyyy)
and day indices relative to a reference date. The reference date is the first day
in the dataset (01-01-1993).

Functions:
    date_to_day_index: Convert date string to day index
    day_index_to_date: Convert day index to date string
    parse_date: Parse date string to datetime object
    validate_date: Validate date string format and range
"""

from datetime import datetime, timedelta
from typing import Union, Tuple


def parse_date(date_str: str) -> datetime:
    """
    Parse date string in dd-mm-yyyy format to datetime object.

    Args:
        date_str (str): Date string in format 'dd-mm-yyyy'

    Returns:
        datetime: Parsed datetime object

    Raises:
        ValueError: If date format is invalid

    Examples:
        >>> parse_date("01-01-1993")
        datetime.datetime(1993, 1, 1, 0, 0)
        >>> parse_date("29-05-2020")
        datetime.datetime(2020, 5, 29, 0, 0)
    """
    try:
        return datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError as e:
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected format: dd-mm-yyyy (e.g., '01-01-1993')"
        ) from e


def date_to_day_index(
    target_date: str,
    reference_date: str = "01-01-1993"
) -> int:
    """
    Convert a date string to day index relative to reference date.

    The day index is the number of days elapsed since the reference date.
    Reference date has index 0.

    Args:
        target_date (str): Target date in format 'dd-mm-yyyy'
        reference_date (str, optional): Reference date in format 'dd-mm-yyyy'.
                                        Defaults to '01-01-1993'

    Returns:
        int: Day index (number of days since reference date)

    Raises:
        ValueError: If date format is invalid or target_date is before reference_date

    Examples:
        >>> date_to_day_index("01-01-1993", "01-01-1993")
        0
        >>> date_to_day_index("02-01-1993", "01-01-1993")
        1
        >>> date_to_day_index("01-01-1994", "01-01-1993")
        365
        >>> date_to_day_index("29-05-2020", "01-01-1993")
        10021
    """
    ref_dt = parse_date(reference_date)
    target_dt = parse_date(target_date)

    if target_dt < ref_dt:
        raise ValueError(
            f"Target date {target_date} is before reference date {reference_date}. "
            f"Day index cannot be negative."
        )

    delta = target_dt - ref_dt
    return delta.days


def day_index_to_date(
    day_index: int,
    reference_date: str = "01-01-1993"
) -> str:
    """
    Convert day index to date string relative to reference date.

    Args:
        day_index (int): Day index (number of days since reference date)
        reference_date (str, optional): Reference date in format 'dd-mm-yyyy'.
                                        Defaults to '01-01-1993'

    Returns:
        str: Date string in format 'dd-mm-yyyy'

    Raises:
        ValueError: If day_index is negative

    Examples:
        >>> day_index_to_date(0, "01-01-1993")
        '01-01-1993'
        >>> day_index_to_date(1, "01-01-1993")
        '02-01-1993'
        >>> day_index_to_date(365, "01-01-1993")
        '01-01-1994'
        >>> day_index_to_date(10021, "01-01-1993")
        '29-05-2020'
    """
    if day_index < 0:
        raise ValueError(f"Day index must be non-negative, got {day_index}")

    ref_dt = parse_date(reference_date)
    target_dt = ref_dt + timedelta(days=day_index)

    return target_dt.strftime("%d-%m-%Y")


def validate_date_range(
    date_str: str,
    min_date: str = "01-01-1993",
    max_date: str = "31-12-2025"
) -> Tuple[bool, str]:
    """
    Validate if a date is within an acceptable range.

    Args:
        date_str (str): Date string to validate in format 'dd-mm-yyyy'
        min_date (str, optional): Minimum acceptable date. Defaults to '01-01-1993'
        max_date (str, optional): Maximum acceptable date. Defaults to '31-12-2025'

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
                         If valid, error_message is empty string

    Examples:
        >>> validate_date_range("15-06-2020")
        (True, '')
        >>> validate_date_range("15-06-1990")
        (False, "Date 15-06-1990 is before minimum date 01-01-1993")
    """
    try:
        date_dt = parse_date(date_str)
        min_dt = parse_date(min_date)
        max_dt = parse_date(max_date)

        if date_dt < min_dt:
            return False, f"Date {date_str} is before minimum date {min_date}"
        if date_dt > max_dt:
            return False, f"Date {date_str} is after maximum date {max_date}"

        return True, ""

    except ValueError as e:
        return False, str(e)


def get_date_from_config(config, reference_date: str = "01-01-1993") -> int:
    """
    Get day index from config object, handling both date and index formats.

    This function provides backward compatibility by accepting either:
    - config.plot.input_date (new format: 'dd-mm-yyyy')
    - config.plot.input_day (old format: integer index)

    Args:
        config: Configuration object with plot settings
        reference_date (str, optional): Reference date. Defaults to '01-01-1993'

    Returns:
        int: Day index for plotting

    Examples:
        >>> class Config:
        ...     class plot:
        ...         input_date = "29-05-2020"
        >>> get_date_from_config(Config())
        10021
    """
    # Try to get reference_date from config first
    ref_date = getattr(config.plot, 'reference_date', reference_date)

    # Try new format first (input_date)
    if hasattr(config.plot, 'input_date'):
        input_date = config.plot.input_date
        return date_to_day_index(input_date, ref_date)

    # Fall back to old format (input_day)
    elif hasattr(config.plot, 'input_day'):
        return config.plot.input_day

    else:
        raise AttributeError(
            "Config must have either 'plot.input_date' (dd-mm-yyyy format) "
            "or 'plot.input_day' (integer index)"
        )


def get_evaluation_start_day(config, reference_date: str = "01-01-1993") -> int:
    """
    Get evaluation start day index from config object, handling both date and index formats.

    This function provides backward compatibility by accepting either:
    - config.evaluation.start_date (new format: 'dd-mm-yyyy')
    - config.evaluation.start_day (old format: integer index)

    Args:
        config: Configuration object with evaluation settings
        reference_date (str, optional): Reference date. Defaults to '01-01-1993'

    Returns:
        int: Day index for evaluation start

    Examples:
        >>> class Config:
        ...     class evaluation:
        ...         start_date = "01-01-2020"
        ...         reference_date = "01-01-1993"
        >>> get_evaluation_start_day(Config())
        9862
    """
    # Try to get reference_date from config first
    ref_date = getattr(config.evaluation, 'reference_date', reference_date)

    # Try new format first (start_date)
    if hasattr(config.evaluation, 'start_date'):
        start_date = config.evaluation.start_date
        return date_to_day_index(start_date, ref_date)

    # Fall back to old format (start_day)
    elif hasattr(config.evaluation, 'start_day'):
        return config.evaluation.start_day

    else:
        raise AttributeError(
            "Config must have either 'evaluation.start_date' (dd-mm-yyyy format) "
            "or 'evaluation.start_day' (integer index)"
        )


def format_date_info(day_index: int, reference_date: str = "01-01-1993") -> str:
    """
    Format date information for display.

    Args:
        day_index (int): Day index
        reference_date (str, optional): Reference date. Defaults to '01-01-1993'

    Returns:
        str: Formatted string with date and index information

    Examples:
        >>> format_date_info(10021)
        'Day 10021 (29-05-2020)'
        >>> format_date_info(0)
        'Day 0 (01-01-1993)'
    """
    date_str = day_index_to_date(day_index, reference_date)
    return f"Day {day_index} ({date_str})"


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Date Utility Functions - Examples")
    print("=" * 60)

    # Example 1: Convert date to day index
    print("\n1. Date to Day Index:")
    print(f"   Reference: 01-01-1993 -> Day {date_to_day_index('01-01-1993')}")
    print(f"   Next day:  02-01-1993 -> Day {date_to_day_index('02-01-1993')}")
    print(f"   One year:  01-01-1994 -> Day {date_to_day_index('01-01-1994')}")
    print(f"   Target:    29-05-2020 -> Day {date_to_day_index('29-05-2020')}")

    # Example 2: Convert day index to date
    print("\n2. Day Index to Date:")
    print(f"   Day 0     -> {day_index_to_date(0)}")
    print(f"   Day 1     -> {day_index_to_date(1)}")
    print(f"   Day 365   -> {day_index_to_date(365)}")
    print(f"   Day 10021 -> {day_index_to_date(10021)}")

    # Example 3: Validate dates
    print("\n3. Date Validation:")
    valid, msg = validate_date_range("15-06-2020")
    print(f"   15-06-2020: {'✓ Valid' if valid else '✗ Invalid - ' + msg}")

    valid, msg = validate_date_range("15-06-1990")
    print(f"   15-06-1990: {'✓ Valid' if valid else '✗ Invalid - ' + msg}")

    # Example 4: Format date info
    print("\n4. Format Date Info:")
    print(f"   {format_date_info(0)}")
    print(f"   {format_date_info(10021)}")

    print("\n" + "=" * 60)
