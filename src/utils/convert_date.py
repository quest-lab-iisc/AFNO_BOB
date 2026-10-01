#!/usr/bin/env python3
"""
Date Conversion Utility Script

A command-line tool to convert between dates and day indices for the
AFNO Bay of Bengal forecasting project.

Usage:
    # Convert date to day index
    python convert_date.py --date 09-06-2020

    # Convert day index to date
    python convert_date.py --index 10021

    # Specify custom reference date
    python convert_date.py --date 15-03-2020 --reference 01-01-1993

Examples:
    $ python convert_date.py --date 01-01-1993
    Date: 01-01-1993 -> Day Index: 0

    $ python convert_date.py --index 10021
    Day Index: 10021 -> Date: 09-06-2020

    $ python convert_date.py --date 09-06-2020
    Date: 09-06-2020 -> Day Index: 10021
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.date_utils import (
    date_to_day_index,
    day_index_to_date,
    validate_date_range,
    format_date_info
)


def main():
    """Main function for date conversion CLI."""
    parser = argparse.ArgumentParser(
        description="Convert between dates (dd-mm-yyyy) and day indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert date to day index
  python convert_date.py --date 09-06-2020

  # Convert day index to date
  python convert_date.py --index 10021

  # Use custom reference date
  python convert_date.py --date 15-03-2020 --reference 01-01-1993

  # Validate a date range
  python convert_date.py --date 15-06-2020 --validate

Note:
  The default reference date is 01-01-1993 (first day in the dataset).
  Day index 0 corresponds to the reference date.
        """
    )

    # Input group - user must specify either date or index
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--date',
        type=str,
        help='Date to convert (format: dd-mm-yyyy, e.g., 09-06-2020)'
    )
    input_group.add_argument(
        '--index',
        type=int,
        help='Day index to convert (e.g., 10021)'
    )

    # Optional arguments
    parser.add_argument(
        '--reference',
        type=str,
        default='01-01-1993',
        help='Reference date (default: 01-01-1993)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate that the date is within acceptable range'
    )

    parser.add_argument(
        '--min-date',
        type=str,
        default='01-01-1993',
        help='Minimum acceptable date for validation (default: 01-01-1993)'
    )

    parser.add_argument(
        '--max-date',
        type=str,
        default='31-12-2025',
        help='Maximum acceptable date for validation (default: 31-12-2025)'
    )

    args = parser.parse_args()

    try:
        # Convert date to day index
        if args.date:
            # Validate if requested
            if args.validate:
                valid, msg = validate_date_range(
                    args.date,
                    args.min_date,
                    args.max_date
                )
                if not valid:
                    print(f"✗ Validation Failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"✓ Date is valid")

            # Convert
            day_index = date_to_day_index(args.date, args.reference)
            print(f"\nDate: {args.date} -> Day Index: {day_index}")
            print(f"Reference: {args.reference}")

        # Convert day index to date
        elif args.index:
            date_str = day_index_to_date(args.index, args.reference)
            print(f"\nDay Index: {args.index} -> Date: {date_str}")
            print(f"Reference: {args.reference}")

            # Validate if requested
            if args.validate:
                valid, msg = validate_date_range(
                    date_str,
                    args.min_date,
                    args.max_date
                )
                if not valid:
                    print(f"✗ Validation Failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"✓ Date is valid")

        print()

    except ValueError as e:
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
