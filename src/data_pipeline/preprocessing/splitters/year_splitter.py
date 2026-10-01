"""
Year-based data splitting for temporal datasets
Uses configuration to split data by years
"""
import pandas as pd


def split_indices_by_year(config, start_year=None, end_year=None):
    """
    Split dataset indices based on configured train/val/test years.

    Args:
        config: Configuration object with data.train_years, data.validate_year, data.test_years
        start_year: Starting year of the dataset (default: config.data.train_years[0])
        end_year: Ending year of the dataset (default: max of test_years)

    Returns:
        tuple: (train_indices, val_indices, test_indices,
                train_indices_atm, val_indices_atm, test_indices_atm)
    """
    # Extract years from config
    train_years_range = config.data.train_years  # [start, end]
    val_year = config.data.validate_year
    test_years_list = config.data.test_years

    # Extract months from config (default to all months)
    train_months = getattr(config.data, 'train_months', list(range(1, 13)))
    val_months = getattr(config.data, 'val_months', list(range(1, 13)))
    test_months = getattr(config.data, 'test_months', list(range(1, 13)))

    # Determine full date range
    if start_year is None:
        start_year = train_years_range[0]
    if end_year is None:
        end_year = max([train_years_range[1], val_year] + test_years_list)

    # Generate complete date range
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='D')

    # Create year lists
    train_years_list = list(range(train_years_range[0], train_years_range[1] + 1))

    # Create masks for train/val/test splits
    train_mask = (dates.year.isin(train_years_list)) & (dates.month.isin(train_months))
    val_mask = (dates.year == val_year) & (dates.month.isin(val_months))
    test_mask = (dates.year.isin(test_years_list)) & (dates.month.isin(test_months))

    # Convert boolean masks to indices
    train_indices = train_mask.nonzero()[0].tolist()
    val_indices = val_mask.nonzero()[0].tolist()
    test_indices = test_mask.nonzero()[0].tolist()

    # Atmospheric data indices (sequential indices within each split) The following logic was needed when atmospheric data only consisted of specific months but ocean data was year-round.
    # train_indices_atm = list(range(len(train_indices)))
    # val_indices_atm = list(range(len(val_indices)))
    # test_indices_atm = list(range(len(test_indices)))


    # For current use-case where both ocean and atmospheric data have same date ranges, we can keep the indices same
    train_indices_atm = train_indices
    val_indices_atm = val_indices
    test_indices_atm = test_indices

    # print(f"\n=== Year Splitter Configuration ===")
    # # Check if train_indices and train_indices_atm are the same array
    # print(f"Train indices and train indices atm are the same: {train_indices == train_indices_atm}")
    # print(f"Val indices and val indices atm are the same: {val_indices == val_indices_atm}")
    # print(f"Test indices and test indices atm are the same: {test_indices == test_indices_atm}")


    print(f"\n=== Data Split Summary ===")
    print(f"Train years: {train_years_range[0]}-{train_years_range[1]}, Samples: {len(train_indices)}")
    print(f"Val year: {val_year}, Samples: {len(val_indices)}")
    print(f"Test years: {test_years_list}, Samples: {len(test_indices)}")
    print(f"=========================\n")

    return train_indices, val_indices, test_indices, train_indices_atm, val_indices_atm, test_indices_atm
