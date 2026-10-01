"""
Data loading orchestration
"""
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from .netcdf_dataset import NetCDFDataset
from ..preprocessing.transformers.normalize_transform import PreprocessTransform
from ..preprocessing.splitters.year_splitter import split_indices_by_year


def create_mask(data):
    """
    Create binary land-sea mask from first timestep

    Args:
        data: 4D array (time, channels, height, width)

    Returns:
        tuple: (dataset_length, mask)
    """
    dataset_length, _, _, _ = data.shape
    first_timestep = data[0]

    # Create mask: 1 for non-NaN values, 0 for NaN values
    mask = ~np.isnan(first_timestep)
    mask = mask.astype(np.float32)

    # Ensure shape (1, 1, height, width)
    if mask.ndim == 3:
        mask = mask[np.newaxis, :, :, :]

    return dataset_length, mask


def load_and_prepare_data(config, k_steps: int = 1):
    """Load and prepare ocean + atmospheric datasets for training.

    Args:
        config: Configuration object with data paths, variables, and split years.
        k_steps (int): Rollout horizon for the training dataset (default 1).
            When k_steps > 1, the training DataLoader yields 3-tuples
            (input_frame, atm_sequence, ocean_sequence) instead of 2-tuples.
            Validation and test datasets always use k_steps=1.

    Returns:
        tuple: (train_loader, val_loader, test_loader, mask)

    Example:
        >>> train_loader, val_loader, test_loader, mask = load_and_prepare_data(config, k_steps=3)
    """
    print("\n### Loading Mean and Variance ###\n")

    # Load mean values from config
    mean_dir = config.data.mean_dir
    mean = {}

    # Ocean variables
    for var in ['thetao', 'so', 'uo', 'vo', 'zos']:
        mean_file = f"{mean_dir}/mean_{var}_1993_2018_all_months.npy"
        try:
            mean[var] = np.load(mean_file)
        except FileNotFoundError:
            mean[var] = None
            print(f"Warning: Mean file not found for {var}, using None")

    # Atmospheric variables
    for var in ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']:
        mean_file = f"{mean_dir}/{var}_mean_1993_2018_all_months.npy"
        try:
            mean[var] = np.load(mean_file)
        except FileNotFoundError:
            mean[var] = None
            print(f"Warning: Mean file not found for {var}, using None")

    # Load land-sea mask
    print("\n### Creating Land-Sea Mask ###\n")
    data_dir = config.data.data_dir
    file_path = Path(data_dir) / f'{config.data.file_prefix}.nc'
    mask_data = xr.open_dataset(file_path)['thetao'].values
    dataset_length, mask = create_mask(mask_data)
    mask = torch.tensor(mask)
    mask = F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)

    # Create transform
    print("\n### Initializing Transform ###\n")
    transform = PreprocessTransform(config)

    # Split indices by year
    print("\n### Splitting Indices by Year ###\n")
    train_indices, val_indices, test_indices, train_indices_atm, val_indices_atm, test_indices_atm = \
        split_indices_by_year(config)

    # Create datasets — val and test always use single-step format (k_steps=1)
    train_dataset = NetCDFDataset(config, mean, transform, train_indices, train_indices_atm,
                                  k_steps=k_steps)
    val_dataset   = NetCDFDataset(config, mean, transform, val_indices, val_indices_atm,
                                  k_steps=1)
    test_dataset  = NetCDFDataset(config, mean, transform, test_indices, test_indices_atm,
                                  k_steps=1)

    # Create dataloaders
    print("\n### Creating DataLoaders ###\n")
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size,
                              shuffle=config.data.shuffle)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size,
                           shuffle=config.data.shuffle)
    test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size,
                            shuffle=config.data.shuffle)

    return train_loader, val_loader, test_loader, mask
