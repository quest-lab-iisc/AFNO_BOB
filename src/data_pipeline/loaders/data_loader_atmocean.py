"""
Data loading orchestration for coupled atmospheric-ocean forecasting
"""
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from .netcdf_dataset_atmocean import AtmOceanDataset
from ..preprocessing.transformers.normalize_transform import PreprocessTransform
from ..preprocessing.splitters.year_splitter import split_indices_by_year


def load_and_prepare_data_atmocean(config):
    """
    Load and prepare data for coupled atmospheric-ocean forecasting.

    Args:
        config: Configuration object

    Returns:
        tuple: (train_loader, val_loader, test_loader, mask)
            mask: (1, 1, 224, 224) land-sea mask derived from ocean thetao
    """
    print("\n### Loading Mean and Variance ###\n")

    mean_dir = config.data.mean_dir
    mean = {}

    # Ocean variable means
    for var in ['thetao', 'so', 'uo', 'vo', 'zos']:
        mean_file = f"{mean_dir}/mean_{var}_1993_2018_all_months.npy"
        try:
            mean[var] = np.load(mean_file)
        except FileNotFoundError:
            mean[var] = None
            print(f"Warning: Mean file not found for {var}, using None")

    # Atmospheric variable means
    for var in ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']:
        mean_file = f"{mean_dir}/{var}_mean_1993_2018_all_months.npy"
        try:
            mean[var] = np.load(mean_file)
        except FileNotFoundError:
            mean[var] = None
            print(f"Warning: Mean file not found for {var}, using None")

    # Build land-sea mask from ocean thetao
    print("\n### Creating Land-Sea Mask ###\n")
    data_dir = config.data.data_dir
    ocean_file = Path(data_dir) / f'{config.data.file_prefix}.nc'
    mask_data = xr.open_dataset(ocean_file)['thetao'].values  # (T, [1,] H, W)

    # Ensure 4-D for create_mask
    if mask_data.ndim == 3:
        mask_data = mask_data[:, np.newaxis, :, :]  # (T, 1, H, W)

    dataset_length = mask_data.shape[0]
    first_timestep = mask_data[0]                   # (1, H, W)
    mask = (~np.isnan(first_timestep)).astype(np.float32)  # (1, H, W)
    mask = mask[np.newaxis, :, :, :]                # (1, 1, H, W)

    mask = torch.tensor(mask)
    mask = F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)

    # Transform (normalization + interpolation)
    print("\n### Initializing Transform ###\n")
    transform = PreprocessTransform(config)

    # Year-based index splitting
    print("\n### Splitting Indices by Year ###\n")
    (train_indices, val_indices, test_indices,
     train_indices_atm, val_indices_atm, test_indices_atm) = split_indices_by_year(config)

    # Datasets
    train_dataset = AtmOceanDataset(config, mean, transform,
                                    train_indices, train_indices_atm)
    val_dataset   = AtmOceanDataset(config, mean, transform,
                                    val_indices, val_indices_atm)
    test_dataset  = AtmOceanDataset(config, mean, transform,
                                    test_indices, test_indices_atm)

    # DataLoaders
    print("\n### Creating DataLoaders ###\n")
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size,
                              shuffle=config.data.shuffle)
    val_loader   = DataLoader(val_dataset,   batch_size=config.data.batch_size,
                              shuffle=config.data.shuffle)
    test_loader  = DataLoader(test_dataset,  batch_size=config.data.batch_size,
                              shuffle=config.data.shuffle)

    return train_loader, val_loader, test_loader, mask
