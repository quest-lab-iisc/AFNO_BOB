"""
NetCDF Dataset for Coupled Atmospheric and Ocean Forecasting

Input:  11 channels at time t  (6 atmospheric + 5 ocean)
Output: 11 channels at time t+1 (6 atmospheric + 5 ocean)

Differences from netcdf_dataset.py:
  - Atmospheric forcing is taken at t (not t+1)
  - Both atmospheric and ocean variables are predicted (11 outputs)
  - Northern boundary masking is disabled (rows = 0)
"""
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


class AtmOceanDataset(Dataset):
    """
    PyTorch Dataset for coupled atmospheric-ocean forecasting.

    Returns:
        input_tensor  (11, 224, 224): [atm(t) | ocean(t)]
        output_tensor (11, 224, 224): [atm(t+1) | ocean(t+1)]
    """

    def __init__(self, config, mean, transform=None, indices_ocean=None, indices_atm=None):
        self.data_dir = config.data.data_dir

        # Ocean dataset
        ocean_file = Path(self.data_dir) / f'{config.data.file_prefix}.nc'
        self.data_ocean = xr.open_dataset(ocean_file)
        self.ocean_variables = config.data.variable          # ['thetao','so','uo','vo','zos']

        # Atmospheric dataset
        atm_file = Path(self.data_dir) / f'{config.data.file_prefix_atm}.nc'
        self.data_atm = xr.open_dataset(atm_file)
        self.atm_variables = config.data.atm_variable        # ['ssr','tp','u10','v10','msl','tcc']

        # Normalization statistics for atmospheric variables
        mean_dir = config.data.mean_dir
        self.ssr_std = np.sqrt(np.load(f'{mean_dir}/ssr_var_1993_2018_all_months.npy'))
        self.tp_std  = np.sqrt(np.load(f'{mean_dir}/tp_var_1993_2018_all_months.npy'))
        self.msl_std = np.sqrt(np.load(f'{mean_dir}/msl_var_1993_2018_all_months.npy'))

        self.transform = transform
        self.mean = mean
        self.indices_ocean = indices_ocean
        self.indices_atm   = indices_atm
        self.rows = 0  # Northern boundary masking disabled

    def __len__(self):
        return len(self.indices_atm)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (input_tensor, output_tensor)
                - input_tensor  (11, H, W): atm(t) followed by ocean(t)
                - output_tensor (11, H, W): atm(t+1) followed by ocean(t+1)
        """
        ocean_idx = self.indices_ocean[idx]
        atm_idx   = self.indices_atm[idx]

        var_inputs  = []
        var_outputs = []

        # --- Atmospheric variables at t (input) and t+1 (output) ---
        for atm_var in self.atm_variables:
            atm_in  = self.data_atm[atm_var][atm_idx:atm_idx+1].values
            atm_out = self.data_atm[atm_var][atm_idx+1:atm_idx+2].values

            if self.transform:
                if atm_var == 'ssr':
                    atm_in  = self.transform(atm_in,  self.mean[atm_var],
                                             variable=atm_var, type='atm', variance=self.ssr_std)
                    atm_out = self.transform(atm_out, self.mean[atm_var],
                                             variable=atm_var, type='atm', variance=self.ssr_std)
                elif atm_var == 'tp':
                    atm_in  = self.transform(atm_in,  self.mean[atm_var],
                                             variable=atm_var, type='atm', variance=self.tp_std)
                    atm_out = self.transform(atm_out, self.mean[atm_var],
                                             variable=atm_var, type='atm', variance=self.tp_std)
                elif atm_var == 'msl':
                    atm_in  = self.transform(atm_in,  self.mean[atm_var],
                                             variable=atm_var, type='atm', variance=self.msl_std)
                    atm_out = self.transform(atm_out, self.mean[atm_var],
                                             variable=atm_var, type='atm', variance=self.msl_std)
                else:
                    atm_in  = self.transform(atm_in,  self.mean[atm_var],
                                             variable=atm_var, type='atm')
                    atm_out = self.transform(atm_out, self.mean[atm_var],
                                             variable=atm_var, type='atm')

            var_inputs.append(atm_in)
            var_outputs.append(atm_out)

        # --- Ocean variables at t (input) and t+1 (output) ---
        for ocean_var in self.ocean_variables:
            ocean_in  = self.data_ocean[ocean_var][ocean_idx:ocean_idx+1].values
            ocean_out = self.data_ocean[ocean_var][ocean_idx+1:ocean_idx+2].values

            if self.transform:
                ocean_in  = self.transform(ocean_in,  self.mean[ocean_var], variable=ocean_var)
                ocean_out = self.transform(ocean_out, self.mean[ocean_var], variable=ocean_var)

            var_inputs.append(ocean_in)
            var_outputs.append(ocean_out)

        # Concatenate all channels
        input_tensor  = torch.cat(var_inputs,  dim=0)  # (11, 1, H, W)
        output_tensor = torch.cat(var_outputs, dim=0)  # (11, 1, H, W)

        return input_tensor.squeeze(1), output_tensor.squeeze(1)  # (11, H, W) each
