"""NetCDF Dataset Loader for Ocean and Atmospheric Data.

Loads daily ocean (GLORYS) and atmospheric (ERA5) fields from NetCDF files,
applies normalisation and 224×224 bilinear interpolation, and returns
PyTorch tensors for training.

Supports single-step (k_steps=1, default) and multi-step rollout (k_steps>1)
batches.  When k_steps=1 the return format is identical to the original
two-tuple (input, target).  When k_steps>1, returns a three-tuple:
    (input_frame, atm_sequence, ocean_sequence)
where atm_sequence holds atmospheric forcing for steps 2..k and
ocean_sequence holds ocean ground truth for steps 1..k.

Inputs:
    config: Configuration object with data paths, variable lists, and
        normalisation directories.
    mean (dict): Climatological mean arrays keyed by variable name.
    transform (PreprocessTransform): Normalisation and interpolation transform.
    indices_ocean (array-like): Day-index array for ocean data splits.
    indices_atm (array-like): Day-index array for atm data splits.
    k_steps (int): Number of rollout steps (default 1 — single-step mode).

Outputs:
    When k_steps=1:
        input_tensor  (torch.Tensor): Shape (in_ch, 224, 224).
        output_tensor (torch.Tensor): Shape (out_ch, 224, 224).
    When k_steps>1:
        input_frame    (torch.Tensor): Shape (in_ch, 224, 224).
        atm_sequence   (torch.Tensor): Shape (k-1, 6, 224, 224).
        ocean_sequence (torch.Tensor): Shape (k, 5, 224, 224).

Example:
    >>> ds = NetCDFDataset(config, mean, transform, train_idx, train_atm_idx, k_steps=3)
    >>> inp, atm_seq, oce_seq = ds[0]   # multi-step
    >>> inp.shape, atm_seq.shape, oce_seq.shape
    (torch.Size([11, 224, 224]), torch.Size([2, 6, 224, 224]), torch.Size([3, 5, 224, 224]))
"""
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class NetCDFDataset(Dataset):
    """PyTorch Dataset for ocean and atmospheric NetCDF data.

    Attributes:
        data_ocean: Open xarray Dataset for ocean.nc.
        data_atm: Open xarray Dataset for atm.nc.
        k_steps (int): Rollout horizon (1 = single-step).
        rows (int): Northern boundary rows masked to zero (20).

    Args:
        config: Configuration object.
        mean (dict): Climatological means keyed by variable name.
        transform (PreprocessTransform): Normalisation + interpolation transform.
        indices_ocean (array-like): Day indices for the ocean split.
        indices_atm (array-like): Day indices for the atm split.
        k_steps (int): Rollout horizon (default 1).

    Example:
        >>> ds = NetCDFDataset(config, mean, transform, idx_o, idx_a, k_steps=1)
        >>> inp, out = ds[0]
        >>> inp.shape
        torch.Size([11, 224, 224])
    """

    def __init__(self, config, mean, transform=None, indices_ocean=None,
                 indices_atm=None, k_steps: int = 1):
        """Initialise dataset, open NetCDF files, and load normalisation stats."""
        self.data_dir = config.data.data_dir

        # Ocean data
        self.file_path_ocean = Path(self.data_dir) / f'{config.data.file_prefix}.nc'
        self.data_ocean = xr.open_dataset(self.file_path_ocean)
        self.variable = config.data.variable
        self.out_variables = config.data.out_variable

        # Atmospheric data
        self.file_path_atm = Path(self.data_dir) / f'{config.data.file_prefix_atm}.nc'
        self.data_atm = xr.open_dataset(self.file_path_atm)
        self.atm_variable = config.data.atm_variable

        # Load variance for normalization
        mean_dir = config.data.mean_dir
        self.ssr_std = np.sqrt(np.load(f'{mean_dir}/ssr_var_1993_2018_all_months.npy'))
        self.tp_std = np.sqrt(np.load(f'{mean_dir}/tp_var_1993_2018_all_months.npy'))
        self.msl_std = np.sqrt(np.load(f'{mean_dir}/msl_var_1993_2018_all_months.npy'))

        # Indices for data splits
        self.indices_ocean = indices_ocean
        self.indices_atm = indices_atm

        # Transform and mean normalization
        self.transform = transform
        self.mean = mean
        self.rows = config.data.north_mask_rows
        self.k_steps = max(1, int(k_steps))

    def __len__(self) -> int:
        """Return number of valid samples, accounting for multi-step rollout boundary.

        Returns:
            int: Number of samples (reduced by k_steps-1 to avoid end-of-split overflow).

        Example:
            >>> len(ds)
            9489
        """
        return len(self.indices_atm) - (self.k_steps - 1)

    def _transform_atm(self, atm_var: str, raw: np.ndarray) -> torch.Tensor:
        """Apply normalisation and interpolation to one atmospheric variable.

        Args:
            atm_var (str): Variable name (e.g. 'ssr', 'u10').
            raw (np.ndarray): Raw data array, shape (1, H, W).

        Returns:
            torch.Tensor: Shape (1, 224, 224) — normalised and interpolated.

        Example:
            >>> t = ds._transform_atm('ssr', raw_ssr)
            >>> t.shape
            torch.Size([1, 224, 224])
        """
        if atm_var == 'ssr':
            return self.transform(raw, self.mean[atm_var],
                                  variable=atm_var, type='atm', variance=self.ssr_std)
        elif atm_var == 'tp':
            return self.transform(raw, self.mean[atm_var],
                                  variable=atm_var, type='atm', variance=self.tp_std)
        elif atm_var == 'msl':
            return self.transform(raw, self.mean[atm_var],
                                  variable=atm_var, type='atm', variance=self.msl_std)
        else:
            return self.transform(raw, self.mean[atm_var],
                                  variable=atm_var, type='atm')

    def _load_atm_step(self, atm_idx: int) -> torch.Tensor:
        """Load and normalise one day of atmospheric forcing.

        Args:
            atm_idx (int): Time index in the atm dataset (t+1 offset already applied
                by the caller, i.e. pass atm_idx+1 for step 1).

        Returns:
            torch.Tensor: Shape (n_atm_vars, 224, 224).

        Example:
            >>> atm = ds._load_atm_step(atm_idx + 1)
            >>> atm.shape
            torch.Size([6, 224, 224])
        """
        parts = []
        for atm_var in self.atm_variable:
            raw = self.data_atm[atm_var][atm_idx:atm_idx + 1].values
            if self.transform:
                t = self._transform_atm(atm_var, raw)
            else:
                t = torch.tensor(raw, dtype=torch.float32)
            parts.append(t)
        return torch.cat(parts, dim=0).squeeze(1)   # (n_atm, 224, 224)

    def _load_ocean_step(self, ocean_idx: int) -> torch.Tensor:
        """Load and normalise ocean state at a single time step.

        Args:
            ocean_idx (int): Time index in the ocean dataset.

        Returns:
            torch.Tensor: Shape (n_ocean_vars, 224, 224).

        Example:
            >>> oce = ds._load_ocean_step(ocean_idx)
            >>> oce.shape
            torch.Size([5, 224, 224])
        """
        parts = []
        for variable in self.variable:
            raw = self.data_ocean[variable][ocean_idx:ocean_idx + 1].values
            if self.transform:
                t = self.transform(raw, self.mean[variable], variable=variable)
                if self.rows > 0:
                    t[:, :, -self.rows:, :] = 0.0
            else:
                t = torch.tensor(raw, dtype=torch.float32)
            parts.append(t)
        return torch.cat(parts, dim=0).squeeze(1)   # (n_ocean, 224, 224)

    def __getitem__(self, idx):
        """Return one training sample, single-step or multi-step depending on k_steps.

        Args:
            idx (int): Sample index.

        Returns:
            When k_steps == 1:
                tuple: (input_tensor, output_tensor), each (in_ch or out_ch, 224, 224).
            When k_steps > 1:
                tuple: (input_frame, atm_sequence, ocean_sequence) where
                    input_frame    — (in_ch, 224, 224),
                    atm_sequence   — (k-1, n_atm, 224, 224),
                    ocean_sequence — (k, n_ocean, 224, 224).

        Example:
            >>> inp, out = ds[0]          # k_steps=1
            >>> inp.shape
            torch.Size([11, 224, 224])
        """
        ocean_idx = self.indices_ocean[idx]
        atm_idx = self.indices_atm[idx]

        # --- Build standard 1-step input (atm at t+1, ocean at t) ---
        atm_t1 = self._load_atm_step(atm_idx + 1)         # (6, 224, 224)
        oce_t0 = self._load_ocean_step(ocean_idx)           # (5, 224, 224)
        input_frame = torch.cat([atm_t1, oce_t0], dim=0)   # (11, 224, 224)

        # --- Ocean target at t+1 ---
        oce_t1 = self._load_ocean_step(ocean_idx + 1)      # (5, 224, 224)

        if self.k_steps == 1:
            return input_frame, oce_t1

        # --- Multi-step: extend to t+2, ..., t+k ---
        ocean_steps = [oce_t1]
        for step in range(2, self.k_steps + 1):
            ocean_steps.append(self._load_ocean_step(ocean_idx + step))
        ocean_sequence = torch.stack(ocean_steps, dim=0)    # (k, 5, 224, 224)

        # --- Atm sequence for steps 2, ..., k (step 1 atm is already in input_frame) ---
        atm_steps = []
        for step in range(2, self.k_steps + 1):
            atm_steps.append(self._load_atm_step(atm_idx + step))
        atm_sequence = torch.stack(atm_steps, dim=0)        # (k-1, 6, 224, 224)

        return input_frame, atm_sequence, ocean_sequence
