"""
Data normalization and preprocessing transformations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreprocessTransform(nn.Module):
    """
    Preprocessing transform for ocean and atmospheric data:
    - Normalization (mean subtraction, variance scaling)
    - Interpolation to target resolution
    - NaN handling
    """
    def __init__(self, config):
        super(PreprocessTransform, self).__init__()
        self.config = config

    def forward(self, sample, mean=None, variable=None, type=None, variance=None):
        """
        Args:
            sample: Input data array
            mean: Mean value for normalization
            variable: Variable name (e.g., 'thetao', 'ssr')
            type: Data type ('atm' for atmospheric, None for ocean)
            variance: Variance for scaling (optional)

        Returns:
            Normalized and interpolated torch tensor
        """
        # Atmospheric data normalization
        if type == 'atm':
            if variable in ['ssr', 'tp', 'msl'] and mean is not None and variance is not None:
                sample = (sample - mean) / variance
        else:
            # Ocean data normalization
            if variable in ['thetao', 'so'] and mean is not None:
                sample = sample - mean
            if variable == 'zos':
                import numpy as np
                sample = np.expand_dims(sample, axis=1)

        # Convert to torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)

        # Interpolate atmospheric data first
        if type == 'atm':
            sample = F.interpolate(sample, size=(224, 224), mode='bilinear', align_corners=False)
            return sample

        # Replace NaN values with mean
        sample_mean = torch.nanmean(sample)
        sample = torch.where(torch.isnan(sample), sample_mean, sample)

        # Interpolate to target resolution
        sample = F.interpolate(sample, size=(224, 224), mode='bilinear', align_corners=False)

        return sample
