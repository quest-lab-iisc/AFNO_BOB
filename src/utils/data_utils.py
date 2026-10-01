"""
Data processing utility functions
"""
import torch
import torch.nn.functional as F
import numpy as np


def diffuse_within_mask(data, mask, num_rows=20):
    """
    Diffuse the masked region in the last num_rows rows
    Used for smoothing the southern boundary of Bay of Bengal

    Args:
        data: Input tensor (B, C, H, W)
        mask: Binary mask (1 in regions to diffuse, 0 elsewhere)
        num_rows: Number of rows to diffuse

    Returns:
        Diffused data tensor
    """
    B, C, H, W = data.shape

    # Linear extrapolation
    known = data[:, :, H - num_rows - 1:H - num_rows, :]
    step = data[:, :, H - num_rows:H - num_rows + 1, :] - known

    for i in range(num_rows):
        value = known + step * (i + 1)
        update_region = H - num_rows + i

        # Only update where mask is 1
        data[:, :, update_region:update_region + 1, :] = (
            mask[:, :, update_region:update_region + 1, :] * value +
            (1 - mask[:, :, update_region:update_region + 1, :]) *
            data[:, :, update_region:update_region + 1, :]
        )

    return data


def preprocess_ocean_data(data, mean=None, variable=None, mask_boundary_rows=0):
    """
    Preprocess ocean data for model input (matches training preprocessing).

    This function replicates the preprocessing logic from PreprocessTransform
    and NetCDFDataset to ensure consistency between training and inference.

    Args:
        data: Input data array (numpy array or torch tensor)
        mean: Mean for normalization (ocean variables only)
        variable: Variable name (e.g., 'thetao', 'so', 'uo', 'vo', 'zos')
        mask_boundary_rows: Number of rows at the bottom to mask to 0.0 (default: 0)

    Returns:
        Preprocessed torch tensor at 224x224 resolution
    """
    # Apply mean subtraction for temperature and salinity
    if variable in ["thetao", "so"] and mean is not None:
        data = data - mean

    # Handle zos dimensions
    if variable == 'zos' and data.ndim == 3:
        data = np.expand_dims(data, axis=1)

    # Convert to torch tensor
    data = torch.tensor(data, dtype=torch.float32) if not torch.is_tensor(data) else data.clone().detach().float()

    # Replace NaN values with spatial mean
    sample_mean = torch.nanmean(data)
    data = torch.where(torch.isnan(data), sample_mean, data)

    # Interpolate to model resolution (224x224)
    data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)

    # Mask southern boundary if specified (matches training behavior)
    if mask_boundary_rows > 0:
        data[:, :, -mask_boundary_rows:, :] = 0.0

    return data


def preprocess_atmospheric_data(data, mean=None, variance=None, variable=None):
    """
    Preprocess atmospheric data for model input (matches training preprocessing).

    This function replicates the preprocessing logic from PreprocessTransform
    to ensure consistency between training and inference.

    Args:
        data: Input data array
        mean: Mean for normalization
        variance: Variance for scaling (used for ssr, tp, msl)
        variable: Variable name (e.g., 'ssr', 'tp', 'msl', 'u10', 'v10', 'tcc')

    Returns:
        Preprocessed torch tensor at 224x224 resolution
    """
    # Normalize variables that have mean and variance
    if variable in ['ssr', 'tp', 'msl'] and mean is not None and variance is not None:
        data = (data - mean) / variance

    # Convert to torch tensor - ensure it's a 4D tensor
    if not isinstance(data, torch.Tensor):
        sample = torch.tensor(data.copy(), dtype=torch.float32)
    else:
        sample = data.clone().detach().float()

    # Ensure 4D shape (B, C, H, W)
    if sample.ndim == 2:
        sample = sample.unsqueeze(0).unsqueeze(0)
    elif sample.ndim == 3:
        sample = sample.unsqueeze(0)

    # Interpolate to model resolution (224x224)
    sample = F.interpolate(sample, size=(224, 224), mode='bilinear', align_corners=False)

    return sample.reshape(1, 1, 224, 224)


def postprocess_to_original_resolution(data, mask, mean=None, variable=None,
                                       original_size=(229, 265), apply_boundary_smoothing=True):
    """
    Postprocess model output and interpolate back to original resolution.

    This function:
    1. Interpolates from 224x224 back to original resolution
    2. Denormalizes the data (adds back mean for thetao/so)
    3. Applies boundary smoothing for specific variables
    4. Applies the land-sea mask

    Args:
        data: Model output tensor at 224x224
        mask: Land-sea mask at original resolution
        mean: Mean value for denormalization
        variable: Variable name (e.g., 'thetao', 'so', 'zos')
        original_size: Target size tuple (height, width)
        apply_boundary_smoothing: Whether to apply smoothing to southern boundary

    Returns:
        Postprocessed tensor at original resolution
    """
    # Interpolate to original resolution FIRST
    data = F.interpolate(data, size=original_size, mode='bilinear', align_corners=False)

    # Then denormalize temperature and salinity
    if variable in ["thetao", "so"] and mean is not None:
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
        data = data + mean

    # Apply boundary smoothing for specific variables (if requested)
    if apply_boundary_smoothing and variable in ["zos", "so"]:
        strip_height = 25
        bottom_strip = data[:, :, -strip_height:, :]

        # Reference: row just above the strip
        ref_row = data[:, :, -strip_height-1, :].unsqueeze(2)  # shape: (B, C, 1, W)

        # Create a smooth distance weighting (cosine decay from 0 to 1)
        weights = torch.linspace(0, 1, steps=strip_height, device=data.device)
        weights = 0.5 * (1 - torch.cos(torch.pi * weights))  # cosine smoothing
        weights = weights.view(1, 1, strip_height, 1)  # reshape for broadcasting

        # Interpolate between ref_row and original bottom strip
        filled_strip = (1 - weights) * ref_row + weights * bottom_strip

        # Replace back
        data[:, :, -strip_height:, :] = filled_strip

    # Apply mask (convert 0s to NaN for visualization)
    mask_nan = torch.where(mask == 0.0, float('nan'), mask)
    data = data * mask_nan

    return data


# Legacy function aliases for backward compatibility
def data_preprocess(data, mean=None, var=None, variable=None, type=None):
    """
    Legacy preprocessing function - redirects to new unified functions.

    DEPRECATED: Use preprocess_ocean_data() or preprocess_atmospheric_data() instead.
    """
    if type == 'atm':
        return preprocess_atmospheric_data(data, mean=mean, variance=var, variable=variable)
    else:
        return preprocess_ocean_data(data, mean=mean, variable=variable, mask_boundary_rows=0)


def data_postprocess(data, mask, mean=None, variable=None):
    """
    Legacy postprocessing function - redirects to new unified function.

    DEPRECATED: Use postprocess_to_original_resolution() instead.
    """
    return postprocess_to_original_resolution(data, mask, mean=mean, variable=variable,
                                             original_size=(229, 265), apply_boundary_smoothing=True)
