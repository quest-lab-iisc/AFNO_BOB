"""
Inference utilities for postprocessing and metrics
"""
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F


def postprocess_ocean_variable(data, mean, variable):
    """
    Reverse preprocessing for ocean variables (normalization and interpolation)

    Args:
        data: Preprocessed data array (224, 224)
        mean: Mean value used in preprocessing
        variable: Variable name

    Returns:
        Postprocessed data in original units and resolution
    """
    if variable in ['thetao', 'so'] and mean is not None:
        # Convert to 2D arrays
        data_2d = data.squeeze()  # (224, 224)
        mean_2d = mean.squeeze()  # (H_orig, W_orig)

        # Interpolate data to match mean's original resolution
        data_tensor = torch.tensor(data_2d).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 224, 224)
        target_size = mean_2d.shape  # (H_orig, W_orig)

        # Interpolate to original resolution
        data_interp = F.interpolate(data_tensor, size=target_size,
                                    mode='bilinear', align_corners=False)
        data_interp = data_interp.squeeze(0).squeeze(0).numpy()  # (H_orig, W_orig)

        # Add mean back (reverse normalization)
        data = data_interp + mean_2d
    else:
        # For uo, vo, zos: just interpolate to original size (no mean subtraction)
        data_2d = data.squeeze()
        if mean is not None:
            mean_2d = mean.squeeze()
            target_size = mean_2d.shape
        else:
            # Fallback: use default ocean grid size
            target_size = (229, 265)

        if data_2d.shape != target_size:
            data_tensor = torch.tensor(data_2d).unsqueeze(0).unsqueeze(0).float()
            data_tensor = F.interpolate(data_tensor, size=target_size,
                                       mode='bilinear', align_corners=False)
            data = data_tensor.squeeze().numpy()
        else:
            data = data_2d

    return data


def compute_metrics(prediction, ground_truth, mask=None):
    """
    Compute evaluation metrics between prediction and ground truth

    Args:
        prediction: Predicted field (numpy array)
        ground_truth: Ground truth field (numpy array)
        mask: Optional mask (1 for valid, 0 for invalid)

    Returns:
        dict: Dictionary containing RMSE, MAE, R2, and Pearson correlation
    """
    # Flatten arrays
    pred_flat = prediction.flatten()
    truth_flat = ground_truth.flatten()

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        pred_flat = pred_flat[mask_flat]
        truth_flat = truth_flat[mask_flat]

    # Remove NaN values
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(truth_flat))
    pred_flat = pred_flat[valid_mask]
    truth_flat = truth_flat[valid_mask]

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(truth_flat, pred_flat))
    mae = mean_absolute_error(truth_flat, pred_flat)
    r2 = r2_score(truth_flat, pred_flat)

    # Pearson correlation
    if len(pred_flat) > 1:
        pearson_corr, _ = pearsonr(pred_flat, truth_flat)
    else:
        pearson_corr = np.nan

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr
    }

    return metrics


def save_metrics_to_file(metrics_dict, save_path):
    """
    Save metrics dictionary to a text file

    Args:
        metrics_dict: Dictionary with variable names as keys and metrics as values
        save_path: Path to save the metrics file
    """
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("INFERENCE METRICS\n")
        f.write("=" * 60 + "\n\n")

        for variable, metrics in metrics_dict.items():
            f.write(f"{variable.upper()}:\n")
            f.write(f"  RMSE:              {metrics['rmse']:.6f}\n")
            f.write(f"  MAE:               {metrics['mae']:.6f}\n")
            f.write(f"  R²:                {metrics['r2']:.6f}\n")
            f.write(f"  Pearson Corr:      {metrics['pearson']:.6f}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")


def date_to_day_index(date_str, reference_date_str="01-01-1993"):
    """
    Convert date string to day index relative to reference date

    Args:
        date_str: Date string in format "dd-mm-yyyy"
        reference_date_str: Reference date string (default: "01-01-1993")

    Returns:
        int: Day index from reference date
    """
    from datetime import datetime

    ref_date = datetime.strptime(reference_date_str, "%d-%m-%Y")
    target_date = datetime.strptime(date_str, "%d-%m-%Y")

    delta = target_date - ref_date
    return delta.days
