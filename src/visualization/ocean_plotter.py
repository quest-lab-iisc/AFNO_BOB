"""
Ocean state visualization with cartopy
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec


class OceanPlotter:
    """
    Visualization class for ocean variables with prediction comparison
    """
    def __init__(self, config, lon_range=None, lat_range=None):
        """
        Initialize ocean plotter

        Args:
            config: Configuration object
            lon_range: Longitude range (min, max)
            lat_range: Latitude range (min, max)
        """
        self.config = config
        self.lon_range = lon_range if lon_range else (77, 99)
        self.lat_range = lat_range if lat_range else (4, 23)

        # Variable metadata
        self.var_metadata = {
            'thetao': {'name': 'Temperature', 'units': '°C', 'cmap': 'RdYlBu_r'},
            'so': {'name': 'Salinity', 'units': 'psu', 'cmap': 'viridis'},
            'uo': {'name': 'Zonal Velocity', 'units': 'm/s', 'cmap': 'RdBu_r'},
            'vo': {'name': 'Meridional Velocity', 'units': 'm/s', 'cmap': 'RdBu_r'},
            'zos': {'name': 'Sea Surface Height', 'units': 'm', 'cmap': 'coolwarm'}
        }

    def plot_comparison(self, prediction, ground_truth, variable, date_str,
                       save_path, vmin=None, vmax=None):
        """
        Plot prediction, ground truth, difference, and relative difference

        Args:
            prediction: Predicted field (H, W)
            ground_truth: Ground truth field (H, W)
            variable: Variable name
            date_str: Date string for title
            save_path: Path to save the plot
            vmin: Minimum value for colorbar (optional)
            vmax: Maximum value for colorbar (optional)
        """
        # Get variable metadata
        meta = self.var_metadata.get(variable, {
            'name': variable, 'units': '', 'cmap': 'viridis'
        })

        # Compute differences
        diff = prediction - ground_truth
        epsilon = 1e-10
        relative_diff = diff / (np.abs(ground_truth) + epsilon)

        # Determine color limits if not provided
        if vmin is None or vmax is None:
            # Use 10th and 90th percentile for prediction and ground truth
            combined = np.concatenate([prediction.flatten(), ground_truth.flatten()])
            vmin_auto = np.nanpercentile(combined, 10)
            vmax_auto = np.nanpercentile(combined, 90)
            vmin = vmin if vmin is not None else vmin_auto
            vmax = vmax if vmax is not None else vmax_auto

        # Difference color limits (symmetric around zero)
        diff_max = np.nanpercentile(np.abs(diff), 90)
        diff_min, diff_max = -diff_max, diff_max

        # Relative difference color limits
        rel_diff_max = np.nanpercentile(np.abs(relative_diff), 90)
        rel_diff_min, rel_diff_max = -rel_diff_max, rel_diff_max

        # Create coordinate grids
        lons = np.linspace(self.lon_range[0], self.lon_range[1], prediction.shape[1])
        lats = np.linspace(self.lat_range[0], self.lat_range[1], prediction.shape[0])
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(7, 5))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        projection = ccrs.PlateCarree()

        # --- Subplot 1: Prediction ---
        ax1 = fig.add_subplot(gs[0, 0], projection=projection)
        self._setup_map(ax1)
        im1 = ax1.contourf(lon_grid, lat_grid, prediction,
                          levels=20, cmap=meta['cmap'],
                          vmin=vmin, vmax=vmax, transform=projection)
        ax1.set_title(f"Prediction\n{meta['name']}", fontsize=10, fontweight='bold')
        plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05,
                    fraction=0.046, label=meta['units'])

        # --- Subplot 2: Ground Truth ---
        ax2 = fig.add_subplot(gs[0, 1], projection=projection)
        self._setup_map(ax2)
        im2 = ax2.contourf(lon_grid, lat_grid, ground_truth,
                          levels=20, cmap=meta['cmap'],
                          vmin=vmin, vmax=vmax, transform=projection)
        ax2.set_title(f"Ground Truth\n{meta['name']}", fontsize=10, fontweight='bold')
        plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05,
                    fraction=0.046, label=meta['units'])

        # --- Subplot 3: Difference (Pred - Truth) ---
        ax3 = fig.add_subplot(gs[1, 0], projection=projection)
        self._setup_map(ax3)
        im3 = ax3.contourf(lon_grid, lat_grid, diff,
                          levels=20, cmap='RdBu_r',
                          vmin=diff_min, vmax=diff_max, transform=projection)
        ax3.set_title(f"Difference\n(Pred - Truth)", fontsize=10, fontweight='bold')
        plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05,
                    fraction=0.046, label=meta['units'])

        # --- Subplot 4: Relative Difference ---
        ax4 = fig.add_subplot(gs[1, 1], projection=projection)
        self._setup_map(ax4)
        im4 = ax4.contourf(lon_grid, lat_grid, relative_diff,
                          levels=20, cmap='RdBu_r',
                          vmin=rel_diff_min, vmax=rel_diff_max, transform=projection)
        ax4.set_title(f"Relative Difference\n(Diff / |Truth|)", fontsize=10, fontweight='bold')
        plt.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.05,
                    fraction=0.046, label='Fraction')

        # Main title
        fig.suptitle(f'{variable.upper()} - {date_str}',
                    fontsize=12, fontweight='bold', y=0.98)

        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _setup_map(self, ax):
        """
        Setup map features for a subplot

        Args:
            ax: Matplotlib axis with cartopy projection
        """
        # Set extent
        ax.set_extent([self.lon_range[0], self.lon_range[1],
                      self.lat_range[0], self.lat_range[1]],
                     crs=ccrs.PlateCarree())

        # Add features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--', alpha=0.5)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                         alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
