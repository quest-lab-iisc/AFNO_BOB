# Seasonal Forecast Visualisation

Four scripts produce seasonal one-day-ahead forecast panels for Bay of Bengal ocean variables. All scripts are run from the **project root** and share a common CLI interface.

---

## Scripts

| Script | Variables | Error type |
|---|---|---|
| `plot_seasonal_forecast.py` | SST, SSS | Signed (°C / psu) |
| `plot_seasonal_forecast_relerr.py` | SST, SSS | Relative (%) |
| `plot_seasonal_uvssh.py` | Speed + UV quiver, SSH | Signed (m/s, m) |
| `plot_seasonal_uvssh_relerr.py` | Speed + UV quiver, SSH | Relative (%) |

All scripts support AFNO and TFNO model architectures, detected automatically from the config file.

---

## Figure layout

Every figure is **4 rows × 6 columns**:

```
              ← Variable A ──────────────────┐  ← Variable B ─────────────────┐
              GLORYS │ Model │ Error          │  GLORYS │ Model │ Error         │
Winter        [map]   [map]   [map]              [map]   [map]   [map]
Pre-monsoon   [map]   [map]   [map]              [map]   [map]   [map]
Monsoon       [map]   [map]   [map]              [map]   [map]   [map]
Post-monsoon  [map]   [map]   [map]              [map]   [map]   [map]
              ──── field cbar ──── │ err cbar    ──── field cbar ──── │ err cbar
```

- **Rows**: four seasonal regimes (Winter, Pre-monsoon, Monsoon, Post-monsoon)
- **Left 3 columns**: Variable A (SST or current speed)
- **Right 3 columns**: Variable B (SSS or SSH)
- **Error panels** use `RdBu_r` — blue = under-prediction, red = over-prediction, white = 0
- **Error metric annotation** in the top-left of every error panel (RMSE + MAE, or RRMSE + RMAE for relative-error scripts)
- **Velocity panels** (UV/SSH scripts only): black quiver arrows overlaid on the GLORYS and Model speed panels to show current direction
- Season labels are rotated 90° (bottom → top) on the left margin
- Lat/lon gridlines drawn at every 5° (80–100 °E, 5–25 °N)

---

## Common options

All four scripts accept:

| Option | Default | Description |
|---|---|---|
| `--config_file` | `afno_bob_surf_e06p1.yaml` | YAML config in `config/`; selects model architecture and data paths |
| `--model_path` | `results/models/{name}.pth` | Explicit path to `.pth` weights |
| `--dates` | Four 2020 season dates (see below) | Comma-separated **input** dates `dd-mm-yyyy`; prediction shown is input + 1 day |
| `--season_labels` | `Winter,Pre-monsoon,Monsoon,Post-monsoon` | Row labels matching `--dates` |
| `--device` | From config | `cuda:0`, `cuda:1`, `cpu`, etc. |
| `--output` | `results/plots/{name}_<suffix>` | Output path **without extension**; `.pdf` and `.png` are written |

### Default dates

| Row | Input date | Prediction date shown |
|---|---|---|
| Winter | 14-Jan-2020 | 15-Jan-2020 |
| Pre-monsoon | 19-Apr-2020 | 20-Apr-2020 |
| Monsoon | 14-Jul-2020 | 15-Jul-2020 |
| Post-monsoon | 19-Oct-2020 | 20-Oct-2020 |

---

## Script-specific options

### `plot_seasonal_forecast.py` and `plot_seasonal_forecast_relerr.py`

| Option | Default | Description |
|---|---|---|
| `--sst_range vmin,vmax` | Auto (2nd–98th pct) | SST field colorbar limits (°C) |
| `--sss_range vmin,vmax` | Auto (2nd–98th pct) | SSS field colorbar limits (psu) |
| `--err_range limit` | Auto (98th pct of \|err\|) | Symmetric ± limit for error colorbar (%; relerr script only) |
| `--relerr_threshold T` | `0.05` | \|truth\| below this uses absolute error instead of relative (relerr script only) |

### `plot_seasonal_uvssh.py` and `plot_seasonal_uvssh_relerr.py`

| Option | Default | Description |
|---|---|---|
| `--speed_range vmin,vmax` | Auto (0 – 98th pct) | Speed field colorbar limits (m/s) |
| `--ssh_range vmin,vmax` | Auto (2nd–98th pct) | SSH field colorbar limits (m) |
| `--err_range limit` | Auto (98th pct of \|err\|) | Symmetric ± limit for relative-error colorbar (%; relerr script only) |
| `--relerr_threshold T` | `0.05` | \|truth\| below this uses absolute error instead of relative (relerr scripts only) |
| `--quiver_stride N` | `8` | Sub-sample every N grid points for quiver arrows |
| `--quiver_scale F` | Auto | Quiver arrow scale in data-units per inch; lower = larger arrows |

---

## Usage examples

```bash
# SST/SSS — best AFNO model, default season dates
python src/visualization/plot_seasonal_forecast.py \
    --config_file afno_bob_surf_e06p1.yaml \
    --device cuda:0

# SST/SSS relative error — TFNO model
python src/visualization/plot_seasonal_forecast_relerr.py \
    --config_file tfno_bob_surf_e01.yaml \
    --device cuda:0

# Velocity + SSH — fixed speed colorbar, denser quiver
python src/visualization/plot_seasonal_uvssh.py \
    --config_file afno_bob_surf_e06p1.yaml \
    --device cuda:0 \
    --speed_range 0,0.8 \
    --quiver_stride 6

# Velocity + SSH relative error — custom dates and labels
python src/visualization/plot_seasonal_uvssh_relerr.py \
    --config_file afno_bob_surf_e06p1.yaml \
    --device cpu \
    --dates 10-01-2020,15-04-2020,10-07-2020,15-10-2020 \
    --season_labels "Jan,Apr,Jul,Oct" \
    --output results/plots/custom_uvssh_relerr
```

---

## Outputs

Each run writes two files:

| File | Description |
|---|---|
| `<output>.pdf` | Vector figure for paper submission |
| `<output>.png` | Raster figure at 300 dpi |

Default output paths follow the pattern `results/plots/{experiment_name}_{suffix}`:

| Script | Suffix |
|---|---|
| `plot_seasonal_forecast.py` | `_seasonal_forecast` |
| `plot_seasonal_forecast_relerr.py` | `_seasonal_relerr` |
| `plot_seasonal_uvssh.py` | `_seasonal_uvssh` |
| `plot_seasonal_uvssh_relerr.py` | `_seasonal_uvssh_relerr` |

---

## Architecture support

The config key `--config_file` determines which model class is loaded:

| Config contains | Model used | Example config |
|---|---|---|
| `afno2d:` section | `AFNONet` | `afno_bob_surf_e06p1.yaml` |
| `tfno:` section | `TFNOWrapper` (neuralop) | `tfno_bob_surf_e01.yaml` |

No code changes are needed to switch architectures — pass the appropriate config file.

---

## Colourmap reference

| Panel | Colourmap | Notes |
|---|---|---|
| SST field | `cmocean.thermal` | fallback: `RdYlBu_r` |
| SSS field | `cmocean.haline` | fallback: `viridis` |
| Speed field | `cmocean.speed` | fallback: `YlOrBr`; always starts at 0 |
| SSH field | `cmocean.matter` | fallback: `YlOrBr`; sequential — SSH is always positive |
| All error panels | `RdBu_r` | diverging, centred at 0 |

Install `cmocean` to get the oceanographic colormaps: `pip install cmocean`.

---

## Notes

- **Relative error near zero**: where `|truth| < --relerr_threshold` (default 0.05), the error panel falls back to absolute error instead of relative to avoid division by very small numbers. This primarily affects current speed in near-calm regions. Raise the threshold if artefacts are visible; lower it to extend relative-error coverage.
- **Quiver scale**: if arrows are too large or too small, adjust `--quiver_stride` (density) and `--quiver_scale` (size). Larger `--quiver_scale` = smaller arrows.
- **Input convention**: ocean-only models (out_chs=5) use atmospheric forcing at t+1; coupled models (out_chs=11) use forcing at t. This is handled automatically.
- **Color ranges** are computed from the 2nd–98th percentile of all valid ocean pixels pooled across all four seasons, ensuring seasonal panels are directly comparable.
