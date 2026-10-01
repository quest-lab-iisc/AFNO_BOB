# Ocean State Inference

One-day-ahead ocean state prediction using trained AFNO model.

## Usage

### Basic Usage

Run inference with the default date from config file:

```bash
cd /path/to/BoB_Surf
python src/inference/run_inference.py
```

### Custom Input Date

Specify a custom input date:

```bash
python src/inference/run_inference.py --input_date "10-07-2020"
```

### Custom Model Path

Use a specific model checkpoint:

```bash
python src/inference/run_inference.py --model_path "results/models/my_model.pth"
```

### Combined Options

```bash
python src/inference/run_inference.py --input_date "15-08-2020" --model_path "results/models/AFNO_BoB_Surf_E01.pth"
```

## Output

The script creates a directory structure:
```
results/
└── {model_name}/
    └── {input_date}/
        ├── thetao_comparison.png
        ├── so_comparison.png
        ├── uo_comparison.png
        ├── vo_comparison.png
        ├── zos_comparison.png
        └── metrics.txt
```

### Comparison Plots

Each PNG file contains a 2x2 grid:
- **Top-left**: Predicted ocean state
- **Top-right**: Ground truth
- **Bottom-left**: Difference (Prediction - Ground Truth)
- **Bottom-right**: Relative difference (Diff / |Truth|)

All plots use cartopy for geographic visualization with coastlines and borders.

### Metrics File

The `metrics.txt` file contains for each variable:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: R-squared score
- **Pearson**: Pearson correlation coefficient

## Configuration

### Color Axis Limits

You can set custom color axis limits in `config/afno_bob_config.yaml`:

```yaml
visualization:
  vmin:
    thetao: 28.0
    so: 32.0
    uo: -0.5
    vo: -0.5
    zos: -0.5
  vmax:
    thetao: 31.0
    so: 35.0
    uo: 0.5
    vo: 0.5
    zos: 0.5
```

If not specified, limits are auto-computed from 10th and 90th percentiles.

## How It Works

1. **Load Model**: Loads the trained AFNO model from checkpoint
2. **Load Data**:
   - Ocean state at time `t` (current day)
   - Atmospheric forcing at time `t+1` (next day)
3. **Preprocess**: Applies same preprocessing as training:
   - Mean subtraction for `thetao` and `so`
   - Variance scaling for atmospheric variables (`ssr`, `tp`, `msl`)
   - Interpolation to 224x224
4. **Predict**: Runs model to predict ocean state at `t+1`
5. **Postprocess**: Reverses preprocessing to get original units
6. **Evaluate**: Computes metrics against ground truth
7. **Visualize**: Creates comparison plots with cartopy

## Variables

### Ocean Variables (Input at t, Output at t+1)
- `thetao`: Sea water potential temperature (°C)
- `so`: Sea water salinity (psu)
- `uo`: Eastward sea water velocity (m/s)
- `vo`: Northward sea water velocity (m/s)
- `zos`: Sea surface height (m)

### Atmospheric Variables (Input at t+1)
- `ssr`: Surface solar radiation
- `tp`: Total precipitation
- `u10`: 10m u-component of wind
- `v10`: 10m v-component of wind
- `msl`: Mean sea level pressure
- `tcc`: Total cloud cover
