"""Download and assemble the GLORYS12, ERA5 and OSTIA input files used by AFNO_BOB.

The models expect, for each period, one ocean file and one atmosphere file with
daily fields on the Bay of Bengal domain (4-23 N, 77-99 E):

    data/1993_2020/ocean.nc            GLORYS12, 1993-01-01 .. 2020-12-31
    data/1993_2020/atm.nc              ERA5 daily means, same dates
    data/1993_2020/ostia_2020.nc       OSTIA SST for 2020 (ensemble figures only)
    data/2021_2025/ocean_2021_2025.nc  GLORYS12, 2021-01-01 .. 2025-09-16
    data/2021_2025/atm_2021_2025.nc    ERA5 daily means, 2021-01-01 .. 2025-11-26

The code indexes these files by day from the first date (1993-01-01 or
2021-01-01), so each file must be daily and gap-free.

Inputs:
    Credentials for the Copernicus Marine Service (``copernicusmarine login``)
        and the Copernicus Climate Data Store (``~/.cdsapirc``).
    CLI subcommand and arguments (see ``--help`` for each subcommand):
        glorys   --start YYYY-MM-DD --end YYYY-MM-DD --out FILE
        era5     --start-year YYYY --end-year YYYY --workdir DIR
        assemble-era5 --workdir DIR --out FILE [--start/--end YYYY-MM-DD]
        ostia    --start YYYY-MM-DD --end YYYY-MM-DD --out FILE

Outputs:
    NetCDF files described above. ``assemble-era5`` writes ERA5 variables with
    dimensions (time, depth=1, latitude, longitude), latitude ascending, which is
    the layout the data loaders and inference scripts index into.

Example:
    python scripts/prepare_data.py glorys --start 1993-01-01 --end 2020-12-31 \\
        --out data/1993_2020/ocean.nc
    python scripts/prepare_data.py era5 --start-year 1993 --end-year 2020 \\
        --workdir data/raw_era5
    python scripts/prepare_data.py assemble-era5 --workdir data/raw_era5 \\
        --start 1993-01-01 --end 2020-12-31 --out data/1993_2020/atm.nc
"""
import argparse
from pathlib import Path

import xarray as xr

BBOX = dict(lat_min=4.0, lat_max=23.0, lon_min=77.0, lon_max=99.0)

GLORYS_DATASET = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'   # GLOBAL_MULTIYEAR_PHY_001_030
GLORYS_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']
GLORYS_DEPTH = 0.49402499198913574                         # uppermost model level (m)

OSTIA_DATASET = 'METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2'

ERA5_DATASET = 'derived-era5-single-levels-daily-statistics'
ERA5_REQUEST_VARS = {                                      # CDS name -> short name
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    'total_precipitation': 'tp',
    'surface_net_solar_radiation': 'ssr',
    'mean_sea_level_pressure': 'msl',
    'total_cloud_cover': 'tcc',
}
ERA5_VARS = ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']      # order used by the models


def download_glorys(start: str, end: str, out: Path) -> None:
    """Download daily GLORYS12 surface fields for the Bay of Bengal domain.

    Args:
        start (str): First date, ``YYYY-MM-DD``.
        end (str): Last date, ``YYYY-MM-DD`` (inclusive).
        out (Path): Output NetCDF path.

    Returns:
        None: Writes ``out`` with variables thetao, so, uo, vo, zos at 0.494 m.

    Example:
        >>> download_glorys('2021-01-01', '2021-01-31', Path('ocean_test.nc'))
    """
    import copernicusmarine
    out.parent.mkdir(parents=True, exist_ok=True)
    copernicusmarine.subset(
        dataset_id=GLORYS_DATASET, variables=GLORYS_VARS,
        minimum_longitude=BBOX['lon_min'], maximum_longitude=BBOX['lon_max'],
        minimum_latitude=BBOX['lat_min'], maximum_latitude=BBOX['lat_max'],
        start_datetime=f'{start}T00:00:00', end_datetime=f'{end}T00:00:00',
        minimum_depth=GLORYS_DEPTH, maximum_depth=GLORYS_DEPTH,
        output_directory=str(out.parent), output_filename=out.name,
    )


def download_ostia(start: str, end: str, out: Path) -> None:
    """Download daily OSTIA foundation SST for the Bay of Bengal domain.

    Args:
        start (str): First date, ``YYYY-MM-DD``.
        end (str): Last date, ``YYYY-MM-DD`` (inclusive).
        out (Path): Output NetCDF path.

    Returns:
        None: Writes ``out`` with variable ``analysed_sst`` (K).

    Example:
        >>> download_ostia('2020-01-01', '2020-12-31', Path('data/1993_2020/ostia_2020.nc'))
    """
    import copernicusmarine
    out.parent.mkdir(parents=True, exist_ok=True)
    copernicusmarine.subset(
        dataset_id=OSTIA_DATASET, variables=['analysed_sst'],
        minimum_longitude=BBOX['lon_min'], maximum_longitude=BBOX['lon_max'],
        minimum_latitude=BBOX['lat_min'], maximum_latitude=BBOX['lat_max'],
        start_datetime=f'{start}T00:00:00', end_datetime=f'{end}T00:00:00',
        output_directory=str(out.parent), output_filename=out.name,
    )


def download_era5(start_year: int, end_year: int, workdir: Path) -> None:
    """Download ERA5 daily means (of hourly data, UTC) per variable and year.

    Args:
        start_year (int): First year.
        end_year (int): Last year (inclusive).
        workdir (Path): Directory for the per-variable, per-year NetCDF files.

    Returns:
        None: Writes ``workdir/era5_{variable}_{year}.nc``; existing files are skipped.

    Example:
        >>> download_era5(2021, 2021, Path('data/raw_era5'))
    """
    import cdsapi
    workdir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    days = [f'{d:02d}' for d in range(1, 32)]
    months = [f'{m:02d}' for m in range(1, 13)]
    for year in range(start_year, end_year + 1):
        for var in ERA5_REQUEST_VARS:
            target = workdir / f'era5_{var}_{year}.nc'
            if target.exists():
                continue
            request = {
                'product_type': 'reanalysis', 'variable': [var], 'year': str(year),
                'month': months, 'day': days, 'daily_statistic': 'daily_mean',
                'time_zone': 'utc+00:00', 'frequency': '1_hourly',
                'area': [BBOX['lat_max'], BBOX['lon_min'], BBOX['lat_min'], BBOX['lon_max']],
            }
            client.retrieve(ERA5_DATASET, request).download(str(target))


def assemble_era5(workdir: Path, out: Path, start: str = None, end: str = None) -> xr.Dataset:
    """Merge per-variable ERA5 downloads into the single file layout the models read.

    Renames ``valid_time`` to ``time``, drops ensemble/experiment coordinates,
    sorts latitude ascending, and adds a length-1 ``depth`` dimension so each
    variable has dimensions (time, depth, latitude, longitude).

    Args:
        workdir (Path): Directory with ``era5_*.nc`` files from :func:`download_era5`.
        out (Path): Output NetCDF path.
        start (str, optional): First date to keep, ``YYYY-MM-DD``.
        end (str, optional): Last date to keep, ``YYYY-MM-DD``.

    Returns:
        xr.Dataset: The assembled dataset (also written to ``out``).

    Example:
        >>> ds = assemble_era5(Path('data/raw_era5'), Path('data/2021_2025/atm_2021_2025.nc'))
    """
    files = sorted(workdir.glob('era5_*.nc'))
    if not files:
        raise FileNotFoundError(f'No era5_*.nc files in {workdir}')
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        ds = ds.drop_vars([c for c in ('number', 'expver') if c in ds.variables])
        parts.append(ds)
    merged = xr.merge([xr.concat([p for p in parts if v in p.data_vars], dim='time')[[v]]
                       for v in ERA5_VARS if any(v in p.data_vars for p in parts)])
    missing = [v for v in ERA5_VARS if v not in merged.data_vars]
    if missing:
        raise ValueError(f'Missing ERA5 variables: {missing}')
    merged = merged.sortby('time').sortby('latitude')
    if start or end:
        merged = merged.sel(time=slice(start, end))
    merged = merged.expand_dims(depth=[0], axis=1)
    merged = merged.transpose('time', 'depth', 'latitude', 'longitude')
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(out)
    return merged


def main() -> None:
    """Parse the subcommand and run the corresponding download or assembly step.

    Args:
        None: Reads ``sys.argv`` (see module docstring).

    Returns:
        None

    Example:
        >>> # python scripts/prepare_data.py assemble-era5 --workdir data/raw_era5 --out atm.nc
    """
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('glorys'); g.add_argument('--start', required=True); g.add_argument('--end', required=True); g.add_argument('--out', type=Path, required=True)
    o = sub.add_parser('ostia'); o.add_argument('--start', required=True); o.add_argument('--end', required=True); o.add_argument('--out', type=Path, required=True)
    e = sub.add_parser('era5'); e.add_argument('--start-year', type=int, required=True); e.add_argument('--end-year', type=int, required=True); e.add_argument('--workdir', type=Path, required=True)
    a = sub.add_parser('assemble-era5'); a.add_argument('--workdir', type=Path, required=True); a.add_argument('--out', type=Path, required=True); a.add_argument('--start'); a.add_argument('--end')
    args = ap.parse_args()
    if args.cmd == 'glorys':
        download_glorys(args.start, args.end, args.out)
    elif args.cmd == 'ostia':
        download_ostia(args.start, args.end, args.out)
    elif args.cmd == 'era5':
        download_era5(args.start_year, args.end_year, args.workdir)
    else:
        assemble_era5(args.workdir, args.out, args.start, args.end)


if __name__ == '__main__':
    main()
