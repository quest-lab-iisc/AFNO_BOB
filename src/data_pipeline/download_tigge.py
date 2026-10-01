"""Download TIGGE ensemble forecast data for Bay of Bengal via CDS API.

Downloads control and perturbed (50-member) forecasts from the ECMWF TIGGE
archive (dataset 'tigge-forecasts') using the cdsapi library. Only the
initialisation dates supplied via --dates are downloaded; dates in the same
calendar month are batched into a single CDS request.

Inputs:
    ~/.cdsapirc (file): CDS API credentials (url and key).
    --dates (str): Comma-separated initialisation dates in dd-mm-yyyy format.
    OUTPUT_DIR (Path): Directory where GRIB files are written (defaults to
        the repo's data/ directory, resolved relative to this script).

Outputs:
    {OUTPUT_DIR}/Tigge_{Mon}_{YYYY}.grib (GRIB2): Control forecasts at
        lead times 24–216 h (1–9 day ahead) for variables u10, v10, msl,
        ssr, tp, tcc over the Bay of Bengal (4–23°N, 77–99°E).
        One file per unique calendar month in the requested dates.
    {OUTPUT_DIR}/Tigge_{Mon}_{YYYY}_ens.grib (GRIB2): Same but for all 50
        perturbed forecast members (numbers 1–50). ~50× larger.

Example:
    # 1. Accept the TIGGE licence at:
    #    https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download#manage-licences
    # 2. Ensure ~/.cdsapirc contains your ECMWF CDS credentials.
    # 3. Run from project root:
    conda activate BoB_Surf_2
    python src/data_pipeline/download_tigge.py \\
        --dates 14-01-2020,14-10-2020             # control forecast only
    python src/data_pipeline/download_tigge.py \\
        --dates 14-01-2020,14-10-2020 --ensemble  # 50-member ensemble only
"""

import sys
import argparse
import calendar
from collections import defaultdict
from pathlib import Path
from datetime import datetime

try:
    import cdsapi
except ImportError:
    print("ERROR: cdsapi is not installed.")
    print("Install with:  pip install cdsapi")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"

# Atmospheric variables as GRIB param codes (unambiguous for MARS):
#   165 = u10  (10m U wind component)
#   166 = v10  (10m V wind component)
#   151 = msl  (mean sea level pressure)
#   176 = ssr  (surface net solar radiation)
#   228228 = tp  (total precipitation)
#   228164 = tcc (total cloud cover)
VARIABLES = ["165", "166", "151", "176", "228228", "228164"]

# Forecast lead times in hours — 1–9 day ahead, matching model evaluation
LEADTIME_HOURS = [str(h) for h in range(24, 217, 24)]

# Bay of Bengal bounding box: [N, W, S, E]
AREA = [23, 77, 4, 99]

# Ensemble member numbers 1–50 (perturbed forecasts; control = member 0)
MEMBERS = [str(n) for n in range(1, 51)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_dates(dates_str: str) -> list[datetime]:
    """Parse a comma-separated list of dd-mm-yyyy date strings.

    Args:
        dates_str (str): Comma-separated dates, e.g. '14-01-2020,19-10-2020'.

    Returns:
        list[datetime]: Parsed datetime objects, one per entry.

    Example:
        >>> parse_dates('14-01-2020,19-10-2020')
        [datetime(2020, 1, 14), datetime(2020, 10, 19)]
    """
    result = []
    for token in dates_str.split(","):
        token = token.strip()
        try:
            result.append(datetime.strptime(token, "%d-%m-%Y"))
        except ValueError:
            print(f"ERROR: Cannot parse date '{token}'. Expected format: dd-mm-yyyy")
            sys.exit(1)
    return result


def group_by_month(dates: list[datetime]) -> dict:
    """Group dates by (year, month), returning sorted day lists.

    Args:
        dates (list[datetime]): List of datetime objects.

    Returns:
        dict: Mapping (year_str, month_str) -> sorted list of zero-padded day
            strings. E.g. {('2020', '01'): ['14'], ('2020', '10'): ['19']}.

    Example:
        >>> group_by_month([datetime(2020, 1, 14), datetime(2020, 1, 19)])
        {('2020', '01'): ['14', '19']}
    """
    groups: dict[tuple, list] = defaultdict(list)
    for dt in dates:
        key = (str(dt.year), f"{dt.month:02d}")
        day_str = f"{dt.day:02d}"
        if day_str not in groups[key]:
            groups[key].append(day_str)
    # Sort days within each month
    return {k: sorted(v) for k, v in groups.items()}


def month_label(month_str: str) -> str:
    """Return three-letter month abbreviation from a two-digit month string.

    Args:
        month_str (str): Two-digit month, e.g. '01'.

    Returns:
        str: Abbreviated month name, e.g. 'Jan'.

    Example:
        >>> month_label('10')
        'Oct'
    """
    return calendar.month_abbr[int(month_str)]


def outfile_name(year: str, month: str, ensemble: bool = False) -> str:
    """Build the output GRIB filename for a given year/month.

    Args:
        year (str): Four-digit year string.
        month (str): Two-digit month string.
        ensemble (bool): True for perturbed-forecast (50-member) file.

    Returns:
        str: Filename, e.g. 'Tigge_Jan_2020.grib' or 'Tigge_Jan_2020_ens.grib'.

    Example:
        >>> outfile_name('2020', '01', ensemble=True)
        'Tigge_Jan_2020_ens.grib'
    """
    label = month_label(month)
    suffix = "_ens" if ensemble else ""
    return f"Tigge_{label}_{year}{suffix}.grib"


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def download_dates(client, year: str, month: str, days: list[str],
                   outfile: str) -> None:
    """Download TIGGE control forecast for specific days to a GRIB file.

    Args:
        client (cdsapi.Client): Authenticated CDS API client.
        year (str): Four-digit year string.
        month (str): Two-digit month string.
        days (list[str]): Zero-padded day strings to download, e.g. ['14', '19'].
        outfile (str): Full path where the output GRIB file is written.

    Returns:
        None: Writes GRIB file to disk as a side effect.

    Example:
        >>> client = cdsapi.Client()
        >>> download_dates(client, '2020', '01', ['14'],
        ...                'data/Tigge_Jan_2020.grib')
    """
    label = month_label(month)
    days_fmt = ", ".join(f"{year}-{month}-{d}" for d in days)
    print(f"\n{'='*60}")
    print(f"Downloading {label} {year} [control]  days: {days_fmt}")
    print(f"Output : {outfile}")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    result = client.retrieve("tigge-forecasts", {
        "origin"       : "ecmwf",
        "year"         : year,
        "month"        : month,
        "day"          : days,
        "time"         : "00:00",
        "level_type"   : "single_level",
        "variable"     : VARIABLES,
        "forecast_type": "control_forecast",
        "leadtime_hour": LEADTIME_HOURS,
        "data_format"  : "grib",
        "area"         : AREA,
    })
    result.download(outfile)

    p = Path(outfile)
    size_mb = p.stat().st_size / 1e6 if p.exists() else 0
    print(f"Done   : {p.name}  ({size_mb:.0f} MB)  [{datetime.now():%H:%M:%S}]")


def download_ensemble_dates(client, year: str, month: str, days: list[str],
                            outfile: str) -> None:
    """Download TIGGE 50-member perturbed ensemble for specific days.

    Args:
        client (cdsapi.Client): Authenticated CDS API client.
        year (str): Four-digit year string.
        month (str): Two-digit month string.
        days (list[str]): Zero-padded day strings to download.
        outfile (str): Full path where the output GRIB file is written.

    Returns:
        None: Writes GRIB file to disk as a side effect. File is ~50× larger
            than the equivalent control-forecast GRIB.

    Example:
        >>> client = cdsapi.Client()
        >>> download_ensemble_dates(client, '2020', '01', ['14'],
        ...                        'data/Tigge_Jan_2020_ens.grib')
    """
    label = month_label(month)
    days_fmt = ", ".join(f"{year}-{month}-{d}" for d in days)
    print(f"\n{'='*60}")
    print(f"Downloading {label} {year} [ensemble, 50 members]  days: {days_fmt}")
    print(f"Output : {outfile}")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    result = client.retrieve("tigge-forecasts", {
        "origin"       : "ecmwf",
        "year"         : year,
        "month"        : month,
        "day"          : days,
        "time"         : "00:00",
        "level_type"   : "single_level",
        "variable"     : VARIABLES,
        "forecast_type": "perturbed_forecast",
        "number"       : MEMBERS,
        "leadtime_hour": LEADTIME_HOURS,
        "data_format"  : "grib",
        "area"         : AREA,
    })
    result.download(outfile)

    p = Path(outfile)
    size_mb = p.stat().st_size / 1e6 if p.exists() else 0
    print(f"Done   : {p.name}  ({size_mb:.0f} MB)  [{datetime.now():%H:%M:%S}]")


def check_credentials():
    """Verify that the CDS credentials file exists and print setup guidance if not.

    Args:
        None

    Returns:
        None: Exits the process with code 1 if the file is missing.

    Example:
        >>> check_credentials()  # exits if ~/.cdsapirc is absent
    """
    creds = Path.home() / ".cdsapirc"
    if not creds.exists():
        print(f"ERROR: CDS credentials file not found: {creds}\n")
        print("Setup steps:")
        print("  1. Register at https://ecds.ecmwf.int  (free account)")
        print(f"  2. Create {creds} with:")
        print("       url: https://ecds.ecmwf.int/api")
        print("       key: YOUR_API_KEY")
        print("  3. Accept the TIGGE licence at:")
        print("     https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download#manage-licences")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Download TIGGE forecasts for user-specified dates.

    Dates in the same calendar month are batched into a single CDS request.
    Without --ensemble: downloads the control (deterministic) forecast only.
    With --ensemble: downloads the 50-member perturbed ensemble only.
    Run twice (once without, once with) to get both forecast types.

    Args:
        None: Reads --dates and --ensemble from sys.argv via argparse.

    Returns:
        None: Writes one GRIB file per unique month to OUTPUT_DIR.

    Example:
        >>> # python src/data_pipeline/download_tigge.py \\
        >>> #     --dates 14-01-2020,14-10-2020
        >>> # python src/data_pipeline/download_tigge.py \\
        >>> #     --dates 14-01-2020,14-10-2020 --ensemble
    """
    parser = argparse.ArgumentParser(description="Download TIGGE ECMWF forecasts")
    parser.add_argument(
        "--dates", required=True,
        help="Comma-separated initialisation dates in dd-mm-yyyy format, "
             "e.g. 14-01-2020,14-10-2020",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Download 50-member perturbed ensemble instead of control forecast",
    )
    args = parser.parse_args()

    check_credentials()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    dates = parse_dates(args.dates)
    groups = group_by_month(dates)  # {(year, month): [days]}

    print(f"\nDates requested: {args.dates}")
    print(f"Grouped into {len(groups)} request(s):")
    for (year, month), days in groups.items():
        print(f"  {month_label(month)} {year}: days {', '.join(days)}")

    downloaded = []

    for (year, month), days in groups.items():
        if args.ensemble:
            # --- 50-member perturbed ensemble only ---
            ens_file = str(OUTPUT_DIR / outfile_name(year, month, ensemble=True))
            if Path(ens_file).exists():
                size_mb = Path(ens_file).stat().st_size / 1e6
                print(f"\nSkipping ensemble {month_label(month)} {year}: "
                      f"{Path(ens_file).name} exists ({size_mb:.0f} MB) — delete to re-download")
            else:
                download_ensemble_dates(client, year, month, days, ens_file)
            downloaded.append(ens_file)
        else:
            # --- Control forecast only ---
            cf_file = str(OUTPUT_DIR / outfile_name(year, month, ensemble=False))
            if Path(cf_file).exists():
                size_mb = Path(cf_file).stat().st_size / 1e6
                print(f"\nSkipping control {month_label(month)} {year}: "
                      f"{Path(cf_file).name} exists ({size_mb:.0f} MB) — delete to re-download")
            else:
                download_dates(client, year, month, days, cf_file)
            downloaded.append(cf_file)

    print(f"\n{'='*60}")
    print("All downloads complete.")
    print(f"Files written to {OUTPUT_DIR}:")
    for f in downloaded:
        p = Path(f)
        if p.exists():
            print(f"  {p.name}  ({p.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
