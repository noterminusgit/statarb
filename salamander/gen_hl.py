"""
High-Low (HL) Signal Generator for Salamander Module

This script generates HL alpha signals from raw price and Barra data. It performs
regression analysis to fit HL mean-reversion signals and produces HDF5 files with
merged data and regression coefficients for each 6-month period.

The HL signal is a mean-reversion strategy based on the position of closing price
relative to the high-low range: close / sqrt(high * low). Signals are computed with
industry demeaning and lagged features.

Usage:
    python gen_hl.py --start=<start_date> --end=<end_date> --dir=<root_directory>

Arguments:
    --start : Starting date for signal generation (format: YYYYMMDD)
              Example: 20100630
    --end   : Ending date for signal generation (format: YYYYMMDD)
              Example: 20130630
    --dir   : Root directory containing data/ subfolder
              Default: '.' (current directory)

Input Requirements:
    The script expects raw data files in data/raw/<YYYYMMDD>/ folders:
    - price_df.csv    : Daily price data (date, gvkey, close, high, low, volume)
    - barra_df.csv    : Barra risk factors (date, gvkey, factor columns)
    - uni_df.csv      : Universe definitions (gvkey, sedol, tradable flags)

    Format: Pipe-delimited (|) CSV files with headers

Output Files:
    data/all/all.<start>-<end>.h5
        HDF5 file with key 'full_df' containing:
        - All input price and Barra data
        - Forward returns (ret0d1, ret0d3, etc.)
        - HL signal components (hl0, hl0_B, hl0_B_ma)
        - Lagged HL features (hl1_B_ma, hl2_B_ma, ...)
        - Regression coefficients (hl1_B_ma_coef, hl2_B_ma_coef, ...)
        - Final HL forecast: sum of lagged features weighted by coefficients

    data/all_graphs/<diagnostics>.png
        Regression diagnostic plots showing coefficient decay and t-statistics

Process:
    1. Divides date range into 6-month periods (Jan-Jun, Jul-Dec)
    2. For each period:
       a. Loads raw data from current period and 2 prior periods (18 months total)
       b. Calculates HL signals with industry demeaning
       c. Runs regression to fit coefficients using prior 12 months
       d. Generates forecasts for current 6-month period
       e. Saves merged data to HDF5 file
    3. Prints unique regression coefficients for each period

Example:
    # Generate HL signals for 2013
    python gen_hl.py --start=20130101 --end=20131231 --dir=.

    # Generate signals for custom date range
    python gen_hl.py --start=20100630 --end=20130630 --dir=/path/to/project

Notes:
    - Requires gen_dir.py to be run first to create folder structure
    - Raw data folders must exist for each 6-month period in the range
    - Regression uses 12-month training window prior to each forecast period
    - HL signals are separated for Energy sector vs. all other sectors
    - Horizon is fixed at 3 days for forward return prediction
    - Output HDF5 files are used by gen_alpha.py to extract alpha signals
"""

import hl_csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--start", help="the starting date, formatted as YYYYMMdd", type=str)
parser.add_argument("--end", help="the ending date, formatted as YYYYMMdd", type=str)
parser.add_argument("--dir", help="the root directory", type=str, default='.')
args = parser.parse_args()
coef_dfs = []
period = []
#pd.set_option('expand_frame_repr', False) to change

d1 = args.start
while d1 < args.end:
    if d1[-4:] == '0101':
        d2 = d1[:4] + '0630'
    else:
        d2 = str(int(d1[:4]) + 1) + '0101'
    print("Creating all.%s-%s.h5..." % (d1, d2))
    period.append(d1 + "-" + d2)
    coef_dfs.append(hl_csv.get_hl(d1, d2, args.dir).drop_duplicates())
    d1 = d2

for i in range(len(period)):
    print("Unique coefficients in the period %s:" % period[i])
    print(coef_dfs[i])
