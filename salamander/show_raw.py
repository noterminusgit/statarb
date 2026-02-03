"""Raw Data Directory Inspector

Simple diagnostic tool to load and display raw CSV data files. Useful for
verifying data structure and checking for loading errors after data generation.

Purpose:
    Quick validation of raw data directories:
    - Verify CSV files parse correctly
    - Check data types and indices
    - Inspect a file called missing_borrow.csv (legacy validation)

Usage:
    python show_raw.py --dir=./data/raw/20130101

Arguments:
    --dir: Path to raw data directory containing CSV files (default: '.')

Expected Files:
    <dir>/barra_df.csv        - Barra factor exposures
    <dir>/price_df.csv        - OHLC price data
    <dir>/missing_borrow.csv  - Legacy file for borrow data diagnostics

Output:
    Prints the contents of uni_df (loaded from missing_borrow.csv)

Index Setup:
    - barra_df: multi-index (gvkey, date)
    - price_df: multi-index (gvkey, date)
    - uni_df: single index (gvkey)

Note:
    The reference to "missing_borrow.csv" suggests this was used during
    development to track securities without borrow data. For production
    validation, use check_all.py or check_hl.py instead.

Customization:
    Modify the print statement to display different DataFrames:
        print(barra_df.head())
        print(price_df[['open', 'high', 'low', 'close']].head())
        print(uni_df[['sedol', 'tid']].head())
"""

import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="the directory where raw files are stored", type=str, default='.')
args = parser.parse_args()
dir = args.dir
barra_df = pd.read_csv("%s/barra_df.csv" % (dir), header=0, sep='|', dtype={'gvkey': str}, parse_dates=[0])
uni_df = pd.read_csv("%s/missing_borrow.csv" % (dir), header=0, sep='|', dtype={'gvkey': str})
price_df = pd.read_csv("%s/price_df.csv" % (dir), header=0, sep='|', dtype={'gvkey': str}, parse_dates=[0])
price_df.set_index(['gvkey', 'date'], inplace=True)
uni_df.set_index('gvkey', inplace=True)
barra_df.set_index(['gvkey', 'date'], inplace=True)

print(uni_df)
