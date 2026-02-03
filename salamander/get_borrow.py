"""Borrow Rate Data Aggregator

Consolidates weekly stock lending availability files into a single CSV for
backtesting. Processes historical borrow rate and share availability data
from multiple files into a unified time series.

Purpose:
    Stock borrow costs impact profitability of short positions. This script
    combines fragmented weekly borrow data files into a single consolidated
    CSV for easy lookup during backtesting.

Usage:
    python get_borrow.py --locates_dir=./data

    Processes all files matching:
        <locates_dir>/locates/Historical_Avail_US_Weekly_*

Arguments:
    --locates_dir: Parent directory containing locates/ subdirectory (default: '.')

Input Files:
    Historical_Avail_US_Weekly_<date>.csv with columns:
        - history_date: Date of the availability snapshot
        - sedol: SEDOL identifier for the security
        - shares: Number of shares available to borrow
        - fee: Annualized borrow fee rate (percentage)
        - ticker: Stock ticker symbol

Output File:
    <locates_dir>/locates/borrow.csv

    Pipe-delimited CSV indexed by date with columns:
        - sedol: Security identifier
        - shares: Available share quantity
        - fee: Borrow rate (annualized percentage)
        - symbol: Stock ticker

Process:
    1. Glob all Historical_Avail_US_Weekly_* files in sorted order
    2. Load each CSV with date parsing
    3. Rename history_date -> date, ticker -> symbol
    4. Concatenate all DataFrames
    5. Set date as index
    6. Write to borrow.csv with | delimiter

Example Output:
                 sedol     shares    fee  symbol
    date
    2013-01-04  2484088  1000000   0.50    AAPL
    2013-01-04  2000019   500000  25.00    GME
    2013-01-11  2484088  1100000   0.45    AAPL

Note:
    SEDOL is used as the join key with uni_df (see change_raw.py). Ensure
    uni_df has SEDOL column before using borrow data in strategies.
"""

import pandas as pd
import argparse
import glob

def get_borrow(locates_dir):
    result_dfs = []
    for ff in sorted(glob.glob(locates_dir + "/locates/Historical_Avail_US_Weekly_*")):
        print("Loading", ff)
        df = pd.read_csv(ff, parse_dates=['history_date'],
                         usecols=['history_date', 'sedol', 'shares', 'fee', 'ticker'])
        df = df.rename(columns={'history_date': 'date', 'ticker': 'symbol'})
        result_dfs.append(df)
    result_df = pd.concat(result_dfs)
    result_df.set_index("date", inplace=True)
    result_df.to_csv(r"%s/locates/borrow.csv" % locates_dir, "|")
    print(result_df)

parser = argparse.ArgumentParser()
parser.add_argument("--locates_dir", help="the directory to the locates folder", type=str, default='.')
args = parser.parse_args()
get_borrow(args.locates_dir)
