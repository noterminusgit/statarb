"""HL Alpha Signal Validator

Diagnostic tool to inspect HL alpha signals and verify data integrity. Finds
the maximum absolute HL value and cross-references it with raw price data to
confirm the signal calculation is reasonable.

Purpose:
    After generating HL alpha files, this script helps validate that:
    1. Alpha signals are within expected ranges
    2. Extreme signals correspond to actual price anomalies
    3. Raw price data matches the signal generation

Usage:
    python check_hl.py --start=20130101 --end=20130630 --dir=./salamander_data

Arguments:
    --start: Start date (YYYYMMDD format)
    --end:   End date (YYYYMMDD format)
    --dir:   Root directory containing data/ subdirectory (default: '.')

Data Requirements:
    <dir>/data/hl/alpha.hl.<start>-<end>.csv   (HL alpha signals)
    <dir>/data/raw/<end>/price_df.csv          (Raw price data)

Output:
    Prints two DataFrames:
    1. The stock-date with the largest absolute HL value
    2. The raw OHLC prices for that stock-date (hardcoded gvkey='011644')

Example Output:
                      hl  hl_abs
    date       gvkey
    2013-04-30 011644  0.523  0.523

                           high    low   open  close
    date       gvkey
    2013-04-30 011644  45.20  43.80  44.10  45.15

Note:
    The second lookup uses a hardcoded gvkey='011644' and date='2013-04-30'.
    This appears to be for a specific historical validation case.
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start")
parser.add_argument("--end")
parser.add_argument("--dir", help="the root directory", default='.')
args = parser.parse_args()
data_dir = args.dir + '/data'
ff = data_dir + '/hl/alpha.hl.'+ args.start +'-'+ args.end +'.csv'
df = pd.read_csv(ff, header=0, parse_dates=['date'], dtype={'gvkey': str})
# print(df)
# import sys
# sys.exit()
df = df.set_index(['date','gvkey'])
df['hl_abs']=df['hl'].abs()
df = df.sort_values('hl_abs',ascending=False)
maxhl=df.iloc[[0]]
print(maxhl)

ff = data_dir + "/raw/" + args.end + "/price_df.csv"
df = pd.read_csv(ff, header=0, delimiter='|', parse_dates=['date'], dtype={'gvkey': str})
df = df.set_index(['date','gvkey']).sort_index()
print(df.loc[('2013-04-30','011644'),['high','low','open','close']])
