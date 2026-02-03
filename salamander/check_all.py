"""HDF5 Full Dataset Inspector

Simple validation tool to load and inspect the complete all.h5 dataset generated
by hl_csv.py. Useful for verifying data structure and checking specific stocks
or sectors after signal generation.

Purpose:
    Quick data sanity checks on the final HDF5 output:
    - Verify file loads correctly
    - Check column availability
    - Inspect specific stocks (symbol, sector, ind1 classifications)
    - Cross-sectional and time-series slicing tests

Usage:
    python check_all.py --start=20130101 --end=20130630 --dir=./salamander_data

Arguments:
    --start: Start date (YYYYMMDD format)
    --end:   End date (YYYYMMDD format)
    --dir:   Root directory containing data/ subdirectory (default: '.')

Data Requirements:
    <dir>/data/all/all.<start>-<end>.h5   (full_df HDF5 file)

Commented Examples:
    The script includes commented-out print statements showing common queries:
    - df[['symbol','sector','ind1']].head()             # First few rows
    - df[['symbol','sector','ind1']].xs('011644',level=1).head()  # By gvkey
    - df.loc[df['symbol']=='AMZN',['symbol','sector','ind1']].head()  # By symbol

Note:
    This is a minimal inspection script. Uncomment and modify the print
    statements for specific validation tasks.
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start")
parser.add_argument("--end")
parser.add_argument("--dir", help="the root directory", default='.')
args = parser.parse_args()
pd.set_option('display.max_columns', 100)
data_dir = args.dir + '/data'
ff = data_dir + '/all/all.'+ args.start +'-'+ args.end +'.h5'
df = pd.read_hdf(ff, 'full_df')
#print(df[['symbol','sector','ind1']].head())
#print(df[['symbol','sector','ind1']].xs('011644',level=1).head())
#print(df.loc[df['symbol']=='AMZN',['symbol','sector','ind1']].head())
