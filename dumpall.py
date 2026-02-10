#!/usr/bin/env python
"""
DUMPALL - Comprehensive Data Export Utility

Loads and exports all data components used in the trading system to HDF5 files
for inspection, debugging, or external analysis. This utility orchestrates the
complete data pipeline from raw data through all calculations and transformations.

Data Components Exported
-------------------------
1. all.h5: Complete intraday dataset with all calculations
   - Intraday 15-minute bars (dbars_df)
   - Bar-level calculations (bbars_df)
   - Merged intraday calculations
   - Daily data merged with intraday
   - Barra factors and risk model data
   - Analyst ratings and estimates
   - Volume profiles
   - Forward returns
   - Price extras

2. all.factors.h5: Factor regression results
   - Calculated alpha factors
   - Factor exposures
   - Factor returns

Usage
-----
    python dumpall.py --start=20130101 --end=20130630

This will create:
    - all.h5: Full dataset with all calculations
    - all.factors.h5: Factor data

Parameters
----------
--start : str
    Start date in YYYYMMDD format
--end : str
    End date in YYYYMMDD format

Data Pipeline
-------------
1. Load universe and filter stocks
2. Load Barra risk model data
3. Load price data (daily bars)
4. Load analyst ratings history
5. Load intraday bars (15-minute frequency)
6. Merge intraday calculations
7. Merge and transform Barra data
8. Load locate availability data
9. Calculate forward returns (5-day horizon)
10. Calculate alpha factors
11. Calculate price extras
12. Calculate volume profiles
13. Export to HDF5

Output Files
------------
The HDF5 files can be read with:
    import pandas as pd
    df = pd.read_hdf('all.h5', 'table')

Use Cases
---------
- Data quality inspection
- External analysis in Python/R/MATLAB
- Debugging data pipeline issues
- Archiving processed datasets
- Offline research and backtesting

Notes
-----
- Requires significant memory for large date ranges
- lookback=30 days used for universe construction
- horizon=5 days used for forward returns
- Output files can be very large (multi-GB)
"""

from __future__ import division, print_function

from calc import *
from loaddata import *
from util import *

parser = argparse.ArgumentParser(description='G')
parser.add_argument("--start",action="store",dest="start",default=None)
parser.add_argument("--end",action="store",dest="end",default=None)
args = parser.parse_args()

start = dateparser.parse(args.start)
end = dateparser.parse(args.end)
lookback = 30
horizon = 5 

uni_df = get_uni(start, end, lookback)    
barra_df = load_barra(uni_df, start, end, None)
price_df = load_prices(uni_df, start, end, None)
uni_df = price_df[['ticker']]
ratings_df = load_ratings_hist(uni_df, start, end, True)
dbars_df = load_daybars(uni_df, start, end, None, freq='15Min')
bbars_df = load_bars(uni_df, start, end, None, freq=15)
intra_df = merge_intra_calcs(dbars_df, bbars_df)
daily_df = merge_barra_data(price_df, barra_df)
daily_df = transform_barra(daily_df)
locates_df = load_locates(uni_df, start, end)
forwards_df = calc_forward_returns(daily_df, horizon)
daily_df = pd.concat( [daily_df, forwards_df, ratings_df], axis=1)
daily_df, factor_df = calc_factors(daily_df)
daily_df = calc_price_extras(daily_df)
intra_df = merge_intra_data(daily_df, intra_df)
intra_df = calc_vol_profiles(intra_df)
dump_hd5(intra_df.sort_index(), "all")
dump_hd5(factor_df.sort_index(), "all.factors")

