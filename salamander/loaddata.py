"""
Salamander Data Loading Module (File-Based)

This module provides simplified, file-based data loading functions for the
salamander standalone backtesting system. Unlike the main loaddata.py which
loads from CSV files with complex universe filtering and multiple data sources,
this module is designed for rapid prototyping and testing with pre-processed data.

Key Differences from Main loaddata.py:
    - File-based only: Loads from pre-generated CSV/HDF5 files (no direct database access)
    - Simplified interface: Fewer configuration parameters and dependencies
    - Python 3 compatible: Uses modern pandas patterns (concat vs append)
    - Pre-computed data: Expects alpha signals and cached data to already exist
    - Minimal filtering: Assumes universe filtering happened during data generation

Functions:
    load_mus(mdir, fcast, start, end):
        Load alpha forecast signals from CSV files.

        Searches for files matching pattern: {mdir}/{fcast}/alpha.{fcast}.{d1}-{d2}.csv
        Files must have columns: date (datetime), gvkey (string), and signal columns
        Returns DataFrame indexed by (date, gvkey) with forecast signals.

        Args:
            mdir: Base directory containing alpha signal subdirectories
            fcast: Forecast/alpha name (e.g., "hl", "bd", "combined")
            start: Start date string in YYYYMMDD format
            end: End date string in YYYYMMDD format

        Returns:
            DataFrame indexed by (date, gvkey) with alpha signal columns

    load_cache(start, end, data_dir, cols=None):
        Load cached market data from HDF5 files.

        Searches for files matching pattern: {data_dir}/all/all.{d1}-{d2}.h5
        Each HDF5 file contains a 'full_df' table with OHLCV prices, returns,
        volume metrics, and other pre-calculated features.

        Args:
            start: datetime, start date for data range
            end: datetime, end date for data range
            data_dir: Base directory containing cached HDF5 files
            cols: Optional list of column names to load (default: all columns)

        Returns:
            DataFrame indexed by (date, gvkey) with market data columns

    load_factor_cache(start, end, data_dir):
        Load cached factor/alpha signals from HDF5 files.

        Similar to load_cache but:
        - Indexes by (date, factor) instead of (date, gvkey)
        - Includes 30-day buffer before start date for lookback calculations
        - Used for loading pre-computed factor returns or exposures

        Args:
            start: datetime, start date for data range
            end: datetime, end date for data range
            data_dir: Base directory containing factor HDF5 files

        Returns:
            DataFrame indexed by (date, factor) with factor values

    load_locates(uni_df, start, end, locates_dir):
        Load short borrow availability data from CSV.

        Loads locate/borrow data from {locates_dir}/locates/borrow.csv
        Merges with universe using both symbol and SEDOL identifiers.
        Forward-fills missing data within available date range.
        Sets borrow_qty=-inf and fee=-0 outside available data range.

        File format: CSV with columns [symbol, sedol, date, shares, fee]
        Separator: pipe (|)

        Args:
            uni_df: Universe DataFrame with columns [gvkey, symbol, sedol, date]
            start: datetime, start date
            end: datetime, end date
            locates_dir: Base directory containing locates subdirectory

        Returns:
            DataFrame indexed by (date, gvkey) with columns [borrow_qty, fee]

Data File Patterns:
    Alpha signals: {mdir}/{fcast}/alpha.{fcast}.YYYYMMDD-YYYYMMDD.csv
    Cached data: {data_dir}/all/all.YYYYMMDD-YYYYMMDD.h5
    Borrow data: {locates_dir}/locates/borrow.csv

Usage Example:
    # Load high-low strategy signals
    alpha_df = load_mus('/data/signals', 'hl', '20130101', '20130630')

    # Load cached market data
    market_df = load_cache(datetime(2013,1,1), datetime(2013,6,30), '/data/cache')

    # Load specific columns only
    price_df = load_cache(start, end, '/data/cache', cols=['close', 'volume'])

Notes:
    - Requires pre-generated data files (use gen_*.py scripts to create)
    - HDF5 files must contain 'full_df' table
    - Date ranges in filenames are inclusive: start-end covers [start, end]
    - gvkey is string type (6-digit Compustat identifier)
"""

import sys
import glob
import re
import math
from util import *
import pandas as pd
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import os
from dateutil import parser as dateparser


def load_mus(mdir, fcast, start, end):
    print("Looking in {}".format(mdir))
    fcast_dfs = []
    for ff in sorted(glob.glob(mdir + "/" + fcast + "/alpha.*")):
        m = re.match(r".*alpha\." + fcast + "\.(\d{8})-(\d{8}).csv", str(ff))
        d1 = m.group(1)
        d2 = m.group(2)
        if start is not None:
            if d2 <= start or d1 >= end: continue
        print("Loading {} from {} to {}".format(ff, d1, d2))
        df = pd.read_csv(ff, header=0, parse_dates=['date'], dtype={'gvkey': str})
        df.set_index(['date', 'gvkey'], inplace=True)
        fcast_dfs.append(df)
    fcast_df = pd.concat(fcast_dfs, verify_integrity=True)
    return fcast_df


def load_cache(start, end, data_dir, cols=None):
    result_df = pd.DataFrame()
    for ff in sorted(glob.glob(data_dir + "/all/all.*")):
        m = re.match(r".*all\.(\d{8})-(\d{8}).h5", str(ff))
        if m is None: continue
        d1 = dateparser.parse(m.group(1))
        d2 = dateparser.parse(m.group(2))
        if d2 <= start or d1 >= end: continue
        print("Loading {}".format(ff))
        df = pd.read_hdf(ff, 'full_df')
        if cols is not None:
            df = df[cols]
        #        df = df.truncate(before=start, after=end)
        result_df = result_df.append(df)
    #    result_df = result_df.truncate(before=start, after=end)
    result_df.index.names = ['date', 'gvkey']
    return result_df


def load_factor_cache(start, end, data_dir):
    dfs = []
    for ff in sorted(glob.glob(data_dir + "/all/all.*")):
        m = re.match(r".*all\.(\d{8})-(\d{8}).h5", str(ff))
        if m is None: continue
        d1 = dateparser.parse(m.group(1))
        d2 = dateparser.parse(m.group(2))
        if d2 <= start or d1 >= end: continue
        print("Loading {}".format(ff))
        df = pd.read_hdf(ff, 'full_df')
        df = df.truncate(before=start - timedelta(days=30), after=end)
        if len(df) > 0:
            dfs.append(df)
    result_df = pd.DataFrame()
    for df in dfs:
        result_df = result_df.append(df)
    result_df.index.names = ['date', 'factor']
    return result_df


def load_locates(uni_df, start, end, locates_dir):
    uni_df = uni_df.reset_index()
    monday_st = start - timedelta(days=start.weekday())
    monday_ed = end - timedelta(days=end.weekday())
    ff = locates_dir + "/locates/borrow.csv"
    print("Loading", ff)
    result_df = pd.read_csv(ff, parse_dates=['date'], usecols=['symbol','sedol', 'date', 'shares', 'fee'], sep='|')
    result_df = result_df.loc[(result_df['date'] >= monday_st) & (result_df['date'] <= monday_ed)]
    # because we have limited borrow data
    borrow_st = result_df['date'].min()
    borrow_ed = result_df['date'].max()
    borrow_ed += timedelta(days=6-borrow_ed.weekday())
    result_df[['borrow_qty','fee']] = -1 * result_df[['shares','fee']]
    del result_df['shares']
    sedol_df = pd.merge(result_df, uni_df[['gvkey', 'sedol']].drop_duplicates(), on=['sedol'])
    symbol_df = pd.merge(result_df, uni_df[['gvkey', 'symbol']].drop_duplicates(), on=['symbol'])
    result_df = pd.merge(symbol_df, sedol_df, on=['date', 'gvkey'], how='outer', suffixes=['', '_dead'])
    result_df[['borrow_qty','fee']].fillna(result_df[['borrow_qty_dead','fee_dead']], inplace=True)
    result_df = result_df[['date','gvkey','borrow_qty','fee']]
    result_df = pd.merge(uni_df, result_df, on=['date', 'gvkey'], how='outer')
    result_df = result_df.sort_values(by=['gvkey', 'date']).groupby(['gvkey'], as_index=False).fillna(method='ffill')
    result_df[['borrow_qty','fee']] = result_df[['borrow_qty','fee']].fillna(0)
    # limited borrow data
    result_df.loc[(result_df['date'] <= borrow_st) & (result_df['date'] >= borrow_ed), ['borrow_qty','fee']] =[-np.inf,-0]
    result_df.set_index(['date', 'gvkey'], inplace=True)
    return result_df
