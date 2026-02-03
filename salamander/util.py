"""
Utility Functions for Salamander Module

This module provides common data manipulation and file I/O utilities for the
salamander backtesting system. It handles merging of price/factor data, filtering,
alpha signal export, and loading of simulation results.

Key Functionality:
    - Data merging: Combine daily price, Barra factors, and intraday bar data
    - Column management: Remove duplicate columns with _dead suffix
    - DataFrame filtering: Filter by tradable universe or market cap
    - Alpha export: Dump forecast signals to timestamped CSV files
    - Results loading: Load simulation outputs from CSV files
    - Date utilities: Extract date ranges from DataFrame indices

Functions:
    merge_barra_data()       : Merge Barra factor data with price data
    merge_intra_eod()        : Merge end-of-day bar data with daily data
    merge_intra_data()       : Merge intraday bar data with daily data
    merge_daily_calcs()      : Merge computed daily features into main DataFrame
    merge_intra_calcs()      : Merge computed intraday features into main DataFrame
    remove_dup_cols()        : Remove columns ending with '_dead' suffix
    filter_expandable()      : Filter to tradable universe
    filter_pca()             : Filter by market cap (> $10B)
    get_overlapping_cols()   : Find non-overlapping columns between DataFrames
    dump_hd5()               : Export DataFrame to HDF5 with date-stamped filename
    dump_all()               : Export full intraday results to timestamped CSV files
    dump_alpha()             : Export single alpha signal to timestamped CSV files
    dump_prod_alpha()        : Export single alpha for production (latest date only)
    dump_daily_alpha()       : Export daily alpha to multiple intraday timestamps
    df_dates()               : Extract date range string from DataFrame index
    load_all_results()       : Load simulation results from CSV files
    load_merged_results()    : Load and merge results from multiple directories

Data Format Conventions:
    - MultiIndex: (date, gvkey) for daily data
    - MultiIndex: (iclose_ts, gvkey) for intraday data
    - Barra data is lagged by 1 day when merging (shift(1))
    - Duplicate columns from merges are suffixed with '_dead' and removed

Test Constants:
    testid  = '001075' : Sample gvkey for debugging
    testid2 = '143356' : Additional sample gvkey for debugging
"""

import sys
import os
import glob
import argparse
import re
import math
from collections import defaultdict
from dateutil import parser as dateparser

import time
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

import os, errno

testid = '001075'  # new #previous is 001045
testid2 = '143356'  # new

def merge_barra_data(price_df, barra_df):
    """
    Merge Barra risk factor data with price data, lagging Barra by 1 day.

    This function ensures that Barra factors are aligned with next-day prices,
    preventing look-ahead bias. Factors from day T are used to predict returns
    from T to T+1.

    Args:
        price_df (pd.DataFrame): Daily price data with MultiIndex (date, gvkey)
                                 Contains: close, high, low, volume, etc.
        barra_df (pd.DataFrame): Barra factor data with MultiIndex (date, gvkey)
                                 Contains: volatility, momentum, size, value, etc.

    Returns:
        pd.DataFrame: Merged DataFrame with MultiIndex (date, gvkey)
                     Contains all price columns + lagged Barra factors
                     Duplicate columns removed (suffixed with '_dead')

    Process:
        1. Unstack barra_df by gvkey (creates wide format)
        2. Shift forward by 1 day (lag the factors)
        3. Stack back to long format with (date, gvkey) index
        4. Merge with price_df on index
        5. Remove duplicate columns

    Example:
        Barra factors from 2013-01-02 will be aligned with prices from 2013-01-03
    """
    barra_df = barra_df.unstack(level=-1).shift(1).stack()
    full_df = pd.merge(price_df, barra_df, left_index=True, right_index=True, sort=True, suffixes=['', '_dead'])
    full_df = remove_dup_cols(full_df)
    return full_df


def remove_dup_cols(result_df):
    """
    Remove columns ending with '_dead' suffix from DataFrame.

    When merging DataFrames with overlapping column names, pandas adds suffixes
    to distinguish them (e.g., 'close' and 'close_dead'). This function removes
    the duplicate columns marked with '_dead' suffix.

    Args:
        result_df (pd.DataFrame): DataFrame potentially containing duplicate columns
                                  with '_dead' suffix

    Returns:
        pd.DataFrame: Same DataFrame with '_dead' columns removed

    Example:
        df with columns ['close', 'close_dead', 'volume']
        returns df with columns ['close', 'volume']
    """
    for col in result_df.columns:
        if col.endswith("_dead"):
            del result_df[col]
    return result_df


def merge_intra_eod(daily_df, intra_df):
    """
    Merge end-of-day (16:00) intraday bar data with daily data.

    Extracts the 16:00 bar from intraday data and merges it with daily data.
    This is useful for comparing daily close prices with intraday 16:00 prices.

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, gvkey)
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, gvkey)
                                 where iclose_ts is datetime with time component

    Returns:
        pd.DataFrame: Merged DataFrame with MultiIndex (date, gvkey)
                     Contains daily data + EOD bar data (suffixed with '_eod')
                     Duplicate columns and 'ticker_eod' removed

    Process:
        1. Unstack intraday data by gvkey
        2. Filter to 16:00 bars only using at_time()
        3. Stack back to long format
        4. Merge with daily data on (date, gvkey)
        5. Clean up duplicate columns and remove ticker_eod
        6. Set MultiIndex back to (date, gvkey)
    """
    print("Merging EOD bar data...")
    eod_df = intra_df.unstack().at_time('16:00').stack()
    merged_df = pd.merge(daily_df.reset_index(), eod_df.reset_index(), left_on=['date', 'gvkey'],
                         right_on=['date', 'gvkey'], sort=True, suffixes=['', '_eod'])
    merged_df = remove_dup_cols(merged_df)
    del merged_df['ticker_eod']
    merged_df.set_index(['date', 'gvkey'], inplace=True)
    return merged_df


def merge_intra_data(daily_df, intra_df):
    """
    Merge daily data into intraday bar data using left join.

    Broadcasts daily-level features (Barra factors, daily returns, etc.) to all
    intraday bars for that date. Each intraday bar gets the same daily features.

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, gvkey)
                                 Contains: Barra factors, daily prices, universe flags
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, gvkey)
                                 Contains: bar open, high, low, close, volume

    Returns:
        pd.DataFrame: Merged DataFrame with MultiIndex (iclose_ts, gvkey)
                     Contains all intraday bar data + daily features
                     Duplicate columns removed

    Process:
        1. Reset indices to enable merge on (date, gvkey)
        2. Left join intraday data with daily data
        3. Intraday rows without matching daily data get NaN
        4. Remove duplicate columns
        5. Set index back to (iclose_ts, gvkey)

    Note:
        - Uses left join to preserve all intraday bars
        - Daily features are broadcast to all bars on same date
        - iclose_ts must have a 'date' component for matching
    """
    print("Merging intra data...")
    merged_df = pd.merge(intra_df.reset_index(), daily_df.reset_index(), how='left', left_on=['date', 'gvkey'],
                         right_on=['date', 'gvkey'], sort=False, suffixes=['', '_dead'])
    merged_df = remove_dup_cols(merged_df)
    merged_df.set_index(['iclose_ts', 'gvkey'], inplace=True)
    return merged_df


def filter_expandable(df):
    """
    Filter DataFrame to expandable universe (tradable stocks).

    Restricts data to stocks that meet expanded tradability criteria:
    - Higher price threshold ($5+ typically)
    - Higher average daily volume ($5M+ typically)
    - Larger market cap
    This is a subset of the full tradable universe.

    Args:
        df (pd.DataFrame): DataFrame with 'expandable' boolean column

    Returns:
        pd.DataFrame: Filtered DataFrame containing only expandable stocks

    Side Effects:
        Prints: "Restricting forecast to expandables: {orig} -> {new}"

    Example:
        1,000 stocks -> 600 stocks after filtering to expandable universe
    """
    origsize = len(df)
    result_df = df.dropna(subset=['expandable'])
    result_df = result_df[result_df['expandable']]
    newsize = len(result_df)
    print("Restricting forecast to expandables: {} -> {}".format(origsize, newsize))
    return result_df


def filter_pca(df):
    """
    Filter DataFrame to large-cap stocks for PCA analysis.

    Restricts data to stocks with market cap > $10 billion. This is typically
    used for PCA decomposition where you want liquid, large-cap stocks to
    identify market-wide principal components.

    Args:
        df (pd.DataFrame): DataFrame with 'mkt_cap' column (market cap in dollars)

    Returns:
        pd.DataFrame: Filtered DataFrame containing only stocks with mkt_cap > $10B

    Side Effects:
        Prints: "Restricting forecast to expandables: {orig} -> {new}"
        Note: Message says "expandables" but actually filters by market cap

    Example:
        1,400 stocks -> 300 large-cap stocks for PCA analysis
    """
    origsize = len(df)
    result_df = df[df['mkt_cap'] > 1e10]
    newsize = len(result_df)
    print("Restricting forecast to expandables: {} -> {}".format(origsize, newsize))
    return result_df


def dump_hd5(result_df, name):
    """
    Export DataFrame to HDF5 file with date-stamped filename.

    Saves DataFrame to HDF5 format with filename including start and end dates
    from the DataFrame's index. Uses zlib compression.

    Args:
        result_df (pd.DataFrame): DataFrame with (date, gvkey) MultiIndex
        name (str): Base name for the file (without extension or dates)

    Output:
        Creates file: <name>.<start_date>-<end_date>.h5
        Example: "hl.20130101-20130630.h5"

    HDF5 Parameters:
        - Key: 'table'
        - Compression: zlib

    Example:
        dump_hd5(df, "alpha_hl")
        Creates: alpha_hl.20130101-20130630.h5
    """
    result_df.to_hdf(name + "." + df_dates(result_df) + ".h5", 'table', complib='zlib')


def dump_all(results_df):
    """
    Export all columns of intraday results to timestamped CSV files.

    Splits DataFrame by timestamp and saves each timestamp group to a separate
    CSV file. This is used for intraday simulations where you have multiple
    forecast times per day.

    Args:
        results_df (pd.DataFrame): Intraday results with MultiIndex (iclose_ts, gvkey)
                                   Contains all forecast columns and features

    Output Files:
        ./all/alpha.all.<YYYYMMDD_HHMM>.csv
        Example: ./all/alpha.all.20130102_1000.csv

    File Format:
        - CSV with headers, no index
        - All columns from results_df
        - One file per unique timestamp

    Process:
        1. Reset index to get iclose_ts as column
        2. Get unique timestamps
        3. For each timestamp (skip NaT):
           a. Filter data for that timestamp
           b. Create 'all' directory if needed
           c. Save to CSV with YYYYMMDD_HHMM format

    Side Effects:
        - Creates ./all/ directory if it doesn't exist
        - Prints progress for each timestamp group

    Example:
        Input: DataFrame with 20 timestamps x 1000 stocks
        Output: 20 CSV files, each with 1000 rows
    """
    print("Dumping alpha files...")
    results_df = results_df.reset_index()
    groups = results_df['iclose_ts'].unique()
    for group in groups:
        if str(group) == 'NaT': continue
        print("Dumping group: {}".format(str(group)))
        date_df = results_df[results_df['iclose_ts'] == group]
        if not len(date_df) > 0:
            print("No data found at ts: {}".format(str(group)))
            continue
        try:
            os.mkdir("all")
        except:
            pass
        filename = "./all/alpha.all." + pd.to_datetime(group).strftime('%Y%m%d_%H%M') + ".csv"
        date_df.to_csv(filename, index=False);


def dump_alpha(results_df, name):
    """
    Export single alpha signal to timestamped CSV files (intraday version).

    Extracts one named column from intraday results and saves each timestamp
    to a separate CSV file. This is used when you only need to export a specific
    alpha forecast, not all columns.

    Args:
        results_df (pd.DataFrame): Intraday results with MultiIndex (iclose_ts, gvkey)
                                   Must contain column specified by 'name'
        name (str): Name of the alpha column to export
                   Also used as directory name for output files

    Output Files:
        ./<name>/alpha.<name>.<YYYYMMDD_HHMM>.csv
        Example: ./hl/alpha.hl.20130102_1000.csv

    File Format:
        - CSV with headers, no index
        - Columns: gvkey, <name>
        - Float precision: 6 decimal places
        - One file per unique timestamp

    Process:
        1. Reset index to get iclose_ts as column
        2. Get unique timestamps
        3. Filter to [gvkey, iclose_ts, name] columns
        4. For each timestamp (skip NaT):
           a. Filter data for that timestamp
           b. Create directory ./<name>/ if needed
           c. Save to CSV with YYYYMMDD_HHMM format

    Side Effects:
        - Creates ./<name>/ directory if it doesn't exist
        - Prints progress for each timestamp group

    Example:
        dump_alpha(df, 'hl')
        Creates: ./hl/alpha.hl.20130102_1000.csv with columns [gvkey, hl]

    Note:
        This function is defined twice in the file (duplicate definition).
        Both versions are identical.
    """
    print("Dumping alpha files...")
    results_df = results_df.reset_index()
    groups = results_df['iclose_ts'].unique()

    results_df = results_df[['gvkey', 'iclose_ts', name]]
    for group in groups:
        if str(group) == 'NaT': continue

        print("Dumping group: {}".format(str(group)))
        date_df = results_df[results_df['iclose_ts'] == group]
        if not len(date_df) > 0:
            print("No data found at ts: {}".format(str(group)))
            continue
        try:
            os.mkdir(name)
        except:
            pass
        filename = "./" + name + "/alpha." + name + "." + pd.to_datetime(group).strftime('%Y%m%d_%H%M') + ".csv"
        date_df.to_csv(filename, index=False, cols=['gvkey', name], float_format="%.6f")


def dump_prod_alpha(results_df, name, outputfile):
    """
    Export single alpha signal for latest date only (production mode).

    This is used for live trading where you only need the most recent forecast.
    Extracts the latest date's alpha signal and saves to a specified file.

    Args:
        results_df (pd.DataFrame): Daily results with MultiIndex (date, gvkey)
                                   Must contain column specified by 'name'
        name (str): Name of the alpha column to export
        outputfile (str): Path to output CSV file

    Output File:
        - CSV with headers, no index
        - Columns: gvkey, <name>
        - Float precision: 6 decimal places
        - Contains only the latest date's data

    Process:
        1. Reset index to get date as column
        2. Find maximum (latest) date
        3. Filter to [gvkey, date, name] columns
        4. Filter to latest date only
        5. Save to specified output file

    Side Effects:
        Prints: "Dumping alpha files..."

    Example:
        dump_prod_alpha(df, 'hl', 'latest_hl.csv')
        Creates: latest_hl.csv with columns [gvkey, hl] for latest date only
    """
    print("Dumping alpha files...")
    results_df = results_df.reset_index()
    group = results_df['date'].unique().max()
    results_df = results_df[['gvkey', 'date', name]]
    date_df = results_df[results_df['date'] == group]
    date_df.to_csv(outputfile, index=False, cols=['gvkey', name], float_format="%.6f")


def dump_daily_alpha(results_df, name):
    """
    Export daily alpha signal to multiple intraday timestamp files.

    Takes daily alpha forecasts and replicates them across all intraday timestamps
    (every 15 minutes from 09:30 to 15:45). This is used when you have a daily
    signal but need it in intraday format for qsim.py.

    Args:
        results_df (pd.DataFrame): Daily results with MultiIndex (date, gvkey)
                                   Must contain column specified by 'name'
        name (str): Name of the alpha column to export
                   Also used as directory name for output files

    Output Files:
        ./<name>/alpha.<name>.<YYYYMMDD_HHMM>.csv
        Example: ./hl/alpha.hl.20130102_0930.csv
                 ./hl/alpha.hl.20130102_0945.csv
                 ... (one file every 15 min from 09:30 to 15:45)

    File Format:
        - CSV with headers, no index
        - Columns: gvkey, <name>
        - Float precision: 6 decimal places
        - Same alpha values replicated for all timestamps of a given date

    Intraday Timestamps:
        09:30, 09:45, 10:00, 10:15, 10:30, 10:45, 11:00, 11:15, 11:30, 11:45,
        12:00, 12:15, 12:30, 12:45, 13:00, 13:15, 13:30, 13:45, 14:00, 14:15,
        14:30, 14:45, 15:00, 15:15, 15:30, 15:45
        (26 timestamps per day)

    Process:
        1. Reset index to get date as column
        2. Get unique dates
        3. Filter to [gvkey, date, name] columns
        4. For each date (skip NaT):
           a. Filter data for that date
           b. Create directory ./<name>/ if needed
           c. For each of 26 intraday timestamps:
              - Create filename with YYYYMMDD_HHMM format
              - Save same data to that file (alpha values constant during day)

    Side Effects:
        - Creates ./<name>/ directory if it doesn't exist
        - Prints progress for each date group
        - Creates 26 files per trading day

    Example:
        dump_daily_alpha(df, 'hl')
        For 1 trading day with 1000 stocks:
        Creates 26 files, each with 1000 rows of identical alpha values

    Use Case:
        Convert daily alpha signals to intraday format for qsim.py intraday backtesting
    """
    print("Dumping daily alpha files...")
    results_df = results_df.reset_index()
    groups = results_df['date'].unique()

    results_df = results_df[['gvkey', 'date', name]]
    for group in groups:
        if str(group) == 'NaT': continue

        print("Dumping group: {}".format(str(group)))
        date_df = results_df[results_df['date'] == group]
        if not len(date_df) > 0:
            print("No data found at ts: {}".format(str(group)))
            continue
        try:
            os.mkdir(name)
        except:
            pass

        for stime in ['0930', '0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130', '1145', '1200', '1215',
                      '1230', '1245', '1300', '1315', '1330', '1345', '1400', '1415', '1430', '1445', '1500', '1515',
                      '1530', '1545']:
            filename = "./" + name + "/alpha." + name + "." + pd.to_datetime(group).strftime(
                '%Y%m%d_' + str(stime)) + ".csv"
            date_df.to_csv(filename, index=False, cols=['gvkey', name], float_format="%.6f")


def df_dates(df):
    """
    Extract date range string from DataFrame index.

    Extracts the first and last dates from a DataFrame's MultiIndex and formats
    them as a date range string. Used for creating date-stamped filenames.

    Args:
        df (pd.DataFrame): DataFrame with MultiIndex where first level is date
                          Index format: (date, gvkey) or (date, other_key)

    Returns:
        str: Date range formatted as "YYYYMMDD-YYYYMMDD"
             Example: "20130101-20130630"

    Process:
        1. Get first element of index: df.index[0][0]
        2. Get last element of index: df.index[-1][0]
        3. Format both as YYYYMMDD using strftime
        4. Join with hyphen

    Example:
        df with dates from Jan 1 2013 to Jun 30 2013
        Returns: "20130101-20130630"

    Note:
        - Assumes index is sorted by date
        - Assumes first level of MultiIndex is datetime
        - If df.index contains raw integers instead of datetime, strftime will fail
    """
    return df.index[0][0].strftime("%Y%m%d") + "-" + df.index[-1][0].strftime("%Y%m%d")
    # new: if hl.py runs on raw data, no strftime


def merge_daily_calcs(full_df, result_df):
    """
    Merge computed daily features into main DataFrame (left join).

    Adds new calculated columns from result_df to full_df without duplicating
    columns that already exist. Uses left join to preserve all rows from full_df.

    Args:
        full_df (pd.DataFrame): Main DataFrame with MultiIndex (date, gvkey)
                                Contains base price, factor, and universe data
        result_df (pd.DataFrame): Computed features with MultiIndex (date, gvkey)
                                  Contains newly calculated columns to add

    Returns:
        pd.DataFrame: Merged DataFrame with MultiIndex (date, gvkey)
                     Contains all full_df columns + new columns from result_df
                     Overlapping columns from full_df are preserved (not replaced)

    Process:
        1. Find columns in result_df that don't exist in full_df
        2. Reset indices for merge
        3. Add 'date' and 'gvkey' to merge columns
        4. Left join on (date, gvkey)
        5. Use suffixes ['_dead', ''] so full_df columns are preserved
        6. Set index back to (date, gvkey)

    Side Effects:
        Prints: "Merging daily results: [list of new columns]"

    Example:
        full_df has: [close, volume, volatility]
        result_df has: [volatility, hl0, hl_forecast]
        result: [close, volume, volatility, hl0, hl_forecast]
        (volatility from full_df is kept, not replaced)
    """
    rcols = set(result_df.columns)
    cols = list(rcols - set(full_df.columns))
    result_df = result_df.reset_index()
    full_df = full_df.reset_index()
    cols.extend(['date', 'gvkey'])
    print("Merging daily results: " + str(cols))
    result_df = pd.merge(full_df, result_df[cols], how='left', left_on=['date', 'gvkey'], right_on=['date', 'gvkey'],
                         sort=False, suffixes=['_dead', ''])
    result_df.set_index(['date', 'gvkey'], inplace=True)
    return result_df


def merge_intra_calcs(full_df, result_df):
    """
    Merge computed intraday features into main DataFrame (left join).

    Adds new calculated columns from result_df to full_df for intraday data.
    Uses index-based merge to preserve the intraday timestamp index.

    Args:
        full_df (pd.DataFrame): Main DataFrame with MultiIndex (iclose_ts, gvkey)
                                Contains base intraday bar and daily feature data
        result_df (pd.DataFrame): Computed features with MultiIndex (iclose_ts, gvkey)
                                  Contains newly calculated intraday columns
                                  May also have a 'date' column

    Returns:
        pd.DataFrame: Merged DataFrame with MultiIndex (iclose_ts, gvkey)
                     Contains all full_df columns + new columns from result_df
                     Overlapping columns from full_df are preserved (not replaced)

    Process:
        1. Delete 'date' column from result_df (prevents NaT issues in merge)
        2. Find columns in result_df that don't exist in full_df
        3. Left join on index (iclose_ts, gvkey)
        4. Use suffixes ['_dead', ''] so full_df columns are preserved
        5. Return merged result

    Side Effects:
        - Modifies result_df by deleting 'date' column
        - Prints: "Merging intra results: [list of new columns]"

    Note:
        Deleting 'date' is important because result_df may have NaT values in
        the 'date' column which can cause issues in subsequent operations.

    Example:
        full_df has: [iopen, iclose, volume, volatility]
        result_df has: [date, volatility, qhl_forecast]
        result: [iopen, iclose, volume, volatility, qhl_forecast]
        ('date' deleted, volatility from full_df kept)
    """
    # important for keeping NaTs out of the following merge
    del result_df['date']
    rcols = set(result_df.columns)
    cols = list(rcols - set(full_df.columns))
    print("Merging intra results: " + str(cols))
    result_df = pd.merge(full_df, result_df[cols], how='left', left_index=True, right_index=True, sort=False,
                         suffixes=['_dead', ''])
    return result_df


def get_overlapping_cols(df1, df2):
    """
    Find columns in df1 that are NOT in df2.

    Returns columns that exist in df1 but not in df2. Despite the name
    "overlapping", this actually returns NON-overlapping columns.

    Args:
        df1 (pd.DataFrame): First DataFrame
        df2 (pd.DataFrame): Second DataFrame

    Returns:
        list: Column names that exist in df1 but not in df2

    Example:
        df1 columns: ['close', 'volume', 'hl']
        df2 columns: ['volume', 'volatility']
        Returns: ['close', 'hl']

    Note:
        Function name is misleading - it returns non-overlapping columns,
        not overlapping ones.
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    res = cols1 - cols1.intersection(cols2)
    return list(res)


def load_merged_results(fdirs, start, end, cols=None):
    """
    Load and merge simulation results from multiple directories.

    Loads alpha files from multiple result directories and merges them on
    the (iclose_ts, gvkey) index. This is useful for combining different
    alpha signals or comparing results from different runs.

    Args:
        fdirs (list of str): List of directory paths to load results from
                            Each should contain an 'all/' subdirectory
        start (str): Starting date filter (format: YYYYMMDD)
        end (str): Ending date filter (format: YYYYMMDD)
        cols (list of str, optional): Specific columns to load from CSV files
                                      If None, loads all columns

    Returns:
        pd.DataFrame: Merged results with MultiIndex (iclose_ts, gvkey)
                     Contains columns from all directories
                     Duplicate columns removed (suffixed with '_dead')

    Process:
        1. For each directory in fdirs:
           a. Load all results using load_all_results()
           b. If first directory, use as merged_df
           c. Otherwise, merge with existing merged_df on index
           d. Remove duplicate columns after each merge
        2. Return final merged DataFrame

    Example:
        fdirs = ['/data/run1', '/data/run2']
        df = load_merged_results(fdirs, '20130101', '20130630')
        # df contains merged results from both directories

    See Also:
        load_all_results() : Loads results from single directory
    """
    merged_df = None
    for fdir in fdirs:
        df = load_all_results(fdir, start, end, cols)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, suffixes=['', '_dead'])
            merged_df = remove_dup_cols(merged_df)
    return merged_df


def load_all_results(fdir, start, end, cols=None):
    """
    Load intraday simulation results from CSV files in a directory.

    Scans a directory for alpha result files, filters by date and time range,
    and loads them into a single DataFrame. Used for analyzing simulation outputs.

    Args:
        fdir (str): Base directory path
                   Will append '/all/' to look for alpha files
        start (str): Starting date filter (format: YYYYMMDD)
                    Only load files with date >= start
        end (str): Ending date filter (format: YYYYMMDD)
                  Only load files with date <= end
        cols (list of str, optional): Specific columns to load from CSV files
                                      If None, loads all columns
                                      Should include ['iclose_ts', 'gvkey'] for index

    Returns:
        pd.DataFrame: Concatenated results with MultiIndex (iclose_ts, gvkey)
                     Contains all matching files' data
                     Index is datetime parsed from iclose_ts

    File Pattern:
        Expects files matching: <fdir>/all/alpha.all.<YYYYMMDD>_<HHMM>.csv

    Filters:
        - Date: YYYYMMDD must be in [start, end] range
        - Time: HHMM must be in [1000, 1530] range (10:00 AM to 3:30 PM)

    Process:
        1. Append '/all/' to fdir
        2. Scan for files matching alpha.* pattern
        3. For each file:
           a. Extract date and time from filename using regex
           b. Skip if time outside [10:00, 15:30] range
           c. Skip if date outside [start, end] range
           d. Load CSV with MultiIndex (iclose_ts, gvkey)
           e. Parse iclose_ts as datetime
           f. Add to list of DataFrames
        4. Concatenate all DataFrames with integrity check
        5. Return concatenated result

    Side Effects:
        Prints: "Looking in <fdir>/all/"
        Prints: "Loading <filename> for <date>" for each file

    Example:
        df = load_all_results('/data/sim1', '20130101', '20130630')
        # Loads all alpha files from /data/sim1/all/ for first half of 2013
        # Only includes timestamps between 10:00 and 15:30

    Raises:
        ValueError: If verify_integrity=True in concat and index has duplicates

    Note:
        Time filter excludes early morning (before 10:00) and late afternoon
        (after 15:30) to focus on main trading hours.
    """
    fdir += "/all/"
    print("Looking in {}".format(fdir))
    fcast_dfs = list()
    for ff in sorted(glob.glob(fdir + "/alpha.*")):
        m = re.match(r".*alpha\.all\.(\d{8})_(\d{4}).*", str(ff))
        fdate = int(m.group(1))
        ftime = int(m.group(2))
        if ftime < 1000 or ftime > 1530: continue
        if fdate < int(start) or fdate > int(end): continue
        print("Loading {} for {}".format(ff, fdate))

        if cols is not None:
            df = pd.read_csv(ff, index_col=['iclose_ts', 'gvkey'], header=0, parse_dates=True, sep=",", usecols=cols)
        else:
            df = pd.read_csv(ff, index_col=['iclose_ts', 'gvkey'], header=0, parse_dates=True, sep=",")

        fcast_dfs.append(df)

    fcast_df = pd.concat(fcast_dfs, verify_integrity=True)

    return fcast_df
