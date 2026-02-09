#!/usr/bin/env python
"""
Utility Functions Module

This module provides common utility functions used throughout the statistical
arbitrage system for data manipulation, file operations, and data merging.

Key Functions:
    mkdir_p(): Create directory with parents (similar to mkdir -p)
    email(): Send email notifications for alerts and reports
    merge_barra_data(): Merge Barra factor data with price data
    merge_intra_eod(): Merge end-of-day bar data with daily data
    merge_intra_data(): Merge intraday data with daily data
    filter_expandable(): Filter securities to expandable universe
    filter_pca(): Filter securities for PCA analysis (large cap only)
    dump_hd5(): Save DataFrame to HDF5 format with compression
    dump_all(): Export alpha signals to files grouped by timestamp
    remove_dup_cols(): Remove duplicate columns after merges

Data Merging Strategies:
    - Lagging Barra data by 1 day to avoid look-ahead bias
    - Joining intraday and daily data on (date, sid) keys
    - Handling time zones and timestamp alignment
    - Removing duplicate columns with '_dead' suffix

The module ensures data integrity and proper temporal alignment to prevent
look-ahead bias in backtesting.
"""

from __future__ import division, print_function

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

testid = 10020731
testid2 = 10000678

def mkdir_p(path):
    """
    Create directory with parent directories (equivalent to mkdir -p).

    Creates the specified directory and all parent directories as needed,
    similar to the Unix 'mkdir -p' command. If the directory already exists,
    no error is raised.

    Args:
        path (str): Path to directory to create

    Raises:
        Exception: If directory creation fails for reasons other than
                   directory already existing

    Example:
        >>> mkdir_p("/tmp/path/to/deep/directory")
        >>> # Creates all intermediate directories if they don't exist
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise Exception("Could not create {}".format(path))

def email(subj, message):
    """
    Send email notification via local SMTP server.

    Used for alerts, error notifications, and daily reports from the
    trading system. Requires local SMTP server running on localhost.

    Args:
        subj (str): Email subject line
        message (str): Email body content (plain text)

    Example:
        >>> email("Backtest Complete", "BSIM finished with Sharpe=1.8")

    Note:
        Configured to send to/from sean@quantbot.com. Modify recipients
        in function body for different email addresses.
    """
    # Import smtplib for the actual sending function
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(message)
    msg['Subject'] = subj
    msg['From'] = "sean@quantbot.com"
    msg['To'] = "sean@quantbot.com"
    s = smtplib.SMTP('localhost')
    s.sendmail("sean@quantbot.com", ["sean@quantbot.com"], msg.as_string())
    s.quit()

def merge_barra_data(price_df, barra_df):
    """
    Merge Barra factor exposures with price data, lagging by 1 day.

    Joins Barra risk model factor exposures and industry classifications
    with price/volume data. Lags Barra data by 1 day to avoid look-ahead
    bias (Barra data for date T is not available until end of day T, so
    we use T-1 data for trading decisions on day T).

    Args:
        price_df (DataFrame): Price data with (date, sid) MultiIndex
        barra_df (DataFrame): Barra factor data with (date, sid) MultiIndex

    Returns:
        DataFrame: Merged data with lagged Barra factors, duplicate columns removed

    Example:
        >>> full_df = merge_barra_data(price_df, barra_df)
        >>> # Barra 'momentum' column is lagged 1 day relative to prices

    Implementation:
        1. Unstack Barra data by sid (columns = sids)
        2. Shift all data forward by 1 day (lag)
        3. Restack to restore (date, sid) MultiIndex
        4. Merge with price data on index
        5. Remove duplicate columns from merge
    """
    # Handle empty Barra DataFrame edge case
    if len(barra_df) == 0:
        return price_df
    barra_df = barra_df.unstack(level=-1).shift(1).stack()
    full_df = pd.merge(price_df, barra_df, left_index=True, right_index=True, sort=True, suffixes=['', '_dead'])
    full_df = remove_dup_cols(full_df)
    return full_df

def remove_dup_cols(result_df):
    """
    Remove duplicate columns created during merge operations.

    After merging DataFrames with overlapping columns, pandas creates
    duplicate columns with '_dead' suffix. This function removes those
    duplicates, keeping only the original column names.

    Args:
        result_df (DataFrame): DataFrame potentially containing '_dead' columns

    Returns:
        DataFrame: Same DataFrame with '_dead' columns removed

    Example:
        >>> # After merge: columns = ['close', 'close_dead', 'volume']
        >>> df = remove_dup_cols(df)
        >>> # After: columns = ['close', 'volume']

    Note:
        Used throughout merge operations in conjunction with
        suffixes=['', '_dead'] parameter in pd.merge()
    """
    for col in result_df.columns:
        if col.endswith("_dead"):
            del result_df[col]
    return result_df

def merge_intra_eod(daily_df, intra_df):
    """
    Merge end-of-day (4:00 PM) bar data with daily data.

    Extracts 16:00 (4:00 PM close) bars from intraday data and merges
    with daily OHLCV data. Used to combine daily fundamental data with
    end-of-day intraday bar statistics.

    Args:
        daily_df (DataFrame): Daily data with (date, sid) MultiIndex
        intra_df (DataFrame): Intraday bar data with (timestamp, sid) MultiIndex

    Returns:
        DataFrame: Merged data with (date, sid) MultiIndex, EOD bars appended
                   with '_eod' suffix

    Example:
        >>> merged = merge_intra_eod(daily_df, bar_df)
        >>> # merged contains both 'close' (daily) and 'iclose_eod' (4pm bar)

    Implementation:
        1. Unstack intraday data by sid
        2. Select only 16:00 timestamp rows
        3. Restack to (timestamp, sid) index
        4. Merge on (date, sid) keys
        5. Remove duplicate columns and redundant ticker_eod
    """
    print("Merging EOD bar data...")
    eod_df = intra_df.unstack().at_time('16:00').stack()
    merged_df = pd.merge(daily_df.reset_index(), eod_df.reset_index(), left_on=['date', 'sid'], right_on=['date', 'sid'], sort=True, suffixes=['', '_eod'])
    merged_df = remove_dup_cols(merged_df)
    del merged_df['ticker_eod']
    merged_df.set_index(['date', 'sid'], inplace=True)
    return merged_df

def merge_intra_data(daily_df, intra_df):
    """
    Merge daily data into intraday data for each bar.

    Left-joins daily fundamental data (Barra factors, market cap, etc.)
    onto every intraday bar, expanding daily data to intraday frequency.
    Used to make daily features available at every 30-minute bar for
    intraday alpha generation.

    Args:
        daily_df (DataFrame): Daily data with (date, sid) MultiIndex
        intra_df (DataFrame): Intraday bar data with (iclose_ts, sid) MultiIndex

    Returns:
        DataFrame: Intraday-indexed data with daily columns appended,
                   indexed by (iclose_ts, sid)

    Example:
        >>> intra_full = merge_intra_data(daily_df, bar_df)
        >>> # Daily 'mkt_cap' now available at every 30-min bar

    Implementation:
        1. Reset both indexes to columns
        2. Left join intra onto daily using (date, sid) keys
        3. Remove duplicate columns
        4. Restore (iclose_ts, sid) MultiIndex

    Note:
        Left join ensures all intraday bars are preserved even if
        daily data is missing (forward-fills daily data across bars)
    """
    print("Merging intra data...")
    merged_df = pd.merge(intra_df.reset_index(), daily_df.reset_index(), how='left', left_on=['date', 'sid'], right_on=['date', 'sid'], sort=False, suffixes=['', '_dead'])
    merged_df = remove_dup_cols(merged_df)
    merged_df.set_index(['iclose_ts', 'sid'], inplace=True)
    return merged_df

def filter_expandable(df):
    """
    Filter DataFrame to only expandable universe securities.

    Restricts data to securities in the expandable universe (higher
    liquidity threshold than base tradable universe). Used when generating
    alpha signals for strategies that require higher capacity.

    Args:
        df (DataFrame): Data with 'expandable' boolean column

    Returns:
        DataFrame: Filtered data containing only expandable=True rows

    Example:
        >>> # Full universe: 1400 stocks, expandable: ~800 stocks
        >>> filtered_df = filter_expandable(full_df)
        >>> # Restricting forecast to expandables: 1400 -> 800

    Prints:
        Count of securities before and after filtering

    Note:
        Expandable universe defined in loaddata.py:
        - Min price: $5 (vs $2 for tradable)
        - Min ADV: $5M (vs $1M for tradable)
    """
    origsize = len(df)
    # Handle empty DataFrame edge case
    if len(df) == 0:
        print("Restricting forecast to expandables: {} -> {}".format(origsize, 0))
        return df
    result_df = df.dropna(subset=['expandable'])
    result_df = result_df[ result_df['expandable'] ]
    newsize = len(result_df)
    print("Restricting forecast to expandables: {} -> {}".format(origsize, newsize))
    return result_df

def filter_pca(df):
    """
    Filter DataFrame to large-cap stocks for PCA analysis.

    Restricts data to securities with market cap > $10B for principal
    component analysis. PCA is performed on large caps to identify market
    and sector factors, which are then applied to the full universe.

    Args:
        df (DataFrame): Data with 'mkt_cap' column (market cap in dollars)

    Returns:
        DataFrame: Filtered data containing only large-cap stocks

    Example:
        >>> # Filter to mega-cap for factor decomposition
        >>> large_cap_df = filter_pca(full_df)
        >>> # Restricting forecast to expandables: 1400 -> 200

    Threshold:
        Market cap > $10 billion (1e10)

    Note:
        Despite print message saying "expandables", this filters by
        market cap for PCA decomposition (see pca.py)
    """
    origsize = len(df)
    result_df = df[ df['mkt_cap'] > 1e10 ]
    newsize = len(result_df)
    print("Restricting forecast to expandables: {} -> {}".format(origsize, newsize))
    return result_df
    

def dump_hd5(result_df, name):
    """
    Save DataFrame to compressed HDF5 file with date range in filename.

    Exports DataFrame to HDF5 format with zlib compression, using the
    date range from the index to generate a descriptive filename.

    Args:
        result_df (DataFrame): Data to save with (date, sid) MultiIndex
        name (str): Base filename (without extension)

    Generates:
        File: {name}.{start_date}-{end_date}.h5

    Example:
        >>> dump_hd5(results_df, "alpha_hl")
        >>> # Creates: alpha_hl.20130101-20130630.h5

    Storage:
        - Format: HDF5 'table' format (query-able)
        - Compression: zlib (good balance of speed/size)

    Note:
        Requires PyTables library for HDF5 support
    """
    result_df.to_hdf(name + "." + df_dates(result_df) + ".h5", 'table', complib='zlib')

def dump_all(results_df):
    """
    Export all alpha signals to CSV files grouped by timestamp.

    Splits intraday alpha signals by timestamp and writes separate CSV
    files for each bar time. Used to export comprehensive intraday alpha
    data for all signals combined.

    Args:
        results_df (DataFrame): Intraday alpha data with (iclose_ts, sid) index

    Generates:
        Directory: ./all/
        Files: alpha.all.{YYYYMMDD_HHMM}.csv for each unique timestamp

    Example:
        >>> dump_all(results_df)
        >>> # Creates ./all/alpha.all.20130115_0930.csv, etc.

    Output Format:
        CSV with all columns from results_df, one file per timestamp

    Note:
        - Skips NaT (Not-a-Time) timestamps
        - Creates 'all' directory if it doesn't exist
        - Used for comprehensive intraday signal archival
    """
    print("Dumping alpha files...")
    results_df = results_df.reset_index()
    groups = results_df['iclose_ts'].unique()
    for group in groups:
        if str(group) == 'NaT': continue
        print("Dumping group: {}".format(str(group)))
        date_df = results_df[ results_df['iclose_ts'] == group ]
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
    Export single alpha signal to CSV files grouped by timestamp.

    Writes intraday alpha signal values to separate CSV files for each
    bar time. Used to export individual alpha strategies (e.g., 'hl', 'bd')
    for loading into simulation or production systems.

    Args:
        results_df (DataFrame): Intraday alpha data with (iclose_ts, sid) index
        name (str): Alpha signal column name and output directory name

    Generates:
        Directory: ./{name}/
        Files: alpha.{name}.{YYYYMMDD_HHMM}.csv for each unique timestamp

    Example:
        >>> dump_alpha(results_df, 'hl')
        >>> # Creates ./hl/alpha.hl.20130115_0930.csv, etc.

    Output Format:
        CSV with columns: sid, {name}
        Float precision: 6 decimal places

    Note:
        - Function is duplicated in source (lines 160-179, 182-201)
        - Skips NaT timestamps
        - Creates output directory if needed
        - Only exports sid and named alpha column
    """
    print("Dumping alpha files...")
    results_df = results_df.reset_index()
    groups = results_df['iclose_ts'].unique()

    results_df = results_df[ ['sid', 'iclose_ts', name] ]
    for group in groups:
        if str(group) == 'NaT': continue

        print("Dumping group: {}".format(str(group)))
        date_df = results_df[ results_df['iclose_ts'] == group ]
        if not len(date_df) > 0:
            print("No data found at ts: {}".format(str(group)))
            continue
        try:
            os.mkdir(name)
        except:
            pass
        filename = "./"+name+"/alpha." + name + "." + pd.to_datetime(group).strftime('%Y%m%d_%H%M') + ".csv"
        date_df.to_csv(filename, index=False, cols=['sid', name], float_format="%.6f")

def dump_prod_alpha(results_df, name, outputfile):
    """
    Export most recent daily alpha signal to CSV for production use.

    Extracts the latest date from daily alpha signals and writes to a
    single CSV file. Used in production to generate today's alpha signals
    for live trading.

    Args:
        results_df (DataFrame): Daily alpha data with (date, sid) index
        name (str): Alpha signal column name to export
        outputfile (str): Output CSV file path

    Generates:
        File: Single CSV with latest date's alpha values

    Example:
        >>> dump_prod_alpha(results_df, 'hl', './prod/alpha_hl.csv')
        >>> # Writes only most recent date to prod/alpha_hl.csv

    Output Format:
        CSV with columns: sid, {name}
        Float precision: 6 decimal places

    Note:
        - Automatically selects max(date) from data
        - No timestamp suffix in filename (specified by user)
        - Used by production alpha generation scripts
    """
    print("Dumping alpha files...")
    results_df = results_df.reset_index()
    group = results_df['date'].unique().max()
    results_df = results_df[ ['sid', 'date', name] ]
    date_df = results_df[ results_df['date'] == group ]
    date_df.to_csv(outputfile, index=False, cols=['sid', name], float_format="%.6f")


def dump_daily_alpha(results_df, name):
    """
    Export daily alpha signals replicated across intraday timestamps.

    Writes daily alpha values to multiple CSV files for each day, with
    one file per 15-minute interval. Same alpha values are replicated
    across all intraday timestamps for that day. Used to provide daily
    alpha signals in intraday format for QSIM.

    Args:
        results_df (DataFrame): Daily alpha data with (date, sid) index
        name (str): Alpha signal column name and output directory name

    Generates:
        Directory: ./{name}/
        Files: alpha.{name}.{YYYYMMDD_HHMM}.csv for each date and time

    Example:
        >>> dump_daily_alpha(results_df, 'hl')
        >>> # Creates ./hl/alpha.hl.20130115_0930.csv
        >>> #         ./hl/alpha.hl.20130115_0945.csv
        >>> #         ... (26 files per day, same values)

    Timestamps:
        Files generated for 15-minute intervals from 9:30 AM to 3:45 PM
        (26 files per trading day)

    Output Format:
        CSV with columns: sid, {name}
        Float precision: 6 decimal places

    Note:
        Daily alpha values are identical across all intraday files for
        a given date. Used to make daily signals compatible with intraday
        simulation infrastructure.
    """
    print("Dumping daily alpha files...")
    results_df = results_df.reset_index()
    groups = results_df['date'].unique()

    results_df = results_df[ ['sid', 'date', name] ]
    for group in groups:
        if str(group) == 'NaT': continue

        print("Dumping group: {}".format(str(group)))
        date_df = results_df[ results_df['date'] == group ]
        if not len(date_df) > 0:
            print("No data found at ts: {}".format(str(group)))
            continue
        try:
            os.mkdir(name)
        except:
            pass

        for stime in ['0930', '0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130', '1145', '1200', '1215', '1230', '1245', '1300', '1315', '1330', '1345', '1400', '1415', '1430', '1445', '1500', '1515', '1530', '1545']:
            filename = "./"+name+"/alpha." + name + "." + pd.to_datetime(group).strftime('%Y%m%d_' + str(stime)) + ".csv"
            date_df.to_csv(filename, index=False, cols=['sid', name], float_format="%.6f")

def df_dates(df):
    """
    Extract date range string from DataFrame index.

    Generates a date range string in format "YYYYMMDD-YYYYMMDD" from
    the first and last dates in a DataFrame's MultiIndex. Used for
    creating descriptive filenames.

    Args:
        df (DataFrame): Data with (date, sid) MultiIndex where date is first level

    Returns:
        str: Date range in format "YYYYMMDD-YYYYMMDD"

    Example:
        >>> df_dates(results_df)
        '20130101-20130630'

    Note:
        Assumes MultiIndex with date as first level ([0][0] = first date)
        Used by dump_hd5() to generate filename with date range
    """
    return df.index[0][0].strftime("%Y%m%d") + "-" + df.index[len(df)-1][0].strftime("%Y%m%d")

def merge_daily_calcs(full_df, result_df):
    """
    Merge daily calculation results into main DataFrame.

    Adds new columns from result_df to full_df, preserving all rows from
    full_df (left join). Only merges columns that don't already exist in
    full_df to avoid duplicates. Used to incrementally add calculated
    features to the main data pipeline.

    Args:
        full_df (DataFrame): Main data with (date, sid) MultiIndex
        result_df (DataFrame): Calculated features with (date, sid) MultiIndex

    Returns:
        DataFrame: full_df with new columns from result_df added,
                   (date, sid) MultiIndex preserved

    Example:
        >>> # full_df has: close, volume, ret_1
        >>> # result_df has: close, ret_1, alpha_hl (new)
        >>> merged = merge_daily_calcs(full_df, result_df)
        >>> # merged adds only alpha_hl column

    Implementation:
        1. Identify columns in result_df not in full_df
        2. Reset both indexes to regular columns
        3. Left merge on (date, sid) keys
        4. Restore (date, sid) MultiIndex

    Note:
        Suffix '_dead' is applied to duplicates from full_df,
        '' (no suffix) to result_df columns
    """
    rcols = set(result_df.columns)
    cols = list(rcols - set(full_df.columns))
    result_df = result_df.reset_index()
    full_df = full_df.reset_index()
    cols.extend(['date', 'sid'])
    print("Merging daily results: " + str(cols))
    result_df = pd.merge(full_df, result_df[cols], how='left', left_on=['date', 'sid'], right_on=['date', 'sid'], sort=False, suffixes=['_dead', ''])
    result_df.set_index(['date', 'sid'], inplace=True)
    return result_df

def merge_intra_calcs(full_df, result_df):
    """
    Merge intraday calculation results into main DataFrame.

    Adds new columns from result_df to full_df using index-based join,
    preserving all rows from full_df (left join). Only merges columns
    that don't already exist in full_df. Used in intraday pipeline to
    add calculated features at each bar timestamp.

    Args:
        full_df (DataFrame): Main intraday data with (iclose_ts, sid) MultiIndex
        result_df (DataFrame): Calculated features with (iclose_ts, sid) MultiIndex

    Returns:
        DataFrame: full_df with new columns from result_df added,
                   (iclose_ts, sid) MultiIndex preserved

    Example:
        >>> # full_df has: iclose, ivol, bar_ret
        >>> # result_df has: date, bar_ret, alpha_qhl (new)
        >>> merged = merge_intra_calcs(full_df, result_df)
        >>> # merged adds only alpha_qhl column

    Implementation:
        1. Delete 'date' column from result_df (can have NaT mismatches)
        2. Identify columns in result_df not in full_df
        3. Left merge on shared MultiIndex
        4. MultiIndex preserved automatically

    Note:
        Deletes 'date' column from result_df to avoid NaT (Not-a-Time)
        mismatches between daily date and intraday timestamp indexes
    """
    #important for keeping NaTs out of the following merge
    del result_df['date']
    rcols = set(result_df.columns)
    cols = list(rcols - set(full_df.columns))
    print("Merging intra results: " + str(cols))
    result_df = pd.merge(full_df, result_df[cols], how='left', left_index=True, right_index=True, sort=False, suffixes=['_dead', ''])
    return result_df

def get_overlapping_cols(df1, df2):
    """
    Get columns in df1 that are NOT in df2.

    Returns the set difference of column names, identifying which columns
    exist in df1 but not in df2. Used for selective merging and column
    comparison operations.

    Args:
        df1 (DataFrame): First DataFrame
        df2 (DataFrame): Second DataFrame

    Returns:
        list: Column names in df1 that are not in df2

    Example:
        >>> # df1 columns: ['a', 'b', 'c']
        >>> # df2 columns: ['b', 'd']
        >>> get_overlapping_cols(df1, df2)
        ['a', 'c']

    Note:
        Despite the name "overlapping", this returns NON-overlapping
        columns (those unique to df1)
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    res = cols1 - cols1.intersection(cols2)
    return list(res)

def load_merged_results(fdirs, start, end, cols=None):
    """
    Load and merge alpha results from multiple directories.

    Loads alpha signal CSV files from multiple directories and merges
    them into a single DataFrame. Used to combine different alpha
    strategies for multi-alpha backtesting or analysis.

    Args:
        fdirs (list): List of directory paths containing alpha files
        start (str/int): Start date in YYYYMMDD format
        end (str/int): End date in YYYYMMDD format
        cols (list, optional): Column names to load (None = all columns)

    Returns:
        DataFrame: Merged alpha signals from all directories,
                   indexed by (iclose_ts, sid)

    Example:
        >>> dirs = ['./hl', './bd', './pca']
        >>> alphas = load_merged_results(dirs, 20130101, 20130630)
        >>> # Returns DataFrame with hl, bd, pca columns merged

    Implementation:
        1. Load each directory using load_all_results()
        2. Sequentially merge DataFrames on index
        3. Remove duplicate columns after each merge

    Note:
        Each directory should contain ./all/ subdirectory with
        alpha.all.*.csv files generated by dump_all()
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
    Load all alpha signal CSV files from a directory for date range.

    Reads all alpha.all.*.csv files from {fdir}/all/ directory, filtering
    by date range and valid market hours. Concatenates into single DataFrame.
    Used to reload previously generated alpha signals for analysis or
    backtesting.

    Args:
        fdir (str): Base directory containing /all/ subdirectory
        start (str/int): Start date in YYYYMMDD format
        end (str/int): End date in YYYYMMDD format
        cols (list, optional): Column names to load (None = all columns)

    Returns:
        DataFrame: Concatenated alpha signals, indexed by (iclose_ts, sid)

    Example:
        >>> alpha_df = load_all_results('./hl', 20130101, 20130630)
        >>> # Loads all ./hl/all/alpha.all.*.csv files in date range

    File Format:
        Expected filename: alpha.all.YYYYMMDD_HHMM.csv
        Index columns: iclose_ts, sid

    Filters:
        - Date range: start <= date <= end
        - Time range: 10:00 <= time <= 15:30 (market hours)

    Raises:
        pandas.errors.InvalidIndexError: If duplicate (iclose_ts, sid)
                                         entries exist (verify_integrity=True)

    Note:
        Concatenates with verify_integrity=True to ensure no duplicate
        index values across files
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
            df = pd.read_csv(ff, index_col=['iclose_ts', 'sid'], header=0, parse_dates=True, sep=",", usecols=cols)
        else:
            df = pd.read_csv(ff, index_col=['iclose_ts', 'sid'], header=0, parse_dates=True, sep=",")

        fcast_dfs.append(df)

    fcast_df = pd.concat(fcast_dfs, verify_integrity=True)

    return fcast_df

