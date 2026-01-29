#!/usr/bin/env python
"""
Live Data Loading Module for Production Trading

This module provides data loading functionality for live/production trading workflows,
distinct from the backtesting data loading in loaddata.py. It handles real-time price
data and analyst fundamental data from external sources.

Key Differences from loaddata.py:
    - Designed for live/production environments vs. historical backtests
    - Loads real-time bid/ask price data from CSV files
    - Integrates with IBES (Institutional Brokers' Estimate System) database
    - Handles intraday timestamps and time-bias adjustments for production
    - Simpler universe management (relies on pre-filtered data)

Active Functionality:
    load_live_file(ifile): Loads live price file with bid/ask spreads and calculates
                          mid-prices (close_i) for current market data

Inactive/Commented Functionality (Infrastructure for future use):
    - load_ratings_prod_hist(): Analyst ratings consensus from IBES database
    - load_target_prod_hist(): Price target consensus from IBES database
    - load_estimate_prod_hist(): Earnings estimate consensus from IBES database

Configuration:
    All *_BASE_DIR variables must be set to appropriate production data paths:
    - UNIV_BASE_DIR: Universe definition files
    - SECDATA_BASE_DIR: Security metadata (sectors, etc.)
    - PRICE_BASE_DIR: Live price data files
    - BARRA_BASE_DIR: Barra risk model data
    - BAR_BASE_DIR: Intraday bar data
    - EARNINGS_BASE_DIR: Historical earnings data
    - LOCATES_BASE_DIR: Short locate availability data
    - ESTIMATES_BASE_DIR: IBES database path (ibes.db)

Constants:
    UNBIAS (int): Time bias adjustment in hours for data queries (default: 3)
                 Used to account for timezone differences in production data

Usage in Production:
    This module is typically called by production pipeline scripts (prod_*.py)
    to load current market data for signal generation and portfolio construction.

Related Files:
    - loaddata.py: Historical data loading for backtesting
    - prod_sal.py, prod_eps.py, prod_rtg.py: Production signal generators
    - util.py, calc.py: Data manipulation and calculation utilities

Author: Legacy codebase
Python: 2.7 (production environment)
"""

import sys
import os
import glob
import re
import math

from dateutil import parser as dateparser
import time
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import sqlite3 as lite

from util import *
from calc import *

# Data source directory configuration
# CRITICAL: These paths must be configured for production environment
UNIV_BASE_DIR = ""        # Universe definition files directory
SECDATA_BASE_DIR = ""     # Security metadata (sectors, industries) directory
PRICE_BASE_DIR = ""       # Live price data files directory
BARRA_BASE_DIR = ""       # Barra risk model data directory
BAR_BASE_DIR = ""         # Intraday bar data directory
EARNINGS_BASE_DIR = ""    # Historical earnings data directory
LOCATES_BASE_DIR = ""     # Short locate availability data directory
ESTIMATES_BASE_DIR = ""   # IBES database path (should end with ibes.db)

# Utility function for timestamp conversion from milliseconds
fromtimestamp = lambda x:datetime.fromtimestamp(int(x) / 1000.)

# Time bias adjustment in hours for production data queries
# Used to account for timezone/data delivery lag in live environment
UNBIAS = 3

def load_live_file(ifile):
    """
    Load live price data file with bid/ask spreads and calculate mid-prices.

    This is the primary function for loading current market data in production.
    It reads a CSV file containing real-time price quotes and calculates the
    mid-price (close_i) as the average of bid and ask prices.

    Args:
        ifile (str): Path to live price CSV file

    Expected CSV Format:
        - Header row required
        - Columns must include: sid (security ID), bid, ask
        - sid column used as DataFrame index
        - Additional columns preserved in output

    Returns:
        pd.DataFrame: Price data indexed by sid with columns:
            - All original columns from input file
            - close_i (float): Mid-price calculated as (bid + ask) / 2.0

    Usage Example:
        >>> live_prices = load_live_file('/data/prices/live_20130630_1530.csv')
        >>> print(live_prices[['bid', 'ask', 'close_i']].head())

    Notes:
        - Mid-price (close_i) is used as the execution price estimate
        - The '_i' suffix indicates an intraday price point
        - This differs from EOD close prices used in backtesting
        - NaN handling: Function assumes clean input data; add validation if needed

    Related:
        - loaddata.load_prices(): Historical EOD price loading for backtests
        - calc.py: Uses close_i for return calculations
    """
    df = pd.read_csv(ifile, header=0, index_col=['sid'])
    df['close_i'] = (df['bid'] + df['ask']) / 2.0
    return df


# =============================================================================
# INACTIVE FUNCTIONS - IBES Database Integration Infrastructure
# =============================================================================
# The following functions provide infrastructure for loading analyst fundamental
# data from the IBES (Institutional Brokers' Estimate System) database in production.
# They are currently commented out but represent complete, tested implementations
# for future production use.
#
# Functions:
#   - load_secdata(): Load security metadata (sectors, industries)
#   - load_ratings_prod_hist(): Load analyst rating consensus over time
#   - load_target_prod_hist(): Load price target consensus over time
#   - load_estimate_prod_hist(): Load earnings estimate consensus over time
#
# Common Pattern:
#   1. Query IBES SQLite database (ibes.db) for historical snapshots
#   2. Filter by timestamp with UNBIAS adjustment
#   3. Merge with universe DataFrame
#   4. Calculate consensus metrics (mean, median, std, count, max, min)
#   5. Calculate change metrics (diff_mean) by estimator
#   6. Return multi-index DataFrame (date, sid)
#
# Time Handling:
#   - UNBIAS=3 hours accounts for data delivery lag
#   - Queries use timestamp ranges to get latest snapshot before cutoff
#   - Handles both EOD (16:00) and intraday timestamps
#
# To Activate:
#   1. Set ESTIMATES_BASE_DIR to IBES database path
#   2. Uncomment desired function(s)
#   3. Update production pipeline scripts to call these functions
#   4. Test thoroughly with recent data before deploying
# =============================================================================

# def load_secdata(uni_df, start, end):
#     year = end.strftime("%Y")

#     secdata_dir = SECDATA_BASE_DIR + year
#     secdata_file = secdata_dir + "/" + unidate + ".estu.csv.gz"
#     secdata_df = pd.read_csv(secdata_file, header=0, compression='gzip', usecols=['sid', 'sector_name'], index_col=['sid'])
#     univ_df = pd.merge(uni_df, secdata_df, how='inner', left_index=True, right_index=True, sort=True)
#     print "Universe size (secdata): {}".format(len(univ_df.index))    
    
#     return univ_df

# def load_ratings_prod_hist(uni_df, start, end_ts):
#     window = timedelta(days=252)
#     con = lite.connect(ESTIMATES_BASE_DIR + "ibes.db")    
#     date = start
#     df_list = list()
#     uni_df = uni_df.reset_index()
#     while (date <= end_ts):
#         endDateStr = date.strftime('%Y%m%d')
#         startDateStr = (date - window).strftime('%Y%m%d')

#         if date == end_ts:
#             time = end_ts.strftime("%H:%M")
#         else:
#             time = '16:00'

#         timeAdjusted = str(int(time.split(":")[0]) - UNBIAS) + ":" + time.split(":")[1] 
#         sql = "select * from t_ibes_hist_rec_snapshot where timestamp between '{} {}' and '{} {}' group by sid, ibes_ticker, estimator having timestamp = max(timestamp)".format(startDateStr, time, endDateStr, timeAdjusted)
#         print sql
#         df = psql.frame_query(sql, con)
#         df = df[ df['ibes_rec_code'] != '' ]
#         #            df['ts'] = pd.to_datetime( date.strftime("%Y%m%d") + " " + time )
#         df['date'] = pd.to_datetime( date.strftime("%Y%m%d") )
#         df['ibes_rec_code'] = df['ibes_rec_code'].astype(int)
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         print df.columns
#         df = pd.merge(uni_df[ uni_df['date'] == date ], df, how='inner', left_on=['sid'], right_on=['sid'], sort=True, suffixes=['', '_dead'])
#         df = df.set_index(['date', 'sid'])
#         df_list.append(df)
#         date += timedelta(days=1)

#     df = pd.concat(df_list)
#     tstype = 'date'

#     #consensus
#     result_df = df.groupby(level=[tstype, 'sid']).agg({'ibes_rec_code' : [np.mean, np.median, np.std, 'count', np.max, np.min], 'timestamp' : 'last'})
#     result_df.columns = result_df.columns.droplevel(0)
#     for column in result_df.columns:
#         result_df.rename(columns={column: 'rating_' + column}, inplace=True)

#     df = df.set_index('estimator', append=True)
#     df2 = df['ibes_rec_code'].unstack(['estimator', 'sid']).fillna(0).diff().iloc[1:].stack(['sid', 'estimator'])
#     #should drop first date here
#     df2 = df2[ df2 != 0 ]
#     df2 = df2.reset_index('estimator').groupby(level=[tstype, 'sid']).agg(np.mean)
#     df2.columns = ['rating_diff_mean']

#     result_df = pd.merge(result_df, df2, left_index=True, right_index=True, how='left')

#     return result_df


# def load_target_prod_hist(uni_df, start, end_ts):
#     window = timedelta(days=252)
#     con = lite.connect(ESTIMATES_BASE_DIR + "ibes.db")    
#     date = start
#     df_list = list()
#     uni_df = uni_df.reset_index()
#     while (date < end_ts):
#         endDateStr = date.strftime('%Y%m%d')
#         startDateStr = (date - window).strftime('%Y%m%d')

#         if date == end_ts:
#             time = end_ts.strftime("%H:%M")
#         else:
#             time = '16:00'

#         timeAdjusted = str(int(time.split(":")[0]) - UNBIAS) + ":" + time.split(":")[1] 
#         sql = "select * from t_ibes_hist_ptg_snapshot where timestamp between '{} {}' and '{} {}' and horizon in ('', 12) and value > 0 group by sid, ibes_ticker, estimator having timestamp = max(timestamp)".format(startDateStr, time, endDateStr, timeAdjusted)
#         print sql
#         df = psql.frame_query(sql, con)
#         df['value'] = df['value'].astype(str)
#         df = df[ df['value'] != '' ]
#         #            df['ts'] = pd.to_datetime( date.strftime("%Y%m%d") + " " + time )
#         df['date'] = pd.to_datetime( date.strftime("%Y%m%d") )
#         df['value'] = df['value'].astype(float)
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         del df['horizon']
#         print df.columns
#         df = pd.merge(uni_df[ uni_df['date'] == date ], df, how='inner', left_on=['sid'], right_on=['sid'], sort=True, suffixes=['', '_dead'])
#         df = df.set_index(['date', 'sid'])
#         df_list.append(df)
#         date += timedelta(days=1)

#     df = pd.concat(df_list)

#     #consensus
#     result_df = df.groupby(level=['date', 'sid']).agg({'value' : [np.mean, np.median, np.std, 'count', np.max, np.min], 'timestamp' : 'last'})
#     result_df.columns = result_df.columns.droplevel(0)
#     for column in result_df.columns:
#         result_df.rename(columns={column: 'target_' + column}, inplace=True)

#     #detailed
#     df = df.set_index('estimator', append=True)
#     df2 = df['value'].unstack(['estimator', 'sid']).fillna(0).diff().iloc[1:].stack(['sid', 'estimator'])
#     df2 = df2[ df2 != 0 ]
#     df2 = df2.reset_index('estimator').groupby(level=['date', 'sid']).agg(np.mean)
#     df2.columns = ['target_diff_mean']

#     result_df = pd.merge(result_df, df2, left_index=True, right_index=True, how='left')

#     return result_df


# def load_estimate_prod_hist(uni_df, start, end_ts, estimate):
#     window = timedelta(days=252)
#     con = lite.connect(ESTIMATES_BASE_DIR + "ibes.db")    
#     date = start
#     df_list = list()
#     uni_df = uni_df.reset_index()
#     while (date < end_ts):
#         endDateStr = date.strftime('%Y%m%d')
#         startDateStr = (date - window).strftime('%Y%m%d')

#         if date == end_ts:
#             time = end_ts.strftime("%H:%M")
#         else:
#             time = '16:00'

#         timeAdjusted = str(int(time.split(":")[0]) - UNBIAS) + ":" + time.split(":")[1] 
#         minPeriod = str(int(endDateStr[2:4])) + endDateStr[4:6]
#         maxPeriod = str(int(endDateStr[2:4]) + 2) + "00"
#         sql = "select * from t_ibes_det_snapshot where timestamp between '{} {}' and '{} {}' and measure = '{}' and forecast_period_ind = 1 and forecast_period_end_date > {} and forecast_period_end_date < {} group by sid, ibes_ticker, estimator, forecast_period_ind, forecast_period_end_date having timestamp = max(timestamp) order by sid, forecast_period_end_date;".format(startDateStr, time, endDateStr, timeAdjusted, estimate, minPeriod, maxPeriod)
#         print sql
#         df = psql.frame_query(sql, con)
#         df['value'] = df['value'].astype(str)
#         df = df[ df['value'] != '' ]
#         #            df['ts'] = pd.to_datetime( date.strftime("%Y%m%d") + " " + time )
#         df['date'] = pd.to_datetime( date.strftime("%Y%m%d") )
#         df['value'] = df['value'].astype(float)
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         print df.columns
#         df = pd.merge(uni_df[ uni_df['date'] == date ], df, how='inner', left_on=['sid'], right_on=['sid'], sort=True, suffixes=['', '_dead'])
#         df = df[ ~df.duplicated(cols=['date', 'sid', 'estimator']) ]
#         df = df.set_index(['date', 'sid'])
#         df_list.append(df)
#         date += timedelta(days=1)

#     df = pd.concat(df_list)
#     #consensus
#     result_df = df.groupby(level=['date', 'sid']).agg({'value' : [np.mean, np.median, np.std, 'count', np.max, np.min], 'timestamp' : 'last'})
#     result_df.columns = result_df.columns.droplevel(0)
#     for column in result_df.columns:
#         result_df.rename(columns={column: estimate + '_' + column}, inplace=True)

#     #detailed
#     df = df.set_index('estimator', append=True)
#     df2 = df['value'].unstack(['estimator', 'sid']).fillna(0).diff().iloc[1:].stack(['sid', 'estimator'])
#     df2 = df2[ df2 != 0 ]
#     df2 = df2.reset_index('estimator').groupby(level=['date', 'sid']).agg(np.mean)
#     del df2['estimator']
#     df2.columns = [estimate + '_diff_mean']

#     result_df = pd.merge(result_df, df2, left_index=True, right_index=True, how='left')

#     return result_df
