"""Raw Data Augmentation Utility

Batch processor to add missing columns to raw CSV data files. Queries SQL
databases to fetch market cap and SEDOL identifiers, then updates existing
raw data folders with the new columns.

Purpose:
    Legacy raw data folders may be missing fields required for newer strategies:
    - mkt_cap: Market capitalization for position sizing
    - sedol: SEDOL identifiers for borrow data linkage

    This script queries Capital IQ databases to backfill these fields across
    all existing raw data directories.

Usage:
    python change_raw.py --dir=./data/raw

    Processes all subdirectories matching pattern ./data/raw/YYYYMMDD/

Arguments:
    --dir: Directory containing raw/ subdirectories (default: '.')

Data Requirements:
    Input:  <dir>/raw/<YYYYMMDD>/{uni_df.csv, price_df.csv}
    Output: Updates the same CSV files in place with new columns

SQL Requirements:
    - Access to dbDevCapIQ SQL Server with xpressfeed database
    - Windows authentication (Trusted_Connection=yes)
    - ODBC Driver 17 for SQL Server

Workflow:
    1. Scans for all raw/<YYYYMMDD>/ directories
    2. For each directory:
       a. Loads uni_df.csv and price_df.csv
       b. Checks for missing 'sedol' column in uni_df
       c. Checks for missing 'mkt_cap' column in price_df
       d. Queries SQL for missing data
       e. Merges and overwrites CSV files

Date Logic:
    For directory ending YYYYMMDD:
    - If MMDD = 0101: start = previous year 0630
    - If MMDD = 0630: start = same year 0101
    Queries data between start and YYYYMMDD.

Example:
    python change_raw.py --dir=./salamander_data/data

    Processes:
    ./salamander_data/data/raw/20130101/
    ./salamander_data/data/raw/20130630/
    ./salamander_data/data/raw/20140101/
    ...

    For 20130101:
        start = 20120630
        end = 20130101 (- 1 trading day)
        Adds sedol to uni_df.csv
        Adds mkt_cap to price_df.csv
"""

import pandas as pd
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import os
import glob
import re
import mysql.connector
import pyodbc
import argparse
from mktcalendar import *


def add_mktcap(uni_df, price_df, start, end, out_dir):
    """Query and add market cap data to price_df.

    Fetches daily market capitalization from Capital IQ and merges it into
    the price DataFrame. Market cap is used for position sizing and universe
    filtering in the backtesting system.

    Args:
        uni_df: Universe DataFrame indexed by gvkey with 'tid' column
        price_df: Price DataFrame indexed by (date, gvkey) with 'tid' column
        start: Start date for query (datetime object)
        end: End date for query (datetime object)
        out_dir: Output directory for saving updated price_df.csv

    Process:
        1. Query Capital IQ for daily market cap by gvkey and tid
        2. Merge with universe to filter to tradable securities
        3. Join with price_df on (date, gvkey, tid)
        4. Overwrite price_df.csv with new mkt_cap column

    SQL Source:
        ciqMarketCap table joined through tradingItemId and gvkey mappings

    Note:
        Currently commented out in main(). Uncomment to enable market cap updates.
    """
    date = end - TDay
    sql = ("SELECT DISTINCT g.gvkey, t.tradingItemId 'tid', m.pricingDate 'date',"
           " m.marketCap 'mkt_cap'"
           " FROM ciqTradingItem t"
           " INNER JOIN ciqGvKeyIID g ON g.objectId = t.tradingItemId"
           " INNER JOIN ciqSecurity s ON t.securityId = s.securityId"
           " INNER JOIN ciqMarketCap m ON s.companyId = m.companyId"
           " WHERE m.pricingDate BETWEEN '%s' AND '%s'"
           " AND g.gvkey IN %s"
           " AND t.tradingItemId In %s"
           % (start, date, tuple(uni_df.index.values), tuple(uni_df['tid'].values)))
    cnxn_s = 'Trusted_Connection=yes;Driver={ODBC Driver 17 for SQL Server};Server=dbDevCapIq;Database=xpressfeed'
    cnxn = pyodbc.connect(cnxn_s)
    add_df = pd.read_sql(sql, cnxn)
    cnxn.close()
    add_df = pd.merge(uni_df[['tid']], add_df, on=['gvkey', 'tid'])
    price_df = pd.merge(price_df, add_df, on=['date', 'gvkey', 'tid'])
    price_df.set_index(['date', 'gvkey'], inplace=True)

    end_s = end.strftime("%Y%m%d")
    dir = '%s/%s/' % (out_dir, end_s)
    print("price_df added a new column:")
    print(price_df[['mkt_cap']].head())
    price_df.to_csv("%sprice_df.csv" % dir, "|")

def add_sedol(uni_df, start, end, out_dir):
    """Query and add SEDOL identifiers to uni_df.

    Fetches SEDOL (Stock Exchange Daily Official List) codes for each gvkey
    from Capital IQ security identifier history. SEDOLs are needed to link
    universe data with borrow rate data from stock lending sources.

    Args:
        uni_df: Universe DataFrame indexed by gvkey
        start: Start date (unused, kept for signature compatibility)
        end: End date for SEDOL validity check (datetime object)
        out_dir: Output directory for saving updated uni_df.csv

    Process:
        1. Query sec_idhist for SEDOL codes valid at end date
        2. Filter to iid='01' (primary issue)
        3. Merge with uni_df on gvkey
        4. Overwrite uni_df.csv with new sedol column

    SQL Source:
        sec_idhist table with effective date range filtering

    Date Logic:
        Uses (end - 1 trading day) to ensure SEDOL is valid at period end.
        Checks that efffrom < date AND effthru >= date.

    Usage:
        Required for strategies using borrow rate data (show_borrow.py,
        get_borrow.py). SEDOL is the join key between universe and locates data.
    """
    date = end - TDay
    sql = ("SELECT DISTINCT gvkey, itemvalue 'sedol'"
           " FROM sec_idhist"
           " WHERE efffrom < '%s'"
           " AND effthru >= '%s'"
           " AND iid = '01'"
           " AND item = 'SEDOL'"
           " AND gvkey IN %s"
           % (date, date, tuple(uni_df.index.values)))
    cnxn_s = 'Trusted_Connection=yes;Driver={ODBC Driver 17 for SQL Server};Server=dbDevCapIq;Database=xpressfeed'
    cnxn = pyodbc.connect(cnxn_s)
    add_df = pd.read_sql(sql, cnxn)
    cnxn.close()
    uni_df = pd.merge(uni_df, add_df, on=['gvkey'])
    uni_df.set_index('gvkey', inplace=True)

    end_s = end.strftime("%Y%m%d")
    dir = '%s/%s/' % (out_dir, end_s)
    print("uni_df added a new column: ")
    print(uni_df[['sedol']].head())
    uni_df.to_csv("%suni_df.csv" % dir, "|")

def main(start_s, end_s, data_dir):
    """Process a single raw data directory and add missing columns.

    Loads uni_df and price_df from a specific YYYYMMDD directory, checks for
    missing columns, and queries SQL to backfill them.

    Args:
        start_s: Start date string (YYYYMMDD) for query window
        end_s: End date string (YYYYMMDD) identifying the directory
        data_dir: Path to raw/ directory containing YYYYMMDD subdirectories

    Process:
        1. Load uni_df.csv and optionally price_df.csv
        2. Check if 'sedol' column exists in uni_df
        3. Check if 'mkt_cap' column exists in price_df (commented out)
        4. Query SQL and merge missing columns
        5. Overwrite CSV files with augmented data

    Currently Active:
        - SEDOL addition to uni_df

    Currently Disabled (commented out):
        - Market cap addition to price_df
        - Uncomment lines 73-74 and 76-77 to enable

    Called by:
        __main__ loop for each discovered raw/<YYYYMMDD>/ directory
    """
    start = datetime.strptime(start_s, "%Y%m%d")
    end = datetime.strptime(end_s, "%Y%m%d")
    pd.set_option('display.max_columns', 100)
    uni_df = pd.read_csv("%s/%s/uni_df.csv" % (data_dir, end_s), header=0, sep='|', dtype={'gvkey': str},
                         parse_dates=[0])
    #price_df = pd.read_csv("%s/%s/price_df.csv" % (data_dir, end_s), header=0, sep='|', dtype={'gvkey': str}, parse_dates=[0])
    uni_df.set_index('gvkey', inplace=True)
    #price_df.set_index(['date', 'gvkey'], inplace=True)
    #if 'mkt_cap' not in price_df.columns:
        #add_mktcap(uni_df, price_df, start, end, data_dir)
    if 'sedol' not in uni_df.columns:
        add_sedol(uni_df, start, end, data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="the directory where raw data folder is stored", type=str, default='.')
    args = parser.parse_args()
    for fd in sorted(glob.glob(args.dir + '/raw/*')):
        m = re.match(r".*\d{8}", str(fd))
        end_s = fd[-8:]
        print("Loading raw data folder %s" % end_s)
        if end_s[-4:] == '0101':
            start_s = str(int(end_s[:4]) - 1) + '0630'
        else:
            start_s = end_s[:4] + '0101'
        main(start_s, end_s, args.dir + '/raw')
