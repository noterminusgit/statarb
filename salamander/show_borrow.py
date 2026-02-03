"""Borrow Data Inspector

Quick diagnostic tool to view borrow rate history for a specific security.
Useful for validating get_borrow.py output and checking short sale costs.

Purpose:
    After running get_borrow.py, use this script to:
    - Verify the consolidated borrow.csv file loaded correctly
    - Check borrow rates for a specific SEDOL over time
    - Diagnose missing or anomalous borrow data

Usage:
    python show_borrow.py

    Hardcoded to:
    - Load from: ./data/locates/borrow.csv
    - Display SEDOL: 2484088 (example security)

Output:
    Prints all rows for the specified SEDOL:

           sedol        date  shares    fee
        2484088  2013-01-04  1000000  0.50
        2484088  2013-01-11  1100000  0.45
        2484088  2013-01-18   950000  0.55
        ...

Customization:
    Edit line 8 to change the SEDOL:
        print(result_df[result_df['sedol']=='YOUR_SEDOL'])

    Edit main("./data") to change the data directory.

Data Format:
    Expects pipe-delimited CSV with columns: sedol, date, shares, fee

Use Case:
    Validate that borrow costs are reasonable before running backtests with
    short positions. High fees (>10%) indicate hard-to-borrow stocks that
    may be unprofitable to short.

Related:
    get_borrow.py - Creates the borrow.csv file
    change_raw.py - Adds SEDOL to uni_df for joining with borrow data
"""

import pandas as pd

def main(locates_dir):
    pd.set_option('display.max_columns', 100)
    ff = locates_dir + "/locates/borrow.csv"
    print("Loading", ff)
    result_df = pd.read_csv(ff, parse_dates=['date'], usecols=['sedol', 'date', 'shares', 'fee'], sep='|')
    print(result_df[result_df['sedol']=='2484088'])

main("./data")
