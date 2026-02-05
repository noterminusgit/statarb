"""Borrow Data Inspector

Quick diagnostic tool to view borrow rate history for a specific security.
Useful for validating get_borrow.py output and checking short sale costs.

Purpose:
    After running get_borrow.py, use this script to:
    - Verify the consolidated borrow.csv file loaded correctly
    - Check borrow rates for a specific SEDOL over time
    - Diagnose missing or anomalous borrow data

Usage:
    # Use defaults (backward compatible):
    python show_borrow.py

    # Specify custom file path:
    python show_borrow.py --file ./custom/locates/borrow.csv

    # Query different SEDOL:
    python show_borrow.py --sedol 1234567

    # Combine both:
    python show_borrow.py --file ./data/locates/borrow.csv --sedol 9876543

Arguments:
    --file: Path to borrow.csv file (default: ./data/locates/borrow.csv)
    --sedol: SEDOL identifier to query (default: 2484088)

Output:
    Prints all rows for the specified SEDOL:

           sedol        date  shares    fee
        2484088  2013-01-04  1000000  0.50
        2484088  2013-01-11  1100000  0.45
        2484088  2013-01-18   950000  0.55
        ...

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

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description='View borrow rate history for a specific security'
    )
    parser.add_argument(
        '--file',
        type=str,
        default='./data/locates/borrow.csv',
        help='Path to borrow.csv file (default: ./data/locates/borrow.csv)'
    )
    parser.add_argument(
        '--sedol',
        type=str,
        default='2484088',
        help='SEDOL identifier to query (default: 2484088)'
    )
    args = parser.parse_args()

    pd.set_option('display.max_columns', 100)

    print("Loading", args.file)
    result_df = pd.read_csv(
        args.file,
        parse_dates=['date'],
        usecols=['sedol', 'date', 'shares', 'fee'],
        sep='|'
    )

    print("\nBorrow data for SEDOL:", args.sedol)
    filtered_df = result_df[result_df['sedol'] == args.sedol]
    if len(filtered_df) == 0:
        print("No data found for SEDOL:", args.sedol)
    else:
        print(filtered_df)

if __name__ == '__main__':
    main()
