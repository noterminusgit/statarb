"""HDF5 Date Format Converter

Quick utility script to fix date index format in HL HDF5 files. Converts date
index from object/string dtype to datetime64[ns] dtype for proper time series
operations.

Purpose:
    Legacy HL files may have date indices stored as strings instead of proper
    datetime objects. This causes issues with pandas time series operations
    like truncate(), resample(), and date-based filtering.

Usage:
    # Use default file path (backward compatible):
    python change_hl.py

    # Specify custom file path:
    python change_hl.py --file ./custom/data.20050101-20050630.h5

Arguments:
    --file: Path to HDF5 file to process (default: ./all/all.20040101-20040630.h5)

Process:
    1. Load full_df from HDF5 file
    2. Reset index to columns
    3. Convert 'date' column to datetime using pd.to_datetime()
    4. Set multi-index back to (date, gvkey)
    5. Overwrite original file with corrected DataFrame

Output:
    Overwrites the original HDF5 file with corrected date types.
    Prints before and after dtypes for verification.

Note:
    For batch processing of multiple files, see change_raw.py.
"""

import argparse
import h5py
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description='Fix date index format in HL HDF5 files'
    )
    parser.add_argument(
        '--file',
        type=str,
        default='./all/all.20040101-20040630.h5',
        help='Path to HDF5 file to process (default: ./all/all.20040101-20040630.h5)'
    )
    args = parser.parse_args()

    pd.set_option('display.max_columns', 100)

    print("Processing file:", args.file)
    df = pd.read_hdf(args.file, key='full_df')
    print("Before - Date dtype:", df.index.levels[0].dtype)

    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index(['date', 'gvkey'], inplace=True)

    print("After - Date dtype:", df.index.levels[0].dtype)
    df.to_hdf(args.file, 'full_df', mode='w')
    print("File updated successfully")

if __name__ == '__main__':
    main()
