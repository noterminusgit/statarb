"""HDF5 Date Format Converter

Quick utility script to fix date index format in HL HDF5 files. Converts date
index from object/string dtype to datetime64[ns] dtype for proper time series
operations.

Purpose:
    Legacy HL files may have date indices stored as strings instead of proper
    datetime objects. This causes issues with pandas time series operations
    like truncate(), resample(), and date-based filtering.

Usage:
    python change_hl.py

    Hardcoded to process: ./all/all.20040101-20040630.h5

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
    This is a one-off utility script. File path is hardcoded and should be
    updated for different files. For batch processing, see change_raw.py.
"""

import h5py
import pandas as pd

pd.set_option('display.max_columns', 100)
filename1 = './all/all.20040101-20040630.h5'
df = pd.read_hdf(filename1, key='full_df')
print(df.index.levels[0].dtype)
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.set_index(['date', 'gvkey'], inplace=True)
print(df.index.levels[0].dtype)
df.to_hdf('./all/all.20040101-20040630.h5', 'full_df', mode='w')
