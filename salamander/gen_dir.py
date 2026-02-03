"""
Directory Structure Generator for Salamander Module

This script creates the required directory structure for the salamander backtesting
system. It sets up the data folders needed to run simulations, store raw data,
intermediate results, and outputs.

Usage:
    python gen_dir.py --dir=<root_directory>

Arguments:
    --dir : Root directory path where the folder structure will be created
            Default: '.' (current directory)

Output Structure:
    <root>/
        data/
            all/           - HDF5 files with merged daily data (all.YYYYMMDD-YYYYMMDD.h5)
            all_graphs/    - Regression diagnostic plots from HL signal generation
            hl/            - Alpha signal CSV files (alpha.hl.YYYYMMDD-YYYYMMDD.csv)
            locates/       - Borrow availability data for short positions
            raw/           - Raw input data folders organized by period (price, barra, universe)
            blotter/       - Trade execution records from simulations
            opt/           - Portfolio optimization results and diagnostics

Example:
    # Create directory structure in current folder
    python gen_dir.py

    # Create directory structure in specific location
    python gen_dir.py --dir=/path/to/my/project

Notes:
    - This script must be run before using other salamander generators (gen_hl.py, gen_alpha.py)
    - All directories are created recursively if parent paths don't exist
    - If directories already exist, the script will skip creation (no error)
    - The raw/ directory needs to be manually populated with data files
"""

import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="the root directory", type=str, default='.')
args = parser.parse_args()
root = args.dir
folders = []
data_dir = root + "/data"
folders.append(data_dir)
folders.append(data_dir + "/all")
folders.append(data_dir + "/all_graphs")
folders.append(data_dir + "/hl")
folders.append(data_dir + "/locates")
folders.append(data_dir + "/raw")
folders.append(data_dir + "/blotter")
folders.append(data_dir + "/opt")

for i in range(len(folders)):
    dir = folders[i]
    if not os.path.exists(dir):
        os.makedirs(dir)
print("A directory structure is created under %s" % root)

