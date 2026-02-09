#!/usr/bin/env python
"""
READCSV - CSV Debugging Utility

Simple command-line utility for parsing and displaying CSV data in a readable
key-value format. Reads CSV from stdin and prints each row as key-value pairs.

This is a debugging tool for quickly inspecting CSV files without loading them
into pandas or Excel.

Usage
-----
    cat data.csv | python readcsv.py
    python readcsv.py < data.csv

Output Format
-------------
For each row after the header, prints:
    column1_name : value1
    column2_name : value2
    ...

Example
-------
Input CSV:
    ticker,price,volume
    AAPL,150.25,1000000
    MSFT,300.50,800000

Output:
    ticker : AAPL
    price : 150.25
    volume : 1000000
    ticker : MSFT
    price : 300.50
    volume : 800000

Limitations
-----------
- Only handles comma-separated values
- No support for quoted fields with commas
- First row must be header
- Does not handle complex CSV edge cases

Notes
-----
This is a minimal utility script for quick data inspection during development.
For production CSV parsing, use pandas.read_csv() or the csv module.
"""

from __future__ import division, print_function

import sys

cnt = 0
keys = dict()
for line in sys.stdin:
    if cnt == 0:
        for item in line.split(","):
            keys[cnt] = item
            cnt += 1
    else:
        ii = 0
        for item in line.split(","):
            print("{} : {}".format(keys[ii], item))
            ii += 1
