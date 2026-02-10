#!/usr/bin/env python
"""
SLIP - Order and Execution Data Merger

Merges QuantBook order and execution data for slippage analysis and fill quality
assessment. Combines order details with execution records to analyze fill prices,
execution quality, and order routing performance.

This script was used for analyzing historical execution data from the QuantBook
trading platform, specifically for the date 2012-02-08.

Functionality
-------------
1. Loads order data from QuantBook ORDER.csv export
2. Loads execution data from QuantBook EXECUTION.csv export
3. Merges orders with executions on order ID
4. Consolidates duplicate symbol and security ID fields
5. Re-indexes by order timestamp and security ID

Data Sources
------------
Order File (ORDER.csv):
    - Order metadata: id, timestamp, symbol, sid
    - Order details: quantity, limit price, order type
    - Routing information

Execution File (EXECUTION.csv):
    - Execution records: order_id, fill price, fill quantity
    - Execution timestamp, venue, fees
    - Partial fill tracking

Output Schema
-------------
Merged DataFrame indexed by ['ts_ord', 'sid'] with columns:
    - symbol: Ticker symbol (consolidated)
    - order_id: Order identifier from QuantBook
    - Fill price and quantity from executions
    - Order details (left join - includes unfilled orders)

Usage
-----
This script is configured for a specific historical analysis:
    python slip.py

The hardcoded paths point to:
    - ../../mus/20120208.ORDER.csv
    - ../../mus/20120208.EXECUTION.csv

Notes
-----
- This appears to be a one-off analysis script
- Contains a typo on line 18: 'merged_Df' should be 'merged_df'
- Currently configured for 2012-02-08 data only
- May be deprecated - check if still used for live execution analysis
- For current slippage calculations, see opt.py slip_nu and slip_beta parameters

Historical Context
------------------
This script was likely used to calibrate the slippage model parameters by
analyzing realized slippage from actual executions. The QuantBook platform
was presumably used for live trading or paper trading during system development.
"""

from __future__ import division, print_function

from loaddata import *
from util import *

ofile = "../../mus/20120208.ORDER.csv"
efile = "../../mus/20120208.EXECUTION.csv"

odf = load_qb_orders(ofile)
edf = load_qb_exec(efile)

merged_df = pd.merge(odf.reset_index(), edf.reset_index(), how='left', left_on=['id'], right_on=['order_id'], suffixes=['_ord', '_exec'])
merged_df['symbol'] = merged_df['symbol_ord']
del merged_df['symbol_ord']
del merged_df['symbol_exec']
del merged_df['index']
del merged_df['id']
assert merged_df['sid_ord'] == merged_df['sid_exec']
merged_df['sid'] = merged_df['sid_ord']
del merged_df['sid_ord']
del merged_df['sid_exec']
merged_df.set_index(['ts_ord', 'sid'], inplace=True)






