#!/usr/bin/env python
"""
FACTORS - Alpha Factor Visualization Tool

Simple plotting utility for visualizing cumulative factor returns over time.
Loads cached factor data and generates time series plots for individual factors.

This tool is used for quick visual inspection of factor performance and
diagnosing factor behavior over specific time periods.

Functionality
-------------
1. Loads factor data from cache (factor returns by date)
2. Unstacks multi-level columns to get factor-level data
3. Computes cumulative sum of factor returns
4. Plots and saves to PNG file

Usage
-----
Edit the script to configure:
    - start: Start date (e.g., "20110101")
    - end: End date (e.g., "20130101")
    - factor: Factor name to plot (e.g., 'growth', 'momentum', 'value')

Then run:
    python factors.py

Output
------
Creates a PNG file named {factor}.png showing cumulative returns.

Example Configuration
---------------------
    start = dateparser.parse("20110101")
    end = dateparser.parse("20130101")
    factor = 'growth'

This will plot the cumulative returns of the 'growth' factor from 2011-2013
and save to growth.png.

Available Factors
-----------------
Common Barra factors that can be plotted:
    - momentum: Price momentum
    - value: Book-to-price value
    - growth: Earnings growth
    - size: Market capitalization
    - volatility: Return volatility
    - leverage: Financial leverage
    - liquidity: Trading liquidity
    - earnings_yield: Earnings-to-price
    (See BARRA_FACTORS in util.py for complete list)

Data Source
-----------
Reads from factor cache files created by:
    - calc_factors() in calc.py
    - regress.py regression output
    - Stored in HDF5 format

Use Cases
---------
- Visual inspection of factor performance
- Diagnosing factor anomalies or regime changes
- Comparing factor behavior across time periods
- Generating charts for research reports

Notes
-----
- Currently hardcoded for specific dates and factor
- For production factor analysis, use more sophisticated tools
- Consider extending to plot multiple factors on same chart
- Add command-line arguments for flexible factor selection
"""

from util import *
from regress import *
from loaddata import *

start = dateparser.parse("20110101")
end = dateparser.parse("20130101")
factor = 'growth'

plt.figure()
df = load_factor_cache(start, end)
df = df.unstack()
df.columns = df.columns.droplevel(0)
df[factor].cumsum().plot()
plt.savefig(factor + ".png")


