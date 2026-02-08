#!/usr/bin/env python
"""
OSIM_SIMPLE - Portfolio Weight Optimizer

Standalone tool for optimizing forecast combination weights based on historical
returns. This is the offline optimization component extracted from OSIM, designed
to find optimal weights that maximize risk-adjusted returns or minimize variance.

Unlike OSIM (which optimizes weights during simulation), this script performs
pure portfolio optimization on pre-computed forecast returns without simulating
actual trading.

Purpose
-------
Find optimal weights for combining multiple alpha forecasts by:
1. Loading historical returns from multiple forecasts
2. Computing rolling covariance matrices
3. Optimizing weights to maximize Sharpe ratio or minimize variance
4. Evaluating out-of-sample performance
5. Combining recent and long-term optimal weights

Optimization Objectives
-----------------------
Two objective functions implemented:

1. fcn() - Variance Minimization (lines 27-43):
   Minimize portfolio variance while ignoring returns
   Returns: 1 / sqrt(variance)
   Use case: Risk parity, minimum variance portfolios

2. sharpe_fcn() - Sharpe Ratio Maximization (lines 45-61):
   Maximize (annual_return / annual_volatility)
   Returns: Sharpe ratio
   Use case: Risk-adjusted return optimization

Currently uses fcn() (variance minimization) in the main loop.

Methodology
-----------
Rolling Window Optimization:
    1. Split data into training and test periods
    2. For each 30-day test window:
       a. Optimize weights on recent 30 days (wtrecent)
       b. Optimize weights on all data from start (wtall)
       c. Average the two: wts = (wtrecent + wtall) / 2
       d. Evaluate on next 30-day out-of-sample period
    3. Compute mean out-of-sample Sharpe ratio

Weight Combination Strategy:
    - Recent weights capture current regime
    - All-history weights provide stability
    - 50/50 average balances adaptability and robustness

Input Data Format
-----------------
Expects text files matching *.txt in current directory with format:
    fcast name date time notional cumpnl dpnl bps turnover other

Example:
    hl 0 20130101 1545 150000000 125000 8500 0.0057 2500000 850
    bd 0 20130201 1545 148000000 128000 3200 0.0022 1800000 650

Columns used:
    - fcast: Forecast name
    - date: Trading date (YYYYMMDD)
    - bps: Daily return in basis points

Data is indexed by [date, fcast] and unstacked to create a returns matrix.

Optimization Parameters
-----------------------
Number of forecasts: 10 (hardcoded)
Weight bounds: [0.0, 1.0] (long-only)
Training window: Rolling 30-day
Test window: 30-day out-of-sample
Total period: 2011-01-01 to 2013-01-01

OpenOpt NSP (Nonlinear Solver Problems):
    - goal='max': Maximize objective function
    - ftol=0.001: Function tolerance for convergence
    - maxFunEvals=300: Maximum function evaluations
    - solver='ralg': Reduced-gradient algorithm

Initial Weights
---------------
Equal weights: [0.5, 0.5, ..., 0.5] for all 10 forecasts
Alternative (commented): [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] (single forecast)

Covariance Calculation
----------------------
Lines 28-35 compute portfolio variance:
    variance = sum_i weights[i]^2 * var[i]^2
             + sum_i sum_j 2 * weights[i] * weights[j] * cov[i,j]

Note: Line 31 appears incorrect - should not square variance terms again.
The covariance matrix already contains variances on the diagonal.

Output
------
Prints per-period results:
    - Optimal weights for recent window
    - Optimal weights for full history
    - Out-of-sample Sharpe ratio for next 30 days
    - Date of evaluation period

Final output:
    - Mean out-of-sample Sharpe ratio across all periods

Example Output
--------------
    hl: 0.45
    bd: 0.32
    pca: 0.23
    ...
    OS: 2011-02-01 1.23
    ...
    1.15  (mean Sharpe across all periods)

Usage
-----
1. Prepare return data files (*.txt) in current directory
2. Ensure data covers 2011-01-01 to 2013-01-01
3. Run: python osim_simple.py

No command-line arguments - all parameters hardcoded.

Dependencies
------------
- pandas: Data manipulation
- numpy: Numerical operations
- openopt: Optimization solver (older library, may need pip install openopt)

Limitations
-----------
- Hardcoded for 10 forecasts
- Hardcoded date range (2011-2013)
- No command-line configuration
- File format not documented
- Variance calculation may have bug (line 31)
- No handling of missing data
- Equal initial weights may bias optimization

Comparison with OSIM
--------------------
OSIM:
    - Optimizes weights during simulation
    - Objective function includes slippage, constraints
    - Online optimization at each timestamp
    - Full order execution modeling

OSIM_SIMPLE:
    - Offline optimization on returns only
    - Pure portfolio optimization
    - No execution modeling
    - Faster, but less realistic

Use Cases
---------
1. Pre-compute optimal weights for OSIM2 (fixed weights)
2. Research on forecast combination methods
3. Validate OSIM weight optimization
4. Quick experiments with different objectives
5. Generate weight recommendations for manual trading

When to Use
-----------
- Before running OSIM2 to determine good weights
- For offline research on forecast combinations
- When you want clean portfolio optimization without execution details
- For teaching portfolio optimization concepts

Notes
-----
- Name "osim_simple" is misleading - this is not a simplified simulator
- Better name: "weight_optimizer.py" or "forecast_combiner.py"
- The variance calculation on lines 28-35 may have a bug
- Consider adding command-line arguments for flexibility
- Could be extended to support different numbers of forecasts
- OpenOpt is an older library - consider switching to scipy.optimize
"""

#from util import *
#from regress import *
#from loaddata import *

from __future__ import division, print_function

import openopt

from collections import defaultdict
from datetime import timedelta
import argparse
import glob

import pandas as pd
import numpy as np

dflist = list()
for file in glob.glob("*.txt"):
    df = pd.read_csv(file, sep=" ", names=['fcast', "blah", "date", "time", "not", "cumpnl", "dpnl", "bps", "turn", "other"])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'fcast'])
    dflist.append(df)
    df = pd.concat(dflist)

cols = df['bps'].unstack().columns

def fcn(weights, start, end):
    cov_one_df = df[ (df.index.get_level_values('date') > start) & (df.index.get_level_values('date') < end) ]['bps'].unstack().fillna(0).cov()
    pvar = 0
    for ii in range(0,10):
        pvar += weights[ii] * cov_one_df.values[ii,ii] * cov_one_df.values[ii,ii] 

    for ii in range(0,10):
        for jj in range(0,10):
            pvar += 2 * weights[ii] * weights[jj] * cov_one_df.values[ii, jj]

    pret = 1
    ret_df = df[ (df.index.get_level_values('date') > start) & (df.index.get_level_values('date') < end) ]['bps'].unstack().fillna(0).mean()
    for ii in range(0,10):
        pret += weights[ii] * ret_df.values[ii]

#    print "{} {} {}".format((pret * 252) / np.sqrt(pvar * 252), pret * 252, np.sqrt(pvar * 252))
    return 1 / np.sqrt(pvar)

def sharpe_fcn(weights, start, end):
    cov_one_df = df[ (df.index.get_level_values('date') > start) & (df.index.get_level_values('date') < end) ]['bps'].unstack().fillna(0).cov()
    pvar = 0
    for ii in range(0,10):
        pvar += weights[ii] * cov_one_df.values[ii,ii] * cov_one_df.values[ii,ii] 

    for ii in range(0,10):
        for jj in range(0,10):
            pvar += 2 * weights[ii] * weights[jj] * cov_one_df.values[ii, jj]

    pret = 0
    ret_df = df[ (df.index.get_level_values('date') > start) & (df.index.get_level_values('date') < end) ]['bps'].unstack().fillna(0).mean()
    for ii in range(0,10):
        pret += weights[ii] * ret_df.values[ii]

    print("{} {} {}".format((pret * 252) / np.sqrt(pvar * 252), pret * 252, np.sqrt(pvar * 252)))
    return (pret * 252) / np.sqrt(pvar * 252)

mean = 0 
cnt = 0
gstart = pd.to_datetime("20110101")
start = pd.to_datetime("20110101")
end = pd.to_datetime("20110101") + timedelta(days=30)
while end < pd.to_datetime("20130101"):
    lb = np.ones(10) * 0.0
    ub = np.ones(10) 
    plotit = False
    initial_weights = np.asarray([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
    #initial_weights = np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    p = openopt.NSP(goal='max', f=fcn, x0=initial_weights, lb=lb, ub=ub)
    p.args.f = (start, end)
    p.ftol = 0.001
    p.maxFunEvals = 300
    r = p.solve('ralg')
    if (r.stopcase == -1 or r.isFeasible == False):
        print(objective_detail(target, *g_params))
        raise Exception("Optimization failed")

    print(r.xf)

    for ii in range(0,10):
        print("{}: {}".format(cols[ii], r.xf[ii]))
        ii += 1

    wtrecent = r.xf

    p = openopt.NSP(goal='max', f=fcn, x0=initial_weights, lb=lb, ub=ub)
    p.args.f = (gstart, end)
    p.ftol = 0.001
    p.maxFunEvals = 300
    r = p.solve('ralg')
    if (r.stopcase == -1 or r.isFeasible == False):
        print(objective_detail(target, *g_params))
        raise Exception("Optimization failed")

    print(r.xf)

    for ii in range(0,10):
        print("{}: {}".format(cols[ii], r.xf[ii]))
        ii += 1

    wtall = r.xf
    
    #fcn(initial_weights, start='20110701', end='20120101')
    wts = np.ones(10) * 0.0
    for ii in range(0, 10):
        wts[ii] = (wtall[ii] + wtrecent[ii]) / 2

    start = end
    end = end + timedelta(days=30)
    sharpe = sharpe_fcn(wts, start, end)
    print("OS: {} {}".format(end, sharpe))
    mean += sharpe
    cnt += 1

print(mean/cnt)

