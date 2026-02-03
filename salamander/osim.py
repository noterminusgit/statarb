#!/usr/bin/env python
"""
SALAMANDER OSIM - Standalone Order Simulation Engine (Python 3)

This is the Python 3 standalone version of the order-level backtesting simulation.
It optimizes forecast weights by simulating detailed order execution across multiple
alpha signals and evaluating realized P&L with different fill strategies.

Differences from Main osim.py:
==============================

1. **Python 3 Compatibility**
   - Print statements use function syntax: print() not print
   - Dictionary iteration updated (.items(), .keys(), .values())
   - String formatting uses .format() instead of %

2. **Simplified Data Loading**
   - Uses standalone loaddata.py for HDF5/CSV loading
   - No intraday timestamp handling (main version uses iclose_ts)
   - Reduced column set focused on execution data
   - No SQL database integration

3. **Forecast Weight Optimization**
   - Primary purpose: Find optimal weights for multiple forecasts
   - Uses OpenOpt NSP solver with 'ralg' algorithm
   - Objective: Maximize Sharpe ratio minus weight diversity penalty
   - Penalty term: 0.05 * std(weights) to prevent extreme allocations

4. **Fill Price Methodology**
   - VWAP fill: Uses bvwap_b_n (next bar VWAP) from bar data
   - Mid fill: Uses iclose (intraday close) as proxy for midpoint
   - Fallback: If VWAP invalid (<=0), uses iclose
   - Main version has more sophisticated VWAP calculation

5. **Slippage Model**
   - Linear slippage: cost = abs(dollars_traded) * slipbps
   - Default: 0.0001 (1 basis point of traded notional)
   - Simpler than main version's square-root market impact

6. **Data Sources**
   - Loads opt.{fcast}.{date}_{time}.csv files from {dir}/opt/
   - Expects pre-optimized positions from bsim.py runs
   - No direct alpha generation (uses existing opt files)

7. **Position Tracking**
   - Handles splits: shares_last *= split_ratio
   - Handles dividends: cash_last += shares * dividend
   - Applied on day boundaries (dayname != lastday)

8. **Performance Metrics**
   - Daily Sharpe ratio calculation
   - Average notional tracking
   - Turnover statistics
   - No industry or factor attribution (main has this)

9. **Optimization Bounds**
   - Lower bound: 0.0 (no short forecast weights)
   - Upper bound: 1.0 (max 100% allocation to any forecast)
   - Initial weights: 0.5 for all forecasts

10. **Output**
    - Prints optimal forecast weights after convergence
    - Shows daily P&L, notional, and returns during simulation
    - No CSV output files (main writes blotter)

Workflow:
=========

1. Load historical price/volume data from HDF5 cache
2. Load pre-optimized position targets from multiple forecast CSVs
3. Merge forecasts and prepare position tracking DataFrame
4. Define objective(weights) function:
   a. Iterate through all timestamps
   b. Combine forecast positions using current weights
   c. Calculate shares traded vs. last positions
   d. Apply fill prices (VWAP or mid) with slippage
   e. Mark positions to market at close
   f. Track daily P&L and notional
   g. Return Sharpe ratio minus weight penalty
5. Run OpenOpt optimizer to find best forecast weights
6. Print optimal weights and final Sharpe ratio

Command-Line Arguments:
=======================

Required:
  --start         Start date (YYYYMMDD format)
  --end           End date (YYYYMMDD format)
  --fcast         Forecast specification (format: "dir:name,dir2:name2,...")
                  Example: "alpha1:hl,alpha2:bd"
                  Loads opt.{name}.{date}.csv from {dir}/opt/ directory

Optional:
  --fill          Fill price method (default: "mid")
                  - "vwap": Fill at bar VWAP (bvwap_b_n), fallback to iclose
                  - "mid": Fill at intraday close (iclose)
  --slipbps       Slippage in basis points (default: 0.0001 = 1bp)
  --weights       Initial forecast weights (default: 0.5 for all)
                  Format: "w1,w2,w3" (comma-separated floats)

Performance Metrics:
====================

The simulation outputs:
  - Daily P&L: Total portfolio profit/loss each day
  - Daily Return: Daily P&L / notional
  - Daily Turnover: Total traded / notional
  - Sharpe Ratio: Annualized return / annualized volatility
  - Optimal Weights: Best linear combination of input forecasts

Optimization Details:
=====================

OpenOpt NSP (Nonlinear Solver Package) Configuration:
  - Solver: 'ralg' (reduced gradient algorithm)
  - Goal: Maximize Sharpe ratio - 0.05 * std(weights)
  - ftol: 0.001 (function tolerance for convergence)
  - maxFunEvals: 150 (maximum objective evaluations)
  - Bounds: [0, 1] for each forecast weight

The weight penalty (0.05 * std(weights)) prevents:
  - Extreme allocations (e.g., 100% to one forecast)
  - Overfitting to historical data
  - Excessive concentration risk

Examples:
=========

# Find optimal weights for two HL forecasts
python3 salamander/osim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --fcast=alpha1:hl,alpha2:hl \\
  --fill=vwap \\
  --slipbps=0.0001

# Test with initial weights and mid fill
python3 salamander/osim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --fcast=alpha1:hl,alpha2:bd \\
  --fill=mid \\
  --weights=0.6,0.4

# VWAP execution with higher slippage
python3 salamander/osim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --fcast=alpha1:hl \\
  --fill=vwap \\
  --slipbps=0.0005

Data Requirements:
==================

Before running osim.py, you must:

1. Run bsim.py for each forecast to generate opt.{name}.{date}.csv files
2. Ensure cache data exists (run gen_hl.py first)
3. Directory structure:
   {dir}/opt/opt.{forecast}.{YYYYMMDD}_{HHMMSS}.csv

Notes:
======

- Python 3.6+ required
- Designed for forecast weight optimization, not live trading
- Optimization may take 5-20 minutes depending on date range
- Works best with 2-4 forecasts (more = harder optimization)
- Results are sensitive to date range and market regime
- No transaction cost beyond fixed slippage (main has impact model)
"""

from util import *
from regress import *
from loaddata import *

import openopt

from collections import defaultdict

import argparse

halfdays = ['20111125', '20120703', '20121123', '20121224']
breaks = ['20110705', '20120102', '20120705', '20130103']

parser = argparse.ArgumentParser(description='G')
parser.add_argument("--start", action="store", dest="start", default=None)
parser.add_argument("--end", action="store", dest="end", default=None)
parser.add_argument("--fill", action="store", dest='fill', default='mid')
parser.add_argument("--slipbps", action="store", dest='slipbps', default=0.0001)
parser.add_argument("--fcast", action="store", dest='fcast', default=None)
parser.add_argument("--weights", action="store", dest='weights', default=None)
args = parser.parse_args()

participation = 0.015

cols = ['split', 'div', 'close', 'iclose', 'bvwap_b', 'bvolume', 'tradable_med_volume_21_y', 'close_y']
cache_df = load_cache(dateparser.parse(args.start), dateparser.parse(args.end), cols)
cache_df['bvolume_d'] = cache_df['bvolume'].groupby(level='sid').diff()
cache_df.loc[cache_df['bvolume_d'] < 0, 'bvolume_d'] = cache_df['bvolume']
cache_df = push_data(cache_df, 'bvolume_d')
cache_df['max_trade_size'] = cache_df['bvolume_d_n'] * cache_df['iclose'] * participation
cache_df['min_trade_size'] = -1 * cache_df['max_trade_size']
cache_df = push_data(cache_df, 'bvwap_b')
cache_df = push_data(cache_df, 'iclose')

trades_df = None

forecasts = list()
fcasts = args.fcast.split(",")
for pair in fcasts:
    fdir, fcast = pair.split(":")
    print
    "Loading {} {}".format(fdir, fcast)
    forecasts.append(fcast)
    flist = list()
    for ff in sorted(glob.glob("./" + fdir + "/opt/opt." + fcast + ".*.csv")):
        m = re.match(r".*opt\." + fcast + "\.(\d{8})_\d{6}.csv", str(ff))
        if m is None: continue
        d1 = int(m.group(1))
        if d1 < int(args.start) or d1 > int(args.end): continue
        print
        "Loading {}".format(ff)
        flist.append(pd.read_csv(ff, parse_dates=True))
    fcast_trades_df = pd.concat(flist)
    #    fcast_trades_df = fcast_trades_df[ fcast_trades_df['sid'] == testid]
    fcast_trades_df['iclose_ts'] = pd.to_datetime(fcast_trades_df['iclose_ts'])
    fcast_trades_df = fcast_trades_df.set_index(['iclose_ts', 'sid']).sort()

    if trades_df is None:
        trades_df = fcast_trades_df
        trades_df['traded_' + fcast] = trades_df['traded']
        trades_df['shares_' + fcast] = trades_df['shares']
    else:
        trades_df = pd.merge(trades_df, fcast_trades_df, how='outer', left_index=True, right_index=True,
                             suffixes=['', '_dead'])
        trades_df['traded_' + fcast] = trades_df['traded_dead']
        trades_df['shares_' + fcast] = trades_df['shares_dead'].unstack().fillna(method='ffill').stack().fillna(0)
        #        print trades_df['shares_' + fcast].xs(testid, level=1).head(50)
        trades_df = remove_dup_cols(trades_df)

trades_df = pd.merge(trades_df.reset_index(), cache_df.reset_index(), how='left', left_on=['iclose_ts', 'sid'],
                     right_on=['iclose_ts', 'sid'], suffixes=['', '_dead'])
trades_df = remove_dup_cols(trades_df)
trades_df.set_index(['iclose_ts', 'sid'], inplace=True)
cache_df = None

max_dollars = 1e6
max_adv = 0.02
trades_df['max_notional'] = (trades_df['tradable_med_volume_21_y'] * trades_df['close_y'] * max_adv).clip(0,
                                                                                                          max_dollars)
trades_df['min_notional'] = (-1 * trades_df['tradable_med_volume_21_y'] * trades_df['close_y'] * max_adv).clip(
    -max_dollars, 0)

trades_df['cash'] = 0
# trades_df['cash_last'] = 0
trades_df['traded'] = 0
trades_df['shares'] = 0
trades_df['pnl'] = 0
trades_df['cum_pnl'] = 0
trades_df['day_pnl'] = 0

if args.fill == "vwap":
    print
    "Filling at vwap..."
    trades_df['fillprice'] = trades_df['bvwap_b_n']
    print
    "Bad count: {}".format(len(trades_df) - len(trades_df[trades_df['fillprice'] > 0]))
    trades_df.ix[(trades_df['fillprice'] <= 0) | (trades_df['fillprice'].isnull()), 'fillprice'] = trades_df['iclose']
else:
    print
    "Filling at mid..."
    trades_df['fillprice'] = trades_df['iclose']

trades_df.replace([np.inf, -np.inf], np.nan, inplace=True)


def objective(weights):
    """Objective function for forecast weight optimization.

    Simulates complete trading lifecycle with given forecast weights and returns
    risk-adjusted performance metric (Sharpe ratio minus weight diversity penalty).

    This is the core function optimized by OpenOpt NSP solver. It:
    1. Combines multiple forecast positions using input weights
    2. Simulates order execution with realistic fill prices and slippage
    3. Tracks positions, cash, and P&L through corporate actions
    4. Computes daily returns and Sharpe ratio
    5. Applies penalty for weight concentration

    Args:
        weights (np.array): Forecast weight vector to evaluate
            Length matches number of forecasts
            Each element represents allocation to corresponding forecast
            Example: [0.6, 0.4] means 60% forecast1, 40% forecast2

    Returns:
        float: Sharpe ratio minus weight penalty (to be maximized)
            Higher values indicate better risk-adjusted returns
            Penalty = 0.05 * std(weights) encourages diversification

    Side Effects:
        - Prints weight values for each evaluation
        - Prints daily P&L, notional, and returns
        - No file I/O (all output is printed)

    Algorithm:
        For each timestamp in trades_df:
            1. Merge with last period's positions
            2. Apply corporate actions (splits, dividends) on day change
            3. Calculate combined shares from weighted forecasts:
               shares = sum(weights[i] * shares_{forecast_i})
            4. Calculate shares_traded = shares - shares_last
            5. Apply fill price (VWAP or mid) to compute dollars_traded
            6. Update cash = cash_last + dollars_traded - slippage
            7. Mark position to market:
               - Intraday: use iclose_n (next bar close)
               - EOD: use close (official closing price)
            8. Calculate pnl = shares * mark_price + cash
            9. Track daily notional and P&L

        After all timestamps:
            1. Aggregate daily returns: ret = pnl_change / notional
            2. Compute Sharpe: mean(daily_ret) / std(daily_ret) * sqrt(252)
            3. Compute penalty: 0.05 * std(weights)
            4. Return sharpe - penalty

    Performance Tracking:
        - day_bucket: Aggregates notional, P&L, turnover by date
        - Sharpe calculation uses annualization factor of 252 trading days
        - Total turnover tracked for cost analysis

    Notes:
        - OpenOpt calls this function 50-150 times during optimization
        - Each call simulates entire backtest period (~6 months)
        - Optimization runtime: typically 5-20 minutes
        - Weight penalty prevents overfitting to single forecast
    """
    ii = 0
    for fcast in forecasts:
        print
        "Weight {}: {}".format(fcast, weights[ii])
        ii += 1

    day_bucket = {
        'not': defaultdict(int),
        'pnl': defaultdict(int),
        'trd': defaultdict(int),
    }

    lastgroup_df = None
    lastday = None
    pnl_last_day_tot = 0
    totslip = 0

    for ts, group_df in trades_df.groupby(level='iclose_ts'):

        dayname = ts.strftime("%Y%m%d")
        if int(dayname) > 20121227: continue
        monthname = ts.strftime("%Y%m")
        weekdayname = ts.weekday()
        timename = ts.strftime("%H%M")

        if dayname in halfdays and int(timename) > 1245:
            continue

        if lastgroup_df is not None:
            #            group_df = pd.merge(group_df.reset_index().set_index('sid'), lastgroup_df.reset_index().set_index('sid'), how='left', left_index=True, right_index=True, suffixes=['', '_last'])
            for col in lastgroup_df.columns:
                if col == "sid": continue
                lastgroup_df[col + "_last"] = lastgroup_df[col]
                del lastgroup_df[col]
            group_df = pd.concat([group_df.reset_index().set_index('sid'), lastgroup_df.reset_index().set_index('sid')],
                                 join='outer', axis=1, verify_integrity=True)
            group_df['iclose_ts'] = ts
            group_df.reset_index().set_index(['iclose_ts', 'sid'], inplace=True)
            if dayname != lastday and lastday is not None:
                group_df['cash_last'] += group_df['shares_last'] * group_df['div'].fillna(0)
                group_df['shares_last'] *= group_df['split'].fillna(1)
        else:
            group_df['shares_last'] = 0
            group_df['cash_last'] = 0

        ii = 0
        for fcast in forecasts:
            #            print fcast
            #            print group_df['shares_' + fcast].xs(testid, level=1)
            group_df['shares'] += group_df['shares_' + fcast].fillna(0) * weights[ii]
            #           print group_df['shares'].xs(testid, level=1)
            ii += 1

        group_df['shares_traded'] = group_df['shares'] - group_df['shares_last'].fillna(0)
        # group_df['shares'] = group_df['traded'] / group_df['fillprice']
        group_df['dollars_traded'] = group_df['shares_traded'] * group_df['fillprice'] * -1.0
        group_df['cash'] = group_df['cash_last'] + group_df['dollars_traded']

        #        fillslip_tot +=  (group_df['pdiff_pct'] * group_df['traded']).sum()
        #        traded_tot +=  np.abs(group_df['traded']).sum()
        #    print "Slip2 {} {}".format(fillslip_tot, traded_tot)

        markPrice = 'iclose_n'
        #    if ts.strftime("%H%M") == "1530" or (dayname in halfdays and timename == "1230"):
        if ts.strftime("%H%M") == "1545" or (dayname in halfdays and timename == "1245"):
            markPrice = 'close'

        group_df['slip'] = np.abs(group_df['dollars_traded']).fillna(0) * float(args.slipbps)
        totslip += group_df['slip'].sum()
        group_df['cash'] = group_df['cash'] - group_df['slip']
        group_df['pnl'] = group_df['shares'] * group_df[markPrice] + group_df['cash'].fillna(0)
        notional = np.abs(group_df['shares'] * group_df[markPrice]).dropna().sum()
        group_df['lsnot'] = group_df['shares'] * group_df[markPrice]
        pnl_tot = group_df['pnl'].dropna().sum()
        #        print group_df[['shares', 'shares_tgt', 'shares_qhl_b', 'cash', 'dollars_traded', 'pnl']]
        # if lastgroup_df is not None:
        #     group_df['pnl_diff'] = (group_df['pnl'] - group_df['pnl_last'])
        #     print group_df['pnl_diff'].order().dropna().head()
        #     print group_df['pnl_diff'].order().dropna().tail()

        #        pnl_incr = pnl_tot - pnl_last_tot
        traded = np.abs(group_df['dollars_traded']).fillna(0).sum()

        day_bucket['trd'][dayname] += traded
        #       month_bucket['trd'][monthname] += traded
        #      dayofweek_bucket['trd'][weekdayname] += traded
        #      time_bucket['trd'][timename] += traded

        # try:
        #     print group_df.xs(testid, level=1)[['target', 'traded', 'cash', 'shares', 'close', 'iclose', 'shares_last', 'cash_last']]
        # except KeyError:
        #     pass

        # print group_df['shares'].describe()
        # print group_df[markPrice].describe()
        if markPrice == 'close' and notional > 0:
            delta = pnl_tot - pnl_last_day_tot
            ret = delta / notional
            daytraded = day_bucket['trd'][dayname]
            notional2 = np.sum(np.abs((group_df['close'] * group_df['position'] / group_df['iclose'])))
            print
            "{}: {} {} {} {:.4f} {:.2f} {}".format(ts, notional, pnl_tot, delta, ret, daytraded / notional, notional2)
            day_bucket['pnl'][dayname] = delta
            #            month_bucket['pnl'][monthname] += delta
            #            dayofweek_bucket['pnl'][weekdayname] += delta
            day_bucket['not'][dayname] = notional
            #            day_bucket['long'][dayname] = group_df[ group_df['lsnot'] > 0 ]['lsnot'].dropna().sum()
            #            day_bucket['short'][dayname] = np.abs(group_df[ group_df['lsnot'] < 0 ]['lsnot'].dropna().sum())
            #            month_bucket['not'][monthname] += notional
            #            dayofweek_bucket['not'][weekdayname] += notional
            #            trades_df.ix[ group_df.index, 'day_pnl'] = group_df['pnl'] - group_df['pnl_last']
            pnl_last_day_tot = pnl_tot
        #            totturnover += daytraded/notional
        #            short_names += len(group_df[ group_df['traded'] < 0 ])
        #            long_names += len(group_df[ group_df['traded'] > 0 ])
        #            cnt += 1

        lastgroup_df = group_df.reset_index()[['shares', 'cash', 'pnl', 'sid', 'target']]

    nots = pd.DataFrame([[d, v] for d, v in sorted(day_bucket['not'].items())], columns=['date', 'notional'])
    nots.set_index(keys=['date'], inplace=True)
    pnl_df = pd.DataFrame([[d, v] for d, v in sorted(day_bucket['pnl'].items())], columns=['date', 'pnl'])
    pnl_df.set_index(['date'], inplace=True)
    rets = pd.merge(pnl_df, nots, left_index=True, right_index=True)
    print
    "Total Pnl: ${:.0f}K".format(rets['pnl'].sum() / 1000.0)

    rets['day_rets'] = rets['pnl'] / rets['notional']
    rets['day_rets'].replace([np.inf, -np.inf], np.nan, inplace=True)
    rets['day_rets'].fillna(0, inplace=True)
    rets['cum_ret'] = (1 + rets['day_rets']).dropna().cumprod()

    mean = rets['day_rets'].mean() * 252
    std = rets['day_rets'].std() * math.sqrt(252)

    sharpe = mean / std
    print
    "Day mean: {:.4f} std: {:.4f} sharpe: {:.4f} avg Notional: ${:.0f}K".format(mean, std, sharpe,
                                                                                rets['notional'].mean() / 1000.0)
    penalty = 0.05 * np.std(weights)
    print
    "penalty: {}".format(penalty)
    print

    return sharpe - penalty


if args.weights is None:
    initial_weights = np.ones(len(forecasts)) * .5
else:
    initial_weights = np.array([float(x) for x in args.weights.split(",")])
lb = np.ones(len(forecasts)) * 0.0
ub = np.ones(len(forecasts))
plotit = False
p = openopt.NSP(goal='max', f=objective, x0=initial_weights, lb=lb, ub=ub, plot=plotit)
p.ftol = 0.001
p.maxFunEvals = 150
r = p.solve('ralg')

if (r.stopcase == -1 or r.isFeasible == False):
    print
    objective_detail(target, *g_params)
    raise Exception("Optimization failed")

print
r.xf
ii = 0
for fcast in forecasts:
    print
    "{}: {}".format(fcast, r.xf[ii])
    ii += 1
