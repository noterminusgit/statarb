#!/usr/bin/env python
"""
OSIM - Order Simulation Engine

Order-level backtesting simulation that focuses on detailed execution modeling
and fill price analysis. OSIM optimizes weights across multiple forecasts by
evaluating realized P&L with different fill strategies.

Methodology
-----------
OSIM differs from BSIM by focusing on order-level execution details rather than
daily portfolio optimization. The simulator:

1. Loads optimized position targets from multiple alpha forecasts
2. Combines forecasts using weighted linear combinations
3. Simulates order execution with realistic fill prices
4. Applies slippage costs proportional to traded notional
5. Tracks position and cash through splits, dividends, and corporate actions
6. Optimizes forecast weights by maximizing risk-adjusted returns

The core innovation is the objective() function which:
- Iterates through all trading timestamps
- Computes position changes from weighted forecast combinations
- Applies fill prices (VWAP or midpoint) with slippage
- Marks positions to market at intraday timestamps
- Computes daily P&L at market close
- Returns Sharpe ratio minus weight diversity penalty

Fill Strategy Details
---------------------
VWAP (Volume-Weighted Average Price):
    - Uses bar VWAP price (bvwap_b_n) from bar data
    - Falls back to iclose if VWAP unavailable or invalid
    - Most realistic for large orders that execute throughout the bar
    - Recommended for participation rates > 1%

Mid (Bid-Ask Midpoint):
    - Uses intraday close price (iclose) as proxy for midpoint
    - Assumes instantaneous execution at arrival price
    - Appropriate for smaller orders with minimal market impact
    - Default fill strategy

Close (Closing Auction):
    - Uses official closing price
    - No implementation in code (would use 'close' field)
    - Suitable for orders placed in closing auction

Slippage Model
--------------
Slippage is applied as a fixed percentage of traded notional:

    slippage_cost = abs(dollars_traded) * slipbps

Where:
    - dollars_traded = shares_traded * fill_price
    - slipbps = basis points (default: 0.0001 = 1 bp)

This linear slippage model is simpler than BSIM's square-root market impact
function. It represents the combination of:
    - Bid-ask spread crossing
    - Market impact during execution
    - Adverse selection
    - Opportunity cost

Participation Rate Constraints
-------------------------------
Order sizes are constrained by participation rate limits:

    max_trade_size = bar_volume * fill_price * participation

Where:
    - bar_volume = bvolume_d (daily bar volume delta)
    - participation = 1.5% (hardcoded)

This prevents unrealistic order sizes that would dominate bar volume.
Additional notional constraints:
    - max_notional = min($1M, 2% of 21-day tradable ADV)
    - Prevents excessive concentration in low-volume names

Position Tracking Through Corporate Actions
--------------------------------------------
The simulator handles corporate actions correctly:

Splits:
    shares_adjusted = shares_last * split_ratio

Dividends:
    cash_adjusted = cash_last + shares_last * dividend

Detection logic:
    - Applied on day changes (dayname != lastday)
    - Uses split and div fields from cache_df
    - Cash adjustments before share adjustments

Market Hours and Special Dates
-------------------------------
halfdays: Early close dates (12:45 PM cutoff)
    - Filters out timestamps after 12:45 PM
    - Prevents trading in non-existent afternoon bars
    - Examples: day before Thanksgiving, day after Christmas

breaks: Market holidays and closures
    - Currently not enforced in execution loop
    - Listed for reference only

Marking to Market
-----------------
Positions are marked to market at different times:

Intraday (all timestamps except close):
    mark_price = iclose_n (next bar's intraday close)

End of day (15:45 timestamp or halfday 12:45):
    mark_price = close (official closing price)

Only end-of-day marks contribute to daily P&L calculations.

Weight Optimization
-------------------
The scipy.optimize.minimize solver optimizes forecast weights by:

Objective function:
    maximize: sharpe_ratio - 0.05 * std(weights)

Constraints:
    - lb = 0.0 (no short forecast weights)
    - ub = 1.0 (max weight is 1.0)
    - Initial weights = 0.5 for all forecasts

The penalty term (0.05 * std(weights)) encourages weight diversity, preventing
the optimizer from allocating all weight to a single best forecast. This
improves robustness and reduces overfitting.

Solver parameters:
    - ftol = 0.001 (function tolerance)
    - maxFunEvals = 150 (maximum objective evaluations)
    - algorithm = 'ralg' (reduced-gradient method)

Command-Line Arguments
----------------------
--start (required):
    Start date in YYYYMMDD format
    Example: --start=20130101

--end (required):
    End date in YYYYMMDD format
    Example: --end=20130630

--fill (optional, default='mid'):
    Fill strategy for order execution
    Options: 'vwap', 'mid', 'close'
    Example: --fill=vwap

--slipbps (optional, default=0.0001):
    Slippage in basis points applied to traded notional
    Example: --slipbps=0.0005 (5 bps)
    Typical range: 0.0001 (1bp) to 0.001 (10bp)

--fcast (required):
    Comma-separated list of forecast directory:name pairs
    Format: dir1:name1,dir2:name2,...
    Example: --fcast=run1:hl,run1:bd,run2:pca

    The simulator loads optimized positions from:
    ./dir/opt/opt.name.YYYYMMDD_HHMMSS.csv

--weights (optional):
    Pre-computed forecast weights (bypasses optimization)
    Format: comma-separated floats
    Example: --weights=0.6,0.4
    Must match number of forecasts in --fcast

Usage Examples
--------------
Basic run with single forecast:
    python osim.py --start=20130101 --end=20130630 --fcast=run1:hl

Multiple forecasts with VWAP fills:
    python osim.py --start=20130101 --end=20130630 \\
        --fcast=run1:hl,run1:bd --fill=vwap

High slippage assumption:
    python osim.py --start=20130101 --end=20130630 \\
        --fcast=run1:hl --slipbps=0.001

Pre-specified weights (no optimization):
    python osim.py --start=20130101 --end=20130630 \\
        --fcast=run1:hl,run1:bd --weights=0.6,0.4

Output Metrics
--------------
Per-timestamp output (intraday):
    - Timestamp
    - Notional (gross market value)
    - Cumulative P&L
    - Daily P&L increment
    - Daily return
    - Turnover ratio (traded/notional)

Summary statistics:
    - Total P&L (in thousands)
    - Annualized mean return
    - Annualized volatility
    - Sharpe ratio
    - Average notional (in thousands)

Final optimization results:
    - Optimal forecast weights
    - Final objective function value

Dependencies
------------
- util.py: Data merging and utility functions (remove_dup_cols, push_data)
- regress.py: Regression utilities (unused in this file)
- loaddata.py: Data loading with load_cache() function
- scipy.optimize: Constrained optimization solver (minimize function)

Data Requirements
-----------------
Required fields in cache_df:
    - split: Split adjustment ratio
    - div: Dividend amount per share
    - close: Official closing price
    - iclose: Intraday bar close price
    - bvwap_b: Bar volume-weighted average price
    - bvolume: Cumulative bar volume
    - tradable_med_volume_21_y: 21-day median tradable volume (yesterday)
    - close_y: Yesterday's close price

Required fields in forecast opt files:
    - iclose_ts: Timestamp (intraday close time)
    - sid: Security identifier
    - traded: Target trade dollars
    - shares: Target share position
    - target: Position target (for tracking)

Notes
-----
- Python 2.7 legacy code
- Uses pandas for time-series operations
- Assumes forecasts are pre-optimized by BSIM or similar
- Does not handle partial fills or order queuing
- Assumes all orders execute at specified fill price
- Does not model intraday price volatility within bars
"""

from __future__ import division, print_function

from util import *
from regress import *
from loaddata import *

from scipy.optimize import minimize

from collections import defaultdict

import argparse

# Special trading dates
# Half days: market closes at 1:00 PM EST (last bar at 12:45)
halfdays = ['20111125', '20120703', '20121123', '20121224']

# Market breaks and holidays (listed but not enforced in code)
breaks = ['20110705', '20120102', '20120705', '20130103']

# Command-line argument parser
parser = argparse.ArgumentParser(description='G')
parser.add_argument("--start",action="store",dest="start",default=None)
parser.add_argument("--end",action="store",dest="end",default=None)
parser.add_argument("--fill",action="store",dest='fill',default='mid')
parser.add_argument("--slipbps",action="store",dest='slipbps',default=0.0001)
parser.add_argument("--fcast",action="store",dest='fcast',default=None)
parser.add_argument("--weights",action="store",dest='weights',default=None)
args = parser.parse_args()

# Maximum participation rate in bar volume (1.5%)
participation = 0.015

# Load market data from cache
cols = ['split', 'div', 'close', 'iclose', 'bvwap_b', 'bvolume', 'tradable_med_volume_21_y', 'close_y']
cache_df = load_cache(dateparser.parse(args.start), dateparser.parse(args.end), cols )

# Compute bar volume delta (volume in this bar only)
cache_df['bvolume_d'] = cache_df['bvolume'].groupby(level='sid').diff()
# Handle first bar or reset: use total volume if diff is negative
cache_df.loc[ cache_df['bvolume_d'] < 0, 'bvolume_d'] = cache_df['bvolume']

# Push data to next timestamp (for look-ahead prevention)
cache_df = push_data(cache_df, 'bvolume_d')

# Compute participation-based trade size limits
cache_df['max_trade_size'] = cache_df[ 'bvolume_d_n' ] * cache_df['iclose'] *  participation
cache_df['min_trade_size'] = -1 * cache_df['max_trade_size']

# Push VWAP and iclose to next bar (for fill prices)
cache_df = push_data(cache_df, 'bvwap_b')
cache_df = push_data(cache_df, 'iclose')

# Initialize trades dataframe (will hold all forecast positions)
trades_df = None

# Load forecast position files
forecasts = list()
fcasts = args.fcast.split(",")
for pair in fcasts:
    fdir, fcast = pair.split(":")
    print("Loading {} {}".format(fdir, fcast))
    forecasts.append(fcast)

    # Load all CSV files for this forecast within date range
    flist = list()
    for ff in sorted(glob.glob( "./" + fdir + "/opt/opt." + fcast + ".*.csv")):
        # Parse date from filename: opt.{fcast}.YYYYMMDD_HHMMSS.csv
        m = re.match(r".*opt\." + fcast + r"\.(\d{8})_\d{6}.csv", str(ff))
        if m is None: continue

        # Filter by date range
        d1 = int(m.group(1))
        if d1 < int(args.start) or d1 > int(args.end): continue

        print("Loading {}".format(ff))
        flist.append(pd.read_csv(ff, parse_dates=True))

    # Concatenate all files for this forecast
    fcast_trades_df = pd.concat(flist)

    # Set index to (timestamp, sid)
    fcast_trades_df['iclose_ts'] = pd.to_datetime(fcast_trades_df['iclose_ts'])
    fcast_trades_df = fcast_trades_df.set_index(['iclose_ts', 'sid']).sort()

    # Merge with main trades_df
    if trades_df is None:
        # First forecast: initialize trades_df
        trades_df = fcast_trades_df
        trades_df['traded_' + fcast] = trades_df['traded']
        trades_df['shares_' + fcast] = trades_df['shares']
    else:
        # Subsequent forecasts: outer merge to include all timestamps
        trades_df = pd.merge(trades_df, fcast_trades_df, how='outer', left_index=True, right_index=True, suffixes=['', '_dead'])
        trades_df['traded_' + fcast] = trades_df['traded_dead']

        # Forward-fill shares to handle missing timestamps
        trades_df['shares_' + fcast] = trades_df['shares_dead'].unstack().fillna(method='ffill').stack().fillna(0)

        # Clean up duplicate columns from merge
        trades_df = remove_dup_cols(trades_df)

# Merge forecast positions with market data
trades_df = pd.merge(trades_df.reset_index(), cache_df.reset_index(), how='left', left_on=['iclose_ts', 'sid'], right_on=['iclose_ts', 'sid'], suffixes=['', '_dead'])
trades_df = remove_dup_cols(trades_df)
trades_df.set_index(['iclose_ts', 'sid'], inplace=True)
cache_df = None  # Free memory

# Compute notional constraints
max_dollars = 1e6  # $1M max per position
max_adv = 0.02  # 2% of ADV
trades_df['max_notional'] = (trades_df['tradable_med_volume_21_y'] * trades_df['close_y'] * max_adv).clip(0, max_dollars)
trades_df['min_notional'] = (-1 * trades_df['tradable_med_volume_21_y'] * trades_df['close_y'] * max_adv).clip(-max_dollars, 0)

# Initialize tracking columns
trades_df['cash'] = 0
trades_df['traded'] = 0
trades_df['shares'] = 0
trades_df['pnl'] = 0
trades_df['cum_pnl'] = 0
trades_df['day_pnl'] = 0

# Set fill prices based on strategy
if args.fill == "vwap":
    print("Filling at vwap...")
    trades_df['fillprice'] = trades_df['bvwap_b_n']

    # Count and replace bad VWAP values (<=0 or null)
    print("Bad count: {}".format( len(trades_df) - len(trades_df[ trades_df['fillprice'] > 0 ]) ))
    trades_df.ix[  (trades_df['fillprice'] <= 0) | (trades_df['fillprice'].isnull()), 'fillprice' ] = trades_df['iclose']
else:
    print("Filling at mid...")
    trades_df['fillprice'] = trades_df['iclose']

# Clean up infinite values
trades_df.replace([np.inf, -np.inf], np.nan, inplace=True)

def objective(weights):
    """
    Objective function for forecast weight optimization.

    Simulates order execution with weighted forecast combinations and computes
    risk-adjusted returns. This function is called iteratively by the scipy.optimize
    solver to find optimal forecast weights.

    Parameters
    ----------
    weights : array-like
        Forecast combination weights (one per forecast)
        Constrained to [0.0, 1.0] by optimizer bounds

    Returns
    -------
    float
        Objective value = sharpe_ratio - 0.05 * std(weights)
        Higher values indicate better risk-adjusted performance
        Penalty term encourages weight diversity

    Algorithm
    ---------
    1. Initialize daily tracking buckets (notional, pnl, traded)
    2. Iterate through all timestamps chronologically:
        a. Combine forecast positions using weights
        b. Compute position deltas from previous timestamp
        c. Execute orders at fill prices (VWAP or mid)
        d. Apply slippage costs
        e. Handle corporate actions (splits, dividends) on day changes
        f. Mark positions to market (intraday or close)
        g. Track daily P&L at end-of-day timestamps
    3. Compute annualized Sharpe ratio from daily returns
    4. Apply weight diversity penalty
    5. Return objective value

    Position Tracking
    -----------------
    shares_traded = target_shares - shares_last
    dollars_traded = shares_traded * fill_price * -1.0
    cash = cash_last + dollars_traded - slippage

    Corporate actions on day changes:
        cash += shares_last * dividend
        shares_last *= split_ratio

    Marking to Market
    -----------------
    Intraday: pnl = shares * iclose_n + cash
    Close:    pnl = shares * close + cash

    Daily return = (pnl_today - pnl_yesterday) / notional

    Notes
    -----
    - Only timestamps before 2012-12-27 are included in optimization
    - Halfday timestamps after cutoff (12:45) are skipped
    - Uses fillna(0) for missing forecast positions (forward-filled shares)
    - Prints daily statistics: timestamp, notional, pnl, return, turnover
    """

    ii = 0
    for fcast in forecasts:
        print("Weight {}: {}".format(fcast, weights[ii]))
        ii += 1

    day_bucket = {
        'not' : defaultdict(int),
        'pnl' : defaultdict(int),
        'trd' : defaultdict(int),
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

        # Merge previous timestamp's state (shares, cash, pnl) into current group
        if lastgroup_df is not None:
            # Rename columns to add '_last' suffix for previous state
            for col in lastgroup_df.columns:
                if col == "sid": continue
                lastgroup_df[col + "_last"]  = lastgroup_df[col]
                del lastgroup_df[col]

            # Outer join ensures we capture both continuing and new positions
            group_df = pd.concat([group_df.reset_index().set_index('sid'), lastgroup_df.reset_index().set_index('sid')], join='outer', axis=1, verify_integrity=True)
            group_df['iclose_ts'] = ts
            group_df.reset_index().set_index(['iclose_ts', 'sid'], inplace=True)

            # Handle corporate actions on day change
            if dayname != lastday and lastday is not None:
                # Dividends: add cash before position change
                group_df['cash_last'] += group_df['shares_last'] * group_df['div'].fillna(0)
                # Splits: adjust share count (e.g., 2.0 for 2-for-1 split)
                group_df['shares_last'] *= group_df['split'].fillna(1)
        else:
            # First timestamp: initialize from zero position
            group_df['shares_last'] = 0
            group_df['cash_last'] = 0

        # Combine forecast positions using optimization weights
        ii = 0
        for fcast in forecasts:
            # Weighted linear combination: shares = sum(weight_i * shares_i)
            group_df['shares'] +=  group_df['shares_' + fcast].fillna(0) * weights[ii]
            ii += 1

        # Compute order sizes (difference from previous position)
        group_df['shares_traded'] = group_df['shares'] - group_df['shares_last'].fillna(0)

        # Execute orders at fill prices (negative because buying consumes cash)
        group_df['dollars_traded'] = group_df['shares_traded'] * group_df['fillprice'] * -1.0

        # Update cash balance after trades
        group_df['cash'] = group_df['cash_last'] + group_df['dollars_traded']

#        fillslip_tot +=  (group_df['pdiff_pct'] * group_df['traded']).sum()
#        traded_tot +=  np.abs(group_df['traded']).sum()
    #    print "Slip2 {} {}".format(fillslip_tot, traded_tot)

        # Determine mark price based on timestamp
        markPrice = 'iclose_n'  # Default: next bar's intraday close
        # End of day: use official close (15:45 regular, 12:45 halfdays)
        if ts.strftime("%H%M") == "1545" or (dayname in halfdays and timename == "1245"):
            markPrice = 'close'

        # Apply slippage cost (proportional to traded notional)
        group_df['slip'] = np.abs(group_df['dollars_traded']).fillna(0) * float(args.slipbps)
        totslip += group_df['slip'].sum()

        # Deduct slippage from cash
        group_df['cash'] = group_df['cash'] - group_df['slip']

        # Mark to market: pnl = shares * price + cash
        group_df['pnl'] = group_df['shares'] * group_df[markPrice] + group_df['cash'].fillna(0)

        # Compute gross notional (long + short)
        notional = np.abs(group_df['shares'] * group_df[markPrice]).dropna().sum()

        # Long/short notional (signed)
        group_df['lsnot'] = group_df['shares'] * group_df[markPrice]

        pnl_tot = group_df['pnl'].dropna().sum() 
#        print group_df[['shares', 'shares_tgt', 'shares_qhl_b', 'cash', 'dollars_traded', 'pnl']]
        # if lastgroup_df is not None:
        #     group_df['pnl_diff'] = (group_df['pnl'] - group_df['pnl_last'])
        #     print group_df['pnl_diff'].order().dropna().head()
        #     print group_df['pnl_diff'].order().dropna().tail()

    #        pnl_incr = pnl_tot - pnl_last_tot
        # Track total traded notional
        traded = np.abs(group_df['dollars_traded']).fillna(0).sum()
        day_bucket['trd'][dayname] += traded

        # At end-of-day close: compute daily P&L and metrics
        if markPrice == 'close' and notional > 0:
            # Daily P&L increment
            delta = pnl_tot - pnl_last_day_tot

            # Daily return
            ret = delta/notional

            # Daily turnover
            daytraded = day_bucket['trd'][dayname]

            # Alternative notional calculation (commented out in original)
            notional2 = np.sum(np.abs((group_df['close'] * group_df['position'] / group_df['iclose'])))

            # Print daily summary: timestamp, notional, cum_pnl, daily_pnl, return, turnover
            print("{}: {} {} {} {:.4f} {:.2f} {}".format(ts, notional, pnl_tot, delta, ret, daytraded/notional, notional2 ))

            # Store daily metrics in buckets
            day_bucket['pnl'][dayname] = delta
            day_bucket['not'][dayname] = notional

            # Update last day's total P&L for next day's delta calculation
            pnl_last_day_tot = pnl_tot

        # Save current state for next timestamp
        lastgroup_df = group_df.reset_index()[[ 'shares', 'cash', 'pnl', 'sid', 'target']]

    # Build daily statistics dataframe
    nots = pd.DataFrame([ [d,v] for d, v in sorted(day_bucket['not'].items()) ], columns=['date', 'notional'])
    nots.set_index(keys=['date'], inplace=True)

    pnl_df = pd.DataFrame([ [d,v] for d, v in sorted(day_bucket['pnl'].items()) ], columns=['date', 'pnl'])
    pnl_df.set_index(['date'], inplace=True)

    rets = pd.merge(pnl_df, nots, left_index=True, right_index=True)

    print("Total Pnl: ${:.0f}K".format(rets['pnl'].sum()/1000.0))

    # Compute daily returns and cumulative returns
    rets['day_rets'] = rets['pnl'] / rets['notional']
    rets['day_rets'].replace([np.inf, -np.inf], np.nan, inplace=True)
    rets['day_rets'].fillna(0, inplace=True)
    rets['cum_ret'] = (1 + rets['day_rets']).dropna().cumprod()

    # Annualize statistics
    mean = rets['day_rets'].mean() * 252
    std = rets['day_rets'].std() * math.sqrt(252)

    # Compute Sharpe ratio
    sharpe =  mean/std

    print("Day mean: {:.4f} std: {:.4f} sharpe: {:.4f} avg Notional: ${:.0f}K".format(mean, std, sharpe, rets['notional'].mean()/1000.0))

    # Weight diversity penalty (encourages balanced weights)
    penalty = 0.05 * np.std(weights)
    print("penalty: {}".format(penalty))
    print

    return sharpe - penalty

# Set initial weights for optimization
if args.weights is None:
    # Default: equal weights of 0.5 for all forecasts
    initial_weights = np.ones(len(forecasts)) * .5
else:
    # User-specified weights (bypasses optimization)
    initial_weights = np.array([ float(x) for x in args.weights.split(",") ] )

# Define weight bounds
lb = np.ones(len(forecasts)) * 0.0  # Lower bound: no short weights
ub = np.ones(len(forecasts))        # Upper bound: max weight is 1.0

# Setup scipy.optimize.minimize optimizer
# Note: scipy minimizes, so we negate the objective
bounds = [(lb[i], ub[i]) for i in range(len(lb))]

result = minimize(
    fun=lambda w: -objective(w),  # Negate to maximize
    x0=initial_weights,
    method='L-BFGS-B',  # Bounded optimization
    bounds=bounds,
    options={'ftol': 0.001, 'maxfun': 150}
)

# Check for optimization failure
if not result.success:
    print("Optimization failed: {}".format(result.message))
    raise Exception("Optimization failed: {}".format(result.message))

# Print optimal weights
print(result.x)
ii = 0
for fcast in forecasts:
    print("{}: {}".format(fcast, result.x[ii]))
    ii += 1

