#!/usr/bin/env python
"""
SSIM - System Simulation Engine

Full lifecycle backtesting simulation that tracks the complete order-to-execution
workflow including positions, cash balances, and P&L across extended periods.

This is the most comprehensive simulator in the system, providing complete lifecycle
tracking from order generation through execution to position management and final
settlement across months or years of trading data.

Simulation Methodology
----------------------
SSIM implements a full stateful simulation with:
    1. Position State Tracking: Maintains complete position state across all timestamps
       with corporate action adjustments (splits, dividends)
    2. Cash Flow Management: Tracks all cash inflows/outflows including trades,
       dividends, slippage costs, and trading fees
    3. Realistic Execution: Applies participation limits, ADV constraints, and
       market impact modeling
    4. Multi-Period Attribution: Aggregates P&L by day, month, time-of-day, and
       day-of-week for comprehensive performance analysis

Position Lifecycle States
-------------------------
Each security goes through a complete lifecycle:
    1. Target Generation: Optimization produces target positions (shares)
    2. Order Creation: Delta between current and target positions
    3. Execution Constraints: Limited by ADV participation and position limits
    4. Fill Processing: Executed at VWAP or mid price with slippage
    5. Position Update: Shares and cash balances updated
    6. Mark-to-Market: Positions marked at next bar's price
    7. Corporate Actions: Automatic adjustment for splits and dividends

Cash Tracking
-------------
Cash balances are maintained throughout the simulation:
    - Cash outflows: Purchase of shares (negative)
    - Cash inflows: Sale of shares (positive), dividend payments
    - Cash costs: Slippage, transaction costs
    - Running Balance: Carried forward across all timestamps
    - P&L Calculation: Position value + cash balance at each mark

Tracking Bucket System
----------------------
Performance metrics are aggregated into multiple bucket types:

    day_bucket: Daily aggregations
        - 'not': Gross notional exposure
        - 'pnl': Daily P&L change
        - 'trd': Daily turnover (total traded notional)
        - 'long': Long-only notional
        - 'short': Short-only notional (absolute value)

    month_bucket: Monthly aggregations
        - 'not': Average monthly notional
        - 'pnl': Monthly P&L
        - 'trd': Monthly turnover

    time_bucket: Intraday time-of-day aggregations
        - 'not': Average notional by time slot (e.g., "0930", "1545")
        - 'pnl': P&L by time slot
        - 'trd': Turnover by time slot

    dayofweek_bucket: Day-of-week aggregations (0=Monday, 6=Sunday)
        - 'not': Average notional by weekday
        - 'pnl': P&L by weekday
        - 'trd': Turnover by weekday

CLI Parameters
--------------
    --file <path>
        Optional CSV file containing pre-computed trades to replay
        Expected columns: iclose_ts, sid, vwap_n, traded

    --start <YYYYMMDD>
        Simulation start date (required)

    --end <YYYYMMDD>
        Simulation end date (required)

    --fill <vwap|mid>
        Fill price methodology (default: vwap)
        - vwap: Fill at bucket VWAP (bvwap_b), fallback to iclose
        - mid: Fill at interval close (iclose)

    --slipbps <float>
        Slippage in basis points applied to absolute traded notional
        Default: 0.0001 (1 bps)

    --fcast <spec>
        Forecast/alpha specification for multi-alpha combination
        Format: "dir:name:weight,dir2:name2:weight2,..."
        Example: "hl:1:0.6,bd:0.8:0.4"

        Each component:
            dir: Directory containing opt/ subfolder with optimization results
            name: Forecast name matching opt.<name>.<date>.csv files
            weight: Multiplier for combining this alpha's positions

    --cond <column>
        Conditional variable for decile-based P&L attribution analysis
        Default: mkt_cap
        Used to bucket stocks and analyze performance by characteristic

Execution Constraints
--------------------
SSIM applies realistic trading constraints:
    - max_dollars: $4M maximum position size per stock
    - max_adv: 2% of 21-day median ADV
    - participation: 1.5% maximum participation rate per bar
    - Constraints computed from tradable_med_volume_21_y and close_y
    - Shares traded capped by max/min_trade_shares per interval

Output and Reporting
--------------------
SSIM produces comprehensive performance analytics:

Console Output:
    - Daily P&L and returns with notional exposure
    - Total P&L, Sharpe ratio, mean return, volatility
    - Average turnover, long/short name counts
    - Fill slippage, opportunity cost, total slippage
    - Monthly P&L breakdown (in bps)
    - Time-of-day P&L breakdown (in bps)
    - Day-of-week P&L breakdown (in bps)
    - Factor exposure and P&L attribution (Barra + proprietary)
    - Conditional P&L by industry and decile
    - Forecast-trade correlation analysis

Generated Plots:
    - stocks.png: Histogram of P&L distribution across stocks
    - maxstock.png: Daily P&L histogram for best-performing stock
    - minstock.png: Daily P&L histogram for worst-performing stock
    - forecast_trade_corr.<period>.png: Scatter plot of forecast vs. actual trades
    - longshorts.<period>.png: Time series of long vs. short exposure
    - notional.<period>.png: Time series of gross notional
    - traded.<period>.png: Time series of daily turnover
    - rets.<period>.png: Cumulative return curve

DataFrame Columns:
    Position tracking: shares, shares_last, cash, cash_last, pnl, cum_pnl, day_pnl
    Trade execution: target, traded, shares_traded, fillprice, slip
    Market data: close, iclose, bvwap_b, split, div
    Attribution: forecast, position, lsnot (long/short notional)
    Constraints: max_notional, min_notional, max_trade_shares, min_trade_shares

Special Date Handling
--------------------
    halfdays: List of early market close dates (e.g., '20111125', '20120703')
        - Simulation stops at 12:45 on half days (vs. 15:45 normal days)
        - Ensures no execution after 12:30 close
        - Positions marked at close price at 12:45 timestamp

Usage Examples
--------------
Single alpha lifecycle simulation:
    python ssim.py --start=20130101 --end=20131231 --fcast=hl:1:1 --slipbps=0.0001

Multi-alpha combination:
    python ssim.py --start=20130101 --end=20131231 --fcast=hl:1:0.6,bd:0.8:0.4 --fill=vwap

Replay pre-computed trades:
    python ssim.py --start=20130101 --end=20131231 --file=trades.csv

Performance attribution by market cap:
    python ssim.py --start=20130101 --end=20131231 --fcast=combined:1:1 --cond=mkt_cap

Full year backtest with detailed analytics:
    python ssim.py --start=20120101 --end=20121231 --fcast=hl:1:1 --slipbps=0.0001

Notes
-----
- Requires pre-computed optimization results in dir/opt/opt.<fcast>.<date>.csv
- Loads data from cache via loaddata.load_cache() for efficiency
- Handles corporate actions automatically (splits, dividends)
- Memory intensive for long periods due to full position tracking
- Uses matplotlib for visualization (non-interactive backend)

See Also
--------
bsim.py : Daily-rebalancing simulation without full lifecycle tracking
osim.py : Order-level simulation with detailed execution modeling
qsim.py : Intraday 30-minute bar simulation
opt.py : Portfolio optimization producing target positions
"""

from __future__ import division, print_function

from util import *
from regress import *
from loaddata import *
import gc

from collections import defaultdict

import argparse

###############################################################################
# CONFIGURATION AND TRACKING STRUCTURES
###############################################################################

# Half-day market close dates (12:30 close instead of 16:00)
# Execution stops at 12:45, positions marked at close
halfdays = ['20111125', '20120703', '20121123', '20121224']

# Daily aggregation bucket: tracks daily performance metrics
# Keys are date strings in YYYYMMDD format
day_bucket = {
    'not' : defaultdict(int),    # Gross notional exposure (sum of abs positions)
    'pnl' : defaultdict(int),    # Daily P&L change
    'trd' : defaultdict(int),    # Daily turnover (sum of abs trades)
    'long' : defaultdict(int),   # Long-only notional
    'short' : defaultdict(int),  # Short-only notional (absolute value)
}

# Monthly aggregation bucket: tracks monthly performance
# Keys are month strings in YYYYMM format
month_bucket = {
    'not' : defaultdict(int),    # Cumulative notional (summed daily notionals)
    'pnl' : defaultdict(int),    # Monthly P&L
    'trd' : defaultdict(int),    # Monthly turnover
}

# Intraday time bucket: tracks performance by time of day
# Keys are time strings in HHMM format (e.g., "0930", "1545")
time_bucket = {
    'not' : defaultdict(int),    # Most recent notional at this time
    'pnl' : defaultdict(int),    # Cumulative P&L during this time slot
    'trd' : defaultdict(int),    # Cumulative turnover during this time slot
}

# Day-of-week bucket: tracks performance by weekday
# Keys are integers 0-6 (0=Monday, 6=Sunday)
dayofweek_bucket = {
    'not' : defaultdict(int),    # Cumulative notional (summed daily notionals)
    'pnl' : defaultdict(int),    # Cumulative P&L for this weekday
    'trd' : defaultdict(int),    # Cumulative turnover for this weekday
}

# Conditional bucket: for custom conditional analysis (currently unused in main loop)
cond_bucket = {
    'not' : defaultdict(int),
    'pnl' : defaultdict(int),
    'trd' : defaultdict(int),
}

# Position counters for tracking winning/losing positions
upnames = 0      # Count of positions with positive P&L
downnames = 0    # Count of positions with negative P&L

###############################################################################
# COMMAND LINE ARGUMENT PARSING
###############################################################################

parser = argparse.ArgumentParser(description='SSIM - System Simulation Engine for full lifecycle backtesting')
parser.add_argument("--file", action="store", dest="file", default=None,
                    help="Optional CSV file with pre-computed trades (columns: iclose_ts, sid, vwap_n, traded)")
parser.add_argument("--start", action="store", dest="start", default=None,
                    help="Simulation start date in YYYYMMDD format (required)")
parser.add_argument("--end", action="store", dest="end", default=None,
                    help="Simulation end date in YYYYMMDD format (required)")
parser.add_argument("--fill", action="store", dest='fill', default='vwap',
                    help="Fill price methodology: 'vwap' (bucket VWAP) or 'mid' (interval close)")
parser.add_argument("--slipbps", action="store", dest='slipbps', default=0.0001,
                    help="Slippage in basis points applied to absolute traded notional (default: 0.0001)")
parser.add_argument("--fcast", action="store", dest='fcast', default=None,
                    help="Forecast specification: 'dir:name:weight' or comma-separated for multi-alpha (required)")
parser.add_argument("--cond", action="store", dest='cond', default='mkt_cap',
                    help="Conditional variable for decile-based P&L attribution (default: mkt_cap)")
args = parser.parse_args()

###############################################################################
# DATA LOADING
###############################################################################

# Parse forecast specification into components for multi-alpha loading
fcasts = args.fcast.split(",")

# Define required columns for simulation
# Includes: corporate actions, prices, risk factors, fundamentals, volume metrics
cols = ['split', 'div', 'close', 'iclose', 'bvwap_b', args.cond, 'indname1', 'srisk_pct',
        'gdate', 'rating_mean', 'ticker', 'tradable_volume', 'tradable_med_volume_21_y',
        'mdvp_y', 'overnight_log_ret', 'date', 'log_ret', 'bvolume', 'capitalization',
        'cum_log_ret', 'dpvolume_med_21', 'volat_21_y', 'close_y']
cols.extend(BARRA_FACTORS)

# Load data from HDF5 cache for date range
cache_df = load_cache(dateparser.parse(args.start), dateparser.parse(args.end), cols)

# Push (forward-fill) VWAP and close prices to next interval for marking positions
# This ensures we mark at the next bar's open/reference price
cache_df = push_data(cache_df, 'bvwap_b')
cache_df = push_data(cache_df, 'iclose')

###############################################################################
# TRADE LOADING AND MULTI-ALPHA COMBINATION
###############################################################################

trades_df = None

if args.file is not None:
    # Mode 1: Load pre-computed trades from CSV file for replay
    trades_df = pd.read_csv(args.file, parse_dates=True, usecols=['iclose_ts', 'sid', 'vwap_n', 'traded'])
else:
    # Mode 2: Load optimization results and combine multiple alphas
    for pair in fcasts:
        # Parse forecast specification: directory:name:weight
        fdir, fcast, weight = pair.split(":")
        print(fdir, fcast, weight)

        # Load all optimization output files for this alpha in date range
        flist = list()
        for ff in sorted(glob.glob("./" + fdir + "/opt/opt." + fcast + ".*.csv")):
            m = re.match(r".*opt\." + fcast + r"\.(\d{8})_\d{6}.csv", str(ff))
            if m is None: continue
            d1 = int(m.group(1))
            if d1 < int(args.start) or d1 > int(args.end): continue
            print("Loading {}".format(ff))
            flist.append(pd.read_csv(ff, parse_dates=True))

        # Concatenate all files for this alpha and set index
        fcast_trades_df = pd.concat(flist)
        fcast_trades_df['iclose_ts'] = pd.to_datetime(fcast_trades_df['iclose_ts'])
        fcast_trades_df = fcast_trades_df.set_index(['iclose_ts', 'sid']).sort()

        print(fcast)
        print(fcast_trades_df.xs(testid, level=1)[['traded','shares']])

        if trades_df is None:
            # First alpha: initialize trades_df with weighted shares
            trades_df = fcast_trades_df
            trades_df['shares'] = trades_df['shares'].fillna(0) * float(weight)
        else:
            # Subsequent alphas: merge and add weighted shares
            # Outer merge ensures all timestamps/sids from both alphas are included
            trades_df = pd.merge(trades_df, fcast_trades_df, how='outer',
                               left_index=True, right_index=True, suffixes=['', '_dead'])

            # Combine shares with forward-fill for missing values, then add weighted component
            trades_df['shares'] = (trades_df['shares'].fillna(method='ffill').fillna(0) +
                                  trades_df['shares_dead'].fillna(method='ffill').fillna(0) * float(weight))

            # Remove duplicate columns created by merge
            trades_df = remove_dup_cols(trades_df)

# Merge trade targets with market data cache
trades_df = pd.merge(trades_df.reset_index(), cache_df.reset_index(), how='left',
                    left_on=['iclose_ts', 'sid'], right_on=['iclose_ts', 'sid'],
                    suffixes=['', '_dead'])
trades_df = remove_dup_cols(trades_df)
trades_df.set_index(['iclose_ts', 'sid'], inplace=True)

# Free cache_df memory
cache_df = None

# Initialize position tracking columns
trades_df['forecast_abs'] = np.abs(trades_df['forecast'])
trades_df['cash'] = 0         # Cash balance (running)
trades_df['pnl'] = 0          # Total P&L (position value + cash)
trades_df['cum_pnl'] = 0      # Cumulative P&L (same as pnl, kept for compatibility)
trades_df['day_pnl'] = 0      # Daily P&L change

###############################################################################
# SIMULATION STATE INITIALIZATION
###############################################################################

# Position state tracking across iterations
lastgroup_df = None          # Previous timestamp's position state (shares, cash, pnl, target)
lastday = None               # Previous day name for corporate action detection
last_ts = None               # Previous timestamp for final slice retrieval

# Running P&L totals
pnl_last_tot = 0            # Total P&L at previous timestamp
pnl_last_day_tot = 0        # Total P&L at previous day's close

# Slippage and turnover tracking
fillslip_tot = 0            # Cumulative fill slippage (bps)
traded_tot = 0              # Cumulative traded notional for slippage calculation
totslip = 0                 # Total slippage cost ($)
totturnover = 0             # Cumulative daily turnover ratio

# Counter and position tracking
cnt = 0                     # Number of days processed
long_names = 0              # Count of long trades across all days
short_names = 0             # Count of short trades across all days

###############################################################################
# FILL PRICE CALCULATION
###############################################################################

if args.fill == "vwap":
    # Fill at bucket VWAP (bvwap_b_n is next bar's VWAP due to push_data)
    print("Filling at vwap...")
    trades_df['fillprice'] = trades_df['bvwap_b_n']
    print("Bad count: {}".format(len(trades_df) - len(trades_df[trades_df['fillprice'] > 0])))

    # Fallback to interval close if VWAP is missing or invalid
    trades_df.ix[(trades_df['fillprice'] <= 0) | (trades_df['fillprice'].isnull()), 'fillprice'] = trades_df['iclose']
else:
    # Fill at interval close (mid price approximation)
    print("Filling at mid...")
    trades_df['fillprice'] = trades_df['iclose']

# Calculate fill quality metrics
trades_df['pdiff'] = trades_df['fillprice'] - trades_df['iclose']
trades_df['pdiff_pct'] = trades_df['pdiff'] / trades_df['iclose']
trades_df['unfilled'] = trades_df['target'] - trades_df['traded']
trades_df['slip2close'] = (trades_df['close'] - trades_df['fillprice']) / trades_df['fillprice']

###############################################################################
# EXECUTION CONSTRAINTS CALCULATION
###############################################################################

# Position size limits
max_dollars = 4e6           # Maximum $4M position per stock
max_adv = 0.02             # Maximum 2% of 21-day median ADV
participation = 0.015       # Maximum 1.5% participation rate per bar

# Calculate maximum notional position based on ADV and dollar limit
trades_df['max_notional'] = (trades_df['tradable_med_volume_21_y'] * trades_df['close_y'] * max_adv).clip(0, max_dollars)
trades_df['min_notional'] = (-1 * trades_df['tradable_med_volume_21_y'] * trades_df['close_y'] * max_adv).clip(-max_dollars, 0)

# Calculate bar volume for participation constraints
# bvolume_d is the volume traded in this bar (diff of cumulative volume)
trades_df['bvolume_d'] = trades_df['bvolume'].groupby(level='sid').diff()

# Handle cases where cumulative volume resets (e.g., new day)
trades_df.loc[trades_df['bvolume_d'] < 0, 'bvolume_d'] = trades_df['bvolume']

# Push forward for next bar's execution
trades_df = push_data(trades_df, 'bvolume_d')

# Maximum shares tradable in one bar based on participation rate
trades_df['max_trade_shares'] = trades_df['bvolume_d_n'] * participation
trades_df['min_trade_shares'] = -1 * trades_df['max_trade_shares']

# Create z-scores for factor analysis
trades_df = create_z_score(trades_df, 'srisk_pct')
trades_df = create_z_score(trades_df, 'rating_mean')
trades_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Force garbage collection before main loop
gc.collect()

###############################################################################
# MAIN SIMULATION LOOP
###############################################################################
# Iterate through each timestamp, updating positions and tracking P&L
# Each iteration:
#   1. Merges with previous position state
#   2. Applies corporate actions (splits, dividends)
#   3. Calculates shares to trade (constrained by participation limits)
#   4. Updates cash balance (trades, slippage, dividends)
#   5. Marks positions to market
#   6. Aggregates metrics into tracking buckets

for ts, group_df in trades_df.groupby(level='iclose_ts'):
    # Extract date/time components for bucketing
    dayname = ts.strftime("%Y%m%d")      # YYYYMMDD format
    monthname = ts.strftime("%Y%m")      # YYYYMM format
    weekdayname = ts.weekday()           # 0-6 (Monday-Sunday)
    timename = ts.strftime("%H%M")       # HHMM format

    # Skip bars after 12:45 on half-days (market closes at 12:30)
    if dayname in halfdays and int(timename) > 1245:
        continue

    # Merge with previous position state for lifecycle tracking
    if lastgroup_df is not None:
        # Join on sid to get previous shares, cash, pnl, target for each stock
        group_df = pd.merge(group_df.reset_index(), lastgroup_df.reset_index(), how='left',
                          left_on=['sid'], right_on=['sid'], suffixes=['', '_last'])
        group_df['iclose_ts'] = ts
        group_df.set_index(['iclose_ts', 'sid'], inplace=True)

        # Apply corporate actions on new day
        if dayname != lastday:
            # Add dividend cash to balance
            group_df['cash_last'] += group_df['shares_last'] * group_df['div'].fillna(0)
            # Adjust shares for splits
            group_df['shares_last'] *= group_df['split'].fillna(1)
    else:
        # First iteration: initialize position state
        group_df['shares_last'] = 0
        group_df['cash_last'] = 0

    # Calculate shares to trade (target - current position)
    group_df['shares1'] = group_df['shares']  # Store original target for debugging
    group_df['shares_traded'] = group_df['shares'] - group_df['shares_last'].fillna(0)

    # Apply participation constraints: limit shares traded per bar
    # Cap at maximum shares tradable (for buys)
    group_df['shares_traded'] = group_df[['shares_traded', 'max_trade_shares']].min(axis=1)
    # Floor at minimum shares tradable (for sells)
    group_df['shares_traded'] = group_df[['shares_traded', 'min_trade_shares']].max(axis=1)

    # Update actual shares held after constrained trade
    group_df['shares'] = group_df['shares_last'] + group_df['shares_traded']

    # Calculate traded notional at fill price
    group_df['traded2'] = group_df['shares_traded'] * group_df['fillprice']
    group_df['traded'] = group_df['traded2']

    # Update cash balance: subtract trade cost, add previous cash
    group_df['cash'] = -1.0 * group_df['traded2'] + group_df['cash_last'].fillna(0)

    # Track fill slippage (difference between fill price and reference price)
    fillslip_tot += (group_df['pdiff_pct'] * group_df['traded']).sum()
    traded_tot += np.abs(group_df['traded']).sum()

    # Determine marking price: use close price at end of day, otherwise next bar's iclose
    markPrice = 'iclose_n'
    if ts.strftime("%H%M") == "1545" or (dayname in halfdays and timename == "1245"):
        # Mark at close price at end of trading day
        markPrice = 'close'

    # Apply slippage cost (in basis points of traded notional)
    group_df['slip'] = np.abs(group_df['traded']).fillna(0) * float(args.slipbps)
    totslip += group_df['slip'].sum()
    group_df['cash'] = group_df['cash'] - group_df['slip']

    # Calculate P&L: position value (shares * mark price) + cash balance
    group_df['pnl'] = trades_df.ix[group_df.index, 'cum_pnl'] = group_df['shares'] * group_df[markPrice] + group_df['cash']

    # Calculate notional exposure (sum of absolute position values)
    notional = np.abs(group_df['shares'] * group_df[markPrice]).dropna().sum()

    # Calculate signed notional (for long/short breakdown)
    group_df['lsnot'] = group_df['shares'] * group_df[markPrice]

    # Sum total P&L across all positions
    pnl_tot = group_df['pnl'].dropna().sum()
    
    # Calculate P&L change since last timestamp
    pnl_incr = pnl_tot - pnl_last_tot

    # Calculate turnover (sum of absolute traded notional)
    traded = np.abs(group_df['traded']).fillna(0).sum()

    # Update tracking buckets with turnover
    day_bucket['trd'][dayname] += traded
    month_bucket['trd'][monthname] += traded
    dayofweek_bucket['trd'][weekdayname] += traded
    time_bucket['trd'][timename] += traded

    # At end of day (close mark): calculate daily metrics and update buckets
    if markPrice == 'close' and notional > 0:
        # Calculate daily P&L and return
        delta = pnl_tot - pnl_last_day_tot
        ret = delta / notional
        daytraded = day_bucket['trd'][dayname]

        # Calculate adjusted notional (accounting for intraday price changes)
        notional2 = np.sum(np.abs((group_df['close'] * group_df['position'] / group_df['iclose'])))

        # Print daily summary
        print("{}: {} {} {} {:.4f} {:.2f} {}".format(ts, notional, pnl_tot, delta, ret,
                                                     daytraded/notional, notional2))

        # Update daily tracking buckets
        day_bucket['pnl'][dayname] = delta
        month_bucket['pnl'][monthname] += delta
        dayofweek_bucket['pnl'][weekdayname] += delta

        day_bucket['not'][dayname] = notional
        day_bucket['long'][dayname] = group_df[group_df['lsnot'] > 0]['lsnot'].dropna().sum()
        day_bucket['short'][dayname] = np.abs(group_df[group_df['lsnot'] < 0]['lsnot'].dropna().sum())

        month_bucket['not'][monthname] += notional
        dayofweek_bucket['not'][weekdayname] += notional

        # Store daily P&L change in main dataframe
        trades_df.ix[group_df.index, 'day_pnl'] = group_df['pnl'] - group_df['pnl_last']

        # Update running totals and counters
        pnl_last_day_tot = pnl_tot
        totturnover += daytraded / notional
        short_names += len(group_df[group_df['traded'] < 0])
        long_names += len(group_df[group_df['traded'] > 0])
        cnt += 1

    # Update intraday time bucket
    time_bucket['pnl'][timename] += pnl_incr
    time_bucket['not'][timename] = notional

    # Track winning vs. losing positions
    upnames += len(group_df[group_df['pnl'] > 0])
    downnames += len(group_df[group_df['pnl'] < 0])

    # Save current state for next iteration
    lastgroup_df = group_df.reset_index()[['shares', 'cash', 'pnl', 'sid', 'target']]
    lastday = dayname
    pnl_last_tot = pnl_tot
    last_ts = ts


###############################################################################
# POST-SIMULATION ANALYSIS AND REPORTING
###############################################################################

period = "{}.{}".format(args.start, args.end)

print()
print()
print("=" * 80)
print("EXECUTION QUALITY METRICS")
print("=" * 80)
print("Fill Slip: {}".format(fillslip_tot/traded_tot))
oppslip = (trades_df['unfilled'] * trades_df['slip2close']).sum()
print("Opp slip: {}".format(oppslip))
print("Totslip: {}".format(totslip))
print("Avg turnover: {}".format(totturnover/cnt))
print("Longs: {}".format(long_names/cnt))
print("Shorts: {}".format(short_names/cnt))
print()


print("=" * 80)
print("CONDITIONAL P&L BREAKDOWN")
print("=" * 80)

# Extract final timestamp slice for attribution analysis
lastslice = trades_df.xs(last_ts, level='iclose_ts')
condname = args.cond

# Industry-level P&L attribution
print("By Industry:")
for ind in INDUSTRIES:
    decile = lastslice[lastslice['indname1'] == ind]
    print("{}: {}".format(ind, decile['cum_pnl'].sum()))

print()

# Decile-based P&L attribution by conditional variable
print("By {} Decile:".format(condname))
lastslice['decile'] = lastslice[condname].rank() / float(len(lastslice)) * 10
lastslice['decile'] = lastslice['decile'].fillna(-1)
lastslice['decile'] = lastslice['decile'].astype(int)
for ii in range(-1, 10):
    decile = lastslice[lastslice['decile'] == ii]
    print("Decile {}: {} mean, {} P&L".format(ii, decile[condname].mean(), decile['cum_pnl'].sum()))

print()
print("=" * 80)
print("STOCK-LEVEL P&L ANALYSIS")
print("=" * 80)

# Extract first and last slices for comparison
firstslice = trades_df.xs(min(trades_df.index)[0], level='iclose_ts')
pnlbystock = lastslice['cum_pnl'].fillna(0)

# Generate P&L distribution histogram
plt.figure()
pnlbystock.hist(bins=1800)
plt.savefig("stocks.png")

# Identify best and worst performing stocks
maxpnlid = pnlbystock.idxmax()
minpnlid = pnlbystock.idxmin()

print("Max pnl stock: {} with P&L {}".format(maxpnlid, pnlbystock.ix[maxpnlid]))
print("Min pnl stock: {} with P&L {}".format(minpnlid, pnlbystock.ix[minpnlid]))

# Generate daily P&L distribution for best-performing stock
plt.figure()
maxstock_df = trades_df.xs(maxpnlid, level=1)
maxstock_df['day_pnl'].hist(bins=100)
plt.savefig("maxstock.png")
print 

print("=" * 80)
print("FACTOR EXPOSURE AND P&L ATTRIBUTION")
print("=" * 80)

# Prepare factor z-scores for first and last slices
firstslice = create_z_score(firstslice, 'srisk_pct')
firstslice = create_z_score(firstslice, 'rating_mean')

# Merge first and last slices for cross-sectional analysis
merge = pd.merge(firstslice.reset_index(), lastslice.reset_index(),
                left_on=['sid'], right_on=['sid'], suffixes=['_first', '_last'])

lastnotional = np.abs(lastslice['position']).sum()

# Calculate factor exposure and P&L attribution for each factor
for factor in BARRA_FACTORS + PROP_FACTORS:
    # Exposure: P&L-weighted average factor value
    exposure = (merge['cum_pnl_last'] * merge[factor + '_first']).sum() / lastnotional

    # P&L attribution: sum of daily P&L weighted by factor value
    pnl = (trades_df['day_pnl'] * trades_df[factor]).sum()

    print("{}: exposure: {:.2f}, pnl: {}".format(factor, exposure, pnl))
print()

print("=" * 80)
print("FORECAST-TRADE CORRELATION")
print("=" * 80)
print(trades_df[['forecast', 'traded', 'target']].corr())

# Generate scatter plot of forecast vs. actual trades
plt.figure()
plt.scatter(trades_df['forecast'], trades_df['traded'])
plt.savefig("forecast_trade_corr." + period + ".png")
print 

###############################################################################
# TIME SERIES VISUALIZATION
###############################################################################

# Generate long/short exposure time series
longs = pd.DataFrame([[d, v] for d, v in sorted(day_bucket['long'].items())],
                     columns=['date', 'long'])
longs.set_index(keys=['date'], inplace=True)

shorts = pd.DataFrame([[d, v] for d, v in sorted(day_bucket['short'].items())],
                      columns=['date', 'short'])
shorts.set_index(keys=['date'], inplace=True)

longshorts = pd.merge(longs, shorts, how='inner', left_index=True, right_index=True)
plt.figure()
longshorts[['long', 'short']].plot()
plt.savefig("longshorts." + period + ".png")

# Generate notional exposure time series
nots = pd.DataFrame([[d, v] for d, v in sorted(day_bucket['not'].items())],
                    columns=['date', 'notional'])
nots.set_index(keys=['date'], inplace=True)
plt.figure()
nots['notional'].plot()
plt.savefig("notional." + period + ".png")

# Generate turnover time series
trds = pd.DataFrame([[d, v] for d, v in sorted(day_bucket['trd'].items())],
                    columns=['date', 'traded'])
trds.set_index(keys=['date'], inplace=True)
plt.figure()
trds['traded'].plot()
plt.savefig("traded." + period + ".png")

###############################################################################
# PERFORMANCE METRICS CALCULATION
###############################################################################

# Build daily returns dataframe
pnl_df = pd.DataFrame([[d, v] for d, v in sorted(day_bucket['pnl'].items())],
                      columns=['date', 'pnl'])
pnl_df.set_index(['date'], inplace=True)

rets = pd.merge(pnl_df, nots, left_index=True, right_index=True)
rets = pd.merge(rets, trds, left_index=True, right_index=True)

print("=" * 80)
print("OVERALL PERFORMANCE SUMMARY")
print("=" * 80)
print("Total Pnl: ${:.0f}K".format(rets['pnl'].sum()/1000.0))

# Calculate daily returns (P&L divided by previous day's notional)
rets['day_rets'] = rets['pnl'] / rets['notional'].shift(1)
rets['day_rets'].replace([np.inf, -np.inf], np.nan, inplace=True)
rets['day_rets'].fillna(0, inplace=True)

# Calculate cumulative return curve
rets['cum_ret'] = (1 + rets['day_rets']).dropna().cumprod()

# Generate cumulative return plot
plt.figure()
rets['cum_ret'].plot()
plt.draw()
plt.savefig("rets." + period + ".png")

# Annualize performance metrics
mean = rets['day_rets'].mean() * 252
std = rets['day_rets'].std() * math.sqrt(252)
sharpe = mean / std

print("Annualized mean: {:.4f}".format(mean))
print("Annualized std: {:.4f}".format(std))
print("Sharpe ratio: {:.4f}".format(sharpe))
print("Avg Notional: ${:.0f}K".format(rets['notional'].mean()/1000.0))
print()

###############################################################################
# TEMPORAL BREAKDOWN ANALYSIS
###############################################################################

print("=" * 80)
print("MONTHLY BREAKDOWN (BPS)")
print("=" * 80)
for month in sorted(month_bucket['not'].keys()):
    notional = month_bucket['not'][month]
    traded = month_bucket['trd'][month]
    if notional > 0:
        # P&L in basis points, turnover ratio
        print("Month {}: {:.4f} bps, {:.4f} turnover".format(
            month, 10000 * month_bucket['pnl'][month]/notional, traded/notional))
print()

print("=" * 80)
print("TIME-OF-DAY BREAKDOWN (BPS)")
print("=" * 80)
for time in sorted(time_bucket['not'].keys()):
    notional = time_bucket['not'][time]
    traded = time_bucket['trd'][time]
    if notional > 0:
        print("Time {}: {:.4f} bps, {:.4f} turnover".format(
            time, 10000 * time_bucket['pnl'][time]/notional, traded/notional))
print()

print("=" * 80)
print("DAY-OF-WEEK BREAKDOWN (BPS)")
print("=" * 80)
dayofweek_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for dayofweek in sorted(dayofweek_bucket['not'].keys()):
    notional = dayofweek_bucket['not'][dayofweek]
    traded = dayofweek_bucket['trd'][dayofweek]
    if notional > 0:
        print("{} ({}): {:.4f} bps, {:.4f} turnover".format(
            dayofweek_names[dayofweek], dayofweek,
            10000 * dayofweek_bucket['pnl'][dayofweek]/notional, traded/notional))
print()

print("=" * 80)
print("POSITION STATISTICS")
print("=" * 80)
print("Win Rate (% positions with positive P&L): {:.4f}".format(
    float(upnames)/(upnames+downnames)))


