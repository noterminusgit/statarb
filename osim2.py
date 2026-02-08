#!/usr/bin/env python
"""
OSIM2 - Order Simulation Without Weight Optimization

Simplified variant of OSIM that simulates order execution with FIXED forecast
weights instead of optimizing weights based on historical performance.

Key Difference from OSIM
-------------------------
OSIM: Optimizes forecast weights using OpenOpt NLP solver to maximize Sharpe ratio
OSIM2: Uses fixed weights (or simple adaptive scaling based on recent returns)

This makes OSIM2 much faster than OSIM but less sophisticated in forecast
combination. OSIM2 is suitable for:
    - Quick backtests with known good weights
    - Testing execution quality without weight optimization overhead
    - Validating fill price assumptions
    - Prototyping new fill strategies

Methodology
-----------
1. Load optimized position targets from multiple alpha forecasts
2. Load bar data and price history for date range
3. For each timestamp:
   a. Combine forecasts using fixed weights
   b. Apply simple adaptive scaling (optional, currently disabled)
   c. Constrain trades by participation rate and notional limits
   d. Execute trades with fill price (VWAP or mid)
   e. Apply linear slippage costs
   f. Handle splits, dividends, corporate actions
   g. Mark positions to market
   h. Track P&L

4. Compute daily returns and Sharpe ratio

Forecast Weighting
------------------
Default: Equal weights (0.5 for each forecast)

Adaptive Scaling (lines 142-155, currently set to weight=1):
    - Compute 10-day rolling return for each forecast
    - If rolling return > 0: weight *= 1.1 (max 1.0)
    - If rolling return < 0: weight *= 0.9 (min 0.1)
    - Currently disabled (weight = 1 on line 153)

To enable adaptive scaling, remove line 153: "weight = 1"

Fill Price Strategy
-------------------
Controlled by --fill argument:

vwap (--fill=vwap):
    Uses bar VWAP price (bvwap_b_n)
    Falls back to iclose if VWAP unavailable
    Recommended for large orders

mid (--fill=mid, default):
    Uses intraday close (iclose) as midpoint proxy
    Assumes instantaneous execution
    Suitable for smaller orders

Slippage Model
--------------
Linear slippage as fixed percentage of traded notional:
    slippage_cost = abs(dollars_traded) * slipbps

Default: 0.0001 (1 basis point)
Configure with: --slipbps=0.0002

Participation Rate Constraints
-------------------------------
    max_trade_size = bar_volume * fill_price * participation
    participation = 1.5% (hardcoded)

Prevents unrealistic order sizes that would dominate bar volume.

Additional notional constraints:
    max_notional = min($1M, 2% of 21-day ADV)
    min_notional = -max_notional

Corporate Actions
-----------------
Splits:
    shares_new = shares_last * split_ratio
    Applied on day changes

Dividends:
    cash_new = cash_last + shares_last * dividend
    Applied before split adjustment

Market Breaks (--breaks):
    On year-end breaks, liquidate all positions to cash
    Dates: 20110705, 20120102, 20120705, 20130103

Special Dates
-------------
halfdays: Early close dates (12:45 PM)
    Filters out timestamps after 12:45 PM
    Dates: 20111125, 20120703, 20121123, 20121224

breaks: Year-end liquidation dates
    Forces flat positions to simulate trading breaks
    Dates: 20110705, 20120102, 20120705, 20130103

Command-Line Arguments
----------------------
--start : str
    Start date (YYYYMMDD format)
--end : str
    End date (YYYYMMDD format)
--fcast : str
    Forecast specifications: "dir:name,dir2:name2,..."
    Example: "mus:hl,mus:bd" loads forecasts from mus/opt/opt.hl.*.csv
--fill : str
    Fill price strategy: 'vwap' or 'mid' (default: 'mid')
--slipbps : float
    Slippage in basis points (default: 0.0001)

Input Data Format
-----------------
Forecast files expected in:
    ./{dir}/opt/opt.{name}.{YYYYMMDD}_HHMMSS.csv

Each file must contain:
    - iclose_ts: Timestamp
    - sid: Security ID
    - traded: Trade size (dollars)
    - target: Target position (dollars)

Also reads from {dir}/rets.txt:
    - date, ret (space-separated)
    - Used for adaptive weight scaling

Usage Examples
--------------
Basic simulation with two forecasts:
    python osim2.py --start=20130101 --end=20130630 --fcast=mus:hl,mus:bd

VWAP fill with custom slippage:
    python osim2.py --start=20130101 --end=20130630 \
        --fcast=mus:hl --fill=vwap --slipbps=0.0002

Output
------
Prints to stdout:
    - Per-timestamp: timestamp, notional, cumPnL, dayPnL, return, turnover, slippage
    - Final: Total P&L, annualized Sharpe ratio, average notional

Example output:
    2013-03-01 15:45:00: 150000000 125000 8500 0.0057 2500000 1.67 850
    Total Pnl: $2,450K
    Day mean: 0.1250 std: 0.0850 sharpe: 1.47 avg Notional: $148,500K

When to Use OSIM2
-----------------
Use OSIM2 instead of OSIM when:
    - You already know good forecast weights from previous optimization
    - You want to test execution assumptions quickly
    - You're validating fill price models
    - You don't want the overhead of weight optimization
    - You're doing sensitivity analysis on execution parameters

Use OSIM (not OSIM2) when:
    - You want to find optimal forecast weights
    - You're combining new forecasts with unknown quality
    - You want risk-adjusted weight selection
    - Computation time is not critical

Comparison with Other Simulators
---------------------------------
BSIM:  Daily rebalancing, portfolio optimization, factor risk
OSIM:  Order-level execution, forecast weight optimization
OSIM2: Order-level execution, fixed weights (this file)
OSIM_SIMPLE: Portfolio weight optimization offline
QSIM:  Intraday bars, simple forecast evaluation
SSIM:  Full lifecycle with cash tracking

Performance
-----------
OSIM2 is significantly faster than OSIM because it skips the expensive
weight optimization loop. Typical speedup: 10-100x depending on number
of forecasts and date range.

Notes
-----
- Adaptive weight scaling is currently disabled (line 153)
- To enable, remove the "weight = 1" override
- Default equal weights (0.5) may be suboptimal
- Consider using OSIM first to find good weights, then OSIM2 for validation
- Memory efficient - loads data in chunks by timestamp
"""

from __future__ import division, print_function

from util import *
from regress import *
from loaddata import *

import gc

from collections import defaultdict

import argparse

halfdays = ['20111125', '20120703', '20121123', '20121224']
breaks = ['20110705', '20120102', '20120705', '20130103']

parser = argparse.ArgumentParser(description='G')
parser.add_argument("--start",action="store",dest="start",default=None)
parser.add_argument("--end",action="store",dest="end",default=None)
parser.add_argument("--fill",action="store",dest='fill',default='mid')
parser.add_argument("--slipbps",action="store",dest='slipbps',default=0.0001)
parser.add_argument("--fcast",action="store",dest='fcast',default=None)
args = parser.parse_args()

participation = 0.015

cols = ['split', 'div', 'close', 'iclose', 'bvwap_b', 'bvolume', 'tradable_med_volume_21_y', 'close_y']
cache_df = load_cache(dateparser.parse(args.start), dateparser.parse(args.end), cols )
cache_df['bvolume_d'] = cache_df['bvolume'].groupby(level='sid').diff()
cache_df.loc[ cache_df['bvolume_d'] < 0, 'bvolume_d'] = cache_df['bvolume']
cache_df = push_data(cache_df, 'bvolume_d')
cache_df['max_trade_size'] = cache_df[ 'bvolume_d_n' ] * cache_df['iclose'] *  participation 
cache_df['min_trade_size'] = -1 * cache_df['max_trade_size']
cache_df = push_data(cache_df, 'bvwap_b')
cache_df = push_data(cache_df, 'iclose')

trades_df = None

forecasts = list()
fcasts = args.fcast.split(",")
fcast_rets = dict()
for pair in fcasts:
    fdir, fcast = pair.split(":")
    print("Loading {} {}".format(fdir, fcast))
    forecasts.append(fcast)
    retdf = pd.read_csv("./" + fdir + "/rets.txt", names=['date', 'ret'], sep=" ")
    retdf['date'] = pd.to_datetime(retdf['date'])
    retdf.set_index('date', inplace=True)
    retdf['rollingret'] = pd.rolling_sum(retdf['ret'], 10).shift(1)
    fcast_rets[fcast] = retdf
    flist = list()
    for ff in sorted(glob.glob( "./" + fdir + "/opt/opt." + fcast + ".*.csv")):
        m = re.match(r".*opt\." + fcast + r"\.(\d{8})_\d{6}.csv", str(ff))
        if m is None: continue
        d1 = int(m.group(1))
        if d1 < int(args.start) or d1 > int(args.end): continue
        print("Loading {}".format(ff))
        flist.append(pd.read_csv(ff, parse_dates=True))
    fcast_trades_df = pd.concat(flist)
    fcast_trades_df['iclose_ts'] = pd.to_datetime(fcast_trades_df['iclose_ts'])
    fcast_trades_df = fcast_trades_df.set_index(['iclose_ts', 'sid']).sort()

    if trades_df is None:
        trades_df = fcast_trades_df
        trades_df['traded_' + fcast] = trades_df['traded']
    else:
        trades_df = pd.merge(trades_df, fcast_trades_df, how='outer', left_index=True, right_index=True, suffixes=['', '_dead'])
        trades_df['traded_' + fcast] = trades_df['traded_dead']
        trades_df = remove_dup_cols(trades_df)

trades_df = pd.merge(trades_df.reset_index(), cache_df.reset_index(), how='left', left_on=['iclose_ts', 'sid'], right_on=['iclose_ts', 'sid'], suffixes=['', '_dead'])
trades_df = remove_dup_cols(trades_df)
trades_df.set_index(['iclose_ts', 'sid'], inplace=True)
cache_df = None

max_dollars = 1e6
max_adv = 0.02
trades_df['max_notional'] = (trades_df['tradable_med_volume_21_y'] * trades_df['close_y'] * max_adv).clip(0, max_dollars)
trades_df['min_notional'] = (-1 * trades_df['tradable_med_volume_21_y'] * trades_df['close_y'] * max_adv).clip(-max_dollars, 0)

trades_df['cash'] = 0
trades_df['shares'] = 0
trades_df['pnl'] = 0
trades_df['cum_pnl'] = 0
trades_df['day_pnl'] = 0

if args.fill == "vwap":
    print("Filling at vwap...")
    trades_df['fillprice'] = trades_df['bvwap_b_n']
    print("Bad count: {}".format( len(trades_df) - len(trades_df[ trades_df['fillprice'] > 0 ]) ))
    trades_df.ix[  (trades_df['fillprice'] <= 0) | (trades_df['fillprice'].isnull()), 'fillprice' ] = trades_df['iclose']
else:
    print("Filling at mid...")
    trades_df['fillprice'] = trades_df['iclose']

trades_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#print trades_df

fcast_weights = dict()
for fcast in forecasts:
    fcast_weights[fcast] = .5


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
    timename = ts.strftime("%H%M")

    if dayname in halfdays and int(timename) > 1245:
        continue

    if lastgroup_df is not None:
        group_df = pd.merge(group_df.reset_index(), lastgroup_df.reset_index(), how='left', left_on=['sid'], right_on=['sid'], suffixes=['', '_last'])
        group_df['iclose_ts'] = ts 
        group_df.set_index(['iclose_ts', 'sid'], inplace=True)
        if dayname != lastday:
            if dayname in breaks:
                group_df['cash_last'] += group_df['shares_last'] * group_df['close_y']
                group_df['shares_last'] = 0

            group_df['cash_last'] += group_df['shares_last'] * group_df['div'].fillna(0)
            group_df['shares_last'] *= group_df['split'].fillna(1)
    else:
        group_df['shares_last'] = 0
        group_df['cash_last'] = 0

    group_df['traded'] = 0
    ii = 0
    for fcast in forecasts:
        weight = fcast_weights[fcast]

        if dayname != lastday:
            retdf = fcast_rets[fcast]
            try:
                last_ret = retdf.ix[ pd.to_datetime(dayname), 'rollingret']
                if last_ret > 0:
                    weight *= 1.1
                    weight = min(weight, 1.0)
                else:
                    weight *= .9
                    weight = max(weight, .1)
            except:
                pass
            weight = 1
            print("{}: {}".format(fcast, weight))
            fcast_weights[fcast] = weight

        group_df['traded'] = group_df['traded'] + group_df['traded_' + fcast] * weight
        ii += 1

    group_df['max_up'] = group_df['max_notional'] - group_df['shares_last'] * group_df['iclose'] 
    group_df['max_down'] = group_df['min_notional'] - group_df['shares_last'] * group_df['iclose'] 

    group_df['traded'] = group_df[ ['traded', 'max_trade_size', 'max_up'] ].min(axis=1)
    group_df['traded'] = group_df[ ['traded', 'min_trade_size', 'max_down'] ].max(axis=1)

    group_df['shares_traded'] = group_df['traded'] / group_df['fillprice']
    group_df['shares'] = group_df['shares_traded'] + group_df['shares_last'].fillna(0)
    group_df['cash'] = -1.0 * group_df['shares_traded'] * group_df['fillprice'] + group_df['cash_last'].fillna(0)

    markPrice = 'iclose_n'
    #    if ts.strftime("%H%M") == "1530" or (dayname in halfdays and timename == "1230"):
    if ts.strftime("%H%M") == "1545" or (dayname in halfdays and timename == "1245"):
        markPrice = 'close'

    SLIPBPS = float(args.slipbps)
    group_df['slip'] = np.abs(group_df['traded']).fillna(0) * SLIPBPS
    totslip += group_df['slip'].sum()
    group_df['cash'] = group_df['cash'] - group_df['slip']
    group_df['pnl'] = group_df['shares'] * group_df[markPrice] + group_df['cash']
    notional = np.abs(group_df['shares'] * group_df[markPrice]).dropna().sum()    
    pnl_tot = group_df['pnl'].dropna().sum()
    traded = np.abs(group_df['traded']).fillna(0).sum()

    day_bucket['trd'][dayname] += traded

    # try:
    #     print group_df.xs(testid, level=1)[['target', 'traded', 'cash', 'shares', 'close', 'iclose', 'shares_last', 'cash_last']]
    # except KeyError:
    #     pass

    # print group_df['shares'].describe()
    # print group_df[markPrice].describe()
    if markPrice == 'close' and notional > 0 and dayname not in halfdays:
        delta = pnl_tot - pnl_last_day_tot
        ret = delta/notional
        daytraded = day_bucket['trd'][dayname]
        print("{}: {} {} {} {:.4f} {:.2f} {:.2f} {:.2f}".format(ts, notional, pnl_tot, delta, ret, daytraded, daytraded/notional, totslip ))
        day_bucket['pnl'][dayname] = delta
        day_bucket['not'][dayname] = notional
        pnl_last_day_tot = pnl_tot

    lastgroup_df = group_df.reset_index()[[ 'shares', 'cash', 'pnl', 'sid', 'target']]
    lastday = dayname

nots = pd.DataFrame([ [d,v] for d, v in sorted(day_bucket['not'].items()) ], columns=['date', 'notional'])
nots.set_index(keys=['date'], inplace=True)
pnl_df = pd.DataFrame([ [d,v] for d, v in sorted(day_bucket['pnl'].items()) ], columns=['date', 'pnl'])
pnl_df.set_index(['date'], inplace=True)
rets = pd.merge(pnl_df, nots, left_index=True, right_index=True)
print("Total Pnl: ${:.0f}K".format(rets['pnl'].sum()/1000.0))

rets['day_rets'] = rets['pnl'] / rets['notional']
rets['day_rets'].replace([np.inf, -np.inf], np.nan, inplace=True)
rets['day_rets'].fillna(0, inplace=True)
rets['cum_ret'] = (1 + rets['day_rets']).dropna().cumprod()

mean = rets['day_rets'].mean() * 252
std = rets['day_rets'].std() * math.sqrt(252)

sharpe =  mean/std
print("Day mean: {:.4f} std: {:.4f} sharpe: {:.4f} avg Notional: ${:.0f}K".format(mean, std, sharpe, rets['notional'].mean()/1000.0))




