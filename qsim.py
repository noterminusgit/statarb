#!/usr/bin/env python
"""
QSIM - Quote/Intraday Simulation Engine

Intraday backtesting simulation that evaluates alpha signals at multiple
time slices throughout the trading day using 30-minute bar data. Unlike daily
simulators (BSIM/OSIM), QSIM operates on intraday bars to capture time-of-day
effects and optimize entry/exit timing.

Key Features:
    - 30-minute bar simulation from 9:45 AM to 4:00 PM
    - Multi-horizon forward return analysis (1-5 bars ahead)
    - Time-of-day performance attribution
    - Day-of-week stratification analysis
    - Intraday volume participation constraints (1% of bar volume)
    - VWAP-based execution modeling with incremental VWAP calculation
    - Market-neutral delta hedging
    - Decile-based conditioning analysis

Time Slices:
    - 9:45, 10:00, 10:15, 10:30, ..., 15:30, 15:45, 16:00
    - Each slice represents a 30-minute bar close time
    - 13 time slices per trading day (9:30-16:00)

Simulation Methodology:
    1. Load 30-minute bar data (bvolume, bvwap, iclose)
    2. Compute incremental VWAP between bars: vwap_n = (bvwap_n * bvolume_n - bvwap * bvolume) / bvolume_d_n
    3. Load intraday alpha forecasts (e.g., qhl_intra, badj_intra)
    4. Mix multiple forecasts with optional weights
    5. Size positions: shares = ALPHA_MULT * forecast, capped at 1% bar volume
    6. Apply max position size: $500K per stock
    7. Calculate P&L: notional * (exp(cum_ret) - 1) - slip
    8. Track P&L across multiple horizons (0 to horizon bars)
    9. Aggregate P&L by time-of-day, day-of-week, month, conditioning deciles

Command-Line Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --fcast: Intraday alpha signal files (format: "name:multiplier" or "name:multiplier:weight")
             Example: "qhl_intra:1" or "qhl_intra:1:0.6,badj_intra:1:0.4"
    --cond: Conditioning variable for decile stratification (default: mkt_cap)
            Stocks are ranked into deciles by this variable to analyze performance
            across different segments (e.g., market cap, volume, volatility)
    --horizon: Forecast horizon in 30-minute bars (default: 3 = 90 minutes)
               QSIM evaluates forward returns from 0 to horizon bars
    --mult: Alpha multiplier for position sizing (default: 1000.0)
            Higher multipliers = more aggressive positions
    --slipbps: Slippage in basis points of notional (default: 0.0001 = 1 bps)
    --vwap: Use incremental VWAP as fill price instead of iclose (default: False)
            If True: fill at bar's incremental VWAP
            If False: fill at bar's iclose (last trade)

Analysis Output:
    - Cumulative P&L curves with Sharpe ratios for each horizon
    - P&L by time-of-day (identifies best intraday trading windows)
    - P&L by day-of-week (Mon=0, Tue=1, ..., Fri=4)
    - P&L by month (identifies seasonal patterns)
    - P&L by conditioning deciles (e.g., small-cap vs. large-cap performance)
    - Forecast correlation matrix
    - Forecast strength time series
    - Alpha distribution histograms
    - Notional bias analysis (long vs. short exposure)
    - Win rate statistics (% profitable positions)

Output Files:
    - rets.{horizon}.{forecast_names}.png: Cumulative returns chart
    - forecast_strength.png: Forecast volatility over time
    - {forecast_name}__hist.png: Alpha distribution histogram
    - stocks.png: Per-stock P&L distribution
    - maxstock.png: Highest P&L stock's daily returns
    - notional_bias.png: Net long/short exposure over time
    - alpha_bias.png: Positive/negative forecast ratio
    - notional.png: Total notional deployed over time
    - debug.csv: Detailed position/P&L data for debugging
    - max_notional_day.csv: Day with maximum notional bias

Use Cases:
    - Testing market-making strategies with intraday signals
    - Identifying optimal entry/exit times (e.g., avoid 9:45-10:00 volatility)
    - Evaluating intraday alpha decay across multiple horizons
    - Optimizing forecast combinations for intraday trading
    - Analyzing time-of-day and day-of-week effects

Example Usage:
    # Single intraday forecast, 3-bar (90-min) horizon
    python qsim.py --start=20130101 --end=20130630 --fcast=qhl_intra:1 --horizon=3

    # Multi-forecast combination with VWAP execution
    python qsim.py --start=20130101 --end=20130630 --fcast=qhl_intra:1:0.6,badj_intra:1:0.4 --vwap=True

    # Analyze performance across volume deciles
    python qsim.py --start=20130101 --end=20130630 --fcast=qhl_intra:1 --cond=tradable_volume

Data Requirements:
    - 30-minute bar data: bvolume, bvwap, iclose (loaded via load_cache)
    - Intraday forecast files: HDF5 files with bar-level alphas
    - Market cap data for delta hedging (mkt_cap)
    - Forward returns: cum_ret1, cum_ret2, ..., cum_ret{horizon}

Performance Notes:
    - QSIM is slower than BSIM due to 13x more time slices per day
    - Memory intensive: loads all bars for entire date range
    - Recommended date range: 6-12 months for development, longer for production
"""

from __future__ import division, print_function

from util import *
from regress import *
from loaddata import *

from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description='QSIM - Intraday 30-minute bar simulation engine')
parser.add_argument("--start",action="store",dest="start",default=None)
parser.add_argument("--end",action="store",dest="end",default=None)
parser.add_argument("--fcast",action="store",dest="fcast",default=None)
parser.add_argument("--cond",action="store",dest="cond",default="mkt_cap")
parser.add_argument("--horizon",action="store",dest="horizon",default=3)
parser.add_argument("--mult",action="store",dest="mult",default=1000.0)
parser.add_argument("--slipbps",action="store",dest="slipbps",default=0.0001)
parser.add_argument("--vwap",action="store",dest="vwap",default=False)
args = parser.parse_args()

ALPHA_MULT = float(args.mult)
horizon = int(args.horizon)
slipbs = float(args.slipbps)
start = args.start
end = args.end

# Define required columns for simulation
# Core bar data: iclose (bar close), bvwap (bar VWAP), bvolume (bar volume)
# Volume metrics: tradable_volume, tradable_med_volume_21_y (21-day median)
# Price data: close (EOD close for forward returns)
# Returns: overnight_log_ret, log_ret, cum_ret1..cum_ret{horizon}
# Conditioning: mkt_cap (for delta hedging and decile analysis)
cols = ['ticker', 'iclose', 'tradable_volume', 'close', 'bvwap', 'tradable_med_volume_21_y', 'mdvp', 'overnight_log_ret', 'date', 'log_ret', 'bvolume', 'mkt_cap']
if args.cond not in cols:
    cols.append(args.cond)
for ii in range(1, horizon+1):
    name = 'cum_ret' + str(ii)
    cols.append( name )

# Parse forecast specification: "name:multiplier" or "name:multiplier:weight"
# Examples: "qhl_intra:1" or "qhl_intra:1:0.6,badj_intra:1:0.4"
fcasts = list()
forecasts = list()
forecastargs = args.fcast.split(',')
for fcast in forecastargs:
    pair = fcast.split(":")
    fcasts.append(pair)
    forecasts.append(pair[1])

# Load 30-minute bar data from HDF5 cache
# MultiIndex: (iclose_ts, sid) where iclose_ts is bar close timestamp
pnl_df = load_cache(dateparser.parse(start), dateparser.parse(end), cols)
pnl_df['forecast'] = np.nan
pnl_df['forecast_abs'] = np.nan

# Calculate incremental bar volume (volume in this 30-minute bar)
# bvolume is cumulative intraday volume, bvolume_d is incremental (delta)
pnl_df['bvolume_d'] = pnl_df['bvolume'].groupby(level='sid').diff()
#pnl_df.loc[ pnl_df['bvolume_d'] < 0, 'bvolume_d'] = pnl_df['bvolume']

# Push data forward (shift -1) to get next bar's values
# Used to calculate incremental VWAP for the upcoming bar
pnl_df = push_data(pnl_df, 'bvolume_d')
pnl_df = push_data(pnl_df, 'bvwap')
pnl_df = push_data(pnl_df, 'bvolume')

# Calculate incremental VWAP for the next bar
# Formula: vwap_next = (cumulative_vwap_next * cumulative_volume_next - cumulative_vwap * cumulative_volume) / incremental_volume_next
# This gives the average price of trades executed during the next 30-minute bar
pnl_df['vwap_n'] = (pnl_df['bvwap_n'] * pnl_df['bvolume_n'] - pnl_df['bvwap'] * pnl_df['bvolume']) / pnl_df['bvolume_d_n']

# Calculate market returns for delta hedging
# Market return = cap-weighted average return
mkt_rets = pnl_df[ ['cum_ret1', 'mkt_cap', 'date'] ].dropna().groupby('date').apply(mkt_ret)

# Load intraday forecast/alpha signals
# Each forecast is stored in HDF5: {fcast[0]}/{fcast[1]}/mus.h5
# Examples: qhl_intra/1/mus.h5, badj_intra/1/mus.h5
for fcast in fcasts:
    mu_df = load_mus(fcast[0], fcast[1], start, end)
pnl_df = pd.merge(mu_df, pnl_df, how='left', left_index=True, right_index=True)

# Sanity check: Compare incremental VWAP vs. bar close price
# Large differences indicate data quality issues
maxpdiff = np.abs(pnl_df['bvwap_n'] - pnl_df['iclose']).idxmax()
print("VWAP Diff")
print(maxpdiff)
print(pnl_df[ [ 'ticker', 'vwap_n', 'iclose'] ].loc[ maxpdiff ])

# P&L aggregation buckets for stratified analysis
# Each bucket stores P&L by (stratification key) -> {date/time: pnl_value}
# Keys 0-5 represent horizons (0 bars, 1 bar, ..., 5 bars ahead)
# Key 'not' stores notional exposure
# Key 'delta' stores market exposure (for delta hedging)

# Day bucket: P&L by calendar date (YYYYMMDD)
day_bucket = {
    'delta': defaultdict(int),
    'not' : defaultdict(int),
    0 : defaultdict(int),
    1 : defaultdict(int),
    2 : defaultdict(int),
    3 : defaultdict(int),
    4 : defaultdict(int),
    5 : defaultdict(int)
}

# Month bucket: P&L by month (YYYYMM)
month_bucket = {
    'not' : defaultdict(int),
    0 : defaultdict(int),
    1 : defaultdict(int),
    2 : defaultdict(int),
    3 : defaultdict(int),
    4 : defaultdict(int),
    5 : defaultdict(int)
}

# Time bucket: P&L by time-of-day (HH:MM:SS)
# Reveals intraday patterns (e.g., 9:45 vs. 15:45 performance)
time_bucket = {
    'not' : defaultdict(int),
    0 : defaultdict(int),
    1 : defaultdict(int),
    2 : defaultdict(int),
    3 : defaultdict(int),
    4 : defaultdict(int),
    5 : defaultdict(int)
}

# Day-of-week bucket: P&L by weekday (0=Mon, 1=Tue, ..., 4=Fri)
dayofweek_bucket = {
    'not' : defaultdict(int),
    0 : defaultdict(int),
    1 : defaultdict(int),
    2 : defaultdict(int),
    3 : defaultdict(int),
    4 : defaultdict(int),
    5 : defaultdict(int)
}

# Win rate tracking
upnames = 0   # Count of profitable positions
downnames = 0  # Count of losing positions

# Conditioning variable buckets for decile analysis
# Stocks are ranked by conditioning variable (e.g., mkt_cap) into 10 deciles
# cond_bucket_day: P&L by decile (e.g., decile 9 = largest market caps)
# cond_bucket_not_day: Notional by decile
cond_bucket_day = defaultdict(int)
cond_bucket_not_day = defaultdict(int)
cond_avg = dict()

#fit_df = pd.DataFrame()

# Mix/combine multiple forecasts
# Each forecast is optionally standardized then summed
# Future enhancement: weighted combination using forecast[2] weight parameter
pnl_df[ 'forecast' ] = 0
for fcast in forecasts:
    #pnl_df.loc[ np.abs(pnl_df[fcast]) < .001, fcast ] = 0
    pnl_df[ fcast + '_adj' ] = pnl_df[ fcast ] #/ pnl_df[ fcast ].std()
    pnl_df[ 'forecast' ] += pnl_df[fcast + '_adj']
pnl_df['forecast_abs'] = np.abs(pnl_df['forecast'])

# Calculate volume constraints for position sizing
# Limit positions to 1% of bar volume to avoid excessive market impact
if 'bvolume' in pnl_df.columns:
    pnl_df['adj_vol'] = pnl_df[ 'bvolume_d_n' ] * .01  # 1% of next bar's incremental volume
else:
    print("WARNING: using tradable_volume instead of bvolume")
    # Fallback: use daily volume / 14 bars (approximate 30-min volume)
    pnl_df['adj_vol'] = 0.01 * pnl_df[ ['tradable_volume', 'tradable_med_volume_21_y']  ].min(axis=1) / 14.0

#zscore
#pnl_df['forecast'] = pnl_df['forecast'].groupby(level=0).transform(lambda x: (x - x.mean())/x.std())
#pnl_df['forecast'] = np.abs(pnl_df['forecast'])
#pnl_df['cur_ret'] = np.log(pnl_df['iclose']/pnl_df['bopen'])
#pnl_df['cdec'] = pnl_df['cur_ret'].rank()/float(len(pnl_df)) * 10
#pnl_df.loc[ np.abs(pnl_df['cur_ret']) > .05, 'forecast'] = 0

# Position sizing: shares = ALPHA_MULT * forecast
# Positive forecast -> long position, negative forecast -> short position
pnl_df['fill_shares'] = fill_shares = ALPHA_MULT * pnl_df['forecast']

# Apply volume constraints (1% of bar volume)
pnl_df['max_shares']  = pnl_df['adj_vol']
pnl_df['min_shares']  = -1 * pnl_df['adj_vol']

#max_adv = 0.0005
#pnl_df['max_shares']  = pnl_df['tradable_med_volume_21_y'] * max_adv
#pnl_df['min_shares']  = -1 * pnl_df['tradable_med_volume_21_y'] * max_adv

# Clip positions to volume limits
pnl_df['fill_shares'] = pnl_df[ ['max_shares', 'fill_shares'] ].min(axis=1)
pnl_df['fill_shares'] = pnl_df[ ['min_shares', 'fill_shares'] ].max(axis=1)

# Convert shares to dollar notional
pnl_df['notional'] = (pnl_df['fill_shares'] * pnl_df['iclose']).fillna(0)

# Apply maximum position size: $500K per stock
max_dollars = 5e5
pnl_df['notional'] = pnl_df['notional'].fillna(0).clip(-max_dollars, max_dollars)

# Calculate slippage cost: fixed basis points of notional
pnl_df['slip'] = np.abs(pnl_df['notional']) * slipbs

# Calculate fill price and forward returns
# Horizon 0: Return from fill price to EOD close (same bar)
# If vwap=True: fill at incremental VWAP, else fill at iclose (bar close)
pnl_df['cum_ret0'] = np.log( pnl_df['close'] / pnl_df['iclose'] )
if args.vwap:
    pnl_df['cum_ret0'] = np.log( pnl_df['close'] / pnl_df['vwap_n'] )

# Calculate P&L for each horizon
# day_pnl0: P&L from fill to EOD close (horizon 0)
# day_pnl1: P&L from fill to 1 bar ahead, etc.
# Formula: notional * (exp(cum_ret) - 1) - slippage
pnl_df['cum_ret_tot0'] = pnl_df['cum_ret0']
pnl_df['day_pnl0'] = pnl_df['notional'] * (np.exp(pnl_df['cum_ret_tot0']) - 1)
pnl_df['day_pnl0'] = pnl_df['day_pnl0'] - pnl_df['slip']

# Multi-horizon P&L calculation
# cum_ret_tot{h} = cum_ret0 (fill to close) + cum_ret{h} (close to h bars ahead)
for hh in range(1, horizon+1):
    pnl_df['cum_ret_tot' + str(hh)] = pnl_df['cum_ret0'] + pnl_df['cum_ret' + str(hh)]
#    pnl_df['cum_ret_tot' + str(hh)] = pnl_df['cum_ret' + str(hh)]
    pnl_df['day_pnl' + str(hh)] = pnl_df['notional'] * (np.exp(pnl_df['cum_ret_tot' + str(hh)]) - 1)
    pnl_df['day_pnl' + str(hh)] = pnl_df['day_pnl' + str(hh)] - pnl_df['slip']
    pnl_df = pnl_df.dropna(subset=['day_pnl' + str(hh)])

pnl_df = pnl_df.dropna(subset=['forecast', 'day_pnl0'])

# Main simulation loop: iterate over each 30-minute bar timestamp
# GroupBy 'iclose_ts' (bar close timestamp) to process all stocks at each time slice
fitlist = list()
it = 0
delta_sum = 0

for name, date_group in pnl_df.groupby(level='iclose_ts'):
#    print "Looking at {}".format(name)

    # Rank stocks into deciles by conditioning variable (e.g., market cap)
    # Decile 0 = smallest, decile 9 = largest
    date_group['decile'] = date_group[args.cond].rank()/float(len(date_group)) * 10
    date_group['decile'] = date_group['decile'].fillna(-1)
    date_group['decile'] = date_group['decile'].astype(int)

    # Print decile cutoffs on first iteration for reference
    if it == 0:
        print("Decile cutoffs")
        for dd in range(10):
            print("Decile {}: {}".format(dd, date_group[ date_group['decile'] == dd ][args.cond].max()))

    # Extract time stratification keys
    dayname = name.strftime("%Y%m%d")      # Calendar date
    monthname = name.strftime("%Y%m")      # Month
    timename = name.strftime("%H:%M:%S")   # Time-of-day
    weekdayname = name.weekday()           # Day-of-week (0=Mon, 4=Fri)

    # Calculate delta hedging P&L: net_notional * market_return
    # This is subtracted from raw P&L to get market-neutral P&L
    delta_pnl = date_group['notional'].sum() * mkt_rets[dateparser.parse(dayname)]
    delta_sum += delta_pnl

    # Aggregate P&L across all horizons for this bar
    for hh in range(0, horizon+1):
        pnlname = 'day_pnl' + str(hh)
        day_pnl = date_group[pnlname]
        daysum = day_pnl.sum() - delta_pnl  # Market-neutral P&L

        # Store P&L in stratification buckets
        day_bucket[hh][dayname] += daysum
        month_bucket[hh][monthname] += daysum
        time_bucket[hh][timename] += daysum
        dayofweek_bucket[hh][weekdayname] += daysum

        # Track win rate at final horizon
        if hh == horizon:
            upnames += len(day_pnl[ day_pnl > 0 ])
            downnames += len(day_pnl[ day_pnl < 0 ])

    # Track notional exposure
    absnotional = np.abs(date_group['notional'].fillna(0)).sum()
    day_bucket['not'][dayname] += absnotional
    month_bucket['not'][monthname] += absnotional
    time_bucket['not'][timename] += absnotional
    dayofweek_bucket['not'][weekdayname] += absnotional

    # Track delta (net market exposure)
    day_bucket['delta'][dayname] += date_group['notional'].sum()

    # Conditioning variable decile analysis
    # Aggregate P&L and notional by decile to identify which segments perform best
    if args.cond is not None:
        condret = 'day_pnl'+str(args.horizon)
        for ii in range (-1,10):
            amt = date_group[ date_group['decile'] == ii][condret].dropna().sum()
            cond_bucket_day[ii] += amt
            cond_bucket_not_day[ii] += np.abs(date_group[ date_group['decile'] == ii]['notional'].dropna()).sum()

        cond_avg[name.strftime("%Y%m%d")] = date_group[args.cond].mean()
    it += 1

# Debug output: export one stock's full history for inspection
pnl_df.xs(testid, level=1).to_csv("debug.csv")
print("Delta Sum {}".format(delta_sum))

print()
print()

# ============================================================================
# DIAGNOSTICS: Forecast analysis and visualization
# ============================================================================

# Forecast correlation matrix: check if forecasts are independent
print("Forecast correlations...")
print(pnl_df[ forecasts ].corr())
print()

# Forecast strength over time: plot standard deviation of alphas by bar
# Declining strength may indicate alpha decay
print("Forecast strength...")
plt.figure()
print(pnl_df[ forecasts ].groupby(level='iclose_ts').std().plot())
plt.savefig("forecast_strength.png")
print()

# Alpha distribution histograms: visualize forecast distributions
print("Generating Total Alpha histogram...")
for forecast in forecasts:
    print("Looking at forecast: {} ".format(forecast))
    fig1 = plt.figure()
    fig1.canvas.set_window_title("Histogram")
    pnl_df[ forecast ].dropna().hist(bins=100)
    plt.savefig(forecast + "__hist.png")
    print(pnl_df[forecast].describe())
print()

# Per-stock P&L distribution: identify outliers and concentration risk
pnlbystock = pnl_df.groupby(level='sid')['day_pnl1'].sum()
plt.figure()
pnlbystock.hist(bins=1800)
plt.savefig("stocks.png")
maxid = pnlbystock.idxmax()

# Drill into highest P&L stock to check for anomalies
print("Max pnl stock pnl distribution: {}".format(pnlbystock.loc[ maxid ]))
plt.figure()
maxstock_df = pnl_df.xs(maxid, level=1)
maxstock_df['day_pnl1'].hist(bins=100)
plt.savefig("maxstock.png")
maxpnlid = maxstock_df['day_pnl1'].idxmax()
#print maxstock_df.xs(maxpnlid)
print 

# Notional bias analysis: check if portfolio is balanced long/short
# Large biases indicate directional exposure (not market-neutral)
longs = pnl_df[ pnl_df['notional'] > 0 ]['notional'].groupby(level='iclose_ts').sum()
shorts = np.abs(pnl_df[ pnl_df['notional'] < 0 ]['notional'].groupby(level='iclose_ts').sum())
nots = longs - shorts
plt.figure()
nots.plot()
plt.savefig("notional_bias.png")
notbiasmax_idx = nots.idxmax()
print("Maximum Notional bias on {}".format(notbiasmax_idx))
print("Bias: {}, Long: {}, Short: {}".format(nots.loc[ notbiasmax_idx ], longs.loc[ notbiasmax_idx ], shorts.loc[ notbiasmax_idx ]))
plt.figure()
pnl_df.xs(notbiasmax_idx, level=0)['notional'].hist(bins=100)
pnl_df.xs(notbiasmax_idx, level=0).to_csv("max_notional_day.csv")
plt.savefig("maxnotional")
print()

# Alpha bias analysis: ratio of positive to negative forecasts
# Balanced alphas should have ~1:1 ratio (50% long, 50% short)
pos = pnl_df[ pnl_df['forecast'] > 0 ].groupby(level='iclose_ts')['forecast'].count()
neg = pnl_df[ pnl_df['forecast'] < 0 ].groupby(level='iclose_ts')['forecast'].count()
ratio = pos.astype(float)/neg.astype(float)
plt.figure()
ratio.plot()
plt.savefig("alpha_bias.png")
maxalpha_idx = ratio.idxmax()
print("Maximum Alpha bias on {} of {}".format(maxalpha_idx, ratio.loc[ maxalpha_idx ]))
plt.figure()
pnl_df.xs(maxalpha_idx, level=0)['forecast'].hist(bins=100)
plt.savefig("maxalphabias.png")
print()

# Free memory: done with full DataFrame
pnl_df = None

# ============================================================================
# PERFORMANCE ANALYSIS: Calculate returns and Sharpe ratios by horizon
# ============================================================================

for ii in range(horizon+1):
    print("Running horizon " + str(ii))
    #    pnl_df = pnl_df.dropna(subset=['cum_ret' + str(ii), 'forecast'])
    #    results_ols = sm.OLS(pnl_df['cum_ret' + str(ii)], sm.add_constant(pnl_df['forecast'])).fit()
    #    print results_ols.summary()

    # Build daily notional series
    nots = pd.DataFrame([ [datetime.strptime(d,'%Y%m%d'),v] for d, v in sorted(day_bucket['not'].items()) ], columns=['date', 'notional'])
    nots.set_index(keys=['date'], inplace=True)

    plt.figure()
    nots['notional'].plot()
    plt.savefig("notional.png")

    # Build daily P&L series for this horizon
    rets = pd.DataFrame([ [datetime.strptime(d,'%Y%m%d'),v] for d, v in sorted(day_bucket[ii].items()) ], columns=['date', 'pnl'])
    rets.set_index(keys=['date'], inplace=True)

    rets = pd.merge(rets, nots, left_index=True, right_index=True)
    print("Total Pnl: ${:.0f}K".format(rets['pnl'].sum()/1000.0))

    # Normalize P&L by horizon length (so different horizons are comparable)
    if ii > 0:
        rets['pnl'] = rets['pnl'] / ii

    # Calculate daily returns: pnl / notional
    rets['day_rets'] = rets['pnl'] / rets['notional']
    rets['day_rets'].replace([np.inf, -np.inf], np.nan, inplace=True)
    rets['day_rets'].fillna(0, inplace=True)

    # Cumulative returns curve
    rets['cum_ret'] = (1 + rets['day_rets']).dropna().cumprod()

    # Plot cumulative returns with optional conditioning variable overlay
    plt.figure()
    if args.cond is not None:
        conds = pd.DataFrame([ [datetime.strptime(d,'%Y%m%d'),v] for d, v in sorted(cond_avg.items()) ], columns=['date', 'cond'])
        conds.set_index(keys=['date'], inplace=True)
        rets[ args.cond ] = conds['cond']
        rets[ 'cum_ret' ].plot(legend=True)
        rets[ args.cond ].plot(secondary_y=True, legend=True)
    else:
        rets['cum_ret'].plot()

    plt.draw()
    plt.savefig("rets." + str(ii) + "." + ".".join(forecasts) + ".png")

    # Annualized statistics (assuming 252 trading days)
    # Note: QSIM has 13 bars/day, but we aggregate to daily returns for Sharpe
    mean = rets['day_rets'].mean() * 252
    std = rets['day_rets'].std() * math.sqrt(252)

    sharpe =  mean/std
    print("Day " + str(ii) + " mean: {:.4f} std: {:.4f} sharpe: {:.4f} avg Notional: ${:.0f}K".format(mean, std, sharpe, rets['notional'].mean()/1000.0))
    print

# ============================================================================
# STRATIFICATION ANALYSIS: Breakdown performance by various dimensions
# ============================================================================

# Conditioning variable (decile) breakdown
# Shows which market cap / volume / other deciles perform best
if args.cond is not None:
    print("Cond {}  breakdown Bps".format(args.cond))
    totnot = 0
    for k, v in cond_bucket_not_day.items():
        totnot += v

    for dec in sorted(cond_bucket_day.keys()):
        notional = cond_bucket_not_day[dec] / 10000.0
        if notional > 0:
            print("Decile {}: {:.4f} {:.4f} {:.4f} {:.2f}%".format(dec, cond_bucket_day[dec]/notional, cond_bucket_day[dec]/notional, cond_bucket_day[dec]/notional, 100.0 * cond_bucket_not_day[dec]/totnot))
    print

# Monthly breakdown: identify seasonal patterns
print("Month breakdown Bps")
for month in sorted(month_bucket['not'].keys()):
    notional = month_bucket['not'][month] / 10000.0
    if notional > 0:
        # Shows P&L in bps for horizons 0, 1, 2, 3, 5
        print("Month {}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(month, month_bucket[0][month]/notional, month_bucket[1][month]/notional, month_bucket[2][month]/notional, month_bucket[3][month]/notional, month_bucket[5][month]/notional))
print()

# Time-of-day breakdown: critical for intraday strategies
# Identifies which 30-minute bars have best/worst performance
print("Time breakdown Bps")
for time in sorted(time_bucket['not'].keys()):
    notional = time_bucket['not'][time] / 10000.0
    if notional > 0:
        print("Time {}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(time, time_bucket[0][time]/notional, time_bucket[1][time]/notional, time_bucket[2][time]/notional, time_bucket[3][time]/notional, time_bucket[5][time]/notional))
print()

# Day-of-week breakdown: Monday effect, Friday effect, etc.
print("Dayofweek breakdown Bps")
for dayofweek in sorted(dayofweek_bucket['not'].keys()):
    notional = dayofweek_bucket['not'][dayofweek] / 10000.0
    if notional > 0:
        print("Dayofweek {}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(dayofweek, dayofweek_bucket[0][dayofweek]/notional, dayofweek_bucket[1][dayofweek]/notional, dayofweek_bucket[2][dayofweek]/notional, dayofweek_bucket[3][dayofweek]/notional, dayofweek_bucket[5][dayofweek]/notional))
print()

# Win rate: percentage of positions that were profitable
print("Up %: {:.4f}".format(float(upnames)/(upnames+downnames)))


