"""
SALAMANDER QSIM - Standalone Intraday Simulation Engine (Python 3)

This is the Python 3 standalone version of the intraday (quote-level) backtesting
simulation. It evaluates alpha signals at 30-minute bar intervals throughout the
trading day, tracking multi-horizon forward returns and performance attribution
across time-of-day, day-of-week, and conditioning deciles.

Differences from Main qsim.py:
==============================

1. **Python 3 Compatibility**
   - Print statements use function syntax: print() not print
   - Dictionary methods: .items() instead of .iteritems()
   - String formatting uses .format() instead of %
   - Integer division uses // where needed

2. **Simplified Data Pipeline**
   - Uses standalone loaddata.py for HDF5/CSV loading
   - Loads from --dir parameter (data stored in {dir}/data/)
   - Expects 'date' and 'gvkey' index (not 'iclose_ts' and 'sid')
   - No intraday timestamp filtering (data is pre-processed)

3. **Reduced Column Set**
   - Core data: symbol, volume, close, mdvp (median daily volume $)
   - Returns: log_ret, overnight_log_ret, cum_ret{1-5}
   - Market data: mkt_cap (for market neutralization)
   - No Barra factors (main version uses 13 factors)
   - No industry classifications (main has 58 industries)

4. **Alpha Loading Format**
   - Format: "name:multiplier:weight"
   - Example: "qhl:1:1" or "qhl:1:0.6,badj:1:0.4"
   - Loads from data/{name}/alpha.{name}.{start}-{end}.csv
   - Main version: loads from directories with more complex paths

5. **Position Sizing**
   - Simple multiplier: shares = ALPHA_MULT * forecast
   - Capped at 1% of bar volume: max_shares = 0.01 * volume
   - Max notional: $500K per position (main: $4M)
   - No ADV-based constraints (main: 2% of 21-day ADV)

6. **Execution Model**
   - Fixed slippage: cost = abs(notional) * slipbps
   - Default: 1 basis point (0.0001)
   - Optional VWAP fill: uses incremental vwap_n calculation
   - Default: fill at iclose (close of bar)

7. **Multi-Horizon Analysis**
   - Tracks P&L from entry (bar 0) to horizon (default: 3 bars = 90 min)
   - Computes cum_ret0, cum_ret1, ..., cum_ret{horizon}
   - Day P&L: notional * (exp(cum_ret_tot{h}) - 1) - slippage
   - Separate Sharpe ratio for each horizon

8. **Market Neutralization**
   - Delta hedging: adjusts for market return component
   - Uses mkt_cap weighted market return: mkt_ret()
   - delta_pnl = notional * mkt_ret (subtracted from P&L)
   - Main version has more sophisticated hedging

9. **Performance Attribution**
   - Time-of-day buckets: 09:45, 10:00, ..., 15:45, 16:00
   - Day-of-week buckets: 0 (Mon) to 4 (Fri)
   - Month buckets: YYYYMM format
   - Conditioning deciles: ranks by --cond variable (default: mkt_cap)

10. **Visualization**
    - Cumulative return plots for each horizon
    - Forecast strength over time
    - Alpha distribution histograms
    - Stock-level P&L distributions
    - Notional bias (long vs short) over time
    - Alpha bias (positive vs negative signals)

Workflow:
=========

1. Load intraday bar data (30-min bars) from HDF5 cache
2. Load alpha forecasts for specified strategies
3. Compute incremental VWAP: vwap_n = (vwap * vol - vwap_old * vol_old) / vol_new
4. Mix multiple forecasts with weights
5. Size positions: shares = ALPHA_MULT * forecast
6. Apply participation constraints and max position limits
7. For each bar:
   a. Calculate notional = shares * close
   b. Apply slippage cost
   c. Compute forward returns from cum_ret0 to cum_ret{horizon}
   d. Track P&L for each horizon
   e. Aggregate into time/day/month/decile buckets
8. Calculate Sharpe ratios for each horizon
9. Generate performance attribution reports and plots

Command-Line Arguments:
=======================

Required:
  --start         Start date (YYYYMMDD format)
  --end           End date (YYYYMMDD format)
  --fcast         Alpha signals (format: "name:mult:weight,name2:mult2:weight2,...")
                  Example: "qhl:1:1" or "qhl:1:0.6,badj:1:0.4"
  --dir           Root directory (data in {dir}/data/, default: '.')

Optional:
  --horizon       Forecast horizon in 30-min bars (default: 3 = 90 minutes)
  --mult          Position sizing multiplier (default: 1000.0)
  --slipbps       Slippage in basis points (default: 0.0001 = 1bp)
  --vwap          Use incremental VWAP fill (default: False = use iclose)
  --cond          Conditioning variable for decile analysis (default: "mkt_cap")

Performance Metrics:
====================

For each horizon (0 to --horizon bars):
  - Total P&L: Sum of all position P&L
  - Annualized Return: Mean daily return * 252
  - Annualized Volatility: Std daily return * sqrt(252)
  - Sharpe Ratio: Return / Volatility
  - Average Notional: Mean absolute position size

Attribution Breakdowns:
  - Time-of-day: P&L and notional by intraday time slot
  - Day-of-week: P&L and notional by weekday
  - Month: P&L and notional by calendar month
  - Conditioning deciles: P&L by deciles of --cond variable

Diagnostic Outputs:
  - Forecast correlations: Correlation matrix of alpha signals
  - Forecast strength: Time series of forecast volatility
  - Alpha histograms: Distribution of alpha values
  - Stock P&L: Distribution of per-stock cumulative P&L
  - Notional bias: Net long/short exposure over time
  - Alpha bias: Ratio of positive to negative signals
  - Win rate: Percentage of profitable positions

Examples:
=========

# Single intraday forecast, 3-bar horizon (90 minutes)
python3 salamander/qsim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --dir=/path/to/workspace \\
  --fcast=qhl:1:1 \\
  --horizon=3

# Multi-forecast with VWAP execution
python3 salamander/qsim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --dir=/path/to/workspace \\
  --fcast=qhl:1:0.6,badj:1:0.4 \\
  --vwap=True

# Analyze by volume deciles with 5-bar horizon
python3 salamander/qsim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --dir=/path/to/workspace \\
  --fcast=qhl:1:1 \\
  --cond=volume \\
  --horizon=5

# Aggressive multiplier with higher slippage
python3 salamander/qsim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --dir=/path/to/workspace \\
  --fcast=qhl:1:1 \\
  --mult=2000 \\
  --slipbps=0.0005

Data Requirements:
==================

1. Processed intraday data files:
   python3 salamander/gen_hl.py \\
     --start=20100630 \\
     --end=20130630 \\
     --dir=/path/to/workspace

2. Intraday alpha signals:
   python3 salamander/gen_alpha.py \\
     --start=20130101 \\
     --end=20130630 \\
     --dir=/path/to/workspace

3. Data structure:
   {dir}/data/all/all.{start}-{end}.h5
   {dir}/data/{alpha}/alpha.{name}.{start}-{end}.csv

Output Files:
=============

Plots (PNG):
  - rets.{horizon}.{forecasts}.png: Cumulative return curves
  - forecast_strength.png: Forecast volatility time series
  - {forecast}__hist.png: Alpha distribution histogram
  - stocks.png: Per-stock P&L distribution
  - maxstock.png: Top P&L stock's daily returns
  - notional_bias.png: Net long/short over time
  - alpha_bias.png: Positive/negative signal ratio
  - notional.png: Total notional deployed

CSV Files:
  - debug.csv: Detailed single-stock trace (testid)
  - max_notional_day.csv: Day with maximum notional bias

Notes:
======

- Python 3.6+ required
- Designed for intraday strategy research and optimization
- Memory efficient: processes in chunks
- Typical runtime: 10-30 minutes for 6-month backtest
- Best for strategies with 1-5 bar (30min-2.5hr) horizons
- No live trading support (research only)
"""

from util import *
from regress import *
from loaddata import *
from collections import defaultdict
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", help="the starting date to generate alpha signals, formatted as 'YYYYMMdd'")
parser.add_argument("--end", help="the end date to generate alpha signals, formatted as 'YYYYMMdd'")
parser.add_argument("--fcast", help="alpha signals, formatted as 'name1:mult1:weight1,name2:mult2:weight2,...'")
parser.add_argument("--cond", default="mkt_cap")
parser.add_argument("--horizon", default=3)
parser.add_argument("--mult", default=1000.0)
parser.add_argument("--slipbps", default=0.0001)
parser.add_argument("--vwap", default=False)
parser.add_argument("--dir", help="the root directory", default='.')
args = parser.parse_args()

ALPHA_MULT = float(args.mult)
horizon = int(args.horizon)
slipbs = float(args.slipbps)
start = args.start
end = args.end
data_dir = args.dir + "/data"

# cols = ['ticker', 'iclose', 'tradable_volume', 'close', 'bvwap', 'tradable_med_volume_21_y', 'mdvp', 'overnight_log_ret', 'date', 'log_ret', 'bvolume', 'mkt_cap']
cols = ['symbol', 'volume', 'close', 'mdvp', 'overnight_log_ret', 'log_ret', 'mkt_cap']  # new
if args.cond not in cols:
    cols.append(args.cond)
for ii in range(1, horizon + 1):
    name = 'cum_ret' + str(ii)
    cols.append(name)

forecasts = []
forecastargs = args.fcast.split(',')
for fcast in forecastargs:
    name, mult, weight = fcast.split(":")
    forecasts.append(name)

pnl_df = load_cache(dateparser.parse(start), dateparser.parse(end), data_dir, cols)
pnl_df['forecast'] = np.nan
pnl_df['forecast_abs'] = np.nan

pnl_df['volume_d'] = pnl_df['volume'].groupby(level='gvkey').diff()
# pnl_df.loc[ pnl_df['bvolume_d'] < 0, 'bvolume_d'] = pnl_df['bvolume']
pnl_df = push_data(pnl_df, 'volume_d')
pnl_df = push_data(pnl_df, 'close')
pnl_df = push_data(pnl_df, 'volume')

# vwap_new = (vwap * volume_tot - vwap_old / volume_old) / volume_new
pnl_df['close_n'] = (pnl_df['close_n'] * pnl_df['volume_n'] - pnl_df['close'] * pnl_df['volume']) / pnl_df['volume_d_n']
mkt_rets = pnl_df[['cum_ret1', 'mkt_cap']].dropna().groupby('date').apply(mkt_ret)

for fcast in forecasts:
    mu_df = load_mus(data_dir, fcast, start, end)
    pnl_df = pd.merge(mu_df, pnl_df, how='left', on=['date', 'gvkey'])

# check vwap diff
maxpdiff = np.abs(pnl_df['close_n'] - pnl_df['close']).idxmax()

print("VWAP Diff")
print(maxpdiff)
# print(pnl_df[[ 'ticker', 'vwap_n', 'close']].loc[ maxpdiff ])
print(pnl_df[['symbol', 'close_n', 'close']].loc[maxpdiff])  # new
day_bucket = {
    'delta': defaultdict(int),
    'not': defaultdict(int),
    0: defaultdict(int),
    1: defaultdict(int),
    2: defaultdict(int),
    3: defaultdict(int),
    4: defaultdict(int),
    5: defaultdict(int)
}

month_bucket = {
    'not': defaultdict(int),
    0: defaultdict(int),
    1: defaultdict(int),
    2: defaultdict(int),
    3: defaultdict(int),
    4: defaultdict(int),
    5: defaultdict(int)
}

time_bucket = {
    'not': defaultdict(int),
    0: defaultdict(int),
    1: defaultdict(int),
    2: defaultdict(int),
    3: defaultdict(int),
    4: defaultdict(int),
    5: defaultdict(int)
}

dayofweek_bucket = {
    'not': defaultdict(int),
    0: defaultdict(int),
    1: defaultdict(int),
    2: defaultdict(int),
    3: defaultdict(int),
    4: defaultdict(int),
    5: defaultdict(int)
}

upnames = 0
downnames = 0

cond_bucket_day = defaultdict(int)
cond_bucket_not_day = defaultdict(int)
cond_avg = dict()

# fit_df = pd.DataFrame()

# MIX FORECASTS
pnl_df['forecast'] = 0
for fcast in forecasts:
    # pnl_df.loc[ np.abs(pnl_df[fcast]) < .001, fcast ] = 0
    pnl_df[fcast + '_adj'] = pnl_df[fcast]  # / pnl_df[ fcast ].std()
    pnl_df['forecast'] += pnl_df[fcast + '_adj']
pnl_df['forecast_abs'] = np.abs(pnl_df['forecast'])

if 'volume' in pnl_df.columns:
    pnl_df['adj_vol'] = pnl_df['volume'] * .01
else:
    print("WARNING: using tradable_volume instead of volume")
    pnl_df['adj_vol'] = 0.01 * pnl_df[['tradable_volume', 'tradable_med_volume_21_y']].min(axis=1) / 14.0

# zscore
# pnl_df['forecast'] = pnl_df['forecast'].groupby(level=0).transform(lambda x: (x - x.mean())/x.std())
# pnl_df['forecast'] = np.abs(pnl_df['forecast'])
# pnl_df['cur_ret'] = np.log(pnl_df['iclose']/pnl_df['bopen'])
# pnl_df['cdec'] = pnl_df['cur_ret'].rank()/float(len(pnl_df)) * 10
# pnl_df.loc[ np.abs(pnl_df['cur_ret']) > .05, 'forecast'] = 0

pnl_df['fill_shares'] = fill_shares = ALPHA_MULT * pnl_df['forecast']

pnl_df['max_shares'] = pnl_df['adj_vol']
pnl_df['min_shares'] = -1 * pnl_df['adj_vol']

# max_adv = 0.0005
# pnl_df['max_shares']  = pnl_df['tradable_med_volume_21_y'] * max_adv
# pnl_df['min_shares']  = -1 * pnl_df['tradable_med_volume_21_y'] * max_adv

pnl_df['fill_shares'] = pnl_df[['max_shares', 'fill_shares']].min(axis=1)
pnl_df['fill_shares'] = pnl_df[['min_shares', 'fill_shares']].max(axis=1)
pnl_df['notional'] = (pnl_df['fill_shares'] * pnl_df['close']).fillna(0)
max_dollars = 5e5
pnl_df['notional'] = pnl_df['notional'].fillna(0).clip(-max_dollars, max_dollars)
pnl_df['slip'] = np.abs(pnl_df['notional']) * slipbs

# set "fill" price
pnl_df['cum_ret0'] = np.log(pnl_df['close'] / pnl_df['close'])
if args.vwap:
    pnl_df['cum_ret0'] = np.log(pnl_df['close'] / pnl_df['vwap_n'])

pnl_df['cum_ret_tot0'] = pnl_df['cum_ret0']
pnl_df['day_pnl0'] = pnl_df['notional'] * (np.exp(pnl_df['cum_ret_tot0']) - 1)
pnl_df['day_pnl0'] = pnl_df['day_pnl0'] - pnl_df['slip']
for hh in range(1, horizon + 1):
    pnl_df['cum_ret_tot' + str(hh)] = pnl_df['cum_ret0'] + pnl_df['cum_ret' + str(hh)]
    #    pnl_df['cum_ret_tot' + str(hh)] = pnl_df['cum_ret' + str(hh)]
    pnl_df['day_pnl' + str(hh)] = pnl_df['notional'] * (np.exp(pnl_df['cum_ret_tot' + str(hh)]) - 1)
    pnl_df['day_pnl' + str(hh)] = pnl_df['day_pnl' + str(hh)] - pnl_df['slip']
    pnl_df = pnl_df.dropna(subset=['day_pnl' + str(hh)])

pnl_df = pnl_df.dropna(subset=['forecast', 'day_pnl0'])

fitlist = []
it = 0
delta_sum = 0
for name, date_group_shallow in pnl_df.groupby(level='date'):
    #    print("Looking at {}".format(name))
    date_group=copy.copy(date_group_shallow)
    date_group['decile'] = date_group[args.cond].rank() / float(len(date_group)) * 10
    date_group['decile'] = date_group['decile'].fillna(-1).astype(int)
    if it == 0:
        print("Decile cutoffs")
        for dd in range(10):
            print("Decile {}: {}".format(dd, date_group[date_group['decile'] == dd][args.cond].max()))

    dayname = name.strftime("%Y%m%d")
    monthname = name.strftime("%Y%m")
    timename = name.strftime("%H:%M:%S")
    weekdayname = name.weekday()
    delta_pnl = date_group['notional'].sum() * mkt_rets[dateparser.parse(dayname)]
    delta_sum += delta_pnl

    # CALCULATE PNLS
    for hh in range(0, horizon + 1):
        pnlname = 'day_pnl' + str(hh)
        day_pnl = date_group[pnlname]
        daysum = day_pnl.sum() - delta_pnl

        day_bucket[hh][dayname] += daysum
        month_bucket[hh][monthname] += daysum
        time_bucket[hh][timename] += daysum
        dayofweek_bucket[hh][weekdayname] += daysum
        if hh == horizon:
            upnames += len(day_pnl[day_pnl > 0])
            downnames += len(day_pnl[day_pnl < 0])

    absnotional = np.abs(date_group['notional'].fillna(0)).sum()
    day_bucket['not'][dayname] += absnotional
    month_bucket['not'][monthname] += absnotional
    time_bucket['not'][timename] += absnotional
    dayofweek_bucket['not'][weekdayname] += absnotional

    day_bucket['delta'][dayname] += date_group['notional'].sum()

    # CALCULATE CONDITIONAL DECILES
    # 9 is the highesto
    if args.cond is not None:
        condret = 'day_pnl' + str(args.horizon)
        for ii in range(-1, 10):
            amt = date_group[date_group['decile'] == ii][condret].dropna().sum()
            cond_bucket_day[ii] += amt
            cond_bucket_not_day[ii] += np.abs(date_group[date_group['decile'] == ii]['notional'].dropna()).sum()

        cond_avg[name.strftime("%Y%m%d")] = date_group[args.cond].mean()
    it += 1

pnl_df.xs(testid, level=1).to_csv("debug.csv")
print("Delta Sum {}".format(delta_sum))

print()
print()
print("Forecast correlations...")
print(pnl_df[forecasts].corr())
print()

print("Forecast strength...")
plt.figure()
print(pnl_df[forecasts].groupby(level='date').std().plot())
plt.savefig("forecast_strength.png")
print()

print("Generating Total Alpha histogram...")
for forecast in forecasts:
    print("Looking at forecast: {} ".format(forecast))
    fig1 = plt.figure()
    fig1.canvas.set_window_title("Histogram")
    pnl_df[forecast].dropna().hist(bins=100)
    plt.savefig(forecast + "__hist.png")
    print(pnl_df[forecast].describe())
print()

pnlbystock = pnl_df.groupby(level='gvkey')['day_pnl1'].sum()
plt.figure()
pnlbystock.hist(bins=1800)
plt.savefig("stocks.png")
maxid = pnlbystock.idxmax()

print("Max pnl stock pnl distribution: {}".format(pnlbystock.loc[maxid]))
plt.figure()
maxstock_df = pnl_df.xs(maxid, level=1)
maxstock_df['day_pnl1'].hist(bins=100)
plt.savefig("maxstock.png")
maxpnlid = maxstock_df['day_pnl1'].idxmax()
# print(maxstock_df.xs(maxpnlid))
print()

longs = pnl_df[pnl_df['notional'] > 0]['notional'].groupby(level='date').sum()
shorts = np.abs(pnl_df[pnl_df['notional'] < 0]['notional'].groupby(level='date').sum())
nots = longs - shorts
plt.figure()
nots.plot()
plt.savefig("notional_bias.png")
notbiasmax_idx = nots.idxmax()
print("Maximum Notional bias on {}".format(notbiasmax_idx))
print("Bias: {}, Long: {}, Short: {}".format(nots.loc[notbiasmax_idx], longs.loc[notbiasmax_idx],
                                             shorts.loc[notbiasmax_idx]))
plt.figure()
pnl_df.xs(notbiasmax_idx, level=0)['notional'].hist(bins=100)
pnl_df.xs(notbiasmax_idx, level=0).to_csv("max_notional_day.csv")
plt.savefig("maxnotional")
print()

pos = pnl_df[pnl_df['forecast'] > 0].groupby(level='date')['forecast'].count()
neg = pnl_df[pnl_df['forecast'] < 0].groupby(level='date')['forecast'].count()
ratio = pos.astype(float) / neg.astype(float)
plt.figure()
ratio.plot()
plt.savefig("alpha_bias.png")
maxalpha_idx = ratio.idxmax()
print("Maximum Alpha bias on {} of {}".format(maxalpha_idx, ratio.loc[maxalpha_idx]))
plt.figure()
pnl_df.xs(maxalpha_idx, level=0)['forecast'].hist(bins=100)
plt.savefig("maxalphabias.png")
print()

pnl_df = None

for ii in range(horizon + 1):
    print("Running horizon " + str(ii))
    #    pnl_df = pnl_df.dropna(subset=['cum_ret' + str(ii), 'forecast'])
    #    results_ols = sm.OLS(pnl_df['cum_ret' + str(ii)], sm.add_constant(pnl_df['forecast'])).fit()
    #    printresults_ols.summary()
    nots = pd.DataFrame([[datetime.strptime(d, '%Y%m%d'), v] for d, v in sorted(day_bucket['not'].items())],
                        columns=['date', 'notional'])
    nots.set_index(keys=['date'], inplace=True)

    plt.figure()
    nots['notional'].plot()
    plt.savefig("notional.png")

    rets = pd.DataFrame([[datetime.strptime(d, '%Y%m%d'), v] for d, v in sorted(day_bucket[ii].items())],
                        columns=['date', 'pnl'])
    rets.set_index(keys=['date'], inplace=True)

    rets = pd.merge(rets, nots, left_index=True, right_index=True)
    print("Total Pnl: ${:.0f}K".format(rets['pnl'].sum() / 1000.0))

    if ii > 0:
        rets['pnl'] = rets['pnl'] / ii
    rets['day_rets'] = rets['pnl'] / rets['notional']
    rets['day_rets'].replace([np.inf, -np.inf], np.nan, inplace=True)
    rets['day_rets'].fillna(0, inplace=True)
    rets['cum_ret'] = (1 + rets['day_rets']).dropna().cumprod()

    plt.figure()
    if args.cond is not None:
        conds = pd.DataFrame([[datetime.strptime(d, '%Y%m%d'), v] for d, v in sorted(cond_avg.items())],
                             columns=['date', 'cond'])
        conds.set_index(keys=['date'], inplace=True)
        rets[args.cond] = conds['cond']
        rets['cum_ret'].plot(legend=True)
        rets[args.cond].plot(secondary_y=True, legend=True)
    else:
        rets['cum_ret'].plot()

    plt.draw()
    plt.savefig("rets." + str(ii) + "." + ".".join(forecasts) + ".png")

    mean = rets['day_rets'].mean() * 252
    std = rets['day_rets'].std() * math.sqrt(252)

    sharpe = mean / std
    print("Day " + str(ii) + " mean: {:.4f} std: {:.4f} sharpe: {:.4f} avg Notional: ${:.0f}K".format(mean, std, sharpe,
                                                                                                      rets[
                                                                                                          'notional'].mean() / 1000.0))
    print()

if args.cond is not None:
    print("Cond {}  breakdown Bps".format(args.cond))
    totnot = 0
    for k, v in cond_bucket_not_day.items():
        totnot += v

    for dec in sorted(cond_bucket_day.keys()):
        notional = cond_bucket_not_day[dec] / 10000.0
        if notional > 0:
            print("Decile {}: {:.4f} {:.4f} {:.4f} {:.2f}%".format(dec, cond_bucket_day[dec] / notional,
                                                                   cond_bucket_day[dec] / notional,
                                                                   cond_bucket_day[dec] / notional,
                                                                   100.0 * cond_bucket_not_day[dec] / totnot))
    print()

print("Month breakdown Bps")
for month in sorted(month_bucket['not'].keys()):
    notional = month_bucket['not'][month] / 10000.0
    if notional > 0:
        print("Month {}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(month, month_bucket[0][month] / notional,
                                                                    month_bucket[1][month] / notional,
                                                                    month_bucket[2][month] / notional,
                                                                    month_bucket[3][month] / notional,
                                                                    month_bucket[5][month] / notional))
print()

print("Time breakdown Bps")
for time in sorted(time_bucket['not'].keys()):
    notional = time_bucket['not'][time] / 10000.0
    if notional > 0:
        print("Time {}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(time, time_bucket[0][time] / notional,
                                                                   time_bucket[1][time] / notional,
                                                                   time_bucket[2][time] / notional,
                                                                   time_bucket[3][time] / notional,
                                                                   time_bucket[5][time] / notional))
print()

print("Dayofweek breakdown Bps")
for dayofweek in sorted(dayofweek_bucket['not'].keys()):
    notional = dayofweek_bucket['not'][dayofweek] / 10000.0
    if notional > 0:
        print("Dayofweek {}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(dayofweek,
                                                                        dayofweek_bucket[0][dayofweek] / notional,
                                                                        dayofweek_bucket[1][dayofweek] / notional,
                                                                        dayofweek_bucket[2][dayofweek] / notional,
                                                                        dayofweek_bucket[3][dayofweek] / notional,
                                                                        dayofweek_bucket[5][dayofweek] / notional))
print()

print("Up %: {:.4f}".format(float(upnames) / (upnames + downnames)))
