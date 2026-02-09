#!/usr/bin/env python
"""
BSIM - Big Simulation Engine

This is the main backtesting simulation engine for daily statistical arbitrage
trading strategies. BSIM combines multiple alpha signals, runs portfolio
optimization at each timestep, and simulates realistic trading with transaction
costs, risk constraints, and position limits.

Workflow:
    1. Load historical price/volume data and alpha forecasts for date range
    2. Merge multiple alpha signals with configurable weights
    3. Load Barra factor exposures and covariance matrix
    4. Calculate residual volatility for specific risk
    5. Main simulation loop (for each timestamp):
       a. Filter tradable universe (price, volume, data quality checks)
       b. Merge last positions and apply corporate actions (splits)
       c. Apply constraints (market cap filters, earnings avoidance, locates)
       d. Set position bounds (max_notional, min_notional per stock)
       e. Call optimizer to find optimal positions maximizing utility
       f. Apply participation constraints to limit trading rate
       g. Calculate actual positions and update last_pos tracker
       h. Write optimization results to CSV file
    6. Output final statistics and email notification

Key Features:
    - Daily rebalancing simulation with optimized positions
    - Multiple alpha signal combination with configurable weights
    - Corporate action handling (splits, dividends)
    - Transaction cost modeling (slippage + execution fees)
    - Factor risk and specific risk optimization
    - Barra factor exposure tracking across dates
    - Earnings avoidance with increased volatility scaling
    - Short locate constraints
    - Participation rate limits (1.5% of daily volume)
    - Position size limits (max 2% ADV, $1M per stock)
    - Market cap and price filters ($1.6B min cap, $500 max price)
    - Industry exclusions (PHARMA sector excluded)
    - P&L attribution and performance metrics

Command-Line Arguments:
    --start: Start date (YYYYMMDD format, required)
    --end: End date (YYYYMMDD format, required)
    --fcast: Alpha forecast specification (required)
             Format: "dir:name:multiplier:weight,dir2:name2:mult2:weight2"
             Example: "hl:1:1,bd:0.5:0.5" combines hl and bd strategies
             Each forecast is scaled by multiplier, then weighted in combination
    --horizon: Forecast horizon in days (default: 3)
               Used to scale volatility and returns to forecast period
    --mult: Global alpha multiplier applied to all forecasts (default: 1.0)
    --kappa: Risk aversion parameter (default: 2.0e-8)
             Higher values = more conservative, lower positions
             Typical range: 2e-8 (aggressive) to 4.3e-5 (conservative)
    --maxnot: Maximum total notional ($, default: $200M)
              Hard cap on sum of absolute position values
    --maxdollars: Maximum position size per stock ($, default: $1M)
                  Caps individual security exposure
    --vwap: Use VWAP execution instead of close (default: False)
    --locates: Require short locates data (default: True)
               When enabled, restricts shorts to available borrows
    --earnings: Earnings avoidance window in days (default: None)
                If set (e.g., "3"), reduces positions N days before/after earnings
                Also scales up residual volatility near earnings (1.5x-3x)
    --slipnu: Market impact coefficient nu in slippage model (default: 0.18)
              Cost = alpha + delta*participation^beta + nu*market_impact
    --slipbeta: Participation power beta in slippage model (default: 0.6)
    --fast: Fast mode - only run at 30-minute intervals (default: False)
    --exclude: Exclude stocks by attribute (format: "attr:threshold", default: None)
               Example: "mkt_cap_y:2e9" excludes stocks below $2B market cap
    --maxforecast: Maximum allowed forecast value (default: 0.0050 = 50bps)
                   Forecasts are clipped to [-maxforecast, +maxforecast]
    --nonegutil: Skip trades with negative utility (default: True)
                 When enabled, keeps last position if optimizer suggests negative utility trade
    --daily: Only run at end of day 15:30+ (default: False)
    --maxiter: Maximum optimizer iterations (default: 1500)

Output Files:
    ./opt/opt.{forecast_names}.{YYYYMMDD}_{HHMMSS}.csv

    Columns:
        - iclose_ts: Timestamp index
        - sid: Security ID
        - target: Optimal position from optimizer ($)
        - dutil: Marginal utility of position
        - eslip: Expected slippage cost ($)
        - dmu: Expected alpha return ($)
        - dsrisk: Expected specific risk contribution ($)
        - dfrisk: Expected factor risk contribution ($)
        - costs: Total transaction costs ($)
        - dutil2: Secondary utility metric
        - traded: Dollar amount traded (target - position_last)
        - shares: Final share position
        - position: Final dollar position (after participation constraints)
        - iclose: Price at close
        - forecast: Combined alpha forecast (clipped to maxforecast)

Performance Metrics:
    - Daily P&L and cumulative returns
    - Sharpe ratio and drawdown statistics
    - Factor exposures (13 Barra factors + 58 industries)
    - Turnover and participation metrics
    - Risk attribution (specific vs factor risk)

Examples:
    # Single alpha strategy with default parameters
    python bsim.py --start=20130101 --end=20130630 --fcast=hl:1:1 --kappa=2e-8

    # Multi-alpha combination with custom weights
    python bsim.py --start=20130101 --end=20130630 \\
        --fcast=hl:1:0.6,bd:0.8:0.4 --kappa=2e-8 --maxnot=200e6

    # Conservative strategy with earnings avoidance
    python bsim.py --start=20130101 --end=20130630 \\
        --fcast=hl:1:1 --kappa=4.3e-5 --earnings=3 --maxnot=50e6

    # Aggressive strategy with high participation
    python bsim.py --start=20130101 --end=20130630 \\
        --fcast=hl:1:1,bd:1:1,pca:0.5:0.5 --kappa=2e-8 --slipnu=0.14

Notes:
    - Requires pre-computed alpha forecasts in forecast directories
    - Requires Barra factor data and price/volume data loaded via loaddata.py
    - Creates opt/ directory for output if it doesn't exist
    - Sends email notification when complete
    - Uses Python 2.7 legacy syntax
    - Memory-intensive: processes full universe (~1400 stocks) at each timestamp
    - Typical runtime: 1-2 hours for 6-month backtest
"""

from __future__ import division, print_function

from util import *
from regress import *
from loaddata import *
from calc import *
import opt
import gc

from collections import defaultdict

import argparse

def pnl_sum(group):
    """
    Calculate cumulative P&L for a group of positions.

    Computes the dollar P&L by applying log returns to position sizes.
    Converts cumulative log returns to simple returns, then multiplies by
    position value to get dollar profit/loss.

    Args:
        group: DataFrame group containing:
            - cum_log_ret_i_now: Cumulative log return at current time
            - cum_log_ret_i_then: Cumulative log return at entry time
            - position_then: Position value at entry ($)

    Returns:
        float: Cumulative dollar P&L for the group

    Formula:
        PnL = sum((exp(log_ret_now - log_ret_then) - 1) * position_value)

    Notes:
        - Uses log returns for accurate compounding over multiple periods
        - Handles missing values by filling with 0
        - Position value includes sign (positive = long, negative = short)
    """
    cum_pnl = ((np.exp(group['cum_log_ret_i_now' ] - group['cum_log_ret_i_then']) - 1) * group['position_then']).fillna(0).sum()
    return cum_pnl

parser = argparse.ArgumentParser(description='G')
parser.add_argument("--start",action="store",dest="start",default=None)
parser.add_argument("--end",action="store",dest="end",default=None)
parser.add_argument("--fcast",action="store",dest="fcast",default=None)
parser.add_argument("--horizon",action="store",dest="horizon",default=3)
parser.add_argument("--mult",action="store",dest="mult",default=1.0)
parser.add_argument("--vwap",action="store",dest="vwap",default=False)
parser.add_argument("--maxiter",action="store",dest="maxiter",default=1500)
parser.add_argument("--kappa",action="store",dest="kappa",default=2.0e-8)
parser.add_argument("--slipnu",action="store",dest="slip_nu",default=.18)
parser.add_argument("--slipbeta",action="store",dest="slip_beta",default=.6)
parser.add_argument("--fast",action="store",dest="fast",default=False)
parser.add_argument("--exclude",action="store",dest="exclude",default=None)
parser.add_argument("--earnings",action="store",dest="earnings",default=None)
parser.add_argument("--locates",action="store",dest="locates",default=True)
parser.add_argument("--maxnot",action="store",dest="maxnot",default=200e6)
parser.add_argument("--maxdollars",action="store",dest="maxdollars",default=1e6)
parser.add_argument("--maxforecast",action="store",dest="maxforecast",default=0.0050)
parser.add_argument("--nonegutil",action="store",dest="nonegutil",default=True)
parser.add_argument("--daily",action="store",dest="daily",default=False)
args = parser.parse_args()

print(args)

# Create output directory for optimization results
mkdir_p("opt")

# Parse command-line parameters
ALPHA_MULT = float(args.mult)  # Global alpha multiplier
horizon = int(args.horizon)  # Forecast horizon (days)
start = args.start  # Start date (YYYYMMDD)
end = args.end  # End date (YYYYMMDD)

# Configure risk and constraint parameters
factors = ALL_FACTORS  # Barra factor list (13 factors)
max_forecast = float(args.maxforecast)  # Clip forecasts to this magnitude (default: 0.005 = 50bps)
max_adv = 0.02  # Max position as fraction of ADV (2%)
max_dollars = float(args.maxdollars)  # Max position per stock ($1M default)
participation = 0.015  # Max participation rate (1.5% of daily volume)

# Optimizer configuration
opt.min_iter = 50  # Minimum iterations
opt.max_iter = int(args.maxiter)  # Maximum iterations (default: 1500)
opt.kappa = float(args.kappa)  # Risk aversion parameter (default: 2e-8)
opt.max_sumnot = float(args.maxnot)  # Max total notional ($200M default)
opt.max_expnot = 0.04  # Max exposure per security (4% of capital)
opt.max_trdnot = 0.5  # Max trade notional (50% of capital)

# Slippage model parameters
opt.slip_alpha = 1.0  # Base cost (1 bps)
opt.slip_delta = 0.25  # Participation coefficient
opt.slip_beta = float(args.slip_beta)  # Participation power (default: 0.6)
opt.slip_gamma = 0  # Volatility coefficient (disabled: 0.3 typical)
opt.slip_nu = float(args.slip_nu)  # Market impact coefficient (default: 0.18)
opt.execFee = 0.00015  # Execution fee (1.5 bps)
opt.num_factors = len(factors)  # Number of Barra factors

# Define columns to load from cache
# Includes: price/volume, Barra factors, industries, risk metrics, corporate actions
cols = ['ticker', 'iclose', 'tradable_volume', 'close', 'bvwap_b', 'tradable_med_volume_21_y', 'mdvp_y', 'overnight_log_ret', 'date', 'log_ret', 'bvolume', 'capitalization', 'cum_log_ret', 'srisk_pct', 'dpvolume_med_21', 'volat_21_y', 'mkt_cap_y', 'cum_log_ret_y', 'open', 'close_y', 'indname1', 'barraResidRet', 'split', 'div', 'gdate', 'rating_mean_z']
cols.extend( BARRA_FACTORS )  # Add 13 Barra factors
# cols.extend( BARRA_INDS )  # Optional: add 58 Barra industries (using INDUSTRIES instead)
cols.extend( INDUSTRIES )  # Add industry classifications

# Parse forecast specification and extract alpha names
# Format: "dir:name:mult:weight,dir2:name2:mult2:weight2"
forecasts = list()
forecastargs = args.fcast.split(',')
for fcast in forecastargs:
    fdir, name, mult, weight = fcast.split(":")
    forecasts.append(name)

# Load data from HDF5 cache
# factor_df: Barra factor covariance matrix (time series)
# pnl_df: Main DataFrame with price/volume/factor data (intraday timestamps x securities)
factor_df = load_factor_cache(dateparser.parse(start), dateparser.parse(end))
pnl_df = load_cache(dateparser.parse(start), dateparser.parse(end), cols)
# print pnl_df.xs(10027954, level=1)['indname1']  # Debug: check industry for specific security

pnl_df = pnl_df.truncate(before=dateparser.parse(start), after=dateparser.parse(end))
pnl_df.index.names = ['iclose_ts', 'sid']
pnl_df['forecast'] = np.nan
pnl_df['forecast_abs'] = np.nan

# Load alpha forecasts from each strategy directory and merge into pnl_df
for fcast in forecastargs:
    print("Loading {}".format(fcast))
    fdir, name, mult, weight = fcast.split(":")
    mu_df = load_mus(fdir, name, start, end)  # Load forecast DataFrame
    pnl_df = pd.merge(pnl_df, mu_df, how='left', left_index=True, right_index=True)

# Extract daily data at 15:45 close
# Unstack converts (timestamp, sid) index to (timestamp) index with sid as columns
# between_time filters to 15:45 snapshots, stack converts back to (timestamp, sid)
# daily_df = pnl_df.unstack().between_time('15:30', '15:30').stack()  # Old: 15:30
daily_df = pnl_df.unstack().between_time('15:45', '15:45').stack()  # New: 15:45
daily_df = daily_df.dropna(subset=['date'])
daily_df = daily_df.reset_index().set_index(['date', 'sid'])

# Create z-score for specific risk percentage
# Standardizes srisk_pct across universe for risk comparisons
daily_df = create_z_score(daily_df, 'srisk_pct')

# Load short locate data (if enabled)
# Restricts short positions to securities with available borrows
if args.locates is not None:
    locates_df = load_locates(daily_df[['ticker']], dateparser.parse(start), dateparser.parse(end))
    daily_df = pd.merge(daily_df, locates_df, how='left', left_index=True, right_index=True, suffixes=['', '_dead'])
    daily_df = remove_dup_cols(daily_df)
    locates_df = None  # Free memory

# Load earnings dates (if earnings avoidance enabled)
# Adds daysToEarn and daysFromEarn columns for risk scaling
if args.earnings is not None:
    earnings_df = load_earnings_dates(daily_df[['ticker']], dateparser.parse(start), dateparser.parse(end))
    daily_df = pd.merge(daily_df, earnings_df, how='left', left_index=True, right_index=True, suffixes=['', '_dead'])
    daily_df = remove_dup_cols(daily_df)
    earnings_df = load_past_earnings_dates(daily_df[['ticker']], dateparser.parse(start), dateparser.parse(end))
    daily_df = pd.merge(daily_df, earnings_df, how='left', left_index=True, right_index=True, suffixes=['', '_dead'])
    daily_df = remove_dup_cols(daily_df)
    earnings_df = None  # Free memory

#daily_df = transform_barra(daily_df)
pnl_df = pd.merge(pnl_df.reset_index(), daily_df.reset_index(), how='left', left_on=['date', 'sid'], right_on=['date', 'sid'], suffixes=['', '_dead'])
pnl_df = remove_dup_cols(pnl_df)                     
pnl_df.set_index(['iclose_ts', 'sid'], inplace=True)

# Calculate residual volatility (specific risk)
# resid_df, factor_df = calc_factors(daily_df)  # Legacy: compute factor residuals
daily_df['residVol'] = horizon * (calc_resid_vol(pnl_df) / 100.0) / np.sqrt(252.0)
factor_df = calc_factor_vol(factor_df)  # Compute factor volatilities

# Merge daily data back into intraday pnl_df
pnl_df = pd.merge(pnl_df.reset_index(), daily_df.reset_index(), how='left', left_on=['date', 'sid'], right_on=['date', 'sid'], suffixes=['', '_dead'])
pnl_df = remove_dup_cols(pnl_df)
pnl_df.set_index(['iclose_ts', 'sid'], inplace=True)

# Calculate residual volatility scaled to forecast horizon
# residVol = horizon * (specific_risk_pct / 100) / sqrt(252 trading days)
pnl_df['residVol'] = horizon * (pnl_df['srisk_pct'] / 100.0) / np.sqrt(252.0)

# Calculate daily volume change (for participation constraints)
# bvolume_d = today's volume (positive if increasing, reset on negative diffs)
pnl_df['bvolume_d'] = pnl_df['bvolume'].groupby(level='sid').diff()
pnl_df.loc[ pnl_df['bvolume_d'] < 0, 'bvolume_d'] = pnl_df['bvolume']

# Push data forward to align with trading time (make yesterday's data available today)
pnl_df = push_data(pnl_df, 'bvolume_d')
pnl_df = push_data(pnl_df, 'bvwap_b')

# COMBINE ALPHA FORECASTS
# Mix multiple alpha signals with specified multipliers and weights
# Final forecast = ALPHA_MULT * sum(weight_i * mult_i * alpha_i)
pnl_df[ 'forecast' ] = 0
for fcast in forecastargs:
    fdir, name, mult, weight = fcast.split(":")
    pnl_df[ name + '_adj' ] = pnl_df[ name ] * float(weight) * float(mult)
    pnl_df[ 'forecast' ] += pnl_df[name + '_adj'].fillna(0)

# Apply global multiplier and clip to max_forecast bounds
# Prevents extreme forecasts from dominating optimization
pnl_df['forecast'] = (ALPHA_MULT * pnl_df['forecast']).clip(-max_forecast, max_forecast)
pnl_df['forecast_abs'] = np.abs(pnl_df['forecast'])

# Calculate max tradeable shares based on participation rate
# Limits trading to participation * today's volume to control market impact
pnl_df['max_trade_shares'] = pnl_df[ 'bvolume_d_n' ] * participation

# Initialize tracking columns (will be populated during simulation loop)
pnl_df['position'] = 0  # Final position ($)
pnl_df['traded'] = 0  # Amount traded ($)
pnl_df['target'] = 0  # Optimizer target ($)
pnl_df['dutil'] = 0  # Marginal utility
pnl_df['dsrisk'] = 0  # Specific risk contribution
pnl_df['dfrisk'] = 0  # Factor risk contribution
pnl_df['dmu'] = 0  # Alpha contribution
pnl_df['eslip'] = 0  # Expected slippage
pnl_df['cum_pnl'] = 0  # Cumulative P&L

# Calculate position bounds based on ADV and max_dollars constraint
# max_notional: max long position = min(2% ADV, $1M)
# min_notional: max short position = max(-2% ADV, -$1M)
pnl_df['max_notional'] = (pnl_df['tradable_med_volume_21_y'] * pnl_df['close_y'] * max_adv).clip(0, max_dollars)
pnl_df['min_notional'] = (-1 * pnl_df['tradable_med_volume_21_y'] * pnl_df['close_y'] * max_adv).clip(-max_dollars, 0)

# Apply locate constraints (if enabled)
# Restricts shorts to available borrow quantity
if args.locates is not None:
    pnl_df['borrow_notional'] = pnl_df['borrow_qty'] * pnl_df['iclose']
    pnl_df['min_notional'] = pnl_df[ ['borrow_notional', 'min_notional'] ].max(axis=1)  # Less negative = tighter constraint
    pnl_df.loc[ pnl_df['fee_rate'] > 0, 'min_notional' ] = 0  # If borrow fee > 0, don't short at all
    
# Initialize position tracker
# Maintains last known position for each security across simulation
# Updated at each timestep to track portfolio state
last_pos = pd.DataFrame(pnl_df.reset_index()['sid'].unique(), columns=['sid'])
last_pos['shares_last'] = 0
last_pos.set_index(['sid'], inplace=True)
last_pos = last_pos.sort()

lastday = None  # Track last trading day for corporate action handling

it = 0
groups = pnl_df.groupby(level='iclose_ts')

# Free memory before main loop
pnl_df = None
daily_df = None
new_pnl_df = None
gc.collect()

# MAIN SIMULATION LOOP
# Iterate through each timestamp in the backtest period
# At each step: filter universe, optimize portfolio, track positions, save results
for name, date_group in groups:
    # Parse timestamp and filter to backtest range
    dayname = name.strftime("%Y%m%d")
    if (int(dayname) < int(start)) or (int(dayname) > int(end)): continue

    # Time-of-day filtering
    hour = int(name.strftime("%H"))
    minute = int(name.strftime("%M"))
    if args.daily:
        if hour < 15 or minute < 30: continue  # Only run at/after 3:30pm

    if args.fast:
        minutes = int(name.strftime("%M"))
        if minutes != 30: continue  # Only run at :30 intervals

    if hour >= 16: continue  # Skip after market close

    print("Looking at {}".format(name))
    monthname = name.strftime("%Y%m")
    timename = name.strftime("%H%M%S")
    weekdayname = name.weekday()

    # Filter tradable universe: require positive price, volume, and ADV data
    date_group = date_group[ (date_group['iclose'] > 0) & (date_group['bvolume_d'] > 0) & (date_group['mdvp_y'] > 0) ].sort()
    if len(date_group) == 0:
        print("No data for {}".format(name))
        continue

    # Merge current universe with last positions
    # Use outer join to capture both new securities and existing positions
    date_group = pd.merge(date_group.reset_index(), last_pos.reset_index(), how='outer', left_on=['sid'], right_on=['sid'], suffixes=['', '_last'])
    date_group['iclose_ts'] = name
    date_group = date_group.dropna(subset=['sid'])
    date_group.set_index(['iclose_ts', 'sid'], inplace=True)

    # Corporate action handling: adjust shares for stock splits
    # Only apply on day transitions to avoid double-counting
    if lastday is not None and lastday != dayname:
        date_group['shares_last'] = date_group['shares_last'] * date_group['split']

    # Calculate last position value at current prices
    date_group['position_last'] = (date_group['shares_last'] * date_group['iclose']).fillna(0)

    # Data quality checks: zero out position bounds for securities with missing/invalid data
    # Prevents optimizer from trading securities with unreliable data
    date_group.loc[ date_group['iclose'].isnull() | date_group['mdvp_y'].isnull() | (date_group['mdvp_y'] == 0) | date_group['bvolume_d'].isnull() | (date_group['bvolume_d'] == 0) | date_group['residVol'].isnull(), 'max_notional' ] = 0
    date_group.loc[ date_group['iclose'].isnull() | date_group['mdvp_y'].isnull() | (date_group['mdvp_y'] == 0) | date_group['bvolume_d'].isnull() | (date_group['bvolume_d'] == 0) | date_group['residVol'].isnull(), 'min_notional' ] = 0

    # Optional: exclude securities by attribute threshold
    # if args.exclude is not None:
    #     attr, val = args.exclude.split(":")
    #     val = float(val)
    #     date_group.loc[ date_group[attr] < val, 'forecast' ] = 0
    #     date_group.loc[ date_group[attr] < val, 'max_notional' ] = 0
    #     date_group.loc[ date_group[attr] < val, 'min_notional' ] = 0

    # Universe filters: exclude small cap, high price, and pharma stocks
    # - Market cap < $1.6B: too illiquid and risky
    # - Price > $500: options may be better vehicle
    # - PHARMA industry: high regulatory risk and volatility
    date_group.loc[ (date_group['mkt_cap_y'] < 1.6e9) | (date_group['iclose'] > 500.0) | (date_group['indname1'] == "PHARMA") , 'forecast' ] = 0
    date_group.loc[ (date_group['mkt_cap_y'] < 1.6e9) | (date_group['iclose'] > 500.0) | (date_group['indname1'] == "PHARMA"), 'max_notional' ] = 0
    date_group.loc[ (date_group['mkt_cap_y'] < 1.6e9) | (date_group['iclose'] > 500.0) | (date_group['indname1'] == "PHARMA"), 'min_notional' ] = 0


    # Earnings avoidance logic (if enabled)
    # Reduces risk exposure around earnings announcements due to high uncertainty
    if args.earnings is not None:
        days = int(args.earnings)

        # Scale up residual volatility near earnings to reduce optimizer position sizes
        # Closer to earnings = higher volatility multiplier (3 days: 1.5x, 2 days: 2x, 1 day: 3x)
        date_group.loc[ date_group['daysToEarn'] == 3, 'residVol'] = date_group.loc[ date_group['daysToEarn'] == 3, 'residVol'] * 1.5
        date_group.loc[ date_group['daysToEarn'] == 2, 'residVol'] = date_group.loc[ date_group['daysToEarn'] == 2, 'residVol'] * 2
        date_group.loc[ date_group['daysToEarn'] == 1, 'residVol'] = date_group.loc[ date_group['daysToEarn'] == 1, 'residVol'] * 3

        # Within earnings window: allow only exit trades (no new positions, no increasing existing)
        # For longs (position_last >= 0): max_notional capped at position_last (can only sell down)
        date_group.ix [ ( (date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (date_group['position_last'] >= 0), 'max_notional'] =   date_group.ix [ ( (date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (date_group['position_last'] >= 0), 'position_last']
        date_group.loc[ ( (date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (date_group['position_last'] >= 0), 'min_notional'] = 0

        # For shorts (position_last <= 0): min_notional capped at position_last (can only cover)
        date_group.loc[ ( (date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (date_group['position_last'] <= 0), 'max_notional'] = 0
        date_group.loc[ ( (date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (date_group['position_last'] <= 0), 'min_notional'] =   date_group.loc [ ( (date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (date_group['position_last'] >= 0), 'position_last']

    # PORTFOLIO OPTIMIZATION
    # Setup optimizer with current universe and constraints
    opt.num_secs = len(date_group)
    opt.init()
    opt.sec_ind = date_group.reset_index().index.copy().values  # Index mapping
    opt.sec_ind_rev = date_group.reset_index()['sid'].copy().values  # Reverse mapping: index -> sid

    # Pass position data to optimizer
    opt.g_positions = date_group['position_last'].copy().values  # Current positions
    opt.g_lbound = date_group['min_notional'].fillna(0).values  # Lower bounds (shorts limited by locates)
    opt.g_ubound = date_group['max_notional'].fillna(0).values  # Upper bounds (position size limits)
    opt.g_mu = date_group['forecast'].copy().fillna(0).values  # Alpha signals (expected returns)
    opt.g_rvar = date_group['residVol'].copy().fillna(0).values  # Specific risk (residual volatility)
    opt.g_advp = date_group[ 'mdvp_y'].copy().fillna(0).values  # ADV in dollars (for participation)
    opt.g_price = date_group['iclose'].copy().fillna(0).values  # Current prices
    opt.g_advpt = (date_group['bvolume_d'] * date_group['iclose']).fillna(0).values  # Today's volume in dollars
    opt.g_vol = date_group['volat_21_y'].copy().fillna(0).values * horizon  # Volatility scaled to horizon
    opt.g_mktcap = date_group['mkt_cap_y'].copy().fillna(0).values  # Market capitalization

    # Debug output for test security
    print(date_group.xs(testid, level=1)[['forecast', 'min_notional', 'max_notional', 'position_last']])

    # Pass Barra factor exposures to optimizer (13 factors x N securities matrix)
    find = 0
    for factor in factors:
        opt.g_factors[ find, opt.sec_ind ] = date_group[factor].fillna(0).values
        find += 1

    # Pass factor covariance matrix to optimizer (13 x 13 matrix)
    # Scaled by horizon to match forecast period
    find1 = 0
    for factor1 in factors:
        find2 = 0
        for factor2 in factors:
            try:
                factor_cov = factor_df[(factor1, factor2)].fillna(0).loc[pd.to_datetime(dayname)]
                # Uncomment to print factor correlations for debugging:
                # factor1_sig = np.sqrt(factor_df[(factor1, factor1)].fillna(0).loc[pd.to_datetime(dayname)])
                # factor2_sig = np.sqrt(factor_df[(factor2, factor2)].fillna(0).loc[pd.to_datetime(dayname)])
                # print "Factor Correlation {}, {}: {}".format(factor1, factor2, factor_cov/(factor1_sig*factor2_sig))
            except:
                # Missing covariance data - assume zero correlation
                # print "No cov found for {} {}".format(factor1, factor2)
                factor_cov = 0

            opt.g_fcov[ find1, find2 ] = factor_cov * horizon
            opt.g_fcov[ find2, find1 ] = factor_cov * horizon

            find2 += 1
        find1 += 1

    # Run optimizer to maximize risk-adjusted returns
    # Returns: target positions, utilities, slippage, alpha, risk decomposition, costs
    try:
        (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2) = opt.optimize()
    except:
        # Save problematic data for debugging
        date_group.to_csv("problem.csv")
        raise

    # Store optimization results in DataFrame
    optresults_df = pd.DataFrame(index=date_group.index, columns=['target', 'dutil', 'eslip', 'dmu', 'dsrisk', 'dfrisk', 'costs', 'dutil2', 'traded'])
    optresults_df['target'] = target  # Optimal target position ($)
    optresults_df['dutil'] = dutil  # Marginal utility of position
    optresults_df['eslip'] = eslip  # Expected slippage cost ($)
    optresults_df['dmu'] = dmu  # Expected alpha return ($)
    optresults_df['dsrisk'] = dsrisk  # Specific risk contribution ($)
    optresults_df['dfrisk'] = dfrisk  # Factor risk contribution ($)
    optresults_df['costs'] = costs  # Total transaction costs ($)
    optresults_df['dutil2'] = dutil2  # Secondary utility metric

    # Legacy: these were used when keeping full pnl_df in memory
    # Now we process date by date to save memory
    # pnl_df.loc[ date_group.index, 'target'] = optresults_df['target']
    # pnl_df.loc[ date_group.index, 'eslip'] = optresults_df['eslip']
    # pnl_df.loc[ date_group.index, 'dutil'] = optresults_df['dutil']
    # pnl_df.loc[ date_group.index, 'dsrisk'] = optresults_df['dsrisk']
    # pnl_df.loc[ date_group.index, 'dfrisk'] = optresults_df['dfrisk']
    # pnl_df.loc[ date_group.index, 'dmu'] = optresults_df['dmu']

    date_group['target'] = optresults_df['target']
    date_group['dutil'] = optresults_df['dutil']

    # Filter out trades with negative utility (if enabled)
    # Negative utility means the trade would reduce expected risk-adjusted returns
    # Keep last position instead of making a value-destroying trade
    if args.nonegutil:
        date_group.loc[ date_group['dutil'] <= 0, 'target'] = date_group.loc[ date_group['dutil'] <= 0, 'position_last']

    # Apply participation constraints to limit trading rate
    # max_trade_shares = participation_rate * today's_volume
    # Prevents excessive market impact by limiting trade size relative to daily volume
    date_group['max_move'] = date_group['position_last'] + date_group['max_trade_shares'] * date_group['iclose']
    date_group['min_move'] = date_group['position_last'] - date_group['max_trade_shares'] * date_group['iclose']
    date_group['position'] = date_group['target']
    date_group['position'] = date_group[ ['position', 'max_move'] ].min(axis=1)  # Cap upside move
    date_group['position'] = date_group[ ['position', 'min_move'] ].max(axis=1)  # Cap downside move

    # Debug output for constrained positions:
    # df = date_group[ date_group['target'] > date_group['max_move']]
    # print df[['max_move', 'min_move', 'target', 'position', 'max_trade_shares', 'position_last', 'bvolume_d_n']].head()
    # print date_group.xs(10000108, level=1)[['max_move', 'min_move', 'target', 'position', 'max_trade_shares', 'position_last', 'bvolume_d_n']]

    # Calculate actual trade size and final share position
    date_group['traded'] = date_group['position'] - date_group['position_last']
    date_group['shares'] = date_group['position'] / date_group['iclose']

    # Update position tracker for next iteration
    postmp = pd.merge(last_pos.reset_index(), date_group['shares'].reset_index(), how='outer', left_on=['sid'], right_on=['sid']).set_index('sid')
    last_pos['shares_last'] = postmp['shares'].fillna(0)
    postmp = None

    # Add additional columns to output DataFrame
    optresults_df['forecast'] = date_group['forecast']
    optresults_df['traded'] = date_group['traded']
    optresults_df['shares'] = date_group['shares']
    optresults_df['position'] = date_group['position']
    optresults_df['iclose'] = date_group['iclose']

    # Clean up index and write results to CSV
    optresults_df = optresults_df.reset_index()
    optresults_df['sid'] = optresults_df['sid'].astype(int)
    optresults_df.set_index(['iclose_ts', 'sid'], inplace=True)

    # Output file: ./opt/opt.{alpha1-alpha2-...}.{YYYYMMDD}_{HHMMSS}.csv
    # Contains all optimization results and final positions for this timestamp
    optresults_df.to_csv("./opt/opt." + "-".join(forecasts) + "." + dayname + "_" + timename + ".csv")

    lastday = dayname
    it += 1
    # groups.remove(name)  # Not needed - iterator handles this

    # Free memory after each iteration
    date_group = None
    gc.collect()

# Send completion notification
email("bsim done: " + args.fcast, "")

#pnl_df.to_csv("debug." + "-".join(forecasts) + "." + str(start) + "." + str(end) + ".csv")
#pnl_df.xs(testid, level=1).to_csv("debug.csv")
    
