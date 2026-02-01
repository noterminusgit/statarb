"""
SALAMANDER BSIM - Daily Backtesting Simulation Engine (Python 3)

This is the Python 3 standalone version of the daily statistical arbitrage
backtesting engine. It provides a simplified, self-contained simulation
environment independent of the main Python 2.7 codebase.

Differences from Main bsim.py:
==============================

1. **Python 3 Compatibility**
   - Print statements use function syntax: print() not print
   - Dictionary iteration updated (.items(), .keys(), .values())
   - Integer division uses // operator where needed
   - Pandas DataFrame indexing uses .loc[] and .iloc[] instead of .ix[]

2. **Simplified Data Pipeline**
   - Uses standalone loaddata.py for HDF5/CSV loading
   - Loads data from --dir parameter (no hardcoded paths)
   - Reduced columns: only core price/volume/factor data
   - No intraday timestamps - daily data only
   - Simpler universe filtering logic

3. **Reduced Factor Model**
   - Only 5 Barra factors (growth, size, divyild, btop, momentum)
   - Main version has 13 factors
   - No industry classifications (main has 58 industries)
   - Simplified factor covariance calculations

4. **Streamlined Alpha Signals**
   - Default: only HL (High-Low) strategy implemented
   - Alpha format: "name:multiplier:weight" (3 fields)
   - Main version format: "dir:name:multiplier:weight" (4 fields)
   - Loads from data/hl/ directory structure

5. **Simplified Optimization**
   - Uses same opt.py module with reduced factor count
   - Fewer constraint types
   - Simpler position bound calculations
   - No earnings date handling by default
   - Simplified locate constraints

6. **Different Index Names**
   - Uses 'date' and 'gvkey' instead of 'iclose_ts' and 'sid'
   - No intraday timestamp tracking
   - Daily rebalancing only (no intraday simulation)

7. **Standalone Directory Structure**
   - Data path: {--dir}/data/
   - Alpha signals: {--dir}/data/hl/
   - Locates: {--dir}/data/locates/
   - Output: {--dir}/data/opt/ and {--dir}/data/blotter/

8. **Performance Tracking**
   - Uses blotter DataFrame to track all trades
   - Calculates P&L from price differences vs traded shares
   - Outputs annualized return, volatility, and Sharpe ratio
   - Main version has more detailed attribution

9. **Memory Management**
   - Groups processed date by date (more memory efficient)
   - Uses garbage collection between iterations
   - No full pnl_df kept in memory during simulation

10. **Output Format**
    - CSV files: opt.{alpha_name}.{YYYYMMDD}.csv (daily, not timestamped)
    - Blotter: blotter.csv with exec amount and action (BUY/SELL)
    - No email notification on completion

Workflow:
=========

1. Load historical data from HDF5 files (data/all/*.h5)
2. Load alpha forecasts from CSV files (data/hl/*.csv)
3. Load optional locates data from borrow.csv
4. Merge alpha signals and calculate combined forecast
5. Calculate residual volatility and factor exposures
6. Main simulation loop (for each date):
   a. Filter tradable universe (price, volume, data quality)
   b. Merge last positions and handle corporate actions
   c. Apply position bounds and constraints
   d. Call optimizer to find optimal positions
   e. Apply participation constraints to limit trade size
   f. Calculate actual positions and track in blotter
   g. Write optimization results to CSV
   h. Calculate daily P&L and performance metrics
7. Output final statistics (total P&L, Sharpe ratio)

Command-Line Arguments:
=======================

Required:
  --start         Start date (YYYYMMDD format)
  --end           End date (YYYYMMDD format)
  --fcast         Alpha forecast specification
                  Format: "name:multiplier:weight"
                  Example: "hl:1:1" or "hl:1:0.6,hl:0.8:0.4"
  --dir           Root directory path (data stored in {dir}/data/)

Optional:
  --horizon       Forecast horizon in days (default: 3)
  --mult          Global alpha multiplier (default: 1.0)
  --kappa         Risk aversion parameter (default: 2.0e-8)
                  Higher = more conservative, lower positions
                  Range: 2e-8 (aggressive) to 4.3e-5 (conservative)
  --maxnot        Maximum total notional in dollars (default: 200M)
  --maxdollars    Maximum position size per stock (default: 1M)
  --maxforecast   Maximum allowed forecast value (default: 0.0050 = 50bps)
                  Forecasts clipped to [-maxforecast, +maxforecast]
  --locates       Use short locates data (default: "True", set "None" to disable)
  --slip_nu       Market impact coefficient nu (default: 0.18)
  --slip_beta     Market impact power beta (default: 0.6)
  --maxiter       Maximum optimizer iterations (default: 1500)
  --nonegutil     Skip trades with negative utility (default: True)
  --vwap          Use VWAP execution (default: False)
  --fast          Fast mode - skip some iterations (default: False)
  --daily         Only run at end of day (default: False)
  --earnings      Earnings avoidance window in days (default: None)
  --exclude       Exclude stocks by attribute "attr:threshold" (default: None)

Performance Metrics:
====================

The simulation outputs:
  - Daily P&L: Dollar profit/loss per day
  - Daily Return: Daily P&L / total absolute notional
  - Total P&L: Cumulative profit/loss over backtest period
  - Annualized Return: Geometric return scaled to 252 trading days
  - Annualized Volatility: Std dev of daily returns × sqrt(252)
  - Sharpe Ratio: Annualized return / annualized volatility

Output Files:
=============

{dir}/data/opt/opt.{alpha_name}.{YYYYMMDD}.csv
  Contains optimization results for each date:
    - date, gvkey: Index (date and security ID)
    - target: Optimal position from optimizer ($)
    - dutil: Marginal utility of position
    - eslip: Expected slippage cost ($)
    - dmu: Expected alpha return ($)
    - dsrisk: Expected specific risk contribution ($)
    - dfrisk: Expected factor risk contribution ($)
    - costs: Total transaction costs ($)
    - dutil2: Secondary utility metric
    - forecast: Combined alpha forecast
    - traded: Dollar amount traded
    - shares: Final share position
    - position: Final dollar position
    - close: Price at close

{dir}/data/blotter/blotter.csv
  Trade execution blotter:
    - date, gvkey: Index
    - exec amount: Absolute dollar amount traded
    - action: BUY or SELL

Examples:
=========

# Basic single-alpha backtest
python3 salamander/bsim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --dir=/path/to/workspace \\
  --fcast=hl:1:1 \\
  --kappa=2e-8 \\
  --maxnot=200e6

# Multi-alpha combination with custom weights
python3 salamander/bsim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --dir=/path/to/workspace \\
  --fcast=hl:1:0.6,hl:0.8:0.4 \\
  --kappa=2e-8

# Conservative strategy without locates
python3 salamander/bsim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --dir=/path/to/workspace \\
  --fcast=hl:1:1 \\
  --kappa=4.3e-5 \\
  --maxnot=50e6 \\
  --locates=None

# Aggressive strategy with low market impact
python3 salamander/bsim.py \\
  --start=20130101 \\
  --end=20130630 \\
  --dir=/path/to/workspace \\
  --fcast=hl:1:1 \\
  --kappa=2e-8 \\
  --slip_nu=0.14 \\
  --slip_beta=0.5

Data Requirements:
==================

Before running bsim.py, you must:

1. Create directory structure:
   python3 salamander/gen_dir.py --dir=/path/to/workspace

2. Generate processed data files (need 1 year prior data):
   python3 salamander/gen_hl.py \\
     --start=20100630 \\
     --end=20130630 \\
     --dir=/path/to/workspace

3. Generate alpha signals:
   python3 salamander/gen_alpha.py \\
     --start=20130101 \\
     --end=20130630 \\
     --dir=/path/to/workspace

4. (Optional) Place borrow.csv in {dir}/data/locates/

Notes:
======

- Python 3.6+ required (uses modern pandas/numpy APIs)
- Designed for research and prototyping, not production
- Simpler than main bsim.py - fewer features, faster setup
- Only HL strategy included by default
- No email notifications (unlike main version)
- Memory efficient: processes date by date, not full dataset
- Typical runtime: 30-60 minutes for 6-month backtest
"""

import opt
from calc import *
from loaddata import *
import argparse


def pnl_sum(group):
    """
    Calculate cumulative P&L for a group of positions.

    Computes the dollar P&L by applying log returns to position sizes.
    Converts cumulative log returns to simple returns, then multiplies by
    position value to get dollar profit/loss.

    Args:
        group (pd.DataFrame): DataFrame group containing:
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

    Example:
        >>> group = pd.DataFrame({
        ...     'cum_log_ret_i_now': [0.05, 0.03],
        ...     'cum_log_ret_i_then': [0.02, 0.01],
        ...     'position_then': [10000, -5000]
        ... })
        >>> pnl_sum(group)
        304.54  # Approx: 10000*(e^0.03-1) + (-5000)*(e^0.02-1)
    """
    cum_pnl = ((np.exp(group['cum_log_ret_i_now'] - group['cum_log_ret_i_then']) - 1) * group['position_then']).fillna(
        0).sum()
    return cum_pnl


# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Salamander BSIM - Daily Backtesting Simulation (Python 3)')

# Required arguments
parser.add_argument("--start", required=False, help="Start date in YYYYMMDD format")
parser.add_argument("--end", required=False, help="End date in YYYYMMDD format")
parser.add_argument("--fcast", help="Alpha signals, formatted as 'name1:mult1:weight1,name2:mult2:weight2,...'")
parser.add_argument("--dir", help="Root directory (data stored in {dir}/data/)", default='.')

# Forecast and alpha parameters
parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon in days (default: 3)")
parser.add_argument("--mult", type=float, default=1.0, help="Global alpha multiplier (default: 1.0)")
parser.add_argument("--maxforecast", type=float, default=0.0050, help="Max forecast value (default: 0.0050 = 50bps)")

# Optimization parameters
parser.add_argument("--kappa", type=float, default=2.0e-8, help="Risk aversion parameter (default: 2e-8)")
parser.add_argument("--maxiter", type=int, default=1500, help="Maximum optimizer iterations (default: 1500)")
parser.add_argument("--maxnot", type=float, default=200e6, help="Maximum total notional in dollars (default: 200M)")
parser.add_argument("--maxdollars", type=float, default=1e6, help="Maximum position per stock in dollars (default: 1M)")
parser.add_argument("--nonegutil", type=bool, default=True, help="Skip trades with negative utility (default: True)")

# Transaction cost parameters
parser.add_argument("--slip_nu", type=float, default=.18, help="Market impact coefficient nu (default: 0.18)")
parser.add_argument("--slip_beta", type=float, default=.6, help="Market impact power beta (default: 0.6)")
parser.add_argument("--vwap", type=bool, default=False, help="Use VWAP execution (default: False)")

# Constraint parameters
parser.add_argument("--locates", default="True", help="Use short locates data (default: 'True', set 'None' to disable)")
parser.add_argument("--earnings", help="Earnings avoidance window in days (default: None)")
parser.add_argument("--exclude", help="Exclude stocks by attribute 'attr:threshold' (default: None)")

# Execution mode parameters
parser.add_argument("--fast", type=bool, default=False, help="Fast mode - skip some iterations (default: False)")
parser.add_argument("--daily", type=bool, default=False, help="Only run at end of day (default: False)")

args = parser.parse_args()

print(args)

# ============================================================================
# CONFIGURATION AND PARAMETER SETUP
# ============================================================================

# Parse alpha and forecast parameters
ALPHA_MULT = float(args.mult)  # Global multiplier applied to all alpha signals
horizon = int(args.horizon)  # Forecast horizon in days (for scaling volatility/returns)
start = args.start  # Start date in YYYYMMDD format
end = args.end  # End date in YYYYMMDD format
data_dir = args.dir + "/data"  # Data directory path: {dir}/data/

# Risk model configuration
factors = ALL_FACTORS  # Barra factor list (5 factors: growth, size, divyild, btop, momentum)
max_forecast = float(args.maxforecast)  # Clip forecasts to this magnitude (default: 0.005 = 50bps)

# Position sizing constraints
max_adv = 0.02  # Max position as fraction of ADV (2% of average daily volume)
max_dollars = float(args.maxdollars)  # Max position per stock in dollars (default: $1M)
participation = 0.015  # Max participation rate (1.5% of daily volume per trade)

# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================

opt.min_iter = 50  # Minimum optimizer iterations
opt.max_iter = int(args.maxiter)  # Maximum optimizer iterations (default: 1500)
opt.kappa = float(args.kappa)  # Risk aversion parameter (default: 2e-8)
                                # Higher = more conservative, lower gross notional
                                # Range: 2e-8 (aggressive) to 4.3e-5 (conservative)

# Portfolio constraints
opt.max_sumnot = float(args.maxnot)  # Max total notional in dollars (default: $200M)
opt.max_expnot = 0.04  # Max exposure per security (4% of total capital)
opt.max_trdnot = 0.5  # Max trade notional (50% of total capital)

# ============================================================================
# TRANSACTION COST MODEL PARAMETERS
# ============================================================================
# Cost = alpha + delta * participation^beta + nu * market_impact + gamma * volatility
#
# Slippage model breakdown:
#   - slip_alpha: Base cost (1 bps fixed cost per trade)
#   - slip_delta: Participation coefficient (multiplies participation term)
#   - slip_beta: Participation power (default 0.6, nonlinear impact)
#   - slip_nu: Market impact coefficient (default 0.18, scales with trade size)
#   - slip_gamma: Volatility coefficient (disabled: 0, would scale with stock volatility)
#   - execFee: Execution fee (1.5 bps per trade)

opt.slip_alpha = 1.0  # Base slippage cost (1 bps)
opt.slip_delta = 0.25  # Participation rate coefficient
opt.slip_beta = float(args.slip_beta)  # Participation power (default: 0.6)
opt.slip_gamma = 0  # Volatility coefficient (disabled, use 0.3 to enable)
opt.slip_nu = float(args.slip_nu)  # Market impact coefficient (default: 0.18)
opt.execFee = 0.00015  # Execution fee (1.5 bps = 0.00015)

opt.num_factors = len(factors)  # Number of Barra factors in covariance matrix

# ============================================================================
# DATA COLUMN SPECIFICATION
# ============================================================================
# Define which columns to load from HDF5 cache files
#
# Salamander uses a simplified column set compared to main bsim.py:
#   - Core data: symbol, close, volume, open
#   - Volume metrics: med_volume_21 (21-day median volume), mdvp (median daily volume in $)
#   - Returns: log_ret (daily log return), overnight_log_ret (close-to-open return)
#   - Risk: volat_21 (21-day volatility)
#   - Market data: mkt_cap (market capitalization), split, div (dividend)
#   - Industry: ind1 (numeric industry code, simplified from main's 58 industries)
#   - Identifiers: sedol (for locate matching)
#   - Barra factors: growth, size, divyild, btop, momentum (5 factors vs 13 in main)
#
# Note: Main bsim.py loads many more columns including:
#   - Intraday timestamps and VWAP data
#   - Full Barra industry classifications (58 industries)
#   - 13 Barra factors vs 5 here
#   - Analyst ratings, earnings estimates, price targets
#   - More granular risk metrics

cols = ['symbol', 'close', 'volume', 'med_volume_21', 'mdvp', 'overnight_log_ret', 'log_ret', 'volat_21', 'mkt_cap',
        'open', 'ind1', 'split', 'div', 'sedol']
cols.extend(BARRA_FACTORS)  # Add 5 Barra factors: growth, size, divyild, btop, momentum
# cols.extend( BARRA_INDS )  # Not used in salamander (simplified)
# cols.extend( INDUSTRIES )  # Not used in salamander (use ind1 instead)

# ============================================================================
# ALPHA FORECAST LOADING
# ============================================================================
# Parse forecast specification and load alpha signals
#
# Format: "name:multiplier:weight,name2:mult2:weight2,..."
# Example: "hl:1:1" = HL strategy, 1x multiplier, 100% weight
# Example: "hl:1:0.6,hl:0.8:0.4" = Two HL variants, 60/40 weighted combination
#
# Main bsim.py uses 4 fields: "dir:name:mult:weight"
# Salamander uses 3 fields: "name:mult:weight" (simpler, alpha dir implicit)

forecasts = []
forecastargs = args.fcast.split(',')
for fcast in forecastargs:
    name, mult, weight = fcast.split(":")
    forecasts.append(name)

# ============================================================================
# DATA LOADING FROM HDF5 CACHE
# ============================================================================

# Load Barra factor covariance matrix time series
# Contains (factor1, factor2) covariances for each date
# Used by optimizer to calculate factor risk
factor_df = load_factor_cache(dateparser.parse(start), dateparser.parse(end), data_dir)

# Load main price/volume/factor data from HDF5 files
# Returns DataFrame with multi-index (date, gvkey) and columns from 'cols' list
# Data comes from data/all/all.{start}-{end}.h5 files generated by gen_hl.py
pnl_df = load_cache(dateparser.parse(start), dateparser.parse(end), data_dir, cols)

# Truncate to exact date range (cache may contain extra dates)
pnl_df = pnl_df.truncate(before=dateparser.parse(start), after=dateparser.parse(end))
pnl_df.index.names = ['date', 'gvkey']

# Initialize forecast columns (will be populated below)
pnl_df['forecast'] = np.nan
pnl_df['forecast_abs'] = np.nan

# Load alpha forecast data for each specified strategy
# Forecasts come from data/{alpha_name}/alpha.{name}.{start}-{end}.csv files
# Generated by gen_alpha.py from the 'all' HDF5 files
for fcast in forecastargs:
    print("Loading {}".format(fcast))
    name, mult, weight = fcast.split(":")
    mu_df = load_mus(data_dir, name, start, end)  # Load alpha forecast DataFrame
    pnl_df = pd.merge(pnl_df, mu_df, how='left', left_index=True, right_index=True)

# ============================================================================
# DAILY DATA EXTRACTION
# ============================================================================
# Salamander version: No intraday timestamps, data is already daily
# Main bsim.py extracts 15:45 snapshots from intraday data using between_time()
#
# Commented out main bsim.py approach:
# daily_df = pnl_df.unstack().between_time('15:30', '15:30').stack()
# daily_df = pnl_df.unstack().between_time('15:45', '15:45').stack()
# daily_df = daily_df.dropna(subset=['date']).reset_index().set_index(['date', 'gvkey'])
# daily_df = create_z_score(daily_df, 'srisk_pct')

# Salamander: data is already daily, no need to filter by time
daily_df = pnl_df

# ============================================================================
# LOAD SHORT LOCATE CONSTRAINTS (OPTIONAL)
# ============================================================================
# If enabled (--locates != "None"), load borrow availability data
# This restricts short positions to securities with available borrows
#
# Locate data format (borrow.csv):
#   - symbol: Stock ticker
#   - sedol: SEDOL identifier (for matching)
#   - date: Trading date
#   - borrow_qty: Shares available to borrow (0 = no borrow, > 0 = available)
#   - fee: Borrow fee rate (> 0 = expensive borrow, avoid shorting)
#
# Applied in optimization as:
#   - min_notional = max(borrow_notional, calculated_min_notional)
#   - If fee > 0: set min_notional = 0 (don't short expensive borrows)

if args.locates != "None":
    locates_df = load_locates(daily_df[['sedol','symbol']], dateparser.parse(start), dateparser.parse(end), data_dir)
    daily_df = pd.merge(daily_df, locates_df, on=['date', 'gvkey'], suffixes=['', '_dead'])
    daily_df = remove_dup_cols(daily_df)
    locates_df = None  # Free memory

    # Debug code for identifying missing borrows (commented out):
    # Finds top 1500 stocks by market cap with no borrow availability
    '''
    test_df = daily_df.sort_values(['date','mkt_cap'],ascending=False).groupby(level='date',group_keys=False).head(1500)
    test_df = test_df[test_df['borrow_qty']==0]
    missed = test_df.xs('2013-01-03',level=0,drop_level=False)
    missed[['symbol','sedol']].to_csv(r"%smissing_borrow.csv" % "./", "|")
    '''

if args.earnings is not None:
    earnings_df = load_earnings_dates(daily_df[['symbol']], dateparser.parse(start), dateparser.parse(end))
    daily_df = pd.merge(daily_df, earnings_df, how='left', left_index=True, right_index=True, suffixes=['', '_dead'])
    daily_df = remove_dup_cols(daily_df)
    earnings_df = load_past_earnings_dates(daily_df[['symbol']], dateparser.parse(start), dateparser.parse(end))
    daily_df = pd.merge(daily_df, earnings_df, how='left', left_index=True, right_index=True, suffixes=['', '_dead'])
    daily_df = remove_dup_cols(daily_df)
    earnings_df = None

# daily_df = transform_barra(daily_df)
pnl_df = pd.merge(pnl_df.reset_index(), daily_df.reset_index(), how='left', on=['date', 'gvkey'],
                  suffixes=['', '_dead'])
pnl_df = remove_dup_cols(pnl_df)
pnl_df.set_index(['date', 'gvkey'], inplace=True)

resid_df, factor_df = calc_factors(daily_df)
# daily_df['residVol'] = horizon * (calc_resid_vol(pnl_df) / 100.0) / np.sqrt(252.0) we dont have barraResidRet

factor_df = calc_factor_vol(factor_df)
pnl_df = pd.merge(pnl_df, daily_df, how='left', on=['date', 'gvkey'], suffixes=['', '_dead'])
pnl_df = remove_dup_cols(pnl_df)

pnl_df['residVol'] = resid_df['barraResidRet'].groupby(level='gvkey').apply(
    lambda x: x.rolling(20).std())
# pnl_df['residVol'] = horizon * (pnl_df['srisk_pct'] / 100.0) / np.sqrt(252.0)

pnl_df['volume_d'] = pnl_df['volume'].groupby(level='gvkey').diff()
pnl_df.loc[pnl_df['volume_d'] < 0, 'volume_d'] = pnl_df['volume']
pnl_df = push_data(pnl_df, 'volume_d')
# pnl_df = push_data(pnl_df, 'bvwap_b')

# MIX FORECASTS
pnl_df['forecast'] = 0
for fcast in forecastargs:
    name, mult, weight = fcast.split(":")
    pnl_df[name + '_adj'] = pnl_df[name] * float(mult) * float(weight)
    pnl_df['forecast'] += pnl_df[name + '_adj'].fillna(0)

pnl_df['forecast'] = (ALPHA_MULT * pnl_df['forecast']).clip(-max_forecast, max_forecast)
pnl_df['forecast_abs'] = np.abs(pnl_df['forecast'])
pnl_df['max_trade_shares'] = pnl_df['volume_d_n'] * participation

pnl_df['position'] = 0
pnl_df['traded'] = 0
pnl_df['target'] = 0
pnl_df['dutil'] = 0
pnl_df['dsrisk'] = 0
pnl_df['dfrisk'] = 0
pnl_df['dmu'] = 0
pnl_df['eslip'] = 0
pnl_df['cum_pnl'] = 0

pnl_df['max_notional'] = (pnl_df['med_volume_21'] * pnl_df['close'] * max_adv).clip(0, max_dollars)
pnl_df['min_notional'] = (-1 * pnl_df['med_volume_21'] * pnl_df['close'] * max_adv).clip(-max_dollars, 0)

if args.locates != "None":
    pnl_df['borrow_notional'] = pnl_df['borrow_qty'] * pnl_df['close']
    pnl_df['min_notional'] = pnl_df[['borrow_notional', 'min_notional']].max(axis=1)
    pnl_df.ix[pnl_df['fee'] > 0, 'min_notional'] = 0

last_pos = pd.DataFrame(pnl_df.reset_index()['gvkey'].unique(), columns=['gvkey'])
last_pos['shares_last'] = 0
last_pos = last_pos.set_index(['gvkey']).sort_index()

# ============================================================================
# SIMULATION INITIALIZATION
# ============================================================================

lastday = None  # Track last trading day for corporate action handling (splits)

it = 0  # Iteration counter
groups = pnl_df.groupby(level='date')  # Group data by date for daily iteration

# Free memory before main loop (process date by date to save RAM)
pnl_df = None
daily_df = None
new_pnl_df = None
gc.collect()

# Initialize blotter DataFrame to track all trades
b_index = pd.MultiIndex(levels=[[], []], labels=[[], []], names=['date', 'gvkey'])
blotter_df = pd.DataFrame(columns=['position', 'traded_shares', 'close'], index=b_index)

# P&L tracking
last_pnl = 0  # Cumulative P&L from previous iteration
daily_returns = []  # List of daily returns for Sharpe calculation

# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================
# Iterate through each trading date in the backtest period
#
# At each date:
#   1. Filter tradable universe (price, volume, data quality checks)
#   2. Merge last positions and apply corporate actions (splits)
#   3. Apply universe filters (market cap, price, industry exclusions)
#   4. Apply locate constraints (if enabled)
#   5. Setup optimizer with current data and constraints
#   6. Run optimization to find optimal positions
#   7. Apply participation constraints to limit trading rate
#   8. Update position tracker for next iteration
#   9. Calculate daily P&L from price changes
#   10. Write results to CSV file
#
# Memory management: Process one date at a time, free memory after each iteration

for name, date_group in groups:
    # ========================================================================
    # DATE FILTERING AND VALIDATION
    # ========================================================================

    dayname = name.strftime("%Y%m%d")
    if (int(dayname) < int(start)) or (int(dayname) > int(end)):
        continue  # Skip dates outside backtest range

    # Time-of-day filtering (salamander is daily, but these checks remain from main version)
    hour = int(name.strftime("%H"))
    minute = int(name.strftime("%M"))

    if args.daily:
        # Only run at end of day (15:30 or later)
        if hour < 15 or minute < 30: continue

    if args.fast:
        # Fast mode: only run at :30 minute intervals
        minutes = int(name.strftime("%M"))
        if minutes != 30: continue

    if hour >= 16:
        continue  # Skip after market close (4pm)

    print("\nLooking at {}".format(name))
    monthname = name.strftime("%Y%m")
    timename = name.strftime("%H%M%S")
    weekdayname = name.weekday()

    # ========================================================================
    # TRADABLE UNIVERSE FILTERING
    # ========================================================================
    # Filter to securities with valid data:
    #   - close > 0: Valid price data
    #   - volume_d > 0: Positive volume (liquidity)
    #   - mdvp > 0: Positive median daily volume in dollars (liquidity filter)

    date_group = date_group[
        (date_group['close'] > 0) & (date_group['volume_d'] > 0) & (date_group['mdvp'] > 0)].sort_index()

    if len(date_group) == 0:
        print("No data for {}".format(name))
        continue  # Skip if no tradable securities on this date

    date_group = pd.merge(date_group.reset_index(), last_pos.reset_index(), how='outer', left_on=['gvkey'],
                          right_on=['gvkey'], suffixes=['', '_last'])
    date_group['date'] = name
    date_group = date_group.dropna(subset=['gvkey'])
    date_group.set_index(['date', 'gvkey'], inplace=True)
    if lastday is not None and lastday != dayname:
        date_group['shares_last'] = date_group['shares_last'] * (date_group['split'].fillna(1))
    date_group['position_last'] = (date_group['shares_last'] * date_group['close']).fillna(0)
    # date_group.ix[ date_group['close'].isnull() | date_group['mdvp'].isnull() | (date_group['mdvp'] == 0) | date_group['volume_d'].isnull() | (date_group['volume_d'] == 0) | date_group['residVol'].isnull(), 'max_notional' ] = 0
    # date_group.ix[ date_group['close'].isnull() | date_group['mdvp'].isnull() | (date_group['mdvp'] == 0) | date_group['volume_d'].isnull() | (date_group['volume_d'] == 0) | date_group['residVol'].isnull(), 'min_notional' ] = 0

    # if args.exclude is not None:
    #     attr, val = args.exclude.split(":")
    #     val = float(val)
    #     date_group.ix[ date_group[attr] < val, 'forecast' ] = 0
    #     date_group.ix[ date_group[attr] < val, 'max_notional' ] = 0
    #     date_group.ix[ date_group[attr] < val, 'min_notional' ] = 0

    # ========================================================================
    # UNIVERSE FILTERS
    # ========================================================================
    # Exclude securities that don't meet trading criteria:
    #
    # 1. Market cap < $1.6B ($1,600M = 1.6e3 M):
    #    - Too illiquid and risky for institutional trading
    #    - Higher transaction costs and wider spreads
    #
    # 2. Price > $500:
    #    - Options may be better vehicle for high-priced stocks
    #    - Position sizing becomes difficult
    #
    # 3. Industry == 3520 (Pharma):
    #    - High regulatory risk (FDA approvals, clinical trials)
    #    - Event-driven volatility (binary outcomes)
    #    - Not suitable for statistical arbitrage
    #
    # For excluded securities: Set forecast = 0, max_notional = 0, min_notional = 0
    # This prevents optimizer from taking any position

    date_group.ix[(date_group['mkt_cap'] < 1.6e3) | (date_group['close'] > 500.0) | (
            date_group['ind1'] == 3520), 'forecast'] = 0  # ind1 == 3520 is PHARMA
    date_group.ix[(date_group['mkt_cap'] < 1.6e3) | (date_group['close'] > 500.0) | (
            date_group['ind1'] == 3520), 'max_notional'] = 0
    date_group.ix[(date_group['mkt_cap'] < 1.6e3) | (date_group['close'] > 500.0) | (
            date_group['ind1'] == 3520), 'min_notional'] = 0

    if args.earnings is not None:
        days = int(args.earnings)
        date_group.ix[date_group['daysToEarn'] == 3, 'residVol'] = date_group.ix[
                                                                       date_group['daysToEarn'] == 3, 'residVol'] * 1.5
        date_group.ix[date_group['daysToEarn'] == 2, 'residVol'] = date_group.ix[
                                                                       date_group['daysToEarn'] == 2, 'residVol'] * 2
        date_group.ix[date_group['daysToEarn'] == 1, 'residVol'] = date_group.ix[
                                                                       date_group['daysToEarn'] == 1, 'residVol'] * 3

        date_group.ix[((date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (
                date_group['position_last'] >= 0), 'max_notional'] = date_group.ix[
            ((date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (
                    date_group['position_last'] >= 0), 'position_last']
        date_group.ix[((date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (
                date_group['position_last'] >= 0), 'min_notional'] = 0
        date_group.ix[((date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (
                date_group['position_last'] <= 0), 'max_notional'] = 0
        date_group.ix[((date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (
                date_group['position_last'] <= 0), 'min_notional'] = date_group.ix[
            ((date_group['daysToEarn'] <= days) | (date_group['daysFromEarn'] < days)) & (
                    date_group['position_last'] >= 0), 'position_last']

    # OPTIMIZATION
    opt.num_secs = len(date_group)
    opt.init()
    opt.sec_ind = date_group.reset_index().index.copy().values
    opt.sec_ind_rev = date_group.reset_index()['gvkey'].copy().values
    opt.g_positions = date_group['position_last'].copy().values
    opt.g_lbound = date_group['min_notional'].fillna(0).values
    opt.g_ubound = date_group['max_notional'].fillna(0).values
    opt.g_mu = date_group['forecast'].copy().fillna(0).values
    opt.g_rvar = date_group['residVol'].copy().fillna(0).values
    opt.g_advp = date_group['mdvp'].copy().fillna(0).values
    opt.g_price = date_group['close'].copy().fillna(0).values
    opt.g_advpt = (date_group['volume_d'] * date_group['close']).fillna(0).values
    opt.g_vol = date_group['volat_21'].copy().fillna(0).values * horizon
    opt.g_mktcap = date_group['mkt_cap'].copy().fillna(0).values

    print(date_group.xs(testid, level=1)[['forecast', 'min_notional', 'max_notional', 'position_last']])

    find = 0
    for factor in factors:
        opt.g_factors[find, opt.sec_ind] = date_group[factor].fillna(0).values
        find += 1

    find1 = 0
    for factor1 in factors:
        find2 = 0
        for factor2 in factors:
            try:
                factor_cov = factor_df[(factor1, factor2)].fillna(0).ix[pd.to_datetime(dayname)]
                #                factor1_sig = np.sqrt(factor_df[(factor1, factor1)].fillna(0).ix[pd.to_datetime(dayname)])
                #               factor2_sig = np.sqrt(factor_df[(factor2, factor2)].fillna(0).ix[pd.to_datetime(dayname)])
                #                print("Factor Correlation {}, {}: {}".format(factor1, factor2, factor_cov/(factor1_sig*factor2_sig)))
            except:
                #                print("No cov found for {} {}".format(factor1, factor2))
                factor_cov = 0

            opt.g_fcov[find1, find2] = factor_cov * horizon
            opt.g_fcov[find2, find1] = factor_cov * horizon

            find2 += 1
        find1 += 1
    try:
        (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2, vol, price) = opt.optimize()
    except:
        date_group.to_csv("problem.csv")
        raise

    optresults_df = pd.DataFrame(index=date_group.index,
                                 columns=['target', 'dutil', 'eslip', 'dmu', 'dsrisk', 'dfrisk', 'costs', 'dutil2',
                                          'traded'])
    optresults_df['target'] = target
    optresults_df['dutil'] = dutil
    optresults_df['eslip'] = eslip
    optresults_df['dmu'] = dmu
    optresults_df['dsrisk'] = dsrisk
    optresults_df['dfrisk'] = dfrisk
    optresults_df['costs'] = costs
    optresults_df['dutil2'] = dutil2

    # pnl_df.ix[ date_group.index, 'target'] = optresults_df['target']
    # pnl_df.ix[ date_group.index, 'eslip'] = optresults_df['eslip']
    # pnl_df.ix[ date_group.index, 'dutil'] = optresults_df['dutil']
    # pnl_df.ix[ date_group.index, 'dsrisk'] = optresults_df['dsrisk']
    # pnl_df.ix[ date_group.index, 'dfrisk'] = optresults_df['dfrisk']
    # pnl_df.ix[ date_group.index, 'dmu'] = optresults_df['dmu']

    date_group['target'] = optresults_df['target']
    date_group['dutil'] = optresults_df['dutil']
    #    tmp = pd.merge(last_pos.reset_index(), date_group['forecast'].reset_index(), how='inner', left_on=['gvkey'], right_on=['gvkey'])
    #    date_group['last_position'] = tmp.set_index(['date', 'gvkey'])['position']

    if args.nonegutil:
        date_group.ix[date_group['dutil'] <= 0, 'target'] = date_group.ix[date_group['dutil'] <= 0, 'position_last']

    date_group['max_move'] = date_group['position_last'] + date_group['max_trade_shares'] * date_group['close']
    date_group['min_move'] = date_group['position_last'] - date_group['max_trade_shares'] * date_group['close']
    date_group['position'] = date_group['target']
    date_group['position'] = date_group[['position', 'max_move']].min(axis=1)
    date_group['position'] = date_group[['position', 'min_move']].max(axis=1)

    # df = date_group[ date_group['target'] > date_group['max_move']]
    # print(df[['max_move', 'min_move', 'target', 'position', 'max_trade_shares', 'position_last', 'volume_d_n']].head())
    # print(date_group.xs(10000108, level=1)[['max_move', 'min_move', 'target', 'position', 'max_trade_shares', 'position_last', 'volume_d_n']])

    date_group['traded'] = date_group['position'] - date_group['position_last']
    date_group['shares'] = date_group['position'] / date_group['close']

    # pnl_df.ix[ date_group.index, 'traded'] = date_group['traded']

    postmp = pd.merge(last_pos.reset_index(), date_group[['shares', 'close', 'position_last']].reset_index(),
                      how='outer', left_on=['gvkey'],
                      right_on=['gvkey']).set_index('gvkey')
    last_pos['shares_last'] = postmp['shares'].fillna(0)
    #    pnl_df.ix[ date_group.index, 'position'] = date_group['position']

    optresults_df['forecast'] = date_group['forecast']
    optresults_df['traded'] = date_group['traded']
    optresults_df['shares'] = date_group['shares']
    optresults_df['position'] = date_group['position']
    optresults_df['close'] = date_group['close']
    optresults_df.to_csv(data_dir + "/opt/opt." + "-".join(forecasts) + "." + dayname + "_" + timename + ".csv")

    date_group['traded_shares'] = date_group['shares'] - date_group['shares_last']
    blotter_df = blotter_df.append(date_group[['traded', 'traded_shares', 'close']])
    blotter_df['diff'] = blotter_df[['close']].groupby(level='date').apply(lambda x: postmp[['close']] - x)
    total_pnl = (blotter_df['diff'] * blotter_df['traded_shares']).fillna(0).sum()
    daily_pnl = total_pnl - last_pnl
    last_pnl = total_pnl
    daily_return = daily_pnl / postmp['position_last'].abs().sum()
    if lastday is not None:
        daily_returns.append(daily_return)
    print("date: %s" % name)
    print("daily_pnl: %.2f" % daily_pnl)
    print("daily_return: %.2f%%" % (daily_return*100))
    lastday = dayname
    it += 1
    #    groups.remove(name)
    date_group = None
    gc.collect()
annualized_days = len(daily_returns) / 252
annualized_return = (total_pnl / postmp['position_last'].abs().sum() + 1) ** (1 / annualized_days) - 1
annualized_volat = np.std(np.array(daily_returns)) * (1 / annualized_days) ** .5
# Sharpe is calculated assuming risk free rate of return is negligible
sharpe = annualized_return / annualized_volat
print("total_pnl: %.2f" % total_pnl)
print("annualized_return: %.2f%%" % (annualized_return * 100))
print("annualized_volat: %.2f%%" % (annualized_volat * 100))
print("sharpe: %.2f" % (sharpe))

print("Saving blotter fields...")
blotter_df = blotter_df[['traded']]
blotter_df['exec amount'] = blotter_df['traded'].abs()
blotter_df['action'] = np.where(blotter_df['traded'] > 0, 'BUY', 'SELL')
blotter_df[['exec amount','action']].to_csv(r"%s/blotter/blotter.csv" % (data_dir))

print("bsim done", args.fcast)

# pnl_df.to_csv("debug." + "-".join(forecasts) + "." + str(start) + "." + str(end) + ".csv")
# pnl_df.xs(testid, level=1).to_csv("debug.csv")
