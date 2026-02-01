#!/usr/bin/env python
"""
Intraday High-Low Mean Reversion Alpha Strategy

Generates alpha signals based on intraday mean reversion from daily high-low ranges.
This is a real-time variant of the daily high-low strategy (hl.py) that uses intraday
close prices to detect deviations from daily extremes during the trading day.

Strategy Concept:
-----------------
This strategy exploits intraday price dislocations from the geometric mean of the
day's high and low prices. Unlike hl.py which uses end-of-day close prices, this
variant uses intraday close prices (e.g., from 30-minute bars) to generate signals
throughout the trading day, allowing for intraday position adjustments.

The core signal measures where the intraday close is relative to the daily high-low range:

    hlC = iclose / sqrt(dhigh * dlow)

Where:
- iclose: Intraday closing price (e.g., 3:00 PM close from 30-min bar)
- dhigh: Daily high price accumulated through the day
- dlow: Daily low price accumulated through the day

When hlC > 1.0: Intraday close is above the geometric midpoint (potential short signal)
When hlC < 1.0: Intraday close is below the geometric midpoint (potential long signal)

The signal is industry-demeaned to isolate stock-specific dislocations from
sector-wide movements, making it market-neutral within industries.

Key Differences from hl.py (Daily Strategy):
---------------------------------------------
1. **Timing**: Uses intraday close (iclose) instead of end-of-day close
   - hl.py: Uses daily close after market close
   - hl_intra.py: Uses intraday close during the trading day

2. **Data Source**:
   - hl.py: Daily OHLC bars only
   - hl_intra.py: Intraday bars (30-min) merged with daily OHLC

3. **Signal Updates**:
   - hl.py: Once per day (EOD)
   - hl_intra.py: Multiple times per day (every bar timestamp)

4. **Signal Combination**:
   - hl.py: Combines hlC (intraday) + hl0, hl1, ... hlN (lagged daily signals)
   - hl_intra.py: Uses only hlC_B_ma (intraday) + lagged daily hl signals from daily_df

5. **Use Case**:
   - hl.py: Daily rebalancing strategies (bsim.py)
   - hl_intra.py: Intraday rebalancing strategies (qsim.py)

6. **Regression Scope**:
   - hl.py: Fits both daily and intraday signals
   - hl_intra.py: Fits only intraday signal, borrows daily coefficients

Signal Generation:
------------------
1. **Intraday Signal (hlC)**:
   - Calculated as: iclose / sqrt(dhigh * dlow)
   - Uses intraday close vs. accumulated daily high/low
   - Provides real-time update as day progresses
   - Winsorized to remove outliers (hlC_B)
   - Industry-demeaned at each timestamp (hlC_B_ma)

2. **Combined Forecast**:
   - Weighted combination of intraday and lagged daily signals
   - Regression-fitted coefficient for hlC_B_ma
   - Borrows coefficients from daily hl signals for lags
   - Separate fits for in-sector (Energy) and ex-sector universes

Expected Characteristics:
-------------------------
- **Holding Period**: Intraday to 1 day (shorter than daily hl.py)
- **Signal Range**: Typically [-3, +3] after winsorization and demeaning
- **Turnover**: Very high (signals update every bar, multiple times per day)
- **Mean Reversion**: Negative coefficients expected (price reverts to mean)
- **Industry Neutral**: Demeaning removes sector effects
- **Intraday Sensitivity**: Captures real-time dislocations during trading

Parameters and Tuning:
---------------------
- **horizon** (int): Number of days ahead to forecast (default: 3)
  - Controls number of lagged daily signals included
  - Typical range: 1-5 days
  - Higher horizon captures slower mean reversion

- **sector_name** (str): Sector for separate in/ex fitting (default: 'Energy')
  - Allows sector-specific coefficient tuning
  - Energy often has different intraday volatility characteristics

Signal Interpretation:
---------------------
The final forecast combines intraday and daily signals:

    forecast = hlC_B_ma * coef_C + sum(lag=0 to horizon-1: hl{lag}_B_ma * coef_{lag})

Where:
- hlC_B_ma: Industry-demeaned intraday signal (this file calculates)
- hl{lag}_B_ma: Industry-demeaned daily signals from daily_df (imported)
- coef_C: Regression-fitted coefficient for intraday signal
- coef_{lag}: Coefficients for lagged daily signals

Positive forecast value: Expect price to increase (buy signal)
Negative forecast value: Expect price to decrease (sell signal)

Note: Line 46 has a bug where `lag` is used before being defined in the loop.
This should be fixed to properly assign lagged signal coefficients.

Usage Example:
-------------
Standalone execution:
    python hl_intra.py

This will:
1. Load 30 days of historical data from 20120601 to 20120610
2. Calculate intraday hlC signals from 30-min bars
3. Run separate regressions for Energy sector vs. other sectors
4. Generate forecast plots showing coefficient values
5. Dump alpha signals to output file

Integration with intraday backtesting:
    python qsim.py --start=20130101 --end=20130630 --fcast=hl_intra --horizon=3

For intraday rebalancing with 30-minute bars.

Multi-alpha combination:
    python qsim.py --fcast=hl_intra:1:0.6,qhl_intra:0.8:0.4 --horizon=3

Combines intraday high-low (60% weight) with quote-based intraday (40% weight).

Related Strategies:
------------------
- hl.py: Daily high-low mean reversion (parent strategy)
- qhl_intra.py: Quote-based intraday high-low (alternative intraday variant)
- qhl_multi.py: Multiple timeframe high-low combination
- qhl_both.py: Combined approach with additional metrics

Data Requirements:
-----------------
- Intraday bars (30-min): iclose, dhigh, dlow, iclose_ts
- Daily prices: Required for lagged hl signals (from daily_df)
- Barra factors: ind1 (industry classification), sector_name
- Universe filters: Expandable universe (liquid stocks)

Output:
------
Generates forecast column in the merged DataFrame containing intraday alpha forecasts.
Also produces diagnostic plots:
- hl_in_intra_<dates>.png: Energy sector intraday fit
- hl_ex_intra_<dates>.png: Ex-Energy intraday fit

Implementation Notes:
--------------------
- Only calculates intraday hlC signal, relies on daily_df for lagged signals
- Industry demeaning done per timestamp to capture cross-sectional effects
- Drops NaN rows to ensure complete intraday data coverage
- Uses left merge to preserve all timestamps from full_df
- Bug on line 46: `lag` undefined before use in coefficient assignment
"""

from alphacalc import *

from dateutil import parser as dateparser

def calc_hl_intra(full_df):
    """
    Calculate intraday high-low mean reversion signals for real-time trading.

    Computes the deviation of intraday closing price from the geometric mean
    of the day's accumulated high and low prices. This provides a real-time
    signal that updates throughout the trading day as new intraday bars arrive.

    Signal Formula:
        hlC = iclose / sqrt(dhigh * dlow)

    Where:
        - iclose: Intraday closing price (e.g., 3:00 PM close from 30-min bar)
        - dhigh: Daily high price accumulated through the trading day
        - dlow: Daily low price accumulated through the trading day
        - sqrt(dhigh * dlow): Geometric mean of daily extremes

    The signal is then:
        1. Winsorized to handle outliers (hlC_B)
        2. Industry-demeaned within each timestamp and ind1 group (hlC_B_ma)

    This differs from calc_hl_intra() in hl.py by:
        - Using only intraday data (no daily signal calculation)
        - Focusing exclusively on the hlC signal
        - Designed for standalone intraday strategy rather than combined daily+intraday

    Args:
        full_df (DataFrame): Input DataFrame with MultiIndex (iclose_ts, sid) containing:
            - iclose (float): Intraday closing prices from 30-min bars
            - dhigh (float): Daily high prices (accumulated intraday)
            - dlow (float): Daily low prices (accumulated intraday)
            - ind1 (int): Barra industry classification (1-58)
            - sid (int): Security identifier
            - iclose_ts (datetime): Intraday timestamp (e.g., 2012-06-01 15:00:00)
            - date (datetime): Trading date (removed before merge to avoid NaT issues)

    Returns:
        DataFrame: full_df merged with new columns:
            - hlC (float): Raw intraday high-low ratio signal
            - hlC_B (float): Winsorized intraday signal (outliers capped)
            - hlC_B_ma (float): Industry-demeaned intraday signal (market-neutral)

    Notes:
        - Only includes expandable universe stocks (liquid, tradeable)
        - Drops rows with any NaN values (requires complete intraday data)
        - Industry demeaning is done within each timestamp (iclose_ts) to remove
          cross-sectional market effects while preserving stock-specific signals
        - The 'date' column is deleted before merge to prevent NaT contamination
          in the MultiIndex join operation
        - Returns left merge preserving all timestamps from full_df
        - Debug output shows tail of result_df to verify calculation

    Example:
        A stock with iclose=$50, dhigh=$52, dlow=$48:
            hlC = 50 / sqrt(52 * 48) = 50 / 49.98 = 1.0004
            (Close is slightly above geometric mean, weak short signal)

        After industry demeaning, if industry average hlC_B is 1.002:
            hlC_B_ma = 1.0004 - 1.002 = -0.0016
            (Stock is below industry average, weak long signal)
    """
    print "Calculating hl intra..."
    result_df = full_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['iclose_ts', 'iclose', 'dhigh', 'dlow', 'date', 'ind1', 'sid' ] ]
    result_df = result_df.dropna(how='any')

    print "Calulating hlC..."
    print result_df.tail()
    result_df['hlC'] = result_df['iclose'] / np.sqrt(result_df['dhigh'] * result_df['dlow'])
    result_df['hlC_B'] = winsorize(result_df['hlC'])

    print "Calulating hlC_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['hlC_B', 'iclose_ts', 'ind1']].groupby(['iclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['hlC_B_ma'] = indgroups['hlC_B']

    #important for keeping NaTs out of the following merge
    del result_df['date']

    print "Merging..."
    result_df.set_index(keys=['iclose_ts', 'sid'], inplace=True)
    result_df = pd.merge(full_df, result_df, how='left', left_index=True, right_index=True, sort=True, suffixes=['_dead', ''])
    result_df = remove_dup_cols(result_df)

    return result_df

def hl_fits(daily_df, intra_df, full_df, horizon, name):
    """
    Run intraday regression to fit high-low signals and generate forecasts.

    Performs weighted least squares regression of the intraday hlC_B_ma signal
    against forward returns to determine the optimal coefficient. Creates a
    diagnostic plot and populates the forecast column with the signal multiplied
    by its coefficient.

    This is a simplified version of hl_fits() in hl.py that:
        - Only fits the intraday signal (not daily signals)
        - Uses a fixed horizon of 1 day for intraday regression
        - Combines intraday signal with lagged daily signals from daily_df
        - Has a bug on line 46 where `lag` is undefined before use

    Regression Process:
        1. Fit intraday signal (hlC_B_ma) against 1-day forward returns
        2. Extract coefficient showing signal strength
        3. Generate forecast as weighted sum of intraday + lagged daily signals

    Forecast Formula:
        forecast = hlC_B_ma * coef0 + sum(lag=0 to horizon-1: hl{lag}_B_ma * coef_{lag})

    Note: The code has a bug on line 46 where it references `lag` before the loop
    that defines it. This line attempts to set 'hl_intra{lag}_B_ma_coef' but should
    likely be 'hlC_B_ma_coef' or be placed inside the loop.

    Args:
        daily_df (DataFrame): Daily data (not used in this function, but passed for API consistency)
        intra_df (DataFrame): Intraday data with hlC_B_ma signal and forward returns
        full_df (DataFrame): Combined DataFrame to populate with forecasts
        horizon (int): Number of days ahead for combining lagged daily signals (not used for fitting)
        name (str): Column name for the forecast output (e.g., "hl_in" or "hl_ex")

    Returns:
        DataFrame: full_df with new columns:
            - {name} (float): Final alpha forecast combining intraday + lagged daily signals
            - hlC_B_ma_coef (float): Intraday signal coefficient
            - hl_intra{lag}_B_ma_coef (float): Coefficient (bug: lag undefined)
            - hl{lag}_B_ma_coef (float): Implied coefficients for lagged daily signals

    Side Effects:
        Generates diagnostic plot:
            - {name}_intra_{dates}.png: Intraday regression results showing:
              * Coefficient value
              * T-statistic
              * Number of observations
              * Standard error

    Notes:
        - Uses regress_alpha_intra() from regress.py for fitting
        - Fixed horizon=1 for intraday regression (overwrites parameter)
        - Assumes daily_df contains lagged hl signals (hl0_B_ma, hl1_B_ma, etc.)
        - Coefficient is applied globally to all rows in full_df (not limited to intra_df timestamps)
        - Bug: Line 46 should be inside the loop or reference a specific lag value

    Example:
        If regression yields coef0 = -0.5 for hlC_B_ma at horizon 1:
            - Stocks with hlC_B_ma = 0.02 get forecast contribution of -0.01
            - Negative coefficient confirms mean reversion (high values -> negative returns)
    """
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fits_df = fits_df.append(regress_alpha_intra(intra_df, 'hlC_B_ma', 1), ignore_index=True)
    plot_fit(fits_df, name + "_intra_" + df_dates(intra_df))
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])

    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)
    horizon = 1
    coef0 = fits_df.ix['hlC_B_ma'].ix[horizon].ix['coef']

    #should only set where key in daily_df
    full_df[ 'hlC_B_ma_coef' ] = coef0
    full_df[ 'hl_intra'+str(lag)+'_B_ma_coef' ] = coef0  # BUG: 'lag' is not defined yet

    full_df[name] = full_df['hlC_B_ma'] * full_df['hlC_B_ma_coef']
    for lag in range(0,horizon):
        full_df[name] += full_df['hl'+str(lag)+'_B_ma'] * full_df['hl'+str(lag)+'_B_ma_coef']

    return full_df

def calc_hl_forecast(daily_df, intra_df, horizon):
    """
    Generate complete intraday high-low mean reversion alpha forecasts.

    Main entry point that orchestrates the intraday HL signal pipeline: intraday
    signal calculation, sector-specific regression fitting, and alpha forecast
    generation. Produces separate coefficient fits for Energy sector vs. all other
    sectors to capture sector-specific intraday volatility characteristics.

    This function is similar to calc_hl_forecast() in hl.py but:
        - Only calculates intraday signals (not daily hl signals)
        - Assumes daily_df already contains lagged hl signals (hl0_B_ma, hl1_B_ma, etc.)
        - Focuses on real-time intraday updates rather than end-of-day signals
        - Uses simpler regression (intraday only, not daily multi-horizon)

    Pipeline Steps:
        1. Calculate intraday hlC signals (calc_hl_intra)
        2. Merge daily and intraday data
        3. Fit intraday regressions separately for:
           - Energy sector (historically more volatile intraday)
           - Non-Energy sectors
        4. Generate forecasts combining intraday + lagged daily signals
        5. Dump alpha signals to output file

    Args:
        daily_df (DataFrame): Daily price and factor data with columns:
            - hl0_B_ma, hl1_B_ma, ...: Lagged daily hl signals (assumed pre-calculated)
            - ind1: Industry classification
            - sector_name: Sector grouping (e.g., 'Energy', 'Technology')
        intra_df (DataFrame): Intraday bar data with columns:
            - iclose: Intraday close prices from 30-min bars
            - dhigh, dlow: Accumulated daily high/low
            - iclose_ts: Intraday timestamp
            - ind1, sector_name: Classifications
        horizon (int): Number of lagged daily signals to include (typically 1-5)

    Returns:
        DataFrame: Merged DataFrame with forecast columns:
            - hl_in (float): Energy sector forecast
            - hl_ex (float): Ex-Energy sector forecast
            - hlC_B_ma_coef (float): Intraday signal coefficient
            - hl{lag}_B_ma_coef (float): Implied daily signal coefficients

    Side Effects:
        - Generates 2 diagnostic plots:
          * hl_in_intra_<dates>.png: Energy sector intraday fit
          * hl_ex_intra_<dates>.png: Ex-Energy intraday fit
        - Writes alpha forecasts via dump_alpha() to 'hl_intra' output file

    Notes:
        - Energy sector is treated separately due to different intraday volatility
        - Separate fits allow coefficients to adapt to sector characteristics
        - The final forecast uses sector-appropriate coefficients for each stock
        - Requires daily_df to have pre-calculated lagged hl signals
        - Assumes daily signals (hl0_B_ma, hl1_B_ma) are available in full_df after merge

    Usage:
        This is the main entry point for standalone execution:
            full_df = calc_hl_forecast(daily_df, intra_df, horizon=3)

        Or called from backtesting framework:
            qsim.py uses this to generate intraday alpha forecasts
    """
    intra_df = calc_hl_intra(intra_df)
    full_df = merge_intra_data(daily_df, intra_df)

    sector_name = 'Energy'
    print "Running hl for sector {}".format(sector_name)
    sector_df = daily_df[ daily_df['sector_name'] == sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] == sector_name ]
    full_df = hl_fits(sector_df, sector_intra_df, full_df, horizon, "hl_in")

    print "Running hl for sector {}".format(sector_name)
    sector_df = daily_df[ daily_df['sector_name'] != sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] != sector_name ]
    full_df = hl_fits(sector_df, sector_intra_df, full_df, horizon, "hl_ex")
    dump_alpha(full_df, 'hl_intra')

    return full_df

if __name__=="__main__":
    """
    Standalone execution for intraday high-low alpha generation and analysis.

    Loads historical intraday data, calculates intraday HL signals, runs regressions,
    and generates diagnostic plots. Useful for strategy development, parameter tuning,
    and analyzing intraday mean reversion characteristics.

    Default Parameters:
        start: 20120601 - Begin date for data loading
        end: 20120610 - End date (10 days for quick testing)
        lookback: 30 - Days of history before start date
        horizon: 3 - Number of lagged daily signals to include

    Data Loading Steps:
        1. get_uni(): Load stock universe (top ~1400 by market cap)
        2. load_barra(): Load Barra risk factors and industry classifications
        3. load_prices(): Load daily OHLC prices
        4. merge_barra_data(): Combine prices with Barra factors
        5. load_daybars(): Load intraday 30-minute bar data
        6. merge_intra_data(): Combine daily and intraday data

    Output:
        - full_df: Complete DataFrame with 'hl_in' and 'hl_ex' alpha forecasts
        - Intraday regression plots showing coefficients for Energy and ex-Energy
        - Alpha dump file 'hl_intra' for further analysis

    Notes:
        - Short date range (10 days) for quick testing
        - Requires intraday bar data (30-min bars) to be available
        - Generates separate fits for Energy vs. other sectors
        - Assumes daily_df has lagged hl signals (may need to pre-calculate)

    Example Modifications:
        # Longer test period
        end = "20120630"

        # More lagged signals
        horizon = 5

        # More lookback for stability
        lookback = 90

        # Different sector split
        In calc_hl_forecast(), change sector_name from 'Energy' to 'Technology'
    """
    start = "20120601"
    end = "20120610"
    lookback = 30
    horizon = 3

    start = dateparser.parse(start)
    end = dateparser.parse(end)

    uni_df = get_uni(start, end, lookback)
    barra_df = load_barra(uni_df, start, end)
    price_df = load_prices(uni_df, start, end)
    daily_df = merge_barra_data(price_df, barra_df)
    daybar_df = load_daybars(uni_df, start, end)
    intra_df = merge_intra_data(daily_df, daybar_df)

    full_df = calc_hl_forecast(daily_df, intra_df, horizon)


