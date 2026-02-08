#!/usr/bin/env python
"""
High-Low Mean Reversion Alpha Strategy

Generates alpha signals based on mean reversion of intraday price ranges relative
to daily high-low midpoints. This is the foundational high-low strategy that exploits
short-term price dislocations from the geometric mean of daily extremes.

Strategy Concept:
-----------------
The high-low mean reversion strategy is based on the empirical observation that when
closing prices deviate from the geometric mean of daily high and low prices, they tend
to revert toward that mean over subsequent days. This reflects a form of intraday
volatility normalization.

The core signal measures where the close is relative to the high-low range:

    hl0 = close / sqrt(high * low)

When hl0 > 1.0: Close is above the geometric midpoint (potential short signal)
When hl0 < 1.0: Close is below the geometric midpoint (potential long signal)

The signal is industry-demeaned to isolate stock-specific dislocations from
sector-wide movements, making it market-neutral within industries.

Signal Generation:
------------------
1. **Daily Signal (hl0)**:
   - Calculated as: close / sqrt(high * low)
   - Winsorized to remove outliers
   - Industry-demeaned (within ind1 industry groups)
   - Lagged versions (hl1, hl2, ...) capture multi-day decay

2. **Intraday Signal (hlC)**:
   - Calculated as: iclose / sqrt(dhigh * dlow)
   - Uses intraday close vs. daily high/low
   - Provides real-time update to daily signal
   - Industry-demeaned at each timestamp

3. **Combined Forecast**:
   - Weighted combination of intraday and lagged daily signals
   - Regression-fitted coefficients determine optimal weights
   - Separate fits for in-sector (Energy) and ex-sector universes
   - Coefficients decay with lag as mean reversion effect weakens

Expected Characteristics:
-------------------------
- **Holding Period**: 1-3 days (horizon parameter)
- **Signal Range**: Typically [-3, +3] after winsorization and demeaning
- **Turnover**: High (signals update daily and intraday)
- **Mean Reversion**: Negative coefficients expected (price reverts to mean)
- **Decay Pattern**: Coefficient magnitude decreases with lag
- **Industry Neutral**: Demeaning removes sector effects

Parameters and Tuning:
---------------------
- **horizon** (int): Number of days ahead to forecast (default: 3)
  - Controls number of lagged signals included
  - Typical range: 1-5 days
  - Higher horizon captures slower mean reversion

- **lookback** (int): Days of history for regression fitting (default: 30)
  - Used in main() for loading historical data
  - More data improves coefficient stability

- **sector_name** (str): Sector for separate in/ex fitting (default: 'Energy')
  - Allows sector-specific coefficient tuning
  - Energy often has different volatility characteristics

Signal Interpretation:
---------------------
The final 'hl' forecast is a weighted sum:

    hl = hlC_B_ma * coef_C + sum(lag=0 to horizon-1: hl{lag}_B_ma * coef_{lag})

Where:
- hlC_B_ma: Industry-demeaned intraday signal
- hl{lag}_B_ma: Industry-demeaned daily signal from {lag} days ago
- coef_*: Regression-fitted coefficients

Positive hl value: Expect price to increase (buy signal)
Negative hl value: Expect price to decrease (sell signal)

Note: Line 86 sets hl = 4.0 unconditionally, suggesting this may be a template
or the actual forecast generation is handled elsewhere in the pipeline.

Usage Example:
-------------
Standalone execution:
    python hl.py

This will:
1. Load 30 days of historical data from 20120601 to 20130101
2. Calculate daily and intraday hl signals
3. Run separate regressions for Energy sector vs. other sectors
4. Generate forecast plots showing coefficient decay
5. Dump alpha signals to output file

Integration with backtesting:
    python bsim.py --start=20130101 --end=20130630 --fcast=hl:1:1 --kappa=2e-8

Where the format is: hl:multiplier:weight
- multiplier: Scales the alpha magnitude (typically 1)
- weight: Portfolio allocation weight (0-1)

Multi-alpha combination:
    python bsim.py --fcast=hl:1:0.6,bd:0.8:0.4 --kappa=2e-8

Combines high-low (60% weight) with beta-adjusted order flow (40% weight).

Related Strategies:
------------------
- hl_intra.py: Intraday-only variant
- qhl_intra.py: Quote-based intraday high-low
- qhl_multi.py: Multiple timeframe high-low
- qhl_both.py: Combined approach with additional metrics

Data Requirements:
-----------------
- Daily prices: close, high, low
- Intraday bars: iclose (intraday close), dhigh/dlow (daily high/low)
- Barra factors: ind1 (industry classification), sector_name
- Universe filters: Expandable universe (liquid stocks)

Output:
------
Generates 'hl' column in the merged DataFrame containing alpha forecasts.
Also produces diagnostic plots:
- hl_intra_in_<dates>.png: Energy sector intraday fit
- hl_daily_in_<dates>.png: Energy sector daily fit
- hl_intra_ex_<dates>.png: Ex-Energy intraday fit
- hl_daily_ex_<dates>.png: Ex-Energy daily fit
"""

from __future__ import division, print_function

from alphacalc import *

from dateutil import parser as dateparser

def calc_hl_daily(full_df, horizon):
    """
    Calculate daily high-low mean reversion signals with multi-day lags.

    Computes the deviation of closing price from the geometric mean of daily
    high and low prices, then creates industry-demeaned signals at multiple
    lag horizons for mean reversion forecasting.

    Signal Formula:
        hl0 = close / sqrt(high * low)

    Where:
        - close: Daily closing price
        - high: Daily high price
        - low: Daily low price
        - sqrt(high * low): Geometric mean of high and low

    The signal is then:
        1. Winsorized to handle outliers (hl0_B)
        2. Industry-demeaned within each date and ind1 group (hl0_B_ma)
        3. Lagged 1 to (horizon-1) days to capture decay (hl1_B_ma, hl2_B_ma, ...)

    Args:
        full_df (DataFrame): Input DataFrame with MultiIndex (date, sid) containing:
            - close (float): Daily closing prices
            - high (float): Daily high prices
            - low (float): Daily low prices
            - ind1 (int): Barra industry classification
            - sid (int): Security identifier
        horizon (int): Number of forecast days ahead (creates lags 1 to horizon-1)

    Returns:
        DataFrame: full_df merged with new columns:
            - hl0 (float): Raw high-low ratio signal
            - hl0_B (float): Winsorized signal
            - hl0_B_ma (float): Industry-demeaned signal (current day)
            - hl1_B_ma through hl{horizon-1}_B_ma (float): Lagged signals

    Notes:
        - Only includes expandable universe stocks (liquid, tradeable)
        - Industry demeaning makes signal market-neutral within sectors
        - Lags allow combining multiple days of mean reversion information
        - Returns left merge preserving all rows from full_df
    """
    print("Caculating daily hl...")
    result_df = full_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['close', 'high', 'low', 'date', 'ind1', 'sid' ]]

    print("Calculating hl0...")
    result_df['hl0'] = result_df['close'] / np.sqrt(result_df['high'] * result_df['low'])
    result_df['hl0_B'] = winsorize(result_df['hl0'])

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['hl0_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['hl0_B_ma'] = indgroups['hl0_B']
    result_df.set_index(keys=['date', 'sid'], inplace=True)

    print("Calulating lags...")
    for lag in range(1,horizon):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['hl'+str(lag)+'_B_ma'] = shift_df['hl0_B_ma']

    result_df = pd.merge(full_df, result_df, how='left', left_index=True, right_index=True, sort=False, suffixes=['', '_dead'])
    result_df = remove_dup_cols(result_df)
    return result_df

def calc_hl_intra(full_df):
    """
    Calculate intraday high-low mean reversion signals.

    Computes the deviation of intraday closing price from the geometric mean
    of the day's high and low prices. This provides a real-time update to the
    daily high-low signal, useful for intraday rebalancing.

    Signal Formula:
        hlC = iclose / sqrt(dhigh * dlow)

    Where:
        - iclose: Intraday closing price (e.g., from 30-min bars)
        - dhigh: Daily high price (accumulated through the day)
        - dlow: Daily low price (accumulated through the day)
        - sqrt(dhigh * dlow): Geometric mean of daily extremes

    The signal is then:
        1. Winsorized to handle outliers (hlC_B)
        2. Industry-demeaned within each timestamp and ind1 group (hlC_B_ma)

    Args:
        full_df (DataFrame): Input DataFrame with MultiIndex (iclose_ts, sid) containing:
            - iclose (float): Intraday closing prices
            - dhigh (float): Daily high prices (updated intraday)
            - dlow (float): Daily low prices (updated intraday)
            - ind1 (int): Barra industry classification
            - sid (int): Security identifier
            - iclose_ts (datetime): Intraday timestamp
            - date (datetime): Trading date (removed before merge)

    Returns:
        DataFrame: full_df merged with new columns:
            - hlC (float): Raw intraday high-low ratio signal
            - hlC_B (float): Winsorized intraday signal
            - hlC_B_ma (float): Industry-demeaned intraday signal

    Notes:
        - Only includes expandable universe stocks
        - Drops rows with any NaN values (requires complete intraday data)
        - Industry demeaning is done within each timestamp to remove
          cross-sectional market effects
        - The 'date' column is deleted before merge to avoid NaT issues
        - Returns left merge preserving all timestamps from full_df
    """
    print("Calculating hl intra...")
    result_df = full_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['iclose_ts', 'iclose', 'dhigh', 'dlow', 'date', 'ind1', 'sid' ] ]
    result_df = result_df.dropna(how='any')

    print("Calulating hlC...")
    result_df['hlC'] = result_df['iclose'] / np.sqrt(result_df['dhigh'] * result_df['dlow'])
    result_df['hlC_B'] = winsorize(result_df['hlC'])

    print("Calulating hlC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['hlC_B', 'iclose_ts', 'ind1']].groupby(['iclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['hlC_B_ma'] = indgroups['hlC_B']

    #important for keeping NaTs out of the following merge
    del result_df['date']

    print("Merging...")
    result_df.set_index(keys=['iclose_ts', 'sid'], inplace=True)
    result_df = pd.merge(full_df, result_df, how='left', left_index=True, right_index=True, sort=True, suffixes=['_dead', ''])
    result_df = remove_dup_cols(result_df)

    return result_df

def hl_fits(daily_df, intra_df, full_df, horizon, name):
    """
    Run regressions to fit high-low signals and generate forecasts.

    Performs weighted least squares regression of high-low signals against forward
    returns to determine optimal coefficients. Creates diagnostic plots and populates
    the 'hl' forecast column with weighted combinations of signals.

    Regression Process:
        1. Fit intraday signal (hlC_B_ma) against 1-day forward returns
        2. Fit daily signals (hl0_B_ma) against multiple horizons
        3. Extract coefficients showing how signal strength decays with lag
        4. Generate forecast as weighted sum of signals

    Forecast Formula:
        hl = hlC_B_ma * coef_C + sum(lag=0 to horizon-1: hl{lag}_B_ma * coef_{lag})

    Where coefficients are:
        - coef_C: Coefficient from intraday regression
        - coef_0: Coefficient from horizon-day daily regression
        - coef_{lag}: Differential coefficient (coef_0 - coef at lag)

    Args:
        daily_df (DataFrame): Daily data with hl0_B_ma signal and forward returns
        intra_df (DataFrame): Intraday data with hlC_B_ma signal
        full_df (DataFrame): Combined DataFrame to populate with forecasts
        horizon (int): Number of days ahead to forecast
        name (str): Suffix for plot filenames ("in" or "ex" for in/ex sector)

    Returns:
        DataFrame: full_df with new columns:
            - hl (float): Final alpha forecast (NOTE: currently set to 4.0 on line 157)
            - hlC_B_ma_coef (float): Intraday signal coefficient
            - hl0_B_ma_coef through hl{horizon}_B_ma_coef (float): Daily coefficients

    Side Effects:
        Generates diagnostic plots:
            - hl_intra_{name}_{dates}.png: Intraday regression results
            - hl_daily_{name}_{dates}.png: Daily regression results across horizons

    Notes:
        - Uses regress_alpha_intra() and regress_alpha_daily() from regress.py
        - Coefficients are fitted on in-sample data (daily_df, intra_df)
        - Forecasts are populated only at intra_df.index timestamps
        - Line 157 overrides the calculated forecast with constant 4.0
          (may be placeholder or signal normalization step)
    """
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fits_df = fits_df.append(regress_alpha_intra(intra_df, 'hlC_B_ma', 1), ignore_index=True)
    plot_fit(fits_df, "hl_intra_"+name+"_" + df_dates(intra_df))
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])

    for lag in range(1,horizon+1):
        fits_df = fits_df.append(regress_alpha_daily(daily_df, 'hl0_B_ma', lag), ignore_index=True)
    plot_fit(fits_df, "hl_daily_" +name+"_"+ df_dates(daily_df))

    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)
    coef0 = fits_df.ix['hl0_B_ma'].ix[horizon].ix['coef']

    if 'hl' not in full_df.columns:
        print("Creating forecast columns...")
        full_df['hl'] = np.nan
        full_df[ 'hlC_B_ma_coef' ] = np.nan
        for lag in range(0, horizon+1):
            full_df[ 'hl' + str(lag) + '_B_ma_coef' ] = np.nan

    full_df.ix[ intra_df.index, 'hlC_B_ma_coef' ] = coef0
    full_df.ix[ intra_df.index, 'hl0_B_ma_coef' ] = coef0
    for lag in range(1,horizon+1):
        full_df.ix[ intra_df.index, 'hl'+str(lag)+'_B_ma_coef' ] = coef0 - fits_df.ix['hl0_B_ma'].ix[lag].ix['coef']

    full_df.ix[ intra_df.index, 'hl'] = full_df['hlC_B_ma'] * full_df['hlC_B_ma_coef']
    for lag in range(0,horizon):
        full_df.ix[ intra_df.index, 'hl'] += full_df['hl'+str(lag)+'_B_ma'] * full_df['hl'+str(lag)+'_B_ma_coef']

    full_df.ix[ intra_df.index, 'hl'] = 4.0

    return full_df

def calc_hl_forecast(daily_df, intra_df, horizon):
    """
    Generate complete high-low mean reversion alpha forecasts.

    Main entry point that orchestrates the full pipeline: signal calculation,
    sector-specific regression fitting, and alpha forecast generation. Produces
    separate coefficient fits for Energy sector vs. all other sectors to capture
    sector-specific mean reversion dynamics.

    Pipeline Steps:
        1. Calculate daily hl signals with lags (calc_hl_daily)
        2. Calculate intraday hl signals (calc_hl_intra)
        3. Merge daily and intraday data
        4. Fit regressions separately for:
           - Energy sector (historically more volatile)
           - Non-Energy sectors
        5. Generate forecasts using sector-appropriate coefficients
        6. Dump alpha signals to output file

    Args:
        daily_df (DataFrame): Daily price and factor data with columns:
            - close, high, low: Daily OHLC prices
            - ind1: Industry classification
            - sector_name: Sector grouping (e.g., 'Energy', 'Technology')
            - Forward return columns (for regression)
        intra_df (DataFrame): Intraday bar data with columns:
            - iclose: Intraday close prices
            - dhigh, dlow: Accumulated daily high/low
            - ind1, sector_name: Classifications
        horizon (int): Forecast horizon in days (typically 1-5)

    Returns:
        DataFrame: Merged DataFrame with 'hl' forecast column containing
                   alpha signals for all stocks and timestamps

    Side Effects:
        - Generates 4 diagnostic plots:
          * hl_intra_in_<dates>.png: Energy sector intraday fit
          * hl_daily_in_<dates>.png: Energy sector daily fit
          * hl_intra_ex_<dates>.png: Ex-Energy intraday fit
          * hl_daily_ex_<dates>.png: Ex-Energy daily fit
        - Writes alpha forecasts via dump_alpha() to output file

    Notes:
        - Energy sector is treated separately due to different volatility regime
        - Separate fits allow coefficients to adapt to sector characteristics
        - The final forecast uses whichever fit applies to each stock's sector
        - All stocks receive forecasts even if not in regression sample
    """
    daily_df = calc_hl_daily(daily_df, horizon)
    intra_df = calc_hl_intra(intra_df)
    full_df = merge_intra_data(daily_df, intra_df)

    sector_name = 'Energy'
    print("Running hl for sector {}".format(sector_name))
    sector_df = daily_df[ daily_df['sector_name'] == sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] == sector_name ]
    full_df = hl_fits(sector_df, sector_intra_df, full_df, horizon, "in")

    print("Running hl for sector {}".format(sector_name))
    sector_df = daily_df[ daily_df['sector_name'] != sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] != sector_name ]
    full_df = hl_fits(sector_df, sector_intra_df, full_df, horizon, "ex")

    dump_alpha(full_df, 'hl')

    return full_df

if __name__=="__main__":
    """
    Standalone execution for high-low alpha generation and analysis.

    Loads historical data, calculates high-low signals, runs regressions,
    and generates diagnostic plots. Useful for strategy development and
    parameter tuning.

    Default Parameters:
        start: 20120601 - Begin date for data loading
        end: 20130101 - End date for data loading
        lookback: 30 - Days of history before start date
        horizon: 3 - Forecast horizon in days

    Data Loading Steps:
        1. get_uni(): Load stock universe (top ~1400 by market cap)
        2. load_barra(): Load Barra risk factors and industry classifications
        3. load_prices(): Load daily OHLC prices
        4. merge_barra_data(): Combine prices with Barra factors
        5. load_daybars(): Load intraday bar data (30-min bars)
        6. merge_intra_data(): Combine daily and intraday data

    Output:
        - full_df: Complete DataFrame with 'hl' alpha forecasts
        - Regression plots showing coefficient decay
        - Alpha dump file for further analysis

    Example Modifications:
        # Longer forecast horizon
        horizon = 5

        # Different date range
        start = "20110101"
        end = "20120101"

        # More lookback for stability
        lookback = 90
    """
    start = "20120601"
    end = "20130101"
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


