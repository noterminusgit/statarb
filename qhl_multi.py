#!/usr/bin/env python

"""Multi-Timeframe High-Low Mean Reversion Strategy.

STRATEGY OVERVIEW
=================
Combines high-low mean reversion signals across multiple time horizons to capture
mean reversion effects at different frequencies. Unlike single-timeframe strategies
(hl.py, qhl_intra.py), this approach generates a composite forecast by weighting
signals from the current intraday period plus multiple lagged daily signals.

KEY INNOVATION: Multi-Timeframe Signal Combination
---------------------------------------------------
The strategy constructs a weighted sum of high-low signals:
    qhl_m = qhlC_B_ma * coef_C + sum(qhl_i_B_ma * coef_i for i=1..horizon-1)

Where:
    - qhlC_B_ma: Current intraday high-low signal (usually zero-weighted)
    - qhl_i_B_ma: Lagged daily high-low signals (i days back)
    - coef_i: Regression-derived weights that capture incremental predictive power

MULTI-TIMEFRAME METHODOLOGY
============================
1. Daily Signals (calc_qhl_daily):
   - Calculate daily hl ratio: qhl0 = close / sqrt(high * low)
   - Winsorize and industry-demean to create qhl0_B_ma
   - Generate lagged versions: qhl1_B_ma, qhl2_B_ma, ..., qhl{horizon}_B_ma

2. Intraday Signals (calc_qhl_intra):
   - Calculate intraday hl ratio: qhlC = iclose / sqrt(qhigh * qlow)
   - Winsorize and industry-demean to create qhlC_B_ma
   - Uses quote-based high/low from daily data with intraday close

3. Coefficient Estimation (qhl_fits):
   - Regress forward returns against qhl0_B_ma at each horizon 1..H
   - Extract incremental coefficients: coef_i = coef_H - coef_i
   - Set current intraday weight to zero (coef_C = 0)
   - Allows different weights for different lookback periods

4. Signal Combination (calc_qhl_forecast):
   - Multiply each timeframe signal by its coefficient
   - Sum to create composite forecast qhl_m
   - Apply sector-specific models (Energy vs non-Energy)

SIGNAL INTERPRETATION
======================
High-Low Ratio (qhl):
    - Values > 1.0: Price closer to recent high (short signal)
    - Values < 1.0: Price closer to recent low (long signal)
    - After demeaning: Positive = expensive vs industry, Negative = cheap

Multi-Timeframe Weights:
    - Longer lags typically get larger weights (stronger mean reversion)
    - Coefficients reflect incremental predictive power beyond shorter lags
    - Zero weight on current intraday signal avoids microstructure noise

PARAMETERS
==========
horizon : int (default: 3)
    Number of daily lags to include in multi-timeframe model.
    Typical range: 2-5 days. Higher horizons capture longer-term mean reversion.

outsample : bool (default: False)
    If True, splits data at midpoint: fit on first half, forecast on second half.
    If False, uses full period for both fitting and forecasting (in-sample).

lookback : int (default: 30)
    Days of history to load before start date for lag calculations.

SECTOR SPLITS
=============
The strategy fits separate models for Energy vs non-Energy sectors to account for
different mean reversion dynamics (e.g., commodity-linked vs non-commodity stocks).

USAGE
=====
Command-line:
    python qhl_multi.py --start=20130101 --end=20130630 --os=True

In QSIM:
    python qsim.py --start=20130101 --end=20130630 --fcast=qhl_m --horizon=3

MATHEMATICAL FORMULATION
=========================
Daily Signal (lag j):
    qhl_j = close_t / sqrt(high_t-j * low_t-j)
    qhl_j_B = winsorize(qhl_j)
    qhl_j_B_ma = qhl_j_B - mean(qhl_j_B | industry)

Intraday Signal:
    qhlC = iclose_t / sqrt(qhigh_t * qlow_t)
    qhlC_B_ma = demean(winsorize(qhlC))

Regression (in-sample):
    r_t+h = alpha + beta_h * qhl0_t + epsilon

Incremental Coefficients:
    coef_j = beta_H - beta_j  for j = 1..(H-1)
    coef_C = 0

Composite Forecast (out-of-sample):
    qhl_m_t = coef_C * qhlC_B_ma_t + sum(coef_j * qhl_j_B_ma_t for j=1..H-1)

EXPECTED CHARACTERISTICS
=========================
Signal Range: [-2, +2] after winsorization and demeaning
Turnover: Medium (daily rebalancing, multi-day persistence)
Decay: Slower than single-timeframe due to multiple lag components
Capacity: Medium (intraday execution at close)
Market Neutrality: Industry-neutral by construction

RELATIONSHIP TO OTHER STRATEGIES
==================================
- hl.py: Single daily timeframe baseline
- qhl_intra.py: Single intraday timeframe
- qhl_multi.py: THIS FILE - combines multiple daily + intraday timeframes
- qhl_both.py: Alternative combination approach
- qhl_both_i.py: Extended version with additional features

See also: hl_intra.py for pure intraday high-low strategy.

DATA REQUIREMENTS
=================
- Daily OHLC prices (price_df)
- Intraday 30-min bars with close prices (intra_df)
- Quote-based daily high/low (qhigh, qlow in daily_df)
- Barra industry classifications (ind1)
- Barra sector classifications (sector_name)

OUTPUT
======
Generates 'qhl_m' forecast column in full_df with multi-timeframe composite signal.
Also generates coefficient columns for diagnostic purposes:
    - qhlC_B_ma_coef
    - qhl1_B_ma_coef, qhl2_B_ma_coef, ..., qhl{H-1}_B_ma_coef
"""

from __future__ import division, print_function

from alphacalc import *

from dateutil import parser as dateparser

def calc_qhl_daily(daily_df, horizon):
    """Calculate daily high-low signals and their lagged versions.

    Generates the daily timeframe component of the multi-timeframe strategy by:
    1. Computing the high-low mean reversion ratio using quote-based highs/lows
    2. Winsorizing to handle outliers
    3. Industry-demeaning to create market-neutral signals
    4. Creating lagged versions to capture multi-day mean reversion patterns

    This function produces the base daily signals that will be combined with
    intraday signals in qhl_fits() to create the composite forecast.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily data indexed by (date, sid) containing:
            - close: Daily closing price
            - qhigh: Quote-based daily high (from tick data)
            - qlow: Quote-based daily low (from tick data)
            - ind1: Barra industry classification (55 industries)
            - Plus universe filter fields (required by filter_expandable)

    horizon : int
        Maximum number of daily lags to generate (e.g., 3 creates qhl1, qhl2, qhl3).
        Controls the lookback window for multi-day mean reversion patterns.
        Typical values: 2-5 days.

    Returns
    -------
    pd.DataFrame
        Enhanced daily_df with added columns:
            - qhl0: Raw high-low ratio = close / sqrt(qhigh * qlow)
            - qhl0_B: Winsorized version (cross-sectional outliers removed)
            - qhl0_B_ma: Industry-demeaned signal (market-neutral)
            - qhl1_B_ma, ..., qhl{horizon}_B_ma: Lagged versions (1-horizon days back)

        All added columns indexed by (date, sid).

    Signal Calculation
    ------------------
    Step 1 - High-Low Ratio:
        qhl0 = close / sqrt(qhigh * qlow)

        Interpretation:
            - Geometric mean of high/low represents "fair value" range midpoint
            - Ratio > 1: Close above midpoint (recently strong, mean revert down)
            - Ratio < 1: Close below midpoint (recently weak, mean revert up)

    Step 2 - Winsorization:
        qhl0_B = winsorize(qhl0)

        Removes cross-sectional outliers to prevent extreme values from dominating
        the portfolio. Applied by date group.

    Step 3 - Industry Demeaning:
        qhl0_B_ma = qhl0_B - mean(qhl0_B | industry)

        Creates industry-neutral signal by subtracting industry average. Isolates
        stock-specific mean reversion from sector-wide movements.

    Step 4 - Lag Generation:
        qhl_j_B_ma = qhl0_B_ma shifted by j days

        Creates historical versions of the signal for multi-timeframe combination.
        Each lag captures mean reversion at a different frequency.

    Multi-Timeframe Context
    ------------------------
    The lagged signals enable differential weighting of recent vs longer-term
    mean reversion patterns:
        - qhl1_B_ma: Yesterday's signal (1-day mean reversion)
        - qhl2_B_ma: 2-day-ago signal (2-day mean reversion)
        - qhl3_B_ma: 3-day-ago signal (3-day mean reversion)

    Later in qhl_fits(), regression determines optimal weights for each lag,
    allowing the model to capture complex multi-day reversal patterns.

    Notes
    -----
    - Uses quote-based high/low (qhigh, qlow) rather than OHLC bar high/low
      for more accurate intraday range measurement
    - filter_expandable() restricts to tradable universe (price, volume filters)
    - Lagged signals propagate NaNs: first 'horizon' days lack full lag set
    - Industry groups use Barra's 55-industry classification (ind1)

    See Also
    --------
    calc_qhl_intra : Calculates intraday component of multi-timeframe signal
    qhl_fits : Combines daily lags with intraday signal using regression weights
    hl.py : Single-timeframe daily high-low strategy (simpler baseline)
    """
    print("Caculating daily qhl...")

    result_df = daily_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['close', 'qhigh', 'qlow', 'date', 'ind1', 'sid' ]]

    print("Calculating qhl0...")
    result_df['qhl0'] = result_df['close'] / np.sqrt(result_df['qhigh'] * result_df['qlow'])
    result_df['qhl0_B'] = winsorize_by_group(result_df[ ['date', 'qhl0'] ], 'date')

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['qhl0_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=True).transform(demean)
    result_df['qhl0_B_ma'] = indgroups['qhl0_B']
    result_df.set_index(keys=['date', 'sid'], inplace=True)

    print("Calulating lags...")
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['qhl'+str(lag)+'_B_ma'] = shift_df['qhl0_B_ma']

    result_df = merge_daily_calcs(daily_df, result_df)
    return result_df

def calc_qhl_intra(intra_df, daily_df):
    """Calculate intraday high-low signals using current intraday close vs daily range.

    Generates the intraday timeframe component of the multi-timeframe strategy by:
    1. Computing high-low ratio using intraday close price vs daily quote range
    2. Winsorizing to handle outliers within each intraday timestamp
    3. Industry-demeaning to create market-neutral intraday signals

    This creates the "current" signal component that will be combined with lagged
    daily signals in qhl_fits(). Note that the intraday signal is typically given
    zero or near-zero weight to avoid microstructure noise.

    Parameters
    ----------
    intra_df : pd.DataFrame
        Intraday 30-minute bar data indexed by (date, sid, iclose_ts) containing:
            - iclose: Intraday closing price at the bar timestamp
            - iclose_ts: Bar close timestamp (e.g., 930, 1000, 1030, ..., 1600)
            - Plus fields required by filter_expandable_intra

    daily_df : pd.DataFrame
        Daily data indexed by (date, sid) containing:
            - qhigh: Quote-based daily high (used for range calculation)
            - qlow: Quote-based daily low (used for range calculation)
            - ind1: Barra industry classification
            - Fields needed for intraday filtering (price, volume, sector)

    Returns
    -------
    pd.DataFrame
        Enhanced intra_df with added columns:
            - qhlC: Raw intraday high-low ratio = iclose / sqrt(qhigh * qlow)
            - qhlC_B: Winsorized version (cross-sectional outliers removed)
            - qhlC_B_ma: Industry-demeaned intraday signal (market-neutral)

        Indexed by (date, sid, iclose_ts).

    Signal Calculation
    ------------------
    Step 1 - Intraday High-Low Ratio:
        qhlC = iclose / sqrt(qhigh * qlow)

        Interpretation:
            - Uses current intraday close vs full daily range (qhigh, qlow)
            - Ratio > 1: Intraday price above daily range midpoint (expensive)
            - Ratio < 1: Intraday price below daily range midpoint (cheap)
            - Measures intraday positioning within daily volatility envelope

    Step 2 - Winsorization:
        qhlC_B = winsorize(qhlC by iclose_ts)

        Removes outliers within each intraday timestamp across all stocks.
        Prevents extreme intraday moves from dominating signal.

    Step 3 - Industry Demeaning:
        qhlC_B_ma = qhlC_B - mean(qhlC_B | industry, iclose_ts)

        Creates industry-neutral intraday signal. Removes sector-wide intraday
        patterns to isolate stock-specific positioning.

    Multi-Timeframe Role
    ---------------------
    In the final multi-timeframe model (qhl_fits), qhlC_B_ma is typically assigned:
        - Zero weight (coef_C = 0): Avoids microstructure noise
        - Small weight: If intraday signal has incremental predictive power

    The primary signal comes from lagged daily components (qhl1, qhl2, etc.),
    while qhlC captures same-day information that is usually too noisy for direct use.

    Hybrid Timeframe Approach
    --------------------------
    This function uses a hybrid of intraday and daily data:
        - Numerator: Intraday close (high-frequency, current)
        - Denominator: Daily quote range (lower-frequency, stable)

    This combination provides more stable range estimates while still incorporating
    intraday price movements.

    Notes
    -----
    - filter_expandable_intra() ensures stock is tradable at the intraday timestamp
    - Uses quote-based daily range (qhigh, qlow) from daily_df
    - Dropna removes bars with missing intraday or daily range data
    - Industry groups by (iclose_ts, ind1) to demean within each bar time
    - Typically calculated for all intraday bars (930-1600 in 30-min increments)

    See Also
    --------
    calc_qhl_daily : Calculates daily lag components for multi-timeframe model
    qhl_fits : Determines optimal weights for combining intraday + daily signals
    qhl_intra.py : Standalone intraday high-low strategy (no daily component)
    """
    print("Calculating qhl intra...")

    result_df = filter_expandable_intra(intra_df, daily_df)
    result_df = result_df.reset_index()
    result_df = result_df[ ['iclose_ts', 'iclose', 'qhigh', 'qlow', 'date', 'ind1', 'sid' ] ]
    result_df = result_df.dropna(how='any')

    print("Calulating qhlC...")
    result_df['qhlC'] = result_df['iclose'] / np.sqrt(result_df['qhigh'] * result_df['qlow'])
    result_df['qhlC_B'] = winsorize_by_group(result_df[ ['iclose_ts', 'qhlC'] ], 'iclose_ts')

    print("Calulating qhlC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['qhlC_B', 'iclose_ts', 'ind1']].groupby(['iclose_ts', 'ind1'], sort=True).transform(demean)
    result_df['qhlC_B_ma'] = indgroups['qhlC_B']

    result_df = merge_intra_calcs(intra_df, result_df)
    return result_df

def qhl_fits(daily_df, intra_df, full_df, horizon, name, middate=None):
    """Fit regression models and generate multi-timeframe composite forecast.

    This is the core function that implements the multi-timeframe combination strategy.
    It performs three key steps:
    1. Regress forward returns against daily qhl signal at multiple horizons
    2. Calculate incremental coefficients to weight each timeframe component
    3. Construct composite forecast as weighted sum of all timeframe signals

    The incremental coefficient approach ensures each lag captures predictive power
    beyond shorter lags, avoiding redundancy in the multi-timeframe model.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily data with qhl signals (output of calc_qhl_daily).
        Must contain qhl0_B_ma and forward return columns.
        Used for regression fitting.

    intra_df : pd.DataFrame
        Intraday data with timestamps (must have 'date' column).
        Used to define in-sample vs out-of-sample splits.

    full_df : pd.DataFrame
        Combined daily+intraday data (output of merge_intra_data).
        Must contain all qhl signal columns (qhlC_B_ma, qhl1_B_ma, ..., qhl{H}_B_ma).
        This dataframe is modified in-place to add forecast and coefficient columns.

    horizon : int
        Maximum forecast horizon for regression (e.g., 3 for 3-day ahead returns).
        Controls number of lag coefficients: generates coef_1, ..., coef_{H-1}.

    name : str
        Identifier for this fit (e.g., "in" for Energy sector, "ex" for non-Energy).
        Used in plot filenames for diagnostic regression plots.

    middate : datetime, optional
        Date to split in-sample vs out-of-sample periods.
        If None, uses full period for both fitting and forecasting (in-sample).
        If provided, fits on data before middate, forecasts on data >= middate.

    Returns
    -------
    pd.DataFrame
        Modified full_df with added columns:
            - qhl_m: Multi-timeframe composite forecast
            - qhlC_B_ma_coef: Weight for intraday signal (typically 0)
            - qhl{j}_B_ma_coef: Weight for lag j daily signal (j=1..horizon-1)

    Multi-Timeframe Coefficient Methodology
    ----------------------------------------
    Step 1 - Horizon Regressions:
        For each lag h = 1, 2, ..., H:
            r_{t+h} = alpha_h + beta_h * qhl0_t + epsilon

        Where:
            - r_{t+h}: Forward return h days ahead
            - qhl0_t: Current daily high-low signal
            - beta_h: Coefficient for h-day ahead forecast

    Step 2 - Incremental Coefficients:
        The incremental coefficient for lag j measures the additional predictive
        power of the j-day-old signal beyond what's captured by more recent signals:

            coef_j = beta_H - beta_j

        Intuition:
            - beta_H: Total predictive power at maximum horizon
            - beta_j: Predictive power using only j-day horizon
            - Difference: Incremental value of longer horizon beyond j days

        Special case:
            coef_C = 0  (intraday signal weight set to zero)

    Step 3 - Composite Forecast:
        qhl_m_t = sum(coef_j * qhl_j_B_ma_t for j=1..H-1) + coef_C * qhlC_B_ma_t
                = coef_1 * qhl1_B_ma_t + coef_2 * qhl2_B_ma_t + ... + 0 * qhlC_B_ma_t

    Example with horizon=3:
        Regressions:
            r_{t+1} = alpha_1 + beta_1 * qhl0_t  -->  beta_1 = 0.05
            r_{t+2} = alpha_2 + beta_2 * qhl0_t  -->  beta_2 = 0.08
            r_{t+3} = alpha_3 + beta_3 * qhl0_t  -->  beta_3 = 0.10

        Incremental coefficients:
            coef_1 = beta_3 - beta_1 = 0.10 - 0.05 = 0.05
            coef_2 = beta_3 - beta_2 = 0.10 - 0.08 = 0.02
            coef_C = 0

        Forecast:
            qhl_m = 0.05 * qhl1_B_ma + 0.02 * qhl2_B_ma + 0 * qhlC_B_ma

    Why Incremental Coefficients?
    ------------------------------
    Direct use of beta coefficients would overweight signals since they're correlated.
    Incremental approach ensures each lag contributes unique information:
        - coef_1: Value of 1-day-old signal (most recent daily)
        - coef_2: Additional value of 2-day-old signal
        - coef_3: Additional value of 3-day-old signal (not included, already in beta_3)

    Typically coef_1 > coef_2 > coef_3, reflecting decay in incremental value.

    In-Sample vs Out-of-Sample
    ---------------------------
    If middate is None (in-sample):
        - Fit regressions on full daily_df
        - Generate forecasts on full intra_df
        - Coefficients applied to all periods (biased performance estimates)

    If middate is provided (out-of-sample):
        - Fit regressions on daily_df before middate
        - Generate forecasts on intra_df >= middate
        - Coefficients from first half applied to second half (unbiased estimates)

    Sector-Specific Models
    -----------------------
    This function is called separately for different sectors (e.g., Energy vs
    non-Energy) to allow sector-specific regression coefficients. Different sectors
    may have different mean reversion dynamics (e.g., commodity vs non-commodity).

    Output Columns
    --------------
    Coefficient columns (for diagnostics):
        - qhlC_B_ma_coef: Intraday signal weight
        - qhl1_B_ma_coef, qhl2_B_ma_coef, ...: Daily lag weights

    Forecast column:
        - qhl_m: Final composite multi-timeframe signal

    Diagnostic Outputs
    ------------------
    Creates regression diagnostic plot:
        - Filename: "qhl_daily_{name}_{start}_{end}.png"
        - Shows coefficient decay and t-statistics across horizons
        - Useful for validating mean reversion strength and persistence

    Notes
    -----
    - regress_alpha() performs WLS regression with Newey-West standard errors
    - plot_fit() generates coefficient decay and significance plots
    - Coefficients stored in full_df for transparency and debugging
    - Only lags 1 to horizon-1 get non-zero weights (lag 0 absorbed into beta_H)
    - Function modifies full_df in-place and returns it

    See Also
    --------
    calc_qhl_daily : Generates daily lag signals used as regressors
    calc_qhl_intra : Generates intraday signal component
    calc_qhl_forecast : Orchestrates full multi-timeframe pipeline
    regress_alpha : Performs individual horizon regressions
    """
    if 'qhl_m' not in full_df.columns:
        print("Creating forecast columns...")
        full_df['qhl_m'] = np.nan
        full_df[ 'qhlC_B_ma_coef' ] = np.nan
        for lag in range(1, horizon+1):
            full_df[ 'qhl' + str(lag) + '_B_ma_coef' ] = np.nan

    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    outsample = False
    if middate is not None:
        outsample = True
        insample_intra_df = intra_df[ intra_df['date'] <  middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate]

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'qhl0_B_ma', lag, outsample, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "qhl_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    print(df_dates(full_df))
    print(df_dates(intra_df))

    coef0 = fits_df.loc['qhl0_B_ma'].loc[horizon].loc['coef']
    full_df.loc[ outsample_intra_df.index, 'qhlC_B_ma_coef' ] = 0 #coef0
    print("Coef0: {}".format(coef0))
    for lag in range(1,horizon):
        coef = coef0 - fits_df.loc['qhl0_B_ma'].loc[lag].loc['coef']
        print("Coef{}: {}".format(lag, coef))
        full_df.loc[ outsample_intra_df.index, 'qhl'+str(lag)+'_B_ma_coef' ] = coef

    full_df.loc[ outsample_intra_df.index, 'qhl_m'] = full_df['qhlC_B_ma'] * full_df['qhlC_B_ma_coef']
    for lag in range(1,horizon):
        full_df.loc[ outsample_intra_df.index, 'qhl_m'] += full_df['qhl'+str(lag)+'_B_ma'] * full_df['qhl'+str(lag)+'_B_ma_coef']

    return full_df

def calc_qhl_forecast(daily_df, intra_df, horizon, outsample):
    """Generate multi-timeframe high-low composite forecast with sector-specific models.

    Main entry point that orchestrates the complete multi-timeframe strategy pipeline:
    1. Calculate daily lag signals (qhl0, qhl1, ..., qhl_H)
    2. Calculate intraday current signal (qhlC)
    3. Merge daily and intraday data into unified dataframe
    4. Fit sector-specific regression models (Energy vs non-Energy)
    5. Generate composite forecasts using incremental coefficient weighting
    6. Optionally filter to out-of-sample period

    This function coordinates all components of the multi-timeframe approach and
    produces the final 'qhl_m' forecast column used by the simulation engines.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLC and factor data indexed by (date, sid) containing:
            - close, qhigh, qlow: Price data for signal calculation
            - ind1: Barra industry classification (for demeaning)
            - sector_name: Barra sector classification (for sector splits)
            - Forward return columns (for regression fitting)
            - Universe filter fields (price, volume, market cap)

    intra_df : pd.DataFrame
        Intraday 30-minute bar data indexed by (date, sid, iclose_ts) containing:
            - iclose: Intraday close price at bar timestamp
            - iclose_ts: Bar timestamp (930, 1000, 1030, ..., 1600)
            - sector_name: Sector classification (must match daily_df)

    horizon : int
        Maximum forecast horizon for regressions (days).
        Controls number of lag coefficients in multi-timeframe model.
        Typical values: 2-5 days.
        Higher horizons capture longer-term mean reversion but increase lag.

    outsample : bool
        If True: Split data at midpoint, fit on first half, forecast on second half.
                 Returns only out-of-sample forecasts (unbiased performance).
        If False: Fit and forecast on full period (in-sample, biased performance).

    Returns
    -------
    pd.DataFrame
        Multi-index dataframe with intraday-frequency forecasts containing:
            - All daily columns (broadcasted to intraday frequency)
            - All intraday columns
            - qhl_m: Multi-timeframe composite forecast (primary output)
            - qhl{j}_B_ma: Daily lag signals (j=1..horizon)
            - qhlC_B_ma: Intraday current signal
            - qhl{j}_B_ma_coef: Coefficients for each component (diagnostic)

        If outsample=True, only returns data after middate.
        If outsample=False, returns full period.

    Pipeline Steps
    --------------
    1. calc_qhl_daily(daily_df, horizon):
        - Computes daily hl ratios: qhl0 = close / sqrt(qhigh * qlow)
        - Winsorizes and industry-demeans: qhl0_B_ma
        - Generates lags: qhl1_B_ma, qhl2_B_ma, ..., qhl{horizon}_B_ma

    2. calc_qhl_intra(intra_df, daily_df):
        - Computes intraday hl ratio: qhlC = iclose / sqrt(qhigh * qlow)
        - Winsorizes and industry-demeans: qhlC_B_ma

    3. merge_intra_data(daily_df, intra_df):
        - Broadcasts daily signals to intraday frequency
        - Creates unified dataframe for model fitting

    4. Sector Split and Model Fitting:
        - Energy sector: Commodity-linked stocks with distinct dynamics
        - Non-Energy: Rest of universe with different mean reversion patterns
        - Separate regressions allow sector-specific coefficients

    5. qhl_fits() for each sector:
        - Regresses forward returns on qhl0_B_ma at horizons 1..H
        - Calculates incremental coefficients: coef_j = beta_H - beta_j
        - Constructs forecast: qhl_m = sum(coef_j * qhl_j_B_ma)

    6. Out-of-Sample Filtering (if outsample=True):
        - Returns only data after middate for unbiased evaluation

    Sector-Specific Models
    -----------------------
    Why split Energy vs non-Energy?
        - Energy stocks: Driven by oil/gas prices, different volatility regime
        - Non-Energy: More diverse fundamental drivers, different mean reversion
        - Separate coefficients improve forecast accuracy for both groups

    The sector split is applied at the fitting stage (qhl_fits), but forecasts
    are combined in the returned full_df. Each stock uses coefficients from its
    sector's regression.

    In-Sample vs Out-of-Sample
    ---------------------------
    In-Sample (outsample=False):
        - middate = None
        - Regressions fit on full daily_df
        - Forecasts generated for full intra_df
        - Returns all periods
        - Use for: Parameter tuning, diagnostic analysis
        - Warning: Biased performance estimates (overfitting risk)

    Out-of-Sample (outsample=True):
        - middate = first_date + (last_date - first_date) / 2
        - Regressions fit on data before middate
        - Forecasts generated for data >= middate
        - Returns only data after middate
        - Use for: Unbiased performance evaluation, production simulation
        - Realistic: Uses only past data for coefficient estimation

    Multi-Timeframe Forecast Construction
    --------------------------------------
    For each stock at each intraday timestamp:

        qhl_m_t = coef_C * qhlC_B_ma_t + sum(coef_j * qhl_j_B_ma_t for j=1..H-1)

    Where coefficients come from sector-specific regressions:
        - Energy stocks: Use Energy sector coefficients
        - Non-Energy stocks: Use non-Energy sector coefficients

    Example with horizon=3:
        Energy stock forecast:
            qhl_m_t = 0 * qhlC_B_ma_t + 0.05 * qhl1_B_ma_t + 0.02 * qhl2_B_ma_t

        Non-Energy stock forecast:
            qhl_m_t = 0 * qhlC_B_ma_t + 0.04 * qhl1_B_ma_t + 0.03 * qhl2_B_ma_t

    Usage in Simulations
    --------------------
    The output qhl_m column can be used directly in intraday simulation engines:

        QSIM (intraday bars):
            python qsim.py --start=20130101 --end=20130630 --fcast=qhl_m --horizon=3

        BSIM (daily close):
            python bsim.py --start=20130101 --end=20130630 --fcast=qhl_m:1:1

    Diagnostic Outputs
    ------------------
    The function creates several diagnostic plots via qhl_fits():
        - qhl_daily_in_{dates}.png: Energy sector regression diagnostics
        - qhl_daily_ex_{dates}.png: Non-Energy sector regression diagnostics

    Plots show coefficient decay, t-statistics, and R-squared across horizons.

    Notes
    -----
    - Requires both daily and intraday data spanning same date range
    - Daily data needs at least 'horizon' days of history before start
    - Sector splits require 'sector_name' column in both dataframes
    - Coefficient columns retained in output for transparency
    - Horizon controls model complexity: higher = more lags but more parameters
    - Energy vs non-Energy split is hardcoded (could be generalized to all sectors)

    See Also
    --------
    calc_qhl_daily : Generates daily lag components
    calc_qhl_intra : Generates intraday component
    qhl_fits : Performs regression fitting and forecast construction
    qsim.py : Intraday simulator that uses qhl_m forecasts
    hl.py : Simpler single-timeframe daily high-low strategy
    """
    daily_df = calc_qhl_daily(daily_df, horizon)
    intra_df = calc_qhl_intra(intra_df, daily_df)
    full_df = merge_intra_data(daily_df, intra_df)

    middate = None
    if outsample:
        middate = intra_df.index[0][0] + (intra_df.index[len(intra_df)-1][0] - intra_df.index[0][0]) / 2
        print("Setting fit period before {}".format(middate))

    sector_name = 'Energy'
    print("Running qhl for sector {}".format(sector_name))
    sector_df = daily_df[ daily_df['sector_name'] == sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] == sector_name ]
    full_df = qhl_fits(sector_df, sector_intra_df, full_df, horizon, "in", middate)

    print("Running qhl for not sector {}".format(sector_name))
    sector_df = daily_df[ daily_df['sector_name'] != sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] != sector_name ]
    full_df = qhl_fits(sector_df, sector_intra_df, full_df, horizon, "ex", middate)

    if outsample:
        full_df = full_df[ full_df['date'] > middate ]
    return full_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--os",action="store",dest="outsample",default=False)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    outsample = args.outsample
    lookback = 30
    horizon = 3
    pname = "./qhl_m" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        barra_df = load_barra(uni_df, start, end)
        price_df = load_prices(uni_df, start, end)
        intra_df = load_daybars(uni_df, start, end)
        daily_df = merge_barra_data(price_df, barra_df)
        daily_df = merge_intra_eod(daily_df, intra_df)
        intra_df = merge_intra_data(daily_df, intra_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    full_df = calc_qhl_forecast(daily_df, intra_df, horizon, outsample)
    dump_alpha(outsample_df, 'qhl_m')
    dump_all(outsample_df)


