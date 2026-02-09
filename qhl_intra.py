#!/usr/bin/env python
"""
Quote-Based Intraday High-Low Mean Reversion Strategy (qhl_intra.py)

This module implements an intraday mean reversion strategy based on the deviation
of intraday closing prices from the geometric mean of quote-based high and low
prices (qhigh/qlow) rather than trade-based daily extremes (dhigh/dlow).

QUOTE-BASED vs TRADE-BASED DISTINCTION:
=======================================
The critical distinction of this "quote-based" strategy is the use of qhigh/qlow
instead of dhigh/dlow in the signal calculation:

- **qhigh/qlow**: Quote-based extremes - the highest bid and lowest ask observed
  during the intraday period. These represent the tightest spread bounds and may
  not correspond to actual executed trades. Quote prices capture liquidity dynamics
  and market maker positioning.

- **dhigh/dlow**: Trade-based extremes - actual executed trade prices representing
  the highest and lowest transaction prices during the period. Used in hl_intra.py.

The quote-based approach captures:
1. **Liquidity dynamics**: Bid-ask extremes reflect market maker behavior
2. **Tighter bounds**: Quotes typically provide narrower ranges than trades
3. **Microstructure effects**: Spread dynamics and order book imbalances
4. **Earlier signals**: Quote movements precede trade executions

STRATEGY OVERVIEW:
==================
The strategy generates mean reversion signals by:
1. Computing qhlC = iclose / sqrt(qhigh * qlow) for each intraday bar
2. Winsorizing to remove outliers (qhlC_B)
3. Industry-demeaning within each timestamp to create market-neutral signals (qhlC_B_ma)
4. Regressing signals against forward returns separately for Energy sector vs others
5. Using time-of-day dependent coefficients (6 hourly periods: 9:30-10:30, ..., 14:30-16:00)
6. Combining fitted coefficients with raw signals to produce forecast (qhl_i)

SIGNAL INTERPRETATION:
======================
- qhlC > 1.0: Price closed above geometric mean of quote extremes (bearish, expect reversion down)
- qhlC < 1.0: Price closed below geometric mean of quote extremes (bullish, expect reversion up)
- Industry demeaning removes sector-wide effects, isolating stock-specific deviations
- Separate Energy sector modeling captures distinct mean reversion dynamics in commodities

PARAMETERS:
===========
- horizon: Forward return horizon for regression (typically 1-6 intraday periods)
- freq: Intraday bar frequency (default: '15Min')
- middate: Split date for in-sample fitting vs out-sample prediction
- sector_name: Sector to model separately (default: 'Energy')

TYPICAL USAGE:
==============
    python qhl_intra.py --start=20130101 --end=20130630 --mid=20130401 --horizon=3 --freq=15Min

Creates alpha forecast 'qhl_i' combining time-dependent coefficients with industry-demeaned
quote-based high-low signals.

DIFFERENCES FROM hl_intra.py:
==============================
1. **Data source**: Uses qhigh/qlow (quote extremes) vs dhigh/dlow (trade extremes)
2. **Signal tightness**: Quote-based ranges are typically narrower, providing more sensitive signals
3. **Microstructure capture**: Incorporates bid-ask spread dynamics vs pure trade flow
4. **Timing**: Quote signals may lead trade-based signals as quotes update continuously

OUTPUT:
=======
Produces time-series forecast 'qhl_i' indexed by (iclose_ts, sid) representing expected
returns based on quote-based mean reversion dynamics.

DEPENDENCIES:
=============
- regress.py: WLS regression and alpha fitting infrastructure
- loaddata.py: Quote-based bar loading (load_daybars with qhigh/qlow columns)
- util.py: Data merging and utility functions
- calc.py: Forward return calculations and winsorization
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def calc_qhl_intra(intra_df):
    """
    Calculate quote-based high-low mean reversion signal for intraday bars.

    Computes the quote-based HL signal by comparing intraday closes to the
    geometric mean of quote-based high and low prices (qhigh/qlow). The key
    distinction from calc_hl_intra() is the use of quote extremes rather than
    trade extremes, capturing bid-ask spread dynamics.

    Signal Calculation Steps:
    1. Filter to expandable universe (liquid stocks meeting criteria)
    2. Compute raw signal: qhlC = iclose / sqrt(qhigh * qlow)
       - qhigh: Highest bid quote during the intraday period
       - qlow: Lowest ask quote during the intraday period
       - Geometric mean provides symmetric reference point
    3. Winsorize by timestamp to remove outliers (qhlC_B)
    4. Industry-demean within each timestamp (qhlC_B_ma)
       - Groups by (timestamp, ind1) to remove sector effects
       - Creates market-neutral, stock-specific signal

    Quote-Based Signal Properties:
    - Values > 1.0: Price closed above quote midpoint (bearish reversion)
    - Values < 1.0: Price closed below quote midpoint (bullish reversion)
    - Typically narrower range than trade-based signals
    - More sensitive to liquidity and spread dynamics
    - May provide earlier signals than trade-based approach

    Args:
        intra_df: DataFrame indexed by (iclose_ts, sid), required columns:
            - iclose: Intraday bar close price
            - qhigh: Quote-based high (highest bid)
            - qlow: Quote-based low (lowest ask)
            - giclose_ts: Global timestamp for grouping
            - ind1: Barra industry classification (for demeaning)
            - expandable: Universe filter flag

    Returns:
        DataFrame with added columns:
            - qhlC: Raw quote-based HL signal (close / sqrt(qhigh * qlow))
            - qhlC_B: Winsorized signal (outliers capped by timestamp)
            - qhlC_B_ma: Industry-demeaned signal (market-neutral)

        Index preserved as (iclose_ts, sid).

    Notes:
        - Quote extremes (qhigh/qlow) differ from trade extremes (dhigh/dlow):
          * Quotes represent bid-ask bounds, not executed trades
          * Typically provide tighter ranges
          * Capture market maker positioning and liquidity
        - Winsorization prevents extreme outliers from distorting regressions
        - Industry demeaning removes common sector movements
        - Only expandable stocks included (liquid, tradable universe)
    """
    print("Calculating qhl intra...")
    result_df = filter_expandable(intra_df)

    print("Calulating qhlC...")
    result_df['qhlC'] = result_df['iclose'] / np.sqrt(result_df['qhigh'] * result_df['qlow'])
    result_df['qhlC_B'] = winsorize_by_ts(result_df[ 'qhlC' ])

    print("Calulating qhlC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['qhlC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=True).transform(demean)
    result_df['qhlC_B_ma'] = indgroups['qhlC_B']

    print("Calculated {} values".format(len(result_df['qhlC_B_ma'].dropna())))
    return result_df

def qhl_fits(daily_df, intra_df, horizon, name, middate=None):
    """
    Fit quote-based HL signal to forward returns with time-of-day dependent coefficients.

    Performs WLS regression of qhlC_B_ma signal against forward returns on in-sample
    data, then applies fitted coefficients to out-sample data with time-of-day variation.
    Mean reversion strength varies by intraday period (morning vs afternoon dynamics).

    Time-of-Day Coefficient Structure:
    - Period 1 (09:30-10:30): Opening hour, high volatility, strong reversion
    - Period 2 (10:30-11:30): Mid-morning stabilization
    - Period 3 (11:30-12:30): Pre-lunch, lower volume
    - Period 4 (12:30-13:30): Lunch period, reduced liquidity
    - Period 5 (13:30-14:30): Afternoon positioning
    - Period 6 (14:30-16:00): Closing period, strong reversion to fair value

    Each period gets independent regression coefficient, capturing time-varying
    mean reversion dynamics.

    Args:
        daily_df: DataFrame, daily data with sector classifications (unused in function
            but maintained for API consistency)
        intra_df: DataFrame indexed by (iclose_ts, sid), must contain:
            - qhlC_B_ma: Industry-demeaned quote-based HL signal
            - date: Trading date for in-sample/out-sample split
            - ticker: Stock ticker (for time grouping)
            - Forward return columns (created by regress_alpha)
        horizon: int, forward return horizon in number of intraday bars (1-6 typical)
        name: str, identifier for output plots ("in" for Energy, "ex" for non-Energy)
        middate: datetime or None, split date between in-sample (< middate) and
            out-sample (>= middate). If None, uses entire period for both.

    Returns:
        DataFrame (out-sample portion) with added columns:
            - qhlC_B_ma_coef: Time-of-day dependent regression coefficient
            - qhl_i: Final forecast = qhlC_B_ma * qhlC_B_ma_coef

        Index preserved as (iclose_ts, sid).

    Side Effects:
        - Generates plot: "qhl_intra_{name}_{date_range}.png" showing:
          * Regression coefficients by horizon
          * T-statistics
          * Number of observations
          * Standard errors

    Process:
        1. Split data into in-sample (fitting) and out-sample (prediction)
        2. Run WLS regression on in-sample: regress_alpha('qhlC_B_ma', horizon)
        3. Extract 6 time-period coefficients from regression results
        4. Map coefficients to out-sample data based on timestamp
        5. Compute forecast: qhl_i = signal * coefficient

    Notes:
        - Time-dependent coefficients capture intraday reversion patterns
        - Separate fits for Energy vs non-Energy sectors (via name parameter)
        - WLS regression accounts for heteroskedasticity in returns
        - Out-sample forecast prevents look-ahead bias
    """
    insample_intra_df = intra_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['qhl_i'] = np.nan
    outsample_intra_df[ 'qhlC_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df = regress_alpha(insample_intra_df, 'qhlC_B_ma', horizon, True, 'intra_eod')
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "qhl_intra_"+name+"_" + df_dates(insample_intra_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)
    unstacked = outsample_intra_df[ ['ticker'] ].unstack()
    coefs = dict()
    coefs[1] = unstacked.between_time('09:30', '10:31').stack().index
    coefs[2] = unstacked.between_time('10:30', '11:31').stack().index
    coefs[3] = unstacked.between_time('11:30', '12:31').stack().index
    coefs[4] = unstacked.between_time('12:30', '13:31').stack().index
    coefs[5] = unstacked.between_time('13:30', '14:31').stack().index
    coefs[6] = unstacked.between_time('14:30', '15:59').stack().index
    print(fits_df.head())
    for ii in range(1,7):
        outsample_intra_df.loc[ coefs[ii], 'qhlC_B_ma_coef' ] = fits_df.loc['qhlC_B_ma'].loc[ii].loc['coef']

    outsample_intra_df[ 'qhl_i'] = outsample_intra_df['qhlC_B_ma'] * outsample_intra_df['qhlC_B_ma_coef']
    return outsample_intra_df

def calc_qhl_forecast(daily_df, intra_df, horizon, middate):
    """
    Generate quote-based high-low mean reversion forecast with sector-specific modeling.

    Orchestrates the complete quote-based HL signal pipeline:
    1. Calculate forward returns for regression targets
    2. Compute quote-based HL signals (qhlC_B_ma)
    3. Merge daily and intraday data with sector classifications
    4. Fit separate models for Energy sector vs all other sectors
    5. Combine forecasts into unified output

    Sector-Specific Modeling Rationale:
    - Energy stocks exhibit distinct mean reversion due to commodity exposure
    - Oil/gas price dynamics create different quote spread behavior
    - Separate coefficients capture Energy-specific microstructure
    - All other sectors pooled for statistical power

    Pipeline Flow:
        Daily Data + Forward Returns
              |
              v
        Intra Data --> calc_qhl_intra() --> qhlC_B_ma signals
              |
              v
        Merge daily + intra (adds sector_name, ind1)
              |
              v
        Split: Energy | Non-Energy
              |           |
              v           v
         qhl_fits()   qhl_fits()
          (in-sample fit, out-sample predict)
              |           |
              v           v
           qhl_i       qhl_i
              |           |
              +-----+-----+
                    v
              Combined forecast

    Args:
        daily_df: DataFrame indexed by (date, sid), must contain:
            - sector_name: Sector classification for split
            - Columns needed for calc_forward_returns
        intra_df: DataFrame indexed by (iclose_ts, sid), must contain:
            - iclose: Intraday close price
            - qhigh: Quote-based high (highest bid)
            - qlow: Quote-based low (lowest ask)
            - giclose_ts: Global timestamp
            - ind1: Industry classification
            - expandable: Universe filter
        horizon: int, forward return horizon in intraday bars (1-6 typical)
        middate: datetime, split date for in-sample/out-sample division

    Returns:
        DataFrame indexed by (iclose_ts, sid) containing:
            - qhl_i: Final quote-based HL forecast
            - qhlC_B_ma: Raw industry-demeaned signal
            - qhlC_B_ma_coef: Time-of-day dependent coefficient
            - All original intra_df columns
            - Merged daily data (sector_name, etc.)

        Coverage: All stocks in intra_df with valid qhigh/qlow data.

    Output Files:
        - "qhl_intra_in_{dates}.png": Energy sector regression diagnostics
        - "qhl_intra_ex_{dates}.png": Non-Energy regression diagnostics

    Notes:
        - Energy sector modeled separately due to commodity linkage
        - Time-of-day coefficients vary by sector
        - verify_integrity=True ensures no duplicate index entries
        - Forecast combines two independent models (no blending)
    """
    daily_results_df = daily_df
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    intra_results_df = calc_qhl_intra(intra_df)
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)

    sector_name = 'Energy'
    print("Running qhl for sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    result1_df = qhl_fits(sector_df, sector_intra_results_df, horizon, "in", middate)

    print("Running qhl for not sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] != sector_name ]
    result2_df = qhl_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)

    result_df = pd.concat([result1_df, result2_df], verify_integrity=True)
    return result_df

if __name__=="__main__":
    """
    Command-line interface for quote-based intraday high-low strategy.

    Usage Examples:
        # Basic usage with 3-bar horizon, 15-minute bars
        python qhl_intra.py --start=20130101 --end=20130630 --mid=20130401 --horizon=3

        # 30-minute bars with 5-bar horizon
        python qhl_intra.py --start=20130101 --end=20130630 --mid=20130401 --horizon=5 --freq=30Min

        # 5-minute bars for high-frequency signals
        python qhl_intra.py --start=20130101 --end=20130630 --mid=20130401 --horizon=1 --freq=5Min

    Arguments:
        --start: Start date (YYYYMMDD format)
        --end: End date (YYYYMMDD format)
        --mid: Split date for in-sample/out-sample (YYYYMMDD format, required)
        --horizon: Forward return horizon in number of bars (default: 0, recommend 1-6)
        --freq: Intraday bar frequency (default: '15Min', options: '5Min', '15Min', '30Min')

    Data Loading:
        - Attempts to load cached HDF5 files: qhl_i{start}.{end}_daily.h5 and _intra.h5
        - If cache miss, loads from raw data:
          * Universe: Top 1,400 stocks by market cap
          * Barra: Industry classifications (ind1)
          * Prices: Daily close prices
          * Intraday bars: close, qhigh, qlow at specified frequency
        - Caches results for subsequent runs

    Output:
        - Alpha forecast file: Contains 'qhl_i' column with predictions
        - Diagnostic plots: qhl_intra_in_{dates}.png, qhl_intra_ex_{dates}.png
        - Console output: Progress messages, regression statistics

    Performance Notes:
        - Initial run loads and caches data (~5-30 minutes depending on period)
        - Subsequent runs use cached data (~1-5 minutes)
        - Memory usage scales with universe size * intraday bars
        - Higher frequency bars (5Min) require more memory than 30Min
    """
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--freq",action="store",dest="freq",default='15Min')
    parser.add_argument("--horizon",action="store",dest="horizon",default=0)
    args = parser.parse_args()

    start = args.start
    end = args.end
    lookback = 30
    horizon = int(args.horizon)
    pname = "./qhl_i" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)
    freq = args.freq
    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        DBAR_COLS = ['close', 'qhigh', 'qlow']
        intra_df = load_daybars(price_df[['ticker']], start, end, DBAR_COLS, freq)

        daily_df = merge_barra_data(price_df, barra_df)
        daily_df = merge_intra_eod(daily_df, intra_df)
        intra_df = merge_intra_data(daily_df, intra_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    result_df = calc_qhl_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(result_df, 'qhl_i')
