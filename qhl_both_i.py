#!/usr/bin/env python
"""
qhl_both_i.py - High-Low Mean Reversion Strategy with Intraday Coefficients

This module implements a high-low mean reversion alpha strategy that combines daily
and intraday signals with sector-specific modeling. The "_i" suffix indicates this
variant uses time-of-day specific coefficients for the intraday component, unlike
qhl_both.py which uses a single coefficient across all intraday periods.

Key Differences from qhl_both.py:
    - Fits separate regression coefficients for each 1-hour intraday period (6 periods)
    - Uses hourly time buckets (09:30-10:31, 10:30-11:31, etc.) to capture time-varying
      mean reversion patterns throughout the trading day
    - More granular coefficient estimation for qhlC_B_ma (current intraday signal)
    - Requires --mid parameter for in-sample/out-of-sample split

Signal Formula:
    qhl_b = qhlC_B_ma * qhlC_B_ma_coef + sum(qhl{lag}_B_ma * qhl{lag}_B_ma_coef)

    Where:
    - qhlC: Current intraday high-low ratio = iclose / sqrt(qhigh * qlow)
    - qhlC_B_ma: Industry-demeaned, winsorized current intraday signal
    - qhlC_B_ma_coef: Time-of-day specific coefficient (varies by hour)
    - qhl{lag}_B_ma: Lagged daily high-low signals
    - qhl{lag}_B_ma_coef: Incremental coefficients from daily regressions

The strategy splits the universe into Energy sector vs non-Energy sector and fits
separate models for each, then combines the predictions.

Typical Usage:
    python qhl_both_i.py --start=20130101 --end=20130630 --mid=20130401

    In qsim.py:
    python qsim.py --start=20130101 --end=20130630 --fcast=qhl_both_i --horizon=3

Dependencies:
    - Daily OHLC data with quarterly high/low (qhigh, qlow)
    - Intraday bar data (15-min default) with quarterly high/low
    - Barra industry classifications (ind1)
    - Requires regress.py for WLS regression fitting
    - Requires loaddata.py for data loading and filtering
    - Requires util.py for data merging utilities

Output:
    Generates alpha forecasts stored as 'qhl_b' in HDF5 format for use by qsim.py.
    Also dumps intermediate signals for analysis: qhlC_B_ma, qhl0_B_ma, qhl1_B_ma, qhl2_B_ma.
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def calc_qhl_daily(daily_df, horizon):
    """
    Calculate daily high-low mean reversion signals with lagged features.

    Computes the ratio of close price to geometric mean of quarterly high and low,
    then winsorizes and industry-demeans the signal. Creates lagged versions of
    the signal for use as predictive features.

    Args:
        daily_df: DataFrame with daily OHLC data indexed by (date, ticker)
                  Must contain columns: close, qhigh, qlow, ind1, gdate
        horizon: Integer number of days to create lagged features (typically 3)

    Returns:
        DataFrame with additional columns:
            - qhl0: Raw high-low ratio = close / sqrt(qhigh * qlow)
            - qhl0_B: Winsorized high-low ratio
            - qhl0_B_ma: Industry-demeaned high-low ratio (the base signal)
            - qhl{1..horizon}_B_ma: Lagged versions of qhl0_B_ma

    Signal Interpretation:
        qhl0 > 1: Close is high relative to quarterly range (potential reversion down)
        qhl0 < 1: Close is low relative to quarterly range (potential reversion up)

    The industry demeaning removes sector-wide effects, isolating stock-specific
    deviations from the quarterly high-low range.
    """
    print("Caculating daily qhl...")
    result_df = filter_expandable(daily_df)

    print("Calculating qhl0...")
    result_df['qhl0'] = result_df['close'] / np.sqrt(result_df['qhigh'] * result_df['qlow'])
    result_df['qhl0_B'] = winsorize_by_date(result_df[ 'qhl0' ])

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['qhl0_B', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    result_df['qhl0_B_ma'] = indgroups['qhl0_B']

    print("Calulating lags...")
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['qhl'+str(lag)+'_B_ma'] = shift_df['qhl0_B_ma']

    return result_df

def calc_qhl_intra(intra_df):
    """
    Calculate intraday high-low mean reversion signal.

    Computes the ratio of bar close to geometric mean of quarterly high and low
    for each intraday bar, then winsorizes by timestamp and industry-demeans.

    Args:
        intra_df: DataFrame with intraday bar data indexed by (timestamp, ticker)
                  Must contain columns: iclose, qhigh, qlow, ind1, giclose_ts

    Returns:
        DataFrame with additional columns:
            - qhlC: Raw intraday high-low ratio = iclose / sqrt(qhigh * qlow)
            - qhlC_B: Winsorized intraday high-low ratio
            - qhlC_B_ma: Industry-demeaned intraday signal (the current intraday feature)

    Signal Interpretation:
        This captures intraday mean reversion relative to the quarterly range.
        The signal is computed at each bar interval (typically 15-minute bars)
        and demeaned within each industry to remove sector effects.

    Note:
        Uses winsorize_by_ts() for timestamp-based winsorization rather than
        winsorize_by_date() to handle intraday volatility patterns.
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
    Fit regression models and generate alpha forecasts with time-of-day specific coefficients.

    This function performs a two-stage regression process:
    1. Intraday regression: Fits qhlC_B_ma signal separately for 6 hourly time buckets
       to capture time-varying mean reversion patterns throughout the trading day
    2. Daily regression: Fits qhl0_B_ma at multiple horizons to get lagged coefficients

    KEY DIFFERENCE FROM qhl_both.py:
    This variant assigns different coefficients to qhlC_B_ma based on the time of day,
    allowing the model to capture that mean reversion strength varies by hour.

    Args:
        daily_df: DataFrame with daily signals and forward returns
        intra_df: DataFrame with intraday signals
        horizon: Number of days ahead to predict (typically 3)
        name: String identifier for plot filenames ("in" for Energy, "ex" for non-Energy)
        middate: Optional date to split in-sample (before) vs out-of-sample (on/after)

    Returns:
        DataFrame with alpha forecast 'qhl_b' and intermediate coefficient columns:
            - qhlC_B_ma_coef: Time-of-day specific coefficient for current intraday signal
            - qhl{1..horizon-1}_B_ma_coef: Incremental coefficients for lagged daily signals

    Time Buckets (Eastern Time):
        1: 09:30-10:31 (market open hour)
        2: 10:30-11:31 (mid-morning)
        3: 11:30-12:31 (late morning)
        4: 12:30-13:31 (early afternoon)
        5: 13:30-14:31 (mid-afternoon)
        6: 14:30-15:59 (market close hour)

    The final forecast combines:
        qhl_b = qhlC_B_ma * qhlC_B_ma_coef(t) + sum(qhl{lag}_B_ma * qhl{lag}_B_ma_coef)

    where qhlC_B_ma_coef(t) varies by the hour of day.
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['qhl_b'] = np.nan
    outsample_intra_df[ 'qhlC_B_ma_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_intra_df[ 'qhl' + str(lag) + '_B_ma_coef' ] = np.nan

    # Fit intraday regression for each hour bucket to get time-varying coefficients
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df = regress_alpha(insample_intra_df, 'qhlC_B_ma', horizon, True, 'intra')
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "qhl_intra_"+name+"_" + df_dates(insample_intra_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    # Assign time-of-day specific coefficients (key difference from qhl_both.py)
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

    # Fit daily regressions at multiple horizons
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'qhl0_B_ma', lag, True, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "qhl_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    # Calculate incremental coefficients for lagged daily signals
    coef0 = fits_df.loc['qhl0_B_ma'].loc[horizon].loc['coef']
    print("Coef{}: {}".format(0, coef0))
#    outsample_intra_df[ 'qhlC_B_ma_coef' ] = coef0
    for lag in range(1,horizon):
        coef = coef0 - fits_df.loc['qhl0_B_ma'].loc[lag].loc['coef']
        print("Coef{}: {}".format(lag, coef))
        outsample_intra_df[ 'qhl'+str(lag)+'_B_ma_coef' ] = coef

    # Combine current intraday signal (with time-varying coef) and lagged daily signals
    outsample_intra_df[ 'qhl_b'] = outsample_intra_df['qhlC_B_ma'] * outsample_intra_df['qhlC_B_ma_coef']
    for lag in range(1,horizon):
        outsample_intra_df[ 'qhl_b'] += outsample_intra_df['qhl'+str(lag)+'_B_ma'] * outsample_intra_df['qhl'+str(lag)+'_B_ma_coef']

    return outsample_intra_df

def calc_qhl_forecast(daily_df, intra_df, horizon, middate):
    """
    Generate high-low mean reversion forecasts with sector-specific models.

    This function orchestrates the complete alpha generation pipeline:
    1. Calculate daily high-low signals and lagged features
    2. Calculate intraday high-low signals
    3. Fit separate models for Energy sector vs non-Energy sector
    4. Combine predictions from both sector models

    Args:
        daily_df: DataFrame with daily OHLC data indexed by (date, ticker)
        intra_df: DataFrame with intraday bar data indexed by (timestamp, ticker)
        horizon: Number of days ahead to predict (typically 3)
        middate: Date to split in-sample vs out-of-sample for walk-forward testing

    Returns:
        DataFrame indexed by (timestamp, ticker) with alpha forecast 'qhl_b'

    Sector Modeling:
        The strategy fits separate regression models for:
        - Energy sector stocks ("in" model)
        - Non-Energy sector stocks ("ex" model)

        This allows the model to capture sector-specific mean reversion dynamics.
        Energy stocks may have different high-low reversion patterns due to
        commodity price correlations and volatility characteristics.

    Process Flow:
        1. calc_qhl_daily(): Compute daily signals and lags
        2. calc_qhl_intra(): Compute intraday signals
        3. merge_intra_data(): Merge daily features into intraday DataFrame
        4. qhl_fits(): Fit regressions and generate forecasts (2x, once per sector)
        5. Concatenate sector-specific predictions into final result

    The final output is ready for use by qsim.py intraday simulation engine.
    """
    daily_results_df = calc_qhl_daily(daily_df, horizon)
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
    Main execution block for standalone alpha generation.

    Command-line Arguments:
        --start: Start date in YYYYMMDD format (e.g., 20130101)
        --end: End date in YYYYMMDD format (e.g., 20130630)
        --mid: Midpoint date for in-sample/out-of-sample split (e.g., 20130401)

    Parameters:
        lookback: 30 days for universe definition
        horizon: 3 days forward prediction horizon
        freq: '15Min' intraday bar frequency

    Data Requirements:
        - Daily close prices
        - Intraday bars (15-min) with close, qhigh, qlow
        - Barra industry classifications (ind1)

    Output Files:
        - qhl_b_<start>.<end>_daily.h5: Cached daily data
        - qhl_b_<start>.<end>_intra.h5: Cached intraday data
        - qhl_b.h5: Final alpha forecast for qsim.py
        - qhlC_B_ma.h5: Current intraday signal (for analysis)
        - qhl0_B_ma.h5, qhl1_B_ma.h5, qhl2_B_ma.h5: Lagged signals (for analysis)

    Example:
        python qhl_both_i.py --start=20130101 --end=20130630 --mid=20130401

    The generated qhl_b forecast can be used in qsim.py:
        python qsim.py --start=20130101 --end=20130630 --fcast=qhl_both_i --horizon=3
    """
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    args = parser.parse_args()

    start = args.start
    end = args.end
    lookback = 30
    horizon = 3
    pname = "./qhl_b" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)
    freq = '15Min'
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
    dump_alpha(result_df, 'qhl_b')
    dump_alpha(result_df, 'qhlC_B_ma')
    dump_alpha(result_df, 'qhl0_B_ma')
    dump_alpha(result_df, 'qhl1_B_ma')
    dump_alpha(result_df, 'qhl2_B_ma')



