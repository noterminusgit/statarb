#!/usr/bin/env python
"""Combined Daily and Intraday Beta-Adjusted Returns Strategy (badj_both)

Combines both daily lag signals and intraday signals using simple beta-division
approach. This is a comprehensive beta-adjusted strategy that captures both
multi-period momentum/reversal (daily lags) and intraday patterns.

IMPORTANT: This strategy does NOT use order flow despite "badj" name.
It uses simple return / beta ratio for beta adjustment.

Strategy Overview:
-----------------
This module implements a "both" approach combining:
1. Daily lags: badj0, badj1, badj2, ... (historical beta-adjusted returns)
2. Intraday: badjC (current intraday beta-adjusted return)

The forecast combines all components with fitted coefficients.

Key Differences from Related Strategies:
---------------------------------------
vs badj_multi.py:
- badj_multi.py: Sets intraday coefficient to 0 (daily-only)
- badj_both.py: Uses both intraday + daily lags
- More comprehensive signal combination

vs badj_intra.py:
- badj_intra.py: Intraday only with time-of-day coefficients
- badj_both.py: Combines intraday + daily lags
- Daily lags provide additional momentum information

vs bd.py (order flow):
- bd.py: Uses order flow imbalance (askHit - bidHit)
- badj_both.py: Uses simple return / beta ratio
- Completely different alpha sources

Methodology:
-----------
1. Daily Beta Adjustment:
   badj0 = log_ret / pbeta

   Simple division by beta to create beta-normalized returns.

2. Intraday Beta Adjustment:
   badjC = (overnight_log_ret + log(iclose/dopen)) / pbeta

   Same approach applied to cumulative intraday returns.

3. Winsorization and Industry Demeaning:
   - Winsorize by date/timestamp to control outliers
   - Industry demean for sector neutrality
   - Creates badj0_B_ma and badjC_B_ma signals

4. Multi-Period Lags:
   - Computes lagged daily signals (badj1, badj2, ..., badj{horizon})
   - Enables multi-horizon regression fitting

5. Regression:
   - Daily: Fits badj0_B_ma at multiple lags
   - Computes residual coefficients for each lag
   - Intraday: Uses current period coefficient (coef0)
   - Combines: badj_b = badjC * coef0 + sum(lagged * residual_coef)

Signal Formulas:
---------------
Daily:
    badj0 = log_ret / pbeta
    badj0_B = winsorize(badj0)
    badj0_B_ma = industry_demean(badj0_B)

Intraday:
    badjC = (overnight_log_ret + log(iclose/dopen)) / pbeta
    badjC_B = winsorize(badjC)
    badjC_B_ma = industry_demean(badjC_B)

Forecast:
    badj_b = badjC_B_ma * coef0 +
             sum_{lag=1}^{horizon-1} (badj{lag}_B_ma * residual_coef[lag])

where:
    residual_coef[lag] = coef0 - fitted_coef[lag]

Sector Splitting:
----------------
- Fits separate regressions for Energy sector vs all others
- Energy sector often has different dynamics
- Two regression outputs: "in" (Energy) and "ex" (ex-Energy)

Use Case:
--------
- Comprehensive beta-adjusted strategy
- Combines short-term (intraday) + medium-term (daily lags)
- Tests pure beta-normalized return patterns
- Simpler data requirements than order flow strategies

Data Requirements:
-----------------
- Price data: log_ret (daily), overnight_log_ret, dopen, iclose
- Barra factors: pbeta (predicted beta), ind1 (industry)
- Universe: Expandable stocks (liquid, tradeable)
- Sector classification: sector_name for split fitting

CLI Usage:
---------
Run backtest with in-sample/out-of-sample split:
    python badj_both.py --start=20130101 --end=20130630 --mid=20130315

Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --mid: Mid-date for in-sample/out-of-sample split

Output:
------
- Regression plots: badj_daily_in_{dates}.png, badj_daily_ex_{dates}.png
- HDF5 cache: badj_b{start}.{end}_daily.h5, badj_b{start}.{end}_intra.h5
- Alpha forecast: 'badj_b' column written via dump_alpha()

Related Modules:
---------------
- badj_multi.py: Daily-only version (intraday coef set to 0)
- badj_intra.py: Intraday-only version with time-of-day coefficients
- bd.py: Order flow based strategy (different alpha source)
- badj2_multi.py: Alternative with market-weighted beta calculation

Notes:
-----
- Horizon fixed at 3 in __main__
- Uses 15-minute bars (freq="15Min")
- Uses daybars not high-frequency intraday bars
- Energy sector treated separately
- Simple beta division (not order flow)
- Combines best of daily momentum + intraday patterns
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def calc_badj_daily(daily_df, horizon):
    """
    Calculate daily beta-adjusted returns (badj) with multiple lags.

    Computes the core daily signal by dividing log returns by predicted beta,
    creating a beta-normalized return series with historical lags.

    Process:
    1. Filter to expandable (tradeable, liquid) universe
    2. Divide log returns by beta (simple beta adjustment)
    3. Winsorize by date to control outliers
    4. Industry demean for sector neutrality
    5. Create lagged versions (badj1, badj2, ..., badj{horizon})

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, sid) containing:
            - log_ret: Daily log returns
            - pbeta: Predicted beta from Barra
            - ind1: Industry classification (for demeaning)
            - gdate: Date grouping column
            - expandable: Boolean filter for tradeable stocks

        horizon (int): Number of lagged signals to create (typically 3)

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - badj0: Raw beta-adjusted return (log_ret / pbeta)
            - badj0_B: Winsorized beta-adjusted return
            - badj0_B_ma: Industry-demeaned signal (main daily signal)
            - badj{1..horizon}_B_ma: Lagged versions of badj0_B_ma

    Signal Formula:
        badj0 = log_ret / pbeta
        badj0_B = winsorize_by_date(badj0)
        badj0_B_ma = industry_demean(badj0_B)

    Notes:
        - Simple division by beta (not subtraction like bd.py)
        - No order flow data required
        - Industry demeaning ensures sector neutrality
        - Lagged signals enable multi-horizon regression fitting
        - Uses gdate (grouped date) for grouping operations
    """
    print("Caculating daily badj...")
    result_df = filter_expandable(daily_df)

    print("Calculating badj0...")
    result_df['badj0'] = result_df['log_ret'] / result_df['pbeta'] 
    result_df['badj0_B'] = winsorize_by_date(result_df[ 'badj0' ])

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['badj0_B', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    result_df['badj0_B_ma'] = indgroups['badj0_B']

    print("Calulating lags...")
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['badj'+str(lag)+'_B_ma'] = shift_df['badj0_B_ma']
    
    return result_df

def calc_badj_intra(intra_df):
    """
    Calculate intraday beta-adjusted returns (badjC).

    Computes the intraday signal by dividing cumulative returns
    (overnight + day's move) by predicted beta.

    Process:
    1. Filter to expandable universe
    2. Calculate total return: overnight + intraday move
    3. Divide by beta for normalization
    4. Winsorize by timestamp
    5. Industry demean within (timestamp, industry) groups

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid):
            - overnight_log_ret: Overnight return (prior close to open)
            - iclose: Intraday bar close price
            - dopen: Day open price
            - pbeta: Predicted beta
            - ind1: Industry classification
            - giclose_ts: Bar close timestamp
            - expandable: Boolean filter

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - badjC: Beta-adjusted intraday return
            - badjC_B: Winsorized beta-adjusted return
            - badjC_B_ma: Industry-demeaned signal (main intraday signal)

    Signal Formula:
        badjC = (overnight_log_ret + log(iclose/dopen)) / pbeta
        badjC_B = winsorize_by_ts(badjC)
        badjC_B_ma = industry_demean(badjC_B)

    Notes:
        - Combines overnight + intraday for cumulative return
        - Simple beta division (not order flow)
        - Industry demeaning ensures sector neutrality
        - This signal will be combined with daily lags in forecast
        - Prints count of valid values for monitoring
    """
    print("Calculating badj intra...")
    result_df = filter_expandable(intra_df)

    print("Calulating badjC...")
    result_df['badjC'] = (result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))) / result_df['pbeta']
    result_df['badjC_B'] = winsorize_by_ts(result_df[ 'badjC' ])

    print("Calulating badjC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['badjC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=True).transform(demean)
    result_df['badjC_B_ma'] = indgroups['badjC_B']
    
    print("Calculated {} values".format(len(result_df['badjC_B_ma'].dropna())))
    return result_df

def badj_fits(daily_df, intra_df, horizon, name, middate=None):
    """
    Fit combined daily+intraday regression and generate beta-adjusted forecast.

    Fits regressions for both daily lags and intraday signals, then combines
    them into a unified forecast. Daily lags provide momentum/reversal while
    intraday captures current patterns.

    Args:
        daily_df (pd.DataFrame): Daily data with badj lag signals
        intra_df (pd.DataFrame): Intraday data with badjC signals
        horizon (int): Forecast horizon in days (typically 3)
        name (str): Name suffix for output plots (e.g., "in", "ex")
        middate (datetime): Split date for in-sample vs out-of-sample
                           If None, uses entire dataset

    Returns:
        pd.DataFrame: outsample_intra_df augmented with:
            - badjC_B_ma_coef: Coefficient for intraday signal
            - badj{1..horizon-1}_B_ma_coef: Coefficients for lagged daily signals
            - badj_b: Combined forecast (main output)

    Regression Strategy:
    -------------------
    Daily Regression:
        - Fits badj0_B_ma at multiple lags (1 to horizon)
        - Uses 'daily' regression mode
        - Extracts coefficient at full horizon (coef0)
        - Computes residual coefficients: coef[lag] = coef0 - fitted_coef[lag]

    Intraday Component:
        - Uses same coef0 from daily regression (full horizon coefficient)
        - No separate intraday regression in this implementation
        - Uniform coefficient across all timestamps

    Final Forecast:
        badj_b = badjC_B_ma * coef0 +
                 sum_{lag=1}^{horizon-1} (badj{lag}_B_ma * residual_coef[lag])

    Output Files:
        - badj_daily_{name}_{dates}.png: Daily regression plot

    Notes:
        - Only out-of-sample data gets forecasts if middate specified
        - In-sample used for coefficient fitting only
        - All coefficients printed to console for monitoring
        - Sector-specific fitting (Energy vs others)
        - Simpler than bd.py which uses time-varying intraday coefficients
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] <  middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['badj_b'] = np.nan
    outsample_intra_df[ 'badjC_B_ma_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_intra_df[ 'badj' + str(lag) + '_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'badj0_B_ma', lag, True, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "badj_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    
    coef0 = fits_df.loc['badj0_B_ma'].loc[horizon].loc['coef']
    outsample_intra_df[ 'badjC_B_ma_coef' ] = coef0
    print("Coef0: {}".format(coef0))
    for lag in range(1,horizon):
        coef = coef0 - fits_df.loc['badj0_B_ma'].loc[lag].loc['coef'] 
        print("Coef{}: {}".format(lag, coef))
        outsample_intra_df[ 'badj'+str(lag)+'_B_ma_coef' ] = coef

    outsample_intra_df['badj_b'] = outsample_intra_df['badjC_B_ma'] * outsample_intra_df['badjC_B_ma_coef']
    for lag in range(1,horizon):
        outsample_intra_df[ 'badj_b'] += outsample_intra_df['badj'+str(lag)+'_B_ma'] * outsample_intra_df['badj'+str(lag)+'_B_ma_coef']

    return outsample_intra_df

def calc_badj_forecast(daily_df, intra_df, horizon, middate):
    """
    Main entry point: compute combined daily+intraday beta-adjusted forecast with sector split.

    Orchestrates the full pipeline from raw data to final forecasts, combining
    both daily momentum and intraday patterns. Fits separate regressions for
    Energy sector vs all other sectors.

    Args:
        daily_df (pd.DataFrame): Daily price/factor data with MultiIndex (date, sid)
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid)
        horizon (int): Forecast horizon in days (typically 3)
        middate (datetime): Split date for in-sample/out-of-sample split

    Returns:
        pd.DataFrame: Combined results with 'badj_b' forecasts for both sectors

    Pipeline Flow:
        daily_df → calc_badj_daily() → daily_results with lags
        daily_df → calc_forward_returns() → forwards_df
        [merge daily_results + forwards]
        intra_df → calc_badj_intra() → intra_results
        [merge daily + intra]
        → badj_fits(Energy sector) → result1_df
        → badj_fits(ex-Energy) → result2_df
        → concat results → final result_df

    Sector Processing:
        1. Energy sector: Fit separate regression for Energy stocks
        2. Ex-Energy: Fit regression for all non-Energy stocks
        3. Concatenate results with verify_integrity=True

    Notes:
        - middate passed to badj_fits for in/out sample split
        - Energy sector handled separately for robustness
        - Forward returns required for regression fitting
        - Final forecast in 'badj_b' column
        - Both sectors combined with integrity verification
        - Uses sector_name column from data
    """
    daily_results_df = calc_badj_daily(daily_df, horizon) 
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    intra_results_df = calc_badj_intra(intra_df)
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)

    sector_name = 'Energy'
    print("Running badj for sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    result1_df = badj_fits(sector_df, sector_intra_results_df, horizon, "in", middate)

    print("Running badj for not sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] != sector_name ]    
    result2_df = badj_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)    

    result_df = pd.concat([result1_df, result2_df], verify_integrity=True)
    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = 3
    pname = "./badj_b" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)
    freq="15Min"
    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1', 'pbeta']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close', 'overnight_log_ret']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        DBAR_COLS = ['close', 'dopen', 'dvolume']
        intra_df = load_daybars(price_df[['ticker']], start, end, DBAR_COLS, freq)

        daily_df = merge_barra_data(price_df, barra_df)
        daily_df = merge_intra_eod(daily_df, intra_df)
        intra_df = merge_intra_data(daily_df, intra_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    result_df = calc_badj_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(result_df, 'badj_b')



