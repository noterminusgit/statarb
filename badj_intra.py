#!/usr/bin/env python
"""Intraday-Only Beta-Adjusted Returns Strategy (badj_intra)

IMPORTANT: Despite the filename "badj", this strategy does NOT use order flow.
Instead, it implements a simple beta-division approach on returns, creating
a "beta-adjusted" signal by dividing returns by predicted beta.

This is the intraday-only variant of badj_multi.py, focusing exclusively on
intraday signals with time-of-day varying coefficients.

Key Differences from badj_multi.py:
----------------------------------
1. Scope:
   - badj_multi.py: Combines daily lags + intraday signal
   - badj_intra.py: ONLY intraday signal (o2cC) - no daily lags
   - Simpler single-signal approach

2. Regression:
   - badj_multi.py: Daily regression with lag coefficients
   - badj_intra.py: Intraday regression with 6 time-of-day buckets
   - Time-varying coefficients capture intraday pattern evolution

3. Forecast:
   - badj_multi.py: Weighted sum of daily lags
   - badj_intra.py: Single intraday signal * time-of-day coefficient
   - No multi-period combination

Differences from bd_intra.py Order Flow:
---------------------------------------
1. Signal Source:
   - bd_intra.py: Order flow imbalance (askHit - bidHit) / total
   - badj_intra.py: Simple return / beta ratio
   - Completely different alpha sources

2. Data Requirements:
   - bd_intra.py: Requires order book data (askHitDollars, etc.)
   - badj_intra.py: Only needs returns and beta
   - Much simpler data requirements

Methodology:
-----------
1. Beta Adjustment (Simple Division):
   o2cC = (overnight_log_ret + log(iclose/dopen)) / pbeta

   Creates a "beta-normalized" intraday return.

2. Winsorization and Industry Demeaning:
   - Winsorize by timestamp to control outliers
   - Industry demean for sector neutrality
   - Creates o2cC_B_ma signal

3. Time-Varying Coefficients:
   - Fits regression with 6 hourly time buckets
   - Captures intraday pattern evolution
   - Overlapping buckets smooth transitions

Signal Formula:
--------------
o2cC = (overnight_log_ret + log(iclose/dopen)) / pbeta
o2cC_B = winsorize_by_group(o2cC, groupby='iclose_ts')
o2cC_B_ma = industry_demean(o2cC_B)

Forecast:
    badj_i = o2cC_B_ma * o2cC_B_ma_coef(time_of_day)

Time-of-Day Buckets:
-------------------
- Bucket 1: 09:30-10:31
- Bucket 2: 10:30-11:31
- Bucket 3: 11:30-12:31
- Bucket 4: 12:30-13:31
- Bucket 5: 13:30-14:31
- Bucket 6: 14:30-15:31

Overlapping buckets provide smooth coefficient transitions.

Sector Splitting:
----------------
- Fits separate regressions for Energy sector vs all others
- Allows sector-specific coefficient optimization
- Two regression outputs: "in" (Energy) and "ex" (ex-Energy)

Use Case:
--------
- Pure intraday trading without overnight positions
- Tests beta-adjusted return patterns intraday
- Lower data requirements (no order book needed)
- Simpler alternative to order flow strategies

Data Requirements:
-----------------
- Price data: overnight_log_ret, dopen, iclose
- Barra factors: pbeta (predicted beta), ind1 (industry)
- Universe: Expandable stocks (liquid, tradeable)

CLI Usage:
---------
Run backtest with optional in-sample/out-of-sample split:
    python badj_intra.py --start=20130101 --end=20130630 --os=True

Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --os: Enable out-of-sample split (default: False)

Output:
------
- Regression plots: badj_intra_in_{dates}.png, badj_intra_ex_{dates}.png
- HDF5 cache: badj_i.{start}.{end}_daily.h5, badj_i.{start}.{end}_intra.h5
- Alpha forecast: 'badj_i' column written via dump_alpha()

Related Modules:
---------------
- badj_multi.py: Multi-period version with daily lags
- bd_intra.py: Order flow based intraday strategy
- badj2_intra.py: Alternative with market-weighted beta calculation

Notes:
-----
- Horizon fixed at 3 in __main__
- Uses daybars (full-day aggregated) not high-frequency intraday bars
- Energy sector treated separately
- Simple division by beta (not order flow based)
"""

from __future__ import division, print_function

from alphacalc import *

from dateutil import parser as dateparser
import argparse

def calc_o2c(daily_df, horizon):
    """
    Calculate daily beta-adjusted returns for reference.

    Creates daily o2c signals but they are NOT used in the intraday-only
    forecast. Included for data consistency and potential future use.

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, sid)
        horizon (int): Number of lagged signals to create

    Returns:
        pd.DataFrame: Daily data with o2c signals (not used in forecast)

    Notes:
        - Daily lags computed but not used in badj_intra forecast
        - Included for consistency with badj_multi.py structure
        - See badj_multi.py for detailed signal documentation
    """
    print("Caculating daily o2c...")

    result_df = daily_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['log_ret', 'pbeta', 'date', 'ind1', 'sid', 'mkt_cap' ]]

    print("Calculating o2c0...")
    result_df['o2c0'] = result_df['log_ret'] / result_df['pbeta'] 
    result_df['o2c0_B'] = winsorize_by_group(result_df[ ['date', 'o2c0'] ], 'date')

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['o2c0_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['o2c0_B_ma'] = indgroups['o2c0_B']
    result_df.set_index(keys=['date', 'sid'], inplace=True)
    
    print("Calulating lags...")
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['o2c' + str(lag) + '_B_ma'] = shift_df['o2c0_B_ma']

    result_df = pd.merge(daily_df, result_df, how='left', left_index=True, right_index=True, sort=True, suffixes=['', '_dead'])
    result_df = remove_dup_cols(result_df)
    return result_df

def calc_o2c_intra(intra_df, daily_df):
    """
    Calculate intraday beta-adjusted returns (o2cC).

    Computes the core intraday signal by dividing cumulative returns
    (overnight + day's move) by predicted beta. This is the main signal
    for the intraday-only strategy.

    Process:
    1. Filter to expandable universe
    2. Calculate total return: overnight + day's move
    3. Divide by beta for normalization
    4. Winsorize by timestamp
    5. Industry demean within (timestamp, industry) groups

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid):
            - iclose: Intraday bar close price
            - dopen: Day open price
            - overnight_log_ret: Overnight return (prior close to open)
            - pbeta: Predicted beta
            - ind1: Industry classification
            - expandable: Boolean filter

        daily_df (pd.DataFrame): Daily reference data for expandable filter

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - o2cC: Beta-adjusted intraday return
            - o2cC_B: Winsorized beta-adjusted return
            - o2cC_B_ma: Industry-demeaned signal (main signal)

    Signal Formula:
        o2cC = (overnight_log_ret + log(iclose/dopen)) / pbeta
        o2cC_B = winsorize_by_group(o2cC, groupby='iclose_ts')
        o2cC_B_ma = industry_demean(o2cC_B)

    Notes:
        - Simple beta division (not order flow)
        - Combines overnight + intraday for total return
        - Industry demeaning ensures sector neutrality
        - This is the primary signal for badj_i forecast
    """
    print("Calculating o2c intra...")

    result_df = filter_expandable_intra(intra_df, daily_df)
    result_df = result_df.reset_index()    
    result_df = result_df[ ['iclose_ts', 'iclose', 'dopen', 'overnight_log_ret', 'pbeta', 'date', 'ind1', 'sid', 'mkt_cap' ] ]
    result_df = result_df.dropna(how='any')

    print("Calulating o2cC...")
    result_df['o2cC'] = (result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))) / result_df['pbeta']
    result_df['o2cC_B'] = winsorize_by_group(result_df[ ['iclose_ts', 'o2cC'] ], 'iclose_ts')

    print("Calulating o2cC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['o2cC_B', 'iclose_ts', 'ind1']].groupby(['iclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['o2cC_B_ma'] = indgroups['o2cC_B']

    #important for keeping NaTs out of the following merge
    del result_df['date']

    print("Merging...")
    result_df.set_index(keys=['iclose_ts', 'sid'], inplace=True)
    result_df = pd.merge(intra_df, result_df, how='left', left_index=True, right_index=True, sort=True, suffixes=['_dead', ''])
    result_df = remove_dup_cols(result_df)

    return result_df

def o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None):
    """
    Fit intraday regression and generate beta-adjusted forecast with time-of-day coefficients.

    Fits a single regression for o2cC_B_ma with time-varying coefficients
    across 6 hourly buckets. Each bucket gets its own coefficient capturing
    how signal efficacy varies throughout the trading day.

    Args:
        daily_df (pd.DataFrame): Daily data (not used, for compatibility)
        intra_df (pd.DataFrame): Intraday data with o2cC signals
        full_df (pd.DataFrame): Full merged dataset for forecast storage
        horizon (int): Forecast horizon (passed to regression)
        name (str): Name suffix for output plots (e.g., "in", "ex")
        middate (datetime): Split date for in-sample vs out-of-sample
                           If None, no split

    Returns:
        pd.DataFrame: full_df augmented with:
            - o2cC_B_ma_coef: Time-of-day specific coefficient
            - badj_i: Intraday forecast (main output)

    Regression Strategy:
    -------------------
    - Fits o2cC_B_ma against forward returns using regress_alpha()
    - Uses outsample=True mode for proper in/out split
    - Generates 6 separate coefficients for time buckets:
      Bucket 1: 09:30-10:31
      Bucket 2: 10:30-11:31
      Bucket 3: 11:30-12:31
      Bucket 4: 12:30-13:31
      Bucket 5: 13:30-14:31
      Bucket 6: 14:30-15:31
    - Overlapping buckets smooth coefficient transitions

    Final Forecast:
        badj_i = o2cC_B_ma * o2cC_B_ma_coef(time_of_day)

    Output Files:
        - badj_intra_{name}_{dates}.png: Intraday regression plot

    Notes:
        - Only out-of-sample data gets forecasts
        - Time-of-day coefficients capture intraday pattern evolution
        - Sector-specific fitting (Energy vs others)
    """
    if 'badj_i' not in full_df.columns:
        print("Creating forecast columns...")
        full_df['badj_i'] = np.nan
        full_df[ 'o2cC_B_ma_coef' ] = np.nan

    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    outsample = False
    if middate is not None:
        outsample = True
        insample_intra_df = intra_df[ intra_df['date'] <  middate ]
        insample_daily_df = daily_df[ daily_df['date'] < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    intra_horizon = horizon
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df, intraForwardRets_df = regress_alpha(insample_intra_df, 'o2cC_B_ma', intra_horizon, outsample, True)
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "badj_intra_"+name+"_" + df_dates(insample_intra_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    unstacked = outsample_intra_df[ ['ticker', 'name'] ].unstack()
    coefs = dict()
    coefs[1] = unstacked.between_time('09:30', '10:31').stack().index
    coefs[2] = unstacked.between_time('10:30', '11:31').stack().index
    coefs[3] = unstacked.between_time('11:30', '12:31').stack().index
    coefs[4] = unstacked.between_time('12:30', '13:31').stack().index
    coefs[5] = unstacked.between_time('13:30', '14:31').stack().index
    coefs[6] = unstacked.between_time('14:30', '15:31').stack().index
    unstacked = None

    for ii in range(1,7):
        full_df.ix[ coefs[ii], 'o2cC_B_ma_coef' ] = fits_df.ix['o2cC_B_ma'].ix[ii].ix['coef']

    full_df.ix[ outsample_intra_df.index, 'badj_i'] = full_df['o2cC_B_ma'] * full_df['o2cC_B_ma_coef']
    
    return full_df

def calc_o2c_forecast(daily_df, intra_df, horizon, outsample):
    """
    Main entry point: compute intraday-only beta-adjusted forecast with sector split.

    Orchestrates the pipeline from raw data to final intraday forecasts,
    with separate regression fitting for Energy sector vs all other sectors.

    Args:
        daily_df (pd.DataFrame): Daily price/factor data with MultiIndex (date, sid)
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid)
        horizon (int): Forecast horizon in days (typically 3)
        outsample (bool): If True, split data in-sample/out-of-sample at midpoint

    Returns:
        tuple: (full_df, outsample_df)
            - full_df: Complete dataset with forecasts
            - outsample_df: Only out-of-sample data (if outsample=True)
                           or full dataset (if outsample=False)

    Pipeline Flow:
        daily_df → calc_o2c() → daily with lags (not used)
        intra_df → calc_o2c_intra() → intra_results
        [merge daily + intra]
        → o2c_fits(Energy sector) → partial forecasts
        → o2c_fits(ex-Energy) → complete forecasts
        → filter to outsample if requested

    Sector Processing:
        1. Energy sector: Fit separate regression with Energy stocks only
        2. Ex-Energy: Fit regression for all non-Energy stocks
        3. Combine results in full_df

    Notes:
        - middate computed as midpoint if outsample=True
        - Energy sector handled separately for robustness
        - Daily lags computed but not used in forecast
        - Final forecast in 'badj_i' column
    """
    daily_df = calc_o2c(daily_df, horizon) 
    intra_df = calc_o2c_intra(intra_df, daily_df)
    full_df = merge_intra_data(daily_df, intra_df)

    middate = None
    if outsample:
        middate = intra_df.index[0][0] + (intra_df.index[len(intra_df)-1][0] - intra_df.index[0][0]) / 2
        print("Setting fit period before {}".format(middate))

    sector_name = 'Energy'
    print("Running o2c for sector {}".format(sector_name))
    sector_df = daily_df[ daily_df['sector_name'] == sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] == sector_name ]
    full_df = o2c_fits(sector_df, sector_intra_df, full_df, horizon, "in", middate)

    print("Running o2c for sector {}".format(sector_name))
    sector_df = daily_df[ daily_df['sector_name'] != sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] != sector_name ]
    full_df = o2c_fits(sector_df, sector_intra_df, full_df, horizon, "ex", middate)

    outsample_df = full_df
    if outsample:
        outsample_df = full_df[ full_df['date'] >= middate ]

    return full_df, outsample_df

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

    pname = "./badj_i." + start + "." + end

    start = dateparser.parse(start)
    end = dateparser.parse(end)

    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
        print("Successfully loaded cached data...")
    except:
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        barra_df = load_barra(uni_df, start, end)
        price_df = load_prices(uni_df, start, end)
        daily_df = merge_barra_data(price_df, barra_df)
        daybar_df = load_daybars(uni_df, start, end)
        intra_df = merge_intra_data(daily_df, daybar_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    full_df, outsample_df = calc_o2c_forecast(daily_df, intra_df, horizon, outsample)

    dump_alpha(outsample_df, 'badj_i')
    dump_all(outsample_df)
