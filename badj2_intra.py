#!/usr/bin/env python
"""Alternative Intraday-Only Beta-Adjusted Returns Strategy (badj2_intra)

Intraday-only variant of badj2_multi.py that uses market-weighted beta
calculation with time-of-day varying coefficients. This is the most
sophisticated intraday beta-adjusted approach without order flow.

IMPORTANT: This strategy does NOT use order flow. It uses market-weighted
beta adjustment on returns.

Key Differences from Related Strategies:
---------------------------------------
vs badj_intra.py:
- badj_intra.py: Simple beta division (return / pbeta)
- badj2_intra.py: Market-weighted beta calculation (wavg2)
- More sophisticated market neutralization

vs badj2_multi.py:
- badj2_multi.py: Combines daily lags + intraday
- badj2_intra.py: Intraday only with time-of-day coefficients
- Simpler single-signal approach

vs bd_intra.py:
- bd_intra.py: Order flow based (askHit - bidHit)
- badj2_intra.py: Return-based beta adjustment
- Completely different alpha sources

Methodology:
-----------
1. Market-Weighted Beta Adjustment:
   For each timestamp:
       market_return = sum(cur_log_ret * weight) / sum(weight)
       where weight = capitalization / 1e6
       o2cC = pbeta * market_return

   Each stock gets its beta-scaled exposure to the cap-weighted market return.

2. Winsorization and Industry Demeaning:
   - Winsorize by timestamp to control outliers
   - Industry demean for sector neutrality
   - Creates o2cC_B_ma signal

3. Time-Varying Coefficients:
   - Fits regression with 6 hourly time buckets
   - Captures how signal efficacy evolves intraday
   - Overlapping buckets smooth transitions

Signal Formula:
--------------
cur_log_ret = overnight_log_ret + log(iclose/dopen)
o2cC = pbeta * (cap_weighted_market_return)
o2cC_B = winsorize_by_group(o2cC, groupby='iclose_ts')
o2cC_B_ma = industry_demean(o2cC_B)

Forecast:
    badj2_i = o2cC_B_ma * o2cC_B_ma_coef(time_of_day)

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
- Most sophisticated beta-adjusted intraday signal
- Tests market-neutralized return patterns intraday
- No order flow data required

Data Requirements:
-----------------
- Price data: overnight_log_ret, dopen, iclose
- Barra factors: pbeta (predicted beta), ind1 (industry)
- Market cap: capitalization for weighting
- Universe: Expandable stocks (liquid, tradeable)

CLI Usage:
---------
Run backtest with optional in-sample/out-of-sample split:
    python badj2_intra.py --start=20130101 --end=20130630 --os=True

Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --os: Enable out-of-sample split (default: False)

Output:
------
- Regression plots: badj2_intra_in_{dates}.png, badj2_intra_ex_{dates}.png
- HDF5 cache: badj2_i.{start}.{end}_daily.h5, badj2_i.{start}.{end}_intra.h5
- Alpha forecast: 'badj2_i' column written via dump_alpha()

Related Modules:
---------------
- badj2_multi.py: Multi-period version with daily lags
- badj_intra.py: Simpler division-based intraday
- bd_intra.py: Order flow based intraday strategy

Test IDs:
--------
- testid = 10020731: Stock ID for testing/debugging
- testid2 = 10000678: Secondary test stock ID

Code Issues:
-----------
- Line 139: Variable name inconsistency (outsample_df condition check)
- Logic issue in return statement

Notes:
-----
- Horizon fixed at 3 in __main__
- Uses daybars (full-day aggregated) not high-frequency bars
- Energy sector treated separately
- Market-weighted approach most sophisticated beta adjustment
- Time-of-day coefficients capture intraday pattern evolution
"""

from __future__ import division, print_function

from alphacalc import *

from dateutil import parser as dateparser
import argparse

testid = 10020731
testid2 = 10000678

def wavg(group):
    """
    Calculate daily market component using cap-weighted market return.

    Used for daily data though badj2_intra focuses on intraday. Included
    for consistency and potential future use.

    Args:
        group (pd.DataFrame): DataFrame group with pbeta, log_ret, capitalization

    Returns:
        pd.Series: Beta-scaled market return for each stock

    Formula:
        market_return = sum(log_ret * weight) / sum(weight)
        where weight = capitalization / 1e6
        result = pbeta * market_return

    Notes:
        - Not actively used in badj2_intra main pipeline
        - Included for compatibility with badj2_multi structure
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['capitalization'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg2(group):
    """
    Calculate intraday market component using cap-weighted market return.

    Core function for market-weighted beta adjustment of intraday returns.
    Computes beta-scaled exposure to the capitalization-weighted market return.

    Args:
        group (pd.DataFrame): DataFrame group containing:
            - pbeta: Predicted beta from Barra risk model
            - cur_log_ret: Current intraday log return (overnight + day's move)
            - capitalization: Market capitalization (in dollars)

    Returns:
        pd.Series: Beta-scaled intraday market return for each stock in group

    Formula:
        market_return = sum(cur_log_ret * weight) / sum(weight)
        where weight = capitalization / 1e6
        result = pbeta * market_return

    Notes:
        - Used in groupby().apply() with grouping by timestamp (iclose_ts)
        - Each timestamp gets its own market return calculation
        - Capitalization scaled by 1e6 (millions) to avoid numerical issues
        - This is the core beta adjustment for badj2_intra
    """
    b = group['pbeta']
    d = group['cur_log_ret']
    w = group['capitalization'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def calc_o2c(daily_df, horizon):
    """
    Calculate daily market-weighted beta-adjusted returns.

    Computes daily signals but they are NOT used in the intraday-only
    forecast. Included for data consistency with badj2_multi structure.

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, sid)
        horizon (int): Number of lagged signals to create

    Returns:
        pd.DataFrame: Daily data with o2c signals (not used in forecast)

    Notes:
        - Daily lags computed but not used in badj2_intra forecast
        - Included for consistency with badj2_multi.py structure
        - See badj2_multi.py for detailed signal documentation
    """
    print("Caculating daily o2c...")

    result_df = daily_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['log_ret', 'pbeta', 'date', 'ind1', 'sid', 'capitalization' ]]

    print("Calculating o2c0...")
    result_df['o2c0'] = result_df[['log_ret', 'pbeta', 'capitalization', 'date']].groupby(['date'], sort=False).apply(wavg)
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
    Calculate intraday market-weighted beta-adjusted returns (o2cC).

    Core signal calculation using market-weighted beta adjustment (wavg2)
    applied to cumulative intraday returns. This is the main signal for
    the badj2_i forecast.

    Process:
    1. Filter to expandable universe
    2. Select required columns and drop NaNs
    3. Calculate cumulative return: overnight + log(iclose/dopen)
    4. Compute market component: o2cC = groupby(iclose_ts).apply(wavg2)
    5. Winsorize by timestamp
    6. Industry demean within (timestamp, industry) groups

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid):
            - overnight_log_ret: Overnight return
            - iclose: Intraday bar close price
            - dopen: Day open price
            - pbeta: Predicted beta
            - capitalization: Market capitalization
            - ind1: Industry classification
            - date: Trading date
            - expandable: Boolean filter

        daily_df (pd.DataFrame): Daily reference data for expandable filter

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - cur_log_ret: Cumulative intraday return
            - o2cC: Market-weighted beta component
            - o2cC_B: Winsorized signal
            - o2cC_B_ma: Industry-demeaned signal (main signal)

    Signal Formula:
        cur_log_ret = overnight_log_ret + log(iclose/dopen)
        o2cC = pbeta * (cap_weighted_intraday_market_return)
        o2cC_B = winsorize_by_group(o2cC, groupby='iclose_ts')
        o2cC_B_ma = industry_demean(o2cC_B)

    Notes:
        - Uses wavg2 for market-weighted calculation
        - Combines overnight + intraday for total return
        - Date column deleted before merge to avoid conflicts
        - More sophisticated than simple beta division
        - This is the primary signal for badj2_i forecast
    """
    print("Calculating o2c intra...")

    result_df = filter_expandable_intra(intra_df, daily_df)
    result_df = result_df.reset_index()    
    result_df = result_df[ ['iclose_ts', 'iclose', 'dopen', 'overnight_log_ret', 'pbeta', 'date', 'ind1', 'sid', 'capitalization' ] ]
    result_df = result_df.dropna(how='any')

    print("Calulating o2cC...")
    result_df['cur_log_ret'] = result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))
    result_df['o2cC'] = result_df[['cur_log_ret', 'pbeta', 'capitalization', 'iclose_ts']].groupby(['iclose_ts'], sort=False).apply(wavg2)
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
    Fit intraday regression and generate market-weighted beta-adjusted forecast.

    Fits a single regression for o2cC_B_ma with time-varying coefficients
    across 6 hourly buckets. Uses market-weighted beta adjustment (most
    sophisticated intraday beta approach).

    Args:
        daily_df (pd.DataFrame): Daily data (not used, for compatibility)
        intra_df (pd.DataFrame): Intraday data with o2cC signals
        full_df (pd.DataFrame): Full merged dataset for forecast storage
        horizon (int): Forecast horizon (passed to regression)
        name (str): Name suffix for output plots (e.g., "in", "ex")
        middate (datetime): Split date for in-sample vs out-of-sample

    Returns:
        pd.DataFrame: full_df augmented with:
            - o2cC_B_ma_coef: Time-of-day specific coefficient
            - badj2_i: Intraday forecast (main output)

    Regression Strategy:
    -------------------
    - Fits o2cC_B_ma against forward returns
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
        badj2_i = o2cC_B_ma * o2cC_B_ma_coef(time_of_day)

    Output Files:
        - badj2_intra_{name}_{dates}.png: Intraday regression plot

    Notes:
        - Only out-of-sample data gets forecasts
        - Time-of-day coefficients capture intraday pattern evolution
        - Sector-specific fitting (Energy vs others)
        - Most sophisticated intraday beta adjustment (market-weighted)
        - Output column named 'badj2_i' (note the '2')
    """
    if 'badj2_i' not in full_df.columns:
        print("Creating forecast columns...")
        full_df['badj2_i'] = np.nan
        full_df[ 'o2cC_B_ma_coef' ] = np.nan

    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    outsample = False
    if middate is not None:
        outsample = True
        insample_intra_df = intra_df[ intra_df['date'] <  middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    intra_horizon = horizon
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df, intraForwardRets_df = regress_alpha(insample_intra_df, 'o2cC_B_ma', intra_horizon, outsample, True)
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "badj2_intra_"+name+"_" + df_dates(insample_intra_df))
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

    full_df.ix[ outsample_intra_df.index, 'badj2_i'] = full_df['o2cC_B_ma'] * full_df['o2cC_B_ma_coef']
    
    return full_df

def calc_o2c_forecast(daily_df, intra_df, horizon, outsample):
    """
    Main entry point: compute intraday market-weighted beta-adjusted forecast.

    Orchestrates the pipeline using market-weighted beta adjustment with
    time-of-day varying coefficients and sector split.

    Args:
        daily_df (pd.DataFrame): Daily price/factor data with MultiIndex (date, sid)
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid)
        horizon (int): Forecast horizon in days (typically 3)
        outsample (bool): If True, split data in-sample/out-of-sample

    Returns:
        tuple: (full_df, outsample_df)
            - full_df: Complete dataset with forecasts
            - outsample_df: Only out-of-sample data (if outsample=True)

    Pipeline Flow:
        daily_df → calc_o2c() → daily with lags (not used)
        intra_df → calc_o2c_intra() → intra with market-weighted signal
        [merge daily + intra]
        → o2c_fits(Energy sector) → partial forecasts
        → o2c_fits(ex-Energy) → complete forecasts

    Sector Processing:
        1. Energy sector: Time-of-day fitting for Energy stocks
        2. Ex-Energy: Time-of-day fitting for non-Energy stocks

    Notes:
        - middate computed as midpoint if outsample=True
        - Energy sector handled separately
        - Daily lags computed but not used in forecast
        - Final forecast in 'badj2_i' column
        - Most sophisticated intraday beta adjustment approach
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

    pname = "./badj2_i." + start + "." + end

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

    dump_alpha(outsample_df, 'badj2_i')
    dump_all(outsample_df)
