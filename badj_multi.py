#!/usr/bin/env python
"""Multi-Period Beta-Adjusted Returns Strategy (badj_multi)

IMPORTANT: Despite the filename "badj", this strategy does NOT use order flow.
Instead, it implements a simple beta-division approach on returns, creating
a "beta-adjusted" signal by dividing log returns by predicted beta.

This is fundamentally different from bd.py which uses order flow imbalance.

Naming Convention:
-----------------
- Internal signal names: "o2c" (overnight-to-close)
- Output forecast name: "badj_m" (beta-adjusted multi-period)
- The "o2c" naming suggests focus on overnight-to-close returns

Methodology:
-----------
1. Beta Adjustment (Simple Division):
   o2c0 = log_ret / pbeta

   This creates a "beta-normalized" return by dividing out systematic risk.
   Stocks with high beta have their returns scaled down, low beta scaled up.

2. Winsorization and Industry Demeaning:
   - Winsorize by date to control outliers
   - Industry demean for sector neutrality
   - Creates o2c0_B_ma signal

3. Multi-Period Lags:
   - Computes lagged versions (o2c1, o2c2, ..., o2c{horizon})
   - Enables multi-period momentum/mean reversion capture

4. Regression:
   - Fits o2c0_B_ma at multiple lags against forward returns
   - Computes residual coefficients for each lag
   - Combines intraday (o2cC) + daily lags (o2c1, o2c2, ...)

Differences from bd.py Order Flow Strategy:
------------------------------------------
1. Signal Source:
   - bd.py: Order flow imbalance (askHit - bidHit) / total
   - badj_multi.py: Simple return / beta ratio
   - Completely different alpha source

2. Beta Adjustment:
   - bd.py: Subtracts beta * market_return from returns
   - badj_multi.py: Divides returns by beta
   - Different normalization approaches

3. Data Requirements:
   - bd.py: Requires order book data (askHitDollars, bidHitDollars, etc.)
   - badj_multi.py: Only needs returns and beta
   - Much simpler data requirements

Signal Formula:
--------------
Daily:
    o2c0 = log_ret / pbeta
    o2c0_B = winsorize(o2c0)
    o2c0_B_ma = industry_demean(o2c0_B)

Intraday:
    o2cC = (overnight_log_ret + log(iclose/dopen)) / pbeta
    o2cC_B = winsorize(o2cC)
    o2cC_B_ma = industry_demean(o2cC_B)

Forecast:
    badj_m = o2cC_B_ma * coef0 +
             sum_{lag=1}^{horizon-1} (o2c{lag}_B_ma * residual_coef[lag])

Sector Splitting:
----------------
- Fits separate regressions for Energy sector vs all others
- Allows sector-specific coefficient optimization
- Two separate regression outputs: "in" (Energy) and "ex" (ex-Energy)

Use Case:
--------
- Simpler alternative to order flow strategies
- Tests pure beta-adjusted return momentum/reversal
- Lower data requirements (no order book needed)
- Captures systematic risk-adjusted return patterns

Data Requirements:
-----------------
- Price data: log_ret (daily returns)
- Barra factors: pbeta (predicted beta), ind1 (industry)
- Intraday: overnight_log_ret, dopen, iclose
- Universe: Expandable stocks (liquid, tradeable)

CLI Usage:
---------
Run backtest with optional in-sample/out-of-sample split:
    python badj_multi.py --start=20130101 --end=20130630 --os=True

Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --os: Enable out-of-sample split (default: False)

Output:
------
- Regression plots: badj_daily_in_{dates}.png, badj_daily_ex_{dates}.png
- HDF5 cache: badj_m{start}.{end}_daily.h5, badj_m{start}.{end}_intra.h5
- Alpha forecast: 'badj_m' column written via dump_alpha()

Related Modules:
---------------
- bd.py: Order flow based strategy (different alpha source)
- badj_intra.py: Intraday-only version of this strategy
- badj_both.py: Combined daily+intraday beta-adjusted
- badj2_multi.py: Alternative implementation with market-weighted beta

Notes:
-----
- Horizon fixed at 3 in __main__
- Uses daybars (full-day aggregated) not intraday bars
- Energy sector treated separately for robustness
- o2cC_B_ma_coef always set to 0 (line 87) - effectively drops intraday component
"""

from alphacalc import *

from dateutil import parser as dateparser
import argparse

def calc_o2c(daily_df, horizon):
    """
    Calculate daily beta-adjusted returns (o2c) with multiple lags.

    Computes the core "o2c" (overnight-to-close / beta-adjusted) signal
    by simply dividing log returns by predicted beta. This creates a
    beta-normalized return series.

    Process:
    1. Filter to expandable (tradeable, liquid) universe
    2. Divide log returns by beta (simple beta adjustment)
    3. Winsorize by date to control outliers
    4. Industry demean for sector neutrality
    5. Create lagged versions (o2c1, o2c2, ..., o2c{horizon})

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, sid) containing:
            - log_ret: Daily log returns
            - pbeta: Predicted beta from Barra
            - ind1: Industry classification (for demeaning)
            - sid: Security identifier
            - expandable: Boolean filter for tradeable stocks

        horizon (int): Number of lagged signals to create (typically 3)

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - o2c0: Raw beta-adjusted return (log_ret / pbeta)
            - o2c0_B: Winsorized beta-adjusted return
            - o2c0_B_ma: Industry-demeaned signal (main daily signal)
            - o2c{1..horizon}_B_ma: Lagged versions of o2c0_B_ma

    Signal Formula:
        o2c0 = log_ret / pbeta
        o2c0_B = winsorize_by_group(o2c0, groupby='date')
        o2c0_B_ma = industry_demean(o2c0_B)

    Notes:
        - Simple division by beta (not subtraction like bd.py)
        - No order flow data required
        - Industry demeaning ensures sector neutrality
        - Lagged signals enable multi-horizon regression fitting
        - "o2c" naming historically meant "overnight-to-close"
    """
    print "Caculating daily o2c..."

    result_df = daily_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['log_ret', 'pbeta', 'date', 'ind1', 'sid', 'mkt_cap' ]]

    print "Calculating o2c0..."
    result_df['o2c0'] = result_df['log_ret'] / result_df['pbeta'] 
    result_df['o2c0_B'] = winsorize_by_group(result_df[ ['date', 'o2c0'] ], 'date')

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['o2c0_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['o2c0_B_ma'] = indgroups['o2c0_B']
    result_df.set_index(keys=['date', 'sid'], inplace=True)
    
    print "Calulating lags..."
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['o2c' + str(lag) + '_B_ma'] = shift_df['o2c0_B_ma']

    result_df = pd.merge(daily_df, result_df, how='left', left_index=True, right_index=True, sort=True, suffixes=['', '_dead'])
    result_df = remove_dup_cols(result_df)
    return result_df

def calc_o2c_intra(intra_df, daily_df):
    """
    Calculate intraday beta-adjusted returns (o2cC) from daybar data.

    Computes the intraday version of the beta-adjusted signal using
    overnight return plus day's price change, normalized by beta.

    This uses full-day aggregate data (daybars) not intraday bars,
    calculating the cumulative return from prior close to current bar.

    Process:
    1. Filter to expandable universe (using daily_df reference)
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
            - o2cC_B_ma: Industry-demeaned signal (main intraday signal)

    Signal Formula:
        o2cC = (overnight_log_ret + log(iclose/dopen)) / pbeta
        o2cC_B = winsorize_by_group(o2cC, groupby='iclose_ts')
        o2cC_B_ma = industry_demean(o2cC_B)

    Notes:
        - Combines overnight + intraday returns for total return
        - Simple division by beta (same as daily)
        - filter_expandable_intra() uses daily_df to determine universe
        - Date column deleted before merge to avoid timestamp conflicts
        - Winsorization by timestamp ensures cross-sectional outlier control
    """
    print "Calculating o2c intra..."

    result_df = filter_expandable_intra(intra_df, daily_df)
    result_df = result_df.reset_index()    
    result_df = result_df[ ['iclose_ts', 'iclose', 'dopen', 'overnight_log_ret', 'pbeta', 'date', 'ind1', 'sid', 'mkt_cap' ] ]
    result_df = result_df.dropna(how='any')

    print "Calulating o2cC..."
    result_df['o2cC'] = (result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))) / result_df['pbeta']
    result_df['o2cC_B'] = winsorize_by_group(result_df[ ['iclose_ts', 'o2cC'] ], 'iclose_ts')

    print "Calulating o2cC_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['o2cC_B', 'iclose_ts', 'ind1']].groupby(['iclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['o2cC_B_ma'] = indgroups['o2cC_B']

    #important for keeping NaTs out of the following merge
    del result_df['date']

    print "Merging..."
    result_df.set_index(keys=['iclose_ts', 'sid'], inplace=True)
    result_df = pd.merge(intra_df, result_df, how='left', left_index=True, right_index=True, sort=True, suffixes=['_dead', ''])
    result_df = remove_dup_cols(result_df)

    return result_df

def o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None):
    """
    Fit regression coefficients and generate beta-adjusted forecast.

    Fits regressions for the beta-adjusted return strategy using daily lags
    of o2c0_B_ma. Computes residual coefficients to capture incremental
    information at each lag.

    The intraday component (o2cC_B_ma) coefficient is set to 0 (line 87),
    effectively making this a daily-only strategy despite having intraday data.

    Args:
        daily_df (pd.DataFrame): Daily data with o2c signals
        intra_df (pd.DataFrame): Intraday data with o2cC signals (not used for coef)
        full_df (pd.DataFrame): Full merged dataset for forecast storage
        horizon (int): Forecast horizon in days (typically 3)
        name (str): Name suffix for output plots (e.g., "in", "ex")
        middate (datetime): Split date for in-sample vs out-of-sample
                           If None, uses entire dataset

    Returns:
        pd.DataFrame: full_df augmented with:
            - o2cC_B_ma_coef: Intraday coefficient (always 0)
            - o2c{1..horizon-1}_B_ma_coef: Coefficients for lagged daily signals
            - badj_m: Combined forecast (main output)

    Regression Strategy:
    -------------------
    - Fits o2c0_B_ma at multiple lags (1 to horizon) against forward returns
    - Uses daily regression mode
    - Extracts coefficient at full horizon (coef0)
    - Computes residual coefficients: coef[lag] = coef0 - fitted_coef[lag]
    - This captures incremental predictive power of recent lags

    Final Forecast:
        badj_m = o2cC_B_ma * 0 +  # Intraday component disabled
                 sum_{lag=1}^{horizon-1} (o2c{lag}_B_ma * residual_coef[lag])

    Output Files:
        - badj_daily_{name}_{dates}.png: Daily regression plot

    Notes:
        - Intraday coefficient hardcoded to 0 (line 87)
        - Only daily lags contribute to forecast
        - Sector-specific fitting (Energy vs others)
        - All coefficients printed to console
        - Creates forecast columns if they don't exist
    """
    if 'badj_m' not in full_df.columns:
        print "Creating forecast columns..."
        full_df['badj_m'] = np.nan
        full_df[ 'o2cC_B_ma_coef' ] = np.nan
        for lag in range(1, horizon+1):
            full_df[ 'o2c' + str(lag) + '_B_ma_coef' ] = np.nan

    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    outsample = False
    if middate is not None:
        outsample = True
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df, dailyForwardRets_df = regress_alpha(insample_daily_df, 'o2c0_B_ma', lag, outsample, False)
        full_df = merge_intra_data(dailyForwardRets_df, full_df)
        fits_df = fits_df.append(fitresults_df, ignore_index=True)  
    plot_fit(fits_df, "badj_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['o2c0_B_ma'].ix[horizon].ix['coef']
    full_df.ix[ outsample_intra_df.index, 'o2cC_B_ma_coef' ] = 0#coef0
    print "{} Coef0: {}".format(name, coef0)
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['o2c0_B_ma'].ix[lag].ix['coef'] 
        print "{} Coef{}: {}".format(name, lag, coef)
        full_df.ix[ outsample_intra_df.index, 'o2c'+str(lag)+'_B_ma_coef' ] = coef

    full_df.ix[ outsample_intra_df.index, 'badj_m'] = full_df['o2cC_B_ma'] * full_df['o2cC_B_ma_coef']
    for lag in range(1,horizon):
        full_df.ix[ outsample_intra_df.index, 'badj_m'] += full_df['o2c'+str(lag)+'_B_ma'] * full_df['o2c'+str(lag)+'_B_ma_coef']

    return full_df

def calc_o2c_forecast(daily_df, intra_df, horizon, outsample):
    """
    Main entry point: compute multi-period beta-adjusted forecast with sector split.

    Orchestrates the full pipeline from raw data to final forecasts, with
    separate regression fitting for Energy sector vs all other sectors.

    The sector split allows sector-specific coefficient optimization,
    accounting for different dynamics in the Energy sector.

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
        daily_df → calc_o2c() → daily_results with lags
        intra_df → calc_o2c_intra() → intra_results
        [merge daily + intra]
        → o2c_fits(Energy sector) → partial forecasts
        → o2c_fits(ex-Energy) → complete forecasts
        → combine and return

    Sector Processing:
        1. Energy sector: Fit separate regression for Energy stocks
        2. Ex-Energy: Fit regression for all non-Energy stocks
        3. Results combined into full_df

    Notes:
        - middate computed as midpoint if outsample=True
        - Energy sector handled separately for robustness
        - Uses module 'sector_name' column from data
        - Final forecast in 'badj_m' column
        - Both full and outsample dataframes returned
    """
    daily_df = calc_o2c(daily_df, horizon) 
    intra_df = calc_o2c_intra(intra_df, daily_df)
    full_df = merge_intra_data(daily_df, intra_df)

    middate = None
    if outsample:
        middate = intra_df.index[0][0] + (intra_df.index[len(intra_df)-1][0] - intra_df.index[0][0]) / 2
        print "Setting fit period before {}".format(middate)

    sector_name = 'Energy'
    print "Running o2c for sector {}".format(sector_name)
    sector_df = daily_df[ daily_df['sector_name'] == sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] == sector_name ]
    full_df = o2c_fits(sector_df, sector_intra_df, full_df, horizon, "in", middate)

    print "Running o2c for sector {}".format(sector_name)
    sector_df = daily_df[ daily_df['sector_name'] != sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] != sector_name ]
    full_df = o2c_fits(sector_df, sector_intra_df, full_df, horizon, "ex", middate)

    outsample_df = full_df
    if outsample:
        outsample_df = full_df[ full_df['date'] > middate ]
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

    pname = "./badj_m" + start + "." + end

    start = dateparser.parse(start)
    end = dateparser.parse(end)

    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print "Did not load cached data..."

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

    dump_alpha(outsample_df, 'badj_m')
    dump_all(outsample_df)

