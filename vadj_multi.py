#!/usr/bin/env python
"""Volume-Adjusted Multi-Period Strategy (vadj_multi)

This module implements a simplified volume-adjusted strategy that focuses on
daily signals only (no intraday component). It differs from vadj.py by removing
the intraday vadjC signal and using only the daily vadj0 signal with multiple
lagged periods.

Key Differences from vadj.py:
-----------------------------
1. **Daily-only signals**: No intraday volume signals (vadjC is commented out)
2. **Simpler volume calculation**: Uses raw relative volume without market adjustment
3. **Direct beta adjustment**: vadj0 = rv * badjret (not sign(badjret))
4. **Fit methodology**: Only daily regression, no intraday hourly fits

Strategy Overview:
-----------------
The multi-period approach uses multiple lagged daily volume-adjusted signals
to capture mean reversion at different horizons. By fitting coefficients at
multiple lags, the model can weight near-term vs. longer-term reversion patterns.

Signal Formula:
--------------
Daily signal:
  rv = tradable_volume / tradable_med_volume_21
  bret = beta * mkt_cap_weighted_market_return
  badjret = log_ret - bret
  vadj0 = rv * badjret  (note: not sign(badjret))
  vadj0_B_ma = industry_demeaned(winsorize(vadj0))

Final forecast:
  vadj_b = sum(vadj{lag}_B_ma * coef_lag for lag in 1..horizon-1)
  where coef_lag = coef_horizon - coef_at_lag

Rationale:
---------
This simpler approach may be preferable when:
- Intraday data quality is poor
- Focus is on longer-term (multi-day) reversion
- Lower computational overhead is desired
- Daily-only trading is preferred

Parameters:
----------
horizon : int
    Number of daily lags to use (default: 3)
    Typical range: 2-5 days

Market Impact:
-------------
- Daily-only signals reduce turnover compared to intraday rebalancing
- Volume normalization still provides liquidity-aware sizing
- Industry neutralization maintains sector balance

Data Requirements:
-----------------
Daily data: close, log_ret, pbeta, tradable_volume, tradable_med_volume_21,
            mkt_cap_y, ind1, sector_name
Intraday data: Only used for merging daily signals, no intraday calculations

Usage:
------
  python vadj_multi.py --start=20130101 --end=20130630 --mid=20130315 --horizon=3

Output:
-------
Creates alpha forecast file: vadj_b.h5
Creates fit diagnostic plot: vadj_daily_*.png
"""

from __future__ import division, print_function

from regress import *
from calc import *
from loaddata import *
from util import *

def wavg(group):
    """Calculate market cap-weighted beta-adjusted returns for a date group.

    Computes the market return as a market cap-weighted average of log returns,
    then multiplies each stock's beta by this market return to get the expected
    return due to market exposure.

    Args:
        group: DataFrame group for a single date containing:
            - pbeta: predicted beta from Barra model
            - log_ret: log return for the day
            - mkt_cap_y: market capitalization (lagged)

    Returns:
        Series: Beta * market_return for each stock in the group
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg2(group):
    b = group['pbeta']
    d = group['cur_log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg_ind(group):
    d = group['vadj0_B']
    w = group['mkt_cap_y'] / 1e6
    res = ((d * w).sum() / w.sum())
    return res

def calc_vadj_daily(daily_df, horizon):
    """Calculate simplified volume-adjusted signals from daily data.

    This version uses a simpler volume calculation compared to vadj.py:
    - No market-wide volume adjustment
    - Direct relative volume calculation
    - Uses badjret magnitude (not sign)

    Signal Construction:
    1. Calculate relative volume: rv = volume / median_volume
    2. Calculate beta-adjusted returns: badjret = log_ret - beta*mkt_return
    3. Combine: vadj0 = rv * badjret (note: uses magnitude, not sign)
    4. Winsorize and industry-neutralize
    5. Create lagged signals for multi-period forecasting

    Args:
        daily_df: DataFrame with daily data, indexed by (date, ticker)
            Required columns: tradable_volume, tradable_med_volume_21,
                             log_ret, pbeta, mkt_cap_y, gdate, ind1
        horizon: Number of lags to create

    Returns:
        DataFrame: Original data plus vadj0_B_ma and vadj{1..horizon}_B_ma columns
    """
    print("Caculating daily vadj...")
    result_df = filter_expandable(daily_df)

    print("Calculating vadj0...")
    result_df['rv'] = result_df['tradable_volume'].astype(float) / result_df['tradable_med_volume_21']

    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_df = result_df.dropna(subset=['log_ret', 'pbeta', 'mkt_cap_y', 'gdate'])
    result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['log_ret'] - result_df['bret']
    result_df['vadj0'] = result_df['rv'] * result_df['badjret']

    result_df = result_df.dropna(subset=['vadj0'])
    result_df['vadj0_B'] = winsorize_by_date(result_df['vadj0'])

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['vadj0_B', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=False).transform(demean)
    result_df['vadj0_B_ma'] = indgroups['vadj0_B']

    print("Calulating lags...")
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['vadj' + str(lag) + '_B_ma'] = shift_df['vadj0_B_ma']

    print("Calculated {} values".format(len(result_df['vadj0_B_ma'].dropna())))
    return result_df

def calc_vadj_intra(intra_df):
    """Calculate intraday volume-adjusted signals (partially disabled).

    NOTE: This function is present for compatibility but the vadjC signal is not
    used in the final forecast. The commented-out sections show it previously
    calculated intraday volume signals similar to vadj.py.

    Current behavior: Calculates vadjC_B_ma but it's not used in vadj_fits().

    Args:
        intra_df: DataFrame with intraday bar data
            Required columns: overnight_log_ret, iclose, dopen, pbeta,
                             mkt_cap_y, dvolume, dvwap, dpvolume_med_21,
                             giclose_ts, date, ind1

    Returns:
        DataFrame: Original data plus vadjC_B_ma column (unused)
    """
    print("Calculating vadj intra...")
    result_df = filter_expandable(intra_df)

    print("Calulating vadjC...")
    result_df['cur_log_ret'] = result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))
#    result_df['c2c_badj'] = result_df['cur_log_ret'] / result_df['pbeta']
    result_df['bret'] = result_df[['cur_log_ret', 'pbeta', 'mkt_cap_y', 'giclose_ts']].groupby(['giclose_ts'], sort=False).apply(wavg2).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['cur_log_ret'] - result_df['bret']
    result_df['rv_i'] = (result_df['dvolume'].astype(float) * result_df['dvwap']) / result_df['dpvolume_med_21']
    result_df['vadjC'] = result_df['rv_i'] * result_df['badjret']
    result_df['vadjC_B'] = winsorize_by_ts(result_df['vadjC'])

    print("Calulating vadjC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['vadjC_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['vadjC_B_ma'] = indgroups['vadjC_B']

    print("Calculated {} values".format(len(result_df['vadjC_B_ma'].dropna())))
    return result_df

def vadj_fits(daily_df, intra_df, horizon, name, middate=None):
    """Fit daily volume-adjusted model only (no intraday component).

    This simplified version:
    - Skips intraday signal fitting (lines commented out)
    - Only fits daily lagged signals at multiple horizons
    - Uses incremental coefficients: coef_lag = coef_horizon - coef_lag

    The forecast combines lagged daily signals:
      vadj_b = sum(vadj{lag}_B_ma * (coef_horizon - coef_lag))

    This assumes the full horizon forecast captures the total reversion, and
    subtracting the lag forecast gives the incremental reversion from lag to horizon.

    Args:
        daily_df: DataFrame with daily vadj signals and forward returns
        intra_df: DataFrame with intraday data (for index/merging only)
        horizon: Number of daily lags to use in forecast
        name: Name suffix for plot files
        middate: Split date for in-sample/out-sample. If None, use all data

    Returns:
        DataFrame: out-sample intraday data with vadj_b forecast column
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['vadj_b'] = np.nan
    outsample_intra_df[ 'vadjC_B_ma_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_intra_df[ 'vadj' + str(lag) + '_B_ma_coef' ] = np.nan

    # fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    # fitresults_df = regress_alpha(insample_intra_df, 'vadjC_B_ma', horizon, True, 'intra')
    # fits_df = fits_df.append(fitresults_df, ignore_index=True)
    # plot_fit(fits_df, "vadj_intra_"+name+"_" + df_dates(insample_intra_df))
    # fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    # unstacked = outsample_intra_df[ ['ticker'] ].unstack()
    # coefs = dict()
    # coefs[1] = unstacked.between_time('09:30', '10:31').stack().index
    # coefs[2] = unstacked.between_time('10:30', '11:31').stack().index
    # coefs[3] = unstacked.between_time('11:30', '12:31').stack().index
    # coefs[4] = unstacked.between_time('12:30', '13:31').stack().index
    # coefs[5] = unstacked.between_time('13:30', '14:31').stack().index
    # coefs[6] = unstacked.between_time('14:30', '15:31').stack().index
    # print fits_df.head()
    # for ii in range(1,7):
    #     outsample_intra_df.ix[ coefs[ii], 'vadjC_B_ma_coef' ] = fits_df.ix['vadjC_B_ma'].ix[ii].ix['coef']
    
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'vadj0_B_ma', lag, True, 'daily') 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "vadj_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['vadj0_B_ma'].ix[horizon].ix['coef']
#    outsample_intra_df[ 'vadjC_B_ma_coef' ] = coef0
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['vadj0_B_ma'].ix[lag].ix['coef'] 
        print("Coef{}: {}".format(lag, coef))
        outsample_intra_df[ 'vadj'+str(lag)+'_B_ma_coef' ] = coef

#    outsample_intra_df[ 'vadj_b'] = outsample_intra_df['vadjC_B_ma'] * outsample_intra_df['vadjC_B_ma_coef']
    outsample_intra_df[ 'vadj_b'] = 0
    for lag in range(1,horizon):
        outsample_intra_df[ 'vadj_b'] += outsample_intra_df['vadj'+str(lag)+'_B_ma'] * outsample_intra_df['vadj'+str(lag)+'_B_ma_coef']
    
    return outsample_intra_df

def calc_vadj_forecast(daily_df, intra_df, horizon, middate):
    """Calculate volume-adjusted forecasts using daily signals only.

    Main pipeline for vadj_multi strategy:
    1. Calculate daily vadj signals with multiple lags
    2. Calculate forward returns for regression
    3. Skip intraday signal calculation (commented out)
    4. Merge daily signals onto intraday index
    5. Fit separate models for Energy vs. other sectors
    6. Combine forecasts

    Note: This function uses the intraday DataFrame only for its index structure,
    not for intraday calculations. The final forecast is still generated at the
    intraday frequency but uses only daily signals.

    Args:
        daily_df: DataFrame with daily data indexed by (date, ticker)
        intra_df: DataFrame with intraday data (used for index only)
        horizon: Number of daily lags to use
        middate: Date to split in-sample (fitting) and out-sample (forecasting)

    Returns:
        DataFrame: Intraday-indexed data with vadj_b forecast column
    """
    daily_results_df = calc_vadj_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    #intra_results_df = calc_vadj_intra(intra_df)
    intra_results_df = intra_df
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)

    sector_name = 'Energy'
    print("Running vadj for sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    result1_df = vadj_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)

    print("Running vadj for sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] != sector_name ]
    result2_df = vadj_fits(sector_df, sector_intra_results_df, horizon, "in", middate)

    result_df = pd.concat([result1_df, result2_df], verify_integrity=True)
    return result_df

def memtest():
    start = "20110101"
    end = "20120101"
    pname = "./vadj_b" + start + "." + end
    daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
    intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
    outsample_df = calc_vadj_forecast(daily_df, intra_df, 3, True)

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--horizon",action="store",dest="horizon",default=3)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = int(args.horizon)

    pname = "./vadj_b" + start + "." + end

    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)

    loaded = False
    try:        
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Could not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)    
        BARRA_COLS = ['ind1', 'pbeta']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close', 'overnight_log_ret', 'tradable_volume', 'tradable_med_volume_21']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        uni_df = price_df[['ticker']]
        DBAR_COLS = ['close', 'dopen', 'dvolume', 'dvwap']
        intra_df = load_daybars(uni_df, start, end, DBAR_COLS)
        daily_df = merge_barra_data(price_df, barra_df)
        intra_df = merge_intra_data(daily_df, intra_df)
        intra_df = calc_vol_profiles(intra_df)
        dump_hd5(daily_df, pname+"_daily.h5")
        dump_hd5(intra_df, pname+"_intra.h5")

    outsample_df = calc_vadj_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(outsample_df, 'vadj_b')
    dump_hd5(outsample_df)

