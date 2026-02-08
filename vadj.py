#!/usr/bin/env python
"""Volume-Adjusted Mean Reversion Strategy (vadj)

This module implements a volume-adjusted mean reversion strategy that combines
daily and intraday signals. The strategy is based on the principle that unusual
volume combined with price moves indicates temporary imbalances that revert.

Strategy Overview:
-----------------
The strategy creates two types of signals:
1. Daily signals (vadj0): Based on daily volume and beta-adjusted returns
2. Intraday signals (vadjC): Based on intraday volume patterns and returns

The signals are combined using regression-fitted coefficients with multiple
lags to capture different reversion horizons.

Core Methodology:
----------------
1. **Volume Normalization**: Calculate relative volume as actual volume divided
   by median volume over 21 days. This identifies unusual volume days.

2. **Beta-Adjusted Returns**: Remove market component from returns using market
   cap-weighted beta adjustment to isolate stock-specific moves.

3. **Volume-Return Interaction**: Multiply relative volume by the sign of
   beta-adjusted returns. High volume + positive move = overbuying signal.

4. **Industry Neutralization**: Demean signals within each industry group to
   create market-neutral positions.

5. **Multi-Horizon Forecasting**: Combine current intraday signal with lagged
   daily signals using regression-fitted coefficients.

Signal Formulas:
---------------
Daily signal:
  rv = tradable_volume / tradable_med_volume_21
  bret = beta * mkt_cap_weighted_market_return
  badjret = log_ret - bret
  vadj0 = rv * sign(badjret)
  vadj0_B_ma = industry_demeaned(winsorize(vadj0))

Intraday signal:
  rv_i = (dvolume * dvwap) / dpvolume_med_21
  cur_ret = overnight_ret + log(iclose/dopen)
  badjret_i = cur_ret - beta * mkt_cap_weighted_intraday_return
  vadjC = rv_i * sign(badjret_i)
  vadjC_B_ma = industry_demeaned(winsorize(vadjC))

Final forecast:
  vadj_b = vadjC_B_ma * coef_intra +
           sum(vadj{lag}_B_ma * coef_lag for lag in 1..horizon-1)

Parameters:
----------
horizon : int
    Number of days to look back for daily signals (default: 2)
    Typical range: 1-5 days
freq : str
    Intraday bar frequency (default: '15Min')
    Used for volume profile calculations
middate : datetime
    Split date between in-sample (for fitting) and out-sample (for forecast)

Market Impact Considerations:
----------------------------
- Volume normalization reduces positions when liquidity is low
- Sign-based signals (not magnitude) prevent excessive trading
- Industry neutralization reduces sector concentration risk
- Multiple lags smooth out reversion timing uncertainty

Sector Handling:
---------------
The strategy fits separate coefficients for Energy sector vs. all other sectors.
This allows the model to capture different reversion dynamics in the Energy
sector which may have different volume/price relationships due to commodity
exposure.

Data Requirements:
-----------------
Daily data: close, log_ret, pbeta, tradable_volume, tradable_med_volume_21,
            mkt_cap_y, ind1, sector_name
Intraday data: dopen, iclose, dvolume, dvwap, dpvolume_med_21, overnight_log_ret

Usage:
------
  python vadj.py --start=20130101 --end=20130630 --mid=20130315 --horizon=2

Output:
-------
Creates alpha forecast file: vadj_b.h5
Creates fit diagnostic plots: vadj_daily_*.png, vadj_intra_*.png
"""

from __future__ import division, print_function

from regress import *
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
            - gdate: group date

    Returns:
        Series: Beta * market_return for each stock in the group
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print("Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum())))
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg2(group):
    """Calculate market cap-weighted beta-adjusted returns for intraday groups.

    Similar to wavg() but uses current intraday returns (overnight + intraday)
    instead of daily close-to-close returns.

    Args:
        group: DataFrame group for a single timestamp containing:
            - pbeta: predicted beta from Barra model
            - cur_log_ret: current log return (overnight + intraday)
            - mkt_cap_y: market capitalization (lagged)

    Returns:
        Series: Beta * intraday_market_return for each stock in the group
    """
    b = group['pbeta']
    d = group['cur_log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg_ind(group):
    """Calculate market cap-weighted average of vadj signals within a group.

    Used for industry-level signal aggregation (not currently used in main flow).

    Args:
        group: DataFrame group containing:
            - vadj0_B: winsorized volume-adjusted signal
            - mkt_cap_y: market capitalization (lagged)

    Returns:
        float: Market cap-weighted average signal
    """
    d = group['vadj0_B']
    w = group['mkt_cap_y'] / 1e6
    res = ((d * w).sum() / w.sum())
    return res

def volmult_i(group):
    """Normalize intraday dollar volume to account for market-wide volume changes.

    Adjusts individual stock volumes by the ratio of total dollar volume to
    total median dollar volume across all stocks in the group. This removes
    the market-wide volume component to focus on stock-specific volume anomalies.

    Args:
        group: DataFrame group for a single timestamp containing:
            - dpvolume: dollar volume (dvolume * dvwap)
            - dpvolume_med_21: median dollar volume over 21 days

    Returns:
        Series: Market-adjusted dollar volume for each stock
    """
    d = group['dpvolume']
    m = group['dpvolume_med_21']
    adj = d.sum()/m.sum()
    res = group['dpvolume'] / adj
    return res

def volmult2(group):
    """Normalize daily tradable volume to account for market-wide volume changes.

    Adjusts individual stock volumes by the ratio of total tradable volume to
    total median tradable volume across all stocks in the group. This removes
    the market-wide volume component to focus on stock-specific volume anomalies.

    Args:
        group: DataFrame group for a single date containing:
            - tradable_volume: tradable volume for the day
            - tradable_med_volume_21: median tradable volume over 21 days

    Returns:
        Series: Market-adjusted tradable volume for each stock
    """
    d = group['tradable_volume']
    m = group['tradable_med_volume_21']
    adj = d.sum()/m.sum()
    res = group['tradable_volume'] / adj
    return res

def calc_vadj_daily(daily_df, horizon):
    """Calculate volume-adjusted signals from daily data with multiple lags.

    This function computes the daily volume-adjusted signal (vadj0) and creates
    lagged versions for multi-period forecasting. The signal combines relative
    volume with beta-adjusted returns to capture volume-driven mean reversion.

    Signal Construction:
    1. Market-adjust volume: tradable_volume_adj accounts for market-wide volume
    2. Calculate relative volume: rv = volume / median_volume
    3. Calculate beta-adjusted returns: badjret = log_ret - beta*mkt_return
    4. Combine: vadj0 = rv * sign(badjret)
    5. Winsorize and industry-neutralize
    6. Create lagged signals for multi-period forecasting

    Args:
        daily_df: DataFrame with daily price and volume data, indexed by (date, ticker)
            Required columns: tradable_volume, tradable_med_volume_21,
                             tradable_med_volume_21_y, log_ret, pbeta,
                             mkt_cap_y, gdate, ind1
        horizon: Number of lags to create (e.g., horizon=2 creates vadj1, vadj2)

    Returns:
        DataFrame: Original data plus vadj0_B_ma and vadj{1..horizon}_B_ma columns
    """
    print("Caculating daily vadj...")
    result_df = filter_expandable(daily_df)

    print("Calculating vadj0...")
    result_df['tradable_volume_adj'] = result_df[['tradable_med_volume_21', 'tradable_volume', 'gdate']].groupby('gdate').apply(volmult2).reset_index(level=0)['tradable_volume']
    result_df['rv'] = result_df['tradable_volume_adj'].astype(float) / result_df['tradable_med_volume_21_y']
    # result_df['dpvolume'] = result_df['dvolume'].astype(float) * result_df['dvwap']
    # result_df['dpvolume_adj'] = result_df[['dpvolume_med_21', 'dpvolume', 'gdate']].groupby('gdate').apply(volmult).reset_index(level=0)['dpvolume']
    # result_df['rv'] = (result_df['dpvolume_adj'] - result_df['dpvolume_med_21']) / result_df['dpvolume_std_21']

    print(result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].head())
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_df = result_df.dropna(subset=['log_ret', 'pbeta', 'mkt_cap_y', 'gdate'])
    result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['log_ret'] - result_df['bret']
    result_df['vadj0'] = result_df['rv'] * np.sign(result_df['badjret']).fillna(0)

#    result_df = result_df.dropna(subset=['vadj0'])
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
    """Calculate volume-adjusted signals from intraday bar data.

    This function computes intraday volume-adjusted signals (vadjC) that capture
    volume anomalies during specific intraday periods. The signal uses the same
    volume-return interaction as the daily signal but at the intraday level.

    Signal Construction:
    1. Calculate current return: overnight_ret + intraday_ret
    2. Market-adjust dollar volume across all stocks in the timestamp group
    3. Calculate relative dollar volume: rv_i = dollar_volume / median
    4. Calculate beta-adjusted returns using intraday market returns
    5. Combine: vadjC = rv_i * sign(badjret_i)
    6. Winsorize and industry-neutralize

    Args:
        intra_df: DataFrame with intraday bar data, indexed by (timestamp, ticker)
            Required columns: overnight_log_ret, iclose, dopen, pbeta, mkt_cap_y,
                             dvolume, dvwap, dpvolume_med_21, giclose_ts, ind1

    Returns:
        DataFrame: Original data plus vadjC_B_ma column (industry-neutral signal)
    """
    print("Calculating vadj intra...")
    result_df = filter_expandable(intra_df)

    print("Calulating vadjC...")
    result_df['cur_log_ret'] = result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))
    result_df['bret_i'] = result_df[['cur_log_ret', 'pbeta', 'mkt_cap_y', 'giclose_ts']].groupby(['giclose_ts'], sort=False).apply(wavg2).reset_index(level=0)['pbeta']
    result_df['badjret_i'] = result_df['cur_log_ret'] - result_df['bret_i']

    result_df['dpvolume'] = result_df['dvolume'].astype(float) * result_df['dvwap']
    result_df['dpvolume_adj'] = result_df[['dpvolume_med_21', 'dpvolume', 'giclose_ts']].groupby('giclose_ts').apply(volmult_i).reset_index(level=0)['dpvolume']
    result_df['rv_i'] = result_df['dpvolume_adj'].astype(float) / result_df['dpvolume_med_21']

    # result_df['rv_i'] = (result_df['dpvolume_adj'] - result_df['dpvolume_med_21']) / result_df['dpvolume_std_21']

    result_df['vadjC'] = result_df['rv_i'] * np.sign(result_df['badjret_i'])
    result_df['vadjC_B'] = winsorize_by_ts(result_df['vadjC'])

    print("Calulating vadjC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['vadjC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['vadjC_B_ma'] = indgroups['vadjC_B']

    print("Calculated {} values".format(len(result_df['vadjC_B_ma'].dropna())))
    return result_df

def vadj_fits(daily_df, intra_df, horizon, name, middate=None):
    """Fit volume-adjusted model and generate forecasts.

    This function:
    1. Splits data into in-sample (before middate) and out-sample (after middate)
    2. Fits intraday signals to forward returns for each hour of the trading day
    3. Fits daily lagged signals to forward returns at multiple horizons
    4. Combines fitted coefficients to generate final forecast (vadj_b)

    The intraday model fits separate coefficients for 6 hourly periods:
      - 9:30-10:30, 10:30-11:30, 11:30-12:30, 12:30-13:30, 13:30-14:30, 14:30-16:00

    The daily model uses incremental coefficients:
      coef_lag = coef_horizon - coef_lag
    This captures the incremental predictive power of each lag.

    Args:
        daily_df: DataFrame with daily vadj signals and forward returns
        intra_df: DataFrame with intraday vadjC signals
        horizon: Number of daily lags to use in forecast
        name: Name suffix for plot files ("in" or "ex" for sector splits)
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

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df = regress_alpha(insample_intra_df, 'vadjC_B_ma', horizon, False, 'intra')
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "vadj_intra_"+name+"_" + df_dates(insample_intra_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    unstacked = outsample_intra_df[ ['ticker'] ].unstack()
    coefs = dict()
    coefs[1] = unstacked.between_time('09:30', '10:31').stack().index
    coefs[2] = unstacked.between_time('10:30', '11:31').stack().index
    coefs[3] = unstacked.between_time('11:30', '12:31').stack().index
    coefs[4] = unstacked.between_time('12:30', '13:31').stack().index
    coefs[5] = unstacked.between_time('13:30', '14:31').stack().index
    coefs[6] = unstacked.between_time('14:30', '15:59').stack().index
    print(fits_df.head(10))
    for ii in range(1,7):
        outsample_intra_df.ix[ coefs[ii], 'vadjC_B_ma_coef' ] = fits_df.ix['vadjC_B_ma'].ix[ii].ix['coef']
    
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'vadj0_B_ma', lag, False, 'daily') 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "vadj_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['vadj0_B_ma'].ix[horizon].ix['coef']
    print("Coef0: {}".format(coef0))
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['vadj0_B_ma'].ix[lag].ix['coef'] 
        print("Coef{}: {}".format(lag, coef))
        outsample_intra_df[ 'vadj'+str(lag)+'_B_ma_coef' ] = coef

    outsample_intra_df[ 'vadj_b'] = outsample_intra_df['vadjC_B_ma'] * outsample_intra_df['vadjC_B_ma_coef']
    for lag in range(1,horizon):
        outsample_intra_df[ 'vadj_b'] += outsample_intra_df['vadj'+str(lag)+'_B_ma'] * outsample_intra_df['vadj'+str(lag)+'_B_ma_coef']
    
    return outsample_intra_df

def calc_vadj_forecast(daily_df, intra_df, horizon, middate):
    """Calculate volume-adjusted forecasts with sector-specific models.

    This is the main entry point that orchestrates the full forecasting pipeline:
    1. Calculate daily vadj signals with lags
    2. Calculate forward returns for regression
    3. Calculate intraday vadjC signals
    4. Merge daily and intraday data
    5. Fit separate models for Energy sector and all other sectors
    6. Combine forecasts

    The strategy uses sector-specific models because the Energy sector can have
    different volume-price dynamics due to commodity exposure and different
    trading patterns.

    Args:
        daily_df: DataFrame with daily data indexed by (date, ticker)
        intra_df: DataFrame with intraday data indexed by (timestamp, ticker)
        horizon: Number of daily lags to use in forecasting
        middate: Date to split in-sample (fitting) and out-sample (forecasting)

    Returns:
        DataFrame: Intraday data with vadj_b forecast column for all stocks
    """
    daily_results_df = calc_vadj_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    intra_results_df = calc_vadj_intra(intra_df)
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)

    sector_name = "Energy"
    results = list()
    print("Running vadj for sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    result_df = vadj_fits(sector_df, sector_intra_results_df, horizon, "in", middate)
    results.append(result_df)

    print("Running vadj excluding sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] != sector_name ]
    result_df = vadj_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)
    results.append(result_df)

    result_df = pd.concat(results, verify_integrity=True)
    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--freq",action="store",dest="freq",default='15Min')
    parser.add_argument("--horizon",action="store",dest="horizon",default=2)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = int(args.horizon)
    freq = args.freq
    pname = "./vadj_b" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)

    loaded = False
    try:        
        print("Looking " + pname+"_daily.h5")
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Could not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)    
        BARRA_COLS = ['ind1', 'pbeta']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close', 'overnight_log_ret', 'tradable_volume', 'tradable_med_volume_21', 'volat_21']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        DBAR_COLS = ['close', 'dopen', 'dvolume', 'dvwap']
        intra_df = load_daybars(price_df[['ticker']], start, end, DBAR_COLS, freq)
        daily_df = merge_barra_data(price_df, barra_df)
        intra_df = merge_intra_data(daily_df, intra_df)
        intra_df = calc_vol_profiles(intra_df)
        print("one")
        print(intra_df.columns)
        daily_df = merge_intra_eod(daily_df, intra_df)
        print("two")
        print(daily_df.columns)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    outsample_df = calc_vadj_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(outsample_df, 'vadj_b')
#    dump_all(outsample_df)

