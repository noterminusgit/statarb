#!/usr/bin/env python
"""Volume-Adjusted Intraday-Only Strategy (vadj_intra)

This module implements an intraday-only volume-adjusted strategy that focuses
exclusively on intraday volume patterns. It differs from vadj.py by removing
the daily signal component entirely and using only the intraday vadjC signal.

Key Differences from vadj.py:
-----------------------------
1. **Intraday-only signals**: No daily vadj0 component
2. **Hourly coefficients**: Fits separate coefficients for each trading hour
3. **Simpler pipeline**: No multi-lag daily signals to combine
4. **End-of-day focus**: Uses 'intra_eod' regression mode

Strategy Overview:
-----------------
The intraday-only approach captures volume-driven mean reversion within the
trading day. By fitting separate coefficients for different hours, the model
can account for different liquidity patterns and price discovery processes
throughout the trading session.

Signal Formula:
--------------
Intraday signal (vadjC):
  cur_ret = overnight_ret + log(iclose/dopen)
  bret_i = beta * mkt_cap_weighted_intraday_return
  badjret_i = cur_ret - bret_i
  rv_i = (dvolume * dvwap) / dpvolume_med_21
  vadjC = rv_i * sign(badjret_i)
  vadjC_B_ma = industry_demeaned(winsorize(vadjC))

Final forecast:
  vadj_i = vadjC_B_ma * coef_hour
  where coef_hour varies by time of day (6 hourly periods)

Hourly Period Coefficients:
--------------------------
The model fits separate coefficients for six intraday periods:
  1. 09:30-10:30 (Market open, high volatility)
  2. 10:30-11:30 (Morning session)
  3. 11:30-12:30 (Mid-day, typically lower volume)
  4. 12:30-13:30 (Afternoon start)
  5. 13:30-14:30 (Late afternoon)
  6. 14:30-16:00 (Market close, high volume)

Rationale:
---------
Different trading hours have different characteristics:
- Open: Large order imbalances from overnight news
- Mid-day: Lower volume, less efficient pricing
- Close: Portfolio rebalancing, index fund flows

Parameters:
----------
horizon : int
    Forward return horizon for fitting (default: 0)
    Note: Only used for forward return calculation, not for lags
freq : str
    Intraday bar frequency (default: '15Min')

Market Impact:
-------------
- Intraday signals can capture short-term liquidity imbalances
- Hourly coefficients adapt to time-varying market microstructure
- Sign-based signals prevent over-trading on small moves
- Industry neutralization maintains sector balance

Data Requirements:
-----------------
Daily data: overnight_log_ret, tradable_volume, tradable_med_volume_21,
            pbeta, mkt_cap_y, ind1, sector_name
Intraday data: dopen, iclose (dclose), dvolume, dvwap, dpvolume_med_21

Usage:
------
  python vadj_intra.py --start=20130101 --end=20130630 --mid=20130315 --freq=15Min

Output:
-------
Creates alpha forecast file: vadj_i.h5
Creates fit diagnostic plot: vadj_intra_*.png
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def wavg(group):
    """Calculate market cap-weighted beta-adjusted returns for a date group.

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

def calc_vadj_intra(intra_df):
    """Calculate intraday-only volume-adjusted signals.

    This is the core signal calculation for the intraday-only strategy. It
    computes volume-adjusted signals using intraday returns and volume patterns.

    Signal Construction:
    1. Calculate current return: overnight_ret + log(iclose/dopen)
    2. Calculate beta-adjusted return using intraday market returns
    3. Calculate relative dollar volume: rv_i = dollar_volume / median
    4. Combine: vadjC = rv_i * sign(badjret_i)
    5. Winsorize and industry-neutralize

    Key differences from vadj.py's calc_vadj_intra:
    - Uses giclose_ts for grouping (not date)
    - Industry demeans by (giclose_ts, ind1) not (date, ind1)

    Args:
        intra_df: DataFrame with intraday bar data, indexed by (timestamp, ticker)
            Required columns: overnight_log_ret, iclose, dopen, pbeta, mkt_cap_y,
                             dvolume, dvwap, dpvolume_med_21, giclose_ts, ind1

    Returns:
        DataFrame: Original data plus vadjC_B_ma column
    """
    print("Calculating vadj intra...")
    result_df = filter_expandable(intra_df)

    print("Calulating vadjC...")
    result_df['cur_log_ret'] = result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))
#    result_df['c2c_badj'] = result_df['cur_log_ret'] / result_df['pbeta']
    result_df['bret'] = result_df[['cur_log_ret', 'pbeta', 'mkt_cap_y', 'giclose_ts']].groupby(['giclose_ts'], sort=False).apply(wavg2).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['cur_log_ret'] - result_df['bret']
    result_df['rv_i'] = (result_df['dvolume'].astype(float) * result_df['dvwap']) / result_df['dpvolume_med_21']
    result_df['vadjC'] = result_df['rv_i'] * np.sign(result_df['badjret'])
    result_df['vadjC_B'] = winsorize_by_ts(result_df['vadjC'])

    print("Calulating vadjC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['vadjC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['vadjC_B_ma'] = indgroups['vadjC_B']

    print("Calculated {} values".format(len(result_df['vadjC_B_ma'].dropna())))
    return result_df

def vadj_fits(daily_df, intra_df, horizon, name, middate=None):
    """Fit intraday volume-adjusted model with hourly coefficients.

    This simplified version:
    - Only fits intraday signals (no daily component)
    - Fits separate coefficients for 6 trading hour periods
    - Uses 'intra_eod' regression mode for end-of-day returns

    The hourly coefficients allow the model to capture different volume-return
    relationships at different times of day:
    - Morning: Large imbalances from overnight news
    - Mid-day: Lower liquidity, potentially stronger mean reversion
    - Close: Institutional rebalancing flows

    Args:
        daily_df: DataFrame with daily data (unused, kept for API compatibility)
        intra_df: DataFrame with intraday vadjC signals
        horizon: Forward return horizon for fitting
        name: Name suffix for plot files
        middate: Split date for in-sample/out-sample. If None, use all data

    Returns:
        DataFrame: out-sample intraday data with vadj_i forecast column
    """
    insample_intra_df = intra_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['vadj_i'] = np.nan
    outsample_intra_df[ 'vadjC_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df = regress_alpha(insample_intra_df, 'vadjC_B_ma', horizon, True, 'intra_eod')
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
        outsample_intra_df.loc[ coefs[ii], 'vadjC_B_ma_coef' ] = fits_df.loc['vadjC_B_ma'].loc[ii].loc['coef']
    
    outsample_intra_df[ 'vadj_i'] = outsample_intra_df['vadjC_B_ma'] * outsample_intra_df['vadjC_B_ma_coef']
    
    return outsample_intra_df

def calc_vadj_forecast(daily_df, intra_df, horizon, middate):
    """Calculate intraday-only volume-adjusted forecasts with sector models.

    Main pipeline for vadj_intra strategy:
    1. Calculate forward returns (for regression fitting)
    2. Calculate intraday vadjC signals
    3. Merge daily context data (beta, market cap, industry) onto intraday data
    4. Fit separate models for Energy vs. other sectors
    5. Combine forecasts

    The sector-specific models account for different volume-price dynamics
    in the Energy sector vs. other sectors.

    Args:
        daily_df: DataFrame with daily data for context
        intra_df: DataFrame with intraday bar data
        horizon: Forward return horizon for fitting
        middate: Date to split in-sample (fitting) and out-sample (forecasting)

    Returns:
        DataFrame: Intraday data with vadj_i forecast column for all stocks
    """
    daily_results_df = daily_df
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    intra_results_df = calc_vadj_intra(intra_df)
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

if __name__=="__main__":            
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
    freq = args.freq
    pname = "./vadj_i" + start + "." + end
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
        PRICE_COLS = ['close', 'overnight_log_ret', 'tradable_volume', 'tradable_med_volume_21']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        DBAR_COLS = ['dclose', 'dopen', 'dvolume', 'dvwap', 'dpvolume_med_21']
        intra_df = load_daybars(price_df[['ticker']], start, end, DBAR_COLS, freq)
        daily_df = merge_barra_data(price_df, barra_df)
        intra_df = merge_intra_data(daily_df, intra_df)
        intra_df = calc_vol_profiles(intra_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    outsample_df = calc_vadj_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(outsample_df, 'vadj_i')
#    dump_all(outsample_df)

