#!/usr/bin/env python
"""Volume-Adjusted Position Sizing Strategy (vadj_pos)

This module implements a volume-adjusted strategy with position sizing emphasis.
It combines daily and intraday signals similar to vadj.py but uses sign-based
directional signals for cleaner position entry/exit.

Key Differences from vadj.py:
-----------------------------
1. **Sign-based signals**: Uses sign(badjret) for both daily and intraday
2. **Cleaner directional signals**: Reduces noise from small price moves
3. **Position sizing focus**: Signal magnitude controlled by volume only
4. **Full model combination**: Includes both intraday and daily components

Strategy Overview:
-----------------
The position sizing approach prioritizes liquidity-aware position management.
By using sign() instead of the raw beta-adjusted return, the signal becomes
purely directional (buy/sell) with magnitude determined by relative volume.

This is particularly useful for:
- Reducing false signals from small random price moves
- Focusing on significant direction changes with volume confirmation
- Cleaner integration with portfolio optimization constraints

Signal Formulas:
---------------
Daily signal:
  rv = tradable_volume / tradable_med_volume_21_y
  bret = beta * mkt_cap_weighted_market_return
  badjret = log_ret - bret
  vadj0 = rv * sign(badjret)  # Direction only, magnitude from volume
  vadj0_B_ma = industry_demeaned(winsorize(vadj0))

Intraday signal:
  rv_i = (dvolume * dvwap) / dpvolume_med_21
  cur_ret = overnight_ret + log(iclose/dopen)
  badjret_i = cur_ret - beta * mkt_cap_weighted_intraday_return
  vadjC = rv_i * sign(badjret_i)  # Direction only, magnitude from volume
  vadjC_B_ma = industry_demeaned(winsorize(vadjC))

Final forecast:
  vadj_b = vadjC_B_ma * coef_intra +
           sum(vadj{lag}_B_ma * coef_lag for lag in 1..horizon-1)

Key Insight:
-----------
By using sign(badjret) instead of badjret:
- Signal is +1/-1 for direction, scaled by relative volume
- High volume + up move = high positive signal (likely overbought)
- High volume + down move = high negative signal (likely oversold)
- Low volume moves have small signals regardless of return magnitude

This creates liquidity-aware position sizing where:
- Larger positions when liquidity is high (rv > 1)
- Smaller positions when liquidity is low (rv < 1)
- Zero positions when volume is at median and no clear direction

Parameters:
----------
horizon : int
    Number of daily lags to use (default: 3)
    Typical range: 2-5 days
freq : str
    Intraday bar frequency (default: '30Min')
    30-minute bars provide good balance of signal and data quality

Market Impact Considerations:
----------------------------
- Position sizing naturally adapts to available liquidity
- Sign-based signals reduce excessive turnover
- Industry neutralization prevents sector concentration
- Multiple lags provide diversification across reversion horizons

Fit Methodology:
---------------
Similar to vadj.py, fits separate models for:
1. Intraday: 6 hourly coefficient periods
2. Daily: Multiple lag coefficients using incremental approach

Data Requirements:
-----------------
Daily data: close, log_ret, pbeta, tradable_volume, tradable_med_volume_21_y,
            tradable_med_volume_21, mkt_cap_y, ind1, sector_name
Intraday data: close, dopen, dvolume, dvwap, dpvolume_med_21, overnight_log_ret

Usage:
------
  python vadj_pos.py --start=20130101 --end=20130630 --mid=20130315 --horizon=3 --freq=30Min

Output:
-------
Creates alpha forecast file: vadj_b.h5
Creates fit diagnostic plots: vadj_daily_*.png, vadj_intra_*.png
Prints forecast distribution statistics
"""

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

def calc_vadj_daily(daily_df, horizon):
    """Calculate volume-adjusted position sizing signals from daily data.

    This version emphasizes position sizing by using sign-based directional
    signals. The signal magnitude comes purely from relative volume, with
    direction determined by the sign of beta-adjusted returns.

    Signal Construction:
    1. Calculate relative volume: rv = volume / median_volume
    2. Calculate beta-adjusted returns: badjret = log_ret - beta*mkt_return
    3. Combine: vadj0 = rv * sign(badjret)
       - Positive: High volume + up move (potential overbought)
       - Negative: High volume + down move (potential oversold)
    4. Winsorize and industry-neutralize
    5. Create lagged signals for multi-period forecasting

    The sign() operation creates cleaner directional signals:
    - Filters out noise from small random price moves
    - Signal magnitude reflects available liquidity (rv)
    - Direction reflects market sentiment (sign of badjret)

    Args:
        daily_df: DataFrame with daily data, indexed by (date, ticker)
            Required columns: tradable_volume, tradable_med_volume_21_y,
                             log_ret, pbeta, mkt_cap_y, gdate, ind1
        horizon: Number of lags to create

    Returns:
        DataFrame: Original data plus vadj0_B_ma and vadj{1..horizon}_B_ma columns
    """
    print "Caculating daily vadj..."
    result_df = filter_expandable(daily_df)

    print "Calculating vadj0..."
    result_df['rv'] = result_df['tradable_volume'].astype(float) / result_df['tradable_med_volume_21_y']

    print result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].head()
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_df = result_df.dropna(subset=['log_ret', 'pbeta', 'mkt_cap_y', 'gdate'])
    result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['log_ret'] - result_df['bret']
    result_df['vadj0'] = result_df['rv'] * np.sign(result_df['badjret'])

    result_df = result_df.dropna(subset=['vadj0'])
    result_df['vadj0_B'] = winsorize_by_date(result_df['vadj0'])

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['vadj0_B', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=False).transform(demean)
    result_df['vadj0_B_ma'] = indgroups['vadj0_B']

    print "Calulating lags..."
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['vadj' + str(lag) + '_B_ma'] = shift_df['vadj0_B_ma']

    print "Calculated {} values".format(len(result_df['vadj0_B_ma'].dropna()))
    return result_df

def calc_vadj_intra(intra_df):
    """Calculate intraday volume-adjusted position sizing signals.

    Similar to daily signals, uses sign-based directional signals for cleaner
    position entry/exit decisions at the intraday level.

    Signal Construction:
    1. Calculate current return: overnight_ret + log(iclose/dopen)
    2. Calculate beta-adjusted return using intraday market returns
    3. Calculate relative dollar volume: rv_i = dollar_volume / median
    4. Combine: vadjC = rv_i * sign(badjret_i)
    5. Winsorize and industry-neutralize

    The sign-based approach at the intraday level:
    - Focuses on clear directional moves confirmed by volume
    - Reduces noise from intraday volatility
    - Provides liquidity-aware signal magnitude

    Args:
        intra_df: DataFrame with intraday bar data, indexed by (timestamp, ticker)
            Required columns: overnight_log_ret, iclose, dopen, pbeta, mkt_cap_y,
                             dvolume, dvwap, dpvolume_med_21, giclose_ts, ind1

    Returns:
        DataFrame: Original data plus vadjC_B_ma column
    """
    print "Calculating vadj intra..."
    result_df = filter_expandable(intra_df)

    print "Calulating vadjC..."
    result_df['cur_log_ret'] = result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))
#    result_df['c2c_badj'] = result_df['cur_log_ret'] / result_df['pbeta']
    result_df['bret'] = result_df[['cur_log_ret', 'pbeta', 'mkt_cap_y', 'giclose_ts']].groupby(['giclose_ts'], sort=False).apply(wavg2).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['cur_log_ret'] - result_df['bret']
    result_df['rv_i'] = (result_df['dvolume'].astype(float) * result_df['dvwap']) / result_df['dpvolume_med_21']
    result_df['vadjC'] = result_df['rv_i'] * np.sign(result_df['badjret'])
    result_df['vadjC_B'] = winsorize_by_ts(result_df['vadjC'])

    print "Calulating vadjC_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['vadjC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['vadjC_B_ma'] = indgroups['vadjC_B']

    print "Calculated {} values".format(len(result_df['vadjC_B_ma'].dropna()))
    return result_df

def vadj_fits(daily_df, intra_df, horizon, name, middate=None):
    """Fit volume-adjusted position sizing model and generate forecasts.

    Fits both intraday and daily components:
    1. Intraday: 6 hourly coefficients for vadjC signal
    2. Daily: Multiple lag coefficients using incremental approach

    The full model combines:
      vadj_b = vadjC_B_ma * coef_hour +
               sum(vadj{lag}_B_ma * (coef_horizon - coef_lag))

    Includes diagnostic output:
    - Prints "Forecasts {name} Dist:" followed by describe() statistics
    - Helps monitor signal distribution and identify outliers

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
    fitresults_df = regress_alpha(insample_intra_df, 'vadjC_B_ma', horizon, True, 'intra')
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
    print fits_df.head(10)
    for ii in range(1,7):
        outsample_intra_df.ix[ coefs[ii], 'vadjC_B_ma_coef' ] = fits_df.ix['vadjC_B_ma'].ix[ii].ix['coef']
    
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'vadj0_B_ma', lag, True, 'daily') 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "vadj_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['vadj0_B_ma'].ix[horizon].ix['coef']
#    outsample_intra_df[ 'vadjC_B_ma_coef' ] = coef0
    print "Coef0: {}".format(coef0)
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['vadj0_B_ma'].ix[lag].ix['coef'] 
        print "Coef{}: {}".format(lag, coef)
        outsample_intra_df[ 'vadj'+str(lag)+'_B_ma_coef' ] = coef

    outsample_intra_df['vadj_b']   = outsample_intra_df['vadjC_B_ma'] * outsample_intra_df['vadjC_B_ma_coef']
    for lag in range(1,horizon):
        outsample_intra_df[ 'vadj_b'] += outsample_intra_df['vadj'+str(lag)+'_B_ma'] * outsample_intra_df['vadj'+str(lag)+'_B_ma_coef']

    print "Forecasts {} Dist:".format(name)
    print outsample_intra_df['vadj_b'].describe()
    
    return outsample_intra_df

def calc_vadj_forecast(daily_df, intra_df, horizon, middate):
    """Calculate volume-adjusted position sizing forecasts with sector models.

    Main pipeline for vadj_pos strategy:
    1. Calculate daily vadj signals with sign-based position sizing
    2. Calculate forward returns for regression
    3. Calculate intraday vadjC signals with sign-based position sizing
    4. Merge daily and intraday data
    5. Fit separate models for Energy sector and all other sectors
    6. Combine forecasts

    The strategy produces liquidity-aware position sizes:
    - Large positions in high-volume, clear-direction moves
    - Small positions in low-volume or unclear moves
    - Sector-neutral through industry demeaning
    - Multi-horizon through lag combination

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

    sector_name = 'Energy'
    print "Running vadj for sector {}".format(sector_name)
    sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    result1_df = vadj_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)

    print "Running vadj for sector {}".format(sector_name)
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
    parser.add_argument("--freq",action="store",dest="freq",default='30Min')
    parser.add_argument("--horizon",action="store",dest="horizon",default=3)
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
        print "Looking " + pname+"_daily.h5"
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print "Could not load cached data..."

    if not loaded:
        uni_df = get_uni(start, end, lookback)    
        BARRA_COLS = ['ind1', 'pbeta']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close', 'overnight_log_ret', 'tradable_volume', 'tradable_med_volume_21_y', 'tradable_med_volume_21']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        DBAR_COLS = ['close', 'dopen', 'dvolume', 'dvwap']
        intra_df = load_daybars(price_df[['ticker']], start, end, DBAR_COLS, freq)
        daily_df = merge_barra_data(price_df, barra_df)
        intra_df = merge_intra_data(daily_df, intra_df)
        intra_df = calc_vol_profiles(intra_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    outsample_df = calc_vadj_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(outsample_df, 'vadj_b')

