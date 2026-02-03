#!/usr/bin/env python
"""
Close-to-Open (C2O) Gap Trading Alpha Strategy

This module implements a mean reversion strategy based on overnight gap returns.
The strategy exploits the tendency for large overnight gaps to partially reverse
during the subsequent trading day.

Strategy Logic:
    1. Calculate overnight return: log(open/previous_close)
    2. Beta-adjust overnight returns to remove market component
    3. Filter out small gaps (< 2% absolute)
    4. Demean within industries for market-neutral positioning
    5. Combine current gap with lagged gaps for multi-day signal
    6. Separate regressions for intraday and daily components

The strategy runs sector-by-sector regressions to capture sector-specific
gap reversal dynamics. Intraday signals vary by time-of-day (6 hourly buckets),
reflecting stronger mean reversion early in the trading session.

Data Requirements:
    - Daily: close prices, overnight returns, market cap, beta, industry codes
    - Intraday: 15-min or 30-min bars with open and close prices
    - Universe: Expandable stocks only (sufficient liquidity)

Parameters:
    horizon: Number of days to include lagged gap signals (default 1)
    middate: Split date between in-sample fitting and out-of-sample prediction
    freq: Intraday bar frequency (default '15Min')

Output:
    'c2o' column with predicted forward returns based on gap signals

Academic Basis:
    Related to overnight return anomaly and gap fade strategies commonly
    used by intraday market makers and statistical arbitrage funds.

Usage:
    python c2o.py --start=20130101 --end=20130630 --mid=20130401 --horizon=1
"""

from regress import *
from loaddata import *
from util import *

def wavg(group):
    """
    Calculate market-cap weighted average return multiplied by beta.

    Used to compute the market component of overnight returns that should
    be removed to isolate stock-specific gap signals.

    Args:
        group: DataFrame group with columns pbeta, overnight_log_ret, mkt_cap_y

    Returns:
        Beta-weighted market return for the group
    """
    b = group['pbeta']
    d = group['overnight_log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg2(group):
    """
    Calculate market-cap weighted overnight return multiplied by beta (intraday version).

    Identical to wavg() but used in intraday context for consistency.

    Args:
        group: DataFrame group with columns pbeta, overnight_log_ret, mkt_cap_y

    Returns:
        Beta-weighted market return for the group
    """
    b = group['pbeta']
    d = group['overnight_log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg3(group):
    """
    Calculate market-cap weighted current return multiplied by beta.

    Used for intraday current-to-open return calculations.

    Args:
        group: DataFrame group with columns pbeta, cur_log_ret, mkt_cap_y

    Returns:
        Beta-weighted market return for the current period
    """
    b = group['pbeta']
    d = group['cur_log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res


def calc_c2o_daily(daily_df, horizon):
    """
    Calculate daily close-to-open gap signals with beta adjustment.

    Formula:
        1. bret = cap_weighted_avg(overnight_ret * beta) by date
        2. badjret = overnight_ret - bret  (beta-adjusted gap)
        3. Filter: Set gaps < 2% absolute to zero
        4. Winsorize by date
        5. Demean within (date, industry) groups for market neutrality
        6. Create lagged features for horizons 1 to horizon

    Args:
        daily_df: DataFrame with columns overnight_log_ret, pbeta, mkt_cap_y, ind1
        horizon: Number of lag days to create (default 1)

    Returns:
        DataFrame with c2o0_B_ma and c2o{lag}_B_ma columns for each lag
    """
    print "Caculating daily c2o..."
    result_df = filter_expandable(daily_df)

    print "Calculating c2o0..."
#    result_df['c2o0'] = result_df['overnight_log_ret'] / result_df['pbeta']
    result_df['bret'] = result_df[['overnight_log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['overnight_log_ret'] - result_df['bret']

   # result_df['c2o0_B'] = result_df['log_ret'] * (1 + np.abs(result_df['badjret'])) ** 3
    result_df['c2o0'] = result_df['badjret']
    result_df.ix[ np.abs(result_df['c2o0']) < .02 , 'c2o0'] = 0
    result_df['c2o0_B'] = winsorize_by_date(result_df['c2o0'])

    result_df = result_df.dropna(subset=['c2o0_B'])

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['c2o0_B', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=False).transform(demean)
    result_df['c2o0_B_ma'] = indgroups['c2o0_B']

    print "Calulating lags..."
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['c2o' + str(lag) + '_B_ma'] = shift_df['c2o0_B_ma']

    return result_df

def calc_c2o_intra(intra_df):
    """
    Calculate intraday close-to-open gap signals with beta adjustment.

    Uses the overnight gap return (prior close to today's open) as the
    primary signal, with beta adjustment to isolate stock-specific component.

    Formula:
        1. cur_log_ret = log(current_close / today_open)
        2. bretC = cap_weighted_avg(cur_log_ret * beta) by timestamp
        3. bret = cap_weighted_avg(overnight_ret * beta) by timestamp
        4. badjret = overnight_ret - bret (beta-adjusted gap)
        5. Filter: Set gaps < 2% absolute to zero
        6. Winsorize by timestamp
        7. Demean within (timestamp, industry) groups

    Args:
        intra_df: Intraday DataFrame with columns iclose, dopen, overnight_log_ret,
                  pbeta, mkt_cap_y, ind1

    Returns:
        DataFrame with c2oC_B_ma column containing intraday gap signal
    """
    print "Calculating c2o intra..."
    result_df = filter_expandable(intra_df)

    print "Calulating c2oC..."
    result_df['cur_log_ret'] = np.log(result_df['iclose']/result_df['dopen'])
    result_df['bretC'] = result_df[['cur_log_ret', 'pbeta', 'mkt_cap_y', 'giclose_ts']].groupby(['giclose_ts'], sort=False).apply(wavg3).reset_index(level=0)['pbeta']
    result_df['badjretC'] = result_df['cur_log_ret'] - result_df['bretC']

    result_df['bret'] = result_df[['overnight_log_ret', 'pbeta', 'mkt_cap_y', 'giclose_ts']].groupby(['giclose_ts'], sort=False).apply(wavg2).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['overnight_log_ret'] - result_df['bret']

#    result_df['c2oC_B'] = result_df['badjretC'] * (1 + np.abs(result_df['badjret'])) ** 3
    result_df['c2oC'] = result_df['badjret']
    result_df.ix[ np.abs(result_df['c2oC']) < .02 , 'c2oC'] = 0
    result_df['c2oC_B'] = winsorize_by_ts(result_df['c2oC'])
    result_df = result_df.dropna(subset=['c2oC_B'])

    print "Calulating c2oC_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['c2oC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['c2oC_B_ma'] = indgroups['c2oC_B']

    return result_df

def c2o_fits(daily_df, intra_df, horizon, name, middate):
    """
    Fit c2o alpha model and generate forecasts.

    Performs two separate regressions:
    1. Intraday regression: Regresses c2oC_B_ma against forward returns for
       6 different time-of-day buckets (09:30-10:30, 10:30-11:30, etc.)
    2. Daily regression: Regresses lagged c2o0_B_ma signals against forward returns

    The intraday signal varies by time of day, capturing the intraday pattern
    of gap mean reversion. Daily lagged signals capture multi-day reversal.

    Final forecast combines:
        c2o = c2oC_B_ma * coef[time_bucket] + sum(c2o{lag}_B_ma * coef{lag})

    Args:
        daily_df: Daily DataFrame with c2o signals and forward returns
        intra_df: Intraday DataFrame with c2oC signals
        horizon: Maximum lag to include in daily regressions
        name: String identifier for plot filenames
        middate: Split date between in-sample and out-of-sample

    Returns:
        DataFrame with 'c2o' forecast column
    """
    # daily_df['dow'] = daily_df['gdate'].apply(lambda x: x.weekday())
    # daily_df['dow'] = daily_df['dow'].clip(0,1)
    # intra_df['dow'] = intra_df['date'].apply(lambda x: x.weekday())
    # intra_df['dow'] = intra_df['dow'].clip(0,1)
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['c2o'] = 0
    outsample_intra_df[ 'c2oC_B_ma_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_intra_df[ 'c2o' + str(lag) + '_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])

    fitresults_df = regress_alpha(insample_intra_df, 'c2oC_B_ma', horizon, True, 'intra_eod')
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "c2o_intra_"+name+"_" + df_dates(insample_intra_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    
    unstacked = outsample_intra_df[ ['ticker'] ].unstack()
    coefs = dict()
    coefs[1] = unstacked.between_time('09:30', '10:31').stack().index
    coefs[2] = unstacked.between_time('10:30', '11:31').stack().index
    coefs[3] = unstacked.between_time('11:30', '12:31').stack().index
    coefs[4] = unstacked.between_time('12:30', '13:31').stack().index
    coefs[5] = unstacked.between_time('13:30', '14:31').stack().index
    coefs[6] = unstacked.between_time('14:30', '16:01').stack().index
    unstacked = None

    for ii in range(1,7):
        outsample_intra_df.ix[ coefs[ii], 'c2oC_B_ma_coef' ] = fits_df.ix['c2oC_B_ma'].ix[ii].ix['coef']

    #DAILY...
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        print insample_daily_df.head()
        fitresults_df = regress_alpha(insample_daily_df, 'c2o0_B_ma', lag, True, 'daily') 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "c2o_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    # for dow in range(0,2):       
    #     coef0 = fits_df.ix['c2o0_B_ma'].ix[horizon * 10 + dow].ix['coef']
    #     for lag in range(1,horizon):
    #         coef = coef0 - fits_df.ix['c2o0_B_ma'].ix[lag * 10 + dow].ix['coef'] 
    #         print "Coef{}: {}".format(lag, coef)
    #         dowidx = outsample_intra_df[ outsample_intra_df['dow'] == dow ].index
    #         outsample_intra_df.ix[ dowidx, 'c2o'+str(lag)+'_B_ma_coef' ] = coef

    coef0 = fits_df.ix['c2o0_B_ma'].ix[horizon].ix['coef']
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['c2o0_B_ma'].ix[lag].ix['coef'] 
        print "Coef{}: {}".format(lag, coef)
        outsample_intra_df[ 'c2o'+str(lag)+'_B_ma_coef' ] = coef

    outsample_intra_df[ 'c2o'] = outsample_intra_df['c2oC_B_ma'] * outsample_intra_df['c2oC_B_ma_coef']
    for lag in range(1,horizon):
        outsample_intra_df[ 'c2o'] += outsample_intra_df['c2o'+str(lag)+'_B_ma'] * outsample_intra_df['c2o'+str(lag)+'_B_ma_coef']

    return outsample_intra_df

def calc_c2o_forecast(daily_df, intra_df, horizon, middate):
    """
    Master function to calculate c2o forecasts with sector-specific regressions.

    Runs the full pipeline:
    1. Calculate daily c2o signals
    2. Calculate forward returns for regression fitting
    3. Calculate intraday c2o signals
    4. Merge daily and intraday data
    5. Run sector-by-sector regressions
    6. Combine results across all sectors

    Running separate regressions by sector captures sector-specific patterns
    in gap reversal dynamics (e.g., Tech gaps may behave differently than Energy).

    Args:
        daily_df: Daily price and factor data
        intra_df: Intraday bar data
        horizon: Number of lag days for daily signals
        middate: Split date between in-sample and out-of-sample

    Returns:
        Intraday DataFrame with 'c2o' forecast column
    """
    daily_results_df = calc_c2o_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    intra_results_df = calc_c2o_intra(intra_df)
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)

    #    sector_name = 'Energy'
    #    print "Running c2o for sector {}".format(sector_name)
    #    sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    #    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]

    results = list()
    for sector_name in daily_results_df['sector_name'].dropna().unique():
        print "Running c2o for sector {}".format(sector_name)
        sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
        sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
        result_df = c2o_fits(sector_df, sector_intra_results_df, horizon, sector_name, middate)
        results.append(result_df)

    # sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
    # sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] != sector_name ]
    # result2_df = c2o_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)

    result_df = pd.concat(results, verify_integrity=True)
    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--horizon",action="store",dest="horizon",default=1)
    parser.add_argument("--freq",action="store",dest="freq",default='15Min')
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    freq = args.freq
    horizon = int(args.horizon)
    pname = "./c2o" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)

    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print "Did not load cached data..."

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1', 'pbeta']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close', 'overnight_log_ret', 'tradable_volume', 'tradable_med_volume_21']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        daily_df = merge_barra_data(price_df, barra_df)
        DBAR_COLS = ['close', 'dopen', 'dvolume']
        daybar_df = load_daybars(price_df[ ['ticker'] ], start, end, DBAR_COLS, freq)
        intra_df = merge_intra_data(daily_df, daybar_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    outsample_df = calc_c2o_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(outsample_df, 'c2o')
#    dump_all(outsample_df)
