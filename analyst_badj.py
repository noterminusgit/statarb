#!/usr/bin/env python
"""
Beta-Adjusted Analyst Rating Strategy

This module implements an alpha signal that combines analyst rating levels
with beta-adjusted stock returns, filtering positions based on rating stability.

Strategy Logic:
    The strategy integrates fundamental analyst ratings with technical beta-adjusted returns:

    1. Load analyst rating history from IBES (rating_diff_mean field)
    2. Calculate beta-adjusted returns: log_ret / pbeta (predicted beta from Barra)
    3. Compute market beta-adjusted return using cap-weighted average
    4. Filter signals by rating stability (std_diff > 0 indicates coverage change)
    5. Combine rating levels with beta-adjusted returns: rtg0_B * rating
    6. Demean by industry (ind1) for market neutrality
    7. Fit multi-lag regression with decay weights

    The key insight: stocks with stable analyst coverage and strong rating
    signals (positive or negative) combined with beta-adjusted price moves
    tend to mean-revert.

Data Requirements:
    - Analyst rating history from ESTIMATES_BASE_DIR/ibes.db
        * rating_diff_mean: Average rating change (from load_ratings_hist)
        * Fields: mean, median, std, count from analyst consensus
    - Barra risk model data:
        * pbeta: Predicted beta for beta adjustment
        * ind1: Industry classification for demeaning
    - Daily price data:
        * log_ret: Daily log returns
        * mkt_cap_y: Market capitalization for weighting

Parameters:
    --start: Start date (YYYYMMDD format)
    --end: End date (YYYYMMDD format)
    --mid: Middate for in-sample/out-of-sample split
    --lag: Horizon for forward returns (default: 4 days, shorter than analyst.py)

Signal Calculation:
    rtg0_B = winsorize(log_ret / pbeta)          # Beta-adjusted return
    rating = -1 * rating_diff_mean                # Invert sign
    ret_rating = rtg0_B * rating                  # Interaction term
    rtg0_B_ma = rtg0_B - industry_mean(rtg0_B)   # Industry-neutral

    Multi-lag weighted signal:
    alpha = sum(w[i] * rtg[i]_B_ma * coef[i]) for i in [0, horizon)
    where w[i] = (horizon - i) / horizon          # Linear decay

Output:
    Writes HDF5 file with 'rtg' alpha column, consumable by bsim.py via:
        --fcast=rtg:1:1  (multiplier=1, weight=1)

Example:
    python analyst_badj.py --start=20130101 --end=20130630 --mid=20130315 --lag=4

Notes:
    - Shorter horizon (4 days) vs analyst.py (20 days) due to beta adjustment
    - Sector-specific handling shown in commented code (Energy separate)
    - Uses multi-lag regression with decay to capture persistence
    - rating_diff_mean is inverted (-1 *) so positive values indicate upgrades
    - Legacy Python 2.7 codebase
"""

from regress import *
from loaddata import *
from util import *

from pandas.stats.moments import ewma

def wavg(group):
    """
    Calculate market-cap weighted average beta-adjusted return.

    Computes the market return contribution to each stock's beta-adjusted return,
    used for debugging and understanding market-wide moves.

    Args:
        group: DataFrame group (typically grouped by date)
               Must contain: pbeta, log_ret, mkt_cap_y, gdate

    Returns:
        Series: Beta times market cap-weighted return (b * market_ret)

    Formula:
        market_return = sum(log_ret * weight) / sum(weight)
        where weight = mkt_cap_y / 1e6
        result = pbeta * market_return

    Notes:
        - Prints market return for each date for monitoring
        - Used to verify beta adjustment is removing market exposure
        - Divides mkt_cap by 1e6 to prevent numerical overflow
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print "Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum()))
    res = b * ((d * w).sum() / w.sum())
    return res

def calc_rtg_daily(daily_df, horizon):
    """
    Calculate beta-adjusted analyst rating signals with multi-lag structure.

    Combines beta-adjusted returns with analyst rating levels, creating
    a series of lagged signals for multi-horizon regression.

    Args:
        daily_df: DataFrame with price, Barra, and analyst data
                  Required columns: log_ret, pbeta, mkt_cap_y, rating_diff_mean,
                                   gdate, ind1
        horizon: int, number of lags to create (default: 4)

    Returns:
        DataFrame with additional columns:
            bret: Market beta-adjusted return (for monitoring)
            rtg0_B: Winsorized beta-adjusted return (log_ret / pbeta)
            rating: Inverted rating_diff_mean (-1 * rating_diff_mean)
            ret_rating: Interaction term (rtg0_B * rating)
            rtg0_B_ma: Industry-demeaned beta-adjusted return
            rtg{i}_B_ma: Lagged versions for i=1 to horizon

    Methodology:
        1. Filter to expandable universe (tradable stocks)
        2. Compute market beta-adjusted return via wavg() for monitoring
        3. Calculate individual stock beta-adjusted returns (log_ret / pbeta)
        4. Winsorize to limit outlier influence
        5. Extract rating signal from rating_diff_mean (inverted sign)
        6. Create interaction term (ret_rating) for sector analysis
        7. Demean by industry (ind1) for market neutrality
        8. Generate lagged versions for multi-horizon regression

    Notes:
        - Beta adjustment removes market exposure before applying ratings
        - Winsorization prevents extreme values from dominating
        - Industry demeaning ensures market-neutral positions
        - Commented code shows alternative formulations
        - ret_rating interaction enables sector-specific analysis (Energy vs others)
    """
    print "Caculating daily rtg..."
    result_df = filter_expandable(daily_df)

    print "Calculating rtg0..."    
#    result_df['cum_ret'] = pd.rolling_sum(result_df['log_ret'], 6)
#    result_df['med_diff'] = result_df['median'].unstack().diff().stack()
#    result_df['rtg0'] = -1.0 * (result_df['median'] - 3) / ( 1.0 + result_df['std'] )

    result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    result_df['rtg0_B'] = winsorize_by_date(result_df['log_ret'] / result_df['pbeta'])

    result_df['rating'] = -1 * result_df['rating_diff_mean'].fillna(0)
    result_df['ret_rating'] = result_df['rtg0_B'] * result_df['rating']
#    result_df['rtg0'] = -1.0 * result_df['med_diff_dk'] * result_df['cum_ret']

    # result_df['rtg0'] = -1.0 * result_df['med_diff_dk']
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['rtg0_B', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    result_df['rtg0_B_ma'] = indgroups['rtg0_B']

    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['rtg'+str(lag)+'_B_ma'] = shift_df['rtg0_B_ma']

    return result_df

def calc_rtg_intra(intra_df):
    """
    Calculate intraday beta-adjusted rating signals (experimental).

    Extends the beta-adjusted rating approach to intraday bars,
    combining overnight and intraday returns normalized by beta.

    Args:
        intra_df: DataFrame with intraday bar data
                  Required columns: overnight_log_ret, iclose, dopen, pbeta,
                                   giclose_ts, ind1

    Returns:
        DataFrame with columns:
            rtgC: Beta-adjusted cumulative return (overnight + intraday)
            rtgC_B: Winsorized rtgC
            rtgC_B_ma: Industry-demeaned rtgC_B

    Formula:
        rtgC = (overnight_ret + log(iclose/dopen)) / pbeta

    Notes:
        - Experimental feature, not used in main strategy
        - Combines overnight gap with intraday move
        - Industry demeaning by timestamp (giclose_ts) instead of date
        - Requires intraday bar data from BAR_BASE_DIR
    """
    print "Calculating rtg intra..."
    result_df = filter_expandable(intra_df)

    print "Calulating rtgC..."
    result_df['rtgC'] = (result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))) / result_df['pbeta']
    result_df['rtgC_B'] = winsorize_by_ts(result_df[ 'rtgC' ])

    print "Calulating rtgC_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['rtgC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=True).transform(demean)
    result_df['rtgC_B_ma'] = indgroups['rtgC_B']
    
    print "Calculated {} values".format(len(result_df['rtgC_B_ma'].dropna()))
    return result_df

def rtg_fits(daily_df, intra_df, horizon, name, middate=None, intercepts=None):
    """
    Fit multi-lag regression for beta-adjusted rating signals.

    Performs regression with multiple lags and decay weighting to capture
    signal persistence over several days.

    Args:
        daily_df: DataFrame with signals and forward returns
        intra_df: DataFrame with intraday signals (unused in current implementation)
        horizon: int, maximum lag for regression (default: 4)
        name: str, name suffix for plot files
        middate: datetime, in-sample/out-of-sample split date
        intercepts: dict, intercept adjustments by horizon (unused)

    Returns:
        DataFrame (out-of-sample) with columns:
            rtg0_B_ma_coef: Coefficient for current signal
            rtg{i}_B_ma_coef: Coefficients for lagged signals (i=1 to horizon-1)
            rtg: Final alpha = sum of weighted lagged signals

    Methodology:
        1. Split at middate into in-sample and out-of-sample
        2. Regress rtg0_B_ma against forward returns at each lag
        3. Plot regression diagnostics
        4. Extract coefficient at target horizon
        5. Compute incremental coefficients: coef[i] = coef[0] - coef[lag_i]
        6. Apply multi-lag weighting to out-of-sample data

    Weighting:
        - Base coefficient applied to current signal (lag 0)
        - Incremental coefficients for lags 1 to horizon-1
        - Each lag contributes: rtg[i]_B_ma * coef[i]

    Outputs:
        - Plot: rtg_daily_{name}_{dates}.png
        - Console: Coefficient values for each lag

    Notes:
        - Multi-lag structure captures signal decay/persistence
        - intra_df parameter exists but is unused (legacy)
        - Sector-specific fitting shown in commented code
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
#        insample_intra_df = intra_df[ intra_df['date'] <  middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_daily_df = daily_df[ daily_df.index.get_level_values('date') >= middate ]
#        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_daily_df['rtg'] = np.nan
    outsample_daily_df[ 'rtg0_B_ma_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_daily_df[ 'rtg' + str(lag) + '_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'rtg0_B_ma', lag, True, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "rtg_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    
    coef0 = fits_df.ix['rtg0_B_ma'].ix[horizon].ix['coef']
    outsample_daily_df[ 'rtg0_B_ma_coef' ] = coef0
    print "Coef0: {}".format(coef0)
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['rtg0_B_ma'].ix[lag].ix['coef'] 
        print "Coef{}: {}".format(lag, coef)
        outsample_daily_df[ 'rtg'+str(lag)+'_B_ma_coef' ] = coef

    outsample_daily_df['rtg'] = outsample_daily_df['rtg0_B_ma'] * outsample_daily_df['rtg0_B_ma_coef']
    for lag in range(1,horizon):
        outsample_daily_df[ 'rtg'] += outsample_daily_df['rtg'+str(lag)+'_B_ma'] * outsample_daily_df['rtg'+str(lag)+'_B_ma_coef']

    return outsample_daily_df


def calc_rtg_forecast(daily_df, horizon, middate):
    """
    Generate beta-adjusted analyst rating forecast.

    Main pipeline orchestrating signal calculation, regression, and forecast
    generation for the beta-adjusted analyst strategy.

    Args:
        daily_df: DataFrame with price, Barra, and analyst data merged
        horizon: int, forward return horizon (default: 4 days)
        middate: datetime, in-sample/out-of-sample split

    Returns:
        DataFrame with 'rtg' column containing final alpha predictions

    Pipeline:
        1. Calculate beta-adjusted signals via calc_rtg_daily()
        2. Calculate forward returns for regression targets
        3. Merge signals with forward returns
        4. Compute intercept adjustments via get_intercept()
        5. Filter by sector and rating interaction (commented code)
        6. Fit regression and generate predictions via rtg_fits()

    Sector Handling (in commented code):
        - Energy sector: treated separately due to different dynamics
        - Other sectors: combined regression
        - Splits by ret_rating sign: positive (upgrade), zero (neutral), negative (downgrade)

    Notes:
        - Current implementation uses single global regression (res5)
        - Sector-specific code is commented out but shows intended design
        - intercept_d provides baseline drift adjustment
        - Results concatenated and sorted by index
    """
    daily_results_df = calc_rtg_daily(daily_df, horizon) 
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

    # results = list()
    # for sector_name in daily_results_df['sector_name'].dropna().unique():
    #     if sector_name == "Utilities" or sector_name == "HealthCare": continue
    #     print "Running rtg for sector {}".format(sector_name)
    #     sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    #     result_df = rtg_fits(sector_df, horizon, sector_name, middate)
    #     results.append(result_df)
    # result_df = pd.concat(results, verify_integrity=True)

    # result_df = rtg_fits(daily_results_df, horizon, "", middate)
    intercept_d = get_intercept(daily_results_df, horizon, 'rtg0_B_ma', middate)
    sector_name = 'Energy'
    print "Running qhl for sector {}".format(sector_name)
    sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
 #   sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    # res1 = rtg_fits( sector_df[ sector_df['ret_rating'] > 0 ], horizon, "en_up", middate)
#    res2 = rtg_fits( sector_df[ sector_df['ret_rating'] == 0 ], None, horizon, "en_eq", middate, intercept_d)
    # res3 = rtg_fits( sector_df[ sector_df['ret_rating'] < 0 ], horizon, "en_dn", middate)

    
    print "Running qhl for not sector {}".format(sector_name)
    sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
#    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] != sector_name ]    
#    res4 = rtg_fits( sector_df[ sector_df['ret_rating'] > 0 ], horizon, "ot_up", middate)
    res5 = rtg_fits( sector_df[ sector_df['ret_rating'] == 0 ], None, horizon, "ot_eq", middate, intercept_d)
#    res6 = rtg_fits( sector_df[ sector_df['ret_rating'] < 0 ], horizon, "ot_dn", middate)

#    result_df = pd.concat([res1, res2, res3, res4, res5, res6], verify_integrity=True).sort()
    result_df = pd.concat([res5], verify_integrity=True).sort()
    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--lag",action="store",dest="lag",default=4)
#    parser.add_argument("--horizon",action="store",dest="horizon",default=20)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = int(args.lag)
    pname = "./rtg" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)
    lag = int(args.lag)

    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        loaded = True
    except:
        print "Did not load cached data..."

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1', 'pbeta']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)

        daily_df = merge_barra_data(price_df, barra_df)
        analyst_df = load_ratings_hist(price_df[['ticker']], start, end, False, True)
        daily_df = merge_daily_calcs(daily_df, analyst_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')

    result_df = calc_rtg_forecast(daily_df, horizon, middate)

    print "Total Alpha Summary"
    print result_df['rtg'].describe()

    dump_daily_alpha(result_df, 'rtg')









