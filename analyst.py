#!/usr/bin/env python
"""
Analyst Rating Mean Reversion Strategy

This module implements a contrarian alpha signal based on analyst rating changes,
exploiting mean reversion patterns when consensus rating shifts occur.

Strategy Logic:
    The strategy generates signals from analyst rating consensus changes over time:

    1. Load analyst rating history from IBES database (ESTIMATES_BASE_DIR/ibes.db)
    2. Compute rating change metrics (rating_diff_mean from load_ratings_hist)
    3. Apply exponentially-weighted moving average (EWMA) to detect trends
    4. Square the signal to amplify large rating shifts
    5. Demean by industry (ind1) to create market-neutral positions
    6. Regress against forward returns to estimate predictive coefficients

    Signal = (EWMA(rating_diff) ^ 2) - industry_mean

Data Requirements:
    - Analyst rating history from ESTIMATES_BASE_DIR/ibes.db
    - Rating scale: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
    - Data fields from load_ratings_hist():
        * rating_mean, rating_median, rating_std: consensus statistics
        * rating_count: number of covering analysts
        * rating_diff_mean: average rating change (upgrades/downgrades)
    - Barra industry classifications (ind1) for demeaning
    - Daily price data for forward returns calculation

Parameters:
    --start: Start date (YYYYMMDD format)
    --end: End date (YYYYMMDD format)
    --mid: Middate for in-sample/out-of-sample split
    --lag: Horizon for forward returns and regression (default: 20 days)

Signal Interpretation:
    - Positive rating_diff_mean: consensus upgrade (ratings decreased numerically)
    - Negative rating_diff_mean: consensus downgrade (ratings increased numerically)
    - Strategy squares the signal, so both upgrades and downgrades generate positive alpha
    - The sign of predictiveness is determined by regression coefficients

Output:
    Writes HDF5 file with 'rtg' alpha column, consumable by bsim.py via:
        --fcast=rtg:1:1  (multiplier=1, weight=1)

Example:
    python analyst.py --start=20130101 --end=20130630 --mid=20130315 --lag=20

Notes:
    - Uses exponential weighting with halflife = horizon/2
    - Industry demeaning provides market-neutral exposure
    - Commented code shows alternative approaches (median-based, cum_ret interaction)
    - Legacy Python 2.7 codebase
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def calc_rtg_daily(daily_df, horizon):
    """
    Calculate daily analyst rating change signals.

    Computes the core alpha signal from analyst rating consensus changes,
    applying exponential decay and industry demeaning.

    Args:
        daily_df: DataFrame with analyst rating history merged with price data
                  Must contain columns: mean, count, log_ret, ind1, gdate
        horizon: int, number of days for forward returns and EWMA halflife

    Returns:
        DataFrame with additional columns:
            cum_ret: Rolling sum of log returns over horizon
            det_diff: Daily change in sum of ratings (mean * count)
            det_diff_dk: EWMA of rating change with halflife=horizon
            rtg0: Squared EWMA signal (det_diff_dk^2)
            rtg0_ma: Industry-demeaned rtg0 signal
            rtg1_ma: Lagged rtg0_ma for multi-day regression

    Methodology:
        1. Filter to expandable universe (tradable stocks)
        2. Compute cumulative returns over horizon
        3. Calculate daily change in aggregate ratings (mean * count)
        4. Apply EWMA to smooth rating changes with halflife
        5. Square the smoothed signal to amplify large shifts
        6. Demean by industry (ind1) to create market-neutral positions
        7. Create lagged version for regression

    Notes:
        - Commented alternatives show median-based and sign-interaction approaches
        - det_diff captures both changes in ratings and analyst coverage
        - Squaring makes both upgrades and downgrades generate positive signal
    """
    print("Caculating daily rtg...")
    result_df = filter_expandable(daily_df)

    print("Calculating rtg0..."    )
    halflife = horizon / 2
#    result_df['dk'] = np.exp( -1.0 * halflife *  (result_df['gdate'] - result_df['last']).astype('timedelta64[D]').astype(int) )

    result_df['cum_ret'] = result_df['log_ret'].rolling(horizon).sum()

    result_df['sum'] = result_df['mean'] * result_df['count']
    result_df['det_diff'] = result_df['sum'].diff()
    result_df['det_diff_dk'] = result_df['det_diff'].ewm(halflife=horizon, adjust=False).mean()
    result_df['rtg0'] = result_df['det_diff_dk'] * result_df['det_diff_dk']

    # result_df['median'] = -1.0 * (result_df['median'] - 3)
    # result_df['med_diff'] = result_df['median'].unstack().diff().stack()
    # result_df['med_diff_dk'] = pd.rolling_sum( result_df['dk'] * result_df['med_diff'], window=horizon )
    # result_df['rtg0'] = (np.sign(result_df['med_diff_dk']) * np.sign(result_df['cum_ret'])).clip(lower=0) * result_df['med_diff_dk']


    demean = lambda x: (x - x.mean())
    indgroups = result_df[['rtg0', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    result_df['rtg0_ma'] = indgroups['rtg0']

#    result_df['rtg0_ma'] = result_df['rtg0_ma'] * (np.sign(result_df['rtg0_ma']) * np.sign(result_df['cum_ret']))

#    result_df['rtg0_ma'] = result_df['rtg0']

    shift_df = result_df.unstack().shift(1).stack()
    result_df['rtg1_ma'] = shift_df['rtg0_ma']

    return result_df

def rtg_fits(daily_df, horizon, name, middate=None):
    """
    Fit regression coefficients for analyst rating signals.

    Performs in-sample regression of rating signals against forward returns,
    then applies learned coefficients to out-of-sample data.

    Args:
        daily_df: DataFrame with rating signals and forward returns
        horizon: int, forward return horizon in days
        name: str, name suffix for plot output files
        middate: datetime, split date between in-sample and out-of-sample
                 If None, entire dataset is in-sample

    Returns:
        DataFrame (out-of-sample portion) with columns:
            rtg0_ma_coef: Regression coefficient for rtg0_ma signal
            rtg: Final alpha prediction (rtg0_ma * coef)

    Methodology:
        1. Split data at middate into in-sample and out-of-sample
        2. Run regress_alpha() for each horizon from 1 to horizon days
        3. Plot fit diagnostics (coefficients, t-stats, stderr vs horizon)
        4. Extract coefficient at target horizon
        5. Apply coefficient to out-of-sample data to generate predictions

    Outputs:
        - Plot: rtg_daily_{name}_{dates}.png showing regression diagnostics
        - Console: Coefficient value for horizon

    Notes:
        - Uses regress_alpha() from regress.py for WLS regression
        - Coefficient should be positive if rating changes predict returns
        - Out-of-sample predictions prevent look-ahead bias
    """
    insample_daily_df = daily_df
    if middate is not None:
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_daily_df = daily_df[ daily_df.index.get_level_values('date') >= middate ]

    outsample_daily_df['rtg'] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'rtg0_ma', ii, False, 'daily') 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "rtg_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.loc['rtg0_ma'].loc[horizon].loc['coef']
    print("Coef{}: {}".format(0, coef0))
    outsample_daily_df[ 'rtg0_ma_coef' ] = coef0

    outsample_daily_df[ 'rtg' ] = outsample_daily_df['rtg0_ma'] * outsample_daily_df['rtg0_ma_coef']
    
    return outsample_daily_df

def calc_rtg_forecast(daily_df, horizon, middate):
    """
    Generate analyst rating forecast by combining signals and regression.

    Main pipeline function that orchestrates signal calculation, regression,
    and forecast generation for the analyst rating strategy.

    Args:
        daily_df: DataFrame with price data and analyst ratings merged
        horizon: int, forward return horizon in days
        middate: datetime, split between in-sample fit and out-of-sample forecast

    Returns:
        DataFrame with 'rtg' column containing final alpha predictions

    Pipeline:
        1. Calculate rating signals via calc_rtg_daily()
        2. Calculate forward returns for regression targets
        3. Merge signals with forward returns
        4. Fit regression and generate predictions via rtg_fits()

    Notes:
        - Commented code shows sector-specific and directional splits
        - Current implementation uses single global regression
        - Could be extended to separate upgrades/downgrades (see comments)
    """
    daily_results_df = calc_rtg_daily(daily_df, horizon) 
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

    # results = list()
    # for sector_name in daily_results_df['sector_name'].dropna().unique():
    #     print "Running rtg for sector {}".format(sector_name)
    #     sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    #     result_df = rtg_fits(sector_df, horizon, sector_name, middate)
    #     results.append(result_df)
    # result_df = pd.concat(results, verify_integrity=True)

    result_df = rtg_fits(daily_results_df, horizon, "", middate)

    # res1 = rtg_fits( daily_results_df[ daily_results_df['med_diff_dk'] > 0 ], horizon, "up", middate)
    # res2 = rtg_fits( daily_results_df[ daily_results_df['med_diff_dk'] < 0 ], horizon, "dn", middate)
    # result_df = pd.concat([res1, res2], verify_integrity=True)

    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--lag",action="store",dest="lag",default=20)
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
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)

        daily_df = merge_barra_data(price_df, barra_df)
        analyst_df = load_ratings_hist(price_df[['ticker']], start, end)
        daily_df = merge_daily_calcs(analyst_df, daily_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')

    result_df = calc_rtg_forecast(daily_df, horizon, middate)
    dump_daily_alpha(result_df, 'rtg')









