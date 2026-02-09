#!/usr/bin/env python
"""
Analyst Rating Change Strategy with Coverage Filtering

This module implements an alpha signal based on analyst rating changes,
with filtering to only trade stocks where analyst coverage is expanding.

Strategy Logic:
    The strategy focuses on rating changes when coverage is increasing,
    under the hypothesis that expanding coverage signals institutional interest
    and rating changes during this period are more informative:

    1. Load analyst rating history from IBES database (rating_diff_mean)
    2. Calculate rating standard deviation changes (std_diff)
    3. Filter: only use rating_diff_mean when std_diff > 0 (coverage expanding)
    4. Set rating_diff_mean = 0 when coverage is stable or contracting
    5. Apply multi-lag regression with linear decay weighting
    6. Generate predictions for filtered signals

    Key Innovation: Coverage filter (std_diff > 0) prevents stale signals
    from stocks with stable/declining analyst interest.

Data Requirements:
    - Analyst rating history from ESTIMATES_BASE_DIR/ibes.db
        * rating_diff_mean: Average rating change across analysts
        * rating_std: Standard deviation of ratings (measures dispersion)
    - Barra industry classifications (ind1)
    - Daily price data for returns

Signal Filtering Logic:
    The critical filtering step (lines 28-32):
        std_diff = rating_std[t] - rating_std[t-1]
        if std_diff <= 0 or std_diff is null:
            rating_diff_mean = 0  # Ignore signal

    This ensures we only trade when:
    - Coverage is expanding (std_diff > 0)
    - Analyst opinions are diverging (higher dispersion)

Parameters:
    --start: Start date (YYYYMMDD format)
    --end: End date (YYYYMMDD format)
    --mid: Middate for in-sample/out-of-sample split
    --lag: Horizon for forward returns (default: 6 days)

Signal Interpretation:
    - rating_diff_mean > 0: Consensus rating improved (upgrade)
    - rating_diff_mean < 0: Consensus rating deteriorated (downgrade)
    - After filtering: only upgrades/downgrades with expanding coverage remain
    - Linear decay weights give more importance to recent signals

Output:
    Writes HDF5 file with 'rtg' alpha column, consumable by bsim.py via:
        --fcast=rtg:1:1  (multiplier=1, weight=1)

Example:
    python rating_diff.py --start=20130101 --end=20130630 --mid=20130315 --lag=6

Notes:
    - Medium horizon (6 days) balances signal decay with predictive power
    - Coverage filter is the key differentiator from analyst.py
    - Current implementation uses global regression (no industry demeaning)
    - Commented code shows alternative sector-specific approaches
    - Legacy Python 2.7 codebase
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def wavg(group):
    """
    Calculate market-cap weighted average beta-adjusted return.

    Used for monitoring market-wide moves and understanding beta contribution.

    Args:
        group: DataFrame group (typically by date)
               Required: pbeta, log_ret, mkt_cap_y, gdate

    Returns:
        Series: Beta times market cap-weighted return

    Notes:
        - Prints market return for debugging
        - Not used in signal calculation, only for diagnostics
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print("Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum())))
    res = b * ((d * w).sum() / w.sum())
    return res

def calc_rtg_daily(daily_df, horizon):
    """
    Calculate filtered analyst rating change signals with multi-lag structure.

    Applies coverage expansion filter to rating changes, creating signals
    only when analyst coverage is increasing (std_diff > 0).

    Args:
        daily_df: DataFrame with price and analyst data
                  Required columns: rating_diff_mean, rating_std, ind1, gdate
        horizon: int, number of lags to create (default: 6)

    Returns:
        DataFrame with additional columns:
            std_diff: Change in rating standard deviation (std[t] - std[t-1])
            rtg0: Filtered rating_diff_mean (zeroed when std_diff <= 0)
            rtg0_ma: Raw signal (no demeaning in current implementation)
            rtg{i}_ma: Lagged signals for i=1 to horizon

    Filtering Logic (critical section):
        1. Calculate std_diff = rating_std.diff() for each stock
        2. If std_diff <= 0 or null: set rating_diff_mean = 0
        3. Else: use rating_diff_mean as-is

    Interpretation:
        - std_diff > 0: Coverage expanding, analyst opinions diverging
        - std_diff <= 0: Coverage stable/contracting, ignore signals
        - null std_diff: First observation, no prior data, ignore

    Debug Output:
        Prints "SEAN" sections showing rating_diff_mean statistics
        before and after filtering to verify filter effectiveness.

    Notes:
        - Lines 28-32 implement the critical coverage filter
        - rtg0_ma currently equals rtg0 (no industry demeaning)
        - Commented code shows alternative industry demeaning approach
        - Multi-lag structure captures signal persistence
        - Debug code at line 34 shows example stock (sid=10000708)
    """
    print("Caculating daily rtg...")
    result_df = filter_expandable(daily_df)

    print("Calculating rtg0..."    )
#    result_df['cum_ret'] = pd.rolling_sum(result_df['log_ret'], 6)
#    result_df['med_diff'] = result_df['median'].unstack().diff().stack()
#    result_df['rtg0'] = -1.0 * (result_df['median'] - 3) / ( 1.0 + result_df['std'] )
#    result_df['rtg0'] = -1 * result_df['mean'] * np.abs(result_df['mean'])
#    result_df['rtg0'] = -1.0 * result_df['med_diff_dk'] * result_df['cum_ret']

    result_df['std_diff'] = result_df['rating_std'].unstack().diff().stack()
    print("SEAN")
    print(result_df['rating_diff_mean'].describe())
    result_df.loc[ (result_df['std_diff'] <= 0) | (result_df['std_diff'].isnull()), 'rating_diff_mean'] = 0
    print(result_df['rating_diff_mean'].describe())
    print("SEAN2")
    print(result_df.xs(10000708, level=1))
    result_df['rtg0'] = result_df['rating_diff_mean'] #* result_df['rating_diff_mean'] * np.sign(result_df['rating_diff_mean'])


    # result_df['rtg0'] = -1.0 * result_df['med_diff_dk']
    # demean = lambda x: (x - x.mean())
    # indgroups = result_df[['rtg0', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    # result_df['rtg0_ma'] = indgroups['rtg0']
    result_df['rtg0_ma'] = result_df['rtg0']

    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['rtg'+str(lag)+'_ma'] = shift_df['rtg0_ma']

    return result_df

def rtg_fits(daily_df, horizon, name, middate=None):
    """
    Fit multi-lag regression with linear decay weighting.

    Performs regression of filtered rating signals against forward returns,
    applying linear decay weights to lagged signals.

    Args:
        daily_df: DataFrame with signals and forward returns
        horizon: int, maximum lag for regression (default: 6)
        name: str, name suffix for plot output
        middate: datetime, in-sample/out-of-sample split date

    Returns:
        DataFrame (out-of-sample) with columns:
            rtg0_ma_coef: Base coefficient for current signal
            rtg: Final weighted alpha prediction

    Methodology:
        1. Split at middate into in-sample and out-of-sample
        2. Regress rtg0_ma against forward returns at each lag
        3. Plot diagnostics (uses plot_fit from regress.py)
        4. Extract coefficient at target horizon
        5. Apply linear decay weighting to lagged signals

    Weighting Formula:
        For lag i in [1, horizon-1]:
            weight[i] = (horizon - i) / horizon
            alpha += rtg[i]_ma * coef0 * weight[i]

        Current signal (lag 0) uses full coefficient:
            alpha = rtg0_ma * coef0

    Example weights for horizon=6:
        lag 0: weight = 1.0 (implicit)
        lag 1: weight = 5/6 = 0.833
        lag 2: weight = 4/6 = 0.667
        lag 3: weight = 3/6 = 0.500
        lag 4: weight = 2/6 = 0.333
        lag 5: weight = 1/6 = 0.167

    Outputs:
        - Plot: rtg_daily_{name}_{dates}.png
        - Console: Coefficient and alpha summary statistics

    Notes:
        - Linear decay assumes signal decays uniformly over time
        - Uses regress_alpha() with neutralize=True for market neutrality
        - fillna(0) ensures missing signals don't propagate NaNs
        - Prints alpha summary for out-of-sample validation
    """
    insample_daily_df = daily_df
    if middate is not None:
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_daily_df = daily_df[ daily_df.index.get_level_values('date') >= middate ]

    outsample_daily_df['rtg'] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'rtg0_ma', ii, True, 'daily', False) 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "rtg_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['rtg0_ma'].ix[horizon].ix['coef']
    print("Coef{}: {}".format(0, coef0))
    outsample_daily_df[ 'rtg0_ma_coef' ] = coef0

    outsample_daily_df[ 'rtg' ] = outsample_daily_df['rtg0_ma'].fillna(0) * outsample_daily_df['rtg0_ma_coef']
    for lag in range(1,horizon):
        weight = (horizon - lag) / float(horizon)
        lagname = 'rtg'+str(lag)+'_ma'
        print("Running lag {} with weight: {}".format(lag, weight))
        outsample_daily_df[ 'rtg'] += outsample_daily_df[lagname].fillna(0) * outsample_daily_df['rtg0_ma_coef'] * weight

    print("Alpha Summary {}".format(name))
    print(outsample_daily_df['rtg'].describe())
    
    return outsample_daily_df

def calc_rtg_forecast(daily_df, horizon, middate):
    """
    Generate analyst rating change forecast with coverage filtering.

    Main pipeline orchestrating filtered signal calculation, regression,
    and forecast generation.

    Args:
        daily_df: DataFrame with price and analyst data merged
        horizon: int, forward return horizon (default: 6 days)
        middate: datetime, in-sample/out-of-sample split

    Returns:
        DataFrame with 'rtg' column containing final alpha predictions

    Pipeline:
        1. Calculate filtered rating signals via calc_rtg_daily()
        2. Calculate forward returns for regression targets
        3. Merge signals with forward returns
        4. Fit regression and generate predictions via rtg_fits()

    Alternative Approaches (commented):
        - Sector-specific regressions by sector_name
        - Separate regressions for upgrades vs downgrades
        - Currently uses single global regression

    Notes:
        - Coverage filter is applied in calc_rtg_daily()
        - Commented code shows upgrade/downgrade split:
            * rating_diff_mean > 0: upgrades
            * rating_diff_mean < 0: downgrades
        - Could be extended to sector or direction-specific models
        - Final result includes only filtered, valid signals
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

    result_df = rtg_fits(daily_results_df, horizon, "", middate)

    # res1 = rtg_fits( daily_results_df[ daily_results_df['rating_diff_mean'] > 0 ], horizon, "up", middate)
    # res2 = rtg_fits( daily_results_df[ daily_results_df['rating_diff_mean'] < 0 ], horizon, "dn", middate)
    # result_df = pd.concat([res1, res2], verify_integrity=True)

    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--lag",action="store",dest="lag",default=6)
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

    print("Total Alpha Summary")
    print(result_df['rtg'].describe())

    dump_daily_alpha(result_df, 'rtg')









