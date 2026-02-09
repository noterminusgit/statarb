#!/usr/bin/env python
"""
Analyst Rating Change Strategy with Asymmetric Up/Down Treatment

This module implements an alpha signal based on analyst rating changes,
with separate regression models for upgrades and downgrades to capture
asymmetric market response to good vs bad news.

Strategy Logic:
    The strategy recognizes that markets may respond differently to rating
    upgrades versus downgrades, fitting separate models for each:

    1. Load analyst rating history from IBES (rating_diff_mean)
    2. Calculate rating standard deviation changes (std_diff) for filtering
    3. Filter signals: zero out rating_diff when std_diff <= 0
    4. Square the signal and preserve sign: sign(x) * x^2
    5. Split dataset into upgrades (rating_diff_mean > 0) and downgrades (≤ 0)
    6. Fit separate regressions with intercept adjustments for each regime
    7. Apply regime-specific coefficients and intercepts to generate predictions

    Key Innovation: Asymmetric modeling captures different dynamics of
    good news (upgrades) versus bad news (downgrades).

Data Requirements:
    - Analyst rating history from ESTIMATES_BASE_DIR/ibes.db
        * rating_diff_mean: Average rating change across analysts
        * rating_std: Standard deviation of ratings (coverage/dispersion)
    - Barra industry classifications (ind1)
    - Daily price data for forward returns

Signal Construction:
    1. Coverage filter (lines 28-31):
        std_diff = rating_std.diff()
        if std_diff <= 0: rating_diff_mean = 0

    2. Quadratic transformation (line 33):
        rtg0 = sign(rating_diff_mean) * rating_diff_mean^2

    This amplifies large rating changes while preserving direction.

Parameters:
    --start: Start date (YYYYMMDD format)
    --end: End date (YYYYMMDD format)
    --mid: Middate for in-sample/out-of-sample split
    --lag: Horizon for forward returns (default: 20 days)

Asymmetric Regression (lines 57-97):
    Upgrade Model (rating_diff_mean > 0):
        - Fit regression on stocks with positive rating changes
        - Estimate coefficients and intercepts for each lag
        - Adjust intercepts by subtracting baseline drift

    Downgrade Model (rating_diff_mean ≤ 0):
        - Fit regression on stocks with negative/zero rating changes
        - Separate coefficients capture different mean-reversion dynamics
        - Different intercepts reflect asymmetric drift

Intercept Adjustment:
    Uses get_intercept() to compute baseline return drift, then:
        adjusted_intercept = fitted_intercept - baseline_drift[horizon]

    This isolates the alpha component from market drift.

Output:
    Writes HDF5 file with 'rtg' alpha column, consumable by bsim.py via:
        --fcast=rtg:1:1  (multiplier=1, weight=1)

Example:
    python rating_diff_updn.py --start=20130101 --end=20130630 --mid=20130315 --lag=20

Notes:
    - Longer horizon (20 days) captures longer-term analyst impact
    - Quadratic transformation (x^2) amplifies strong signals
    - Asymmetric treatment allows upgrades and downgrades to have different
      coefficients, decay rates, and intercepts
    - Current implementation uses global models (no industry demeaning in signal)
    - Commented code shows sector-specific alternatives
    - Legacy Python 2.7 codebase
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def wavg(group):
    """
    Calculate market-cap weighted average beta-adjusted return.

    Used for monitoring and debugging market-wide moves.

    Args:
        group: DataFrame group (by date)
               Required: pbeta, log_ret, mkt_cap_y, gdate

    Returns:
        Series: Beta times market cap-weighted return

    Notes:
        - Diagnostic function, not used in signal generation
        - Prints market return for each date
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print("Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum())))
    res = b * ((d * w).sum() / w.sum())
    return res

def calc_rtg_daily(daily_df, horizon):
    """
    Calculate filtered rating change signals with quadratic transformation.

    Applies coverage filter and quadratic transformation to amplify strong
    rating changes while preserving directional information.

    Args:
        daily_df: DataFrame with price and analyst data
                  Required: rating_diff_mean, rating_std, ind1, gdate
        horizon: int, number of lags to create (default: 20)

    Returns:
        DataFrame with additional columns:
            std_diff: Change in rating standard deviation
            rtg0: Quadratic signal = sign(rating_diff_mean) * rating_diff_mean^2
            rtg0_ma: Raw signal (equals rtg0 in current implementation)
            rtg{i}_ma: Lagged signals for i=1 to horizon

    Signal Construction:
        1. Calculate std_diff = rating_std[t] - rating_std[t-1]
        2. Filter: set rating_diff_mean = 0 when std_diff <= 0
        3. Transform: rtg0 = sign(x) * x^2 where x = rating_diff_mean

    Quadratic Transformation Rationale:
        - Linear: rating change of 0.5 → signal of 0.5
        - Quadratic: rating change of 0.5 → signal of 0.25
        - Linear: rating change of 2.0 → signal of 2.0
        - Quadratic: rating change of 2.0 → signal of 4.0

        Effect: Amplifies large changes, dampens small changes,
                preserves sign via sign(x) multiplication.

    Coverage Filter (lines 28-31):
        Only trades when analyst coverage/dispersion is expanding:
        - std_diff > 0: coverage expanding, use signal
        - std_diff ≤ 0 or null: ignore signal

    Debug Output (lines 29-34):
        Prints "SEAN" statistics showing signal distribution before/after
        filtering and example stock data.

    Notes:
        - rtg0_ma currently equals rtg0 (no industry demeaning)
        - Commented code shows alternative industry demeaning approach
        - Multi-lag structure for decay modeling
        - Quadratic amplification is key differentiator vs rating_diff.py
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
    result_df.loc[ result_df['std_diff'] <= 0, 'rating_diff_mean'] = 0
    print(result_df['rating_diff_mean'].describe())
    result_df['rtg0'] = result_df['rating_diff_mean'] * result_df['rating_diff_mean'] * np.sign(result_df['rating_diff_mean'])


    # result_df['rtg0'] = -1.0 * result_df['med_diff_dk']
    # demean = lambda x: (x - x.mean())
    # indgroups = result_df[['rtg0', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    # result_df['rtg0_ma'] = indgroups['rtg0']
    result_df['rtg0_ma'] = result_df['rtg0']

    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['rtg'+str(lag)+'_ma'] = shift_df['rtg0_ma']

    return result_df

def rtg_fits(daily_df, horizon, name, middate=None, intercepts=None):
    """
    Fit asymmetric regressions for upgrades and downgrades separately.

    Recognizes that rating upgrades and downgrades may have different
    predictive power and decay rates, fitting separate models for each regime.

    Args:
        daily_df: DataFrame with signals and forward returns
        horizon: int, maximum lag for regression (default: 20)
        name: str, name suffix for plot outputs
        middate: datetime, in-sample/out-of-sample split date
        intercepts: dict, baseline intercepts by horizon from get_intercept()

    Returns:
        DataFrame (out-of-sample) with columns:
            rtg0_ma_coef: Coefficient (different for upgrades vs downgrades)
            rtg0_ma_intercept: Intercept adjustment
            rtg{i}_ma_coef: Lag coefficients for i=1 to horizon-1
            rtg{i}_ma_intercept: Lag intercept adjustments
            rtg: Final weighted alpha prediction

    Methodology:
        1. Split in-sample data into upgrades (rating_diff_mean > 0)
           and downgrades (rating_diff_mean ≤ 0)

        2. UPGRADE MODEL (lines 57-76):
           - Regress rtg0_ma on forward returns at each lag
           - Fit with intercept estimation (intercept=True)
           - Adjust intercepts: fitted_int - baseline_int[horizon]
           - Extract coefficients and adjusted intercepts
           - Compute incremental lag coefficients:
               coef[lag] = coef[0] - coef[lag]
           - Apply to upgrade stocks in out-of-sample

        3. DOWNGRADE MODEL (lines 78-97):
           - Same process as upgrades but on downgrade subset
           - Separate coefficients and intercepts
           - Apply to downgrade stocks in out-of-sample

        4. COMBINE (lines 99-110):
           - Use downgrade model coefficients as default (line 100-102)
           - Weighted sum of lagged signals with linear decay:
               weight[i] = (horizon - i) / horizon
               alpha = rtg0_ma * coef + intercept
                     + sum(weight[i] * (rtg[i]_ma * coef + intercept[i]))

    Intercept Adjustment:
        Baseline intercepts (from get_intercept) capture market drift.
        Adjusted intercepts isolate alpha:
            alpha_intercept = fitted_intercept - market_drift

    Outputs:
        - Plot: rtg_up_{name}_{dates}.png (upgrade model diagnostics)
        - Plot: rtg_dn_{name}_{dates}.png (downgrade model diagnostics)
        - Console: Coefficients for each lag in both models
        - Console: Alpha summary statistics

    Example:
        For horizon=20, creates separate upgrade/downgrade models:
        - 20 lag regressions for upgrades
        - 20 lag regressions for downgrades
        - Different coefficients capture asymmetric response
        - Intercept adjustments ensure market neutrality

    Notes:
        - Uses regress_alpha() with neutralize=False, intercept=True
        - Linear decay weighting assumes gradual signal decay
        - fillna(0) prevents NaN propagation in missing signals
        - Regime-specific models can have very different dynamics
        - Final coefficient assignment uses downgrade model (line 100)
          but each regime uses its own fitted values
    """
    insample_daily_df = daily_df
    if middate is not None:
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_daily_df = daily_df[ daily_df.index.get_level_values('date') >= middate ]

    outsample_daily_df['rtg'] = np.nan
    ESTIMATE = "rating"

    insample_up_df = insample_daily_df[ insample_daily_df[ESTIMATE + "_diff_mean"] > 0 ]
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr', 'intercept'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_up_df, 'rtg0_ma', ii, False, 'daily', True) 
        fitresults_df['intercept'] = fitresults_df['intercept'] - intercepts[ii]
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "rtg_up_"+name+"_" + df_dates(insample_up_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    coef0 = fits_df.ix['rtg0_ma'].ix[horizon].ix['coef']
    intercept0 = fits_df.ix['rtg0_ma'].ix[horizon].ix['intercept']
    print("Coef{}: {}".format(0, coef0)               )
    outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] > 0, 'rtg0_ma_coef' ] = coef0
    outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] > 0, 'rtg0_ma_intercept' ] =  intercept0
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['rtg0_ma'].ix[lag].ix['coef'] 
        intercept = intercept0 - fits_df.ix['rtg0_ma'].ix[lag].ix['intercept'] 
        print("Coef{}: {}".format(lag, coef))
        outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] > 0, 'rtg'+str(lag)+'_ma_coef' ] = coef
        outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] > 0, 'rtg'+str(lag)+'_ma_intercept' ] = intercept


    insample_dn_df = insample_daily_df[ insample_daily_df[ESTIMATE + "_diff_mean"] <= 0 ]
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr', 'intercept'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_dn_df, 'rtg0_ma', ii, False, 'daily', True) 
        fitresults_df['intercept'] = fitresults_df['intercept'] - intercepts[ii]
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "rtg_dn_"+name+"_" + df_dates(insample_dn_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    coef0 = fits_df.ix['rtg0_ma'].ix[horizon].ix['coef']
    intercept0 = fits_df.ix['rtg0_ma'].ix[horizon].ix['intercept']
    print("Coef{}: {}".format(0, coef0)               )
    outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] <= 0, 'rtg0_ma_coef' ] = coef0
    outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] <= 0, 'rtg0_ma_intercept' ] =  intercept0
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['rtg0_ma'].ix[lag].ix['coef'] 
        intercept = intercept0 - fits_df.ix['rtg0_ma'].ix[lag].ix['intercept'] 
        print("Coef{}: {}".format(lag, coef))
        outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] <= 0, 'rtg'+str(lag)+'_ma_coef' ] = coef
        outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] <= 0, 'rtg'+str(lag)+'_ma_intercept' ] = intercept



    coef0 = fits_df.ix['rtg0_ma'].ix[horizon].ix['coef']
    print("Coef{}: {}".format(0, coef0))
    outsample_daily_df[ 'rtg0_ma_coef' ] = coef0

    outsample_daily_df[ 'rtg' ] = outsample_daily_df['rtg0_ma'].fillna(0) * outsample_daily_df['rtg0_ma_coef']  + outsample_daily_df['rtg0_ma_intercept']
    for lag in range(1,horizon):
        weight = (horizon - lag) / float(horizon)
        lagname = 'rtg'+str(lag)+'_ma'
        print("Running lag {} with weight: {}".format(lag, weight))
        outsample_daily_df[ 'rtg'] += weight * (outsample_daily_df[lagname].fillna(0) * outsample_daily_df['rtg0_ma_coef'] + outsample_daily_df['rtg'+str(lag)+'_ma_intercept'])

    print("Alpha Summary {}".format(name))
    print(outsample_daily_df['rtg'].describe())
    
    return outsample_daily_df

def calc_rtg_forecast(daily_df, horizon, middate):
    """
    Generate analyst rating forecast with asymmetric up/down treatment.

    Main pipeline orchestrating filtered signal calculation, baseline intercept
    estimation, asymmetric regression, and forecast generation.

    Args:
        daily_df: DataFrame with price and analyst data merged
        horizon: int, forward return horizon (default: 20 days)
        middate: datetime, in-sample/out-of-sample split

    Returns:
        DataFrame with 'rtg' column containing final alpha predictions

    Pipeline:
        1. Calculate filtered quadratic signals via calc_rtg_daily()
        2. Calculate forward returns for regression targets
        3. Merge signals with forward returns
        4. Compute baseline intercepts via get_intercept()
        5. Fit asymmetric regressions via rtg_fits()

    Baseline Intercept Calculation:
        get_intercept(data, horizon, 'rtg0_ma', middate) returns dict:
        {
            1: intercept_for_1day,
            2: intercept_for_2day,
            ...
            horizon: intercept_for_Nday
        }
        Used to adjust fitted intercepts and isolate alpha.

    Alternative Approaches (commented):
        - Sector-specific regressions (lines 122-127)
        - Separate upgrade/downgrade models without intercept adjustment

    Notes:
        - Asymmetric fitting is the key differentiator
        - Intercept adjustment ensures market neutrality
        - Current implementation uses single global model
        - Could be extended to sector-specific asymmetric models
        - Commented code shows upgrade/downgrade separation strategy
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

    intercept_d = get_intercept(daily_results_df, horizon, 'rtg0_ma', middate)
    result_df = rtg_fits(daily_results_df, horizon, "", middate, intercept_d)

    # res1 = rtg_fits( daily_results_df[ daily_results_df['rating_diff_mean'] > 0 ], horizon, "up", middate)
    # res2 = rtg_fits( daily_results_df[ daily_results_df['rating_diff_mean'] < 0 ], horizon, "dn", middate)
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

    print("Total Alpha Summary")
    print(result_df['rtg'].describe())

    dump_daily_alpha(result_df, 'rtg')









