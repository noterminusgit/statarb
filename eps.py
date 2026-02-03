#!/usr/bin/env python
"""
Earnings Surprise (EPS) Alpha Strategy - Post-Earnings Announcement Drift (PEAD)

This module implements an event-driven earnings surprise strategy based on the well-documented
Post-Earnings Announcement Drift (PEAD) anomaly. The strategy exploits the tendency of stocks
to continue drifting in the direction of their earnings surprise for several days/weeks after
the announcement.

Academic Basis:
    The PEAD anomaly was first documented by Ball and Brown (1968) and extensively studied
    in behavioral finance literature. Stocks that report positive earnings surprises tend
    to outperform over the subsequent 60 trading days, while negative surprises lead to
    underperformance. This drift is attributed to market underreaction to earnings news.

Strategy Overview:
    1. Calculate earnings surprise: (actual EPS - consensus estimate) / median estimate
    2. Signal is triggered when new estimates arrive (std_diff > 0)
    3. Use distributed lag structure to capture signal decay over horizon
    4. Regress surprise signal against forward returns to estimate coefficients
    5. Generate forecasts using fitted coefficients on out-of-sample data

Signal Calculation:
    eps0 = EPS_diff_mean / EPS_median

    Where:
        - EPS_diff_mean: Mean difference between actual and estimated EPS
        - EPS_median: Median analyst EPS estimate
        - Signal is only active when std_diff > 0 (new estimate data available)

Event Windows:
    - Signal generated on earnings announcement dates
    - Distributed lag structure captures decay over 'horizon' days (typically 15-20 days)
    - Coefficients estimated via WLS regression against forward returns

Data Requirements:
    - Earnings estimates and actuals from ESTIMATES_BASE_DIR
    - Required columns: EPS_diff_mean, EPS_median, EPS_std
    - Barra factors for beta adjustment and industry classification
    - Daily price data for return calculations

Parameters:
    horizon (int): Forecast horizon in days (default 20)
        Controls signal decay period and regression horizon
    middate (datetime): Split date between in-sample and out-of-sample
        Used for walk-forward testing and coefficient estimation
    lag (int): Alias for horizon parameter

Usage:
    python eps.py --start=20130101 --end=20130630 --mid=20130401 --lag=20

Output:
    Generates 'eps' alpha signal saved via dump_daily_alpha()
    Signal is winsorized and can be combined with other alphas

References:
    - Ball, R. and Brown, P. (1968). "An Empirical Evaluation of Accounting Income Numbers"
    - Bernard, V. and Thomas, J. (1989). "Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?"
"""

from regress import *
from loaddata import *
from util import *

from pandas.stats.moments import ewma

def wavg(group):
    """
    Calculate market-cap weighted average returns adjusted by beta.

    Computes the market return for a given date group, then scales each stock's
    beta by this market return to create a beta-adjusted expected return component.

    Args:
        group (DataFrame): Date-grouped dataframe with columns:
            - pbeta: Predicted beta from Barra model
            - log_ret: Log returns for the day
            - mkt_cap_y: Market capitalization in dollars
            - gdate: Trading date

    Returns:
        Series: Beta-adjusted market return for each stock in the group

    Note:
        This is used to remove market beta exposure from returns, creating
        market-neutral signals for the earnings surprise strategy.
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print "Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum()))
    res = b * ((d * w).sum() / w.sum())
    return res


def calc_eps_daily(daily_df, horizon):
    """
    Calculate daily earnings surprise signals with distributed lag structure.

    Computes the core EPS surprise signal (eps0) and creates lagged versions
    to capture signal decay over the forecast horizon. The signal is only
    active when new estimate data arrives (std_diff > 0).

    Signal Formula:
        eps0 = EPS_diff_mean / EPS_median

        Only active when: std_diff = EPS_std.diff() > 0
        (i.e., new estimate data has arrived)

    The function creates a distributed lag structure by computing:
        - eps0_ma: Current day signal
        - eps1_ma through eps{horizon}_ma: Lagged signals

    This allows the regression to estimate how the signal predictive power
    decays over time after the earnings announcement.

    Args:
        daily_df (DataFrame): Daily stock data with columns:
            - EPS_diff_mean: Mean difference (actual - estimate)
            - EPS_median: Median analyst EPS estimate
            - EPS_std: Standard deviation of estimates
            - All columns required by filter_expandable()
        horizon (int): Number of days for signal decay (typically 15-20)

    Returns:
        DataFrame: Original data with added columns:
            - eps0_ma: Current earnings surprise signal
            - eps1_ma through eps{horizon}_ma: Lagged signals
            - std_diff: Change in estimate std (trigger for new data)

    Note:
        Commented code shows alternative approaches tried during development:
        - Beta-adjusted returns
        - Cumulative returns
        - Median rating changes
        - Industry demeaning
        These were tested but not used in final implementation.
    """
    print "Caculating daily eps..."
    result_df = filter_expandable(daily_df)

    print "Calculating eps0..."    
    halflife = horizon / 2
#    result_df['dk'] = np.exp( -1.0 * halflife *  (result_df['gdate'] - result_df['last']).astype('timedelta64[D]').astype(int) )

    # result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    # result_df['badjret'] = result_df['log_ret'] - result_df['bret']
    # result_df['badj0_B'] = winsorize_by_date(result_df[ 'badjret' ])

    # result_df['cum_ret'] = pd.rolling_sum(result_df['log_ret'], horizon)

    result_df['std_diff'] = result_df['EPS_std'].unstack().diff().stack()
    result_df.loc[ (result_df['std_diff'] <= 0) | (result_df['std_diff'].isnull()), 'EPS_diff_mean'] = 0
    print "SEAN2"
    print result_df.xs(testid, level=1)
    result_df['eps0'] = result_df['EPS_diff_mean'] / result_df['EPS_median']

    # print result_df.columns
    # result_df['sum'] = result_df['EPS_median'] 
    # result_df['det_diff'] = (result_df['sum'].diff())
    # result_df['det_diff_sum'] = pd.rolling_sum( result_df['det_diff'], window=2)
    # #result_df['det_diff_dk'] = ewma(result_df['det_diff'], halflife=horizon )   
    # result_df['eps0'] = result_df['det_diff'] 

    # result_df['median'] = -1.0 * (result_df['median'] - 3)
    # result_df['med_diff'] = result_df['median'].unstack().diff().stack()
    # result_df['med_diff_dk'] = pd.rolling_sum( result_df['dk'] * result_df['med_diff'], window=horizon )
    # result_df['eps0'] = (np.sign(result_df['med_diff_dk']) * np.sign(result_df['cum_ret'])).clip(lower=0) * result_df['med_diff_dk']


    # demean = lambda x: (x - x.mean())
    # indgroups = result_df[['eps0', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    # result_df['eps0_ma'] = indgroups['eps0']

#    result_df['eps0_ma'] = result_df['eps0_ma'] * (np.sign(result_df['eps0_ma']) * np.sign(result_df['cum_ret']))

    result_df['eps0_ma'] = result_df['eps0']

    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['eps'+str(lag)+'_ma'] = shift_df['eps0_ma']

    return result_df

def eps_fits(daily_df, horizon, name, middate=None):
    """
    Fit distributed lag regression of EPS surprise signal against forward returns.

    Uses weighted least squares (WLS) regression to estimate how the earnings
    surprise signal predicts forward returns at various horizons. Estimates
    coefficients on in-sample data and applies them to out-of-sample data.

    Regression Model:
        forward_return[t+h] = coef[h] * eps0_ma[t] + controls + error

    The distributed lag structure allows for:
        forecast[t] = sum(coef[lag] * eps{lag}_ma[t] for lag in 0..horizon-1)

    This captures the signal decay pattern over time.

    Args:
        daily_df (DataFrame): Daily data with eps signals and forward returns
        horizon (int): Maximum forecast horizon for regression
        name (str): Identifier for plot filename (e.g., sector name)
        middate (datetime, optional): Split date for in/out-of-sample
            If None, uses all data for fitting

    Returns:
        DataFrame: Out-of-sample data with added columns:
            - eps0_ma_coef through eps{horizon-1}_ma_coef: Fitted coefficients
            - eps: Combined forecast signal

    Process:
        1. Split data at middate into train/test sets
        2. Run WLS regression for each horizon (1 to horizon days)
        3. Plot fit quality diagnostics
        4. Calculate incremental coefficients for distributed lag
        5. Apply coefficients to out-of-sample signals
        6. Sum lagged components to create final forecast

    Note:
        Incremental coefficients computed as:
            coef[lag] = coef[horizon] - coef[horizon-lag]
        This captures the marginal contribution of each lag.
    """
    insample_daily_df = daily_df
    if middate is not None:
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_daily_df = daily_df[ daily_df.index.get_level_values('date') >= middate ]

    print insample_daily_df['eps0_ma'].describe()
    print outsample_daily_df['eps0_ma'].describe()
    outsample_daily_df['eps'] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'eps0_ma', ii, False, 'daily', False) 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "eps_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['eps0_ma'].ix[horizon].ix['coef']
    print "Coef{}: {}".format(0, coef0)               
    outsample_daily_df[ 'eps0_ma_coef' ] = coef0
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['eps0_ma'].ix[lag].ix['coef'] 
        print "Coef{}: {}".format(lag, coef)
        outsample_daily_df[ 'eps'+str(lag)+'_ma_coef' ] = coef

    print "SEAN1"
    print outsample_daily_df['eps0_ma'].describe()
    outsample_daily_df[ 'eps' ] = outsample_daily_df['eps0_ma'].fillna(0) * outsample_daily_df['eps0_ma_coef']
    print outsample_daily_df['eps'].describe()
    for lag in range(1,horizon):
        outsample_daily_df[ 'eps'] += outsample_daily_df['eps'+str(lag)+'_ma'].fillna(0) * outsample_daily_df['eps'+str(lag)+'_ma_coef']
        print outsample_daily_df['eps'].describe()
    
    return outsample_daily_df

def calc_eps_forecast(daily_df, horizon, middate):
    """
    Main entry point for generating EPS-based alpha forecasts.

    Orchestrates the complete workflow:
        1. Calculate earnings surprise signals with distributed lags
        2. Calculate forward returns for regression
        3. Fit regression models and generate out-of-sample forecasts

    The function includes commented code showing experimental approaches:
        - Sector-specific models (Energy vs rest)
        - Separate models for positive vs negative surprises
        - Direction-filtered signals
    These were tested but final implementation uses a unified model.

    Args:
        daily_df (DataFrame): Daily stock data with required columns:
            - Price, returns, market cap
            - Barra factors (pbeta, ind1)
            - Earnings estimates (EPS_diff_mean, EPS_median, EPS_std)
        horizon (int): Forecast horizon in days
        middate (datetime): Train/test split date for walk-forward testing

    Returns:
        DataFrame: Out-of-sample data with 'eps' alpha forecast column

    Process Flow:
        1. calc_eps_daily(): Generate surprise signals and lags
        2. calc_forward_returns(): Generate regression targets
        3. eps_fits(): Fit models and generate forecasts

    Note:
        Final implementation uses single unified model across all stocks.
        Commented code shows sector-specific and directional variants that
        were tested during development.
    """
    daily_results_df = calc_eps_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

    #results = list()
    # for sector_name in daily_results_df['sector_name'].dropna().unique():
    #     print "Running eps for sector {}".format(sector_name)
    #     sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    #     result_df = eps_fits(sector_df, horizon, sector_name, middate)
    #     results.append(result_df)
    # result_df = pd.concat(results, verify_integrity=True)


#    result_df = eps_fits(daily_results_df, horizon, "", middate)

#    daily_results_df = daily_results_df[ daily_results_df['det_diff'] > 0]

#     results = list()
#     sector_name = 'Energy'
#     print "Running eps for sector {}".format(sector_name)
#     sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
#     res1 = eps_fits( sector_df[ sector_df['det_diff'] > 0 ], horizon, "energy_up", middate)
# #    res2 = eps_fits( sector_df[ sector_df['det_diff'] < 0 ], horizon, "energy_dn", middate)
#     results.append(res1)
# #    results.append(res2)

#     print "Running eps for not sector {}".format(sector_name)
#     sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
#     res1 = eps_fits( sector_df[ sector_df['det_diff'] > 0 ], horizon, "rest_up", middate)
# #    res2 = eps_fits( sector_df[ sector_df['det_diff'] < 0 ], horizon, "rest_dn", middate)
#     results.append(res1)
# #    results.append(res2)

#     result_df = pd.concat(results, verify_integrity=True)

#    res1 = eps_fits( daily_results_df[ daily_results_df['EPS_diff_mean'] > 0 ], horizon, "up", middate)
#    res2 = eps_fits( daily_results_df[ daily_results_df['EPS_diff_mean'] < 0 ], horizon, "both", middate)
    res2 = eps_fits( daily_results_df, horizon, "both", middate)
    result_df = pd.concat([res2], verify_integrity=True)

    return result_df

if __name__=="__main__":
    """
    Command-line interface for earnings surprise alpha generation.

    Usage:
        python eps.py --start=20130101 --end=20130630 --mid=20130401 --lag=20

    Arguments:
        --start: Start date for data loading (YYYYMMDD format)
        --end: End date for data loading (YYYYMMDD format)
        --mid: Split date between in-sample and out-of-sample (YYYYMMDD format)
        --lag: Forecast horizon in days (default 20)
            Also controls distributed lag window for signal decay

    Data Loading:
        - Attempts to load cached HDF5 file first for faster execution
        - If cache missing, loads from raw sources:
            * Universe definition (top 1,400 stocks)
            * Barra factors (pbeta, ind1)
            * Daily prices (close)
            * Analyst earnings estimates (EPS data)
        - Caches loaded data to HDF5 for future runs

    Output:
        - Saves 'eps' alpha signal via dump_daily_alpha()
        - Signal is out-of-sample forecast from middate onwards
        - Can be used in bsim.py via --fcast=eps:1:1 format

    Performance:
        - Uses HDF5 caching to avoid reloading data
        - Cache file: ./eps{start}.{end}_daily.h5
        - Compressed with zlib for smaller file size

    Example:
        # Generate EPS alpha for H1 2013, split at April 1
        python eps.py --start=20130101 --end=20130630 --mid=20130401 --lag=15
    """
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
    pname = "./eps" + start + "." + end
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
        analyst_df = load_estimate_hist(price_df[['ticker']], start, end, "EPS")
        daily_df = merge_daily_calcs(analyst_df, daily_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')

    result_df = calc_eps_forecast(daily_df, horizon, middate)
    dump_daily_alpha(result_df, 'eps')









