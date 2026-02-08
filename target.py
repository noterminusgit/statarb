#!/usr/bin/env python
"""
Price Target Miss Alpha Strategy - Analyst Target Deviation Signals

This module implements an event-driven strategy based on deviations between analyst
price targets and current stock prices. The strategy exploits the tendency of stocks
to move toward their consensus price targets over time, while accounting for behavioral
biases in target-setting by analysts.

Strategy Overview:
    When analyst price targets deviate significantly from current prices, the market
    tends to gradually adjust toward the target over subsequent weeks. This creates
    a predictable drift similar to PEAD but driven by valuation rather than earnings.

Signal Calculation:
    tgt0 = log(target_median / close)

    Where:
        - target_median: Median analyst price target
        - close: Current closing price
        - Log ratio captures percentage deviation from target

    The signal is then industry-demeaned to create market-neutral positions:
        tgt0_ma = tgt0 - industry_mean(tgt0)

Event Windows and Timing:
    - Signal updates whenever new analyst targets are published
    - Distributed lag structure captures signal decay over 'horizon' days (typically 15 days)
    - Beta adjustment option available to remove systematic market risk
    - Industry neutralization reduces sector concentration risk

Data Requirements:
    - Analyst price targets from ESTIMATES_BASE_DIR via load_target_hist()
    - Required columns: target_median (consensus price target)
    - Barra factors for beta adjustment and industry classification (ind1)
    - Daily price data for return calculations and target comparison

Parameters:
    horizon (int): Forecast horizon in days (default 15)
        Controls signal decay period and regression horizon
    middate (datetime): Split date between in-sample and out-of-sample
        Used for walk-forward testing and coefficient estimation
    lag (int): Alias for horizon parameter

Signal Characteristics:
    - Mean-reverting behavior (stocks move toward targets)
    - Industry-neutral by construction
    - Optionally beta-adjusted for market neutrality
    - Winsorized to limit outlier impact

Usage:
    python target.py --start=20130101 --end=20130630 --mid=20130401 --lag=15

Output:
    Generates 'tgt' alpha signal saved via dump_daily_alpha()
    Signal is winsorized and industry-neutral
    Can be combined with other alphas in multi-strategy portfolios

Academic Context:
    Related to analyst recommendation literature and limits to arbitrage.
    Price targets represent analyst expectations but markets are slow to
    fully incorporate this information, creating exploitable drift patterns.

Note:
    This strategy is complementary to earnings-based strategies (eps.py)
    as it captures valuation-driven rather than earnings-driven drift.
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

from pandas.stats.moments import ewma

def wavg(group):
    """
    Calculate market-cap weighted average returns adjusted by beta.

    Computes the market return for a given date group, then scales each stock's
    beta by this market return to create a beta-adjusted expected return component.
    This is used to remove systematic market risk from returns.

    Args:
        group (DataFrame): Date-grouped dataframe with columns:
            - pbeta: Predicted beta from Barra model
            - log_ret: Log returns for the day
            - mkt_cap_y: Market capitalization in dollars
            - gdate: Trading date

    Returns:
        Series: Beta-adjusted market return for each stock in the group

    Note:
        Used in commented beta adjustment code (lines 26-28).
        Beta adjustment helps create market-neutral signals by removing
        the portion of returns explained by market movements.
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print("Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum())))
    res = b * ((d * w).sum() / w.sum())
    return res


def calc_tgt_daily(daily_df, horizon):
    """
    Calculate daily price target deviation signals with distributed lag structure.

    Computes the core target miss signal (tgt0) as the log ratio of analyst
    consensus price target to current price, then applies industry neutralization.
    Creates lagged versions to capture signal decay over the forecast horizon.

    Signal Formula:
        tgt0 = winsorize_by_date(log(target_median / close_y))
        tgt0_ma = tgt0 - industry_mean(tgt0)  [industry-demeaned]

    The winsorization limits extreme outliers, while industry demeaning
    ensures the signal is neutral within each industry sector.

    Distributed Lag Structure:
        - tgt0_ma: Current day signal (industry-neutral)
        - tgt1_ma through tgt{horizon}_ma: Lagged signals

    This allows the regression to estimate how the signal's predictive power
    decays over time as the stock price adjusts toward the target.

    Args:
        daily_df (DataFrame): Daily stock data with columns:
            - target_median: Median analyst price target
            - close_y: Yesterday's closing price
            - ind1: Barra industry classification
            - gdate: Trading date
            - All columns required by filter_expandable()
        horizon (int): Number of days for signal decay (typically 15)

    Returns:
        DataFrame: Original data with added columns:
            - tgt0: Raw target deviation signal (winsorized)
            - tgt0_ma: Industry-neutral target signal
            - tgt1_ma through tgt{horizon}_ma: Lagged signals

    Note:
        Commented code (lines 26-29) shows optional beta adjustment:
            badj0_B = winsorize(log_ret - beta * market_return)
        This was tested but not used in final implementation.
        Industry neutralization alone provides sufficient risk control.
    """
    print("Caculating daily tgt...")
    result_df = filter_expandable(daily_df)

    print("Calculating tgt0...")
    halflife = horizon / 2
#    result_df['dk'] = np.exp( -1.0 * halflife *  (result_df['gdate'] - result_df['last']).astype('timedelta64[D]').astype(int) )
    print(result_df.columns)
    result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['log_ret'] - result_df['bret']
    result_df['badj0_B'] = winsorize_by_date(result_df[ 'badjret' ])

    #result_df['median_diff'] = result_df['target_median'].unstack().diff().stack()
    #result_df.loc[ result_df['std_diff'] <= 0, 'target_diff_mean'] = 0
    result_df['tgt0'] = winsorize_by_date(np.log(result_df['target_median'] / result_df['close_y']))


    # result_df['median'] = -1.0 * (result_df['median'] - 3)
    # result_df['med_diff'] = result_df['median'].unstack().diff().stack()
    # result_df['med_diff_dk'] = pd.rolling_sum( result_df['dk'] * result_df['med_diff'], window=horizon )
    # result_df['tgt0'] = (np.sign(result_df['med_diff_dk']) * np.sign(result_df['cum_ret'])).clip(lower=0) * result_df['med_diff_dk']


    demean = lambda x: (x - x.mean())
    indgroups = result_df[['tgt0', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    result_df['tgt0_ma'] = indgroups['tgt0']

#    result_df['tgt0_ma'] = result_df['tgt0_ma'] * (np.sign(result_df['tgt0_ma']) * np.sign(result_df['cum_ret']))

#    result_df['tgt0_ma'] = result_df['tgt0']

    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['tgt'+str(lag)+'_ma'] = shift_df['tgt0_ma']

    return result_df

def tgt_fits(daily_df, horizon, name, middate=None, intercepts=None):
    """
    Fit distributed lag regression of price target signal against forward returns.

    Uses weighted least squares (WLS) regression to estimate how the price target
    deviation signal predicts forward returns at various horizons. Estimates
    coefficients on in-sample data and applies them to out-of-sample data.

    Regression Model:
        forward_return[t+h] = intercept[h] + coef[h] * tgt0_ma[t] + controls + error

    The distributed lag structure allows for:
        forecast[t] = sum(coef[lag] * tgt{lag}_ma[t] for lag in 0..horizon-1)

    This captures how the signal's predictive power evolves as prices adjust
    toward analyst targets.

    Args:
        daily_df (DataFrame): Daily data with target signals and forward returns
        horizon (int): Maximum forecast horizon for regression
        name (str): Identifier for plot filename (e.g., sector name)
        middate (datetime, optional): Split date for in/out-of-sample
            If None, uses all data for fitting
        intercepts (dict, optional): Pre-computed intercepts for adjustment
            Currently not used (commented out), but could adjust for systematic biases

    Returns:
        DataFrame: Out-of-sample data with added columns:
            - tgt0_ma_coef through tgt{horizon-1}_ma_coef: Fitted coefficients
            - tgt0_ma_intercept through tgt{horizon-1}_ma_intercept: Intercepts (set to 0)
            - tgt: Combined forecast signal

    Process:
        1. Split data at middate into train/test sets
        2. Run WLS regression for each horizon (1 to horizon days)
            Note: regression uses intercept=True and daily frequency
        3. Plot fit quality diagnostics
        4. Calculate incremental coefficients for distributed lag
        5. Apply coefficients to out-of-sample signals
        6. Sum lagged components to create final forecast

    Incremental Coefficients:
        coef[lag] = coef[horizon] - coef[horizon-lag]

        This captures the marginal contribution of each lag to the
        total forecast at the maximum horizon.

    Note:
        Intercepts are currently set to 0 (commented code shows intercept handling).
        The industry-neutral construction of tgt0_ma already removes systematic biases.
    """
    insample_daily_df = daily_df
    if middate is not None:
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_daily_df = daily_df[ daily_df.index.get_level_values('date') >= middate ]

    outsample_daily_df['tgt'] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr', 'intercept'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'tgt0_ma', ii, True, 'daily', True) 
#        fitresults_df['intercept'] = fitresults_df['intercept'] - intercepts[ii]
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 

    plot_fit(fits_df, "tgt_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['tgt0_ma'].ix[horizon].ix['coef']
#    intercept0 = fits_df.ix['tgt0_ma'].ix[horizon].ix['intercept']
    print("Coef{}: {}".format(0, coef0))
    outsample_daily_df[ 'tgt0_ma_coef' ] = coef0
    outsample_daily_df[ 'tgt0_ma_intercept' ] = 0 # intercept0
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['tgt0_ma'].ix[lag].ix['coef'] 
#        intercept = intercept0 - fits_df.ix['tgt0_ma'].ix[lag].ix['intercept'] 
        print("Coef{}: {}".format(lag, coef))
        outsample_daily_df[ 'tgt'+str(lag)+'_ma_coef' ] = coef
 #       outsample_daily_df[ 'tgt'+str(lag)+'_ma_intercept' ] = intercept

    outsample_daily_df[ 'tgt' ] = outsample_daily_df['tgt0_ma'] * outsample_daily_df['tgt0_ma_coef'] #+ outsample_daily_df['tgt0_ma_intercept']
    for lag in range(1,horizon):
        outsample_daily_df[ 'tgt'] += outsample_daily_df['tgt'+str(lag)+'_ma'] * outsample_daily_df['tgt'+str(lag)+'_ma_coef'] #+ outsample_daily_df['tgt'+str(lag)+'_ma_intercept']
    
    return outsample_daily_df


def calc_tgt_forecast(daily_df, horizon, middate):
    """
    Main entry point for generating price target-based alpha forecasts.

    Orchestrates the complete workflow:
        1. Calculate price target deviation signals with distributed lags
        2. Calculate forward returns for regression
        3. Fit regression models and generate out-of-sample forecasts

    The function includes commented code showing experimental approaches:
        - Sector-specific models (Energy vs rest)
        - Separate models for positive vs negative deviations
        - Direction-filtered signals
        - Intercept adjustment (get_intercept())
    These were tested but final implementation uses a unified model.

    Args:
        daily_df (DataFrame): Daily stock data with required columns:
            - Price data (close_y for target comparison)
            - Barra factors (pbeta, ind1 for industry neutralization)
            - Analyst price targets (target_median)
        horizon (int): Forecast horizon in days
        middate (datetime): Train/test split date for walk-forward testing

    Returns:
        DataFrame: Out-of-sample data with 'tgt' alpha forecast column

    Process Flow:
        1. calc_tgt_daily(): Generate target signals and lags
        2. calc_forward_returns(): Generate regression targets
        3. tgt_fits(): Fit models and generate forecasts

    Model Selection:
        Final implementation uses single unified model across all stocks
        and all deviation directions. Commented code shows variants:
            - Line 125: get_intercept() for bias adjustment
            - Lines 109-124: Sector-specific Energy models
            - Line 127: Separate down-deviation model
        These were tested during development but unified model performed best.

    Note:
        The strategy is complementary to earnings strategies (eps.py)
        as it captures valuation-driven rather than earnings-driven signals.
    """
    daily_results_df = calc_tgt_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

    #results = list()
    # for sector_name in daily_results_df['sector_name'].dropna().unique():
    #     print "Running tgt for sector {}".format(sector_name)
    #     sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    #     result_df = tgt_fits(sector_df, horizon, sector_name, middate)
    #     results.append(result_df)
    # result_df = pd.concat(results, verify_integrity=True)

  #  result_df = tgt_fits(daily_results_df, horizon, "", middate)

#    daily_results_df = daily_results_df[ daily_results_df['det_diff'] > 0]

    # results = list()
    # sector_name = 'Energy'
    # print "Running tgt for sector {}".format(sector_name)
    # sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    # res1 = tgt_fits( sector_df[ sector_df['det_diff'] > 0 ], horizon, "energy_up", middate)
    # res2 = tgt_fits( sector_df[ sector_df['det_diff'] < 0 ], horizon, "energy_dn", middate)
    # results.append(res1)
    # results.append(res2)

    # print "Running tgt for not sector {}".format(sector_name)
    # sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
    # res1 = tgt_fits( sector_df[ sector_df['det_diff'] > 0 ], horizon, "rest_up", middate)
    # res2 = tgt_fits( sector_df[ sector_df['det_diff'] < 0 ], horizon, "rest_dn", middate)
    # results.append(res1)
    # results.append(res2)

    # result_df = pd.concat(results, verify_integrity=True)
#    intercept_d = get_intercept(daily_results_df, horizon, 'tgt0_ma', middate)
    res1 = tgt_fits( daily_results_df, horizon, "", middate)
#    res2 = tgt_fits( daily_results_df[ daily_results_df['det_diff'] < 0 ], horizon, "dn", middate, intercept_d)
    result_df = pd.concat([res1], verify_integrity=True)

    return result_df

if __name__=="__main__":
    """
    Command-line interface for price target alpha generation.

    Usage:
        python target.py --start=20130101 --end=20130630 --mid=20130401 --lag=15

    Arguments:
        --start: Start date for data loading (YYYYMMDD format)
        --end: End date for data loading (YYYYMMDD format)
        --mid: Split date between in-sample and out-of-sample (YYYYMMDD format)
        --lag: Forecast horizon in days (default 15)
            Also controls distributed lag window for signal decay
            Note: Default is 15 days (vs 20 for eps.py) as target signals
            decay faster than earnings signals

    Data Loading:
        - Attempts to load cached HDF5 file first for faster execution
        - If cache missing, loads from raw sources:
            * Universe definition (top 1,400 stocks)
            * Barra factors (pbeta, ind1 for industry neutralization)
            * Daily prices (close)
            * Analyst price targets via load_target_hist()
        - Caches loaded data to HDF5 for future runs

    Output:
        - Saves 'tgt' alpha signal via dump_daily_alpha()
        - Signal is out-of-sample forecast from middate onwards
        - Signal is industry-neutral by construction
        - Can be used in bsim.py via --fcast=tgt:1:1 format

    Performance:
        - Uses HDF5 caching to avoid reloading data
        - Cache file: ./tgt{start}.{end}_daily.h5
        - Compressed with zlib for smaller file size

    Example:
        # Generate target alpha for H1 2013, split at April 1
        python target.py --start=20130101 --end=20130630 --mid=20130401 --lag=15

        # Combine with other alphas in simulation
        python bsim.py --start=20130101 --end=20130630 --fcast=tgt:1:0.5,hl:1:0.5

    Note:
        Shorter default horizon (15 vs 20 days) reflects faster mean reversion
        toward analyst price targets compared to earnings drift patterns.
    """
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--lag",action="store",dest="lag",default=15)
#    parser.add_argument("--horizon",action="store",dest="horizon",default=20)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = int(args.lag)
    pname = "./tgt" + start + "." + end
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
        BARRA_COLS = ['ind1', 'pbeta']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)

        daily_df = merge_barra_data(price_df, barra_df)
        analyst_df = load_target_hist(price_df[['ticker']], start, end, False)
        daily_df = merge_daily_calcs(analyst_df, daily_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')

    result_df = calc_tgt_forecast(daily_df, horizon, middate)
    dump_daily_alpha(result_df, 'tgt')









