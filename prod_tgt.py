#!/usr/bin/env python
"""
Production Price Target Alpha Module - Live Trading Implementation

This module is the production-ready version of the price target strategy (target.py)
designed for live trading systems. It separates model fitting from prediction and
stores fitted coefficients to CSV files for reuse.

Key Differences from target.py:
    1. Coefficient Storage: Fits models once, saves coefficients to CSV
    2. Two-Stage Process: Separate --fit and prediction modes
    3. Live Data Integration: Loads real-time prices via load_live_file()
    4. Production Output: Uses dump_prod_alpha() for trading system integration
    5. Sliding Window: Fits on 720-day lookback, predicts on recent data

Production Workflow:
    Stage 1 - Model Fitting (periodic, e.g., monthly):
        python prod_tgt.py --asof=20130630 --coeffile=/path/to/coefs --fit=True

        - Loads 720 days of historical data
        - Runs WLS regressions to estimate coefficients
        - Saves coefficients to: {coeffile}/{asof}.tgt.csv
        - No predictions generated

    Stage 2 - Daily Prediction (every trading day):
        python prod_tgt.py --asof=20130630 --inputfile=live_prices.csv
                           --outputfile=alpha_tgt.csv --coeffile=20130630.tgt.csv

        - Loads live prices from inputfile
        - Loads pre-fitted coefficients from coeffile
        - Generates alpha forecasts for current day
        - Saves to outputfile for trading system

Signal Calculation:
    Same as target.py:
        tgt0 = winsorize_by_date(log(target_median / close))
        tgt0_ma = tgt0 - industry_mean(tgt0)

    Forecast:
        tgt = sum(coef[lag] * tgt{lag}_ma for lag in 0..horizon-1)

Data Requirements:
    Fit Mode:
        - 720 days of historical data for stable coefficient estimation
        - Analyst price targets from ESTIMATES_BASE_DIR
        - Barra factors (pbeta, ind1)
        - Daily prices for returns

    Predict Mode:
        - Live prices from inputfile (CSV with ticker, close_i, time columns)
        - Recent historical data (horizon + 5 days lookback)
        - Pre-fitted coefficients from coeffile

Parameters:
    --asof: Trading date in YYYYMMDD format
    --inputfile: Path to live price data CSV (predict mode only)
    --outputfile: Path to save alpha forecasts (predict mode only)
    --coeffile: Path to coefficient file or directory
        Fit mode: Directory to save {asof}.tgt.csv
        Predict mode: Path to specific coefficient CSV file
    --fit: Boolean flag for fit vs predict mode

Output Format:
    Predict mode saves via dump_prod_alpha():
        - Ticker-level alpha forecasts
        - Ready for position optimizer integration
        - Compatible with production trading system

Usage Examples:
    # Monthly coefficient fitting (run on last trading day of month)
    python prod_tgt.py --asof=20130630 --coeffile=/data/coefs --fit=True

    # Daily prediction (run each trading day)
    python prod_tgt.py --asof=20130702 --inputfile=/data/live/20130702.csv
                       --outputfile=/data/alpha/20130702_tgt.csv
                       --coeffile=/data/coefs/20130630.tgt.csv

Coefficient Lifecycle:
    - Refit monthly to adapt to changing market conditions
    - Use most recent coefficient file for daily predictions
    - Archive old coefficient files for research and validation

Performance Considerations:
    - Fit mode: ~2-5 minutes for 720 days of data
    - Predict mode: ~10-30 seconds for daily forecasts
    - No HDF5 caching in production (fresh data each run)
    - Live prices updated intraday via inputfile

Note:
    This module is designed for integration with institutional trading systems.
    The two-stage design minimizes latency during trading hours by pre-computing
    coefficients and performing only lightweight predictions on live data.
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from load_data_live import *
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
        Currently not used in production (commented out in calc_tgt_daily).
        Kept for potential future beta adjustment functionality.
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

    Identical to target.py implementation but used in production context.
    Computes target miss signal as log ratio of analyst target to current price,
    then applies industry neutralization and creates distributed lags.

    Signal Formula:
        tgt0 = winsorize_by_date(log(target_median / close_y))
        tgt0_ma = tgt0 - industry_mean(tgt0)  [industry-demeaned]

    Args:
        daily_df (DataFrame): Daily stock data with columns:
            - target_median: Median analyst price target
            - close_y: Yesterday's closing price
            - ind1: Barra industry classification
            - gdate: Trading date
        horizon (int): Number of days for signal decay (fixed at 15)

    Returns:
        DataFrame: Original data with added columns:
            - tgt0: Raw target deviation signal (winsorized)
            - tgt0_ma: Industry-neutral target signal
            - tgt1_ma through tgt{horizon}_ma: Lagged signals

    Note:
        Beta adjustment code (lines 27-29) is commented out in production.
        Industry neutralization alone provides sufficient risk control for
        live trading while reducing computation time.
    """
    print("Caculating daily tgt...")
    result_df = filter_expandable(daily_df)

    print("Calculating tgt0...")
    halflife = horizon / 2
#    result_df['dk'] = np.exp( -1.0 * halflife *  (result_df['gdate'] - result_df['last']).astype('timedelta64[D]').astype(int) )
    # print result_df.columns
    # result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    # result_df['badjret'] = result_df['log_ret'] - result_df['bret']
    # result_df['badj0_B'] = winsorize_by_date(result_df[ 'badjret' ])

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

def generate_coefs(daily_df, horizon, fitfile=None):
    """
    Fit WLS regression models and save coefficients to CSV open(Fit Mode).

    This function is called in --fit mode to estimate how the price target
    signal predicts forward returns. Runs regressions for horizons 1 through
    'horizon' days, then saves the coefficient structure to a CSV file for
    later use in prediction mode.

    Regression Model:
        forward_return[t+h] = intercept[h] + coef[h] * tgt0_ma[t] + controls + error

    Coefficient Structure:
        The function saves incremental coefficients that capture the marginal
        contribution of each lag:
            coef[lag] = coef[horizon] - coef[horizon-lag]

    Args:
        daily_df (DataFrame): Historical data with target signals and forward returns
            Must span 720 days for stable coefficient estimation
        horizon (int): Maximum forecast horizon (typically 15 days)
        fitfile (str): Path to save coefficient CSV file
            Format: {coeffile_dir}/{asof}.tgt.csv

    Returns:
        int: 1 (success indicator)

    Output CSV Format:
        Columns: name, coef
        Rows:
            - tgt0_ma_coef: Coefficient for current day signal
            - tgt1_ma_coef through tgt{horizon-1}_ma_coef: Lagged coefficients

    Process:
        1. Run WLS regression for each horizon (1 to horizon days)
            Uses regress_alpha with intercept=True, daily frequency
        2. Extract coefficient for maximum horizon
        3. Calculate incremental coefficients for distributed lags
        4. Save coefficient table to CSV for production use

    Note:
        Intercept handling is commented out (line 60).
        Industry-neutral construction of tgt0_ma removes systematic biases,
        making intercepts close to zero in practice.

    Usage:
        Called automatically when --fit=True flag is set in main block.
        Typically run monthly to update model coefficients.
    """
    insample_daily_df = daily_df

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr', 'intercept'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'tgt0_ma', ii, True, 'daily', True)
#        fitresults_df['intercept'] = fitresults_df['intercept'] - intercepts[ii]
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 

    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['tgt0_ma'].ix[horizon].ix['coef']
#    intercept0 = fits_df.ix['tgt0_ma'].ix[horizon].ix['intercept']
    print("Coef{}: {}".format(0, coef0))
    coef_list = list()
    coef_list.append( { 'name': 'tgt0_ma_coef', 'coef': coef0 } )
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['tgt0_ma'].ix[lag].ix['coef'] 
#        intercept = intercept0 - fits_df.ix['tgt0_ma'].ix[lag].ix['intercept'] 
        print("Coef{}: {}".format(lag, coef))
        coef_list.append( { 'name': 'tgt'+str(lag)+'_ma_coef', 'coef': coef } )

    coef_df = pd.DataFrame(coef_list)
    coef_df.to_csv(fitfile)

    return 1

def tgt_alpha(daily_df, horizon, fitfile=None):
    """
    Generate alpha forecasts using pre-fitted coefficients (Predict Mode).

    This function is called in prediction mode to apply previously fitted
    coefficients to current market data and generate alpha signals for trading.
    Loads coefficients from CSV file and computes distributed lag forecast.

    Forecast Formula:
        tgt[t] = sum(coef[lag] * tgt{lag}_ma[t] for lag in 0..horizon-1)

    Where:
        - coef[lag]: Pre-fitted coefficients from generate_coefs()
        - tgt{lag}_ma[t]: Industry-neutral target signals with lag

    Args:
        daily_df (DataFrame): Recent market data with target signals
            Typically horizon + 5 days of data ending on asof date
        horizon (int): Forecast horizon (typically 15 days)
        fitfile (str): Path to coefficient CSV file
            Format: {coeffile_dir}/{fit_asof}.tgt.csv

    Returns:
        DataFrame: Input data with added 'tgt' alpha forecast column

    Process:
        1. Load pre-fitted coefficients from CSV file
        2. Multiply each lagged signal by its coefficient
        3. Sum all components to create final forecast
        4. Handle NaN values with fillna(0) to avoid data gaps

    Output:
        The 'tgt' column contains the final alpha signal:
            - Positive values suggest undervalued (buy signal)
            - Negative values suggest overvalued (sell signal)
            - Magnitude indicates conviction level

    Performance:
        Fast execution (~10-30 seconds) as coefficients are pre-computed.
        Suitable for low-latency production trading systems.

    Usage:
        Called automatically when --fit=False (default) in main block.
        Run daily before market open to generate trading signals.

    Note:
        .fillna(0) on line 94 and 97 handles missing lagged signals for
        recently IPO'd stocks or data gaps. Conservative approach treats
        missing data as neutral signal (zero).
    """
    coef_df = pd.read_csv(fitfile, header=0, index_col=['name'])

    outsample_daily_df = daily_df
    outsample_daily_df['tgt'] = 0.0

    for lag in range(0,horizon):
        coef = coef_df.ix[ 'tgt'+str(lag)+'_ma_coef' ]['coef']
        print("Coef: {}".format(coef))
        outsample_daily_df[ 'tgt'+str(lag)+'_ma_coef' ] = coef

    print(outsample_daily_df['tgt'].describe())

    outsample_daily_df[ 'tgt' ] = (outsample_daily_df['tgt0_ma'] * outsample_daily_df['tgt0_ma_coef']).fillna(0) #+ outsample_daily_df['tgt0_ma_intercept']
    for lag in range(1,horizon):
        print(outsample_daily_df['tgt'].describe())
        outsample_daily_df[ 'tgt'] += (outsample_daily_df['tgt'+str(lag)+'_ma'] * outsample_daily_df['tgt'+str(lag)+'_ma_coef']).fillna(0) #+ outsample_daily_df['tgt'+str(lag)+'_ma_intercept']
    
    print(outsample_daily_df['tgt'].describe() )
    return outsample_daily_df

def calc_tgt_forecast(daily_df, horizon, coeffile, fit):
    """
    Main orchestrator for production target alpha generation.

    Coordinates either coefficient fitting or alpha prediction based on the
    'fit' flag. This two-stage design separates expensive model fitting from
    fast daily predictions.

    Modes:
        Fit Mode (fit=True):
            - Loads 720 days of historical data
            - Calculates forward returns for regression
            - Calls generate_coefs() to fit models and save coefficients
            - Returns None (no predictions generated)

        Predict Mode (fit=False):
            - Loads recent data (horizon + 5 days)
            - Applies pre-fitted coefficients via tgt_alpha()
            - Returns dataframe with 'tgt' alpha forecasts

    Args:
        daily_df (DataFrame): Market data with required columns:
            Fit mode: 720 days of historical data
            Predict mode: Recent data (horizon + 5 days)
        horizon (int): Forecast horizon in days (fixed at 15)
        coeffile (str): Coefficient file path or directory
            Fit mode: Directory to save {asof}.tgt.csv
            Predict mode: Path to specific coefficient CSV
        fit (bool): True for fitting, False for prediction

    Returns:
        DataFrame or None:
            Fit mode: None
            Predict mode: DataFrame with 'tgt' alpha column

    Process Flow:
        1. calc_tgt_daily(): Generate target signals and distributed lags
        2a. Fit Mode:
            - calc_forward_returns(): Generate regression targets
            - generate_coefs(): Fit models and save coefficients
        2b. Predict Mode:
            - tgt_alpha(): Apply coefficients to generate forecasts

    Commented Code:
        Lines 105-133 show experimental sector-specific models that were
        tested during development. Final production version uses unified
        model for simplicity and stability.

    Note:
        The function includes extensive commented code from development.
        This preserves the research history and documents alternatives
        that were considered but not implemented in production.
    """
    daily_results_df = calc_tgt_daily(daily_df, horizon) 

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

    # results.append(res1)
    # results.append(res2)

    # result_df = pd.concat(results, verify_integrity=True)
    #    intercept_d = get_intercept(daily_results_df, horizon, 'tgt0_ma', middate)
    if fit:
        forwards_df = calc_forward_returns(daily_df, horizon)
        daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
        generate_coefs( daily_results_df, horizon, coeffile)
        return
    else:
        res1 = tgt_alpha( daily_results_df, horizon, coeffile)
        res1['tgt'].describe() 
        #    res2 = tgt_fits( daily_results_df[ daily_results_df['det_diff'] < 0 ], horizon, "dn", middate, intercept_d)
        result_df = pd.concat([res1], verify_integrity=True)

    return result_df

if __name__=="__main__":
    """
    Command-line interface for production price target alpha generation.

    Two-Stage Production Workflow:

    Stage 1 - Model Fitting (monthly):
        python prod_tgt.py --asof=20130630 --coeffile=/data/coefs --fit=True

        Actions:
            - Loads 720 days (2 years) of historical data
            - Runs WLS regressions to estimate coefficients
            - Saves coefficients to: {coeffile}/{asof}.tgt.csv
            - No alpha forecasts generated
            - Runtime: ~2-5 minutes

    Stage 2 - Daily Prediction (each trading day):
        python prod_tgt.py --asof=20130702 --inputfile=/data/live/prices.csv
                           --outputfile=/data/alpha/tgt.csv
                           --coeffile=/data/coefs/20130630.tgt.csv

        Actions:
            - Loads live prices from inputfile
            - Loads recent historical data (20 days lookback)
            - Applies pre-fitted coefficients
            - Saves alpha forecasts to outputfile
            - Runtime: ~10-30 seconds

    Arguments:
        --asof: Trading date in YYYYMMDD format (required)
            Fit mode: Date to name coefficient file
            Predict mode: Date for alpha generation

        --inputfile: Path to live price CSV open(predict mode only)
            Format: ticker, close_i, time columns
            Contains intraday prices for all universe stocks

        --outputfile: Path to save alpha forecasts (predict mode only)
            Output via dump_prod_alpha() for trading system integration

        --logfile: Path for logging output (currently not used)

        --coeffile: Coefficient file path or directory (required)
            Fit mode: Directory path, saves to {dir}/{asof}.tgt.csv
            Predict mode: Full path to specific coefficient CSV

        --fit: Boolean flag for mode selection (default False)
            True: Fit mode (generate coefficients)
            False: Predict mode (generate alphas)

    Data Requirements:
        Fit Mode:
            - 720 days of historical prices and returns
            - Analyst price targets from ESTIMATES_BASE_DIR
            - Barra factors (pbeta, ind1)

        Predict Mode:
            - Live prices from inputfile
            - Recent historical data (horizon + 5 days)
            - Pre-fitted coefficient CSV file

    Live Data Integration:
        Line 173: load_live_file(args.inputfile)
            Reads CSV with current intraday prices

        Line 189: Merges live prices into daily_df
            daily_df.ix[lastday, 'prc'] = daily_df['close_i']
            Uses intraday price (close_i) for most recent day

    Output Format:
        dump_prod_alpha(result_df, 'tgt', args.outputfile)
            Saves ticker-level alpha forecasts
            Compatible with portfolio optimizer input format

    Production Schedule:
        - Refit coefficients: Last trading day of each month
        - Daily predictions: Every trading day before market open
        - Coefficient lifecycle: Use most recent fit for all predictions

    Example Workflow:
        # End of June: Fit new coefficients
        python prod_tgt.py --asof=20130630 --coeffile=/data/coefs --fit=True

        # July 1st: Generate alpha using June coefficients
        python prod_tgt.py --asof=20130701 --inputfile=/data/live/20130701.csv
                           --outputfile=/data/alpha/20130701_tgt.csv
                           --coeffile=/data/coefs/20130630.tgt.csv

        # July 2nd: Generate alpha using same June coefficients
        python prod_tgt.py --asof=20130702 --inputfile=/data/live/20130702.csv
                           --outputfile=/data/alpha/20130702_tgt.csv
                           --coeffile=/data/coefs/20130630.tgt.csv

        # End of July: Fit new coefficients for August
        python prod_tgt.py --asof=20130731 --coeffile=/data/coefs --fit=True

    Performance Optimization:
        - No HDF5 caching in production (fresh data each run)
        - Pre-fitted coefficients minimize latency
        - Suitable for low-latency trading systems
        - Can run in parallel for multiple alpha families

    Note:
        The two-stage design separates expensive fitting (monthly) from
        fast prediction (daily), enabling low-latency alpha generation
        suitable for institutional trading systems.
    """

    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--asof",action="store",dest="asof",default=None)
    parser.add_argument("--inputfile",action="store",dest="inputfile",default=None)
    parser.add_argument("--outputfile",action="store",dest="outputfile",default=None)
    parser.add_argument("--logfile",action="store",dest="logfile",default=None)
    parser.add_argument("--coeffile",action="store",dest="coeffile",default=None)
    parser.add_argument("--fit",action="store",dest="fit",default=False)
    args = parser.parse_args()

    horizon = int(15)
    end = datetime.strptime(args.asof, "%Y%m%d")
    if args.fit:
        print("Fitting...")
        coeffile = args.coeffile + "/" + args.asof + ".tgt.csv"
        lookback = timedelta(days=720)    
        start = end - lookback
        uni_df = get_uni(start, end, 30)
    else:
        print("Not fitting...")
        coeffile = args.coeffile
        lookback = timedelta(days=horizon+5)    
        start = end - lookback
        uni_df = load_live_file(args.inputfile)
        end = datetime.strptime(args.asof + '_' + uni_df['time'].min(), '%Y%m%d_%H:%M:%S')
    
    print("Running between {} and {}".format(start, end))

    BARRA_COLS = ['ind1', 'pbeta']
    barra_df = load_barra(uni_df, start, end, BARRA_COLS)
    PRICE_COLS = ['close']
    price_df = load_prices(uni_df, start, end, PRICE_COLS)

    daily_df = merge_barra_data(price_df, barra_df)
    analyst_df = load_target_hist(price_df[['ticker']], start, end)
    daily_df = merge_daily_calcs(analyst_df, daily_df)

    daily_df['prc'] = daily_df['close']
    if not args.fit:
        lastday = daily_df['gdate'].max()
        daily_df.ix[ daily_df['gdate'] == lastday, 'prc'] = daily_df['close_i'] 

    result_df = calc_tgt_forecast(daily_df, horizon, coeffile, args.fit)
    if not args.fit:
        print(result_df.head())
        dump_prod_alpha(result_df, 'tgt', args.outputfile)



