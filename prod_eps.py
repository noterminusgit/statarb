#!/usr/bin/env python
"""
Production EPS Alpha Generator

Generates alpha signals based on sell-side analyst EPS (Earnings Per Share) estimate
revisions for production trading. The strategy exploits the information content in
analyst estimate changes, particularly when analyst confidence is increasing.

Strategy Logic:
--------------
The core signal is based on the change in mean analyst EPS estimates normalized by
the median EPS estimate:

    eps0 = EPS_diff_mean / EPS_median

Where EPS_diff_mean is only used when analyst confidence is increasing (std_diff > 0).
The strategy uses lagged versions of this signal with regression-fitted coefficients
to optimize the forecast horizon.

Operating Modes:
---------------
1. **Fit Mode** (--fit=True):
   - Uses 720-day lookback window
   - Loads historical universe via get_uni()
   - Runs regression analysis to determine optimal coefficients
   - Saves coefficients to: <coeffile>/<asof>.eps.csv
   - Does NOT generate alpha forecasts

2. **Production Mode** (--fit=False):
   - Uses horizon+5 day lookback window
   - Loads live data from --inputfile
   - Applies pre-fitted coefficients from --coeffile
   - Generates alpha forecasts and writes to --outputfile
   - Output format: ticker, date, alpha value

Data Requirements:
-----------------
- Barra factors: ind1 (industry), pbeta (predicted beta)
- Price data: close prices
- Analyst estimates: EPS estimate history with mean, median, std

CLI Usage:
---------
Fit coefficients (backtest mode):
    python prod_eps.py --asof=20130630 --coeffile=./coefs --fit=True

Generate production alpha:
    python prod_eps.py --asof=20130630 --inputfile=live_data.csv \\
                       --outputfile=eps_alpha.csv --coeffile=./coefs/20130630.eps.csv

Parameters:
    --asof: Date for production run (YYYYMMDD format)
    --inputfile: Live data CSV file (production mode only)
    --outputfile: Alpha output file (production mode only)
    --coeffile: Coefficient file path (production) or directory (fit mode)
    --fit: Enable fit mode (default: False)

Related Modules:
---------------
- prod_sal.py: Similar pattern for analyst estimate revision strategies
- prod_rtg.py: Analyst rating-based production alpha
- eps.py: Backtest version of EPS strategy
- load_data_live.py: Live data loading infrastructure
- regress.py: Regression fitting framework

Output:
------
Production mode writes alpha forecasts via dump_prod_alpha() with:
- ticker: Stock identifier
- gdate: Forecast date
- eps: Alpha signal value (winsorized, demeaned)
"""

from regress import *
from load_data_live import *
from loaddata import *
from util import *

from pandas.stats.moments import ewma

def wavg(group):
    """
    Calculate market-cap-weighted average return scaled by beta.

    Computes the beta-adjusted market return for a group (typically a date).
    Used for calculating beta-adjusted returns in the strategy.

    Args:
        group: DataFrame group with columns:
            - pbeta: Predicted beta values
            - log_ret: Log returns
            - mkt_cap_y: Market capitalization
            - gdate: Group date (for logging)

    Returns:
        Series: Beta values scaled by market-cap-weighted average return
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print "Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum()))
    res = b * ((d * w).sum() / w.sum())
    return res


def calc_eps_daily(daily_df, horizon):
    """
    Calculate daily EPS revision signals with lagged features.

    Generates the core EPS alpha signal based on analyst estimate revisions.
    The signal is only active when analyst confidence is increasing (std_diff > 0).
    Creates lagged versions (eps1_ma through eps<horizon>_ma) for regression fitting.

    Signal Logic:
        - std_diff = change in EPS estimate standard deviation
        - If std_diff <= 0 or null: set EPS_diff_mean = 0 (ignore low-confidence revisions)
        - eps0 = EPS_diff_mean / EPS_median (normalized revision)
        - eps0_ma = eps0 (current implementation, historical versions tested other transforms)

    Args:
        daily_df: DataFrame with columns:
            - EPS_std: Standard deviation of analyst EPS estimates
            - EPS_diff_mean: Change in mean analyst EPS estimates
            - EPS_median: Median analyst EPS estimate
        horizon: Forecast horizon in days (determines number of lags to create)

    Returns:
        DataFrame: Input df with added columns:
            - eps0_ma: Current EPS revision signal
            - eps1_ma through eps<horizon>_ma: Lagged signals for regression
    """
    print "Caculating daily eps..."
    result_df = filter_expandable(daily_df)

    print "Calculating eps0..."    
    #halflife = horizon / 2
#    result_df['dk'] = np.exp( -1.0 * halflife *  (result_df['gdate'] - result_df['last']).astype('timedelta64[D]').astype(int) )

    # result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    # result_df['badjret'] = result_df['log_ret'] - result_df['bret']
    # result_df['badj0_B'] = winsorize_by_date(result_df[ 'badjret' ])

    # result_df['cum_ret'] = pd.rolling_sum(result_df['log_ret'], horizon)

    result_df['std_diff'] = result_df['EPS_std'].unstack().diff().stack()
    result_df.loc[ (result_df['std_diff'] <= 0) | (result_df['std_diff'].isnull()), 'EPS_diff_mean'] = 0
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

def generate_coefs(daily_df, horizon, name, coeffile=None):
    """
    Fit regression coefficients for EPS alpha signal.

    Runs WLS regression for each horizon (1 to horizon) to determine optimal
    coefficients for combining lagged eps signals. Uses incremental coefficient
    approach where lag coefficients are computed as differences from the base
    coefficient.

    Coefficient Logic:
        - coef0 = regression coefficient for eps0_ma at full horizon
        - coef[lag] = coef0 - regression_coef[lag] (incremental adjustment)

    Args:
        daily_df: DataFrame with eps signals and forward returns
        horizon: Maximum forecast horizon in days
        name: Strategy name (for logging, currently unused)
        coeffile: Output CSV path for coefficient file

    Returns:
        None (writes coefficients to CSV file)

    Output CSV Format:
        Columns: name, coef
        Rows: eps0_ma_coef, eps1_ma_coef, ..., eps<horizon-1>_ma_coef
    """
    insample_daily_df = daily_df

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'eps0_ma', ii, False, 'daily', False) 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['eps0_ma'].ix[horizon].ix['coef']
    print "Coef{}: {}".format(0, coef0)               
    coef_list = list()
    coef_list.append( { 'name': 'eps0_ma_coef', 'coef': coef0 } )
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['eps0_ma'].ix[lag].ix['coef'] 
        print "Coef{}: {}".format(lag, coef)
        coef_list.append( { 'name': 'eps' + str(lag) + '_ma_coef', 'coef': coef } )

    coef_df = pd.DataFrame(coef_list)
    coef_df.to_csv(coeffile)

    return 

def eps_alpha(daily_df, horizon, name, coeffile):
    """
    Generate EPS alpha forecast using pre-fitted coefficients.

    Applies learned regression coefficients to EPS revision signals to produce
    final alpha forecast. This is the production-mode function that uses
    coefficients fitted during backtest/calibration runs.

    Args:
        daily_df: DataFrame with eps0_ma through eps<horizon-1>_ma columns
        horizon: Forecast horizon (must match coefficient file)
        name: Strategy name (for logging, currently unused)
        coeffile: Path to CSV coefficient file from generate_coefs()

    Returns:
        DataFrame: Input df with added 'eps' column containing alpha forecast
            eps = sum(eps[i]_ma * coef[i]) for i in 0 to horizon-1
    """
    print "Loading coeffile: {}".format(coeffile)
    coef_df = pd.read_csv(coeffile, header=0, index_col=['name'])
    outsample_daily_df = daily_df
    outsample_daily_df['eps'] = 0.0

    coef0 = coef_df.ix['eps0_ma_coef'].ix['coef']
    print "Coef{}: {}".format(0, coef0)               
    outsample_daily_df[ 'eps0_ma_coef' ] = coef0
    for lag in range(0,horizon):
        coef = coef_df.ix[ 'eps'+str(lag)+'_ma_coef' ].ix['coef']
        outsample_daily_df[ 'eps'+str(lag)+'_ma_coef' ] = coef

    outsample_daily_df[ 'eps' ] = (outsample_daily_df['eps0_ma'].fillna(0) * outsample_daily_df['eps0_ma_coef']).fillna(0)
    print outsample_daily_df['eps'].describe()
    for lag in range(1,horizon):
        outsample_daily_df[ 'eps'] += (outsample_daily_df['eps'+str(lag)+'_ma'].fillna(0) * outsample_daily_df['eps'+str(lag)+'_ma_coef']).fillna(0)
        print outsample_daily_df['eps'].describe()
    
    return outsample_daily_df

def calc_eps_forecast(daily_df, horizon, coeffile, fit):
    """
    Main orchestrator for EPS alpha generation.

    Coordinates the full workflow for either fitting coefficients (backtest mode)
    or generating production alpha forecasts (production mode).

    Workflow:
        1. Calculate daily EPS signals via calc_eps_daily()
        2a. If fit=True: Add forward returns, fit coefficients, save to file, return None
        2b. If fit=False: Apply coefficients via eps_alpha(), return forecast DataFrame

    Args:
        daily_df: Base DataFrame with price, Barra, and analyst estimate data
        horizon: Forecast horizon in days (typically 10)
        coeffile: Coefficient file path (input for prod, output for fit)
        fit: Boolean - True for fitting mode, False for production mode

    Returns:
        DataFrame with 'eps' column (production mode) or None (fit mode)
    """
    daily_results_df = calc_eps_daily(daily_df, horizon) 


    if fit:
        forwards_df = calc_forward_returns(daily_df, horizon)
        daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

        generate_coefs( daily_results_df, horizon, "all", coeffile)
        return
    else:
        res = eps_alpha( daily_results_df, horizon, "all", coeffile)
        result_df = pd.concat([res], verify_integrity=True)

    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--asof",action="store",dest="asof",default=None)
    parser.add_argument("--inputfile",action="store",dest="inputfile",default=None)
    parser.add_argument("--outputfile",action="store",dest="outputfile",default=None)
    parser.add_argument("--logfile",action="store",dest="logfile",default=None)
    parser.add_argument("--coeffile",action="store",dest="coeffile",default=None)
    parser.add_argument("--fit",action="store",dest="fit",default=False)
    args = parser.parse_args()

    horizon = int(10)
    end = datetime.strptime(args.asof, "%Y%m%d")

    if args.fit:
        print "Fitting..."
        coeffile = args.coeffile + "/" + args.asof + ".eps.csv"
        lookback = timedelta(days=720)    
        start = end - lookback
        uni_df = get_uni(start, end, 30)
    else:
        print "Not fitting..."
        coeffile = args.coeffile
        lookback = timedelta(days=horizon+5)    
        start = end - lookback
        uni_df = load_live_file(args.inputfile)
        end = datetime.strptime(args.asof + '_' + uni_df['time'].min(), '%Y%m%d_%H:%M:%S')
    
    print "Running between {} and {}".format(start, end)

    BARRA_COLS = ['ind1', 'pbeta']
    barra_df = load_barra(uni_df, start, end, BARRA_COLS)
    PRICE_COLS = ['close']
    price_df = load_prices(uni_df, start, end, PRICE_COLS)

    daily_df = merge_barra_data(price_df, barra_df)
    analyst_df = load_estimate_hist(price_df[['ticker']], start, end, "EPS")
    daily_df = merge_daily_calcs(analyst_df, daily_df)

    result_df = calc_eps_forecast(daily_df, horizon, coeffile, args.fit)
    if not args.fit:
        dump_prod_alpha(result_df, 'eps', args.outputfile)



