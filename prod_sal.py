#!/usr/bin/env python
"""Production alpha generator for sell-side analyst estimate revisions.

This module generates trading signals based on changes in analyst estimates,
specifically tracking the "SAL" (Sell-side Analyst Liquidity) estimate type.
The strategy capitalizes on information content in analyst estimate revisions.

**Strategy Logic:**
1. Tracks changes in analyst consensus estimates (SAL_diff_mean)
2. Normalizes estimate changes by median estimate level (SAL_median)
3. Filters for positive standard deviation changes (increasing uncertainty)
4. Fits separate regression models for estimate upgrades vs. downgrades
5. Generates lagged multi-day forecasts using fitted coefficients

**Operating Modes:**
- **Fitting mode** (--fit): Calibrates regression coefficients using 720-day lookback
  - Performs WLS regression of alpha factors against forward returns
  - Separates into "up" (positive revisions) and "dn" (negative revisions) regimes
  - Saves coefficients to CSV for production use

- **Production mode**: Generates alpha signals for live trading
  - Loads pre-fitted coefficients from CSV
  - Applies coefficients to recent estimate data (horizon+5 days)
  - Outputs signals via dump_prod_alpha()

**Signal Construction:**
- Base signal: (SAL_diff_mean / SAL_median) when SAL_std increases
- Beta-adjusted returns used for market neutrality
- Multi-lag structure (0 to horizon-1 days) with declining coefficients
- Separate treatment for positive vs. negative estimate revisions

**Data Requirements:**
- Price and volume data (via loaddata.py)
- Barra risk factors (ind1, pbeta)
- Analyst estimate history from estimates database
- Pre-fitted coefficient open(production mode only)

**Usage Examples:**

Fit coefficients (calibration):
    python prod_sal.py --asof=20130630 --coeffile=./coefs --fit=True

Generate production signals:
    python prod_sal.py --asof=20130701 --inputfile=live_data.csv \\
                       --outputfile=sal_alpha.csv --coeffile=./coefs/20130630.sal.csv

**Key Parameters:**
- horizon: Forecast horizon in days (default: 20)
- lookback: 720 days for fitting, horizon+5 for production
- ESTIMATE: Fixed to "SAL" for analyst estimate tracking

**Output:**
- Fit mode: Coefficient CSV with lagged weights for up/down regimes
- Production mode: Alpha forecast CSV with 'sal' column

**Dependencies:**
- regress.py: WLS regression and coefficient fitting
- load_data_live.py: Live data loading for production mode
- loaddata.py: Historical data loading and estimate history
- util.py: Data merging and output utilities

**Author Notes:**
- Originally designed for analyst estimate momentum/reversal signals
- SAL estimate type may refer to specific data vendor or estimate category
- Beta adjustment ensures market-neutral signal construction
"""

from __future__ import division, print_function

from regress import *
from load_data_live import *
from loaddata import *
from util import *

ESTIMATE = "SAL"  # Estimate type identifier for analyst data queries

def wavg(group):
    """Calculate market-cap weighted beta-adjusted returns for a date group.

    Computes the market return for each date using cap-weighted log returns,
    then multiplies by each stock's predicted beta to get expected return.

    Args:
        group: DataFrame group for single date with columns:
            - pbeta: Predicted beta from Barra
            - log_ret: Log returns
            - mkt_cap_y: Market capitalization
            - gdate: Date identifier

    Returns:
        Series: Beta * market_return for each stock in the group

    Notes:
        - Market cap is scaled to millions for numerical stability
        - Prints market return for monitoring/debugging
        - Used to calculate beta-adjusted returns (actual - expected)
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print("Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum())))
    res = b * ((d * w).sum() / w.sum())
    return res


def calc_sal_daily(daily_df, horizon):
    """Calculate daily SAL (analyst estimate revision) alpha signals.

    Constructs alpha signals from analyst estimate changes, filtering for cases
    where estimate uncertainty is increasing (positive std_diff). Normalizes
    estimate changes by median estimate level and computes multi-lag features.

    Args:
        daily_df: DataFrame with columns:
            - SAL_diff_mean: Change in mean analyst estimate
            - SAL_median: Median analyst estimate
            - SAL_std: Standard deviation of analyst estimates
            - log_ret: Daily log returns
            - pbeta: Predicted beta
            - mkt_cap_y: Market capitalization
            - gdate: Date identifier
        horizon: Forecast horizon in days (default: 20)

    Returns:
        DataFrame with additional columns:
            - bret: Beta-adjusted expected return (beta * market_return)
            - badjret: Beta-adjusted residual return (log_ret - bret)
            - badj0_B: Winsorized beta-adjusted returns
            - cum_ret: Rolling sum of returns over horizon
            - std_diff: Change in estimate standard deviation
            - sal0: Base signal (SAL_diff_mean / SAL_median) when std increases
            - sal0_ma: Moving average of base signal
            - sal1_ma, sal2_ma, ...: Lagged versions of sal0_ma

    Notes:
        - Only generates signals when std_diff > 0 (increasing uncertainty)
        - Sets sal0=0 when estimate std is decreasing (stability implies no signal)
        - Creates horizon lags of the base signal for distributed forecasts
        - Uses filter_expandable() to limit universe
    """
    print("Caculating daily sal...")
    result_df = filter_expandable(daily_df)

    print("Calculating sal0..."    )
    halflife = horizon / 2
#    result_df['dk'] = np.exp( -1.0 * halflife *  (result_df['gdate'] - result_df['last']).astype('timedelta64[D]').astype(int) )

    result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['log_ret'] - result_df['bret']
    result_df['badj0_B'] = winsorize_by_date(result_df[ 'badjret' ])

    result_df['cum_ret'] = result_df['log_ret'].rolling(horizon).sum()

    print(result_df[ESTIMATE + '_diff_mean'].describe())
    result_df['std_diff'] = result_df[ESTIMATE + '_std'].unstack().diff().stack()
    result_df.loc[ result_df['std_diff'] <= 0, ESTIMATE + '_diff_mean'] = 0
    result_df['sal0'] = result_df[ESTIMATE + '_diff_mean'] / result_df[ESTIMATE + '_median']

    # print result_df.columns
    # result_df['sum'] = result_df['SAL_median'] 
    # result_df['det_diff'] = (result_df['sum'].diff())
    # result_df['det_diff_sum'] = pd.rolling_sum( result_df['det_diff'], window=2)
    # #result_df['det_diff_dk'] = ewma(result_df['det_diff'], halflife=horizon )   
    # result_df['sal0'] = result_df['det_diff'] 

    # result_df['median'] = -1.0 * (result_df['median'] - 3)
    # result_df['med_diff'] = result_df['median'].unstack().diff().stack()
    # result_df['med_diff_dk'] = pd.rolling_sum( result_df['dk'] * result_df['med_diff'], window=horizon )
    # result_df['sal0'] = (np.sign(result_df['med_diff_dk']) * np.sign(result_df['cum_ret'])).clip(lower=0) * result_df['med_diff_dk']


    # demean = lambda x: (x - x.mean())
    # indgroups = result_df[['sal0', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    # result_df['sal0_ma'] = indgroups['sal0']

#    result_df['sal0_ma'] = result_df['sal0_ma'] - result_df['sal0_ma'].dropna().mean()

#    result_df['sal0_ma'] = result_df['sal0_ma'] * (np.sign(result_df['sal0_ma']) * np.sign(result_df['cum_ret']))

    result_df['sal0_ma'] = result_df['sal0']

    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['sal'+str(lag)+'_ma'] = shift_df['sal0_ma']

    return result_df

def generate_coefs(daily_df, horizon, name, coeffile=None, intercepts=None):
    """Fit regression coefficients for SAL alpha signal.

    Performs separate WLS regressions for positive and negative estimate revisions,
    fitting lagged coefficients to predict forward returns. Saves coefficients to
    CSV for use in production signal generation.

    Args:
        daily_df: DataFrame with SAL signals and forward returns
        horizon: Forecast horizon in days
        name: Strategy name identifier (typically "all")
        coeffile: Path to output coefficient CSV file
        intercepts: Dict mapping lag -> intercept adjustment values
            Subtracted from fitted intercepts to adjust for bias

    Returns:
        None (writes coefficient CSV to disk)

    Coefficient Structure:
        For each lag (0 to horizon-1) and regime (up/dn):
        - sal{lag}_ma_coef: Slope coefficient for lagged signal
        - sal{lag}_ma_intercept: Intercept term

    Regression Details:
        - Uses regress_alpha() with WLS for heteroskedasticity
        - Fits sal0_ma against forward returns at each horizon
        - Computes incremental coefficients: coef[lag] = coef[horizon] - coef[lag]
        - Separate models for up (positive SAL_diff_mean) and dn (negative/zero)

    Output CSV Format:
        Columns: name, group, coef
        Index: MultiIndex(name, group) where group in ['up', 'dn']

    Notes:
        - Intercept adjustment critical for bias correction
        - Separate up/dn models capture asymmetric estimate revision effects
        - Incremental coefficient structure ensures signals decay with lag
    """
    insample_daily_df = daily_df

    insample_up_df = insample_daily_df[ insample_daily_df[ESTIMATE + "_diff_mean"] > 0 ]
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr', 'intercept'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_up_df, 'sal0_ma', ii, False, 'daily', True) 
        print("INTERCEPT {} {}".format(ii, intercepts[ii]))
        fitresults_df['intercept'] = fitresults_df['intercept'] - float(intercepts[ii])
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    coef0 = fits_df.loc['sal0_ma'].loc[horizon].loc['coef']
    intercept0 = fits_df.loc['sal0_ma'].loc[horizon].loc['intercept']
    coef_list = list()
    coef_list.append( { 'name': 'sal0_ma_coef', 'group': "up", 'coef': coef0 } )
    coef_list.append( { 'name': 'sal0_ma_intercept', 'group': 'up', 'coef': intercept0 } )
    print("Coef{}: {}".format(0, coef0)               )
    for lag in range(1,horizon):
        coef = coef0 - fits_df.loc['sal0_ma'].loc[lag].loc['coef'] 
        intercept = intercept0 - fits_df.loc['sal0_ma'].loc[lag].loc['intercept'] 
        print("Coef{}: {}".format(lag, coef))
        coef_list.append( { 'name': 'sal' + str(lag) + '_ma_coef', 'group': "up", 'coef': coef } )
        coef_list.append( { 'name': 'sal' + str(lag) + '_ma_intercept', 'group': "up", 'coef': intercept } )

    insample_dn_df = insample_daily_df[ insample_daily_df[ESTIMATE + "_diff_mean"] <= 0 ]
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr', 'intercept'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_dn_df, 'sal0_ma', ii, False, 'daily', True) 
        fitresults_df['intercept'] = fitresults_df['intercept'] - intercepts[ii]
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    coef0 = fits_df.loc['sal0_ma'].loc[horizon].loc['coef']
    intercept0 = fits_df.loc['sal0_ma'].loc[horizon].loc['intercept']
    coef_list.append( { 'name': 'sal0_ma_coef', 'group': "dn", 'coef': coef0 } )
    coef_list.append( { 'name': 'sal0_ma_intercept', 'group': 'dn', 'coef': intercept0 } )

    print("Coef{}: {}".format(0, coef0)               )
    for lag in range(1,horizon):
        coef = coef0 - fits_df.loc['sal0_ma'].loc[lag].loc['coef'] 
        intercept = intercept0 - fits_df.loc['sal0_ma'].loc[lag].loc['intercept'] 
        print("Coef{}: {}".format(lag, coef))
        coef_list.append( { 'name': 'sal' + str(lag) + '_ma_coef', 'group': "dn", 'coef': coef } )
        coef_list.append( { 'name': 'sal' + str(lag) + '_ma_intercept', 'group': "dn", 'coef': intercept } )


    coef_df = pd.DataFrame(coef_list)
    coef_df.to_csv(coeffile)

    return

def sal_alpha(daily_df, horizon, name, coeffile):
    """Apply fitted coefficients to generate SAL alpha forecasts.

    Loads pre-fitted regression coefficients and applies them to SAL signals,
    using separate coefficient sets for positive vs. negative estimate revisions.

    Args:
        daily_df: DataFrame with SAL signals (sal0_ma, sal1_ma, ..., sal{horizon-1}_ma)
        horizon: Forecast horizon in days (must match coefficient fitting)
        name: Strategy name identifier (typically "all")
        coeffile: Path to coefficient CSV file from generate_coefs()

    Returns:
        DataFrame with additional 'sal' column containing alpha forecast

    Alpha Calculation:
        For each stock-date:
        1. Select coefficient set based on SAL_diff_mean sign (up vs. dn)
        2. sal = sum over lags: (sal{lag}_ma * coef + intercept)
        3. Lagged signals provide distributed multi-day forecast

    Coefficient Application:
        - Positive SAL_diff_mean: Use "up" regime coefficients
        - Negative/zero SAL_diff_mean: Use "dn" regime coefficients
        - Each lag (0 to horizon-1) contributes to final signal
        - Missing values filled with 0

    Notes:
        - Requires coefficient CSV with MultiIndex(name, group)
        - Separate up/dn treatment captures asymmetric revision effects
        - Output 'sal' column ready for portfolio optimization
    """
    coef_df = pd.read_csv(coeffile, header=0, index_col=['name', 'group'])
    outsample_daily_df = daily_df
    outsample_daily_df['sal'] = 0.0

    coef0 = coef_df.loc['sal0_ma_coef'].loc["up"].loc['coef']
    intercept0 = coef_df.loc['sal0_ma_intercept'].loc["up"].loc['coef']
    print("Coef{}: {}".format(0, coef0)               )
    outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] > 0, 'sal0_ma_coef' ] = coef0
    outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] > 0, 'sal0_ma_intercept' ] =  intercept0
    for lag in range(1,horizon):
        coef = coef_df.loc['sal' + str(lag) + '0_ma_coef'].loc["up"].loc['coef']
        intercept = coef_df.loc['sal' + str(lag) + '_ma_intercept'].loc["up"].loc['coef']
        outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] > 0, 'sal'+str(lag)+'_ma_coef' ] = coef
        outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] > 0, 'sal'+str(lag)+'_ma_intercept' ] = intercept

    coef0 = coef_df.loc['sal0_ma_coef'].loc["dn"].loc['coef']
    intercept0 = coef_df.loc['sal0_ma_intercept'].loc["dn"].loc['coef']
    print("Coef{}: {}".format(0, coef0)               )
    outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] <= 0, 'sal0_ma_coef' ] = coef0
    outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] <= 0, 'sal0_ma_intercept' ] =  intercept0
    for lag in range(1,horizon):
        coef = coef_df.loc['sal' + str(lag) + '0_ma_coef'].loc["dn"].loc['coef']
        intercept = coef_df.loc['sal' + str(lag) + '_ma_intercept'].loc["dn"].loc['coef']
        print("Coef{}: {}".format(lag, coef))
        outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] <= 0, 'sal'+str(lag)+'_ma_coef' ] = coef
        outsample_daily_df.loc[ outsample_daily_df[ESTIMATE + '_diff_mean'] <= 0, 'sal'+str(lag)+'_ma_intercept' ] = intercept


    outsample_daily_df[ 'sal' ] = (outsample_daily_df['sal0_ma'].fillna(0) * outsample_daily_df['sal0_ma_coef'] + outsample_daily_df['sal0_ma_intercept']).fillna(0)
    for lag in range(1,horizon):
        outsample_daily_df[ 'sal'] += (outsample_daily_df['sal'+str(lag)+'_ma'].fillna(0) * outsample_daily_df['sal'+str(lag)+'_ma_coef'] + outsample_daily_df['sal'+str(lag)+'_ma_intercept']).fillna(0)
    
    return outsample_daily_df

def calc_sal_forecast(daily_df, horizon, coeffile, fit):
    """Main entry point for SAL alpha generation (fitting or production).

    Orchestrates the complete SAL alpha workflow: signal calculation,
    coefficient fitting (if requested), and alpha forecast generation.

    Args:
        daily_df: DataFrame with price, Barra, and analyst estimate data
        horizon: Forecast horizon in days (typically 20)
        coeffile: Path to coefficient CSV (input for production, output for fitting)
        fit: Boolean flag:
            - True: Fit mode - calculate coefficients and save to coeffile
            - False: Production mode - apply existing coefficients to generate signals

    Returns:
        - Fit mode (fit=True): None (writes coefficients to disk)
        - Production mode (fit=False): DataFrame with 'sal' alpha column

    Workflow:
        1. Calculate base SAL signals via calc_sal_daily()
        2. If fitting:
           a. Compute forward returns for regression targets
           b. Calculate intercept adjustments via get_intercept()
           c. Fit coefficients via generate_coefs()
           d. Save coefficients to coeffile
        3. If production:
           a. Load coefficients from coeffile
           b. Apply coefficients via sal_alpha()
           c. Return DataFrame ready for portfolio optimization

    Notes:
        - Fit mode used for periodic coefficient recalibration (e.g., monthly)
        - Production mode used for daily signal generation
        - Intercept adjustment critical for bias correction in fitting
        - Output suitable for merge with other alpha signals
    """
    daily_results_df = calc_sal_daily(daily_df, horizon) 

    if fit:
        forwards_df = calc_forward_returns(daily_df, horizon)
        daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

        intercept_d = get_intercept(daily_results_df, horizon, 'sal0_ma')
        generate_coefs( daily_results_df, horizon, "all", coeffile, intercept_d)
        return
    else:
        res = sal_alpha( daily_results_df, horizon, "all", coeffile)
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

    horizon = int(20)
    end = datetime.strptime(args.asof, "%Y%m%d")

    if args.fit:
        print("Fitting...")
        coeffile = args.coeffile + "/" + args.asof + ".sal.csv"
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
    analyst_df = load_estimate_hist(price_df[['ticker']], start, end, "SAL")
    daily_df = merge_daily_calcs(analyst_df, daily_df)

    
    result_df = calc_sal_forecast(daily_df, horizon, coeffile, args.fit)
    if not args.fit:
        dump_prod_alpha(result_df, 'sal', args.outputfile)


