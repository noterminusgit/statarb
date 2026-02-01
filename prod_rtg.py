#!/usr/bin/env python
"""Production alpha generator for sell-side analyst rating revisions.

This module generates trading signals based on changes in analyst stock ratings,
tracking rating upgrades and downgrades to capture momentum in analyst sentiment.

**Strategy Logic:**
1. Tracks changes in analyst consensus ratings (rating_diff_mean)
2. Filters for increasing rating uncertainty (std_diff > 0)
3. Cubes the rating change to amplify strong signals: rtg0 = rating_diff_mean^3
4. Fits regression coefficients using WLS against forward returns
5. Generates lagged multi-day forecasts using fitted coefficients

**Operating Modes:**
- **Fitting mode** (--fit): Calibrates regression coefficients using 720-day lookback
  - Performs WLS regression of alpha factors against forward returns
  - Computes incremental coefficients with linearly declining weights
  - Saves coefficients to CSV for production use

- **Production mode**: Generates alpha signals for live trading
  - Loads pre-fitted coefficients from CSV
  - Applies coefficients to recent rating data (horizon+5 days)
  - Outputs signals via dump_prod_alpha()

**Signal Construction:**
- Base signal: rating_diff_mean^3 when rating_std increases
  - Cubing amplifies strong rating changes while preserving sign
  - Only active when std_diff > 0 (increasing analyst disagreement)
- Multi-lag structure (0 to horizon-1 days) with declining weights
  - Weight for lag k: (horizon - k) / horizon
  - Ensures recent rating changes have stronger impact

**Data Requirements:**
- Price and volume data (via loaddata.py)
- Barra risk factors (ind1 industry classification)
- Analyst rating history from ratings database
- Pre-fitted coefficient file (production mode only)

**Usage Examples:**

Fit coefficients (calibration):
    python prod_rtg.py --asof=20130630 --coeffile=./coefs --fit=True

Generate production signals:
    python prod_rtg.py --asof=20130701 --inputfile=live_data.csv \\
                       --outputfile=rtg_alpha.csv --coeffile=./coefs/20130630.rtg.csv

**Key Parameters:**
- horizon: Forecast horizon in days (default: 6)
- lookback: 720 days for fitting, horizon+5 for production
- Rating scale: Analyst ratings typically on 1-5 scale (1=Strong Buy, 5=Strong Sell)

**Output:**
- Fit mode: Coefficient CSV with lagged weights
- Production mode: Alpha forecast CSV with 'rtg' column

**Dependencies:**
- regress.py: WLS regression and coefficient fitting
- load_data_live.py: Live data loading for production mode
- loaddata.py: Historical data loading and rating history
- util.py: Data merging and output utilities

**Differences from prod_sal.py and prod_eps.py:**
- Uses rating changes instead of estimate revisions
- Single regime (no separate up/dn treatment)
- Cubed signal amplification instead of linear normalization
- Shorter default horizon (6 days vs 20/10)
- Simpler coefficient structure with linear decay weights

**Author Notes:**
- Rating changes often precede price moves due to analyst herding behavior
- Cubing the signal helps distinguish strong conviction changes from noise
- Shorter horizon reflects faster market reaction to rating changes vs. estimates
"""

import logging

from regress import *
from loaddata import *
from load_data_live import *
from util import *

from pandas.stats.moments import ewma

def wavg(group):
    """Calculate market-cap weighted average return scaled by beta.

    Computes the market-cap-weighted average return for a date group and scales
    it by each stock's beta. Used for beta-adjustment calculations, though not
    currently used in the main rating signal (commented out in calc_rtg_daily).

    Args:
        group: DataFrame group for single date with columns:
            - pbeta: Predicted beta from Barra
            - log_ret: Log returns
            - mkt_cap_y: Market capitalization
            - gdate: Date identifier

    Returns:
        Series: Beta values scaled by market-cap-weighted average return

    Notes:
        - Market cap is scaled to millions for numerical stability
        - Prints market return for monitoring/debugging
        - Currently not used in active signal construction
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    print "Mkt return: {} {}".format(group['gdate'], ((d * w).sum() / w.sum()))
    res = b * ((d * w).sum() / w.sum())
    return res

def calc_rtg_daily(daily_df, horizon):
    """Calculate daily rating revision alpha signals with lagged features.

    Generates alpha signals from analyst rating changes, filtering for cases
    where rating uncertainty is increasing (positive std_diff). The signal cubes
    the rating change to amplify strong conviction moves while preserving sign.

    Args:
        daily_df: DataFrame with columns:
            - rating_diff_mean: Change in mean analyst rating
            - rating_std: Standard deviation of analyst ratings
            - Additional columns from loaddata (ticker, gdate, etc.)
        horizon: Forecast horizon in days (default: 6)

    Returns:
        DataFrame with additional columns:
            - std_diff: Change in rating standard deviation
            - rtg0: Base signal (rating_diff_mean^3) when std increases
            - rtg0_ma: Moving average of base signal (currently equals rtg0)
            - rtg1_ma, rtg2_ma, ...: Lagged versions of rtg0_ma

    Signal Construction:
        1. Compute std_diff = change in rating_std
        2. Set rating_diff_mean = 0 when std_diff <= 0 (confidence filter)
        3. rtg0 = rating_diff_mean^3 (cubed to amplify strong signals)
        4. Create horizon lags of rtg0 for distributed forecasts

    Notes:
        - Only generates signals when std_diff > 0 (increasing analyst disagreement)
        - Cubing preserves sign: positive changes stay positive, negative stay negative
        - Commented code shows historical signal variations (median-based, cumulative returns)
        - Uses filter_expandable() to limit universe
        - Industry demeaning commented out (rtg0_ma = rtg0 directly)
    """
    print "Caculating daily rtg..."
    result_df = filter_expandable(daily_df)

    print "Calculating rtg0..."
#    result_df['cum_ret'] = pd.rolling_sum(result_df['log_ret'], 6)
#    result_df['med_diff'] = result_df['median'].unstack().diff().stack()
#    result_df['rtg0'] = -1.0 * (result_df['median'] - 3) / ( 1.0 + result_df['std'] )
#    result_df['rtg0'] = -1 * result_df['mean'] * np.abs(result_df['mean'])
#    result_df['rtg0'] = -1.0 * result_df['med_diff_dk'] * result_df['cum_ret']

    result_df['std_diff'] = result_df['rating_std'].unstack().diff().stack()
    print result_df['rating_diff_mean'].describe()
    result_df.loc[ (result_df['std_diff'] <= 0) | (result_df['std_diff'].isnull()), 'rating_diff_mean'] = 0
    print result_df['rating_diff_mean'].describe()
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

def generate_coefs(daily_df, horizon, fitfile=None):
    """Fit regression coefficients for rating alpha signal.

    Performs WLS regression to fit the rating signal against forward returns,
    then computes lagged coefficients using a linear decay weighting scheme.
    Unlike prod_sal.py and prod_eps.py, this uses simple proportional weights
    rather than incremental differences.

    Args:
        daily_df: DataFrame with rating signals and forward returns
        horizon: Forecast horizon in days (default: 6)
        fitfile: Path to output coefficient CSV file

    Returns:
        int: Always returns 1 (success indicator)

    Coefficient Structure:
        - rtg0_ma_coef: Base coefficient from horizon-day regression
        - rtg{lag}_ma_coef: Weighted coefficient = coef0 * (horizon - lag) / horizon

    Weighting Logic:
        - lag 0: weight = 1.0 (full coefficient)
        - lag 1: weight = (horizon-1)/horizon
        - lag 2: weight = (horizon-2)/horizon
        - ...
        - Ensures more recent rating changes have stronger impact

    Regression Details:
        - Uses regress_alpha() with WLS for heteroskedasticity
        - Fits rtg0_ma against forward returns at each horizon (1 to horizon)
        - Uses horizon-day coefficient as base, scales for shorter lags
        - No separate up/dn regimes (unlike prod_sal.py)
        - No intercept terms (unlike some variations)

    Output CSV Format:
        Columns: name, coef
        Rows: rtg0_ma_coef, rtg1_ma_coef, ..., rtg{horizon-1}_ma_coef

    Notes:
        - Simpler than prod_sal/prod_eps which use incremental coefficients
        - Linear decay reflects diminishing relevance of older rating changes
        - Prints weights during fitting for monitoring
    """
    insample_daily_df = daily_df

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'rtg0_ma', ii, True, 'daily', False)
        fits_df = fits_df.append(fitresults_df, ignore_index=True)

    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    coef0 = fits_df.ix['rtg0_ma'].ix[horizon].ix['coef']
    print "Coef{}: {}".format(0, coef0)

    coef_list = list()
    coef_list.append( { 'name': 'rtg0_ma_coef', 'coef': coef0 } )
    for lag in range(1,horizon):
        weight = (horizon - lag) / float(horizon)
        lagname = 'rtg'+str(lag)+'_ma'
        coef = coef0 * weight
        print "Running lag {} with weight: {}".format(lag, weight)
        coef_list.append( { 'name': 'rtg'+str(lag)+'_ma_coef', 'coef': coef } )

    coef_df = pd.DataFrame(coef_list)
    coef_df.to_csv(fitfile)

    return 1

def rtg_alpha(daily_df, horizon, coeffile=None):
    """Apply fitted coefficients to generate rating alpha forecasts.

    Loads pre-fitted regression coefficients and applies them to rating signals
    to produce final alpha forecast. Uses a single coefficient set (no regime
    separation like prod_sal.py).

    Args:
        daily_df: DataFrame with rating signals (rtg0_ma, rtg1_ma, ..., rtg{horizon-1}_ma)
        horizon: Forecast horizon in days (must match coefficient fitting)
        coeffile: Path to coefficient CSV file from generate_coefs()

    Returns:
        DataFrame with additional 'rtg' column containing alpha forecast

    Alpha Calculation:
        rtg = sum over lags: (rtg{lag}_ma * coef{lag})
        where lags range from 0 to horizon-1

    Coefficient Application:
        - Load all coefficients from CSV
        - Apply same coefficients to all stocks (no up/dn split)
        - Each lag (0 to horizon-1) contributes to final signal
        - Missing values filled with 0

    Notes:
        - Simpler than prod_sal.py which has separate up/dn regimes
        - No intercept terms (commented out code shows where they would go)
        - Prints coefficient values during application for monitoring
        - Prints incremental rtg statistics for debugging
        - Output 'rtg' column ready for portfolio optimization
    """
    coef_df = pd.read_csv(coeffile, header=0, index_col=['name'])

    outsample_daily_df = daily_df
    outsample_daily_df['rtg'] = 0

    for lag in range(0,horizon):
        coef = coef_df.ix[ 'rtg'+str(lag)+'_ma_coef' ]['coef']
        print "Coef: {}".format(coef)
        outsample_daily_df[ 'rtg'+str(lag)+'_ma_coef' ] = coef
    print outsample_daily_df['rtg'].describe()

    outsample_daily_df[ 'rtg' ] = (outsample_daily_df['rtg0_ma'] * outsample_daily_df['rtg0_ma_coef']).fillna(0) #+ outsample_daily_df['rtg0_ma_intercept']
    for lag in range(1,horizon):
        print outsample_daily_df['rtg'].describe()
        outsample_daily_df[ 'rtg'] += (outsample_daily_df['rtg'+str(lag)+'_ma'] * outsample_daily_df['rtg'+str(lag)+'_ma_coef']).fillna(0) #+ outsample_daily_df['rtg'+str(lag)+'_ma_intercept']

    return outsample_daily_df
def calc_rtg_forecast(daily_df, horizon, coeffile, fit):
    """Main entry point for rating alpha generation (fitting or production).

    Orchestrates the complete rating alpha workflow: signal calculation,
    coefficient fitting (if requested), and alpha forecast generation.

    Args:
        daily_df: DataFrame with price, Barra, and analyst rating data
        horizon: Forecast horizon in days (default: 6)
        coeffile: Path to coefficient CSV (input for production, output for fitting)
        fit: Boolean flag:
            - True: Fit mode - calculate coefficients and save to coeffile
            - False: Production mode - apply existing coefficients to generate signals

    Returns:
        - Fit mode (fit=True): None (writes coefficients to disk)
        - Production mode (fit=False): DataFrame with 'rtg' alpha column

    Workflow:
        1. Calculate base rating signals via calc_rtg_daily()
        2. If fitting:
           a. Compute forward returns for regression targets
           b. Fit coefficients via generate_coefs()
           c. Save coefficients to coeffile
           d. Return None
        3. If production:
           a. Load coefficients from coeffile
           b. Apply coefficients via rtg_alpha()
           c. Return DataFrame ready for portfolio optimization

    Notes:
        - Fit mode used for periodic coefficient recalibration (e.g., monthly)
        - Production mode used for daily signal generation
        - Commented code shows historical sector-based and regime-based variations
        - No intercept adjustment needed (unlike prod_sal.py)
        - Output suitable for merge with other alpha signals
    """
    daily_results_df = calc_rtg_daily(daily_df, horizon)

    # results = list()
    # for sector_name in daily_results_df['sector_name'].dropna().unique():
    #     if sector_name == "Utilities" or sector_name == "HealthCare": continue
    #     print "Running rtg for sector {}".format(sector_name)
    #     sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    #     result_df = rtg_fits(sector_df, horizon, sector_name, middate)
    #     results.append(result_df)
    # result_df = pd.concat(results, verify_integrity=True)


    # res1 = rtg_fits( daily_results_df[ daily_results_df['rating_diff_mean'] > 0 ], horizon, "up", middate)
    # res2 = rtg_fits( daily_results_df[ daily_results_df['rating_diff_mean'] < 0 ], horizon, "dn", middate)
    # result_df = pd.concat([res1, res2], verify_integrity=True)

    if fit:
        forwards_df = calc_forward_returns(daily_df, horizon)
        daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

        generate_coefs( daily_results_df, horizon, coeffile)
        return
    else:
        res1 = rtg_alpha( daily_results_df, horizon, coeffile)
        #    res2 = tgt_fits( daily_results_df[ daily_results_df['det_diff'] < 0 ], horizon, "dn", middate, intercept_d)
        result_df = pd.concat([res1], verify_integrity=True)

    return result_df

if __name__=="__main__":
    """Command-line interface for rating alpha production.

    CLI Parameters:
        --asof: Date for production run in YYYYMMDD format (required)
        --inputfile: Live data CSV file (required for production mode)
        --outputfile: Alpha output CSV file (required for production mode)
        --logfile: Log file path (currently unused)
        --coeffile: Coefficient file path or directory:
            - Fit mode: Directory where <asof>.rtg.csv will be saved
            - Production mode: Full path to coefficient CSV file
        --fit: Enable fit mode (default: False)

    Mode-Specific Behavior:

    Fit Mode (--fit=True):
        - Lookback: 720 days from asof date
        - Universe: Top 1400 stocks via get_uni(start, end, 30)
        - Output: Coefficient CSV at <coeffile>/<asof>.rtg.csv
        - No alpha forecast generated

    Production Mode (--fit=False):
        - Lookback: horizon+5 days (11 days total)
        - Universe: Loaded from --inputfile via load_live_file()
        - End time: Adjusted to first timestamp in live data
        - Output: Alpha forecast CSV via dump_prod_alpha()
        - Prints alpha summary statistics

    Data Pipeline:
        1. Load Barra factors: ind1 (industry classification)
        2. Load price data: close prices only
        3. Merge Barra and price data
        4. Load analyst rating history via load_ratings_hist()
        5. Merge rating data with daily DataFrame
        6. Generate forecast via calc_rtg_forecast()
        7. Write output (fit or production mode)

    Example Commands:
        Fit coefficients:
            python prod_rtg.py --asof=20130630 --coeffile=./coefs --fit=True

        Generate production signals:
            python prod_rtg.py --asof=20130701 --inputfile=live_data.csv \\
                               --outputfile=rtg_alpha.csv --coeffile=./coefs/20130630.rtg.csv

    Notes:
        - Horizon fixed at 6 days (shorter than prod_sal/prod_eps)
        - Does NOT load pbeta (unlike prod_sal.py and prod_eps.py)
        - Rating data from load_ratings_hist() (not load_estimate_hist())
        - Output column name: 'rtg' (for portfolio optimization merging)
    """
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--asof",action="store",dest="asof",default=None)
    parser.add_argument("--inputfile",action="store",dest="inputfile",default=None)
    parser.add_argument("--outputfile",action="store",dest="outputfile",default=None)
    parser.add_argument("--logfile",action="store",dest="logfile",default=None)
    parser.add_argument("--coeffile",action="store",dest="coeffile",default=None)
    parser.add_argument("--fit",action="store",dest="fit",default=False)
    args = parser.parse_args()

    horizon = int(6)

    end = datetime.strptime(args.asof, "%Y%m%d")
    if args.fit:
        print "Fitting..."
        coeffile = args.coeffile + "/" + args.asof + ".rtg.csv"
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

    BARRA_COLS = ['ind1']
    barra_df = load_barra(uni_df, start, end, BARRA_COLS)

    PRICE_COLS = ['close']
    price_df = load_prices(uni_df, start, end, PRICE_COLS)

    daily_df = merge_barra_data(price_df, barra_df)
    analyst_df = load_ratings_hist(price_df[['ticker']], start, end)
    daily_df = merge_daily_calcs(analyst_df, daily_df)


    result_df = calc_rtg_forecast(daily_df, horizon, coeffile, args.fit)

    if not args.fit:
        print "Total Alpha Summary"
        print result_df['rtg'].describe()
        dump_prod_alpha(result_df, 'rtg', args.outputfile)


