"""
Regression Analysis Module for Salamander (Python 3)

This module performs weighted least squares (WLS) regression to fit alpha factors
against forward returns and extract predictive coefficients for statistical arbitrage.

Key Features:
    - Weighted Least Squares (WLS) regression with market cap weighting
    - Multiple horizon regression (1-N days ahead)
    - Outlier handling via winsorization
    - Cross-sectional regression by date
    - Statistical diagnostics (t-stats, standard errors)
    - Visualization of regression results
    - Out-of-sample validation via median regression

Regression Types:
    - Daily: Cross-sectional regression of daily alphas vs forward returns
    - Intraday: Time-slice regression for intraday signals (30-min bars)
    - Intraday EOD: Intraday signals vs end-of-day returns
    - Day-of-Week: Separate regressions by day of week to detect calendar effects

Python 3 Features:
    - Modern pandas syntax (no .ix, uses .loc/.iloc)
    - Print functions with parentheses
    - Compatible with statsmodels 0.9+
    - Proper exception handling

Weighting Methodology:
    - Weights = mdvp^ADV_POWER (default: 0.5)
    - mdvp = market cap / median dollar volume product
    - Balances small-cap vs large-cap influence
    - Reduces heteroskedasticity from varying market cap

Statistical Approach:
    1. Extract alpha factor and forward return columns
    2. Drop NaN and infinite values
    3. Apply ADV-based weighting (mdvp^0.5)
    4. Winsorize returns by date (handle extreme days)
    5. Convert log returns to simple returns
    6. Winsorize alpha factors (handle outliers)
    7. Fit WLS regression with or without intercept
    8. Extract coefficients, t-stats, standard errors

Functions:
    plot_fit(): Visualize coefficient decay across horizons
    extract_results(): Extract statistics from statsmodels regression
    get_intercept(): Extract intercepts across multiple horizons
    regress_alpha(): Main regression dispatcher with out-of-sample validation
    regress_alpha_daily(): Cross-sectional daily regression
    regress_alpha_intra(): Intraday forward-looking regression
    regress_alpha_intra_eod(): Intraday alpha vs EOD returns
    regress_alpha_dow(): Day-of-week stratified regression

Constants:
    ADV_POWER: Power parameter for ADV-based weighting (default: 0.5)

Dependencies:
    - calc: Winsorization functions (winsorize, winsorize_by_date, winsorize_by_ts)
    - statsmodels: WLS regression with robust statistics
    - matplotlib: Coefficient visualization
    - pandas: DataFrame operations and time-series handling

Usage Example:
    # Daily regression for high-low alpha at 3-day horizon
    >>> daily_df = load_daily_data()
    >>> results = regress_alpha_daily(daily_df, 'hl', horizon=3)
    >>> print(results[['coef', 'tstat', 'nobs']])

    # Out-of-sample validation with 3-fold split
    >>> results = regress_alpha(daily_df, 'hl', 3, median=True, rtype='daily')

    # Intraday regression with 3-bar horizon (90 minutes)
    >>> intra_df = load_intraday_data()
    >>> results = regress_alpha_intra(intra_df, 'qhl_intra', horizon=3)

Differences from Main regress.py:
    - Python 3 compatible (print functions, modern pandas)
    - Uses .loc/.iloc instead of deprecated .ix
    - Updated pandas rolling functions (pd.rolling_sum deprecated)
    - Compatible with newer statsmodels versions

Note:
    Results include coefficients, t-statistics, and standard errors for
    alpha signal calibration in portfolio optimization.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from calc import *

ADV_POWER = 1 / 2


def plot_fit(fits_df, name):
    """
    Visualize regression coefficients across multiple horizons.

    Creates an error bar plot showing how alpha coefficients and intercepts
    vary across different forecast horizons (typically 1-5 days). Useful for
    diagnosing alpha decay and coefficient stability over time.

    Args:
        fits_df (DataFrame): Regression results with columns:
            - horizon (int): Forecast horizon in days
            - coef (float): Alpha coefficient
            - stderr (float): Standard error of coefficient
            - intercept (float): Regression intercept
        name (str): Output filename prefix (saves as name.png)

    Output:
        Saves PNG plot showing:
        - Blue points: Coefficients with 2-sigma error bars (95% confidence)
        - Red points: Intercepts
        - Horizontal line at y=0 for reference

    Use Case:
        Diagnostic tool to assess:
        - Alpha decay: Do coefficients decrease with horizon?
        - Stability: Are error bars small and consistent?
        - Bias: Are intercepts close to zero?

    Example:
        >>> fits_df = regress_multiple_horizons(daily_df, 'hl', max_horizon=5)
        >>> plot_fit(fits_df, 'hl_decay')
        # Saves hl_decay.png with coefficient decay visualization
    """
    print("Plotting fits...")
    plt.figure()
    plt.xlim(0, fits_df.horizon.max() + 1)
    plt.errorbar(fits_df.horizon, fits_df.coef, yerr=fits_df.stderr * 2, fmt='o')
    plt.errorbar(fits_df.horizon, fits_df.intercept, yerr=fits_df.stderr * 0, fmt='o', color='red')
    plt.axhline(0, color='black')
    plt.savefig(name + ".png")


def extract_results(results, indep, horizon):
    """
    Extract key statistics from a statsmodels regression result.

    Converts statsmodels WLS regression output into a standardized DataFrame
    format with coefficient, standard error, t-statistic, and intercept.
    Handles both intercept and no-intercept regressions automatically.

    Args:
        results (RegressionResults): Fitted statsmodels WLS model object
        indep (str): Name of independent variable (alpha factor)
        horizon (int): Forecast horizon in days (or timeslice index for intraday)

    Returns:
        DataFrame: Single-row DataFrame with columns:
            - indep (str): Independent variable name
            - horizon (int): Forecast horizon
            - nobs (int): Number of observations in fit
            - coef (float): Alpha coefficient estimate
            - stderr (float): Standard error of coefficient
            - tstat (float): T-statistic for coefficient
            - intercept (float): Regression intercept (0 if no-intercept model)

    Logic:
        - If len(params) > 1: Intercept model, params[0]=intercept, params[1]=coef
        - If len(params) == 1: No-intercept model, params[0]=coef, intercept=0

    Example:
        >>> wls_result = sm.WLS(y, X, weights=w).fit()
        >>> df = extract_results(wls_result, 'hl', horizon=3)
        >>> print(df[['coef', 'tstat', 'nobs']])
    """
    ret = dict()
    ret['indep'] = [indep]
    ret['horizon'] = [horizon]
    ret['nobs'] = [results.nobs]
    if len(results.params) > 1:
        ret['coef'] = [results.params[1]]
        ret['stderr'] = [results.bse[1]]
        ret['tstat'] = [results.tvalues[1]]
        ret['intercept'] = [results.params[0]]
    else:
        ret['coef'] = [results.params[0]]
        ret['stderr'] = [results.bse[0]]
        ret['tstat'] = [results.tvalues[0]]
        ret['intercept'] = [0]

    return pd.DataFrame(ret)


def get_intercept(daily_df, horizon, name, middate=None):
    """
    Extract regression intercepts across multiple horizons for time-series analysis.

    Fits alpha regressions for horizons 1 through max horizon using median regression
    (out-of-sample 3-fold splits) and extracts intercept values. Useful for detecting
    systematic biases or drift in forward returns that are not explained by the alpha.

    Args:
        daily_df (DataFrame): Daily results indexed by (date, sid) with columns:
            - cum_ret{N}: Cumulative log returns for various horizons
            - {name}: Alpha factor values
            - mdvp: Market cap weighting factor
        horizon (int): Maximum forecast horizon (fits 1 to horizon)
        name (str): Alpha factor column name
        middate (datetime, optional): If specified, use only data before this
            date for in-sample fitting (for out-of-sample validation)

    Returns:
        dict: Mapping from horizon (int) to intercept (float)
            {1: intercept_1day, 2: intercept_2day, ..., horizon: intercept_Nday}

    Interpretation:
        Non-zero intercepts may indicate:
        - Systematic market drift not captured by alpha
        - Alpha miscalibration or bias
        - Missing risk factors
        - Need for intercept correction in portfolio construction

    Example:
        >>> intercepts = get_intercept(daily_df, horizon=5, name='hl')
        >>> print(intercepts)
        {1: 0.0002, 2: 0.0003, 3: 0.0004, 4: 0.0005, 5: 0.0006}
        # Positive intercepts suggest upward drift not explained by hl

    Note:
        Uses median=True in regress_alpha() for out-of-sample validation,
        which provides more robust estimates than in-sample fitting.
    """
    insample_daily_df = daily_df
    if middate is not None:
        insample_daily_df = daily_df[daily_df.index.get_level_values('date') < middate]

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr', 'intercept'])
    for ii in range(1, horizon + 1):
        fitresults_df = regress_alpha(insample_daily_df, name, ii, True, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    result = dict()
    for ii in range(1, horizon + 1):
        result[ii] = float(fits_df.loc[name].loc[ii].loc['intercept'])

    return result


def regress_alpha(results_df, indep, horizon, median=False, rtype='daily', intercept=True, start=None, end=None):
    """
    Main regression function to fit alpha factors against forward returns.

    Dispatches to specialized regression functions based on regression type.
    Supports out-of-sample validation via median regression (3-fold split),
    which provides more robust coefficient estimates by reducing overfitting.

    Args:
        results_df (DataFrame): Results with alpha factors and forward returns:
            - Daily: indexed by (date, sid)
            - Intraday: indexed by (date, time, sid)
        indep (str): Independent variable (alpha factor column name)
        horizon (int): Forecast horizon in days (daily) or bars (intraday)
        median (bool): If True, perform 3-fold out-of-sample validation and
            return median coefficients. If False, fit on full dataset.
            Default: False
        rtype (str): Regression type, one of:
            - 'daily': Cross-sectional daily regression
            - 'intra': Intraday time-slice regression with forward horizon
            - 'dow': Day-of-week specific regression
            - 'intra_eod': Intraday regression vs EOD returns
            Default: 'daily'
        intercept (bool): Include intercept in regression. Default: True
        start (str, optional): Start date for regression (YYYYMMDD format)
        end (str, optional): End date for regression (YYYYMMDD format)

    Returns:
        DataFrame: Regression results with columns:
            - indep (str): Independent variable name
            - horizon (int): Forecast horizon
            - coef (float): Alpha coefficient
            - stderr (float): Standard error
            - tstat (float): T-statistic
            - nobs (int): Number of observations
            - intercept (float): Regression intercept

    Out-of-Sample Validation (median=True):
        1. Split data into 3 equal temporal windows
        2. Fit regression separately on each window
        3. Return median coefficients across 3 fits
        4. Reduces overfitting and provides robust estimates

    Example:
        # In-sample daily regression
        >>> results = regress_alpha(daily_df, 'hl', 3, rtype='daily')

        # Out-of-sample validation
        >>> results = regress_alpha(daily_df, 'hl', 3, median=True, rtype='daily')

        # Intraday regression with 3-bar horizon
        >>> results = regress_alpha(intra_df, 'qhl_intra', 3, rtype='intra')

        # Day-of-week analysis
        >>> results = regress_alpha(daily_df, 'hl', 1, rtype='dow')

    Note:
        median=True implements walk-forward out-of-sample validation, which is
        critical for assessing true predictive power and avoiding overfitting.
    """
    if start is not None and end is not None:
        print("restrict fit from {} to {}".format(start, end))
        results_df = results_df.truncate(before=dateparser.parse(start), after=dateparser.parse(end))

    if median:
        medians_df = pd.DataFrame(columns=['indep', 'horizon', 'coef', 'stderr', 'tstat', 'nobs', 'intercept'],
                                  dtype=float)
        start = 0
        cnt = len(results_df)
        window = int(cnt / 3)
        end = window
        while end <= cnt:
            print("Looking at rows {} to {} out of {}".format(start, end, cnt))
            timeslice_df = results_df.iloc[start:end]
            if rtype == 'intra_eod':
                fitresults_df = regress_alpha_intra_eod(timeslice_df, indep)
            elif rtype == 'daily':
                fitresults_df = regress_alpha_daily(timeslice_df, indep, horizon, intercept)
            elif rtype == 'dow':
                fitresults_df = regress_alpha_dow(timeslice_df, indep, horizon)
            elif rtype == 'intra':
                fitresults_df = regress_alpha_intra(timeslice_df, indep, horizon)
            else:
                raise "Bad regression type: {}".format(rtype)

            medians_df = medians_df.append(fitresults_df)
            start += window
            end += window

        print("Out of sample coefficients:")
        print(medians_df)
        ret = medians_df.groupby(['indep', 'horizon']).median().reset_index()
        return ret
    else:
        timeslice_df = results_df
        if rtype == 'intra':
            return regress_alpha_intra(timeslice_df, indep, horizon)
        elif rtype == 'daily':
            return regress_alpha_daily(timeslice_df, indep, horizon, intercept)
        elif rtype == 'dow':
            return regress_alpha_dow(timeslice_df, indep, horizon)


def regress_alpha_daily(daily_df, indep, horizon, intercept=True):
    """
    Cross-sectional daily regression of alpha factor vs forward returns.

    Fits weighted least squares regression to predict horizon-day forward returns
    from alpha factor values. Uses market cap weighting (mdvp^0.5) to balance
    small/large cap influence. Winsorizes both returns and alpha to handle outliers.

    Args:
        daily_df (DataFrame): Daily data indexed by (date, sid) with columns:
            - cum_ret{horizon}: Cumulative log return over horizon days
            - mdvp: Market cap / median dollar volume product
            - {indep}: Alpha factor values
        indep (str): Alpha factor column name
        horizon (int): Forward return horizon in days (typically 1-5)
        intercept (bool): Include intercept term in regression. Default: True

    Returns:
        DataFrame: Single-row result with columns:
            - indep: Alpha factor name
            - horizon: Forecast horizon
            - coef: Alpha coefficient estimate
            - stderr: Standard error
            - tstat: T-statistic
            - nobs: Number of observations
            - intercept: Regression intercept (or 0 if intercept=False)

    Methodology:
        1. Extract {indep}, cum_ret{horizon}, mdvp columns
        2. Replace infinite values with NaN and drop missing data
        3. Set weights = mdvp^ADV_POWER (default: mdvp^0.5)
           - Balances small-cap vs large-cap influence
           - Reduces heteroskedasticity
        4. Winsorize returns by date (handle extreme days)
        5. Convert log returns to simple returns: exp(log_ret) - 1
        6. Winsorize alpha factor (handle outlier stocks)
        7. Add constant if intercept=True
        8. Fit WLS: simple_returns ~ alpha + intercept
        9. Extract and return statistics

    Weighting Rationale:
        - mdvp^0.5 weighting prevents large caps from dominating
        - Equivalent to heteroskedasticity correction
        - Balances statistical power across market cap spectrum

    Example:
        # Fit hl (high-low) alpha for 3-day returns
        >>> results = regress_alpha_daily(daily_df, 'hl', horizon=3)
        >>> print("Coefficient: {:.4f}, T-stat: {:.2f}".format(
        ...     results['coef'].iloc[0], results['tstat'].iloc[0]))

        # No-intercept regression for pure alpha
        >>> results = regress_alpha_daily(daily_df, 'hl', 3, intercept=False)

    Note:
        Positive coefficient indicates alpha predicts returns in same direction.
        T-stat > 2 suggests statistical significance at 95% confidence level.
    """
    print("Regressing alphas daily for {} with horizon {}...".format(indep, horizon))
    retname = 'cum_ret' + str(horizon)

    fitdata_df = daily_df[[retname, 'mdvp', indep]]
    #    print(fitdata_df.tail())
    fitdata_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fitdata_df = fitdata_df.dropna()

    weights = fitdata_df['mdvp'] ** ADV_POWER
    ys = winsorize_by_date(fitdata_df[retname])
    ys = np.exp(ys) - 1
    xs = winsorize(fitdata_df[indep])
    if intercept:
        xs = sm.add_constant(xs)
    results_wls = sm.WLS(ys, xs, weights=weights).fit()
    print(results_wls.summary())
    results_df = extract_results(results_wls, indep, horizon)
    return results_df


def regress_alpha_intra_eod(intra_df, indep):
    """
    Intraday alpha regression predicting end-of-day returns from intraday signals.

    For each hourly timeslice (10:00-15:00), fits alpha factor observed at that
    time against the simple return from market open to current bar close. Tests
    whether intraday signals predict accumulated returns during the trading day.

    Args:
        intra_df (DataFrame): Intraday bar data indexed by (date, time, sid):
            - log_ret: Log return for this bar
            - {indep}: Alpha factor value
            - mdvp: Market cap weighting factor
            - close: Current bar close price
            - iclose: Initial (opening) price for the day
        indep (str): Alpha factor column name

    Returns:
        DataFrame: 6-row result (one per timeslice) with columns:
            - horizon (int): Timeslice index (1=10:00, 2=11:00, ..., 6=15:00)
            - coef (float): Alpha coefficient
            - stderr (float): Standard error
            - tstat (float): T-statistic
            - nobs (int): Number of observations
            - intercept (float): Regression intercept

    Methodology:
        For each hourly timeslice (10:00, 11:00, 12:00, 13:00, 14:00, 15:00):
        1. Extract bars at that specific time across all dates
        2. Calculate day_ret = (close - iclose) / iclose
           - Measures cumulative return from open to current time
        3. Winsorize day_ret to handle outliers
        4. Set weights = mdvp^ADV_POWER (default: 0.5)
        5. Fit WLS: day_ret ~ alpha + constant
        6. Extract and store results

    Use Case:
        Diagnostic tool to assess:
        - Do intraday alphas predict cumulative intraday returns?
        - Or do they only predict bar-to-bar changes?
        - How does predictive power evolve through trading day?

    Example:
        >>> results = regress_alpha_intra_eod(intra_df, 'qhl_intra')
        >>> print(results[['horizon', 'coef', 'tstat']])
        # Shows how qhl_intra coefficient varies 10:00-15:00

    Note:
        Compare with regress_alpha_intra() which uses forward-looking horizons
        instead of cumulative from-open returns.
    """
    print("Regressing intra alphas for {} on EOD...".format(indep))
    results_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'], dtype=float)
    fitdata_df = intra_df[['log_ret', indep, 'mdvp', 'close', 'iclose']]
    fitdata_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fitdata_df = fitdata_df.dropna()

    it = 1
    for timeslice in ['10:00', '11:00', '12:00', '13:00', '14:00', '15:00']:
        print("Fitting for timeslice: {}".format(timeslice))

        timeslice_df = fitdata_df.unstack().between_time(timeslice, timeslice).stack()
        timeslice_df['day_ret'] = (timeslice_df['close'] - timeslice_df['iclose']) / timeslice_df['iclose']
        #       timeslice_df['day_ret'] = np.log(timeslice_df['close'] / timeslice_df['iclose'])

        weights = np.sqrt(timeslice_df['mdvp'])
        weights = timeslice_df['mdvp'] ** ADV_POWER
        results_wls = sm.WLS(winsorize(timeslice_df['day_ret']), sm.add_constant(timeslice_df[indep]),
                             weights=weights).fit()
        print(results_wls.summary())
        results_df = results_df.append(extract_results(results_wls, indep, it), ignore_index=True)

        it += 1

    return results_df


def regress_alpha_intra(intra_df, indep, horizon):
    """
    Intraday forward-looking regression with multiple bar horizon.

    For each 30-minute timeslice, fits alpha observed at time T against the
    cumulative return from market open through T+horizon bars. Tests whether
    intraday signals predict forward intraday returns over multiple 30-min bars.

    Args:
        intra_df (DataFrame): Intraday bar data indexed by (date, time, sid):
            - log_ret: Log return for each bar
            - {indep}: Alpha factor value
            - mdvp: Market cap weighting factor
            - close: Current bar close price
            - iclose: Initial (opening) price for the day
        indep (str): Alpha factor column name
        horizon (int): Number of 30-minute bars to look ahead
            - 1 bar = 30 minutes
            - 3 bars = 90 minutes
            - 6 bars = 3 hours

    Returns:
        DataFrame: 6-row result (one per timeslice) with columns:
            - horizon (int): Timeslice index (1-6 for 10:30-15:30)
            - coef (float): Alpha coefficient
            - stderr (float): Standard error
            - tstat (float): T-statistic
            - nobs (int): Number of observations
            - intercept (float): Regression intercept

    Methodology:
        For each 30-minute timeslice (10:30, 11:30, 12:30, 13:30, 14:30, 15:30):
        1. Extract bars at that specific time across all dates
        2. Shift log_ret forward by horizon bars using shift(-horizon)
        3. Sum shifted log returns over horizon window:
           cum_ret = sum of next {horizon} bar log returns
        4. Calculate total return from open:
           day_ret = exp(log(close/iclose) + cum_ret) - 1
           - Combines return to current bar + forward return
        5. Winsorize day_ret by timeslice (handle outlier days at each time)
        6. Set weights = mdvp^ADV_POWER (default: 0.5)
        7. Fit WLS: day_ret ~ alpha + constant
        8. Extract and store results

    Example:
        # Test if 10:30 alpha predicts cumulative return through 12:00
        >>> results = regress_alpha_intra(intra_df, 'qhl_intra', horizon=3)
        >>> print(results.loc[results.horizon == 1, ['coef', 'tstat']])

        # 1-bar ahead prediction (30 minutes)
        >>> results = regress_alpha_intra(intra_df, 'qhl_intra', horizon=1)

    Horizon Interpretation:
        horizon=1: Predicts next 30-min bar
        horizon=3: Predicts cumulative return over next 90 minutes
        horizon=6: Predicts rest of trading day (if called at 10:30)

    Note:
        - Uses winsorize_by_ts() which winsorizes separately for each timeslice
        - pd.rolling_sum() may need update to .rolling().sum() in newer pandas
        - Compare with regress_alpha_intra_eod() for cumulative-from-open prediction
    """
    print("Regressing intra alphas for {} on horizon {}...".format(indep, horizon))
    assert horizon > 0
    results_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'], dtype=float)
    retname = 'cum_ret' + str(horizon)
    fitdata_df = intra_df[['log_ret', indep, 'mdvp', 'close', 'iclose']]
    fitdata_df[retname] = np.nan
    fitdata_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    it = 1
    for timeslice in ['10:30', '11:30', '12:30', '13:30', '14:30', '15:30']:
        print("Fitting for timeslice: {} at horizon {}".format(timeslice, horizon))

        timeslice_df = fitdata_df.unstack().between_time(timeslice, timeslice).stack()
        shift_df = timeslice_df.unstack().shift(-horizon).stack()
        timeslice_df[retname] = shift_df['log_ret'].groupby(level='sid').apply(lambda x: x.rolling(horizon).sum())
        #        intra_df.loc[ timeslice_df.index, retname ] = timeslice_df[retname]
        timeslice_df['day_ret'] = np.exp(
            np.log(timeslice_df['close'] / timeslice_df['iclose']) + timeslice_df[retname]) - 1
        timeslice_df = timeslice_df.dropna()

        weights = np.sqrt(timeslice_df['mdvp'])
        weights = timeslice_df['mdvp'] ** ADV_POWER
        ys = winsorize_by_ts(timeslice_df['day_ret'])
        results_wls = sm.WLS(ys, sm.add_constant(timeslice_df[indep]), weights=weights).fit()
        print(results_wls.summary())
        results_df = results_df.append(extract_results(results_wls, indep, it), ignore_index=True)
        it += 1

    return results_df


def regress_alpha_dow(daily_df, indep, horizon):
    """
    Day-of-week stratified regression to detect calendar effects.

    Fits separate regressions for each day of the week (Monday-Friday) to test
    whether alpha coefficients vary by day. Useful for detecting calendar effects
    and optimizing rebalancing schedules.

    Args:
        daily_df (DataFrame): Daily data indexed by (date, sid) with columns:
            - cum_ret{horizon}: Cumulative log return over horizon days
            - mdvp: Market cap weighting factor
            - {indep}: Alpha factor values
            - dow: Day of week (0=Monday, 1=Tuesday, 2=Wed, 3=Thu, 4=Friday)
        indep (str): Alpha factor column name
        horizon (int): Forward return horizon in days

    Returns:
        DataFrame: 5-row result (one per weekday) with columns:
            - horizon (int): Encoded as horizon*10 + dow
              (e.g., 30=3-day Mon, 31=3-day Tue, ..., 34=3-day Fri)
            - coef (float): Alpha coefficient for this day
            - stderr (float): Standard error
            - tstat (float): T-statistic
            - nobs (int): Number of observations
            - intercept (float): Regression intercept

    Methodology:
        1. Group data by day of week (dow column: 0-4)
        2. For each day, fit separate WLS regression:
           - Weight by mdvp^ADV_POWER (default: 0.5)
           - Winsorize returns by date within each day group
           - Fit: returns ~ alpha + constant
        3. Encode results with horizon*10 + dow for identification

    Use Cases:
        - Detect day-of-week effects:
          - Monday reversal (coefficient more negative)
          - Friday momentum (coefficient more positive)
        - Optimal rebalancing schedule:
          - Rebalance on days with strongest alpha signal
        - Alpha decay patterns:
          - Does alpha decay differently Mon-Fri?

    Example:
        # Test if hl alpha varies by day of week
        >>> results = regress_alpha_dow(daily_df, 'hl', horizon=1)
        >>> results['day'] = results['horizon'] % 10
        >>> results['weekday'] = results['day'].map({
        ...     0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri'})
        >>> print(results[['weekday', 'coef', 'tstat']])

    Horizon Encoding:
        horizon=10: 1-day return, Monday (10 = 1*10 + 0)
        horizon=11: 1-day return, Tuesday (11 = 1*10 + 1)
        horizon=30: 3-day return, Monday (30 = 3*10 + 0)
        horizon=34: 3-day return, Friday (34 = 3*10 + 4)

    Note:
        Significant variation across days suggests calendar effects that could
        be exploited for trading schedule optimization.
    """
    print("Regressing alphas day of week for {} with horizon {}...".format(indep, horizon))
    retname = 'cum_ret' + str(horizon)
    fitdata_df = daily_df[[retname, 'mdvp', indep, 'dow']]
    fitdata_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fitdata_df = fitdata_df.dropna()
    results_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'], dtype=float)
    for name, daygroup in fitdata_df.groupby('dow'):
        weights = np.sqrt(daygroup['mdvp'])
        weights = daygroup['mdvp'] ** ADV_POWER
        ys = winsorize_by_date(daygroup[retname])
        results_wls = sm.WLS(ys, sm.add_constant(daygroup[indep]), weights=weights).fit()
        print(results_wls.summary())
        results_df = results_df.append(extract_results(results_wls, indep, horizon * 10 + int(name)), ignore_index=True)

    return results_df
