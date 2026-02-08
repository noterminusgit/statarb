#!/usr/bin/env python
"""
Regression Analysis Module

This module performs regression analysis to fit alpha factors against forward
returns and extract predictive coefficients for the statistical arbitrage system.

Key Features:
    - Weighted Least Squares (WLS) regression with market cap weighting
    - Multiple horizon regression (1-5 days ahead)
    - Outlier handling via winsorization
    - Cross-sectional regression by date
    - Statistical diagnostics (t-stats, standard errors, R-squared)
    - Visualization of regression results

Regression Types:
    - Daily: Cross-sectional regression of daily alphas vs. forward returns
    - Intraday: Time-slice regression for intraday signals
    - Day-of-Week: Separate regressions by day of week

Functions:
    regress_alpha(): Main regression function with WLS
    extract_results(): Extract coefficients and statistics from regression
    plot_fit(): Visualize coefficient decay across horizons
    regress_day_alpha(): Day-specific regression analysis
    run_intra_regression(): Intraday time-slice regression

The module uses statsmodels for robust regression with proper handling of
missing data and outliers. Results include coefficients, t-statistics, and
standard errors for alpha signal calibration.

ADV_POWER: Power parameter for ADV-based weighting (default: 0.5)
"""

from __future__ import division, print_function

import sys
import os
import glob
import argparse
import re
import math
import logging
from collections import defaultdict
from dateutil import parser as dateparser

import time
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import statsmodels.api as sm

from util import *
from calc import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ADV_POWER = 1/2

def plot_fit(fits_df, name):
    """
    Visualize regression coefficients across multiple horizons.

    Creates an error bar plot showing how alpha coefficients and intercepts
    vary across different forecast horizons (typically 1-5 days). Useful for
    diagnosing alpha decay and coefficient stability.

    Args:
        fits_df (DataFrame): Regression results with columns:
            - horizon (int): Forecast horizon in days
            - coef (float): Alpha coefficient
            - stderr (float): Standard error of coefficient
            - intercept (float): Regression intercept
        name (str): Output filename prefix (saves as name.png)

    Output:
        Saves PNG plot showing:
        - Blue points: Coefficients with 2-sigma error bars
        - Red points: Intercepts
        - Horizontal line at y=0 for reference
    """
    print("Plotting fits...")
    print(fits_df)
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
    Handles both intercept and no-intercept regressions.

    Args:
        results (RegressionResults): Fitted statsmodels WLS model
        indep (str): Name of independent variable (alpha factor)
        horizon (int): Forecast horizon in days (or timeslice index)

    Returns:
        DataFrame: Single-row DataFrame with columns:
            - indep: Independent variable name
            - horizon: Forecast horizon
            - nobs: Number of observations in fit
            - coef: Alpha coefficient estimate
            - stderr: Standard error of coefficient
            - tstat: T-statistic for coefficient
            - intercept: Regression intercept (0 if no-intercept model)
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

    Fits alpha regressions for horizons 1 through horizon using median regression
    (out-of-sample splits) and extracts intercept values. Useful for detecting
    systematic biases or drift in forward returns that are not explained by alpha.

    Args:
        daily_df (DataFrame): Daily results with alpha factors and returns
        horizon (int): Maximum forecast horizon (fits 1 to horizon)
        name (str): Alpha factor column name
        middate (datetime, optional): If specified, use only data before this
            date for in-sample fitting (out-of-sample validation)

    Returns:
        dict: Mapping from horizon (int) to intercept (float)
            {1: intercept_1day, 2: intercept_2day, ...}

    Note:
        Non-zero intercepts may indicate:
        - Systematic market drift
        - Alpha miscalibration
        - Missing risk factors
    """
    insample_daily_df = daily_df
    if middate is not None:
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr', 'intercept'])
    for ii in range(1, horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, name, ii, True, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    result = dict()
    for ii in range(1, horizon+1):
        result[ii] = float(fits_df.ix[name].ix[ii].ix['intercept'])

    return result

def regress_alpha(results_df, indep, horizon, median=False, rtype='daily', intercept=True, start=None, end=None):
    """
    Main regression function to fit alpha factors against forward returns.

    Dispatches to specialized regression functions based on regression type.
    Supports out-of-sample validation via median regression (3-fold split).

    Args:
        results_df (DataFrame): Results with alpha factors and forward returns,
            indexed by (date, sid) for daily or (date, time, sid) for intraday
        indep (str): Independent variable (alpha factor column name)
        horizon (int): Forecast horizon in days (daily) or bars (intraday)
        median (bool): If True, perform 3-fold out-of-sample validation and
            return median coefficients. If False, fit on full dataset.
        rtype (str): Regression type:
            - 'daily': Cross-sectional daily regression
            - 'intra': Intraday time-slice regression
            - 'dow': Day-of-week specific regression
            - 'intra_eod': Intraday regression vs EOD returns
        intercept (bool): Include intercept in regression (default True)
        start (str, optional): Start date for regression (YYYYMMDD format)
        end (str, optional): End date for regression (YYYYMMDD format)

    Returns:
        DataFrame: Regression results with columns:
            - indep: Independent variable name
            - horizon: Forecast horizon
            - coef: Alpha coefficient
            - stderr: Standard error
            - tstat: T-statistic
            - nobs: Number of observations
            - intercept: Regression intercept

    Note:
        median=True implements out-of-sample validation by splitting data into
        thirds, fitting each third separately, and returning median coefficients.
    """
    if start is not None and end is not None:
        print("restrict fit from {} to {}".format(start, end))
        results_df = results_df.truncate(before=dateparser.parse(start), after=dateparser.parse(end))

    if median:
        medians_df = pd.DataFrame(columns=['indep', 'horizon', 'coef', 'stderr', 'tstat', 'nobs', 'intercept'], dtype=float)
        start = 1
        cnt = len(results_df)
        window = int(cnt/3)
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

            print(fitresults_df)
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
        horizon (int): Forward return horizon in days
        intercept (bool): Include intercept term (default True)

    Returns:
        DataFrame: Single-row result with coef, stderr, tstat, nobs, intercept

    Methodology:
        1. Extract {indep}, cum_ret{horizon}, mdvp columns
        2. Drop NaN and infinite values
        3. Set weights = mdvp^0.5 (balance small/large cap)
        4. Winsorize returns by date (handles extreme days)
        5. Convert log returns to simple returns: exp(log_ret) - 1
        6. Winsorize alpha factor (handles outliers)
        7. Fit WLS: returns ~ alpha (with optional intercept)

    Example:
        To fit hl (high-low) alpha for 3-day returns:
        >>> regress_alpha_daily(daily_df, 'hl', horizon=3)
    """
    # Validate inputs
    if daily_df is None or daily_df.empty:
        raise ValueError("daily_df cannot be None or empty")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer, got: {}".format(horizon))
    if not isinstance(indep, str) or not indep:
        raise ValueError("indep must be a non-empty string")

    print("Regressing alphas daily for {} with horizon {}...".format(indep, horizon))
    retname = 'cum_ret'+str(horizon)

    # Check for required columns
    required_cols = [retname, 'mdvp', indep]
    missing_cols = [col for col in required_cols if col not in daily_df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: {}".format(missing_cols))

    fitdata_df = daily_df[ [retname, 'mdvp', indep] ]
    fitdata_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fitdata_df = fitdata_df.dropna()

    # Check for sufficient observations
    if len(fitdata_df) < 10:
        raise ValueError("Insufficient observations for regression: {} rows (need at least 10)".format(len(fitdata_df)))

    # Check for data quality
    if fitdata_df['mdvp'].min() <= 0:
        logging.warning("Found non-positive mdvp values, may cause issues with weighting")
    if fitdata_df[indep].std() == 0:
        raise ValueError("Alpha factor {} has zero variance, cannot perform regression".format(indep))

    weights = fitdata_df['mdvp'] ** ADV_POWER

    # Validate weights
    if weights.isnull().any() or (weights <= 0).any():
        logging.warning("Found invalid weights, setting to 1 where invalid")
        weights = weights.fillna(1)
        weights[weights <= 0] = 1

    ys = winsorize_by_date(fitdata_df[retname])
    ys = np.exp(ys) - 1
    xs = winsorize(fitdata_df[indep])
    if intercept:
        xs = sm.add_constant(xs)

    try:
        results_wls = sm.WLS(ys, xs, weights=weights).fit()
        print(results_wls.summary())
        results_df = extract_results(results_wls, indep, horizon)
        return results_df
    except Exception as e:
        raise ValueError("Regression failed for {} at horizon {}: {}".format(indep, horizon, str(e)))

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
            - horizon: Timeslice index (1=10:00, 2=11:00, ..., 6=15:00)
            - coef: Alpha coefficient
            - stderr: Standard error
            - tstat: T-statistic
            - nobs: Number of observations
            - intercept: Regression intercept

    Methodology:
        For each hourly timeslice:
        1. Extract bars at that specific time
        2. Calculate day_ret = (close - open) / open
        3. Winsorize day_ret to handle outliers
        4. Fit WLS: day_ret ~ alpha + constant, weighted by mdvp^0.5

    Use Case:
        Diagnostic tool to see if intraday alphas predict cumulative intraday
        returns, or if they only predict bar-to-bar changes.
    """
    print("Regressing intra alphas for {} on EOD...".format(indep))
    results_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'], dtype=float)
    fitdata_df = intra_df[  ['log_ret', indep, 'mdvp', 'close', 'iclose'] ]
    fitdata_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fitdata_df = fitdata_df.dropna()

    it = 1
    for timeslice in ['10:00', '11:00', '12:00', '13:00', '14:00', '15:00' ]:
        print("Fitting for timeslice: {}".format(timeslice))

        timeslice_df = fitdata_df.unstack().between_time(timeslice, timeslice).stack()
        timeslice_df['day_ret'] = (timeslice_df['close'] - timeslice_df['iclose']) / timeslice_df['iclose']
 #       timeslice_df['day_ret'] = np.log(timeslice_df['close'] / timeslice_df['iclose'])

        weights = np.sqrt(timeslice_df['mdvp'])
        weights = timeslice_df['mdvp'] ** ADV_POWER
        results_wls = sm.WLS(winsorize(timeslice_df['day_ret']), sm.add_constant(timeslice_df[indep]), weights=weights).fit()
        print(results_wls.summary())
        results_df = results_df.append(extract_results(results_wls, indep, it), ignore_index=True)

        it += 1

    return results_df

def regress_alpha_intra(intra_df, indep, horizon):
    """
    Intraday forward-looking regression with multiple bar horizon.

    For each 30-minute timeslice, fits alpha observed at time T against the
    cumulative return from market open through T+horizon bars. Tests whether
    intraday signals predict forward intraday returns over multiple bars.

    Args:
        intra_df (DataFrame): Intraday bar data indexed by (date, time, sid):
            - log_ret: Log return for each bar
            - {indep}: Alpha factor value
            - mdvp: Market cap weighting factor
            - close: Current bar close price
            - iclose: Initial (opening) price for the day
        indep (str): Alpha factor column name
        horizon (int): Number of 30-minute bars to look ahead (e.g., 3 = 90 min)

    Returns:
        DataFrame: 6-row result (one per timeslice) with columns:
            - horizon: Timeslice index (1-6 for 10:30-15:30)
            - coef: Alpha coefficient
            - stderr: Standard error
            - tstat: T-statistic
            - nobs: Number of observations
            - intercept: Regression intercept

    Methodology:
        For each 30-minute timeslice (10:30, 11:30, ..., 15:30):
        1. Extract bars at that specific time
        2. Shift log_ret forward by horizon bars
        3. Sum shifted log returns over horizon window (cum_ret)
        4. Calculate day_ret = exp(log(close/open) + cum_ret) - 1
        5. Winsorize day_ret by timeslice
        6. Fit WLS: day_ret ~ alpha + constant, weighted by mdvp^0.5

    Example:
        horizon=3 at 10:30 tests if 10:30 alpha predicts cumulative return
        from open through 12:00 (3 bars forward: 11:00, 11:30, 12:00).
    """
    print("Regressing intra alphas for {} on horizon {}...".format(indep, horizon))
    assert horizon > 0
    results_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'], dtype=float)
    retname = 'cum_ret'+str(horizon)
    fitdata_df = intra_df[  ['log_ret', indep, 'mdvp', 'close', 'iclose'] ]
    fitdata_df[retname] = np.nan
    fitdata_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    it = 1
    for timeslice in ['10:30', '11:30', '12:30', '13:30', '14:30', '15:30' ]:
        print("Fitting for timeslice: {} at horizon {}".format(timeslice, horizon))

        timeslice_df = fitdata_df.unstack().between_time(timeslice, timeslice).stack()
        shift_df = timeslice_df.unstack().shift(-horizon).stack()
        timeslice_df[retname] = shift_df['log_ret'].groupby(level='sid').apply(lambda x: pd.rolling_sum(x, horizon))
#        intra_df.ix[ timeslice_df.index, retname ] = timeslice_df[retname]
        timeslice_df['day_ret'] = np.exp(np.log(timeslice_df['close'] / timeslice_df['iclose']) + timeslice_df[retname]) - 1
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
    whether alpha coefficients vary by day. Useful for detecting:
    - Day-of-week effects (Monday reversal, Friday momentum, etc.)
    - Alpha decay patterns across the trading week
    - Optimal rebalancing schedules

    Args:
        daily_df (DataFrame): Daily data indexed by (date, sid) with columns:
            - cum_ret{horizon}: Cumulative log return over horizon days
            - mdvp: Market cap weighting factor
            - {indep}: Alpha factor values
            - dow: Day of week (0=Monday, 1=Tuesday, ..., 4=Friday)
        indep (str): Alpha factor column name
        horizon (int): Forward return horizon in days

    Returns:
        DataFrame: 5-row result (one per weekday) with columns:
            - horizon: Encoded as horizon*10 + dow (e.g., 30=3-day Mon, 31=3-day Tue)
            - coef: Alpha coefficient for this day
            - stderr: Standard error
            - tstat: T-statistic
            - nobs: Number of observations
            - intercept: Regression intercept

    Methodology:
        1. Group data by day of week (dow column)
        2. For each day (0-4), fit separate WLS regression:
           - Weight by mdvp^0.5
           - Winsorize returns by date within each day group
           - Fit: returns ~ alpha + constant
        3. Encode results with horizon*10 + dow for identification

    Example:
        To test if hl alpha has different strength Mon-Fri:
        >>> regress_alpha_dow(daily_df, 'hl', horizon=1)
    """
    print("Regressing alphas day of week for {} with horizon {}...".format(indep, horizon))
    retname = 'cum_ret'+str(horizon)
    fitdata_df = daily_df[ [retname, 'mdvp', indep, 'dow'] ]
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


