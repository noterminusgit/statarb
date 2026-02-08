#!/usr/bin/env python
"""
Factor Calculation and Analysis Module

This module provides functions for calculating alpha factors, forward returns,
and statistical transformations used in the statistical arbitrage system.

Key Features:
    - Forward return calculation at multiple horizons (1-5 days)
    - Volume profile calculation for intraday patterns
    - Data winsorization to handle outliers
    - Rolling correlations and exponentially-weighted calculations
    - Price and volume ratio calculations

Factor Categories:
    BARRA_FACTORS: Standard risk factors (13 factors)
        - growth, size, sizenl, divyild, btop, earnyild
        - beta, resvol, betanl, momentum, leverage, liquidty, country

    INDUSTRIES: Industry classifications (58 industries)
        - Barra GICS-based industry groupings

    PROP_FACTORS: Proprietary factors (2 factors)
        - srisk_pct_z: Standardized specific risk percentage
        - rating_mean_z: Standardized analyst rating mean

Key Functions:
    calc_forward_returns(): Calculate cumulative returns at multiple horizons
    calc_vol_profiles(): Calculate intraday volume participation patterns
    winsorize(): Trim outliers at specified standard deviation levels
    winsorize_by_date(): Apply winsorization within each date
    calc_price_extras(): Calculate volatility ratios and volume metrics
    rolling_ew_corr_pairwise(): Exponentially-weighted pairwise correlations

The module uses pandas DataFrames with MultiIndex (date, sid) for efficient
time-series operations across securities.
"""

from __future__ import division, print_function

import numpy as np
import pandas as pd
import gc
import logging

from scipy import stats
from pandas.stats.api import ols
from pandas.stats import moments
from lmfit import minimize, Parameters, Parameter, report_errors
from collections import defaultdict

from util import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


INDUSTRIES = ['CONTAINR', 'HLTHSVCS', 'SPLTYRET', 'SPTYSTOR', 'DIVFIN', 'GASUTIL', 'BIOLIFE', 'SPTYCHEM', 'ALUMSTEL', 'AERODEF', 'COMMEQP', 'HOUSEDUR', 'CHEM', 'LEISPROD', 'AUTO', 'CONGLOM', 'HOMEBLDG', 'CNSTENG', 'LEISSVCS', 'OILGSCON', 'MEDIA', 'FOODPROD', 'PSNLPROD', 'OILGSDRL', 'SOFTWARE', 'BANKS', 'RESTAUR', 'FOODRET', 'ROADRAIL', 'APPAREL', 'INTERNET', 'NETRET', 'PAPER', 'WIRELESS', 'PHARMA', 'MGDHLTH', 'CNSTMACH', 'OILGSEQP', 'REALEST', 'COMPELEC', 'BLDGPROD', 'TRADECO', 'MULTUTIL', 'CNSTMATL', 'HLTHEQP', 'PRECMTLS', 'INDMACH', 'TRANSPRT', 'SEMIEQP', 'TELECOM', 'OILGSEXP', 'INSURNCE', 'AIRLINES', 'SEMICOND', 'ELECEQP', 'ELECUTIL', 'LIFEINS', 'COMSVCS', 'DISTRIB']
BARRA_FACTORS = ['country', 'growth', 'size', 'sizenl', 'divyild', 'btop', 'earnyild', 'beta', 'resvol', 'betanl', 'momentum', 'leverage', 'liquidty']
PROP_FACTORS = ['srisk_pct_z', 'rating_mean_z']
ALL_FACTORS = BARRA_FACTORS + INDUSTRIES + PROP_FACTORS

def calc_vol_profiles(full_df):
    """
    Calculate intraday volume participation profiles at 15-minute intervals.

    Computes 21-day rolling median and standard deviation of dollar volume
    at each 15-minute timeslice from 09:45 to 16:00 (market hours).

    Args:
        full_df: DataFrame with MultiIndex (date, sid) containing:
            - dvolume: Intraday bar volume
            - dvwap: Intraday bar VWAP
            - tradable_med_volume_21: 21-day median tradable volume
            - close: Close price

    Returns:
        DataFrame with added columns:
            - dpvolume: Dollar volume (dvolume * dvwap)
            - dpvolume_med_21: 21-day trailing median dollar volume by timeslice
            - dpvolume_std_21: 21-day trailing std dev dollar volume by timeslice

    Notes:
        - Processes 26 timeslices per trading day (15-min intervals)
        - Uses shifted data (shift(1)) to avoid look-ahead bias
        - Prints average volume fraction at each timeslice for debugging
    """
    full_df['dpvolume_med_21'] = np.nan
    full_df['dpvolume_std_21'] = np.nan
    full_df['dpvolume'] = full_df['dvolume'] * full_df['dvwap']
    print("Calculating trailing volume profile...")
    for timeslice in ['09:45', '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00' ]:
        timeslice_df = full_df[ ['dpvolume', 'tradable_med_volume_21', 'close'] ]
        timeslice_df = timeslice_df.unstack().between_time(timeslice, timeslice).stack()
        timeslice_df = timeslice_df.dropna()
        if len(timeslice_df) == 0: continue
        timeslice_df['dpvolume_med_21'] = timeslice_df['dpvolume'].groupby(level='sid').apply(lambda x: pd.rolling_median(x.shift(1), 21))
        timeslice_df['dpvolume_std_21'] = timeslice_df['dpvolume'].groupby(level='sid').apply(lambda x: pd.rolling_std(x.shift(1), 21))
        m_df = timeslice_df.dropna()
        print(m_df.head())
        print("Average dvol frac at {}: {}".format(timeslice, (m_df['dpvolume_med_21'] / (m_df['tradable_med_volume_21'] * m_df['close'])).mean()))
        full_df.ix[ timeslice_df.index, 'dpvolume_med_21'] = timeslice_df['dpvolume_med_21']
        full_df.ix[ timeslice_df.index, 'dpvolume_std_21'] = timeslice_df['dpvolume_std_21']

    return full_df

def calc_price_extras(daily_df):
    """
    Calculate derived volatility and volume metrics.

    Args:
        daily_df: DataFrame with MultiIndex (date, sid) containing:
            - volat_21: 21-day rolling volatility
            - volat_60: 60-day rolling volatility
            - tradable_volume: Daily tradable volume
            - shares_out: Shares outstanding
            - comp_volume: Composite volume

    Returns:
        DataFrame with added columns:
            - volat_ratio: Short/long volatility ratio (21-day / 60-day)
            - volume_ratio: Turnover ratio (volume / comp_volume, overwrites previous)
            - volat_move: Day-over-day change in 21-day volatility

    Notes:
        - volume_ratio is calculated twice, second calculation overwrites first
        - volat_ratio indicates volatility regime changes (>1 = increasing vol)
        - volat_move captures volatility momentum
    """
    daily_df['volat_ratio'] = daily_df['volat_21'] / daily_df['volat_60']
    daily_df['volume_ratio'] = daily_df['tradable_volume'] / daily_df['shares_out']
    daily_df['volume_ratio'] = daily_df['tradable_volume'] / daily_df['comp_volume']
    daily_df['volat_move'] = daily_df['volat_21'].diff()
    return daily_df

def calc_forward_returns(daily_df, horizon):
    """
    Calculate forward cumulative returns at multiple horizons.

    Computes future returns from 1 to N days ahead for alpha backtesting.
    Uses negative shift to look forward in time.

    Args:
        daily_df: DataFrame with MultiIndex (date, sid) containing:
            - log_ret: Daily log returns
        horizon: Maximum forward horizon (1-5 typical)

    Returns:
        DataFrame with columns:
            - cum_ret1: 1-day forward cumulative log return
            - cum_ret2: 2-day forward cumulative log return
            - ... up to cum_ret{horizon}

    Notes:
        - Uses shift(-ii) to look forward ii days
        - Prevents look-ahead bias in alpha generation
        - Returns are per-security using groupby(level='sid')
        - Used by regress.py to fit alpha factors to future returns

    Example:
        If today is day T:
        - cum_ret1 = log_ret[T+1]
        - cum_ret2 = log_ret[T+1] + log_ret[T+2]
        - cum_ret5 = sum of log_ret from T+1 to T+5
    """
    # Validate inputs
    if daily_df is None or daily_df.empty:
        raise ValueError("daily_df cannot be None or empty")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer, got: {}".format(horizon))
    if horizon > 20:
        logging.warning("horizon={} is unusually large".format(horizon))
    if 'log_ret' not in daily_df.columns:
        raise ValueError("daily_df must contain 'log_ret' column")

    # Check for data quality issues in log_ret
    if daily_df['log_ret'].isnull().all():
        raise ValueError("All log_ret values are NaN")
    if np.isinf(daily_df['log_ret']).any():
        logging.warning("Found {} infinite values in log_ret".format(np.isinf(daily_df['log_ret']).sum()))

    print("Calculating forward returns...")
    results_df = pd.DataFrame( index=daily_df.index )
    for ii in range(1, horizon+1):
        retname = 'cum_ret'+str(ii)
        cum_rets = daily_df['log_ret'].groupby(level='sid').apply(lambda x: pd.rolling_sum(x, ii))
        shift_df = cum_rets.unstack().shift(-ii).stack()
        results_df[retname] = shift_df
    return results_df

def winsorize(data, std_level=5):
    """
    Winsorize data by capping outliers at N standard deviations from mean.

    Trims extreme values to reduce impact of outliers on statistics and
    regression. Values beyond mean ± N*std are clamped to the threshold.

    Args:
        data: Series or array of numeric values
        std_level: Number of standard deviations for threshold (default: 5)

    Returns:
        Winsorized data with same shape as input

    Notes:
        - Default std_level=5 is conservative (retains ~99.9999% of normal data)
        - Commonly used std_level values: 3 (99.7%), 5 (99.9999%)
        - Applied to alpha factors before regression to handle fat tails
        - Preserves ranking while limiting magnitude of extremes

    Example:
        If mean=0, std=1, std_level=5:
        - Values > 5 become 5
        - Values < -5 become -5
        - Values in [-5, 5] unchanged
    """
    # Validate inputs
    if data is None or len(data) == 0:
        raise ValueError("data cannot be None or empty")
    if not isinstance(std_level, (int, float)) or std_level <= 0:
        raise ValueError("std_level must be a positive number, got: {}".format(std_level))
    if std_level > 10:
        logging.warning("std_level={} is unusually high, may not winsorize effectively".format(std_level))

    result = data.copy()

    # Check for all NaN
    if result.isnull().all():
        logging.warning("All values are NaN in winsorize, returning as-is")
        return result

    # Check for infinite values
    if np.isinf(result).any():
        logging.warning("Found {} infinite values in data, replacing with NaN".format(np.isinf(result).sum()))
        result = result.replace([np.inf, -np.inf], np.nan)

    std = result.std() * std_level
    mean = result.mean()

    # Handle case where std is 0 or NaN
    if pd.isnull(std) or std == 0:
        logging.warning("Standard deviation is {} in winsorize, returning data unchanged".format(std))
        return result

    result[result > mean + std] = mean + std
    result[result < mean - std] = mean - std
    return result

def winsorize_by_date(data):
    """
    Apply winsorization within each date cross-section.

    Winsorizes data separately for each trading day, ensuring outlier
    detection is relative to same-day distribution across securities.

    Args:
        data: Series with MultiIndex (date, sid)

    Returns:
        Winsorized Series maintaining same index

    Notes:
        - Prevents outliers from one date affecting other dates
        - Critical for cross-sectional alpha signals
        - Uses default std_level=5 from winsorize()
    """
    print("Winsorizing by day...")
    return data.groupby(level='date', sort=False).transform(winsorize)

def winsorize_by_ts(data):
    """
    Apply winsorization within each intraday timestamp.

    Winsorizes data separately for each intraday bar timestamp.

    Args:
        data: Series with MultiIndex (iclose_ts, sid)

    Returns:
        Winsorized Series maintaining same index

    Notes:
        - Used for intraday factor calculations
        - iclose_ts is intraday bar close timestamp
    """
    print("Winsorizing by day...")
    return data.groupby(level='iclose_ts', sort=False).transform(winsorize)

def winsorize_by_group(data, group):
    """
    Apply winsorization within custom groups.

    Winsorizes data separately for each value of specified grouping variable.

    Args:
        data: Series to winsorize
        group: Grouping variable (e.g., industry, sector)

    Returns:
        Winsorized Series maintaining same index

    Notes:
        - Enables industry-neutral or sector-neutral winsorization
        - Prevents cross-contamination between groups
    """
    print("Winsorizing by day...")
    return data.groupby([group], sort=False).transform(winsorize)

def rolling_ew_corr_pairwise(df, halflife):
    """
    Calculate exponentially-weighted pairwise correlations for all column pairs.

    Computes time-varying correlation matrix using exponential weighting with
    specified halflife. Returns 3D panel with correlation time series.

    Args:
        df: DataFrame with time series columns
        halflife: Halflife for exponential weighting (days)

    Returns:
        Panel with dimensions (time, col1, col2) containing correlation series
        for each column pair

    Notes:
        - Uses pandas.stats.moments.ewmcorr (deprecated in newer pandas)
        - Span parameter = (halflife-1)/2.0 for ewm calculation
        - Returns full symmetric correlation matrix at each timestamp
        - Used for analyzing factor co-movement and regime changes

    Example:
        If df has columns ['alpha1', 'alpha2', 'alpha3']:
        Returns Panel[t, 'alpha1', 'alpha2'] = corr(alpha1, alpha2) at time t
    """
    all_results = {}
    for col, left in df.items():
        all_results[col] = col_results = {}
        for col, right in df.items():
            col_results[col] = moments.ewmcorr(left, right, span=(halflife-1)/2.0)

    ret = pd.Panel(all_results)
    ret = ret.swapaxes(0,1, copy=False)
    return ret

def push_data(df, col):
    """
    Shift specified column forward 1 period and merge with original DataFrame.

    Creates forward-looking (t+1) version of a column. WARNING: Can leak data
    across date boundaries.

    Args:
        df: DataFrame with MultiIndex (date, sid)
        col: Column name to shift forward

    Returns:
        DataFrame with original columns plus col_n (shifted forward 1 period)

    Notes:
        - Uses shift(-1) to look forward
        - Suffix '_n' indicates "next" period
        - CAUTION: May push data to next calendar day, creating look-ahead bias
        - Use carefully in alpha generation to avoid future data leakage
    """
    #Careful, can push to next day...
    lagged_df = df[[col]].unstack(level='sid').shift(-1).stack()
    merged_df = pd.merge(df, lagged_df, left_index=True, right_index=True, sort=True, suffixes=['', '_n'])
    return merged_df

def lag_data(daily_df):
    """
    Lag entire DataFrame by 1 period and merge with original.

    Creates previous-period (t-1) versions of all columns.

    Args:
        daily_df: DataFrame with MultiIndex (date, sid)

    Returns:
        DataFrame with original columns plus lagged versions (suffix '_y')

    Notes:
        - Uses shift(1) to look backward
        - Suffix '_y' indicates lagged values
        - Safe for alpha generation (no look-ahead bias)
        - Commonly used to create momentum/reversal signals
        - Example: price_y is yesterday's price
    """
    lagged_df = daily_df.unstack(level=-1).shift(1).stack()
    merged_df = pd.merge(daily_df, lagged_df, left_index=True, right_index=True, sort=True, suffixes=['', '_y'])
    return merged_df

def calc_med_price_corr(daily_df):
    """
    Calculate median price correlation. [NOT IMPLEMENTED]

    Args:
        daily_df: DataFrame with MultiIndex (date, sid)

    Returns:
        None (function stub)

    Notes:
        - Placeholder for future implementation
        - Likely intended for pair trading or sector correlation analysis
    """
    pass

def calc_resid_vol(daily_df):
    """
    Calculate Barra residual volatility (specific/idiosyncratic risk).

    Computes 20-day rolling standard deviation of Barra residual returns,
    measuring stock-specific risk after removing factor exposures.

    Args:
        daily_df: DataFrame with MultiIndex (date, sid) containing:
            - barraResidRet: Barra residual returns (from calc_factors())

    Returns:
        Series of residual volatility (annualized standard deviation)

    Notes:
        - Uses 20-day rolling window
        - Residual volatility = sqrt(var(barraResidRet))
        - Measures idiosyncratic risk not explained by factors
        - Lower residual vol indicates more factor-driven returns
        - Used in risk models and portfolio optimization
        - Requires barraResidRet from calc_factors() or calc_intra_factors()

    Dependencies:
        Requires barraResidRet column from calc_factors()
    """
    lookback = 20
    daily_df['barraResidVol'] = np.sqrt(pd.rolling_var(daily_df['barraResidRet'], lookback))
    return daily_df['barraResidVol']

def calc_factor_vol(factor_df):
    """
    Calculate exponentially-weighted factor covariance matrix.

    Computes time-varying covariance matrix for all factor pairs using
    exponential weighting. Used for factor risk modeling.

    Args:
        factor_df: DataFrame with MultiIndex (date, factor) containing:
            - ret: Factor returns from calc_factors() or calc_intra_factors()

    Returns:
        Dictionary keyed by (factor1, factor2) tuples, values are Series
        of exponentially-weighted covariances over time

    Notes:
        - Halflife = 20 days for exponential weighting
        - Covers all 73 factors (13 Barra + 58 industries + 2 proprietary)
        - Returns symmetric covariance matrix: cov(A,B) = cov(B,A)
        - Uses pandas.stats.moments.ewmcov (deprecated in newer pandas)
        - Commented-out code shows alternative: 20-day rolling covariance
        - Used by opt.py for portfolio risk calculations

    Example:
        ret[(('beta', 'size')] = time series of beta-size covariance
        ret[('BANKS', 'BANKS')] = time series of banking industry variance

    Dependencies:
        Requires factor returns from calc_factors() or calc_intra_factors()
    """
    halflife = 20.0
    #    factors = factor_df.index.get_level_values('factor').unique()
    factors = ALL_FACTORS
    ret = dict()
    for factor1 in factors:
        for factor2 in factors:
            key = (factor1, factor2)
            if key not in ret.keys():
                ret[key] = moments.ewmcov(factor_df.xs(factor1, level=1)['ret'], factor_df.xs(factor2, level=1)['ret'], span=(halflife-1)/2.0)
#                ret[key] = pd.rolling_cov(factor_df.xs(factor1, level=1)['ret'], factor_df.xs(factor2, level=1)['ret'], window=20)
#                print "Created factor Cov on {} from {} to {}".format(key, min(ret[key].index), max(ret[key].index))
    return ret

weights_df = None

def create_z_score(daily_df, name):
    """
    Standardize column to z-score within each date cross-section.

    Converts raw factor values to standardized scores (mean=0, std=1) for
    each trading day, enabling cross-sectional comparison.

    Args:
        daily_df: DataFrame with MultiIndex (date, sid) containing:
            - name: Column to standardize
            - gdate: Grouping date column

    Returns:
        DataFrame with added column {name}_z containing z-scores

    Notes:
        - Z-score = (x - mean) / std, calculated per date
        - Enables rank-based alpha signals
        - Makes factors comparable across time periods
        - Groups by 'gdate' (not 'date'), allowing custom date grouping
        - Used for srisk_pct_z and rating_mean_z proprietary factors

    Example:
        create_z_score(df, 'srisk_pct') adds 'srisk_pct_z' column
        with cross-sectional z-scores
    """
    zscore = lambda x: ( (x - x.mean()) / x.std())
    indgroups = daily_df[[name, 'gdate']].groupby(['gdate'], sort=True).transform(zscore)
    daily_df[name + "_z"] = indgroups[name]
    return daily_df
    
def calc_factors(daily_df, barraOnly=False):
    """
    Decompose stock returns into factor returns and residual returns.

    Performs daily cross-sectional regression to separate returns into:
    1. Factor returns (systematic risk)
    2. Residual returns (idiosyncratic/stock-specific risk)

    Uses Weighted Least Squares with market cap weighting and lmfit minimization.

    Args:
        daily_df: DataFrame with MultiIndex (date, sid) containing:
            - log_ret: Daily log returns
            - All factor loadings (BARRA_FACTORS, INDUSTRIES)
            - capitalization: Market cap for weighting
            - indname1: Industry classification
            - srisk_pct: Specific risk percentage (if barraOnly=False)
            - rating_mean: Analyst rating mean (if barraOnly=False)
        barraOnly: If True, use only Barra+industry factors (73 total).
                   If False, add proprietary factors (75 total). Default: False.

    Returns:
        Tuple of (daily_df, factorRets_df):
            - daily_df: Input DataFrame with added 'barraResidRet' column
            - factorRets_df: MultiIndex (date, factor) with 'ret' column

    Notes:
        - Processes each date independently to avoid look-ahead bias
        - Uses log(market cap) as regression weights
        - Country factor constrained as cap-weighted sum of industry returns
        - Creates z-scores for proprietary factors (srisk_pct_z, rating_mean_z)
        - Uses factorize() function for WLS regression via lmfit
        - Memory cleanup with gc.collect() after each date
        - Prints regression diagnostics for monitoring

    Factor Categories:
        - BARRA_FACTORS: 13 risk factors (beta, size, momentum, etc.)
        - INDUSTRIES: 58 industry classifications
        - PROP_FACTORS: 2 proprietary (srisk_pct_z, rating_mean_z)

    Dependencies:
        - Requires Barra factor loadings from loaddata.load_barra()
        - Calls factorize() for regression fitting
        - Calls create_z_score() for proprietary factor standardization

    See Also:
        calc_intra_factors() - Intraday version using 30-min bars
        factorize() - WLS regression implementation
    """
    print("Calculating factors...")

    allreturns_df = pd.DataFrame(columns=['barraResidRet'], index=daily_df.index)
    if barraOnly:
        factors = BARRA_FACTORS + INDUSTRIES
    else:
        daily_df = create_z_score(daily_df, 'srisk_pct')
        daily_df = create_z_score(daily_df, 'rating_mean')
        factors = ALL_FACTORS

    print("Total len: {}".format(len(daily_df)))
    cnt = 0
    cnt1 = 0
    factorrets = list()
    for name, group in daily_df.groupby(level='date'):
        print("Regressing {}".format(name))
        cnt1 += len(group)
        print("Size: {} {}".format(len(group), cnt1))

        loadings_df = group[ factors ]
        loadings_df = loadings_df.reset_index().fillna(0)

        del loadings_df['sid']
        del loadings_df['date']

#        print "loadings len {}".format(len(loadings_df))
#        print loadings_df.head()

        returns_df = group['log_ret'].fillna(0)
 #       print "returns len {}".format(len(returns_df))

 #       print returns_df.head()
        global weights_df
        weights_df = np.log(group['capitalization']).fillna(0)
#        print weights_df.head()
        weights_df = pd.DataFrame( np.diag(weights_df) )

        #        print "weights len {}".format(len(weights_df))
        indwgt = dict()
        capsum = (group['capitalization'] / 1e6).sum()
        for ind in INDUSTRIES:
            indwgt[ind] = (group[ group['indname1'] == ind]['capitalization'] / 1e6).sum() / capsum
#        print returns_df.head()

        fRets, residRets = factorize(loadings_df, returns_df, weights_df, indwgt)        
        print("Factor Returns:")
#        print fRets
#        print residRets
        
        cnt += len(residRets)
        print("Running tally: {}".format(cnt))
        fdf = pd.DataFrame([ [i,v] for i, v in fRets.items() ], columns=['factor', 'ret'])
        fdf['date'] = name
        factorrets.append( fdf )
        allreturns_df.ix[ group.index, 'barraResidRet'] = residRets

        fRets = residRets = None
        gc.collect()

#    print allreturns_df.tail()
    factorRets_df = pd.concat(factorrets).set_index(['date', 'factor']).fillna(0)
    print("Final len {}".format(len(allreturns_df)))
    daily_df['barraResidRet'] = allreturns_df['barraResidRet']
    return daily_df, factorRets_df

def calc_intra_factors(intra_df, barraOnly=False):
    """
    Decompose intraday stock returns into factor and residual components.

    Intraday version of calc_factors() using 30-minute bars. Performs
    cross-sectional regression at each bar timestamp to extract factor returns.

    Args:
        intra_df: DataFrame with MultiIndex (iclose_ts, sid) containing:
            - overnight_log_ret: Overnight return (close to open)
            - iclose: Intraday bar close price
            - dopen: Daily open price
            - All factor loadings (BARRA_FACTORS, INDUSTRIES, PROP_FACTORS)
            - capitalization: Market cap for weighting
            - indname1: Industry classification
        barraOnly: If True, use only Barra+industry factors.
                   If False, include proprietary factors. Default: False.

    Returns:
        Tuple of (intra_df, factorRets_df):
            - intra_df: Input DataFrame with added 'barraResidRetI' column
            - factorRets_df: MultiIndex (iclose_ts, factor) with 'ret' column

    Notes:
        - Processes each intraday timestamp independently
        - Return calculation: overnight_log_ret + log(iclose/dopen)
        - Uses log(market cap) as regression weights
        - Groups by 'iclose_ts' instead of 'date'
        - Suffix 'I' (barraResidRetI) indicates intraday
        - Used by qsim.py for intraday backtesting

    Factor Categories:
        Same as calc_factors():
        - 13 Barra factors
        - 58 industries
        - 2 proprietary factors (if barraOnly=False)

    Dependencies:
        - Requires intraday bar data from loaddata.load_bars()
        - Calls factorize() for regression fitting

    See Also:
        calc_factors() - Daily version
        factorize() - WLS regression implementation
    """
    print("Calculating intra factors...")

    allreturns_df = pd.DataFrame(columns=['barraResidRetI'], index=intra_df.index)

    if barraOnly:
        factors = BARRA_FACTORS + INDUSTRIES
    else:
        factors = ALL_FACTORS
    
    print("Total len: {}".format(len(intra_df)))
    cnt = 0
    cnt1 = 0
    factorrets = list()
    for name, group in intra_df.groupby(level='iclose_ts'):
        print("Regressing {}".format(name))
        cnt1 += len(group)
        print("Size: {} {}".format(len(group), cnt1))

        loadings_df = group[ factors ]
        loadings_df = loadings_df.reset_index().fillna(0)

        del loadings_df['sid']
        del loadings_df['iclose_ts']

#        print "loadings len {}".format(len(loadings_df))
#        print loadings_df.head()

        returns_df = (group['overnight_log_ret'] + np.log(group['iclose'] / group['dopen'])).fillna(0)
 #       print "returns len {}".format(len(returns_df))

 #       print returns_df.head()
        global weights_df
        weights_df = np.log(group['capitalization']).fillna(0)
#        print weights_df.head()
        weights_df = pd.DataFrame( np.diag(weights_df) )
        #        print "weights len {}".format(len(weights_df))
        indwgt = dict()
        capsum = (group['capitalization'] / 1e6).sum()
        for ind in INDUSTRIES:
            indwgt[ind] = (group[ group['indname1'] == ind]['capitalization'] / 1e6).sum() / capsum
#        print returns_df.head()

        fRets, residRets = factorize(loadings_df, returns_df, weights_df, indwgt)        
        print("Factor Returns:")
        print(fRets)
#        print residRets
        
        cnt += len(residRets)
        print("Running tally: {}".format(cnt))
        fdf = pd.DataFrame([ [i,v] for i, v in fRets.items() ], columns=['factor', 'ret'])
        fdf['iclose_ts'] = name
        factorrets.append( fdf )
        allreturns_df.ix[ group.index, 'barraResidRetI'] = residRets

        fRets = residRets = None
        gc.collect()

#    print allreturns_df.tail()
    factorRets_df = pd.concat(factorrets).set_index(['iclose_ts', 'factor']).fillna(0)
    print("Final len {}".format(len(allreturns_df)))
    intra_df['barraResidRetI'] = allreturns_df['barraResidRetI']
    return intra_df, factorRets_df

def factorize(loadings_df, returns_df, weights_df, indwgt):
    """
    Estimate factor returns using Weighted Least Squares regression.

    Fits factor model: returns = loadings @ factor_returns + residuals
    using market cap weighted regression via lmfit minimization.

    Args:
        loadings_df: DataFrame (N securities x M factors) with factor exposures
        returns_df: Series (N securities) of realized returns
        weights_df: DataFrame (N x N) diagonal matrix of log(market cap) weights
        indwgt: Dict mapping industry name to cap-weighted fraction

    Returns:
        Tuple of (fRets_d, residRets_na):
            - fRets_d: Dict mapping factor name to estimated factor return
            - residRets_na: Array of residual returns (N securities)

    Notes:
        - Uses lmfit.minimize() for nonlinear least squares
        - Minimization target: weighted residual from fcn2min()
        - Country factor constrained to be cap-weighted sum of industries
        - Prints statistical significance (2-sigma test) for each factor
        - Exits on optimization failure (result.success == False)
        - Returns stderr for uncertainty quantification

    Constraint:
        country = sum(industry[i] * indwgt[i]) for all industries

    Algorithm:
        1. Initialize Parameters object with factor returns = 0
        2. Add country constraint as weighted sum of industries
        3. Minimize fcn2min() using lmfit
        4. Extract factor returns and residuals from optimization result
        5. Test statistical significance using stderr

    Dependencies:
        - Uses global weights_df for fcn2min() calculation
        - Calls fcn2min() as objective function

    See Also:
        fcn2min() - Objective function for minimization
        calc_factors() - Caller for daily returns
        calc_intra_factors() - Caller for intraday returns
    """
    print("Factorizing...")
    params = Parameters()
    for colname in loadings_df.columns:
        expr = None

        if colname == 'country':
            expr = "0"
            for ind in INDUSTRIES:
                expr += "+" + ind + "*" + str(indwgt[ind])
#                expr += "+" + ind
            print(expr)
        params.add(colname, value=0.0, expr=expr)

    print("Minimizing...")
    result = minimize(fcn2min, params, args=(loadings_df, returns_df))
    print("Result: " )
    if not result.success:
        print("ERROR: failed fit")
        exit(1)

    fRets_d = dict()
    for param in params:
        val = params[param].value
        error = params[param].stderr

        fRets_d[param] = val

        upper = val + error * 2
        lower = val - error * 2
        if upper * lower < 0:
            print("{} not significant: {}, {}".format(param, val, error))

    print("SEAN")
    print(result)
    print(result.residual)
    print(result.message)
    print(result.lmdif_message)
    print(result.nfev)
    print(result.ndata)

    residRets_na = result.residual
    return fRets_d, residRets_na

def fcn2min(params, x, data):
    """
    Objective function for lmfit minimization in factorize().

    Computes market-cap weighted residuals for factor model regression.
    Minimizing this function yields WLS factor return estimates.

    Args:
        params: lmfit.Parameters object containing factor returns to estimate
        x: Array (N x M) of factor loadings
        data: Series (N) of realized returns

    Returns:
        Array (N) of weighted residuals: sqrt(weight) * (predicted - actual)

    Notes:
        - Uses global weights_df (diagonal matrix of log market cap)
        - Model: predicted = x @ factor_returns
        - Residual = predicted - actual
        - Weighted residual = residual * weight (market cap weighting)
        - Returns diagonal of weight matrix multiplication
        - Type conversion to float64 for numerical stability

    Algorithm:
        1. Extract factor return values from params
        2. Reshape to column vector (M x 1)
        3. Compute predicted returns: x @ f
        4. Calculate residuals: predicted - actual
        5. Apply weights: residual * cap_weights
        6. Extract diagonal and flatten to 1D array

    Global Dependencies:
        - weights_df: Diagonal weight matrix from factorize()

    Mathematical Form:
        min_f sum_i [ weight_i * (sum_j x_ij * f_j - ret_i)^2 ]

    See Also:
        factorize() - Caller that uses this as objective function
    """
    # f1 = params['BBETANL_b'].value
    # f2 = params['SIZE_b'].value
    # print "f1: " + str(type(f1))
    # print f1
    ps = list()
    for param in params:
        val = params[param].value
        # if val is None: val = 0.0
        ps.append(val)
#        print "adding {} of {}".format(param, val)
    # print ps
    f = np.array(ps)
    f.shape = (len(params),1)
    # print "f: " + str(f.shape)
    # print f
    # print "x: " + str(type(x)) + str(x.shape)
    # print x
    model = np.dot(x, f)
 #   print "model: " + str(type(model)) + " " + str(model.shape)
    # print model
#    print "data: " + str(type(data)) + " " + str(data.shape)
    #
    # print data

    global weights_df
    cap_sq = weights_df.as_matrix()
#    cap_sq.shape = (cap_sq.shape[0], 1)

#    print model.shape
#    print data.values.shape
#    print cap_sq.shape
    # print "SEAN2"
    # print model
    # print data.values
    # print cap_sq

    #ret = np.multiply((model - data.values), cap_sq) / cap_sq.mean()
    ret = np.multiply((model - data.values), cap_sq)

    # print str(ret)
#    ret = model - data

    ret = ret.diagonal()
    # print ret.shape
#    ret = ret.as_matrix()
    ret.shape = (ret.shape[0], )

    #UGH XXX should really make sure types are correct at a higher level
    ret = ret.astype(np.float64, copy=False)

    # print
 #   print "ret: " + str(type(ret)) + " " + str(ret.shape)
    # print ret
    return ret

def mkt_ret(group):
    """
    Calculate market-cap weighted average return for a group.

    Computes cap-weighted mean return, typically for a date or sector group.

    Args:
        group: DataFrame group containing:
            - cum_ret1: 1-day cumulative return (from calc_forward_returns())
            - mkt_cap: Market capitalization

    Returns:
        Float representing weighted average return

    Notes:
        - Weights normalized by dividing market cap by 1e6 (millions)
        - Formula: sum(return_i * weight_i) / sum(weight_i)
        - Used for market benchmarking and sector return calculation
        - Typically applied via groupby().apply() pattern

    Example:
        daily_df.groupby(level='date').apply(mkt_ret)
        Returns market return for each date
    """
    d = group['cum_ret1']
    w = group['mkt_cap'] / 1e6
    res = (d * w).sum() / w.sum()
    return res
