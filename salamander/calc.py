"""
Factor Calculation and Analysis Module (Python 3 - Salamander)

This module provides functions for calculating alpha factors, forward returns,
and statistical transformations used in the salamander statistical arbitrage system.

DIFFERENCES FROM MAIN CALC.PY:
    - Python 3 compatible syntax (print functions, dict.items())
    - Simplified factor configuration (5 Barra factors vs 13)
    - No industry factors (INDUSTRIES = [])
    - No proprietary factors (PROP_FACTORS = [])
    - Uses 'gvkey' index level instead of 'sid'
    - Uses 'mkt_cap' field instead of 'capitalization'
    - Removed deprecated pandas.stats.moments imports
    - Rolling window functions use modern pandas syntax (.rolling())
    - Deprecated pd.Panel handling removed

Key Features:
    - Forward return calculation at multiple horizons (1-N days)
    - Volume profile calculation for intraday patterns
    - Data winsorization to handle outliers
    - Rolling correlations and exponentially-weighted calculations
    - Factor decomposition via Weighted Least Squares regression
    - Barra-style factor risk model

Factor Categories (Simplified):
    BARRA_FACTORS (5 factors - reduced from main calc.py):
        - growth: Growth factor exposure
        - size: Market capitalization factor
        - divyild: Dividend yield factor
        - btop: Book-to-price ratio factor
        - momentum: Price momentum factor

    INDUSTRIES: Empty list (main calc.py has 58 industries)
    PROP_FACTORS: Empty list (main calc.py has 2 proprietary factors)
    ALL_FACTORS: Combined list of all factors (5 total)

Key Functions:
    calc_forward_returns(): Calculate cumulative returns at multiple horizons
    calc_vol_profiles(): Calculate intraday volume participation patterns
    winsorize(): Trim outliers at specified standard deviation levels
    winsorize_by_date(): Apply winsorization within each date
    winsorize_by_ts(): Apply winsorization within intraday timestamps
    calc_price_extras(): Calculate volatility ratios and volume metrics
    calc_factors(): Decompose returns into factor and residual components
    calc_intra_factors(): Intraday version of factor decomposition
    factorize(): Weighted Least Squares regression for factor estimation
    fcn2min(): Objective function for WLS minimization
    calc_resid_vol(): Calculate residual (idiosyncratic) volatility
    calc_factor_vol(): Calculate factor covariance matrix

The module uses pandas DataFrames with MultiIndex (date, gvkey) for efficient
time-series operations across securities.

Mathematical Background:
    Factor Model: r_i = sum_j(beta_ij * f_j) + epsilon_i
    where:
        r_i = return of security i
        beta_ij = loading of security i on factor j
        f_j = return of factor j
        epsilon_i = residual (idiosyncratic) return

    Weighted Least Squares:
        min_f sum_i[w_i * (r_i - sum_j(beta_ij * f_j))^2]
    where w_i = log(market_cap_i)
"""

import numpy as np
import pandas as pd
import gc

from scipy import stats
# from pandas.stats.api import ols
# from pandas.stats import moments
from lmfit import minimize, Parameters, Parameter, report_errors
from collections import defaultdict

from util import *

# INDUSTRIES = ['CONTAINR', 'HLTHSVCS', 'SPLTYRET', 'SPTYSTOR', 'DIVFIN', 'GASUTIL', 'BIOLIFE', 'SPTYCHEM', 'ALUMSTEL', 'AERODEF', 'COMMEQP', 'HOUSEDUR', 'CHEM', 'LEISPROD', 'AUTO', 'CONGLOM', 'HOMEBLDG', 'CNSTENG', 'LEISSVCS', 'OILGSCON', 'MEDIA', 'FOODPROD', 'PSNLPROD', 'OILGSDRL', 'SOFTWARE', 'BANKS', 'RESTAUR', 'FOODRET', 'ROADRAIL', 'APPAREL', 'INTERNET', 'NETRET', 'PAPER', 'WIRELESS', 'PHARMA', 'MGDHLTH', 'CNSTMACH', 'OILGSEQP', 'REALEST', 'COMPELEC', 'BLDGPROD', 'TRADECO', 'MULTUTIL', 'CNSTMATL', 'HLTHEQP', 'PRECMTLS', 'INDMACH', 'TRANSPRT', 'SEMIEQP', 'TELECOM', 'OILGSEXP', 'INSURNCE', 'AIRLINES', 'SEMICOND', 'ELECEQP', 'ELECUTIL', 'LIFEINS', 'COMSVCS', 'DISTRIB']
# BARRA_FACTORS = ['country', 'growth', 'size', 'sizenl', 'divyild', 'btop', 'earnyild', 'beta', 'resvol', 'betanl', 'momentum', 'leverage', 'liquidty']
BARRA_FACTORS = ['growth', 'size', 'divyild', 'btop', 'momentum']
# PROP_FACTORS = ['srisk_pct_z', 'rating_mean_z']
INDUSTRIES = []
PROP_FACTORS = []
ALL_FACTORS = BARRA_FACTORS + INDUSTRIES + PROP_FACTORS


def calc_vol_profiles(full_df):
    """
    Calculate intraday volume participation profiles at 15-minute intervals.

    Computes 21-day rolling median and standard deviation of dollar volume
    at each 15-minute timeslice from 09:45 to 16:00 (US market hours).

    Args:
        full_df: DataFrame with MultiIndex (date, gvkey) containing:
            - dvolume: Intraday bar volume
            - dvwap: Intraday bar VWAP (volume-weighted average price)
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
        - Groups by 'gvkey' instead of 'sid' (difference from main calc.py)
        - Uses modern .rolling() syntax (Python 3 compatible)

    Implementation:
        For each timeslice:
        1. Unstack to wide format (dates as index, gvkeys as columns)
        2. Filter to specific time using between_time()
        3. Restack to long format
        4. Calculate rolling statistics per security
        5. Merge back into main DataFrame
    """
    full_df['dpvolume_med_21'] = np.nan
    full_df['dpvolume_std_21'] = np.nan
    full_df['dpvolume'] = full_df['dvolume'] * full_df['dvwap']
    print("Calculating trailing volume profile...")
    for timeslice in ['09:45', '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15',
                      '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00',
                      '15:15', '15:30', '15:45', '16:00']:
        timeslice_df = full_df[['dpvolume', 'tradable_med_volume_21', 'close']]
        timeslice_df = timeslice_df.unstack().between_time(timeslice, timeslice).stack()
        timeslice_df = timeslice_df.dropna()
        if len(timeslice_df) == 0: continue
        timeslice_df['dpvolume_med_21'] = timeslice_df['dpvolume'].groupby(level='gvkey').apply(
            lambda x: x.shift(1).rolling(21).median())
        timeslice_df['dpvolume_std_21'] = timeslice_df['dpvolume'].groupby(level='gvkey').apply(
            lambda x: x.shift(1).rolling(21).std())
        m_df = timeslice_df.dropna()
        print(m_df.head())
        print("Average dvol frac at {}: {}".format(timeslice, (
                m_df['dpvolume_med_21'] / (m_df['tradable_med_volume_21'] * m_df['close'])).mean()))
        full_df.loc[timeslice_df.index, 'dpvolume_med_21'] = timeslice_df['dpvolume_med_21']
        full_df.loc[timeslice_df.index, 'dpvolume_std_21'] = timeslice_df['dpvolume_std_21']

    return full_df


def calc_price_extras(daily_df):
    """
    Calculate derived volatility and volume metrics.

    Computes ratio-based features useful for regime detection and
    alpha generation.

    Args:
        daily_df: DataFrame with MultiIndex (date, gvkey) containing:
            - volat_21: 21-day rolling volatility
            - volat_60: 60-day rolling volatility
            - tradable_volume: Daily tradable volume
            - shares_out: Shares outstanding
            - comp_volume: Composite volume

    Returns:
        DataFrame with added columns:
            - volat_ratio: Short/long volatility ratio (21-day / 60-day)
            - volume_ratio: Turnover ratio (volume / comp_volume, overwrites first calc)
            - volat_move: Day-over-day change in 21-day volatility

    Notes:
        - volume_ratio is calculated twice, second calculation overwrites first
          (first: volume/shares_out, second: volume/comp_volume)
        - volat_ratio > 1 indicates increasing volatility regime
        - volat_ratio < 1 indicates decreasing volatility regime
        - volat_move captures volatility momentum/acceleration

    Mathematical Formulas:
        volat_ratio = σ_21 / σ_60
        volume_ratio = V_t / V_comp
        volat_move = Δσ_21 = σ_21(t) - σ_21(t-1)
    """
    daily_df['volat_ratio'] = daily_df['volat_21'] / daily_df['volat_60']
    daily_df['volume_ratio'] = daily_df['tradable_volume'] / daily_df['shares_out']
    daily_df['volume_ratio'] = daily_df['tradable_volume'] / daily_df['comp_volume']
    daily_df['volat_move'] = daily_df['volat_21'].diff()
    return daily_df


def calc_forward_returns(daily_df, horizon):
    """
    Calculate forward cumulative returns at multiple horizons.

    Computes future returns from 1 to N days ahead for alpha backtesting
    and regression fitting. Uses negative shift to look forward in time.

    Args:
        daily_df: DataFrame with MultiIndex (date, gvkey) containing:
            - log_ret: Daily log returns
        horizon: Maximum forward horizon (typically 1-5 days)

    Returns:
        DataFrame with columns:
            - cum_ret1: 1-day forward cumulative log return
            - cum_ret2: 2-day forward cumulative log return
            - ... up to cum_ret{horizon}

    Notes:
        - Uses shift(-ii) to look forward ii days
        - Prevents look-ahead bias in alpha generation
        - Returns are per-security using groupby(level='gvkey')
        - Uses modern .rolling(ii).sum() syntax (Python 3 compatible)
        - Difference from main calc.py: uses 'gvkey' instead of 'sid'
        - Used by regress.py to fit alpha factors to future returns

    Mathematical Formula:
        cum_ret_h = sum_{i=1}^{h} log(P_{t+i} / P_{t+i-1})
                  = log(P_{t+h} / P_t)

    Example:
        If today is day T:
        - cum_ret1 = log_ret[T+1] = log(P_{T+1}/P_T)
        - cum_ret2 = log_ret[T+1] + log_ret[T+2] = log(P_{T+2}/P_T)
        - cum_ret5 = sum of log_ret from T+1 to T+5 = log(P_{T+5}/P_T)

    Implementation:
        For each horizon h:
        1. Calculate rolling sum of log returns over h days
        2. Shift backward by h days (shift(-h) looks forward)
        3. Unstack/stack to handle multi-index properly
        4. Store in results_df as cum_ret{h}
    """
    print("Calculating forward returns...")
    results_df = pd.DataFrame(index=daily_df.index)
    for ii in range(1, horizon + 1):
        retname = 'cum_ret' + str(ii)
        cum_rets = daily_df['log_ret'].groupby(level='gvkey').apply(lambda x: x.rolling(ii).sum())
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
        - Default std_level=5 is conservative (captures ~99.9999% of normal distribution)
        - Commonly used std_level values:
          * 3σ: 99.7% of normal distribution
          * 5σ: 99.9999% of normal distribution
        - Applied to alpha factors before regression to handle fat tails
        - Preserves ranking while limiting magnitude of extremes
        - More robust than truncation (keeps all data points)

    Mathematical Formula:
        For data point x:
        winsorized(x) = {
            μ + k*σ,  if x > μ + k*σ
            μ - k*σ,  if x < μ - k*σ
            x,        otherwise
        }
        where μ = mean(data), σ = std(data), k = std_level

    Example:
        If mean=0, std=1, std_level=5:
        - Values > 5 become 5
        - Values < -5 become -5
        - Values in [-5, 5] unchanged

        winsorize([1, 2, 3, 100], std_level=2)
        → [1, 2, 3, mean+2*std] (caps 100 to threshold)
    """
    result = data.copy()
    std = result.std() * std_level
    mean = result.mean()
    result[result > mean + std] = mean + std
    result[result < mean - std] = mean - std
    return result


def winsorize_by_date(data):
    """
    Apply winsorization within each date cross-section.

    Winsorizes data separately for each trading day, ensuring outlier
    detection is relative to same-day distribution across securities.

    Args:
        data: Series with MultiIndex (date, gvkey)

    Returns:
        Winsorized Series maintaining same index

    Notes:
        - Prevents outliers from one date affecting other dates
        - Critical for cross-sectional alpha signals
        - Uses default std_level=5 from winsorize()
        - Each date gets independent mean and std calculation
        - Preserves cross-sectional ranking within each date

    Example:
        If stock A has return of 50% on day 1 (outlier):
        - Only capped relative to day 1 distribution
        - Doesn't affect winsorization on other days
    """
    print("Winsorizing by day...")
    return data.groupby(level='date', sort=False).transform(winsorize)


def winsorize_by_ts(data):
    """
    Apply winsorization within each intraday timestamp.

    Winsorizes data separately for each intraday bar timestamp,
    ensuring outlier detection is relative to same-timestamp distribution.

    Args:
        data: Series with MultiIndex (iclose_ts, gvkey)

    Returns:
        Winsorized Series maintaining same index

    Notes:
        - Used for intraday factor calculations (qsim.py)
        - iclose_ts is intraday bar close timestamp
        - Prevents outliers from one timestamp affecting others
        - Each timestamp gets independent statistics

    Example:
        Winsorizes 10:00 bar separately from 10:30 bar, etc.
    """
    print("Winsorizing by day...")
    return data.groupby(level='iclose_ts', sort=False).transform(winsorize)


def winsorize_by_group(data, group):
    """
    Apply winsorization within custom groups.

    Winsorizes data separately for each value of specified grouping variable,
    enabling industry-neutral or sector-neutral outlier handling.

    Args:
        data: Series to winsorize
        group: Grouping variable (e.g., industry, sector, factor)

    Returns:
        Winsorized Series maintaining same index

    Notes:
        - Enables industry-neutral or sector-neutral winsorization
        - Prevents cross-contamination between groups
        - Useful for neutralizing sector effects
        - Each group gets independent mean and std

    Example:
        winsorize_by_group(returns, 'industry')
        - Tech stocks winsorized separately from financials
        - Prevents tech volatility from affecting finance thresholds
    """
    print("Winsorizing by day...")
    return data.groupby([group], sort=False).transform(winsorize)


def rolling_ew_corr_pairwise(df, halflife):
    """
    Calculate exponentially-weighted pairwise correlations for all column pairs.

    Computes time-varying correlation matrix using exponential weighting with
    specified halflife. Returns 3D structure (dict of dicts) with correlation time series.

    Args:
        df: DataFrame with time series columns
        halflife: Halflife for exponential weighting (days)

    Returns:
        Dict of dicts with dimensions (col1, col2) containing correlation series
        for each column pair

    Notes:
        - Uses pandas.Series.ewm().corr() (modern pandas API)
        - Span parameter = (halflife-1)/2.0 for ewm calculation
        - Returns full symmetric correlation matrix at each timestamp
        - Used for analyzing factor co-movement and regime changes
        - Note: Returns dict instead of deprecated Panel (Panel removed in pandas 1.0+)

    Mathematical Formula:
        ρ_t(X,Y) = Cov_t(X,Y) / (σ_t(X) * σ_t(Y))
        where Cov_t, σ_t use exponential weighting:
        Cov_t = α * Cov_{t-1} + (1-α) * (X_t - μ_X)(Y_t - μ_Y)
        α = exp(-ln(2) / halflife)

    Example:
        If df has columns ['alpha1', 'alpha2', 'alpha3']:
        Returns ret['alpha1']['alpha2'] = corr(alpha1, alpha2) time series
        ret['alpha1']['alpha2']['2020-01-01'] = 0.73
    """
    all_results = {}
    span = (halflife - 1) / 2.0
    for col1, left in df.items():
        all_results[col1] = col_results = {}
        for col2, right in df.items():
            # Modern pandas API: Series.ewm(span=...).corr(other_series)
            col_results[col2] = left.ewm(span=span, adjust=False).corr(right)

    return all_results


def push_data(df, col):
    """
    Shift specified column forward 1 period and merge with original DataFrame.

    Creates forward-looking (t+1) version of a column. WARNING: Can leak data
    across date boundaries, creating look-ahead bias if not used carefully.

    Args:
        df: DataFrame with MultiIndex (date, gvkey)
        col: Column name to shift forward

    Returns:
        DataFrame with original columns plus col_n (shifted forward 1 period)

    Notes:
        - Uses shift(-1) to look forward (negative shift)
        - Suffix '_n' indicates "next" period value
        - Difference from main calc.py: uses 'gvkey' instead of 'sid'
        - CAUTION: May push data to next calendar day, creating look-ahead bias
        - Use carefully in alpha generation to avoid future data leakage
        - Unstack/stack pattern ensures proper per-security shifting

    Example:
        If df has 'close' column:
        push_data(df, 'close') adds 'close_n' = tomorrow's close
        WARNING: Using close_n in today's trading decisions = look-ahead bias!

    Valid Use Cases:
        - Creating labels for machine learning (supervised learning)
        - Backtesting evaluation (not for signal generation)
    """
    # Careful, can push to next day...
    lagged_df = df[[col]].unstack(level='gvkey').shift(-1).stack()
    merged_df = pd.merge(df, lagged_df, left_index=True, right_index=True, sort=True, suffixes=['', '_n'])
    return merged_df


def lag_data(daily_df):
    """
    Lag entire DataFrame by 1 period and merge with original.

    Creates previous-period (t-1) versions of all columns, enabling
    momentum and reversal signal calculations.

    Args:
        daily_df: DataFrame with MultiIndex (date, gvkey)

    Returns:
        DataFrame with original columns plus lagged versions (suffix '_y')

    Notes:
        - Uses shift(1) to look backward (positive shift)
        - Suffix '_y' indicates lagged/yesterday's values
        - Safe for alpha generation (no look-ahead bias)
        - Commonly used to create momentum/reversal signals
        - Lags ALL columns in DataFrame

    Example:
        If df has columns ['close', 'volume']:
        Returns df with additional columns:
        - close_y: yesterday's close
        - volume_y: yesterday's volume

        Momentum signal: close / close_y - 1 (1-day return)
        Reversal signal: -(close / close_y - 1) (fade yesterday's move)
    """
    lagged_df = daily_df.unstack(level=-1).shift(1).stack()
    merged_df = pd.merge(daily_df, lagged_df, left_index=True, right_index=True, sort=True, suffixes=['', '_y'])
    return merged_df


def calc_med_price_corr(daily_df):
    """
    Calculate median price correlation. [NOT IMPLEMENTED]

    Args:
        daily_df: DataFrame with MultiIndex (date, gvkey)

    Returns:
        None (function stub)

    Notes:
        - Placeholder for future implementation
        - Likely intended for pair trading or sector correlation analysis
        - Could calculate rolling median correlation between stock pairs
        - Potential use: identify correlation regime changes
    """
    pass


def calc_resid_vol(daily_df):
    """
    Calculate Barra residual volatility (idiosyncratic/specific risk).

    Computes 20-day rolling standard deviation of Barra residual returns,
    measuring stock-specific risk after removing factor exposures.

    Args:
        daily_df: DataFrame with MultiIndex (date, gvkey) containing:
            - barraResidRet: Barra residual returns (from calc_factors())

    Returns:
        Series of residual volatility (idiosyncratic standard deviation)

    Notes:
        - Uses 20-day rolling window
        - Residual volatility = sqrt(var(barraResidRet))
        - Measures idiosyncratic risk not explained by factors
        - Lower residual vol → more factor-driven returns
        - Higher residual vol → more stock-specific risk
        - Used in risk models and portfolio optimization
        - Requires barraResidRet column from calc_factors() or calc_intra_factors()
        - Uses modern .rolling() syntax (Python 3 compatible)

    Mathematical Formula:
        σ_resid(t) = sqrt( (1/20) * sum_{i=0}^{19} ε_{t-i}^2 )
        where ε_t = barraResidRet_t (residual return)

    Interpretation:
        - High residual vol: Stock moves independently of factors
        - Low residual vol: Stock moves mostly with factor exposures
        - Total risk = factor risk + idiosyncratic risk

    Dependencies:
        Requires barraResidRet column from calc_factors()

    See Also:
        calc_factors() - Generates barraResidRet
        calc_factor_vol() - Calculates factor covariance
    """
    lookback = 20
    daily_df['barraResidVol'] = np.sqrt(daily_df['barraResidRet'].rolling(lookback).var())
    return daily_df['barraResidVol']


def calc_factor_vol(factor_df):
    """
    Calculate exponentially-weighted factor covariance matrix.

    Computes time-varying covariance matrix for all factor pairs using
    exponential weighting. Used for factor risk modeling in portfolio optimization.

    Args:
        factor_df: DataFrame with MultiIndex (date, factor) containing:
            - ret: Factor returns from calc_factors() or calc_intra_factors()

    Returns:
        Dictionary keyed by (factor1, factor2) tuples, values are Series
        of exponentially-weighted covariances over time

    Notes:
        - Halflife = 20 days for exponential weighting
        - Covers all 5 Barra factors (simplified from main calc.py's 73 factors)
        - Returns symmetric covariance matrix: cov(A,B) = cov(B,A)
        - Uses modern .ewm().cov() syntax (Python 3 compatible)
        - Commented-out code shows alternative: 20-day rolling covariance
        - Used by opt.py for portfolio risk calculations
        - More responsive to recent factor behavior than rolling window

    Mathematical Formula:
        Cov_t(f_i, f_j) = E_t[(f_i - μ_i)(f_j - μ_j)]
        where E_t uses exponential weighting:
        Cov_t = α * Cov_{t-1} + (1-α) * (f_{i,t} - μ_i)(f_{j,t} - μ_j)
        α = exp(-ln(2) / halflife)
        span = (halflife - 1) / 2.0

    Example:
        ret[('size', 'momentum')] = time series of size-momentum covariance
        ret[('size', 'size')] = time series of size factor variance
        ret[('growth', 'btop')] = time series of growth-value covariance

    Risk Model:
        Factor risk = β' * Σ_factor * β
        where Σ_factor is the covariance matrix from this function

    Dependencies:
        Requires factor returns from calc_factors() or calc_intra_factors()

    See Also:
        calc_factors() - Generates daily factor returns
        calc_intra_factors() - Generates intraday factor returns
        calc_resid_vol() - Calculates idiosyncratic risk
    """
    halflife = 20.0
    #    factors = factor_df.index.get_level_values('factor').unique()
    factors = ALL_FACTORS
    ret = {}
    for factor1 in factors:
        for factor2 in factors:
            key = (factor1, factor2)
            if key not in ret.keys():
                ret[key] = factor_df.xs(factor1, level=1)['ret'].ewm(span=(halflife - 1) / 2.0).cov(
                    factor_df.xs(factor2, level=1)['ret'].ewm(span=(halflife - 1) / 2.0))
    #                ret[key] = pd.rolling_cov(factor_df.xs(factor1, level=1)['ret'], factor_df.xs(factor2, level=1)['ret'], window=20)
    #                print "Created factor Cov on {} from {} to {}".format(key, min(ret[key].index), max(ret[key].index))
    return ret


weights_df = None  # Global variable used by factorize() and fcn2min() for market cap weighting


def create_z_score(daily_df, name):
    """
    Standardize column to z-score within each date cross-section.

    Converts raw factor values to standardized scores (mean=0, std=1) for
    each trading day, enabling cross-sectional comparison and ranking.

    Args:
        daily_df: DataFrame with MultiIndex (date, gvkey) containing:
            - name: Column to standardize
            - gdate: Grouping date column

    Returns:
        DataFrame with added column {name}_z containing z-scores

    Notes:
        - Z-score = (x - mean) / std, calculated per date
        - Enables rank-based alpha signals
        - Makes factors comparable across time periods
        - Groups by 'gdate' (not 'date'), allowing custom date grouping
        - Used for proprietary factors (not used in salamander - empty PROP_FACTORS)
        - In main calc.py: creates srisk_pct_z and rating_mean_z

    Mathematical Formula:
        z_i,t = (x_i,t - μ_t) / σ_t
        where:
        μ_t = mean(x_t) across all securities
        σ_t = std(x_t) across all securities

    Properties:
        - E[z_t] = 0 (zero mean)
        - Var[z_t] = 1 (unit variance)
        - Preserves ranking: rank(z) = rank(x)
        - Distribution shape unchanged (only location/scale)

    Example:
        create_z_score(df, 'srisk_pct') adds 'srisk_pct_z' column
        with cross-sectional z-scores per date:

        Date       gvkey   srisk_pct   srisk_pct_z
        2020-01-01 AAPL    0.05        1.2
        2020-01-01 MSFT    0.03        0.0
        2020-01-01 GOOG    0.02       -0.8
        (each date standardized independently)
    """
    zscore = lambda x: ((x - x.mean()) / x.std())
    indgroups = daily_df[[name, 'gdate']].groupby(['gdate'], sort=True).transform(zscore)
    daily_df[name + "_z"] = indgroups[name]
    return daily_df


def calc_factors(daily_df, barraOnly=False):
    """
    Decompose stock returns into factor returns and residual returns.

    Performs daily cross-sectional regression to separate returns into:
    1. Factor returns (systematic risk explained by common factors)
    2. Residual returns (idiosyncratic/stock-specific risk)

    Uses Weighted Least Squares with market cap weighting via lmfit minimization.

    Args:
        daily_df: DataFrame with MultiIndex (date, gvkey) containing:
            - log_ret: Daily log returns
            - All factor loadings from BARRA_FACTORS list
            - mkt_cap: Market capitalization for weighting
            - ind1: Industry classification (if INDUSTRIES not empty)
        barraOnly: If True, use only Barra factors (5 total in salamander).
                   If False, add proprietary factors (none in salamander).
                   Default: False.

    Returns:
        Tuple of (daily_df, factorRets_df):
            - daily_df: Input DataFrame with added 'barraResidRet' column
            - factorRets_df: MultiIndex (date, factor) with 'ret' column

    Notes:
        - Processes each date independently to avoid look-ahead bias
        - Uses log(market cap) as regression weights (larger caps = higher weight)
        - Simplified from main calc.py: only 5 Barra factors, no industries
        - Uses 'mkt_cap' field (main calc.py uses 'capitalization')
        - Uses 'gvkey' index level (main calc.py uses 'sid')
        - Memory cleanup with gc.collect() after each date
        - Prints regression diagnostics for monitoring

    DIFFERENCES FROM MAIN CALC.PY:
        - Simplified factors: 5 Barra (vs 13 Barra + 58 industries + 2 proprietary)
        - Uses 'gvkey' instead of 'sid'
        - Uses 'mkt_cap' instead of 'capitalization'
        - Uses 'ind1' instead of 'indname1'
        - No country factor constraint (INDUSTRIES is empty)
        - Python 3 syntax (print functions, dict.items())

    Factor Model:
        r_i,t = sum_j(β_i,j * f_j,t) + ε_i,t
        where:
        r_i,t = log return of stock i on date t
        β_i,j = loading of stock i on factor j (from Barra model)
        f_j,t = return of factor j on date t (estimated via WLS)
        ε_i,t = residual return (idiosyncratic component)

    Regression:
        Minimize: sum_i[w_i * (r_i - sum_j(β_i,j * f_j))^2]
        where w_i = log(mkt_cap_i)

    Algorithm:
        For each date:
        1. Extract factor loadings for all stocks
        2. Extract returns for all stocks
        3. Create diagonal weight matrix W = diag(log(mkt_cap))
        4. Call factorize() to solve WLS regression
        5. Store factor returns in factorRets_df
        6. Store residual returns in barraResidRet column
        7. Clean up memory

    Dependencies:
        - Requires Barra factor loadings from loaddata
        - Calls factorize() for WLS regression
        - Uses global weights_df in factorize()

    See Also:
        calc_intra_factors() - Intraday version using 30-min bars
        factorize() - WLS regression implementation
        fcn2min() - Objective function for minimization
        calc_resid_vol() - Uses barraResidRet output
    """
    print("Calculating factors...")

    allreturns_df = pd.DataFrame(columns=['barraResidRet'], index=daily_df.index)
    if barraOnly:
        factors = BARRA_FACTORS + INDUSTRIES
    else:
        # daily_df = create_z_score(daily_df, 'srisk_pct')
        # daily_df = create_z_score(daily_df, 'rating_mean')
        factors = ALL_FACTORS

    print("Total len: {}".format(len(daily_df)))
    cnt = 0
    cnt1 = 0
    factorrets = []
    for name, group in daily_df.groupby(level='date'):
        print("Regressing {}".format(name))
        cnt1 += len(group)
        print("Size: {} {}".format(len(group), cnt1))

        loadings_df = group[factors]
        loadings_df = loadings_df.reset_index().fillna(0)

        del loadings_df['gvkey']
        del loadings_df['date']

        #        print "loadings len {}".format(len(loadings_df))
        #        printloadings_df.head()

        returns_df = group['log_ret'].fillna(0)
        #       print "returns len {}".format(len(returns_df))

        #       printreturns_df.head()
        global weights_df
        weights_df = np.log(group['mkt_cap']).fillna(0)
        #        printweights_df.head()
        weights_df = pd.DataFrame(np.diag(weights_df))

        #        print "weights len {}".format(len(weights_df))
        indwgt = {}
        capsum = (group['mkt_cap'] / 1e6).sum()
        for ind in INDUSTRIES:
            indwgt[ind] = (group[group['ind1'] == ind]['mkt_cap'] / 1e6).sum() / capsum
        #        printreturns_df.head()

        fRets, residRets = factorize(loadings_df, returns_df, weights_df, indwgt)
        print("Factor Returns:")
        #        printfRets
        #        printresidRets

        cnt += len(residRets)
        print("Running tally: {}".format(cnt))
        fdf = pd.DataFrame([[i, v] for i, v in fRets.items()], columns=['factor', 'ret'])
        fdf['date'] = name
        factorrets.append(fdf)
        allreturns_df.loc[group.index, 'barraResidRet'] = residRets

        fRets = residRets = None
        gc.collect()

    #    printallreturns_df.tail()
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
        intra_df: DataFrame with MultiIndex (date, gvkey) containing:
            - overnight_log_ret: Overnight return (close to open)
            - close: Intraday bar close price
            - dopen: Daily open price
            - All factor loadings (BARRA_FACTORS)
            - mkt_cap: Market capitalization for weighting
            - ind1: Industry classification
        barraOnly: If True, use only Barra factors (5 in salamander).
                   If False, include proprietary factors (none in salamander).
                   Default: False.

    Returns:
        Tuple of (intra_df, factorRets_df):
            - intra_df: Input DataFrame with added 'barraResidRetI' column
            - factorRets_df: MultiIndex (date, factor) with 'ret' column

    Notes:
        - Processes each date independently (not intraday timestamp)
        - Return calculation: overnight_log_ret + log(close/dopen)
        - Uses log(market cap) as regression weights
        - Groups by 'date' level (not 'iclose_ts' like main calc.py)
        - Suffix 'I' (barraResidRetI) indicates intraday
        - Used by qsim.py for intraday backtesting

    DIFFERENCES FROM MAIN CALC.PY:
        - Groups by 'date' instead of 'iclose_ts'
        - Uses 'close' instead of 'iclose'
        - Simplified factors (5 vs 73)
        - Uses 'gvkey' instead of 'sid'
        - Uses 'mkt_cap' instead of 'capitalization'
        - Uses 'ind1' instead of 'indname1'

    Return Calculation:
        Total intraday return = overnight component + intraday component
        r_intra = log(open_t / close_{t-1}) + log(close_bar / open_t)
                = overnight_log_ret + log(close/dopen)

    Factor Model:
        Same as calc_factors():
        r_i = sum_j(β_i,j * f_j) + ε_i

    Algorithm:
        For each date:
        1. Calculate total return (overnight + intraday)
        2. Extract factor loadings
        3. Create market cap weight matrix
        4. Call factorize() for WLS regression
        5. Store factor returns and residuals

    Dependencies:
        - Requires intraday bar data from loaddata
        - Calls factorize() for regression fitting
        - Uses global weights_df

    See Also:
        calc_factors() - Daily version
        factorize() - WLS regression implementation
        fcn2min() - Objective function
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
    for name, group in intra_df.groupby(level='date'):
        print("Regressing {}".format(name))
        cnt1 += len(group)
        print("Size: {} {}".format(len(group), cnt1))

        loadings_df = group[factors]
        loadings_df = loadings_df.reset_index().fillna(0)

        del loadings_df['gvkey']
        del loadings_df['date']

        #        print "loadings len {}".format(len(loadings_df))
        #        printloadings_df.head()

        returns_df = (group['overnight_log_ret'] + np.log(group['close'] / group['dopen'])).fillna(0)
        #       print "returns len {}".format(len(returns_df))

        #       printreturns_df.head()
        global weights_df
        weights_df = np.log(group['mkt_cap']).fillna(0)
        #        printweights_df.head()
        weights_df = pd.DataFrame(np.diag(weights_df))
        #        print "weights len {}".format(len(weights_df))
        indwgt = dict()
        capsum = (group['mkt_cap'] / 1e6).sum()
        for ind in INDUSTRIES:
            indwgt[ind] = (group[group['ind1'] == ind]['mkt_cap'] / 1e6).sum() / capsum
        #        printreturns_df.head()

        fRets, residRets = factorize(loadings_df, returns_df, weights_df, indwgt)
        print("Factor Returns:")
        print(fRets)
        #        printresidRets

        cnt += len(residRets)
        print("Running tally: {}".format(cnt))
        fdf = pd.DataFrame([[i, v] for i, v in fRets.items()], columns=['factor', 'ret'])
        fdf['date'] = name
        factorrets.append(fdf)
        allreturns_df.loc[group.index, 'barraResidRetI'] = residRets

        fRets = residRets = None
        gc.collect()

    #    printallreturns_df.tail()
    factorRets_df = pd.concat(factorrets).set_index(['date', 'factor']).fillna(0)
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
            Columns are factor names (e.g., 'size', 'momentum')
            Rows are securities (one row per security on given date)
        returns_df: Series (N securities) of realized returns
        weights_df: DataFrame (N x N) diagonal matrix of log(market cap) weights
            Used for WLS - larger caps get higher weight in regression
        indwgt: Dict mapping industry name to cap-weighted fraction
            Used for country factor constraint (empty in salamander)

    Returns:
        Tuple of (fRets_d, residRets_na):
            - fRets_d: Dict mapping factor name to estimated factor return
            - residRets_na: Array of residual returns (N securities)

    Notes:
        - Uses lmfit.minimize() for nonlinear least squares
        - Minimization target: weighted residual from fcn2min()
        - Country factor constrained to be cap-weighted sum of industries
          (not applicable in salamander since INDUSTRIES is empty)
        - Prints statistical significance (2-sigma test) for each factor
        - Exits on optimization failure (result.success == False)
        - Returns stderr for uncertainty quantification
        - Uses global weights_df variable shared with fcn2min()

    Constraint (when INDUSTRIES non-empty):
        country = sum_i(industry_i * indwgt_i) for all industries
        Ensures country factor represents market-wide movement

    Mathematical Model:
        r = X * f + ε
        where:
        r = (N x 1) vector of returns
        X = (N x M) matrix of factor loadings
        f = (M x 1) vector of factor returns (to estimate)
        ε = (N x 1) vector of residuals

        Weighted Least Squares:
        min_f sum_i[w_i * (r_i - sum_j(X_ij * f_j))^2]

    Algorithm:
        1. Initialize Parameters object with factor returns = 0
        2. Add country constraint (if industries present)
        3. Minimize fcn2min() using lmfit
        4. Extract factor returns from optimization result
        5. Test statistical significance (2-sigma test):
           - If confidence interval crosses zero, factor not significant
           - interval = [f - 2*stderr, f + 2*stderr]
        6. Return factor returns dict and residual array

    Optimization Details:
        - Method: Levenberg-Marquardt (lmfit default)
        - Objective: fcn2min() - weighted squared residuals
        - Parameters: One per factor
        - Constraints: Linear (country = industry sum)

    Statistical Significance Test:
        For each factor f with stderr σ:
        - Upper bound = f + 2σ
        - Lower bound = f - 2σ
        - If upper * lower < 0, bounds cross zero → not significant

    Dependencies:
        - Uses global weights_df for fcn2min() calculation
        - Calls fcn2min() as objective function
        - Requires lmfit library

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
    print("Result: ")
    if not result.success:
        print("ERROR: failed fit")
        exit(1)

    fRets_d = dict()
    for param in result.params:
        val = result.params[param].value
        error = result.params[param].stderr

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
            Keys are factor names (e.g., 'size', 'momentum')
            Values are current estimates (updated during optimization)
        x: DataFrame (N x M) of factor loadings
            N = number of securities
            M = number of factors
            x[i,j] = loading of security i on factor j
        data: Series (N) of realized returns
            data[i] = actual return of security i

    Returns:
        Array (N) of weighted residuals for minimization:
            residual_i = sqrt(weight_i) * (predicted_i - actual_i)

    Notes:
        - Uses global weights_df (diagonal matrix of log market cap)
        - Model: predicted = x @ factor_returns
        - Residual = predicted - actual
        - Weighted residual = residual * weight (market cap weighting)
        - Returns diagonal of weight matrix multiplication
        - Type conversion to float64 for numerical stability
        - Called repeatedly by lmfit.minimize()

    Global Dependencies:
        - weights_df: Diagonal weight matrix from factorize()
          Must be set before calling this function

    Mathematical Form:
        Minimize: sum_i [ w_i * (sum_j(x_ij * f_j) - r_i)^2 ]
        where:
        w_i = log(mkt_cap_i) = weight for security i
        x_ij = loading of security i on factor j
        f_j = return of factor j (being estimated)
        r_i = actual return of security i

    Algorithm:
        1. Extract factor return values from params → vector f
        2. Reshape f to column vector (M x 1)
        3. Compute predicted returns: x @ f → (N x 1)
        4. Calculate residuals: predicted - actual → (N x 1)
        5. Create weight matrix from global weights_df
        6. Apply weights: W @ residual → (N x N)
        7. Extract diagonal to get (N x 1)
        8. Flatten to 1D array and convert to float64

    WLS Intuition:
        - Larger cap stocks get higher weight (w_i = log(mkt_cap_i))
        - Regression "cares more" about fitting large caps correctly
        - Minimizes cap-weighted prediction error

    Numerical Details:
        - Uses matrix diagonal trick to avoid full matrix multiplication
        - Type conversion to float64 prevents overflow in optimizer
        - Commented code shows alternative normalization (divide by mean)

    Example:
        If 3 stocks with log caps [10, 12, 11]:
        - Stock 2 (log_cap=12) has highest weight
        - Optimizer prioritizes fitting stock 2's return
        - Residuals scaled by log caps before summing

    See Also:
        factorize() - Caller that uses this as objective function
        calc_factors() - Top-level caller for daily returns
    """
    # f1 = params['BBETANL_b'].value
    # f2 = params['SIZE_b'].value
    # print "f1: " + str(type(f1))
    # printf1
    ps = list()
    for param in params:
        val = params[param].value
        # if val is None: val = 0.0
        ps.append(val)
    #        print "adding {} of {}".format(param, val)
    # printps
    f = np.array(ps)
    f.shape = (len(params), 1)
    # print "f: " + str(f.shape)
    # printf
    # print "x: " + str(type(x)) + str(x.shape)
    # printx
    model = np.dot(x, f)
    #   print "model: " + str(type(model)) + " " + str(model.shape)
    # printmodel
    #    print "data: " + str(type(data)) + " " + str(data.shape)
    #
    # printdata

    global weights_df
    cap_sq = weights_df.as_matrix()
    #    cap_sq.shape = (cap_sq.shape[0], 1)

    #    printmodel.shape
    #    printdata.values.shape
    #    printcap_sq.shape
    # print "SEAN2"
    # printmodel
    # printdata.values
    # printcap_sq

    # ret = np.multiply((model - data.values), cap_sq) / cap_sq.mean()
    ret = np.multiply((model - data.values), cap_sq)

    # printstr(ret)
    #    ret = model - data

    ret = ret.diagonal()
    # printret.shape
    #    ret = ret.as_matrix()
    ret.shape = (ret.shape[0],)

    # UGH XXX should really make sure types are correct at a higher level
    ret = ret.astype(np.float64, copy=False)

    # print
    #   print "ret: " + str(type(ret)) + " " + str(ret.shape)
    # printret
    return ret


def mkt_ret(group):
    """
    Calculate market-cap weighted average return for a group.

    Computes cap-weighted mean return, typically for a date or sector group.
    Used for market benchmarking and performance attribution.

    Args:
        group: DataFrame group containing:
            - cum_ret1: 1-day cumulative return (from calc_forward_returns())
            - mkt_cap: Market capitalization

    Returns:
        Float representing weighted average return for the group

    Notes:
        - Weights normalized by dividing market cap by 1e6 (millions)
        - Formula: sum(return_i * weight_i) / sum(weight_i)
        - Used for market benchmarking and sector return calculation
        - Typically applied via groupby().apply() pattern
        - Difference from main calc.py: uses 'mkt_cap' (not 'capitalization')

    Mathematical Formula:
        r_market = sum_i(r_i * w_i) / sum_i(w_i)
        where:
        r_i = return of stock i
        w_i = market cap of stock i (in millions)

    Example Usage:
        # Calculate market return for each date
        daily_df.groupby(level='date').apply(mkt_ret)

        # Calculate sector returns
        daily_df.groupby(['date', 'sector']).apply(mkt_ret)

    Returns:
        Date       Market Return
        2020-01-01    0.015  (1.5% market return)
        2020-01-02   -0.008  (-0.8% market return)

    Use Cases:
        - Market benchmark for alpha evaluation
        - Sector rotation analysis
        - Market-neutral strategy verification
        - Performance attribution
    """
    d = group['cum_ret1']
    w = group['mkt_cap'] / 1e6
    res = (d * w).sum() / w.sum()
    return res
