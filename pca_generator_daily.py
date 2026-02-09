#!/usr/bin/env python
"""
PCA Generator Daily - Daily Residual Signal Extraction

This module generates alpha signals by decomposing daily return correlations
using Principal Component Analysis (PCA) and extracting stock-specific residuals.

Strategy Logic:
    1. Calculate overnight + previous day return (2-day combined return)
    2. Winsorize and demean returns by date
    3. Compute rolling exponentially-weighted correlation matrices (5-period halflife)
    4. Fit PCA model to capture top 4 principal components
    5. Project returns onto principal components
    6. Calculate residuals (actual returns - reconstructed returns from PCA)
    7. Use residuals as mean-reverting signals

Difference from pca_generator.py:
    - pca_generator.py: Intraday bars with simple rolling correlations
    - pca_generator_daily.py: Daily bars with exponentially-weighted correlations
    - This version combines overnight + previous day returns for smoother signals
    - Uses exponentially-weighted correlation with 5-day halflife (more responsive)

The exponentially-weighted correlation gives more weight to recent data, making
the PCA decomposition more adaptive to changing market conditions.

Data Requirements:
    - Daily prices with overnight_log_ret and today_log_ret
    - Universe: Top 1200 stocks by market cap (expandable universe)
    - Minimum 5 periods of history for correlation calculation

Parameters:
    COMPONENTS: Number of principal components to extract (default 4)
    CORR_LOOKBACK: Window for correlation calculation (default 20, not actively used)

Output:
    Daily DataFrame with PCA decomposition diagnostics printed to console

Note: Like pca_generator.py, the residual calculation is commented out (lines 51-56).
This appears to be analysis/diagnostic code rather than production signal generation.
Uncomment residual calculation lines to generate actual signals.

Usage:
    python pca_generator_daily.py --start=20130101 --end=20130630
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *
from calc import *

from sklearn.decomposition import PCA

COMPONENTS = 4
CORR_LOOKBACK = 20

def calc_pca_daily(daily_df):
    """
    Calculate daily PCA decomposition for correlation structure analysis.

    Formula:
        1. combined_ret = overnight_ret[t] + today_ret[t-1]
        2. log_ret_B = winsorize(combined_ret)
        3. log_ret_B_ma = demean(log_ret_B) by date
        4. corr_matrix[t] = exp_weighted_correlation(log_ret_B_ma, halflife=5)
        5. pca_fit = PCA(n_components=4).fit(corr_matrix[t])
        6. Print explained variance and average correlation diagnostics
        7. (Optional) residuals = ret - reconstruct(ret, pca_components)

    The 2-day combined return (overnight + previous day) smooths out intraday
    noise and captures more persistent return patterns for PCA decomposition.

    Exponentially-weighted correlation with 5-day halflife means:
        - Most recent day has weight 1.0
        - 5 days ago has weight 0.5
        - 10 days ago has weight 0.25
        - Adapts quickly to regime changes

    Args:
        daily_df: Daily DataFrame with overnight_log_ret and today_log_ret columns

    Returns:
        DataFrame with PCA analysis results (currently no signal columns added)

    Note: Residual calculation (lines 51-56) is commented out. This function
          primarily serves as analysis/diagnostics rather than signal generation.
          To generate 'pca0' signal column, uncomment residual calculation lines.
    """
    print("Caculating daily pca...")
    result_df = filter_expandable(daily_df)

    demean = lambda x: (x - x.mean())
    # result_df['log_ret_B'] = winsorize_by_date(result_df['log_ret'])
    # dategroups = result_df[['log_ret_B', 'gdate']].groupby(['gdate'], sort=False).transform(demean)
    # result_df['log_ret_B_ma'] = dategroups['log_ret_B']
    # result_df['log_ret_B_ma_l'] = result_df['log_ret_B_ma'].shift(1)

    result_df['yesterday_log_ret'] = result_df['today_log_ret'].shift(1)
    result_df['log_ret_B'] = winsorize_by_date(result_df['overnight_log_ret'] + result_df['yesterday_log_ret'])
    dategroups = result_df[['log_ret_B', 'gdate']].groupby(['gdate'], sort=False).transform(demean)
    result_df['log_ret_B_ma'] = dategroups['log_ret_B']


    # unstacked_df = result_df[['log_ret_B_ma_l']].unstack()
    # unstacked_df.columns = unstacked_df.columns.droplevel(0)
    # unstacked_df = unstacked_df.fillna(0)

    unstacked_overnight_df = result_df[['log_ret_B_ma']].unstack()
    unstacked_overnight_df.columns = unstacked_overnight_df.columns.droplevel(0)
    unstacked_overnight_df = unstacked_overnight_df.fillna(0)

    corr_matrices = rolling_ew_corr_pairwise(unstacked_overnight_df, 5)

    pca = PCA(n_components=COMPONENTS)
    lastpcafit = None
    for dt, grp in result_df.groupby(level='date'):
        df = corr_matrices.xs(dt, axis=0)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        rets = unstacked_overnight_df.xs(dt)
        print("Average correlation: {} {} {}".format(dt, df.unstack().mean(), df.unstack().std()))
        try:
            pcafit =  pca.fit(np.asarray(df))
        except:
            pcafit = lastpcafit
        print("PCA explained variance {}: {}".format(dt, pcafit.explained_variance_ratio_))

        pcarets = pca.transform(rets)
        pr = np.dot(pcarets, pcafit.components_)
        resids = rets - pr.T.reshape(len(df))
        result_df.loc[ grp.index, 'pca0' ] = resids.values
        lastpcafit = pcafit

    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--freq",action="store",dest="freq",default='5Min')
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    freq = args.freq
    start = dateparser.parse(start)
    end = dateparser.parse(end)

    uni_df = get_uni(start, end, lookback, 1200)
    PRICE_COLS = ['close', 'overnight_log_ret', 'today_log_ret']
    price_df = load_prices(uni_df, start, end, PRICE_COLS)
    result_df = calc_pca_daily(price_df)

 
