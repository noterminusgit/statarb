#!/usr/bin/env python
"""
PCA Generator - Intraday Residual Signal Extraction

This module generates alpha signals by decomposing intraday return correlations
using Principal Component Analysis (PCA) and extracting stock-specific residuals.

Strategy Logic:
    1. Calculate intraday bar-to-bar log returns
    2. Compute rolling correlation matrices (10-period window)
    3. Fit PCA model to capture top 4 principal components
    4. Project returns onto principal components
    5. Calculate residuals (actual returns - reconstructed returns from PCA)
    6. Use residuals as mean-reverting signals

The intuition is that principal components capture systematic market and sector
movements. The residuals represent stock-specific noise that should mean-revert.

Relationship to pca.py:
    - pca.py: Implements PCA decomposition for analysis and backtesting
    - pca_generator.py: Generates PCA-based signals for live trading
    - pca_generator_daily.py: Daily version of this strategy

Data Requirements:
    - Intraday bars: 5-min or 15-min frequency with close prices
    - Universe: Top 1200 stocks by market cap (expandable universe)
    - Minimum 10 periods of history for correlation calculation

Parameters:
    COMPONENTS: Number of principal components to extract (default 4)
    CORR_LOOKBACK: Window for correlation calculation (default 20, but code uses 10)
    freq: Bar frequency for intraday data (default '5Min')

Output:
    'pcaC_B_ma' column with PCA residual signal (winsorized and demeaned)

Note: The code has a bug - explained_variance is printed but residuals ('pcaC')
are never actually calculated (commented out lines 43-46). This appears to be
incomplete implementation that would require the residual calculation to be
uncommented.

Usage:
    python pca_generator.py --start=20130101 --end=20130630 --freq=5Min
"""

from regress import *
from loaddata import *
from util import *
from calc import *

from sklearn.decomposition import PCA

COMPONENTS = 4
CORR_LOOKBACK = 20

def calc_pca_intra(intra_df):
    """
    Calculate intraday PCA residual signals.

    Formula:
        1. logret[t] = log(close[t] / close[t-1])
        2. corr_matrix[t] = rolling_correlation(logret, window=10)
        3. pca_fit = PCA(n_components=4).fit(corr_matrix[t])
        4. residuals = logret - reconstruct(logret, pca_components)
        5. Winsorize residuals by timestamp
        6. Return winsorized residuals as signal

    The PCA captures the top 4 systematic factors in the correlation structure.
    Residuals represent idiosyncratic movements that should mean-revert.

    Args:
        intra_df: Intraday DataFrame with iclose prices indexed by (ticker, iclose_ts)

    Returns:
        DataFrame with 'pcaC_B_ma' column containing PCA residual signal

    Note: Current implementation has residual calculation commented out (lines 43-46).
          The pcaC column is never actually populated, making this incomplete.
          Uncomment those lines to get actual residual signals.
    """
    print "Calculating pca intra..."
    result_df = filter_expandable(intra_df)

    result_df['iclose_l'] = result_df['iclose'].shift(1)
    result_df['logret'] = np.log(result_df['iclose']/result_df['iclose_l'])

    unstacked_rets_df = result_df[['logret']].unstack()
    unstacked_rets_df = unstacked_rets_df.replace([np.inf, -np.inf], np.nan)
    unstacked_rets_df = unstacked_rets_df.fillna(0)

    corr_matrices = pd.rolling_corr_pairwise(unstacked_rets_df, 10)

    pca = PCA(n_components=COMPONENTS)
    lastpcafit = None

    for dt, grp in result_df.groupby(level='iclose_ts'):
        df = corr_matrices.xs(dt, axis=0)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        rets = unstacked_rets_df.xs(dt)
        ids = rets.index.droplevel(0)
        df = df[ ids ].ix[ ids ]

        try:
            pcafit =  pca.fit(np.asarray(df))
        except:
            pcafit = lastpcafit
        print "PCA explained variance {}: {}".format(dt, pcafit.explained_variance_ratio_)
#        pcarets = pca.transform(rets)
#        pr = np.dot(pcarets, pcafit.components_)
#        resids = rets - pr.T.reshape(len(df))
#        result_df.ix[ grp.index, 'pcaC' ] = resids.values
        lastpcafit = pcafit

    print "Calulating pcaC_ma..."
    result_df['pcaC_B'] = winsorize_by_ts(result_df['pcaC'])
 #   demean = lambda x: (x - x.mean())
#    dategroups = result_df[['pcaC_B', 'giclose_ts']].groupby(['giclose_ts'], sort=False).transform(demean)
    result_df['pcaC_B_ma'] = result_df['pcaC_B']

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
    PRICE_COLS = ['close', 'overnight_log_ret']
    price_df = load_prices(uni_df, start, end, PRICE_COLS)
    DBAR_COLS = ['close', 'dvolume', 'dopen']
    daybar_df = load_daybars(price_df[ ['ticker'] ], start, end, DBAR_COLS, freq)
    intra_df = merge_intra_data(daily_df, daybar_df)
    intra_df = calc_pca_intra(intra_df)

 
