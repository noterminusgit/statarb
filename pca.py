#!/usr/bin/env python
"""
PCA Factor Decomposition Module

Principal Component Analysis (PCA) module for decomposing stock returns into
systematic factors and measuring market-neutral alpha as residuals.

Key Concepts:
    - Decomposes returns into 5 principal components
    - Uses 30-day rolling windows for component calculation
    - Separates systematic (market) risk from idiosyncratic returns
    - Market-neutral alpha = actual returns - PCA predicted returns

Configuration:
    COMPONENTS: Number of principal components (default: 5)
    WINDOW: Rolling window size in days (default: 30)

Functions:
    calc_pca_daily(): Daily PCA decomposition and alpha calculation
        - Fits PCA on 30-day rolling window of returns
        - Projects current returns onto principal components
        - Calculates residuals (market-neutral returns)
        - Generates pca0, pca1, ..., pca4 factor loadings

    calc_pca_intra(): Intraday PCA for 30-minute bars
        - Time-slice specific PCA decomposition
        - Handles intraday return patterns
        - Supports lagged component analysis

PCA Alpha Interpretation:
    - pca0: Market-neutral return after removing first component
    - Higher components: Additional systematic factors (size, momentum, etc.)
    - Residual (alpha): Stock-specific return after factor removal

The module uses scikit-learn's PCA implementation with rolling window updates
to adapt to changing market conditions while maintaining statistical stability.

Usage:
    For large-cap stocks (market cap > $10B) to ensure liquid, tradable universe
    with sufficient cross-sectional diversity for robust PCA estimation.
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *
from calc import *

from sklearn.decomposition import PCA

COMPONENTS = 5
WINDOW = 30

cache = dict()

def calc_pca_daily(daily_df, horizon):
    """
    Perform daily PCA decomposition on stock returns to extract market-neutral alpha.

    Uses a rolling 30-day window to fit PCA models and calculate residuals as the
    difference between actual returns and PCA-predicted returns. The residuals represent
    market-neutral alpha after removing systematic factors.

    Mathematical Approach:
        1. Winsorize log returns to reduce outlier impact
        2. For each day t with window [t-30, t]:
           - Standardize returns (demean each stock's time series)
           - Fit PCA with 5 components on standardized returns
           - Transform actual returns to PCA space
           - Reconstruct predicted returns: predicted = U * components
           - Calculate residuals: alpha = actual - predicted
        3. Generate lagged features for horizon-day forecasting

    PCA Decomposition:
        Let R be the (stocks x days) return matrix over the window.
        PCA finds orthogonal components C such that:
            R_predicted = (R * C) * C^T

        Explained variance ratio shows how much return variance each component captures.
        First component typically explains 20-40% (market factor).
        Residuals are returns orthogonal to all 5 components.

    Args:
        daily_df (pd.DataFrame): Daily stock data with MultiIndex (date, sid).
            Required columns: 'log_ret', 'gdate'
        horizon (int): Forecast horizon in days. Creates lagged features pca1_B_ma
            through pca{horizon-1}_B_ma for regression analysis.

    Returns:
        pd.DataFrame: Input dataframe with added columns:
            - pca0: Raw residuals (actual - PCA predicted returns)
            - pca0_B: Winsorized residuals
            - pca0_B_ma: Market-adjusted residuals (same as pca0_B)
            - pca1_B_ma through pca{horizon-1}_B_ma: Lagged residuals

    Side Effects:
        Populates global cache dict with window dataframes keyed by date.
        This cache is reused by calc_pca_intra() for intraday analysis.

    Example:
        >>> daily_df = load_daily_data(start, end)
        >>> daily_df = calc_pca_daily(daily_df, horizon=3)
        >>> # Use pca0_B_ma as market-neutral alpha signal
        >>> # Use pca1_B_ma, pca2_B_ma as lagged features in regression

    Notes:
        - Requires at least 30 days of data to begin generating signals
        - First 30 days will have pca0 = 0 (insufficient window)
        - Explained variance printed for each day shows model fit quality
        - Higher explained variance = stronger systematic factors
        - Lower explained variance = more idiosyncratic returns
    """
    print("Caculating daily pca...")
    result_df = filter_expandable(daily_df)

    print("Calculating pca0...")
    result_df['log_ret_B'] = winsorize_by_date(result_df['log_ret'])

    unstacked_rets_df = result_df[['log_ret']].unstack()
    unstacked_rets_df.columns = unstacked_rets_df.columns.droplevel(0)
    unstacked_rets_df = unstacked_rets_df.fillna(0)

    result_df['pca0'] = 0
    pca = PCA(n_components=COMPONENTS)
    last_sigma = 99999.0
    for ii in range(WINDOW, len(unstacked_rets_df)):
        window_df = unstacked_rets_df[ii-WINDOW:ii]
        dt = window_df.index.max()
        sids = result_df.xs(dt, level=0).index

        window_df = window_df.replace([np.inf, -np.inf], np.nan)
        window_df = window_df.fillna(0)
        cache[dt] = window_df

        std_df = window_df.copy()
        for col in std_df.columns:
            if col in sids:
                rets = winsorize(std_df[col])
                std_df[col] = (rets - rets.mean()) 
            else:
                del std_df[col]
                del window_df[col]

        std_df = std_df.replace([np.inf, -np.inf], np.nan)
        std_df = std_df.fillna(0)

        window_df = window_df.T
    
        pcafit =  pca.fit(np.asarray(std_df.T))


        actual = window_df.loc[:,WINDOW-1]        
        pcarets = pca.transform(window_df)
        pr = np.dot(pcarets, pcafit.components_)
        pr = pr[:,[WINDOW-1]].reshape(-1)
        predicted =  pd.Series(pr, index=actual.index)
        
        predicted_sigma = predicted.std()
        resids = actual - predicted
    
        # if predicted_sigma > .01:
        #     resids = resids * 0.0

        print("PCA explained variance {}: {} {}".format(dt, predicted_sigma, pcafit.explained_variance_ratio_))

        resids.index = result_df[ result_df['gdate'] == dt].index
        result_df.loc[ result_df[ result_df['gdate'] == dt].index , 'pca0'] = resids 

        last_sigma = predicted_sigma

    print(result_df['pca0'].describe())
    result_df['pca0_B'] = winsorize_by_date(result_df['pca0'])
#    dategroups = result_df[['pca0_B', 'gdate']].groupby(['gdate'], sort=False).transform(demean)
#    result_df['pca0_B_ma'] = dategroups['pca0_B']
    result_df['pca0_B_ma'] = result_df['pca0_B']
    print("Calculated {} values".format(len(result_df)))

    print("Calulating lags...")
    for lag in range(1,horizon):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['pca'+str(lag)+'_B_ma'] = shift_df['pca0_B_ma']

    return result_df

def calc_pca_intra(intra_df):
    """
    Perform intraday PCA decomposition on 30-minute bar returns.

    Extends the daily PCA methodology to intraday data by combining the 30-day
    rolling window of daily returns (from cache) with current intraday returns
    for each 30-minute time slice. This allows capture of time-of-day specific
    patterns in market-neutral returns.

    Mathematical Approach:
        1. Calculate intraday return: overnight_ret + log(iclose/dopen)
        2. For each (date, timeslice) combination:
           - Retrieve 30-day window from cache (populated by calc_pca_daily)
           - Replace last column with current timeslice returns
           - Standardize the augmented window (demean each stock)
           - Fit PCA with 5 components
           - Calculate residuals for current timeslice
        3. Winsorize residuals by timeslice to create pcaC_B_ma

    Time-Slice Adaptation:
        Unlike daily PCA which uses a single model per day, intraday PCA
        fits a separate model for each 30-minute bar. This captures:
        - Opening volatility (9:30-10:30 higher variance)
        - Midday patterns (lunch hour effects)
        - Closing dynamics (15:30-16:00 increased trading)

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (timestamp, sid).
            Required columns: 'overnight_log_ret', 'iclose', 'dopen', 'gdate',
            'giclose_ts'

    Returns:
        pd.DataFrame: Input dataframe with added columns:
            - dret: Total return from prior close to current bar close
            - pcaC: Raw intraday PCA residuals
            - pcaC_B: Winsorized residuals by timeslice
            - pcaC_B_ma: Market-adjusted intraday residuals

    Side Effects:
        Requires global cache dict to be populated by calc_pca_daily().
        Will only process dates present in cache.

    Example:
        >>> daily_df = calc_pca_daily(daily_df, horizon=3)  # Populates cache
        >>> intra_df = calc_pca_intra(intra_df)
        >>> # Use pcaC_B_ma as intraday market-neutral signal
        >>> # Combine with daily pca0_B_ma for multi-timeframe analysis

    Notes:
        - Depends on calc_pca_daily() being called first to populate cache
        - Each timeslice gets a separate PCA fit (adaptive to intraday patterns)
        - Residuals winsorized by timeslice to handle time-varying volatility
        - Silent on explained variance (too many prints for all timeslices)
    """
    print("Calculating pca intra...")
    result_df = filter_expandable(intra_df)

    print("Calulating pcaC...")
    result_df['dret'] = result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))

    unstacked_rets_df = result_df[['dret']].unstack()
    unstacked_rets_df = unstacked_rets_df.replace([np.inf, -np.inf], np.nan)
    unstacked_rets_df = unstacked_rets_df.fillna(0)
    result_df['pcaC'] = 0

    pca = PCA(n_components=COMPONENTS)
    last_sigma = 99999.0
    for dt in cache.keys():
        window_df = cache[dt].T

        for ts in result_df[ result_df['gdate'] == dt ]['giclose_ts'].unique():
            today = unstacked_rets_df.loc[ts]
            today.index = today.index.droplevel(0)

            orig = result_df[ result_df['giclose_ts'] == ts ]
            today = today.loc[ orig.index.droplevel(0) ]            

            del window_df[window_df.columns.max()]
            window_df.index.name = 'sid'
            window_df = window_df.join(today, how='right')
            window_df = window_df.fillna(0)

            std_df = window_df.copy()
            for col in std_df.columns:
                rets = winsorize(std_df[col])
                std_df[col] = (rets - rets.mean()) 

            pcafit =  pca.fit(np.asarray(std_df))
#            print "PCA explained variance {}: {}".format(ts, pcafit.explained_variance_ratio_)
        
            actual = window_df.loc[:,WINDOW-1]        
            pcarets = pca.transform(window_df)
            pr = np.dot(pcarets, pcafit.components_)
            pr = pr[:,[WINDOW-1]].reshape(-1)
            predicted =  pd.Series(pr, index=actual.index)

            predicted_sigma = predicted.std()
            resids = actual - predicted

            # if predicted_sigma > .01:
            #     resids = resids * 0.0

            resids.index = result_df[ result_df['giclose_ts'] == ts].index
            result_df.loc[ result_df[ result_df['giclose_ts'] == ts].index , 'pcaC'] = resids 
            last_sigma = predicted_sigma

    print("Calulating pcaC_ma...")
    result_df['pcaC_B'] = winsorize_by_ts(result_df['pcaC'])
 #   demean = lambda x: (x - x.mean())
#    dategroups = result_df[['pcaC_B', 'giclose_ts']].groupby(['giclose_ts'], sort=False).transform(demean)
    result_df['pcaC_B_ma'] = result_df['pcaC_B']

    return result_df


def pca_fits(daily_df, intra_df, horizon, name, middate):
    """
    Fit regression models on PCA residuals and generate forecasts.

    Performs in-sample regression of PCA residuals against forward returns,
    then applies fitted coefficients to out-of-sample data to generate
    forecast signals. Handles both intraday (pcaC) and daily (pca0) residuals
    with horizon-specific and time-of-day specific coefficients.

    Regression Methodology:
        1. Intraday (pcaC_B_ma):
           - Regress pcaC_B_ma against forward returns by timeslice and horizon
           - Each 30-min window (9:30-10:30, 10:30-11:30, etc.) gets separate coef
           - Captures time-of-day patterns in alpha predictability

        2. Daily (pca0_B_ma):
           - Regress pca0_B_ma against forward returns for each lag 1..horizon
           - Calculate incremental coefficients: coef_lag = coef_h - coef_lag
           - Combines lagged daily signals with intraday signal

    Forecast Construction:
        forecast = pcaC_B_ma * pcaC_coef + sum(pca_lag_B_ma * pca_lag_coef)

        Where:
        - pcaC_coef varies by time-of-day (6 timeslices)
        - pca_lag_coef captures multi-day persistence

    Args:
        daily_df (pd.DataFrame): Daily data with pca0_B_ma, pca1_B_ma, etc.
        intra_df (pd.DataFrame): Intraday data with pcaC_B_ma
        horizon (int): Forecast horizon in days (typically 3)
        name (str): Name suffix for plot files (e.g., sector name or "")
        middate (datetime): Split date for in-sample vs out-of-sample.
            If None, uses entire dataset for both fit and forecast.

    Returns:
        pd.DataFrame: Out-of-sample intraday dataframe with added columns:
            - pcaC_B_ma_coef: Time-of-day specific coefficient for intraday residual
            - pca1_B_ma_coef through pca{horizon-1}_B_ma_coef: Daily lag coefficients
            - pca: Combined forecast signal (weighted sum of residuals)

    Side Effects:
        Creates regression diagnostic plots:
        - pca_intra_{name}_{dates}.png: Intraday fit quality by timeslice/horizon
        - pca_daily_{name}_{dates}.png: Daily fit quality by lag

    Example:
        >>> # Train on first half, forecast on second half
        >>> mid = datetime(2013, 6, 30)
        >>> forecast_df = pca_fits(daily_df, intra_df, horizon=3,
        ...                         name="tech", middate=mid)
        >>> # Use forecast_df['pca'] as final alpha signal
        >>> # Positive pca = expect outperformance, negative = underperformance

    Notes:
        - If middate is None, overfits by using same data for fit and forecast
        - Intraday coefficients: 6 values for 30-min windows from 9:30 to 16:00
        - Daily coefficients: horizon-1 values for lags
        - Incremental daily coefficients prevent double-counting persistence
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['pca'] = np.nan
    outsample_intra_df[ 'pcaC_B_ma_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_intra_df[ 'pca' + str(lag) + '_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'], dtype=float)
    fitresults_df = regress_alpha(insample_intra_df, 'pcaC_B_ma', horizon, True, 'intra')
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "pca_intra_"+name+"_" + df_dates(insample_intra_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    unstacked = outsample_intra_df[ ['ticker'] ].unstack()
    coefs = dict()
    coefs[1] = unstacked.between_time('09:30', '10:31').stack().index
    coefs[2] = unstacked.between_time('10:30', '11:31').stack().index
    coefs[3] = unstacked.between_time('11:30', '12:31').stack().index
    coefs[4] = unstacked.between_time('12:30', '13:31').stack().index
    coefs[5] = unstacked.between_time('13:30', '14:31').stack().index
    coefs[6] = unstacked.between_time('14:30', '15:59').stack().index
    print(fits_df.head())
    for ii in range(1,7):
        outsample_intra_df.loc[ coefs[ii], 'pcaC_B_ma_coef' ] = fits_df.loc['pcaC_B_ma'].loc[ii].loc['coef']
    
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'], dtype=float)
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'pca0_B_ma', lag, True, 'daily') 
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "pca_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.loc['pca0_B_ma'].loc[horizon].loc['coef']
#    outsample_intra_df[ 'pcaC_B_ma_coef' ] = coef0
    for lag in range(1,horizon):
        coef = coef0 - fits_df.loc['pca0_B_ma'].loc[lag].loc['coef'] 
        print("Coef{}: {}".format(lag, coef))
        outsample_intra_df[ 'pca'+str(lag)+'_B_ma_coef' ] = coef

    outsample_intra_df[ 'pca'] = outsample_intra_df['pcaC_B_ma'] * outsample_intra_df['pcaC_B_ma_coef']
    for lag in range(1,horizon):
        outsample_intra_df[ 'pca'] += outsample_intra_df['pca'+str(lag)+'_B_ma'] * outsample_intra_df['pca'+str(lag)+'_B_ma_coef']
    
    return outsample_intra_df

def calc_pca_forecast(daily_df, intra_df, horizon, middate):
    """
    Generate PCA-based alpha forecasts across entire universe.

    Wrapper function that applies pca_fits() to generate market-neutral
    alpha forecasts. Can optionally partition by sector for sector-specific
    PCA decomposition (currently disabled).

    Workflow:
        1. Call pca_fits() on full universe (sector partitioning commented out)
        2. Returns intraday dataframe with 'pca' forecast column

    Args:
        daily_df (pd.DataFrame): Daily stock data with PCA residuals
            (pca0_B_ma, pca1_B_ma, etc.)
        intra_df (pd.DataFrame): Intraday bar data with PCA residuals
            (pcaC_B_ma)
        horizon (int): Forecast horizon in days
        middate (datetime): Split date for in-sample/out-of-sample,
            or None to use full dataset

    Returns:
        pd.DataFrame: Intraday dataframe with 'pca' forecast column containing
            combined alpha signal from daily and intraday residuals

    Example:
        >>> daily_df = calc_pca_daily(daily_df, horizon=3)
        >>> intra_df = calc_pca_intra(intra_df)
        >>> intra_df = merge_intra_data(daily_df, intra_df)
        >>> forecast_df = calc_pca_forecast(daily_df, intra_df,
        ...                                  horizon=3, middate=None)
        >>> # Use forecast_df['pca'] in portfolio optimization

    Notes:
        - Sector-specific PCA is commented out but available for activation
        - Sector PCA would fit separate models per sector (e.g., tech, finance)
        - Currently uses single universe-wide PCA for all stocks
        - Returns intraday granularity for compatibility with qsim/osim
    """
    daily_results_df = daily_df
    intra_results_df = intra_df

    # results = list()
    # for sector_name in daily_results_df['sector_name'].unique():
    #     print "Running pca for sector {}".format(sector_name)
    #     sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    #     sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    #     result_df = pca_fits(sector_df, sector_intra_results_df, horizon, sector_name, middate)
    #     results.append(result_df)

#    result_df = pd.concat(results)

    result_df = pca_fits(daily_results_df, intra_results_df, horizon, "", middate)
    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--freq",action="store",dest="freq",default='30Min')
    parser.add_argument("--horizon",action="store",dest="horizon",default=3)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    freq = args.freq
    horizon = int(args.horizon)
    pname = "./pca" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)

    loaded = False
    try:        
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Could not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback, 1200)
        barra_df = load_barra(uni_df, start, end)    
        barra_df = transform_barra(barra_df)
        PRICE_COLS = ['close', 'overnight_log_ret']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        daily_df = merge_barra_data(price_df, barra_df)
        DBAR_COLS = ['close', 'dvolume', 'dopen']
        daybar_df = load_daybars(price_df[ ['ticker'] ], start, end, DBAR_COLS, freq)
        intra_df = merge_intra_data(daily_df, daybar_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    daily_df = calc_pca_daily(daily_df, horizon) 
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_df = pd.concat( [daily_df, forwards_df], axis=1)
    intra_df = calc_pca_intra(intra_df)
    intra_df = merge_intra_data(daily_df, intra_df)
    full_df = calc_pca_forecast(daily_df, intra_df, horizon, middate)

    dump_alpha(full_df, 'pca')
