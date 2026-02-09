#!/usr/bin/env python
"""
Residual Return Betting (RRB) Alpha Strategy

This module generates alpha signals based on Barra factor model residual returns.
The strategy exploits mean reversion in stock-specific (idiosyncratic) returns
after controlling for systematic factor exposure.

Strategy Logic:
    1. Calculate Barra factor model residuals (barraResidRet)
    2. Winsorize and demean residuals by date
    3. Create lagged features for multi-day signals
    4. Regress lagged residuals against forward returns
    5. Apply coefficients to generate intraday forecasts

Economic Intuition:
    - Barra residuals represent idiosyncratic return after removing exposure
      to market, size, value, momentum, and other systematic factors
    - Large residual returns indicate temporary stock-specific shocks
    - These shocks should mean-revert as the market digests information
    - Unlike pca.py, this uses the full Barra model rather than simple PCA

Relationship to Other Modules:
    - pca.py: PCA decomposition of returns (data-driven factors)
    - rrb.py: Barra model residuals (theory-driven factors)
    - Both capture idiosyncratic mean reversion, but Barra is more interpretable

Data Requirements:
    - Full Barra factor loadings (all factors, not just subset)
    - Daily and intraday returns for residual calculation
    - calc_factors() and calc_intra_factors() for Barra decomposition
    - Minimum 3+ days of history to populate lagged features

Parameters:
    horizon: Number of lag days to include (default 3)
    freq: Intraday bar frequency (default '15Min')

Output:
    'rrb' column with residual-based forward return forecast

Note: Current implementation excludes Energy sector (line 94). This may be
due to different dynamics in commodity-driven sectors. Consider sector-specific
calibration for production use.

Usage:
    python rrb.py --start=20130101 --end=20130630 --mid=20130401 --horizon=3
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *
from calc import *

def calc_rrb_daily(daily_df, horizon):
    """
    Calculate daily Barra residual return signals.

    Formula:
        1. rrb0 = barraResidRet (from calc_factors())
        2. rrb0_B = winsorize(rrb0) by date
        3. rrb0_B_ma = demean(rrb0_B) by date
        4. Create lagged features rrb{lag}_B_ma for lag=1 to horizon-1

    The barraResidRet is calculated in calc.py using:
        residual = return - sum(factor_loading[i] * factor_return[i])

    Winsorization prevents extreme outliers from dominating the signal.
    Date demeaning removes any daily market-wide residual drift.

    Args:
        daily_df: Daily DataFrame with barraResidRet column (must run calc_factors first)
        horizon: Number of lag days to create (default 3)

    Returns:
        DataFrame with rrb0_B_ma through rrb{horizon-1}_B_ma columns
    """
    print("Caculating daily rrb...")
    result_df = filter_expandable(daily_df)

    print("Calculating rrb0...")
    result_df['rrb0'] = result_df['barraResidRet']
    print(result_df['rrb0'].head())
    result_df['rrb0_B'] = winsorize_by_date(result_df['rrb0'])

    demean = lambda x: (x - x.mean())
    dategroups = result_df[['rrb0_B', 'gdate']].groupby(['gdate'], sort=False).transform(demean)
    result_df['rrb0_B_ma'] = dategroups['rrb0_B']
    print("Calculated {} values".format(len(result_df)))

    print("Calulating lags...")
    for lag in range(1,horizon):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['rrb'+str(lag)+'_B_ma'] = shift_df['rrb0_B_ma']

    return result_df

def calc_rrb_intra(intra_df):
    """
    Calculate intraday Barra residual return signals.

    Formula:
        1. rrbC = barraResidRetI (from calc_intra_factors())
        2. rrbC_B = winsorize(rrbC) by timestamp
        3. rrbC_B_ma = demean(rrbC_B) by timestamp

    The intraday residual uses the same Barra factor model but applied to
    intraday bar returns. This captures intraday mean reversion in residuals.

    Args:
        intra_df: Intraday DataFrame with barraResidRetI column
                  (must run calc_intra_factors first)

    Returns:
        DataFrame with rrbC_B_ma column containing intraday residual signal
    """
    print("Calculating rrb intra...")
    result_df = filter_expandable(intra_df)

    print("Calulating rrbC...")
    result_df['rrbC'] = result_df['barraResidRetI']
    result_df['rrbC_B'] = winsorize_by_ts(result_df['rrbC'])

    print(result_df['rrbC'].tail())

    print("Calulating rrbC_ma...")
    demean = lambda x: (x - x.mean())
    dategroups = result_df[['rrbC_B', 'giclose_ts']].groupby(['giclose_ts'], sort=False).transform(demean)
    result_df['rrbC_B_ma'] = dategroups['rrbC_B']

    return result_df

def rrb_fits(daily_df, intra_df, horizon, name, middate):
    """
    Fit RRB regression model and generate intraday forecasts.

    Performs daily-frequency WLS regression of rrb0_B_ma against forward returns
    at multiple horizons. Applies coefficients to intraday data for high-frequency
    trading on residual mean reversion.

    The coefficient structure:
        - coef[0] = total effect at horizon
        - coef[lag] = incremental effect of lag-day residuals
        - Final coef[lag] = coef[0] - coef[lag] (differential effect)

    Combines current intraday residual (rrbC_B_ma) with lagged daily residuals
    for a comprehensive signal.

    Args:
        daily_df: Daily DataFrame with rrb signals and forward returns
        intra_df: Intraday DataFrame with rrbC signal
        horizon: Maximum lag to include in regressions (default 3)
        name: String identifier for plot filenames
        middate: Split date between in-sample and out-of-sample

    Returns:
        Intraday DataFrame with 'rrb' forecast column
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['rrb'] = np.nan
    outsample_intra_df[ 'rrbC_B_ma_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_intra_df[ 'rrb' + str(lag) + '_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        print(insample_daily_df.head())
        fitresults_df = regress_alpha(insample_daily_df, 'rrb0_B_ma', lag, True, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "rrb_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    coef0 = fits_df.loc['rrb0_B_ma'].loc[horizon].loc['coef']
    outsample_intra_df[ 'rrbC_B_ma_coef' ] = coef0
    print("Coef0: {}".format(coef0))
    for lag in range(1,horizon):
        coef = coef0 - fits_df.loc['rrb0_B_ma'].loc[lag].loc['coef']
        print("Coef{}: {}".format(lag, coef))
        outsample_intra_df[ 'rrb'+str(lag)+'_B_ma_coef' ] = coef

    outsample_intra_df['rrb'] = outsample_intra_df['rrbC_B_ma'] * outsample_intra_df['rrbC_B_ma_coef']
#    outsample_intra_df['rrb_b'] = 0
    for lag in range(1,horizon):
        outsample_intra_df[ 'rrb'] += outsample_intra_df['rrb'+str(lag)+'_B_ma'] * outsample_intra_df['rrb'+str(lag)+'_B_ma_coef']

    return outsample_intra_df

def calc_rrb_forecast(daily_df, intra_df, horizon, middate):
    """
    Master function to calculate RRB forecasts (excluding Energy sector).

    Runs the full pipeline on non-Energy stocks only. The Energy sector is
    excluded because its dynamics may be driven more by commodity prices
    than factor model residuals.

    Pipeline:
    1. Takes pre-calculated daily and intraday Barra residuals
    2. Fits regression model on non-Energy stocks
    3. Generates intraday forecasts

    Note: To include Energy or other sector-specific models, uncomment the
    sector-specific regression lines and concatenate results.

    Args:
        daily_df: Daily DataFrame with rrb signals (already calculated)
        intra_df: Intraday DataFrame with rrbC signals (already calculated)
        horizon: Number of lag days for signals (default 3)
        middate: Split date between in-sample and out-of-sample

    Returns:
        Intraday DataFrame with 'rrb' forecast column (Energy sector excluded)
    """
    daily_results_df = daily_df
    intra_results_df = intra_df

    sector_name = 'Energy'
    # print "Running rrb for sector {}".format(sector_name)
    # sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    # sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    # result1_df = rrb_fits(sector_df, sector_intra_results_df, horizon, "in", middate)

    print("Running rrb for sector {}".format(sector_name))
    sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] != sector_name ]
    result2_df = rrb_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)

    result_df = pd.concat([result2_df], verify_integrity=True)
    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--horizon",action="store",dest="horizon",default=3)
    parser.add_argument("--freq",action="store",dest="freq",default='15Min')
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = int(args.horizon)
    pname = "./rrb" + start + "." + end
    freq = args.freq
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
        uni_df = get_uni(start, end, lookback)
        barra_df = load_barra(uni_df, start, end)    
        barra_df = transform_barra(barra_df)
        PRICE_COLS = ['close', 'overnight_log_ret']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        daily_df = merge_barra_data(price_df, barra_df)
        DBAR_COLS = ['close', 'dvolume', 'dopen']
        daybar_df = load_daybars(price_df[ ['ticker'] ], start, end, DBAR_COLS, freq)
        intra_df = merge_intra_data(daily_df, daybar_df)

        daily_df, factorRets_df = calc_factors(daily_df, True)
        daily_df = calc_rrb_daily(daily_df, horizon) 
        forwards_df = calc_forward_returns(daily_df, horizon)
        daily_df = pd.concat( [daily_df, forwards_df], axis=1)
        intra_df, factorRets_df = calc_intra_factors(intra_df, True)
        intra_df = calc_rrb_intra(intra_df)
        intra_df = merge_intra_data(daily_df, intra_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    full_df = calc_rrb_forecast(daily_df, intra_df, horizon, middate)
    print(full_df.columns)
    dump_alpha(full_df, 'rrb')
    # dump_alpha(full_df, 'rrbC_B_ma')
    # dump_alpha(full_df, 'rrb0_B_ma')
    # dump_all(full_df)

