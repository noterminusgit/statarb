#!/usr/bin/env python
"""
Hard-to-Borrow (HTB) Fee Rate Alpha Strategy

This module generates alpha signals based on equity borrow fee rates from the
stock loan market. High borrow fees indicate strong shorting demand and limited
supply, which can predict future returns.

Strategy Logic:
    1. Use fee_rate from locates data as primary signal
    2. Winsorize fee rates by date to reduce outlier influence
    3. Create lagged features (0 to horizon days)
    4. Regress lagged fee rates against forward returns
    5. Apply coefficients to generate intraday forecasts

The economic intuition:
    - High borrow fees indicate pessimistic sentiment and crowded shorts
    - When shorts get too crowded, mean reversion or short squeezes occur
    - Fee rates are a cleaner signal than short interest (real-time, price-based)

Relationship to Short Availability:
    Fee rates directly measure how difficult (expensive) it is to borrow shares
    for shorting. When fee_rate is high:
    - Stock is "hard to borrow" (HTB)
    - Short supply is limited relative to demand
    - Potential for short squeeze if positive news emerges

Data Requirements:
    - Daily fee_rate data from LOCATES_BASE_DIR
    - Intraday bars (30-min default) for forecast generation
    - Minimum 5+ days of history to populate lagged features

Parameters:
    horizon: Number of lag days to include (default 5)
    freq: Intraday bar frequency (default '30Min')

Output:
    'htb' column with fee rate-based forward return forecast

Note: Despite the name suggesting hard-to-borrow indicators, this specifically
uses borrow fee rates as a quantitative signal rather than binary HTB flags.

Usage:
    python htb.py --start=20130101 --end=20130630 --mid=20130401 --freq=30Min
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def calc_htb_daily(daily_df, horizon):
    """
    Calculate hard-to-borrow fee rate signals with lags.

    Formula:
        1. htbC = fee_rate (raw borrow cost)
        2. htbC_B = winsorize(htbC) by date
        3. Create lagged features htb{lag}_B for lag=0 to horizon

    Winsorization prevents extreme outliers (very rare stocks with 100%+ fees)
    from dominating the regression. Lagged features capture the persistence
    and mean reversion of borrow fee dynamics.

    Args:
        daily_df: Daily DataFrame with fee_rate column from locates data
        horizon: Number of lag days to create (default 5)

    Returns:
        DataFrame with htb0_B through htb{horizon}_B columns
    """
    print("Caculating daily htb...")
    result_df = filter_expandable(daily_df)

    print("Calculating htb0...")
    result_df['htbC'] = result_df['fee_rate']
    result_df['htbC_B'] = winsorize_by_date(result_df[ 'htbC' ])

    print("Calulating lags...")
    for lag in range(0,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['htb'+str(lag) + "_B"] = shift_df['htbC_B']

    return result_df

def htb_fits(daily_df, intra_df, horizon, name, middate=None):
    """
    Fit HTB regression model and generate intraday forecasts.

    Performs daily-frequency WLS regression of htb0_B against forward returns
    at multiple horizons. Then applies the fitted coefficients to intraday data
    for high-frequency trading.

    The coefficient structure captures:
        - coef[0] = total effect at horizon
        - coef[lag] = incremental effect of lag-day fee rates
        - Final coef[lag] = coef[0] - coef[lag] (differential effect)

    This differential structure ensures consistent forecast as new lag signals
    arrive each day.

    Args:
        daily_df: Daily DataFrame with htb signals and forward returns
        intra_df: Intraday DataFrame to receive forecasts
        horizon: Maximum lag to include in regressions (default 5)
        name: String identifier for plot filenames
        middate: Split date between in-sample and out-of-sample

    Returns:
        Intraday DataFrame with 'htb' forecast column
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['htb'] = np.nan
    outsample_intra_df[ 'htbC_B_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_intra_df[ 'htb' + str(lag) + '_B_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'htb0_B', lag, True, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "htb_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    coef0 = fits_df.ix['htb0_B'].ix[horizon].ix['coef']
    outsample_intra_df['htbC_B_coef'] = coef0
    print("Coef0: {}".format(coef0))
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['htb0_B'].ix[lag].ix['coef']
        print("Coef{}: {}".format(lag, coef))
        outsample_intra_df[ 'htb'+str(lag)+'_B_coef' ] = coef

    outsample_intra_df['htb'] = outsample_intra_df['htbC_B'] * outsample_intra_df['htbC_B_coef']
    for lag in range(1,horizon):
        outsample_intra_df[ 'htb'] += outsample_intra_df['htb'+str(lag)+'_B'] * outsample_intra_df['htb'+str(lag)+'_B_coef']

    return outsample_intra_df

def calc_htb_forecast(daily_df, intra_df, horizon, middate):
    """
    Master function to calculate HTB forecasts.

    Runs the full pipeline:
    1. Calculate htb signals (fee rates with lags)
    2. Calculate forward returns for regression fitting
    3. Merge daily htb signals into intraday bars
    4. Fit regression model and generate intraday forecasts

    The daily signals are broadcast to all intraday bars within each day,
    allowing high-frequency trading on daily fee rate changes.

    Args:
        daily_df: Daily price and locates data
        intra_df: Intraday bar data
        horizon: Number of lag days for signals (default 5)
        middate: Split date between in-sample and out-of-sample

    Returns:
        Intraday DataFrame with 'htb' forecast column
    """
    daily_results_df = calc_htb_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    #    intra_results_df = calc_htb_intra(intra_df)
    intra_results_df = intra_df
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)

    result_df = htb_fits(daily_results_df, intra_results_df, horizon, "", middate)

    return result_df

if __name__== "__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--freq",action="store",dest="freq",default='30Min')
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = 5
    pname = "./htb" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)
    freq = args.freq
    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        DBAR_COLS = ['close', 'qhigh', 'qlow']
        intra_df = load_daybars(price_df[['ticker']], start, end, DBAR_COLS, freq)
        daily_df = merge_barra_data(price_df, barra_df)
        intra_df = merge_intra_data(daily_df, intra_df)

        locates_df = load_locates(price_df[['ticker']], start, end)
        daily_df = pd.merge(daily_df, locates_df, how='left', left_index=True, right_index=True, suffixes=['', '_dead'])
        daily_df = remove_dup_cols(daily_df)         

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    result_df = calc_htb_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(result_df, 'htb')



