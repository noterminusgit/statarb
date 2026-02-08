#!/usr/bin/env python
"""
Annual Momentum Alpha Strategy

This module implements a long-term momentum strategy based on lagged price returns.
Despite the filename "mom_year", it uses a 232-day (roughly 1 year) lag combined
with a 20-day rolling sum signal.

Strategy Logic:
    1. Calculate 20-day cumulative log return as base momentum signal
    2. Demean within (date, industry) groups for market neutrality
    3. Lag the signal by 232 trading days (approximately 1 calendar year)
    4. Regress lagged momentum against forward returns
    5. Apply regression coefficients to generate forecast

The 232-day lag is designed to capture the well-documented annual momentum effect
while avoiding short-term reversal and medium-term continuation zones.

Academic Basis:
    Related to the momentum anomaly (Jegadeesh & Titman 1993) and the long-term
    reversal literature (DeBondt & Thaler 1985). The 1-year lag avoids the
    short-term reversal (1 month) and medium-term continuation (3-12 months) zones,
    potentially capturing value mean reversion after extreme annual performance.

Data Requirements:
    - Daily close prices for log return calculation
    - Industry codes (ind1) for within-industry demeaning
    - Minimum 252+ days of history to populate 232-day lagged features

Parameters:
    horizon: Forward return horizon for regression fitting (default 20 days)
    middate: Split date between in-sample fitting and out-of-sample prediction

Output:
    'mom' column with momentum-based forward return forecast

Note: The strategy requires at least 1 year of history before generating signals
(232-day lag + 20-day rolling window).

Usage:
    python mom_year.py --start=20130101 --end=20130630 --mid=20130401
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def calc_mom_daily(daily_df, horizon):
    """
    Calculate annual momentum signal with 232-day lag.

    Formula:
        1. mom0 = sum(log_ret[t-19:t])  (20-day cumulative return)
        2. mom0_ma = demean(mom0) within (date, industry) groups
        3. mom1_ma = lag(mom0_ma, 232 days)

    The 20-day rolling sum smooths out daily noise while capturing recent
    price trends. The 232-day lag shifts this signal back approximately one
    year to capture long-term reversal patterns.

    Industry demeaning ensures the strategy is industry-neutral, betting on
    relative momentum within sectors rather than absolute momentum.

    Args:
        daily_df: Daily DataFrame with log_ret and ind1 columns
        horizon: Forward return horizon (not used in signal calculation)

    Returns:
        DataFrame with 'mom1_ma' column containing 1-year lagged momentum signal
    """
    print("Caculating daily mom...")
    result_df = filter_expandable(daily_df)

    print("Calculating mom0...")
    result_df['mom0'] = pd.rolling_sum(result_df['log_ret'], 20)

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['mom0', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    result_df['mom0_ma'] = indgroups['mom0']
    shift_df = result_df.unstack().shift(232).stack()
    result_df['mom1_ma'] = shift_df['mom0_ma']

    return result_df

def mom_fits(daily_df, horizon, name, middate=None):
    """
    Fit momentum regression model and generate forecasts.

    Performs WLS regression of mom1_ma against forward returns to estimate
    the predictive coefficient. Applies coefficient to out-of-sample data.

    Important: In-sample period starts 252 days after data begins to ensure
    sufficient history for the 232-day lag feature.

    Args:
        daily_df: Daily DataFrame with mom1_ma signal and forward returns
        horizon: Forward return horizon for regression (default 20)
        name: String identifier for plot filename
        middate: Split date between in-sample and out-of-sample

    Returns:
        DataFrame with 'mom' forecast column = mom1_ma * fitted_coefficient
    """
    insample_daily_df = daily_df
    if middate is not None:
        startdate = daily_df.index.get_level_values('date').min() + timedelta(days=252)
        insample_daily_df = daily_df[ (daily_df.index.get_level_values('date') < middate) & (daily_df.index.get_level_values('date') > startdate) ]
        outsample_daily_df = daily_df[ daily_df.index.get_level_values('date') >= middate ]

    outsample_daily_df['mom'] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df = regress_alpha(insample_daily_df, 'mom1_ma', horizon, True, 'daily')
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "mom_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    coef0 = fits_df.ix['mom1_ma'].ix[horizon].ix['coef']
    print("Coef{}: {}".format(0, coef0))
    outsample_daily_df[ 'mom1_ma_coef' ] = coef0

    outsample_daily_df[ 'mom'] = outsample_daily_df['mom1_ma'] * outsample_daily_df['mom1_ma_coef']

    return outsample_daily_df

def calc_mom_forecast(daily_df, horizon, middate):
    """
    Master function to calculate momentum forecasts.

    Runs the full pipeline:
    1. Calculate mom1_ma (232-day lagged 20-day momentum)
    2. Calculate forward returns for regression fitting
    3. Fit regression model and generate forecasts

    Args:
        daily_df: Daily price and factor data
        horizon: Forward return horizon for regression (default 20)
        middate: Split date between in-sample and out-of-sample

    Returns:
        DataFrame with 'mom' forecast column
    """
    daily_results_df = calc_mom_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

    result_df = mom_fits(daily_results_df, horizon, "", middate)

    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = 20
    pname = "./mom" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)

    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        loaded = True
    except:
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)

        daily_df = merge_barra_data(price_df, barra_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')

    result_df = calc_mom_forecast(daily_df, horizon, middate)
    dump_daily_alpha(result_df, 'mom')



