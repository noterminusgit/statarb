#!/usr/bin/env python
"""Day-of-Week Multi-Period Beta-Adjusted Returns Strategy (badj_dow_multi)

Extension of badj_multi.py that incorporates day-of-week effects in the
regression fitting. Fits separate coefficients for each weekday (Monday-Friday)
to capture calendar-specific patterns in beta-adjusted return predictability.

IMPORTANT: This strategy does NOT use order flow. It uses simple return / beta
ratio for beta adjustment, same as badj_multi.py.

Strategy Enhancement:
--------------------
vs badj_multi.py:
- badj_multi.py: Single set of coefficients for all days
- badj_dow_multi.py: Separate coefficients for each weekday
- Captures day-of-week effects in signal efficacy

Day-of-Week Effects:
-------------------
Research shows return patterns vary by day of week:
- Monday: Often negative (weekend news)
- Tuesday-Thursday: Mid-week patterns
- Friday: End-of-week positioning

This strategy tests whether beta-adjusted return signals have different
predictive power on different weekdays.

Methodology:
-----------
1. Beta Adjustment (Same as badj_multi.py):
   o2c0 = log_ret / pbeta

2. Day-of-Week Encoding:
   - Adds 'dow' column: weekday() method returns 0-6 (Mon-Sun)
   - Monday=0, Tuesday=1, ..., Friday=4

3. Regression Mode:
   - Uses regress_alpha(..., mode='dow')
   - Fits separate coefficients for each weekday
   - Generates horizon * 10 + day index for coefficient lookup

4. Coefficient Application:
   - Each lag gets 5 separate coefficients (Mon-Fri)
   - Applied based on forecast date's weekday
   - Enables day-specific signal weighting

Signal Formulas:
---------------
(Same as badj_multi.py)

Daily:
    o2c0 = log_ret / pbeta
    o2c0_B = winsorize(o2c0)
    o2c0_B_ma = industry_demean(o2c0_B)

Intraday:
    o2cC = (overnight_log_ret + log(iclose/dopen)) / pbeta
    o2cC_B = winsorize(o2cC)
    o2cC_B_ma = industry_demean(o2cC_B)

Forecast (Day-Specific):
    For each weekday d:
        badj_m = o2cC_B_ma * 0 +  # Intraday disabled (line 86)
                 sum_{lag=1}^{horizon-1} (o2c{lag}_B_ma * coef[lag][d])

where coef[lag][d] is the coefficient for lag at weekday d.

Coefficient Indexing:
--------------------
Coefficients indexed as: horizon * 10 + day
- Lag 1, Monday: coef['o2c0_B_ma'][10]
- Lag 1, Tuesday: coef['o2c0_B_ma'][11]
- Lag 2, Monday: coef['o2c0_B_ma'][20]
- Lag 3, Friday: coef['o2c0_B_ma'][34]

This encoding allows efficient lookupof day-specific coefficients.

Sector Splitting:
----------------
- Fits separate regressions for Energy sector vs all others
- PLUS day-of-week split within each sector
- Total: 2 sectors × 5 weekdays = 10 coefficient sets

Use Case:
--------
- Tests calendar effects in beta-adjusted returns
- Exploits day-of-week anomalies if they exist
- More parameters = better fit but risk of overfitting
- Useful for understanding signal time-variation

Data Requirements:
-----------------
(Same as badj_multi.py plus)
- Date information for weekday calculation
- Sufficient history across all weekdays for stable estimates

CLI Usage:
---------
Run backtest with optional in-sample/out-of-sample split:
    python badj_dow_multi.py --start=20130101 --end=20130630 --os=True

Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --os: Enable out-of-sample split (default: False)

Output:
------
- Regression plots: badj_daily_in_{dates}.png, badj_daily_ex_{dates}.png
  (Separate coefficients by day but plotted together)
- HDF5 cache: badj_m{start}.{end}_daily.h5, badj_m{start}.{end}_intra.h5
- Alpha forecast: 'badj_m' column written via dump_alpha()

Related Modules:
---------------
- badj_multi.py: Base version without day-of-week effects
- badj_intra.py: Intraday-only version
- badj2_multi.py: Alternative with market-weighted beta

Notes:
-----
- Horizon fixed at 3 in __main__
- Intraday coefficient hardcoded to 0 (line 86)
- Requires sufficient data for each weekday
- More parameters = higher risk of overfitting
- Day-of-week effects may be time-varying themselves
- Uses merge_intra_calcs() for data merging
"""

from __future__ import division, print_function

from alphacalc import *

from dateutil import parser as dateparser
import argparse

def calc_o2c(daily_df, horizon):
    """
    Calculate daily beta-adjusted returns (o2c) with multiple lags.

    Identical to badj_multi.py version. Day-of-week differentiation happens
    in the regression fitting phase, not signal calculation.

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, sid)
        horizon (int): Number of lagged signals to create (typically 3)

    Returns:
        pd.DataFrame: Daily data augmented with o2c signals and lags

    Notes:
        - Signal calculation identical to badj_multi.py
        - Day-of-week encoding added later in pipeline
        - See badj_multi.py documentation for detailed signal formulas
    """
    print("Caculating daily o2c...")
    result_df = daily_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['log_ret', 'pbeta', 'date', 'ind1', 'sid' ]]

    print("Calculating o2c0...")
    result_df['o2c0'] = result_df['log_ret'] / result_df['pbeta'] 
    result_df['o2c0_B'] = winsorize_by_group(result_df[ ['date', 'o2c0'] ], 'date')

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['o2c0_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['o2c0_B_ma'] = indgroups['o2c0_B']
    result_df.set_index(keys=['date', 'sid'], inplace=True)
    
    print("Calulating lags...")
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['o2c' + str(lag) + '_B_ma'] = shift_df['o2c0_B_ma']

    result_df = merge_daily_calcs(daily_df, result_df)
    return result_df

def calc_o2c_intra(intra_df, daily_df):
    """
    Calculate intraday beta-adjusted returns (o2cC).

    Identical to badj_multi.py version. Day-of-week encoding added later
    in the o2c_fits() function.

    Args:
        intra_df (pd.DataFrame): Intraday bar data
        daily_df (pd.DataFrame): Daily reference data for expandable filter

    Returns:
        pd.DataFrame: Intraday data augmented with o2cC signals

    Notes:
        - Signal calculation identical to badj_multi.py
        - Day-of-week differentiation applied in regression phase
        - See badj_multi.py documentation for detailed formulas
    """
    print("Calculating o2c intra...")

    result_df = filter_expandable_intra(intra_df, daily_df)
    result_df = result_df[ ['iclose', 'dopen', 'overnight_log_ret', 'pbeta', 'date', 'ind1' ] ]
    result_df = result_df.dropna(how='any')

    print("Calulating o2cC...")
    result_df['o2cC'] = (result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))) / result_df['pbeta']
    result_df['o2cC_B'] = winsorize_by_ts(result_df[ ['o2cC'] ])

    print("Calulating o2cC_ma...")
    result_df.reset_index(inplace=True)
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['o2cC_B', 'iclose_ts', 'ind1']].groupby(['iclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['o2cC_B_ma'] = indgroups['o2cC_B']
    result_df.set_index(['iclose_ts', 'sid'], inplace=True)

    result_df = merge_intra_calcs(intra_df, result_df)
    return result_df

def o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None):
    """
    Fit day-of-week specific regression and generate beta-adjusted forecast.

    Key innovation: Adds day-of-week encoding and uses 'dow' regression mode
    to fit separate coefficients for each weekday. This allows signal efficacy
    to vary by day of week.

    Process:
    1. Add 'dow' column to daily and intraday data (0=Mon, 4=Fri)
    2. Fit regressions with mode='dow' for day-specific coefficients
    3. Apply coefficients based on each observation's weekday
    4. Combine daily lags with day-specific weighting

    Args:
        daily_df (pd.DataFrame): Daily data with o2c signals
        intra_df (pd.DataFrame): Intraday data with o2cC signals
        full_df (pd.DataFrame): Full merged dataset for forecast storage
        horizon (int): Forecast horizon in days (typically 3)
        name (str): Name suffix for output plots (e.g., "in", "ex")
        middate (datetime): Split date for in-sample vs out-of-sample

    Returns:
        pd.DataFrame: full_df augmented with:
            - o2cC_B_ma_coef: Intraday coefficient (set to 0)
            - o2c{1..horizon-1}_B_ma_coef: Day-specific lag coefficients
            - badj_m: Combined forecast with day-of-week adjustments

    Regression Strategy:
    -------------------
    - Uses regress_alpha(..., mode='dow') for day-specific fitting
    - Fits o2c0_B_ma at multiple lags (1 to horizon)
    - Each lag gets 5 coefficients (one per weekday)
    - Coefficient indexing: horizon * 10 + day
      Example: Lag 3, Wednesday (day=2) → index 32

    Coefficient Application:
        For each observation with weekday d and lag L:
            coef = fits_df['o2c0_B_ma'][L * 10 + d]['coef']

        badj_m = 0 * o2cC_B_ma +  # Intraday disabled
                 sum_{lag=1}^{horizon-1} (o2c{lag}_B_ma * coef[lag][dow])

    Output Files:
        - badj_daily_{name}_{dates}.png: Regression plot (all days combined)

    Notes:
        - Intraday coefficient hardcoded to 0 (line 86)
        - Groupby dow to process each weekday separately
        - More coefficients = better fit but overfitting risk
        - Prints coefficients for each day and lag
        - Only out-of-sample gets forecasts if middate specified
    """
    if 'badj_m' not in full_df.columns:
        print("Creating forecast columns...")
        full_df['badj_m'] = np.nan
        full_df[ 'o2cC_B_ma_coef' ] = np.nan
        for lag in range(1, horizon+1):
            full_df[ 'o2c' + str(lag) + '_B_ma_coef' ] = np.nan

    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    outsample = False
    if middate is not None:
        outsample = True
        insample_intra_df = intra_df[ intra_df['date'] < middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    insample_daily_df = insample_daily_df.reset_index()
    insample_daily_df['dow'] = insample_daily_df['date'].apply(lambda x: x.weekday())
    insample_daily_df.set_index(['date', 'sid'], inplace=True)
    outsample_intra_df['dow'] = outsample_intra_df['date'].apply(lambda x: x.weekday())

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'o2c0_B_ma', lag, True, 'dow')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)  
    plot_fit(fits_df, "badj_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    for day, daygroup in insample_daily_df.groupby('dow'):
        idx = outsample_intra_df[ outsample_intra_df['dow'] == day ].index
        day = int(day)
        coef0 = fits_df.loc['o2c0_B_ma'].loc[horizon * 10 + day].loc['coef']
        full_df.loc[ idx, 'o2cC_B_ma_coef' ] = 0#coef0
        print("{} {} Coef0: {}".format(name, day, coef0))
        for lag in range(1,horizon):
            coef = coef0 - fits_df.loc['o2c0_B_ma'].loc[lag * 10 + day].loc['coef'] 
            print("{} {} Coef{}: {}".format(name, day, lag, coef))
            full_df.loc[ idx, 'o2c'+str(lag)+'_B_ma_coef' ] = coef

    full_df.loc[ outsample_intra_df.index, 'badj_m'] = full_df['o2cC_B_ma'] * full_df['o2cC_B_ma_coef']
    for lag in range(1,horizon):
        full_df.loc[ outsample_intra_df.index, 'badj_m'] += full_df['o2c'+str(lag)+'_B_ma'] * full_df['o2c'+str(lag)+'_B_ma_coef']

    return full_df

def calc_o2c_forecast(daily_df, intra_df, horizon, outsample):
    """
    Main entry point: compute day-of-week adjusted beta-adjusted forecast.

    Orchestrates the pipeline with sector split and day-of-week specific
    coefficient fitting. Combines both sector and calendar effects.

    Args:
        daily_df (pd.DataFrame): Daily price/factor data with MultiIndex (date, sid)
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid)
        horizon (int): Forecast horizon in days (typically 3)
        outsample (bool): If True, split data in-sample/out-of-sample

    Returns:
        pd.DataFrame: Full dataset with day-of-week adjusted 'badj_m' forecasts

    Pipeline Flow:
        daily_df → calc_o2c() → daily with lags
        daily_df → calc_forward_returns() → forwards
        [merge]
        intra_df → calc_o2c_intra() → intra results
        [merge daily + intra]
        → o2c_fits(Energy sector) → partial forecasts
        → o2c_fits(ex-Energy) → complete forecasts

    Sector Processing:
        1. Energy sector: Day-of-week fitting for Energy stocks
        2. Ex-Energy: Day-of-week fitting for non-Energy stocks
        3. Each sector gets 5 × horizon coefficient sets

    Notes:
        - Combines sector split + day-of-week effects
        - Total complexity: 2 sectors × 5 days × horizon lags
        - middate computed as midpoint if outsample=True
        - Only outsample data returned if outsample=True
        - Final forecast in 'badj_m' column
    """
    daily_df = calc_o2c(daily_df, horizon) 
    daily_df = calc_forward_returns(daily_df, horizon)
    intra_df = calc_o2c_intra(intra_df, daily_df)
    full_df = merge_intra_data(daily_df, intra_df)

    middate = None
    if outsample:
        middate = daily_df.index[0][0] + (daily_df.index[len(daily_df)-1][0] - daily_df.index[0][0]) / 2
        print("Setting fit period before {}".format(middate))

    sector_name = 'Energy'
    print("Running o2c for sector {}".format(sector_name))
    sector_df = daily_df[ daily_df['sector_name'] == sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] == sector_name ]
    full_df = o2c_fits(sector_df, sector_intra_df, full_df, horizon, "in", middate)

    print("Running o2c for sector {}".format(sector_name))
    sector_df = daily_df[ daily_df['sector_name'] != sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] != sector_name ]
    full_df = o2c_fits(sector_df, sector_intra_df, full_df, horizon, "ex", middate)

    if outsample:
        full_df = full_df[ full_df['date'] > middate ]
    return full_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--os",action="store",dest="outsample",default=False)
    args = parser.parse_args()    

    start = args.start
    end = args.end
    outsample = args.outsample
    lookback = 30
    horizon = 3

    pname = "./badj_m" + start + "." + end

    start = dateparser.parse(start)
    end = dateparser.parse(end)

    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        barra_df = load_barra(uni_df, start, end)
        price_df = load_prices(uni_df, start, end)
        daily_df = merge_barra_data(price_df, barra_df)
        daybar_df = load_daybars(uni_df, start, end)
        intra_df = merge_intra_data(daily_df, daybar_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    outsample_df = calc_o2c_forecast(daily_df, intra_df, horizon, outsample)

    dump_alpha(outsample_df, 'badj_m')
    dump_all(outsample_df)

