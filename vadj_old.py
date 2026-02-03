#!/usr/bin/env python
"""Volume-Adjusted Strategy - Legacy Version (vadj_old)

LEGACY CODE: This is an older implementation of the volume-adjusted strategy.
It has been superseded by vadj.py, vadj_multi.py, vadj_intra.py, and vadj_pos.py.

Key Differences from Current Versions:
--------------------------------------
1. **Different data pipeline**: Uses alphacalc module instead of regress/calc
2. **Logarithmic relative volume**: Uses log(volume/median) instead of raw ratio
3. **Beta division**: Uses log_ret/pbeta instead of beta*market_return subtraction
4. **Different regression functions**: Uses regress_alpha_daily/intra vs regress_alpha
5. **Index handling**: Different approach to merging daily and intraday data

Why It Was Replaced:
--------------------
1. **Better beta adjustment**: Current versions properly compute market returns
   using market cap weighting, rather than simple division by beta
2. **Cleaner volume calculation**: Raw relative volume is more interpretable
   than log relative volume
3. **Improved merge logic**: Current versions handle index alignment better
4. **Unified codebase**: Current versions use consistent data loading patterns

Signal Formulas (Legacy):
------------------------
Daily signal:
  rv = log(tradable_volume / tradable_med_volume_21)
  vadj0 = rv * (log_ret / pbeta)
  vadj0_B_ma = industry_demeaned(winsorize(vadj0))

Intraday signal:
  rv_i = log((dvolume / dvol_frac_med_21) / tradable_med_volume_21_y)
  cur_ret = overnight_ret + log(iclose/dopen)
  vadjC = rv_i * (cur_ret / pbeta)
  vadjC_B_ma = industry_demeaned(winsorize(vadjC))

Final forecast:
  vadj = vadjC_B_ma * coef_intra +
         sum(vadj{lag}_B_ma * coef_lag for lag in 1..horizon-1)

Notable Implementation Details:
------------------------------
1. Uses alphacalc module which has different helper functions
2. Uses regress_alpha_intra which returns intraForwardRets_df
3. Merges forward returns back into full_df during the fitting process
4. Uses remove_dup_cols to handle duplicate columns from merges
5. Creates full_df that combines daily and intraday data structures

When to Use This:
----------------
Generally, you should NOT use this legacy version. Use one of the current versions:
- vadj.py: Full model with daily and intraday signals
- vadj_multi.py: Daily-only multi-period model
- vadj_intra.py: Intraday-only model with hourly coefficients
- vadj_pos.py: Position sizing emphasis with sign-based signals

This file is retained for:
- Historical reference
- Comparison with current implementations
- Understanding evolution of the strategy

Usage (for reference only):
--------------------------
  python vadj_old.py --start=20130101 --end=20130630

Output:
-------
Creates alpha forecast file: vadj.h5
Creates fit diagnostic plots: vadj_daily_*.png, vadj_intra_*.png
"""

from alphacalc import *

from dateutil import parser as dateparser

def calc_vadj(full_df, horizon):
    """Calculate legacy daily volume-adjusted signals.

    LEGACY: Uses log(volume/median) and log_ret/pbeta formulation.
    Current versions use better beta adjustment methodology.

    Args:
        full_df: DataFrame with daily data
        horizon: Number of lags to create

    Returns:
        DataFrame: Original data plus vadj0_B_ma and lagged columns
    """
    print "Caculating daily vadj..."
    result_df = full_df.reset_index()
    result_df = filter_expandable(result_df)

    result_df = result_df[ ['close', 'pbeta', 'tradable_volume', 'tradable_med_volume_21', 'log_ret', 'date', 'ind1', 'sid' ] ]

    print "Calculating vadj0..."
    rv = np.log(result_df['tradable_volume'] / result_df['tradable_med_volume_21'])
    result_df['vadj0'] = rv * result_df['log_ret'] / result_df['pbeta']
    result_df['vadj0_B'] = winsorize(result_df['vadj0'])

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['vadj0_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['vadj0_B_ma'] = indgroups['vadj0_B']
    result_df.set_index(keys=['date', 'sid'], inplace=True)

    print "Calulating lags..."
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['vadj' + str(lag) + '_B_ma'] = shift_df['vadj0_B_ma']

    result_df = pd.merge(full_df, result_df, how='left', left_index=True, right_index=True, sort=False, suffixes=['', '_dead'])
    result_df = remove_dup_cols(result_df)
    return result_df

def calc_vadj_intra(full_df):
    """Calculate legacy intraday volume-adjusted signals.

    LEGACY: Uses log volume ratios and simple beta division.
    Current versions use better market adjustment and beta methodology.

    Args:
        full_df: DataFrame with intraday data

    Returns:
        DataFrame: Original data plus vadjC_B_ma column
    """
    print "Calculating vadj intra..."
    result_df = full_df.reset_index()
    result_df = filter_expandable(result_df)

    result_df = result_df[ ['iclose_ts', 'dopen', 'iclose', 'pbeta', 'dvolume', 'overnight_log_ret', 'tradable_med_volume_21_y', 'dvol_frac_med_21', 'date', 'ind1', 'sid' ] ]
    result_df = result_df.dropna(how='any')

    print "Calulating vadjC..."
    rv = np.log((result_df['dvolume'] / result_df['dvol_frac_med_21']) / result_df['tradable_med_volume_21_y'])
    result_df['vadjC'] = rv * (result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))) / result_df['pbeta']
    result_df['vadjC_B'] = winsorize(result_df['vadjC'])

    print "Calulating vadjC_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['vadjC_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['vadjC_B_ma'] = indgroups['vadjC_B']

    #important for keeping NaTs out of the following merge
    del result_df['date']

    print "Merging..."
    result_df.set_index(keys=['iclose_ts', 'sid'], inplace=True)
    result_df = pd.merge(full_df, result_df, how='left', left_index=True, right_index=True, sort=True, suffixes=['_dead', ''])
    result_df = remove_dup_cols(result_df)

    return result_df

def vadj_fits(daily_df, intra_df, full_df, horizon, name):
    """Fit legacy volume-adjusted model.

    LEGACY: Uses regress_alpha_intra/daily which have different return signatures.
    Also merges forward returns back into full_df during fitting.

    Args:
        daily_df: DataFrame with daily signals
        intra_df: DataFrame with intraday signals
        full_df: Combined DataFrame to write forecasts into
        horizon: Number of lags for daily model
        name: Name suffix for plots

    Returns:
        DataFrame: full_df with vadj forecast column added
    """
    regress_intra_df = intra_df
    regress_daily_df = daily_df
#    middate = intra_df.index[0][0] + (intra_df.index[len(intra_df)-1][0] - intra_df.index[0][0]) / 2
#    print "Setting fit period before {}".format(middate)
#    regress_intra_df = intra_df[ intra_df['date'] <  middate ]

    intra_horizon = 3
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df, intraForwardRets_df = regress_alpha_intra(regress_intra_df, 'vadjC_B_ma', intra_horizon)
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "vadj_intra_"+name+"_" + df_dates(regress_intra_df))

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])

    if 'vadj' not in full_df.columns:
        print "Creating forecast columns..."
        full_df[name] = np.nan
        full_df[ 'vadjC_B_ma_coef' ] = np.nan
        for lag in range(0, horizon+1):
            full_df[ 'vadj' + str(lag) + '_B_ma_coef' ] = np.nan

    for lag in range(1,horizon+1):
        fitresults_df, dailyForwardRets_df = regress_alpha_daily(regress_daily_df, 'vadj0_B_ma', lag)
        full_df = merge_intra_data(dailyForwardRets_df, full_df)
        fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "vadj_daily_"+name+"_" + df_dates(regress_daily_df))
    
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    coef0 = fits_df.ix['vadj0_B_ma'].ix[horizon].ix['coef']
    full_df.ix[ intra_df.index, 'vadjC_B_ma_coef' ] = coef0
    for lag in range(1,horizon+1):
        coef = coef0 - fits_df.ix['vadj0_B_ma'].ix[lag].ix['coef'] 
        print "Coef{}: {}".format(lag, coef)
        full_df.ix[ intra_df.index, 'vadj'+str(lag)+'_B_ma_coef' ] = coef

    full_df.ix[ intra_df.index, 'vadj'] = full_df['vadjC_B_ma'] * full_df['vadjC_B_ma_coef']
    for lag in range(1,horizon):
        full_df.ix[ intra_df.index, 'vadj'] += full_df['vadj'+str(lag)+'_B_ma'] * full_df['vadj'+str(lag)+'_B_ma_coef']
    
    return full_df

def calc_vadj_forecast(daily_df, intra_df, horizon):
    """Calculate legacy volume-adjusted forecasts.

    LEGACY: Uses Energy sector split and combines daily+intraday signals.
    The full_df structure allows incremental updates during fitting.

    Args:
        daily_df: DataFrame with daily data
        intra_df: DataFrame with intraday data
        horizon: Number of daily lags

    Returns:
        DataFrame: Combined data with vadj forecast column
    """
    daily_df = calc_vadj(daily_df, horizon)
    intra_df = calc_vadj_intra(intra_df)
    full_df = merge_intra_data(daily_df, intra_df)

    sector_name = 'Energy'
    print "Running vadj for sector {}".format(sector_name)
    sector_df = daily_df[ daily_df['sector_name'] == sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] == sector_name ]
    full_df = vadj_fits(sector_df, sector_intra_df, full_df, horizon, "vadj")

    print "Running vadj for sector {}".format(sector_name)
    sector_df = daily_df[ daily_df['sector_name'] != sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] != sector_name ]
    full_df = vadj_fits(sector_df, sector_intra_df, full_df, horizon, "vadj")
  
    return full_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = 5

    start = dateparser.parse(start)
    end = dateparser.parse(end)

    uni_df = get_uni(start, end, lookback)

    barra_df = load_barra(uni_df, start, end)
    price_df = load_prices(uni_df, start, end)
    daily_df = merge_barra_data(price_df, barra_df)

#    volume_df = load_volume_profile(uni_df, start, end)
    daybar_df = load_daybars(uni_df, start, end)
    intra_df = merge_intra_data(daily_df, daybar_df)

    intra_df = calc_vol_profiles(intra_df)

   # intra_df = pd.merge(daybar_df, volume_df, left_index=True, right_index=True, sort=False, suffixes=['', '_dead'])
   # intra_df = remove_dup_cols(intra_df)

    full_df = calc_vadj_forecast(daily_df, intra_df, horizon)

    dump_alpha(full_df, 'vadj')
    dump_all(full_df)

