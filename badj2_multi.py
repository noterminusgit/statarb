#!/usr/bin/env python
"""Alternative Multi-Period Beta-Adjusted Returns Strategy (badj2_multi)

Alternative implementation of beta-adjusted returns that uses market-weighted
beta calculation instead of simple division. This approach is closer to bd.py's
methodology but still operates on returns rather than order flow.

IMPORTANT: This strategy does NOT use order flow. It uses a more sophisticated
beta adjustment calculation than badj_multi.py.

Key Differences from badj_multi.py:
----------------------------------
1. Beta Adjustment Method:
   - badj_multi.py: o2c0 = log_ret / pbeta (simple division)
   - badj2_multi.py: o2c0 = groupby(date).apply(wavg) (market-weighted)
   - More sophisticated market-neutralization

2. Market Return Calculation:
   wavg computes: pbeta * market_return
   where market_return = cap-weighted average of all returns

   Then result is subtracted (implicitly through groupby.apply)

3. Philosophy:
   - badj_multi.py: Beta normalization (divide by beta)
   - badj2_multi.py: Beta adjustment (remove beta * market component)
   - Closer to bd.py's approach but without order flow

Key Differences from bd.py:
--------------------------
1. Signal Source:
   - bd.py: Order flow imbalance (askHit - bidHit) / total
   - badj2_multi.py: Beta-adjusted returns (not order flow)
   - Still fundamentally different alpha sources

2. Market Adjustment:
   - bd.py: Subtracts beta * market_return from log_ret
   - badj2_multi.py: Uses wavg() to compute adjusted component
   - Similar approach but applied to returns not order flow

Methodology:
-----------
1. Daily Beta Adjustment (Market-Weighted):
   For each date:
       market_return = sum(log_ret * weight) / sum(weight)
       where weight = capitalization / 1e6
       o2c0 = pbeta * market_return

   This computes the market component of each stock's return.

2. Intraday Beta Adjustment:
   For each timestamp:
       market_return = sum(cur_log_ret * weight) / sum(weight)
       o2cC = pbeta * market_return

   Same approach for intraday returns.

3. Winsorization and Industry Demeaning:
   - Winsorize by date/timestamp
   - Industry demean for additional sector neutrality
   - Creates o2c0_B_ma and o2cC_B_ma signals

4. Multi-Period Lags and Regression:
   - Same as badj_multi.py once signals are computed
   - Fits at multiple lags with residual coefficients

Signal Formulas:
---------------
Daily:
    o2c0 = pbeta * (cap_weighted_market_return)
    o2c0_B = winsorize(o2c0)
    o2c0_B_ma = industry_demean(o2c0_B)

Intraday:
    o2cC = pbeta * (cap_weighted_intraday_market_return)
    o2cC_B = winsorize(o2cC)
    o2cC_B_ma = industry_demean(o2cC_B)

Forecast:
    badj2_m = o2cC_B_ma * 0 +  # Intraday disabled (line 93)
              sum_{lag=1}^{horizon-1} (o2c{lag}_B_ma * residual_coef[lag])

Market Return Calculation:
-------------------------
wavg(group):
    market_return = sum(log_ret * weight) / sum(weight)
    where weight = capitalization / 1e6
    return pbeta * market_return

This gives each stock its beta-scaled exposure to the market return.

Sector Splitting:
----------------
- Fits separate regressions for Energy sector vs all others
- Two regression outputs: "in" (Energy) and "ex" (ex-Energy)

Use Case:
--------
- More sophisticated beta adjustment than simple division
- Tests market-neutralized return patterns
- Closer to factor model approach
- Still no order flow data required

Data Requirements:
-----------------
- Price data: log_ret (daily returns)
- Barra factors: pbeta (predicted beta), ind1 (industry)
- Market cap: capitalization for weighting
- Intraday: overnight_log_ret, dopen, iclose, mkt_cap_y
- Universe: Expandable stocks (liquid, tradeable)

CLI Usage:
---------
Run backtest with optional in-sample/out-of-sample split:
    python badj2_multi.py --start=20130101 --end=20130630 --os=True

Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --os: Enable out-of-sample split (default: False)

Output:
------
- Regression plots: badj_daily_in_{dates}.png, badj_daily_ex_{dates}.png
- HDF5 cache: badj2_m{start}.{end}_daily.h5, badj2_m{start}.{end}_intra.h5
- Alpha forecast: 'badj2_m' column written via dump_alpha()

Related Modules:
---------------
- badj_multi.py: Simple division version
- badj2_intra.py: Intraday-only version of this approach
- bd.py: Order flow based (different alpha source)
- badj_both.py: Combined daily+intraday

Code Issues:
-----------
- Line 166: References undefined 'outsample_df' (should be 'full_df')
- Variable name inconsistency in return statement

Notes:
-----
- Horizon fixed at 3 in __main__
- Intraday coefficient hardcoded to 0 (line 93)
- Uses merge_daily_calcs() for data merging
- Market-weighted approach more sophisticated than simple division
- Requires capitalization data for proper market return calculation
"""

from alphacalc import *

from dateutil import parser as dateparser
import argparse

def wavg(group):
    """
    Calculate market component of returns using cap-weighted market return.

    Computes beta * market_return where market_return is the capitalization-
    weighted average return across all stocks in the group (typically all
    stocks on a given date).

    This gives each stock's systematic (market) component of returns based
    on its beta and the overall market move.

    Args:
        group (pd.DataFrame): DataFrame group containing:
            - pbeta: Predicted beta from Barra risk model
            - log_ret: Log return for the period
            - capitalization: Market capitalization (in dollars)

    Returns:
        pd.Series: Beta-scaled market return for each stock in group.
                   All stocks get same market return but scaled by their beta.

    Formula:
        market_return = sum(log_ret * weight) / sum(weight)
        where weight = capitalization / 1e6
        result = pbeta * market_return

    Notes:
        - Capitalization scaled by 1e6 (millions) to avoid numerical issues
        - Used in groupby().apply() pattern with grouping by date
        - Each stock gets its own beta but same underlying market return
        - This is the "market component" that can be subtracted for neutrality
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['capitalization'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg2(group):
    """
    Calculate intraday market component using cap-weighted market return.

    Identical to wavg() but operates on intraday returns (cur_log_ret) rather
    than daily returns. Used for beta adjustment within intraday bars.

    Args:
        group (pd.DataFrame): DataFrame group with pbeta, cur_log_ret, mkt_cap_y

    Returns:
        pd.Series: Beta-scaled intraday market return for each stock

    Formula:
        market_return = sum(cur_log_ret * weight) / sum(weight)
        where weight = mkt_cap_y / 1e6
        result = pbeta * market_return

    Notes:
        - Used in groupby().apply() with grouping by timestamp (iclose_ts)
        - Enables beta adjustment for each intraday bar
        - Uses mkt_cap_y (different column name than wavg's capitalization)
    """
    b = group['pbeta']
    d = group['cur_log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def calc_o2c(daily_df, horizon):
    """
    Calculate daily market-weighted beta-adjusted returns with multiple lags.

    Computes the core daily signal using market-weighted beta adjustment
    (wavg function) rather than simple division. This creates a more
    sophisticated market-neutral signal.

    Process:
    1. Filter to expandable (tradeable, liquid) universe
    2. Compute market component: o2c0 = groupby(date).apply(wavg)
    3. Winsorize by date to control outliers
    4. Industry demean for sector neutrality
    5. Create lagged versions (o2c1, o2c2, ..., o2c{horizon})

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, sid) containing:
            - log_ret: Daily log returns
            - pbeta: Predicted beta from Barra
            - capitalization: Market cap for weighting (note: different from mkt_cap_y)
            - ind1: Industry classification (for demeaning)
            - expandable: Boolean filter for tradeable stocks

        horizon (int): Number of lagged signals to create (typically 3)

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - o2c0: Market-weighted beta component
            - o2c0_B: Winsorized signal
            - o2c0_B_ma: Industry-demeaned signal (main daily signal)
            - o2c{1..horizon}_B_ma: Lagged versions of o2c0_B_ma

    Signal Formula:
        o2c0 = pbeta * (cap_weighted_market_return)
        o2c0_B = winsorize_by_group(o2c0, groupby='date')
        o2c0_B_ma = industry_demean(o2c0_B)

    Notes:
        - More sophisticated than badj_multi.py's simple division
        - Requires capitalization data for proper weighting
        - Uses merge_daily_calcs() for merging
        - Lagged signals enable multi-horizon regression fitting
    """
    print "Caculating daily o2c..."

    result_df = daily_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ ['log_ret', 'pbeta', 'date', 'ind1', 'sid', 'mkt_cap_y' ]]

    print "Calculating o2c0..."
    result_df['o2c0'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'date']].groupby(['date'], sort=False).apply(wavg)
    result_df['o2c0_B'] = winsorize_by_group(result_df[ ['date', 'o2c0'] ], 'date')

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['o2c0_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['o2c0_B_ma'] = indgroups['o2c0_B']
    result_df.set_index(keys=['date', 'sid'], inplace=True)
    
    print "Calulating lags..."
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['o2c' + str(lag) + '_B_ma'] = shift_df['o2c0_B_ma']

    result_df = merge_daily_calcs(daily_df, result_df)
    return result_df

def calc_o2c_intra(intra_df, daily_df):
    """
    Calculate intraday market-weighted beta-adjusted returns.

    Computes intraday signal using market-weighted beta adjustment (wavg2)
    applied to cumulative returns (overnight + day's move).

    Process:
    1. Filter to expandable universe
    2. Calculate cumulative return: overnight + log(iclose/dopen)
    3. Compute market component: o2cC = groupby(iclose_ts).apply(wavg2)
    4. Winsorize by timestamp
    5. Industry demean within (timestamp, industry) groups

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid):
            - overnight_log_ret: Overnight return
            - iclose: Intraday bar close price
            - dopen: Day open price
            - pbeta: Predicted beta
            - mkt_cap_y: Market capitalization
            - ind1: Industry classification
            - expandable: Boolean filter

        daily_df (pd.DataFrame): Daily reference data for expandable filter

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - cur_log_ret: Cumulative intraday return
            - o2cC: Market-weighted beta component
            - o2cC_B: Winsorized signal
            - o2cC_B_ma: Industry-demeaned signal (main intraday signal)

    Signal Formula:
        cur_log_ret = overnight_log_ret + log(iclose/dopen)
        o2cC = pbeta * (cap_weighted_intraday_market_return)
        o2cC_B = winsorize_by_group(o2cC, groupby='iclose_ts')
        o2cC_B_ma = industry_demean(o2cC_B)

    Notes:
        - Uses wavg2 for market-weighted calculation
        - Combines overnight + intraday for total return
        - Uses merge_intra_calcs() for merging
        - More sophisticated than simple beta division
    """
    print "Calculating o2c intra..."

    result_df = filter_expandable_intra(intra_df, daily_df)
    result_df = result_df.reset_index()    
    result_df = result_df[ ['iclose_ts', 'iclose', 'dopen', 'overnight_log_ret', 'pbeta', 'date', 'ind1', 'sid', 'mkt_cap_y' ] ]
    result_df = result_df.dropna(how='any')

    print "Calulating o2cC..."
    result_df['cur_log_ret'] = result_df['overnight_log_ret'] + (np.log(result_df['iclose']/result_df['dopen']))
    result_df['o2cC'] = result_df[['cur_log_ret', 'pbeta', 'mkt_cap_y', 'iclose_ts']].groupby(['iclose_ts'], sort=False).apply(wavg2)
    result_df['o2cC_B'] = winsorize_by_group(result_df[ ['iclose_ts', 'o2cC'] ], 'iclose_ts')

    print "Calulating o2cC_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['o2cC_B', 'iclose_ts', 'ind1']].groupby(['iclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['o2cC_B_ma'] = indgroups['o2cC_B']

    result_df = merge_intra_calcs(intra_df, result_df)
    return result_df

def o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None):
    """
    Fit regression and generate market-weighted beta-adjusted forecast.

    Fits regressions using the market-weighted o2c0_B_ma signal, then combines
    with lagged signals to create final forecast. Structure similar to
    badj_multi.py but uses more sophisticated signal.

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
            - o2c{1..horizon-1}_B_ma_coef: Coefficients for lagged daily signals
            - badj2_m: Combined forecast (main output)

    Regression Strategy:
    -------------------
    - Fits o2c0_B_ma at multiple lags (1 to horizon)
    - Uses 'daily' regression mode
    - Extracts coefficient at full horizon (coef0)
    - Computes residual coefficients: coef[lag] = coef0 - fitted_coef[lag]

    Final Forecast:
        badj2_m = o2cC_B_ma * 0 +  # Intraday disabled (line 93)
                  sum_{lag=1}^{horizon-1} (o2c{lag}_B_ma * residual_coef[lag])

    Output Files:
        - badj_daily_{name}_{dates}.png: Daily regression plot

    Notes:
        - Intraday coefficient hardcoded to 0 (line 93)
        - Only daily lags contribute to forecast
        - Sector-specific fitting (Energy vs others)
        - All coefficients printed to console
        - Output column named 'badj2_m' (note the '2')
    """
    if 'badj_m' not in full_df.columns:
        print "Creating forecast columns..."
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

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'o2c0_B_ma', lag, outsample, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)  
    plot_fit(fits_df, "badj_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['o2c0_B_ma'].ix[horizon].ix['coef']
    full_df.ix[ outsample_intra_df.index, 'o2cC_B_ma_coef' ] = 0#coef0
    print "{} Coef0: {}".format(name, coef0)
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['o2c0_B_ma'].ix[lag].ix['coef'] 
        print "{} Coef{}: {}".format(name, lag, coef)
        full_df.ix[ outsample_intra_df.index, 'o2c'+str(lag)+'_B_ma_coef' ] = coef

    full_df.ix[ outsample_intra_df.index, 'badj2_m'] = full_df['o2cC_B_ma'] * full_df['o2cC_B_ma_coef']
    for lag in range(1,horizon):
        full_df.ix[ outsample_intra_df.index, 'badj2_m'] += full_df['o2c'+str(lag)+'_B_ma'] * full_df['o2c'+str(lag)+'_B_ma_coef']

    return full_df

def calc_o2c_forecast(daily_df, intra_df, horizon, outsample):
    """
    Main entry point: compute market-weighted beta-adjusted forecast with sector split.

    Orchestrates the pipeline using market-weighted beta adjustment approach,
    with separate fitting for Energy vs other sectors.

    Args:
        daily_df (pd.DataFrame): Daily price/factor data with MultiIndex (date, sid)
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid)
        horizon (int): Forecast horizon in days (typically 3)
        outsample (bool): If True, split data in-sample/out-of-sample

    Returns:
        pd.DataFrame: Full dataset with 'badj2_m' forecasts

    Pipeline Flow:
        daily_df → calc_o2c() → daily with market-weighted lags
        intra_df → calc_o2c_intra() → intra with market-weighted signal
        [merge daily + intra]
        → o2c_fits(Energy sector) → partial forecasts
        → o2c_fits(ex-Energy) → complete forecasts

    Sector Processing:
        1. Energy sector: Market-weighted fitting for Energy stocks
        2. Ex-Energy: Market-weighted fitting for non-Energy stocks

    Notes:
        - middate computed as midpoint if outsample=True
        - Only outsample data returned if outsample=True
        - Final forecast in 'badj2_m' column
        - Uses market-weighted approach (not simple division)
        - Code has bug on line 166: references undefined outsample_df variable
    """
    daily_df = calc_o2c(daily_df, horizon) 
    intra_df = calc_o2c_intra(intra_df, daily_df)
    full_df = merge_intra_data(daily_df, intra_df)

    middate = None
    if outsample:
        middate = intra_df.index[0][0] + (intra_df.index[len(intra_df)-1][0] - intra_df.index[0][0]) / 2
        print "Setting fit period before {}".format(middate)

    sector_name = 'Energy'
    print "Running o2c for sector {}".format(sector_name)
    sector_df = daily_df[ daily_df['sector_name'] == sector_name ]
    sector_intra_df = intra_df[ intra_df['sector_name'] == sector_name ]
    full_df = o2c_fits(sector_df, sector_intra_df, full_df, horizon, "in", middate)

    print "Running o2c for sector {}".format(sector_name)
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
    pname = "./badj2_m" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print "Did not load cached data..."

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        barra_df = load_barra(uni_df, start, end)
        price_df = load_prices(uni_df, start, end)
        daily_df = merge_barra_data(price_df, barra_df)
        daybar_df = load_daybars(uni_df, start, end)
        intra_df = merge_intra_data(daily_df, daybar_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    full_df = calc_o2c_forecast(daily_df, intra_df, horizon, outsample)

    dump_alpha(outsample_df, 'badj2_m')
    dump_all(outsample_df)

