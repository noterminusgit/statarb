#!/usr/bin/env python
"""Beta-Adjusted Order Flow Alpha Strategy

Generates trading signals based on order book flow imbalance (bid/ask hit ratio),
adjusted for market beta to create a market-neutral signal. The strategy exploits
the informational content in aggressive order flow while removing market-driven
effects through beta adjustment.

Methodology:
-----------
1. Calculate beta-adjusted returns by removing market component:
   - Compute market return as cap-weighted average of individual returns
   - Beta-adjust: badjret = log_ret - (pbeta * market_return)

2. Compute order flow imbalance signal:
   - bd0 = (askHitDollars - bidHitDollars) / (askHitDollars + midHitDollars + bidHitDollars)
   - Normalize by sqrt(spread_bps) to adjust for liquidity
   - Winsorize to control outliers

3. Industry demean to ensure sector neutrality:
   - Within each date/industry group, subtract industry mean
   - Creates bd0_B_ma (market-adjusted, industry-demeaned signal)

4. Regression-based coefficient estimation:
   - Fit daily lagged signals (bd1, bd2, ...) against forward returns
   - Fit intraday signal (bdC) with time-of-day coefficients (6 hourly buckets)
   - Combine signals: bdma = sum(coef[i] * bd[i]_B_ma)

Signal Interpretation:
---------------------
- Positive bd signal: More aggressive buying (ask hits) than selling (bid hits)
  -> Expect continuation (mean reversion of order flow)
- Negative bd signal: More aggressive selling than buying
  -> Expect downward pressure

The beta adjustment removes confounding market effects, ensuring the signal
captures stock-specific order flow information rather than broad market moves.

Strategy Variants:
-----------------
The 'bd' family includes several implementations:
- bd.py: Base beta-adjusted order flow (this file)
- bd1.py: Simplified variant
- bd_intra.py: Pure intraday implementation
- badj_*.py: Alternative beta-adjustment approaches
- badj_dow_multi.py: Day-of-week variant for calendar effects

Operating Modes:
---------------
The module operates on both daily and intraday timeframes:

Daily Mode (calc_bd_daily):
- Computes lagged daily signals (bd0, bd1, bd2, ...)
- Beta-adjusts overnight returns
- Stores historical lags for regression fitting

Intraday Mode (calc_bd_intra):
- Computes current-bar signal (bdC) from intraday bar data
- Beta-adjusts intraday returns (open-to-current)
- Generates time-of-day specific coefficients

Combined Forecast (bd_fits):
- Fits regressions separately for intraday and daily signals
- Applies time-varying coefficients for intraday (6 hourly buckets)
- Computes residual coefficients for lagged daily signals
- Final forecast: bdma = weighted sum of all components

Data Requirements:
-----------------
- Barra factors: ind1 (industry), pbeta (predicted beta)
- Price data: close, log_ret, mkt_cap_y
- Bar data: askHitDollars, bidHitDollars, midHitDollars, spread_bps, bopen
- Filtered to expandable universe (liquid, tradeable stocks)

CLI Usage:
---------
Run full backtest with in-sample/out-of-sample split:
    python bd.py --start=20130101 --end=20130630 --mid=20130315 --horizon=3 --freq=15

Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --mid: Mid-date for in-sample/out-of-sample split (optional)
    --freq: Bar frequency in minutes (default: 15)
    --horizon: Forecast horizon in days (default: 3)

Output:
------
- Regression plots: bdma_intra_{name}_{dates}.png, bdma_daily_{name}_{dates}.png
- HDF5 cache: bd{start}.{end}_daily.h5, bd{start}.{end}_intra.h5
- Alpha forecast: 'bdma' column written via dump_alpha()

Related Modules:
---------------
- new1.py: Similar beta-adjusted approach using 'insideness' metric
- regress.py: Regression fitting framework
- calc.py: Forward returns and winsorization utilities
- util.py: Data merging and helper functions

References:
----------
Order flow literature suggests aggressive order imbalance predicts short-term
returns. This implementation combines that insight with beta-neutral positioning
to isolate stock-specific information from market-wide effects.
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def wavg(group):
    """
    Calculate market return weighted by market cap and scaled by beta.

    Used to compute the market component of returns that will be subtracted
    from individual stock returns to create beta-adjusted (market-neutral) returns.

    The formula computes:
        beta * market_return

    where market_return is the capitalization-weighted average return across
    all stocks in the group (typically all stocks on a given date).

    Args:
        group (pd.DataFrame): DataFrame group containing:
            - pbeta: Predicted beta from Barra risk model
            - log_ret: Log return for the day
            - mkt_cap_y: Market capitalization (in dollars)

    Returns:
        pd.Series: Beta-scaled market return for each stock in the group.
                   Has same length as input group, with each stock getting
                   its beta times the overall market return.

    Formula:
        market_return = sum(log_ret * weight) / sum(weight)
        where weight = mkt_cap_y / 1e6 (market cap in millions)

        result = pbeta * market_return

    Notes:
        - Market cap is scaled by 1e6 (millions) to avoid numerical issues
        - Each stock gets same market return but scaled by its own beta
        - Used in groupby().apply() pattern with grouping by date
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg2(group):
    """
    Calculate intraday market return weighted by market cap and scaled by beta.

    Identical to wavg() but operates on intraday returns (cur_log_ret) rather
    than daily returns. Used to beta-adjust intraday bar returns.

    Args:
        group (pd.DataFrame): DataFrame group containing:
            - pbeta: Predicted beta from Barra risk model
            - cur_log_ret: Current intraday log return (open to current bar)
            - mkt_cap_y: Market capitalization (in dollars)

    Returns:
        pd.Series: Beta-scaled intraday market return for each stock.

    Notes:
        - Used in groupby().apply() with grouping by timestamp (giclose_ts)
        - Enables beta adjustment within each intraday bar
        - Same methodology as wavg() but for intraday data
    """
    b = group['pbeta']
    d = group['cur_log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def calc_bd_daily(daily_df, horizon):
    """
    Calculate daily beta-adjusted order flow signals with multiple lags.

    Computes the core 'bd' (beta-adjusted order flow) signal for daily data,
    including beta-adjusted returns and lagged versions for use in multi-period
    regression fitting.

    Process:
    1. Filter to expandable (tradeable, liquid) universe
    2. Compute beta-adjusted returns (remove market component)
    3. Calculate order flow imbalance (ask vs bid hits)
    4. Normalize by spread and winsorize
    5. Industry demean for sector neutrality
    6. Create lagged versions (bd1, bd2, ..., bd{horizon})

    Args:
        daily_df (pd.DataFrame): Daily data with MultiIndex (date, sid) containing:
            - log_ret: Daily log returns
            - pbeta: Predicted beta from Barra
            - mkt_cap_y: Market capitalization
            - askHitDollars: Dollar volume of aggressive buys (ask hits)
            - bidHitDollars: Dollar volume of aggressive sells (bid hits)
            - midHitDollars: Dollar volume at midpoint
            - spread_bps: Bid-ask spread in basis points
            - ind1: Industry classification (for demeaning)
            - expandable: Boolean filter for tradeable stocks

        horizon (int): Number of lagged signals to create (typically 3-5 days)

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - bret: Beta-scaled market return
            - badjret: Beta-adjusted return (log_ret - bret)
            - bd0: Raw order flow imbalance ratio
            - bd0_B: Winsorized, spread-adjusted order flow
            - bd0_B_ma: Industry-demeaned signal (main daily signal)
            - bd{1..horizon}_B_ma: Lagged versions of bd0_B_ma
            - bd{1..horizon}_B: Lagged versions of bd0_B

    Signal Formula:
        bd0 = (askHitDollars - bidHitDollars) /
              (askHitDollars + midHitDollars + bidHitDollars)

        bd0_B = winsorize(bd0 / sqrt(spread_bps) / 10000.0)
        bd0_B_ma = industry_demean(bd0_B)

    Notes:
        - Normalization by sqrt(spread_bps) adjusts for liquidity differences
        - Division by 10000 scales the signal to reasonable magnitudes
        - Industry demeaning ensures sector neutrality
        - Commented code shows experimental variants (interaction with returns)
        - Lagged signals enable multi-horizon regression fitting
    """
    print("Caculating daily bd...")
    result_df = filter_expandable(daily_df)

#    decile = lambda x: 10.0 * x.rank()/float(len(x))
#    result_df['log_ret_decile'] = result_df[['log_ret', 'gdate']].groupby(['gdate'], sort=False).transform(decile)['log_ret']
    result_df['bret'] = result_df[['log_ret', 'pbeta', 'mkt_cap_y', 'gdate']].groupby('gdate').apply(wavg).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['log_ret'] - result_df['bret']

    print("Calculating bd0...")
    result_df['bd0'] = (result_df['askHitDollars'] - result_df['bidHitDollars']) / (result_df['askHitDollars'] + result_df['midHitDollars'] + result_df['bidHitDollars'])
    result_df['bd0_B'] = winsorize_by_date( result_df['bd0'] / np.sqrt(result_df['spread_bps']) / 10000.0)

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['bd0_B', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=False).transform(demean)
    result_df['bd0_B_ma'] = indgroups['bd0_B']
 #   result_df['bd0_B_ma'] = result_df['bd0_B_ma'] * np.abs(result_df['badjret'])
#    result_df['bd0_B_ma'] =  result_df['bd0_B_ma'].clip(0,1000) * np.sign(result_df['log_ret'])
    #    result_df.loc[ (result_df['log_ret_decile'] < 2) | (result_df['log_ret_decile'] == 9), 'bd0_B_ma'] = np.nan

    print("Calulating lags...")
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['bd'+str(lag)+'_B_ma'] = shift_df['bd0_B_ma']
        result_df['bd'+str(lag)+'_B'] = shift_df['bd0_B']

    return result_df

def calc_bd_intra(intra_df):
    """
    Calculate intraday beta-adjusted order flow signals.

    Computes the 'bdC' (current bar) signal for intraday 30-minute bar data,
    with beta adjustment and industry demeaning. The 'C' suffix indicates
    "current" or intraday signal.

    Process:
    1. Filter to expandable universe
    2. Compute intraday return (open to current bar close)
    3. Beta-adjust intraday returns
    4. Calculate order flow imbalance for the bar
    5. Normalize by spread and winsorize (by timestamp)
    6. Industry demean within each timestamp

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid):
            - iclose: Intraday bar close price
            - bopen: Beginning-of-day open price
            - pbeta: Predicted beta from Barra
            - mkt_cap_y: Market capitalization
            - askHitDollars: Aggressive buy volume in bar
            - bidHitDollars: Aggressive sell volume in bar
            - midHitDollars: Mid-price trade volume
            - spread_bps: Bid-ask spread in basis points
            - ind1: Industry classification
            - giclose_ts: Bar close timestamp
            - expandable: Boolean filter for tradeable stocks

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - cur_log_ret: Intraday log return (bopen to iclose)
            - bret: Beta-scaled market return for this timestamp
            - badjret: Beta-adjusted intraday return
            - bdC: Raw order flow imbalance for current bar
            - bdC_B: Winsorized, spread-adjusted order flow
            - bdC_B_ma: Industry-demeaned intraday signal (main intraday signal)

    Signal Formula:
        Same as daily: bdC = (ask - bid) / (ask + mid + bid)
        bdC_B = winsorize_by_ts(bdC / sqrt(spread_bps) / 10000.0)
        bdC_B_ma = industry_demean(bdC_B)

    Notes:
        - winsorize_by_ts applies winsorization within each timestamp cross-section
        - Industry demeaning is done within (timestamp, industry) groups
        - Commented code shows experimental variants:
          * Interaction with beta-adjusted returns
          * Clipping and directional filters
          * Time-of-day decay scaling (distance from EOD)
        - Does not create lags (intraday lags handled differently)
    """
    print("Calculating bd intra...")
    result_df = filter_expandable(intra_df)

    result_df['cur_log_ret'] = np.log(result_df['iclose']/result_df['bopen'])
    result_df['bret'] = result_df[['cur_log_ret', 'pbeta', 'mkt_cap_y', 'giclose_ts']].groupby(['giclose_ts'], sort=False).apply(wavg2).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['cur_log_ret'] - result_df['bret']

#    decile = lambda x: 10.0 * x.rank()/float(len(x))
#    result_df['cur_log_ret_decile'] = result_df[['cur_log_ret', 'giclose_ts']].groupby(['giclose_ts'], sort=False).transform(decile)['cur_log_ret']

    print("Calulating bdC...")
    result_df['bdC'] = (result_df['askHitDollars'] - result_df['bidHitDollars']) / (result_df['askHitDollars'] + result_df['midHitDollars'] + result_df['bidHitDollars'])
    result_df['bdC_B'] = winsorize_by_ts(result_df['bdC'] / np.sqrt(result_df['spread_bps']) / 10000.0)

    print("Calulating bdC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['bdC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['bdC_B_ma'] = indgroups['bdC_B']
#    result_df['bdC_B_ma'] = result_df['bdC_B_ma'] * np.abs(result_df['badjret'])

#    result_df['bdC_B_ma'] =  result_df['bdC_B_ma'].clip(0,1000) * np.sign(result_df['cur_log_ret'])
#    result_df.loc[ (result_df['cur_log_ret_decile'] < 1) | (result_df['cur_log_ret_decile'] == 9), 'bdC_B_ma'] = np.nan
#    result_df['bdC_B_ma'] = result_df['bdC_B_ma'] * (2 - result_df['cur_log_ret_r'])

    # result_df['eod_ts'] = result_df['date'].apply(lambda x: x + timedelta(hours=15, minutes=30))
    # result_df['scale'] = result_df['eod_ts'] - result_df['giclose_ts']
    # result_df['scale'] = result_df['scale'].apply(lambda x: 1.0 - (x/np.timedelta64(1, 's'))/(360*60))
    # result_df[ result_df['scale'] > 1 ] = 0
    # result_df['bdC_B_ma_tod'] = result_df['bdC_B_ma'] * result_df['scale']

    return result_df

def bd_fits(daily_df, intra_df, horizon, name, middate):
    """
    Fit regression coefficients and generate combined beta-adjusted order flow forecast.

    This is the core forecasting function that:
    1. Splits data into in-sample (fitting) and out-of-sample (forecasting) periods
    2. Fits separate regressions for intraday and daily signals
    3. Applies time-varying coefficients for intraday (6 hourly buckets)
    4. Computes residual coefficients for lagged daily signals
    5. Combines all signals into final 'bdma' forecast

    The strategy uses two types of signals:
    - Intraday (bdC_B_ma): Current bar order flow, fitted by time-of-day
    - Daily lags (bd1_B_ma, bd2_B_ma, ...): Historical daily signals

    Args:
        daily_df (pd.DataFrame): Daily data with bd signals (from calc_bd_daily)
        intra_df (pd.DataFrame): Intraday data with bdC signals (from calc_bd_intra)
        horizon (int): Forecast horizon in days (typically 3)
        name (str): Name suffix for output plots (e.g., "in", "out", "")
        middate (datetime): Split date for in-sample vs out-of-sample
                           If None, uses entire dataset for both fit and forecast

    Returns:
        pd.DataFrame: Out-of-sample intraday dataframe with added columns:
            - bdC_B_ma_coef: Time-of-day specific coefficient for intraday signal
            - bd{1..horizon-1}_B_ma_coef: Coefficients for lagged daily signals
            - bdma: Combined forecast (main output)

    Regression Strategy:
    -------------------
    Intraday Regression:
        - Fits bdC_B_ma against forward returns using regress_alpha(..., 'intra')
        - Uses median regression (3-fold) for robustness
        - Generates 6 separate coefficients for time-of-day buckets:
          Bucket 1: 09:30-10:31
          Bucket 2: 10:30-11:31
          Bucket 3: 11:30-12:31
          Bucket 4: 12:30-13:31
          Bucket 5: 13:30-14:31
          Bucket 6: 14:30-15:59
        - Overlapping buckets smooth transitions between time periods

    Daily Regression:
        - Fits bd0_B_ma at multiple lags (1 to horizon days)
        - Uses daily regression against cumulative forward returns
        - Extracts coefficient at full horizon (coef0)
        - Computes residual coefficients: coef[lag] = coef0 - fitted_coef[lag]
        - This captures incremental information from recent lags

    Final Forecast:
        bdma = bdC_B_ma * bdC_B_ma_coef +
               sum_{lag=1}^{horizon-1} (bd{lag}_B_ma * bd{lag}_B_ma_coef)

    Output Files:
        - bdma_intra_{name}_{dates}.png: Intraday regression plot
        - bdma_daily_{name}_{dates}.png: Daily regression plot

    Notes:
        - Only out-of-sample data gets forecasts (in-sample used for fitting only)
        - Time-of-day coefficients capture intraday patterns in signal efficacy
        - Lagged daily signals provide persistence/momentum information
        - All coefficients printed to console for monitoring
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    outsample = False
    if middate is not None:
        outsample = True
        insample_intra_df = intra_df[ intra_df['date'] <  middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['bdma'] = np.nan
    outsample_intra_df['bdC_B_ma_coef'] = np.nan
    for lag in range(0, horizon+1):
        outsample_intra_df[ 'bd' + str(lag) + '_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df = regress_alpha(insample_intra_df, 'bdC_B_ma', horizon, True, 'intra')
    fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "bdma_intra_"+name+"_" + df_dates(insample_intra_df))
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
        outsample_intra_df.loc[ coefs[ii], 'bdC_B_ma_coef' ] = fits_df.loc['bdC_B_ma'].loc[ii].loc['coef']

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'bd0_B_ma', lag, outsample, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "bdma_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    coef0 = fits_df.loc['bd0_B_ma'].loc[horizon].loc['coef']
#    full_df.loc[ outsample_intra_df.index, 'bdC_B_ma_coef' ] = coef0
    print("Coef0: {}".format(coef0))
    for lag in range(1,horizon):
        coef = coef0 - fits_df.loc['bd0_B_ma'].loc[lag].loc['coef']
        print("Coef{}: {}".format(lag, coef))
        outsample_intra_df[ 'bd'+str(lag)+'_B_ma_coef' ] = coef

    outsample_intra_df[ 'bdma'] = outsample_intra_df['bdC_B_ma'] * outsample_intra_df['bdC_B_ma_coef']
    for lag in range(1,horizon):
        outsample_intra_df['bdma'] += outsample_intra_df['bd'+str(lag)+'_B_ma'] * outsample_intra_df['bd'+str(lag)+'_B_ma_coef']

    return outsample_intra_df

def calc_bd_forecast(daily_df, intra_df, horizon):
    """
    Main entry point: compute beta-adjusted order flow forecast end-to-end.

    Orchestrates the full pipeline from raw data to final alpha forecast:
    1. Calculate daily bd signals with lags
    2. Calculate forward returns for regression fitting
    3. Calculate intraday bdC signals
    4. Merge daily and intraday data
    5. Fit regressions and generate forecast

    Args:
        daily_df (pd.DataFrame): Daily price/factor data with MultiIndex (date, sid)
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid)
        horizon (int): Forecast horizon in days (typically 3)

    Returns:
        pd.DataFrame: Intraday dataframe with 'bdma' forecast column

    Pipeline Flow:
        daily_df → calc_bd_daily() → daily_results_df
        daily_df → calc_forward_returns() → forwards_df
        [merge daily_results_df + forwards_df]
        intra_df → calc_bd_intra() → intra_results_df
        [merge daily + intra data]
        → bd_fits() → full_df with 'bdma' forecast

    Notes:
        - Uses module-level 'middate' variable for in/out sample split
        - Forward returns required for regression fitting
        - Final output suitable for dump_alpha() or backtesting
    """
    daily_results_df = calc_bd_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    intra_results_df = calc_bd_intra(intra_df)
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)

    full_df = bd_fits(daily_results_df, intra_results_df, horizon, "", middate)

    return full_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--freq",action="store",dest="freq",default=15)
    parser.add_argument("--horizon",action="store",dest="horizon",default=3)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = int(args.horizon)
    freq = int(args.freq)
    pname = "./bd" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)

    loaded = False
    try:
        daily_df = pd.read_hdf(pname+"_daily.h5", 'table')
        intra_df = pd.read_hdf(pname+"_intra.h5", 'table')
        loaded = True
    except:
        print("Did not load cached data...")

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1', 'pbeta']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close', 'overnight_log_ret', 'tradable_volume', 'tradable_med_volume_21']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        BAR_COLS = ['askHitDollars', 'midHitDollars', 'bidHitDollars', 'bopen', 'spread_bps']
        intra_df = load_bars(price_df[['ticker']], start, end, BAR_COLS, freq)
        daily_df = merge_barra_data(price_df, barra_df)
        daily_df = merge_intra_eod(daily_df, intra_df)
        intra_df = merge_intra_data(daily_df, intra_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')
    
    outsample_df = calc_bd_forecast(daily_df, intra_df, horizon)
    dump_alpha(outsample_df, 'bdma')
    # dump_alpha(outsample_df, 'bdC_B_ma')
    # dump_alpha(outsample_df, 'bd0_B_ma')
    # dump_all(outsample_df)



