#!/usr/bin/env python
"""Pure Intraday Beta-Adjusted Order Flow Alpha Strategy

Focuses exclusively on intraday order flow signals with beta adjustment,
without incorporating daily lag components. This is a streamlined version
of bd.py designed for pure intraday trading signals.

Key Differences from bd.py:
---------------------------
1. Scope:
   - bd.py: Combines daily lags (bd1, bd2, ...) + intraday (bdC)
   - bd_intra.py: ONLY intraday signal (bdC) - no daily components
   - Simpler single-signal approach

2. Signal Normalization:
   - bd.py: Normalizes by sqrt(spread_bps) to adjust for liquidity
   - bd_intra.py: No spread normalization - just winsorization
   - Simpler preprocessing

3. Regression:
   - bd.py: Separate regressions for intraday + daily lags
   - bd_intra.py: Single intraday regression only
   - Uses 'intra_eod' regression mode for end-of-day returns

4. Forward Returns:
   - bd.py: Computes forward returns from daily data
   - bd_intra.py: calc_forward_returns() on daily_df only
   - No forward return calculation from intraday bars

Methodology:
-----------
1. Beta-adjust intraday returns:
   - Compute market return as cap-weighted average at each timestamp
   - Subtract beta-scaled market return: badjret = cur_log_ret - (pbeta * mkt_ret)

2. Calculate order flow imbalance for current bar:
   - bdC = (askHitDollars - bidHitDollars) / (total hit dollars)
   - Winsorize by timestamp to control outliers
   - Industry demean for sector neutrality

3. Time-varying coefficients:
   - Fit regression using 'intra_eod' mode
   - Generates 6 hourly coefficients (09:30-15:59)
   - Overlapping buckets smooth transitions

Signal Formula:
--------------
bdC = (askHitDollars - bidHitDollars) /
      (askHitDollars + midHitDollars + bidHitDollars)

bdC_B = winsorize_by_ts(bdC)
bdC_B_ma = industry_demean(bdC_B)

Final forecast: bdma_i = bdC_B_ma * bdC_B_ma_coef(time_of_day)

Beta Adjustment Formula:
-----------------------
cur_log_ret = log(iclose / bopen)
market_return = sum(cur_log_ret * weight) / sum(weight)
    where weight = mkt_cap_y / 1e6
badjret = cur_log_ret - (pbeta * market_return)

Use Case:
--------
- Pure intraday trading without overnight positions
- Testing order flow signals in isolation
- Lower latency forecasting (no daily lag dependencies)
- Intraday execution strategies

Data Requirements:
-----------------
- Barra factors: ind1 (industry), pbeta (predicted beta)
- Price data: bopen (beginning-of-day open), iclose (bar close)
- Bar data: askHitDollars, bidHitDollars, midHitDollars
- Market cap: mkt_cap_y for market return calculation
- Filtered to expandable universe

CLI Usage:
---------
Run full backtest with in-sample/out-of-sample split:
    python bd_intra.py --start=20130101 --end=20130630 --mid=20130315 --freq=15

Arguments:
    --start: Start date (YYYYMMDD)
    --end: End date (YYYYMMDD)
    --mid: Mid-date for in-sample/out-of-sample split
    --freq: Bar frequency in minutes (default: 15)

Output:
------
- Regression plot: bdma_intra_{dates}.png
- HDF5 cache: bd{start}.{end}_daily.h5, bd{start}.{end}_intra.h5
- Alpha forecast: 'bdma_i' column written via dump_alpha()

Related Modules:
---------------
- bd.py: Full beta-adjusted order flow (daily + intraday)
- bd1.py: Simplified variant using differenced order flow
- regress.py: Regression fitting framework with 'intra_eod' mode
- calc.py: Forward returns calculation

Notes:
-----
- horizon parameter in __main__ set to 0 (likely should be 3-5)
- Commented code shows experimental variants (interactions, clipping)
- Time-of-day coefficients capture intraday pattern evolution
- Beta adjustment removes market-driven effects from order flow
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def wavg(group):
    """
    Calculate daily market return weighted by market cap and scaled by beta.

    Used for daily beta adjustment (though bd_intra.py focuses on intraday).
    Included for compatibility but not actively used in main pipeline.

    Args:
        group (pd.DataFrame): DataFrame group with pbeta, log_ret, mkt_cap_y

    Returns:
        pd.Series: Beta-scaled market return for each stock

    Formula:
        market_return = sum(log_ret * weight) / sum(weight)
        result = pbeta * market_return
    """
    b = group['pbeta']
    d = group['log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def wavg2(group):
    """
    Calculate intraday market return weighted by market cap and scaled by beta.

    Computes the market component of intraday returns that will be subtracted
    to create beta-adjusted (market-neutral) returns for the current bar.

    Args:
        group (pd.DataFrame): DataFrame group containing:
            - pbeta: Predicted beta from Barra risk model
            - cur_log_ret: Current intraday log return (open to current bar)
            - mkt_cap_y: Market capitalization (in dollars)

    Returns:
        pd.Series: Beta-scaled intraday market return for each stock in group

    Formula:
        market_return = sum(cur_log_ret * weight) / sum(weight)
        where weight = mkt_cap_y / 1e6
        result = pbeta * market_return

    Notes:
        - Used in groupby().apply() pattern with grouping by giclose_ts
        - Each timestamp gets its own market return calculation
        - Market cap scaled by 1e6 (millions) to avoid numerical issues
    """
    b = group['pbeta']
    d = group['cur_log_ret']
    w = group['mkt_cap_y'] / 1e6
    res = b * ((d * w).sum() / w.sum())
    return res

def calc_bd_intra(intra_df):
    """
    Calculate pure intraday beta-adjusted order flow signals.

    Computes the 'bdC' (current bar) order flow signal with full beta
    adjustment and industry demeaning. This is the core signal for
    intraday-only trading without daily lag components.

    Process:
    1. Filter to expandable (tradeable, liquid) universe
    2. Compute intraday return (beginning-of-day open to current bar close)
    3. Beta-adjust intraday returns (remove market component)
    4. Calculate order flow imbalance for current bar
    5. Winsorize by timestamp to control outliers
    6. Industry demean within (timestamp, industry) groups

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid):
            - iclose: Intraday bar close price
            - bopen: Beginning-of-day open price
            - pbeta: Predicted beta from Barra
            - mkt_cap_y: Market capitalization
            - askHitDollars: Aggressive buy volume in bar
            - bidHitDollars: Aggressive sell volume in bar
            - midHitDollars: Mid-price trade volume
            - ind1: Industry classification
            - giclose_ts: Bar close timestamp
            - expandable: Boolean filter for tradeable stocks

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - cur_log_ret: Intraday log return (bopen to iclose)
            - bret: Beta-scaled market return for this timestamp
            - badjret: Beta-adjusted intraday return
            - bdC: Raw order flow imbalance for current bar
            - bdC_B: Winsorized order flow
            - bdC_B_ma: Industry-demeaned signal (main intraday signal)

    Signal Formula:
        bdC = (askHitDollars - bidHitDollars) /
              (askHitDollars + midHitDollars + bidHitDollars)
        bdC_B = winsorize_by_ts(bdC)
        bdC_B_ma = industry_demean(bdC_B)

    Notes:
        - No spread normalization (unlike bd.py)
        - Beta adjustment removes market-driven effects
        - Industry demeaning ensures sector neutrality
        - Commented code shows experimental variants:
          * Interaction with beta-adjusted returns
          * Clipping and directional filters
          * Time-of-day decay scaling
    """
    print("Calculating bd intra...")
    result_df = filter_expandable(intra_df)

    result_df['cur_log_ret'] = np.log(result_df['iclose']/result_df['bopen'])
    result_df['bret'] = result_df[['cur_log_ret', 'pbeta', 'mkt_cap_y', 'giclose_ts']].groupby(['giclose_ts'], sort=False).apply(wavg2).reset_index(level=0)['pbeta']
    result_df['badjret'] = result_df['cur_log_ret'] - result_df['bret']

#    decile = lambda x: 10.0 * x.rank()/float(len(x))
#    result_df['cur_log_ret_decile'] = result_df[['cur_log_ret', 'giclose_ts']].groupby(['giclose_ts'], sort=False).transform(decile)['cur_log_ret']

    print("Calulating bdC..."    )
    result_df['bdC'] = (result_df['askHitDollars'] - result_df['bidHitDollars']) / (result_df['askHitDollars'] + result_df['midHitDollars'] + result_df['bidHitDollars'])
    result_df['bdC_B'] = winsorize_by_ts(result_df['bdC'])

    print("Calulating bdC_ma...")
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['bdC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=False).transform(demean)
    result_df['bdC_B_ma'] = indgroups['bdC_B']
#    result_df['bdC_B_ma'] = result_df['bdC_B_ma'] * np.abs(result_df['badjret'])

#    result_df['bdC_B_ma'] =  result_df['bdC_B_ma'].clip(0,1000) * np.sign(result_df['cur_log_ret'])
#    result_df.ix[ (result_df['cur_log_ret_decile'] < 1) | (result_df['cur_log_ret_decile'] == 9), 'bdC_B_ma'] = np.nan
#    result_df['bdC_B_ma'] = result_df['bdC_B_ma'] * (2 - result_df['cur_log_ret_r'])

    # result_df['eod_ts'] = result_df['date'].apply(lambda x: x + timedelta(hours=15, minutes=30))
    # result_df['scale'] = result_df['eod_ts'] - result_df['giclose_ts'] 
    # result_df['scale'] = result_df['scale'].apply(lambda x: 1.0 - (x/np.timedelta64(1, 's'))/(360*60))
    # result_df[ result_df['scale'] > 1 ] = 0
    # result_df['bdC_B_ma_tod'] = result_df['bdC_B_ma'] * result_df['scale']

    return result_df

def bd_fits(intra_df, horizon, name, middate):
    """
    Fit intraday regression and generate beta-adjusted order flow forecast.

    Fits a single regression for the intraday bdC_B_ma signal with time-varying
    coefficients across 6 hourly buckets. This is simpler than bd.py which
    also fits daily lag components.

    The regression uses 'intra_eod' mode which fits signals against end-of-day
    returns, capturing the predictive power of intraday order flow for
    close-to-close returns.

    Args:
        intra_df (pd.DataFrame): Intraday data with bdC signals
        horizon (int): Forecast horizon (used in regression, typically 0-5)
        name (str): Name suffix for output plots (e.g., "in", "out", "")
        middate (datetime): Split date for in-sample vs out-of-sample
                           If None, uses entire dataset for both fit and forecast

    Returns:
        pd.DataFrame: Out-of-sample intraday dataframe with added columns:
            - bdC_B_ma_coef: Time-of-day specific coefficient for intraday signal
            - bdma_i: Combined forecast (main output)

    Regression Strategy:
    -------------------
    - Fits bdC_B_ma against forward returns using 'intra_eod' mode
    - Generates 6 separate coefficients for time-of-day buckets:
      Bucket 1: 09:30-10:31
      Bucket 2: 10:30-11:31
      Bucket 3: 11:30-12:31
      Bucket 4: 12:30-13:31
      Bucket 5: 13:30-14:31
      Bucket 6: 14:30-15:59
    - Overlapping buckets smooth transitions between time periods

    Final Forecast:
        bdma_i = bdC_B_ma * bdC_B_ma_coef(time_of_day)

    Output Files:
        - bdma_intra_{name}_{dates}.png: Intraday regression plot

    Notes:
        - Only out-of-sample data gets forecasts
        - In-sample used for fitting coefficients only
        - Time-of-day coefficients capture intraday pattern evolution
        - No daily lag components unlike bd.py
        - All coefficients printed to console for monitoring
    """
    insample_intra_df = intra_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] <  middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['bdma'] = np.nan
    outsample_intra_df['bdC_B_ma_coef'] = np.nan
    for lag in range(0, horizon+1):
        outsample_intra_df[ 'bd' + str(lag) + '_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df = regress_alpha(insample_intra_df, 'bdC_B_ma', horizon, True, 'intra_eod')
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
        outsample_intra_df.ix[ coefs[ii], 'bdC_B_ma_coef' ] = fits_df.ix['bdC_B_ma'].ix[ii].ix['coef']

    outsample_intra_df[ 'bdma_i'] = outsample_intra_df['bdC_B_ma'] * outsample_intra_df['bdC_B_ma_coef']
    return outsample_intra_df

def calc_bd_forecast(daily_df, intra_df, horizon):
    """
    Main entry point: compute pure intraday beta-adjusted order flow forecast.

    Orchestrates the simplified pipeline from raw data to final intraday forecast:
    1. Calculate forward returns from daily data (for regression targets)
    2. Calculate intraday bdC signals with beta adjustment
    3. Merge daily and intraday data
    4. Fit regression and generate forecast

    Args:
        daily_df (pd.DataFrame): Daily price/factor data with MultiIndex (date, sid)
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid)
        horizon (int): Forecast horizon in days (typically 0-5)

    Returns:
        pd.DataFrame: Intraday dataframe with 'bdma_i' forecast column

    Pipeline Flow:
        daily_df → calc_forward_returns() → forwards_df
        intra_df → calc_bd_intra() → intra_results_df
        [merge forwards + intra_results]
        → bd_fits() → full_df with 'bdma_i' forecast

    Notes:
        - Uses module-level 'middate' variable for in/out sample split
        - Forward returns required for regression fitting
        - No daily lag calculation (simpler than bd.py)
        - Final output suitable for dump_alpha() or backtesting
    """
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = forwards_df
    intra_results_df = calc_bd_intra(intra_df)
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)
    full_df = bd_fits(intra_results_df, horizon, "", middate)
    return full_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--freq",action="store",dest="freq",default=15)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = 0
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
        BAR_COLS = ['askHitDollars', 'midHitDollars', 'bidHitDollars', 'bopen']
        intra_df = load_bars(price_df[['ticker']], start, end, BAR_COLS, freq)
        daily_df = merge_barra_data(price_df, barra_df)
        daily_df = merge_intra_eod(daily_df, intra_df)
        intra_df = merge_intra_data(daily_df, intra_df)
        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')
    
    outsample_df = calc_bd_forecast(daily_df, intra_df, horizon)
    dump_alpha(outsample_df, 'bdma_i')




