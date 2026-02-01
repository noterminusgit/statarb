#!/usr/bin/env python
"""
Combined Daily+Intraday High-Low Mean Reversion Alpha Strategy

Generates alpha signals by combining BOTH daily lagged signals and intraday real-time
signals based on high-low mean reversion. This "both" approach leverages the complementary
information from historical daily patterns and current intraday dislocations.

Strategy Concept:
-----------------
This strategy extends the basic high-low mean reversion approach by integrating two
temporal dimensions:

1. **Daily Component**: Multi-lag daily signals (hl0, hl1, hl2, ...) that capture
   mean reversion patterns over multiple days. These signals use end-of-day prices
   relative to daily high-low ranges.

2. **Intraday Component**: Real-time intraday signal (hlC) that captures current
   dislocation from the daily high-low range as the day progresses.

The "both" refers to combining both temporal components with regression-fitted weights
that optimize predictive power for forward returns.

Signal Generation:
------------------
**Daily Signals (qhl0, qhl1, ..., qhlN)**:
    - qhl0 = close / sqrt(qhigh * qlow)
    - qhigh/qlow are quote-based daily high/low (more granular than trade prices)
    - Winsorized to remove outliers (qhl0_B)
    - Industry-demeaned within ind1 groups (qhl0_B_ma)
    - Lagged versions (qhl1_B_ma, qhl2_B_ma, ...) created via shifting
    - Each lag captures decaying mean reversion effect

**Intraday Signal (qhlC)**:
    - qhlC = iclose / sqrt(qhigh * qlow)
    - iclose is intraday close price at current timestamp
    - qhigh/qlow are the full day's quote high/low
    - Winsorized by timestamp (qhlC_B)
    - Industry-demeaned at each timestamp (qhlC_B_ma)
    - Provides real-time signal updates throughout the trading day

**Coefficient Fitting**:
    - In-sample regression fits each signal (qhl0_B_ma, qhl1_B_ma, ...) to forward returns
    - Horizon parameter determines forecast target (typically 3 days)
    - Coefficients decay with lag: coef[lag] = coef[horizon] - coef[lag]
    - Separate fits for in-sector (Energy) and ex-sector universes
    - This sector split allows Energy stocks to have different reversion dynamics

**Final Combined Signal (qhl_b)**:
    qhl_b = qhlC_B_ma * coef[0] +
            qhl1_B_ma * coef[1] +
            qhl2_B_ma * coef[2] + ...

    Where coef[i] are regression-fitted weights that optimize predictive power.

Expected Characteristics:
-------------------------
- **Holding Period**: 1-3 days (horizon parameter, default=3)
- **Signal Range**: Typically [-2, +2] after coefficient weighting
- **Turnover**: High due to daily signal updates and intraday component
- **Decay Pattern**: Coefficients decrease with lag as mean reversion weakens
- **Universe Split**: Separate parameters for Energy vs. non-Energy sectors
- **Market Neutrality**: Industry-demeaned signals provide sector neutrality

Key Parameters:
---------------
- horizon: Forecast horizon in days (default=3). Determines regression target
           and number of daily lags to incorporate.
- middate: Split date for in-sample fit vs. out-of-sample forecast. Regression
           coefficients fitted on data before middate, applied after middate.
- freq: Intraday bar frequency (default='15Min'). Trade-off between granularity
        and data quality.
- lookback: Days of universe history to load (default=30)

Data Requirements:
------------------
- Daily OHLC prices (close, qhigh, qlow)
- Intraday bar data at specified frequency (iclose, qhigh, qlow)
- Barra industry classifications (ind1)
- Sector classifications (sector_name)

Usage Example:
--------------
    python qhl_both.py --start=20130101 --end=20130630 --mid=20130401 --freq=15Min

    This will:
    1. Load daily and 15-minute intraday data from Jan-Jun 2013
    2. Fit coefficients on data before April 1, 2013
    3. Generate forecasts for data after April 1, 2013
    4. Output qhl_b signal to alpha file

Integration:
------------
In bsim.py/qsim.py, use as:
    --fcast=qhl_both:1:1  (100% weight)
    --fcast=qhl_both:3:0.5,bd:1:0.5  (combine with beta-adjusted order flow)

The signal integrates seamlessly with the optimization pipeline, providing a
high-turnover, market-neutral mean reversion component to the portfolio.

Comparison to Related Strategies:
----------------------------------
- qhl_intra.py: Uses ONLY intraday signals with time-of-day specific coefficients
- qhl_both.py (this): Combines daily lags AND intraday signals with uniform coefficients
- hl.py: Daily-only high-low strategy without intraday component
- qhl_multi.py: Multiple timeframe aggregation approach

The "both" approach provides richer information by incorporating both historical
daily patterns and current intraday dislocations, but requires more data and
computational resources.
"""

from regress import *
from loaddata import *
from util import *

def calc_qhl_daily(daily_df, horizon):
    """
    Calculate daily high-low mean reversion signals with multiple lags.

    This is the "daily component" of the combined strategy. It computes the base
    high-low signal at each day's close and creates lagged versions to capture
    multi-day mean reversion patterns.

    Args:
        daily_df (pd.DataFrame): Daily price data with MultiIndex (date, ticker).
                                 Must contain columns: close, qhigh, qlow, ind1
        horizon (int): Number of daily lags to create. Typically 3 for a 3-day
                       forecast horizon. Creates signals qhl1_B_ma through
                       qhl{horizon}_B_ma.

    Returns:
        pd.DataFrame: Input dataframe augmented with columns:
            - qhl0: Raw daily signal = close / sqrt(qhigh * qlow)
            - qhl0_B: Winsorized version of qhl0 (outliers capped)
            - qhl0_B_ma: Industry-demeaned version (market-neutral)
            - qhl1_B_ma, qhl2_B_ma, ..., qhl{horizon}_B_ma: Lagged signals

    Signal Construction:
        1. Filter to expandable universe (minimum liquidity requirements)
        2. Calculate base signal: qhl0 = close / sqrt(qhigh * qlow)
           - Uses geometric mean of quote-based high/low as reference
           - Values > 1.0 indicate close above midpoint (overextended)
           - Values < 1.0 indicate close below midpoint (oversold)
        3. Winsorize by date to remove outliers (typically at 5th/95th percentile)
        4. Industry-demean within ind1 groups to achieve sector neutrality
        5. Create lagged versions by shifting back in time
           - qhl1_B_ma is yesterday's signal, qhl2_B_ma is 2 days ago, etc.
           - Captures decaying predictive power over multiple days

    Notes:
        - Quote-based high/low (qhigh, qlow) are more granular than trade-based
        - Industry demeaning removes sector-wide movements, isolating stock effects
        - Lagged signals capture mean reversion decay over holding period
        - All signals are industry-neutral by construction
    """
    print "Caculating daily qhl..."
    result_df = filter_expandable(daily_df)

    print "Calculating qhl0..."
    result_df['qhl0'] = result_df['close'] / np.sqrt(result_df['qhigh'] * result_df['qlow'])
    result_df['qhl0_B'] = winsorize_by_date(result_df[ 'qhl0' ])

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['qhl0_B', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    result_df['qhl0_B_ma'] = indgroups['qhl0_B']

    print "Calulating lags..."
    for lag in range(1,horizon+1):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['qhl'+str(lag)+'_B_ma'] = shift_df['qhl0_B_ma']

    return result_df

def calc_qhl_intra(intra_df):
    """
    Calculate intraday high-low mean reversion signals at each timestamp.

    This is the "intraday component" of the combined strategy. It computes real-time
    high-low signals as the day progresses, measuring where the current intraday close
    is relative to the day's quote-based high-low range.

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (timestamp, ticker).
                                 Must contain columns: iclose, qhigh, qlow, ind1,
                                 giclose_ts (grouped timestamp for intraday close)

    Returns:
        pd.DataFrame: Input dataframe augmented with columns:
            - qhlC: Raw intraday signal = iclose / sqrt(qhigh * qlow)
            - qhlC_B: Winsorized version of qhlC across all timestamps
            - qhlC_B_ma: Industry-demeaned version (market-neutral)

    Signal Construction:
        1. Filter to expandable universe (minimum liquidity requirements)
        2. Calculate intraday signal: qhlC = iclose / sqrt(qhigh * qlow)
           - iclose is the intraday bar close price at current timestamp
           - qhigh/qlow are the full day's quote-based high/low (from daily data)
           - Signal updates in real-time as intraday prices evolve
           - Values > 1.0 indicate current price above daily midpoint
           - Values < 1.0 indicate current price below daily midpoint
        3. Winsorize by timestamp to remove outliers across all bars
        4. Industry-demean within (timestamp, ind1) groups
           - Removes sector-wide intraday movements
           - Isolates stock-specific intraday dislocations

    Notes:
        - Uses giclose_ts (grouped intraday close timestamp) for grouping
        - Industry demeaning done at each timestamp separately
        - Provides higher-frequency signal updates than daily-only approach
        - Complements daily signals by capturing current market conditions
        - Signal quality depends on intraday bar frequency (typically 15-30 min)
    """
    print "Calculating qhl intra..."
    result_df = filter_expandable(intra_df)

    print "Calulating qhlC..."
    result_df['qhlC'] = result_df['iclose'] / np.sqrt(result_df['qhigh'] * result_df['qlow'])
    result_df['qhlC_B'] = winsorize_by_ts(result_df[ 'qhlC' ])

    print "Calulating qhlC_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['qhlC_B', 'giclose_ts', 'ind1']].groupby(['giclose_ts', 'ind1'], sort=True).transform(demean)
    result_df['qhlC_B_ma'] = indgroups['qhlC_B']

    print "Calculated {} values".format(len(result_df['qhlC_B_ma'].dropna()))
    return result_df

def qhl_fits(daily_df, intra_df, horizon, name, middate=None):
    """
    Fit regression coefficients and generate combined daily+intraday forecasts.

    This is the core "combination" function that implements the "both" approach.
    It regresses daily signals against forward returns to determine optimal weights,
    then applies those weights to combine daily lagged signals with the intraday signal.

    Args:
        daily_df (pd.DataFrame): Daily data with qhl0_B_ma and forward returns.
                                 MultiIndex (date, ticker).
        intra_df (pd.DataFrame): Intraday data with qhlC_B_ma and lagged qhl signals.
                                 MultiIndex (timestamp, ticker).
        horizon (int): Forecast horizon in days. Determines regression target and
                       number of lags to combine. Typically 3.
        name (str): Name suffix for plot output (e.g., "in", "ex" for sector split).
        middate (datetime, optional): Split date for in-sample vs. out-of-sample.
                                      If None, uses all data for both fitting and forecast.

    Returns:
        pd.DataFrame: Intraday dataframe augmented with:
            - qhl_b: Combined forecast signal
            - qhlC_B_ma_coef: Coefficient for intraday component
            - qhl1_B_ma_coef, qhl2_B_ma_coef, ...: Coefficients for lagged daily components

    Regression Methodology:
        1. Split data at middate into in-sample (fit) and out-of-sample (forecast)
        2. For each lag k in [1, horizon]:
           - Regress qhl0_B_ma against k-day forward returns on in-sample data
           - Extract coefficient coef[k] and statistics (t-stat, stderr, nobs)
        3. Plot regression results showing coefficient decay with lag
        4. Compute differential coefficients for signal combination:
           - coef[0] = coef[horizon]  (weight for intraday/current signal)
           - coef[k] = coef[horizon] - coef[k]  (weight for k-day lagged signal)
           This differential weighting emphasizes recent signals while capturing
           multi-day decay patterns.

    Signal Combination:
        qhl_b = qhlC_B_ma * coef[0] +
                qhl1_B_ma * coef[1] +
                qhl2_B_ma * coef[2] + ...

        Where:
        - qhlC_B_ma is the current intraday signal
        - qhl{k}_B_ma are k-day lagged daily signals
        - coef[k] are regression-fitted weights

    Notes:
        - Coefficients are fitted on daily data but applied to intraday timestamps
        - This allows real-time forecast updates as intraday signal evolves
        - Separate fits for different sectors (via name parameter) allow different
          mean reversion dynamics
        - Plot output saved as "qhl_daily_{name}_{daterange}.png" for diagnostics
        - Differential coefficients (coef[k] = coef[horizon] - coef[k]) create
          a decay structure that emphasizes recent signals
    """
    insample_intra_df = intra_df
    insample_daily_df = daily_df
    outsample_intra_df = intra_df
    if middate is not None:
        insample_intra_df = intra_df[ intra_df['date'] <  middate ]
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_intra_df = intra_df[ intra_df['date'] >= middate ]

    outsample_intra_df['qhl_b'] = np.nan
    outsample_intra_df[ 'qhlC_B_ma_coef' ] = np.nan
    for lag in range(1, horizon+1):
        outsample_intra_df[ 'qhl' + str(lag) + '_B_ma_coef' ] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    for lag in range(1,horizon+1):
        fitresults_df = regress_alpha(insample_daily_df, 'qhl0_B_ma', lag, True, 'daily')
        fits_df = fits_df.append(fitresults_df, ignore_index=True)
    plot_fit(fits_df, "qhl_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)

    coef0 = fits_df.ix['qhl0_B_ma'].ix[horizon].ix['coef']
    outsample_intra_df['qhlC_B_ma_coef'] = coef0
    print "Coef0: {}".format(coef0)
    for lag in range(1,horizon):
        coef = coef0 - fits_df.ix['qhl0_B_ma'].ix[lag].ix['coef']
        print "Coef{}: {}".format(lag, coef)
        outsample_intra_df[ 'qhl'+str(lag)+'_B_ma_coef' ] = coef

    outsample_intra_df['qhl_b'] = outsample_intra_df['qhlC_B_ma'] * outsample_intra_df['qhlC_B_ma_coef']
    for lag in range(1,horizon):
        outsample_intra_df[ 'qhl_b'] += outsample_intra_df['qhl'+str(lag)+'_B_ma'] * outsample_intra_df['qhl'+str(lag)+'_B_ma_coef']

    return outsample_intra_df

def calc_qhl_forecast(daily_df, intra_df, horizon, middate):
    """
    Generate combined daily+intraday high-low forecasts with sector-specific fits.

    This is the top-level orchestration function that coordinates the entire "both"
    strategy pipeline. It processes both daily and intraday data, splits the universe
    by sector, fits separate models for each sector, and produces the final combined
    forecast signal (qhl_b).

    Args:
        daily_df (pd.DataFrame): Daily price data with columns: close, qhigh, qlow,
                                 ind1, sector_name. MultiIndex (date, ticker).
        intra_df (pd.DataFrame): Intraday bar data with columns: iclose, qhigh, qlow,
                                 ind1, sector_name. MultiIndex (timestamp, ticker).
        horizon (int): Forecast horizon in days. Typically 3 for 3-day ahead forecast.
        middate (datetime): Split date for in-sample regression vs. out-of-sample forecast.
                           Coefficients fitted on data before middate, applied after.

    Returns:
        pd.DataFrame: Intraday dataframe with qhl_b forecast signal for all stocks.
                      Combines in-sector and ex-sector results.

    Pipeline Steps:
        1. **Daily Signal Generation**:
           - Calculate qhl0_B_ma (base daily signal)
           - Create lagged versions (qhl1_B_ma, qhl2_B_ma, ...)
           - Calculate forward returns for regression targets

        2. **Intraday Signal Generation**:
           - Calculate qhlC_B_ma (intraday signal at each timestamp)
           - Merge with daily data to get lagged signals at each intraday point

        3. **Sector Split**:
           - Split universe into Energy sector vs. non-Energy sectors
           - Rationale: Energy stocks may have different mean reversion dynamics
             due to commodity price exposure and sector-specific flows
           - Allows separate coefficient fits for each group

        4. **Coefficient Fitting**:
           - For Energy stocks: Fit coefficients on in-sample data, apply to out-sample
           - For non-Energy stocks: Fit coefficients on in-sample data, apply to out-sample
           - Regression plots saved as "qhl_daily_in_*.png" and "qhl_daily_ex_*.png"

        5. **Signal Combination**:
           - Combine daily lagged signals and intraday signal using fitted coefficients
           - Generate qhl_b forecast for each timestamp in out-of-sample period

        6. **Universe Reconstruction**:
           - Concatenate Energy and non-Energy results back into single dataframe
           - verify_integrity=True ensures no overlapping stocks between sectors

    Sector-Specific Modeling:
        Energy stocks (in-sector):
            - Often have different volatility patterns due to oil/gas price shocks
            - May exhibit stronger/weaker mean reversion than broader market
            - Separate coefficients allow capturing these dynamics

        Non-Energy stocks (ex-sector):
            - Larger universe providing more robust coefficient estimates
            - More representative of general equity mean reversion

    Notes:
        - Sector split hardcoded to 'Energy' but could be parameterized
        - Both sector models use same signal construction, only coefficients differ
        - verify_integrity=True in concat ensures no ticker appears in both sectors
        - Forward returns calculated to horizon for regression target
        - All signals are industry-neutral (demeaned within ind1 groups)
    """
    daily_results_df = calc_qhl_daily(daily_df, horizon)
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)
    intra_results_df = calc_qhl_intra(intra_df)
    intra_results_df = merge_intra_data(daily_results_df, intra_results_df)

    sector_name = 'Energy'
    print "Running qhl for sector {}".format(sector_name)
    sector_df = daily_results_df[ daily_results_df['sector_name'] == sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] == sector_name ]
    result1_df = qhl_fits(sector_df, sector_intra_results_df, horizon, "in", middate)

    print "Running qhl for not sector {}".format(sector_name)
    sector_df = daily_results_df[ daily_results_df['sector_name'] != sector_name ]
    sector_intra_results_df = intra_results_df[ intra_results_df['sector_name'] != sector_name ]
    result2_df = qhl_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)

    result_df = pd.concat([result1_df, result2_df], verify_integrity=True)
    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--freq",action="store",dest="freq",default='15Min')
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = 3
    pname = "./qhl_b" + start + "." + end
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
        print "Did not load cached data..."

    if not loaded:
        uni_df = get_uni(start, end, lookback)
        BARRA_COLS = ['ind1']
        barra_df = load_barra(uni_df, start, end, BARRA_COLS)
        PRICE_COLS = ['close']
        price_df = load_prices(uni_df, start, end, PRICE_COLS)
        DBAR_COLS = ['close', 'qhigh', 'qlow']
        intra_df = load_daybars(price_df[['ticker']], start, end, DBAR_COLS, freq)

        daily_df = merge_barra_data(price_df, barra_df)
        daily_df = merge_intra_eod(daily_df, intra_df)
        intra_df = merge_intra_data(daily_df, intra_df)

        daily_df.to_hdf(pname+"_daily.h5", 'table', complib='zlib')
        intra_df.to_hdf(pname+"_intra.h5", 'table', complib='zlib')

    result_df = calc_qhl_forecast(daily_df, intra_df, horizon, middate)
    dump_alpha(result_df, 'qhl_b')



