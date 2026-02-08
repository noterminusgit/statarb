#!/usr/bin/env python
"""
REV - Mean Reversion Alpha Strategy

Generates alpha forecasts based on short-term mean reversion in stock returns.
The strategy bets that stocks with extreme recent returns will revert toward
their industry mean.

Strategy Logic
--------------
1. Compute rolling sum of returns over lookback window (e.g., 21 days)
2. Demean returns within each industry group
3. Fit linear regression: forward_returns ~ lagged_industry_demeaned_returns
4. Generate forecasts by multiplying coefficients with current signals

The core hypothesis is that industry-relative outperformers will underperform
and vice versa over the forecast horizon.

Reversal Signal Construction
-----------------------------
rev0: Raw rolling return sum
    rev0 = sum(log_ret, lag)
    Example: lag=21 means sum of returns over past 21 days

rev0_ma: Industry-demeaned rolling returns
    rev0_ma = rev0 - industry_mean(rev0)
    Removes industry-level momentum to isolate stock-specific effects

rev1_ma: Lagged signal for prediction
    rev1_ma = shift(rev0_ma, 1)
    Yesterday's industry-relative returns predict tomorrow's reversals

Regression Framework
--------------------
In-sample: Fit regression on data before middate
    forward_return[t+horizon] = beta * rev1_ma[t] + error

Out-sample: Generate forecasts after middate
    rev_forecast[t] = beta * rev1_ma[t]

The beta coefficient captures the strength of mean reversion:
    - Negative beta: Mean reversion (expected)
    - Positive beta: Momentum (unexpected for this strategy)

Functions
---------
calc_rev_daily(daily_df, horizon, lag):
    Compute reversal signals on daily data.

    Parameters:
        daily_df: Price data with log_ret, industry (ind1), date
        horizon: Forward return horizon (days)
        lag: Rolling window for reversal signal (days)

    Returns:
        DataFrame with rev0, rev0_ma, rev1_ma columns

    Steps:
        1. Filter to expandable universe (tradable stocks)
        2. Calculate rev0 = rolling_sum(log_ret, lag)
        3. Demean by [date, industry] groups
        4. Shift to create lagged signal (rev1_ma)

rev_fits(daily_df, horizon, name, middate):
    Fit regression and generate out-of-sample forecasts.

    Parameters:
        daily_df: Data with reversal signals and forward returns
        horizon: Prediction horizon (days)
        name: Strategy name for output files
        middate: Split between in-sample and out-sample

    Returns:
        Out-of-sample data with rev_{name} forecast column

    Steps:
        1. Split data at middate
        2. Run regression on in-sample data
        3. Plot regression diagnostics
        4. Apply coefficients to out-sample data
        5. Generate forecasts

calc_rev_forecast(daily_df, horizon, middate, lag):
    Main function combining signal calculation and regression.

    Returns:
        DataFrame with rev_{lag} forecast column

Command-Line Arguments
----------------------
--start : str
    Start date (YYYYMMDD format)
--end : str
    End date (YYYYMMDD format)
--mid : str
    Midpoint date for in-sample/out-sample split
--lag : int
    Rolling window for reversal signal (default: 21)

The horizon is set equal to lag (line 65).

Usage Examples
--------------
21-day reversal with 2012 split:
    python rev.py --start=20110101 --end=20130630 --mid=20120101 --lag=21

5-day reversal:
    python rev.py --start=20110101 --end=20130630 --mid=20120101 --lag=5

Output Files
------------
1. {pname}_daily.h5: Cached price and Barra data
   - Used for faster re-runs without reloading
   - Contains: close prices, industry assignments

2. Regression plots: rev_daily_{lag}_{date_range}.png
   - Coefficient significance
   - T-statistics
   - Number of observations

3. Alpha forecast: Written via dump_daily_alpha()
   - Forecast column name: rev_{lag}
   - Can be loaded by simulation engines

Data Requirements
-----------------
Requires:
    - Price data with close prices
    - Barra industry classifications (ind1)
    - Date range covering --start to --end
    - Additional lookback for rolling calculations

Caching
-------
First run: Loads and caches data to {pname}_daily.h5
Subsequent runs: Reads from cache (much faster)
To force reload: Delete .h5 file

Industry Demeaning
------------------
Critical to strategy - removes industry momentum effects:
    - Tech sector rallies: long relative losers, short relative winners
    - Energy sector sell-offs: opposite within energy
    - Isolates stock-specific reversals from sector trends

This is implemented via grouped transform (lines 14-16):
    demean = lambda x: (x - x.mean())
    result_df['rev0_ma'] = result_df.groupby(['gdate', 'ind1']).transform(demean)['rev0']

Typical Parameter Values
-------------------------
Short-term reversal:
    - lag=5 to 10 days
    - Captures weekly mean reversion
    - Higher turnover, needs low transaction costs

Medium-term reversal:
    - lag=21 to 30 days (default)
    - Monthly rebalancing frequency
    - More stable, lower turnover

Long-term reversal:
    - lag=60 to 252 days
    - May capture longer-term overreaction
    - Risk of becoming momentum at very long horizons

Relationship to Other Alphas
-----------------------------
This strategy complements:
    - hl.py: High-low mean reversion (intraday)
    - pca.py: Market-neutral mean reversion
    - bd.py: Order flow momentum (opposite signal)

Performance Characteristics
----------------------------
Expected:
    - Negative in trending markets (2013 rally)
    - Positive in range-bound markets (2011-2012)
    - Works best in high-dispersion environments
    - Capacity limited by turnover

Risk Factors:
    - Momentum crashes (reversals accelerate)
    - Low dispersion regimes (nothing to revert)
    - Transaction costs erode edge quickly

Integration with Simulators
----------------------------
After running rev.py, use the forecast in simulations:

BSIM (daily):
    python bsim.py --start=20130101 --end=20130630 --fcast=rev_21:1:1

Multi-alpha combination:
    python bsim.py --start=20130101 --end=20130630 \
        --fcast=hl:1:0.6,rev_21:0.8:0.4

Notes
-----
- Horizon is set equal to lag (not independently configurable)
- Industry classification uses ind1 from Barra
- Requires filter_expandable() for universe
- Regression uses daily frequency ('daily' parameter)
- Could be extended to intraday bars (similar to qhl_*.py)
- Consider adding volume or volatility filters
- May benefit from regime-dependent coefficients

Theoretical Foundation
----------------------
Mean reversion is a well-documented market anomaly:
    - Overreaction hypothesis (DeBondt & Thaler)
    - Liquidity provision to momentum traders
    - Statistical arbitrage around fair value
    - Market microstructure effects

The industry demeaning is critical because:
    - Industry momentum is persistent (Jegadeesh & Titman)
    - Stock-level reversals exist within industries
    - Without demeaning, momentum would dominate signal
"""

from __future__ import division, print_function

from regress import *
from loaddata import *
from util import *

def calc_rev_daily(daily_df, horizon, lag):
    print("Caculating daily rev...")
    result_df = filter_expandable(daily_df)

    print("Calculating rev0...")
    result_df['rev0'] = pd.rolling_sum(result_df['log_ret'], lag)

    demean = lambda x: (x - x.mean())
    indgroups = result_df[['rev0', 'gdate', 'ind1']].groupby(['gdate', 'ind1'], sort=True).transform(demean)
    result_df['rev0_ma'] = indgroups['rev0']
    shift_df = result_df.unstack().shift(1).stack()
    result_df['rev1_ma'] = shift_df['rev0_ma']

    return result_df

def rev_fits(daily_df, horizon, name, middate=None):
    insample_daily_df = daily_df
    if middate is not None:
        insample_daily_df = daily_df[ daily_df.index.get_level_values('date') < middate ]
        outsample_daily_df = daily_df[ daily_df.index.get_level_values('date') >= middate ]

    outsample_daily_df['rev'] = np.nan

    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fitresults_df = regress_alpha(insample_daily_df, 'rev1_ma', horizon, True, 'daily') 
    fits_df = fits_df.append(fitresults_df, ignore_index=True) 
    plot_fit(fits_df, "rev_daily_"+name+"_" + df_dates(insample_daily_df))
    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    

    coef0 = fits_df.ix['rev1_ma'].ix[horizon].ix['coef']
    print("Coef{}: {}".format(0, coef0))
    outsample_daily_df[ 'rev1_ma_coef' ] = coef0

    outsample_daily_df[ 'rev_' + name ] = outsample_daily_df['rev1_ma'] * outsample_daily_df['rev1_ma_coef']
    
    return outsample_daily_df

def calc_rev_forecast(daily_df, horizon, middate, lag):
    daily_results_df = calc_rev_daily(daily_df, horizon, lag) 
    forwards_df = calc_forward_returns(daily_df, horizon)
    daily_results_df = pd.concat( [daily_results_df, forwards_df], axis=1)

    result_df = rev_fits(daily_results_df, horizon, str(lag), middate)

    return result_df

if __name__=="__main__":            
    parser = argparse.ArgumentParser(description='G')
    parser.add_argument("--start",action="store",dest="start",default=None)
    parser.add_argument("--end",action="store",dest="end",default=None)
    parser.add_argument("--mid",action="store",dest="mid",default=None)
    parser.add_argument("--lag",action="store",dest="lag",default=21)
#    parser.add_argument("--horizon",action="store",dest="horizon",default=20)
    args = parser.parse_args()
    
    start = args.start
    end = args.end
    lookback = 30
    horizon = int(args.lag)
    pname = "./rev" + start + "." + end
    start = dateparser.parse(start)
    end = dateparser.parse(end)
    middate = dateparser.parse(args.mid)
    lag = int(args.lag)

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

    result_df = calc_rev_forecast(daily_df, horizon, middate, lag)
    dump_daily_alpha(result_df, 'rev_' + str(lag))



