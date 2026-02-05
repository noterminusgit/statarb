#!/usr/bin/env python
"""Simplified Beta-Adjusted Order Flow Alpha Strategy (BD1)

Simplified variant of bd.py that uses order flow delta (change in hit dollars)
rather than absolute levels. Focuses purely on intraday signals without daily
lag components, making it a leaner implementation for testing order flow momentum.

Methodology Differences from bd.py:
-----------------------------------
1. Signal Calculation:
   - bd.py: Uses absolute order flow ratios (askHit - bidHit) / (total volume)
   - bd1.py: Uses DIFFERENCED order flow - .diff() on hit dollars
   - This captures momentum/acceleration in order flow rather than levels

2. Scope:
   - bd.py: Combines daily lags (bd1, bd2, ...) + intraday (bdC)
   - bd1.py: Pure intraday signal only - no daily lag components
   - Simpler regression with just current bar signal

3. Normalization:
   - bd.py: Normalizes by sqrt(spread_bps) for liquidity adjustment
   - bd1.py: No spread normalization - just winsorization

Signal Formula:
--------------
bd1 = (askHitDollars.diff() - bidHitDollars.diff()) /
      (askHitDollars.diff() + midHitDollars.diff() + bidHitDollars.diff())

Where .diff() computes the change from previous bar, capturing order flow
acceleration rather than absolute levels.

Industry Demeaning:
------------------
- Groups by (date, industry)
- Subtracts industry mean to ensure sector neutrality
- Creates bd1_B_ma signal (market-adjusted, industry-demeaned)

Regression Strategy:
-------------------
- Fits bd1_B_ma and bd1_B separately against forward returns
- Uses regress_alpha_intra() for intraday regression
- Single horizon (typically 1 period forward)
- Produces two forecasts:
  * bd1ma: Industry-demeaned forecast
  * bd1: Raw winsorized forecast

Use Case:
--------
This simplified variant is useful for:
- Testing order flow momentum effects (changes vs levels)
- Rapid prototyping without daily lag complexity
- Isolating intraday order flow patterns
- Lower computational overhead than full bd.py

Data Requirements:
-----------------
- Intraday bar data: askHitDollars, bidHitDollars, midHitDollars
- Barra factors: ind1 (industry classification)
- Universe: Expandable stocks (liquid, tradeable)

CLI Usage:
---------
This module runs as a standalone script with hardcoded dates:
    python bd1.py

Modify start/end dates in __main__ block for different test periods.

Output:
------
- Regression plots: bd1ma_intra_*.png, bd1_intra_*.png
- Alpha forecasts: 'bd1' and 'bd1ma' columns via dump_alpha()

Related Modules:
---------------
- bd.py: Full beta-adjusted order flow with daily+intraday
- bd_intra.py: Pure intraday with beta adjustment (no order flow differencing)
- alphacalc.py: Helper functions for alpha calculations
- regress.py: Regression fitting framework

Notes:
-----
- This appears to be a quick experimental variant for testing differenced signals
- May produce different signal characteristics than level-based order flow
"""

from alphacalc import *

from dateutil import parser as dateparser

def calc_bd_intra(intra_df):
    """
    Calculate simplified intraday beta-adjusted order flow using differenced signals.

    Computes the bd1 signal using CHANGES in order flow (first differences)
    rather than absolute levels. This captures acceleration/momentum in
    aggressive order flow rather than the level of imbalance.

    The key innovation is using .diff() on hit dollars to detect shifts in
    order flow patterns rather than sustained imbalances.

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid):
            - askHitDollars: Dollar volume of aggressive buys (ask hits)
            - bidHitDollars: Dollar volume of aggressive sells (bid hits)
            - midHitDollars: Dollar volume at midpoint
            - ind1: Industry classification (for demeaning)
            - expandable: Boolean filter for tradeable stocks
            - iclose: Bar close price
            - date: Trading date

    Returns:
        pd.DataFrame: Input dataframe augmented with:
            - bd1: Differenced order flow ratio signal
            - bd1_B: Winsorized bd1 signal
            - bd1_B_ma: Industry-demeaned signal

    Signal Formula:
        bd1 = (askHitDollars.diff() - bidHitDollars.diff()) /
              (askHitDollars.diff() + midHitDollars.diff() + bidHitDollars.diff())

    Process:
        1. Reset index and filter to expandable universe
        2. Select required columns only
        3. Drop any rows with missing data
        4. Calculate differenced order flow ratio (bd1)
        5. Winsorize to control outliers
        6. Industry demean within (date, industry) groups
        7. Merge back into original intra_df

    Notes:
        - No beta adjustment despite "beta-adjusted" in module name
        - No spread normalization unlike bd.py
        - First bar of each day will have NaN for .diff() calculation
        - Industry demeaning ensures sector neutrality
        - Date column deleted before merge to avoid timestamp conflicts
    """
    print "Calculating bd1 intra..."

    result_df = intra_df.reset_index()
    result_df = filter_expandable(result_df)
    result_df = result_df[ [ 'iclose', 'iclose_ts', 'bidHitDollars', 'midHitDollars', 'askHitDollars', 'date', 'ind1', 'sid' ] ]
    result_df = result_df.dropna(how='any')

    print "Calulating bd1..."
    result_df['bd1'] = (result_df['askHitDollars'].diff() - result_df['bidHitDollars'].diff()) / (result_df['askHitDollars'].diff() + result_df['midHitDollars'].diff() + result_df['bidHitDollars'].diff())
    result_df['bd1_B'] = winsorize(result_df['bd1'])

    print "Calulating bd1_ma..."
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['bd1_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['bd1_B_ma'] = indgroups['bd1_B']

    #important for keeping NaTs out of the following merge
    del result_df['date']

    print "Merging..."
    result_df.set_index(keys=['iclose_ts', 'sid'], inplace=True)
    result_df = pd.merge(intra_df, result_df, how='left', left_index=True, right_index=True, sort=True, suffixes=['_dead', ''])
    result_df = remove_dup_cols(result_df)

    return result_df

def bd_fits(daily_df, intra_df, full_df, name):
    """
    Fit regression coefficients for simplified order flow signals.

    Fits two separate regressions for the bd1 strategy:
    1. bd1_B_ma: Industry-demeaned (market-adjusted) signal
    2. bd1_B: Raw winsorized signal without industry adjustment

    This allows comparison of sector-neutral vs raw signal efficacy.

    Args:
        daily_df (pd.DataFrame): Daily data (used for plot dating only)
        intra_df (pd.DataFrame): Intraday data with bd1 signals
        full_df (pd.DataFrame): Full merged dataset for forecast storage
        name (str): Name prefix for output plots (e.g., "bd1")

    Returns:
        pd.DataFrame: full_df augmented with:
            - bd1_B_ma_coef: Coefficient for industry-demeaned signal
            - bd1_B_coef: Coefficient for raw signal
            - bd1ma: Forecast from industry-demeaned signal
            - bd1: Forecast from raw signal

    Regression Process:
        - Uses regress_alpha_intra() for intraday regression
        - Fits against forward returns (horizon typically 1)
        - Produces separate plots for each signal variant
        - Applies coefficients uniformly across all timestamps

    Output Files:
        - {name}ma_intra_{dates}.png: Industry-demeaned regression plot
        - {name}_intra_{dates}.png: Raw signal regression plot

    Notes:
        - Unlike bd.py, no time-of-day varying coefficients
        - Simple uniform coefficient application
        - Both forecasts generated for comparison
        - 'horizon' variable used but not defined in function scope
    """
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])
    fits_df = fits_df.append(regress_alpha_intra(intra_df, 'bd1_B_ma', 1), ignore_index=True)
    fits_df = fits_df.append(regress_alpha_intra(intra_df, 'bd1_B', 1), ignore_index=True)

    plot_fit(fits_df[ fits_df['indep'] == 'bd1_B_ma' ], name + "ma_intra_" + df_dates(daily_df))
    plot_fit(fits_df[ fits_df['indep'] == 'bd1_B' ], name + "_intra_" + df_dates(daily_df))

    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)    
    coef0 = fits_df.ix['bd1_B_ma'].ix[horizon].ix['coef']
    full_df[ 'bd1_B_ma_coef' ] = coef0
    full_df['bd1ma'] = full_df['bd1_B_ma'] * full_df['bd1_B_ma_coef']

    coef0 = fits_df.ix['bd1_B'].ix[horizon].ix['coef']
    full_df[ 'bd1_B_coef' ] = coef0
    full_df['bd1'] = full_df['bd1_B'] * full_df['bd1_B_coef']

    return full_df

def calc_bd_forecast(intra_df):
    """
    Main entry point: compute simplified bd1 forecast end-to-end.

    Orchestrates the simplified pipeline from intraday bar data to final forecasts:
    1. Calculate bd1 differenced order flow signals
    2. Merge with daily data
    3. Fit regressions and generate forecasts

    Args:
        intra_df (pd.DataFrame): Intraday bar data with MultiIndex (iclose_ts, sid)

    Returns:
        pd.DataFrame: Full dataset with 'bd1' and 'bd1ma' forecast columns

    Pipeline Flow:
        intra_df → calc_bd_intra() → intra_results
        [merge with daily_df] → full_df
        → bd_fits() → forecasts

    Notes:
        - Requires module-level 'daily_df' variable (side effect dependency)
        - Simpler pipeline than bd.py (no daily lag calculation)
        - No in-sample/out-of-sample split
        - Produces two forecast variants for comparison
    """
    intra_df = calc_bd_intra(intra_df)
    full_df = merge_intra_data(daily_df, intra_df)

    full_df = bd_fits(daily_df, intra_df, full_df, "bd1")

    return full_df

if __name__=="__main__":            
    start = "20120101"
    end = "20120115"
    lookback = 30

    start = dateparser.parse(start)
    end = dateparser.parse(end)

    uni_df = get_uni(start, end, lookback)
    barra_df = load_barra(uni_df, start, end)
    price_df = load_prices(uni_df, start, end)
    daily_df = merge_barra_data(price_df, barra_df)
    ibar_df = load_bars(uni_df, start, end)
    intra_df = merge_intra_data(daily_df, ibar_df)

    full_df = calc_bd_forecast(intra_df)

    dump_alpha(full_df, 'bd1')
    dump_alpha(full_df, 'bdma1')
