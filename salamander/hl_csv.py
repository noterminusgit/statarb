"""High-Low Mean Reversion Strategy - CSV-Based Production Version

This module implements the production version of the HL mean reversion strategy
with CSV-based data loading and coefficient persistence. It improves upon hl.py
by adding rolling regression windows and structured data output.

Key Improvements over hl.py:
    - Loads data from CSV files in structured raw/ directories
    - Implements rolling 6-month regression windows for stability
    - Saves regression coefficients separately for analysis
    - Uses median-based regression for robustness
    - Supports configurable data and output directories

Strategy Overview:
    Generates mean reversion signals based on close price position within the
    daily high-low range. The hl0 ratio (close / sqrt(high*low)) captures this
    position, and industry-demeaned lagged values predict future returns.

Data Flow:
    1. Load 18 months of data (3 x 6-month periods)
    2. Fit coefficients on months T-18 to T-6
    3. Generate forecasts for months T-6 to T
    4. Save both forecasts and coefficients

File Structure:
    Input:  <dir>/data/raw/<YYYYMMDD>/{barra_df.csv, uni_df.csv, price_df.csv}
    Output: <dir>/data/all/all.<start>-<end>.h5
            <dir>/data/all_graphs/hl_daily_{in,ex}_<dates>.png

Usage:
    python hl_csv.py --start=20130101 --end=20130630 --dir=/path/to/data

Parameters:
    --start: Start date for forecast generation (YYYYMMDD)
    --end:   End date for forecast generation (YYYYMMDD)
    --dir:   Root data directory containing data/raw/ subdirectories

Example:
    python hl_csv.py --start=20130101 --end=20130630 --dir=./salamander_data

    This loads raw data from 20110630, 20120101, 20120630, 20130101, 20130630,
    fits coefficients on 20110630-20120630, and generates forecasts for 20130101-20130630.
"""

from regress import *
from util import *
from dateutil import parser as dateparser


def calc_hl_daily(full_df, horizon):
    """Calculate high-low ratio and industry-demeaned lags.

    Identical to hl.py version. Computes hl0 = close / sqrt(high*low), winsorizes
    it, demeans by industry, and generates lagged features.

    Args:
        full_df: DataFrame with multi-index (date, gvkey), columns: close, high, low, ind1
        horizon: Number of lag periods (typically 3)

    Returns:
        DataFrame with added columns:
            - hl0, hl0_B, hl0_B_ma: Base HL ratio and transformations
            - hl<N>_B_ma: Industry-demeaned lagged ratios for N=1 to horizon-1
            - hl3: 3-period lag of raw hl0
    """
    print("Caculating daily hl...")
    result_df = full_df.reset_index()
    # result_df = filter_expandable(result_df)
    result_df = result_df[['close', 'high', 'low', 'date', 'ind1', 'gvkey']]

    print("Calculating hl0...")
    result_df['hl0'] = result_df['close'] / np.sqrt(result_df['high'] * result_df['low'])
    result_df['hl0_B'] = winsorize(result_df['hl0'])

    result_df = result_df.dropna()
    demean = lambda x: (x - x.mean())
    indgroups = result_df[['hl0_B', 'date', 'ind1']].groupby(['date', 'ind1'], sort=False).transform(demean)
    result_df['hl0_B_ma'] = indgroups['hl0_B']
    result_df.set_index(['date', 'gvkey'], inplace=True)

    result_df['hl3'] = result_df['hl0'].unstack().shift(3).stack()  # new

    print("Calulating lags...")
    for lag in range(1, horizon):
        shift_df = result_df.unstack().shift(lag).stack()
        result_df['hl' + str(lag) + '_B_ma'] = shift_df['hl0_B_ma']
    result_df = pd.merge(full_df, result_df, how='left', on=['date', 'gvkey'], sort=False,
                         suffixes=['', '_dead'])  # new
    result_df = remove_dup_cols(result_df)
    return result_df

def hl_fits(daily_df, full_df, horizon, name, reg_st, reg_ed, out_dir):
    """Fit HL coefficients with windowed regression and generate forecasts.

    Enhanced version of hl_fits() that uses median-based regression on a
    specific date window for robustness. This prevents look-ahead bias by
    only using data from reg_st to reg_ed for coefficient estimation.

    Args:
        daily_df: DataFrame subset to fit on (sector-specific)
        full_df: Full DataFrame to populate with forecasts
        horizon: Maximum lag to regress (typically 3)
        name: Label for plot ("in" or "ex" for Energy sector)
        reg_st: Start date for regression window (string YYYYMMDD)
        reg_ed: End date for regression window (string YYYYMMDD)
        out_dir: Directory for saving diagnostic plots

    Returns:
        DataFrame with forecast column 'hl' and coefficient columns hl<N>_B_ma_coef

    Key Differences from hl.py:
        - Uses regress_alpha() with median=True instead of regress_alpha_daily()
        - Restricts regression to [reg_st, reg_ed] window
        - Saves plots to configurable out_dir
        - Uses .loc[] instead of .loc[] for modern pandas compatibility
    """
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])

    for lag in range(1, horizon + 1):
        fits_df = fits_df.append(regress_alpha(daily_df, 'hl0_B_ma', lag, median=True, start=reg_st, end=reg_ed),
                                 ignore_index=True)
    plot_fit(fits_df, out_dir + "/" + "hl_daily_" + name + "_" + df_dates(daily_df))

    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)
    coef0 = fits_df.loc['hl0_B_ma'].loc[horizon].loc['coef']

    if 'hl' not in full_df.columns:
        print("Creating forecast columns...")
        full_df['hl'] = 0
        full_df['hlC_B_ma_coef'] = np.nan
        for lag in range(1, horizon + 1):
            full_df['hl' + str(lag) + '_B_ma_coef'] = np.nan

    for lag in range(1, horizon + 1):
        full_df.loc[daily_df.index, 'hl' + str(lag) + '_B_ma_coef'] = coef0 - fits_df.loc['hl0_B_ma'].loc[lag].loc['coef']

    for lag in range(1, horizon):
        full_df.loc[daily_df.index, 'hl'] += full_df['hl' + str(lag) + '_B_ma'] * full_df[
            'hl' + str(lag) + '_B_ma_coef']
    return full_df


def calc_hl_forecast(daily_df, horizon, reg_st, reg_ed, output_dir):
    """Calculate HL forecast with sector-based fitting and return coefficients.

    Generates HL signals using sector-split regression (Energy vs. all others)
    with specified regression window. Returns both the full forecast DataFrame
    and a separate coefficient DataFrame for analysis.

    Args:
        daily_df: Full DataFrame with price, Barra, and forward returns
        horizon: Number of lags (typically 3)
        reg_st: Start date for regression window (YYYYMMDD string)
        reg_ed: End date for regression window (YYYYMMDD string)
        output_dir: Directory for saving diagnostic plots

    Returns:
        Tuple of (full_df, coef_df):
            - full_df: Complete DataFrame with 'hl' forecast column
            - coef_df: DataFrame containing only coefficient columns for persistence

    Methodology:
        Same sector-split approach as hl.py but with windowed regression.
        Energy sector (10) provides in-sample stability while non-Energy
        provides out-of-sample validation.
    """
    daily_df = calc_hl_daily(daily_df, horizon)

    sector = 10  # 'Energy'
    print("Running hl for sector code %d" % (sector))
    sector_df = daily_df[daily_df['sector'] == sector]
    full_df = hl_fits(sector_df, daily_df, horizon, "in", reg_st, reg_ed, output_dir)

    print("Running hl for sector code %d" % (sector))
    sector_df = daily_df[daily_df['sector'] != sector]
    full_df = hl_fits(sector_df, daily_df, horizon, "ex", reg_st, reg_ed, output_dir)

    coefs = []
    for lag in range(1, horizon + 1):
        coefs.append('hl' + str(lag) + '_B_ma_coef')
    coef_df = full_df[coefs]

    # dump_alpha(full_df, 'hl')

    return full_df, coef_df


def six_months_before(date_s):
    """Calculate the previous 6-month period boundary.

    Converts YYYYMMDD date strings backward by 6 months, respecting the
    semi-annual data structure (0101 and 0630 boundaries).

    Args:
        date_s: Date string in YYYYMMDD format (e.g., "20130630")

    Returns:
        Previous 6-month boundary as YYYYMMDD string

    Examples:
        "20130101" -> "20120630"
        "20130630" -> "20130101"
        "20120101" -> "20110630"

    Note:
        Assumes data is organized in 6-month periods ending Jan 1 and Jun 30.
    """
    if date_s[-4:] == '0101':
        return str(int(date_s[:4]) - 1) + '0630'
    else:
        return date_s[:4] + '0101'


def get_hl(start_s, end_s, dir):
    """Main entry point: load rolling CSV data, fit HL, save forecasts.

    Loads 18 months of historical data (3 x 6-month periods) from structured
    CSV directories, fits regression coefficients on the earlier 12 months,
    and generates forecasts for the requested 6-month period.

    Args:
        start_s: Start date for forecast output (YYYYMMDD format)
        end_s: End date for forecast output (YYYYMMDD format)
        dir: Root directory containing data/raw/<YYYYMMDD>/ subdirectories

    Returns:
        DataFrame containing regression coefficients for the forecast period

    Process:
        1. Load 3 periods of raw data (end_s, -6mo, -12mo)
        2. Merge price, Barra, universe, and forward returns
        3. Fit coefficients using data from T-18mo to T-6mo
        4. Generate forecasts for T-6mo to T (start_s to end_s)
        5. Save full_df to HDF5: <dir>/data/all/all.<start>-<end>.h5
        6. Save plots to <dir>/data/all_graphs/

    Data Requirements:
        <dir>/data/raw/<YYYYMMDD>/barra_df.csv   (Barra factors)
        <dir>/data/raw/<YYYYMMDD>/uni_df.csv     (universe with sedol)
        <dir>/data/raw/<YYYYMMDD>/price_df.csv   (OHLC price data)

    Output Files:
        <dir>/data/all/all.<start>-<end>.h5      (HDF5 with full_df)
        <dir>/data/all_graphs/hl_daily_*.png     (diagnostic plots)

    Example:
        get_hl("20130101", "20130630", "./salamander_data")

        Loads: 20130630, 20130101, 20120630 data
        Fits:  20110630 to 20130101
        Saves: Forecasts for 20130101 to 20130630

    Parameters:
        lookback: 30 (unused, legacy from hl.py)
        horizon: 3 days (changed from 5 in hl.py for production)
    """
    lookback = 30
    horizon = 3 # new
    d2 = end_s
    dfs = []
    for i in range(3):
        print("Loading raw data folder %s..." % d2)
        barra_df = pd.read_csv("%s/data/raw/%s/barra_df.csv" % (dir, d2), header=0, sep='|', dtype={'gvkey': str},
                               parse_dates=[0])
        uni_df = pd.read_csv("%s/data/raw/%s/uni_df.csv" % (dir, d2), header=0, sep='|', dtype={'gvkey': str},
                             parse_dates=[0])
        price_df = pd.read_csv("%s/data/raw/%s/price_df.csv" % (dir, d2), header=0, sep='|', dtype={'gvkey': str},
                               parse_dates=[0])
        price_df.set_index(['date', 'gvkey'], inplace=True)
        uni_df.set_index('gvkey', inplace=True)
        barra_df.set_index(['date', 'gvkey'], inplace=True)

        daily_df = merge_barra_data(price_df, barra_df)
        result_df = calc_forward_returns(daily_df, horizon)
        daily_df = daily_df.merge(result_df, on=['date', 'gvkey'])
        daily_df = daily_df.join(uni_df[['sedol']],on='gvkey', how='inner')
        daily_df.index.names=['date','gvkey']
        # intra_df = merge_intra_data(daily_df, daybar_df)
        dfs.append(daily_df)
        d2 = six_months_before(d2)
    reg_st = d2
    reg_ed = start_s
    daily_df = pd.concat(dfs).sort_index()
    graphs_dir = dir + "/data/all_graphs"
    full_df, coef_df = calc_hl_forecast(daily_df, horizon, reg_st, reg_ed, graphs_dir)
    full_df = full_df.truncate(before=dateparser.parse(start_s), after=dateparser.parse(end_s))
    output_dir = dir+ "/data/all"
    full_df.to_hdf('%s/all.%s-%s.h5' % (output_dir, start_s, end_s), 'full_df', mode='w')
    return coef_df
