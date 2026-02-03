"""High-Low Mean Reversion Strategy

This module implements a high-low mean reversion alpha strategy for the salamander
backtesting system. The strategy generates signals based on the ratio of close price
to the geometric mean of high and low prices, industry-demeaned and regressed against
forward returns.

Key Features:
    - Calculates hl0 ratio: close / sqrt(high * low)
    - Industry-demeaned signals (by ind1)
    - Winsorized to reduce outlier impact
    - Lagged coefficient regression to predict mean reversion
    - Sector-based in-sample and out-of-sample fitting (sector 10 = Energy)

Strategy Logic:
    The hl0 ratio captures price position within the daily high-low range. Values
    above 1.0 indicate closes near the high (potential reversal down), below 1.0
    indicates closes near the low (potential reversal up). Industry demeaning removes
    sector-wide effects.

Data Requirements:
    - Price data with high, low, close fields
    - Barra sector and ind1 classifications
    - Minimum lookback of 30 days for regression stability

Output:
    - HDF5 file: all.<start>-<end>.h5 containing full_df with hl forecast column
    - Forecast column 'hl' contains the predicted alpha signal
    - Coefficient columns hl1_B_ma_coef through hl<horizon>_B_ma_coef

Usage:
    python hl.py --start=20130101 --end=20130630

Note:
    This is an early prototype version. For production use, see hl_csv.py which
    adds CSV-based coefficient persistence and improved regression windowing.
"""

from regress import *
from util import *
from s_loaddata import *
from dateutil import parser as dateparser


def calc_hl_daily(full_df, horizon):
    """Calculate high-low ratio and industry-demeaned lags.

    Computes the hl0 ratio (close / sqrt(high*low)), winsorizes it, and creates
    industry-demeaned versions. Then generates lagged features for regression.

    Args:
        full_df: DataFrame with multi-index (date, gvkey) containing price data
                 Required columns: close, high, low, ind1, gvkey, date
        horizon: Number of lag periods to generate (typically 3-5)

    Returns:
        DataFrame with added columns:
            - hl0: Raw high-low ratio
            - hl0_B: Winsorized hl0
            - hl0_B_ma: Industry-demeaned hl0_B
            - hl<N>_B_ma: Lagged industry-demeaned ratios for N=1 to horizon-1
            - hl3: 3-period lagged hl0 (legacy field)

    Process:
        1. Calculate hl0 = close / sqrt(high * low)
        2. Winsorize to reduce outliers
        3. Demean by industry (ind1) within each date
        4. Generate lagged versions by shifting time series
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


def hl_fits(daily_df, full_df, horizon, name):
    """Fit HL coefficients using lagged regression and generate forecasts.

    Regresses the hl0_B_ma signal at various lags against forward returns to
    estimate the decay of predictive power over time. Uses coefficient differences
    to construct a composite forecast.

    Args:
        daily_df: DataFrame subset to fit on (e.g., sector-specific or date range)
        full_df: Full DataFrame to populate with forecasts
        horizon: Maximum lag to regress (typically 5)
        name: Label for diagnostic plot output (e.g., "in" or "ex" for sector)

    Returns:
        DataFrame with added columns:
            - hl: Composite forecast signal (sum of lagged signals * coefficients)
            - hl<N>_B_ma_coef: Coefficient for lag N
            - hlC_B_ma_coef: Placeholder for close-based coefficient

    Regression Logic:
        For each lag L from 1 to horizon:
            Regress hl0_B_ma(t-L) against forward_returns(t)
            Extract coefficient c(L)

        For forecasting at time t:
            coef(L) = c(horizon) - c(L)  (coefficient decay from horizon)
            forecast += hl(t-L) * coef(L)  for L=0 to horizon-1

    Diagnostic Output:
        Saves plot of coefficients vs lag to visualize decay pattern.
    """
    fits_df = pd.DataFrame(columns=['horizon', 'coef', 'indep', 'tstat', 'nobs', 'stderr'])

    for lag in range(1, horizon + 1):
        fits_df = fits_df.append(regress_alpha_daily(daily_df, 'hl0_B_ma', lag), ignore_index=True)
    plot_fit(fits_df, "hl_daily_" + name + "_" + df_dates(daily_df))

    fits_df.set_index(keys=['indep', 'horizon'], inplace=True)
    coef0 = fits_df.ix['hl0_B_ma'].ix[horizon].ix['coef']

    if 'hl' not in full_df.columns:
        print("Creating forecast columns...")
        full_df['hl'] = 0
        full_df['hlC_B_ma_coef'] = np.nan
        for lag in range(0, horizon + 1):
            full_df['hl' + str(lag) + '_B_ma_coef'] = np.nan

    for lag in range(1, horizon + 1):
        full_df.ix[daily_df.index, 'hl' + str(lag) + '_B_ma_coef'] = coef0 - fits_df.ix['hl0_B_ma'].ix[lag].ix['coef']

    for lag in range(0, horizon):
        full_df.ix[daily_df.index, 'hl'] += full_df['hl' + str(lag) + '_B_ma'] * full_df['hl' + str(lag) + '_B_ma_coef']

    return full_df


def calc_hl_forecast(daily_df, horizon):
    """Calculate HL forecast using sector-based in/out-of-sample fitting.

    Implements a pseudo-out-of-sample approach by fitting the Energy sector (sector 10)
    separately from all other sectors. This provides robustness by ensuring coefficients
    are not overfit to a single sector's dynamics.

    Args:
        daily_df: Full DataFrame with price and Barra data
        horizon: Number of lags for regression (typically 5)

    Returns:
        DataFrame with 'hl' forecast column populated for all dates

    Methodology:
        1. Calculate hl0_B_ma and lagged features
        2. Fit coefficients using Energy sector only (in-sample)
        3. Fit coefficients using all non-Energy sectors (out-of-sample for Energy)
        4. Combine forecasts (later fits overwrite earlier ones)

    Note:
        The sector-based splitting helps prevent overfitting to sector-specific
        patterns and provides more robust coefficients.
    """
    daily_df = calc_hl_daily(daily_df, horizon)

    sector = 10  # 'Energy'
    print("Running hl for sector code %d" % (sector))
    sector_df = daily_df[daily_df['sector'] == sector]
    full_df = hl_fits(sector_df, daily_df, horizon, "in")

    print("Running hl for sector code %d" % (sector))
    sector_df = daily_df[daily_df['sector'] != sector]
    full_df = hl_fits(sector_df, daily_df, horizon, "ex")

    # dump_alpha(full_df, 'hl')

    return full_df


def get_hl(start_s, end_s):
    """Main entry point: load data, calculate HL forecast, save to HDF5.

    Orchestrates the full HL signal generation pipeline from data loading through
    forecast generation and output.

    Args:
        start_s: Start date string in YYYYMMDD format (e.g., "20130101")
        end_s: End date string in YYYYMMDD format (e.g., "20130630")

    Process:
        1. Load universe with 30-day lookback
        2. Load Barra factor data and price data
        3. Merge datasets and calculate forward returns
        4. Generate HL forecast using calc_hl_forecast()
        5. Save to HDF5 file: all.<start>-<end>.h5

    Output:
        HDF5 file containing full_df DataFrame with all price, Barra, and HL columns.
        Key column: 'hl' contains the alpha forecast signal.

    Parameters:
        lookback: 30 days (for data loading stability)
        horizon: 5 days (forward return and lag window)
    """
    lookback = 30
    horizon = 5
    pd.set_option('display.max_columns', 100)
    start = dateparser.parse(start_s)
    end = dateparser.parse(end_s)
    uni_df = get_uni(start, end, lookback)
    barra_df = load_barra(uni_df, start, end)
    price_df = load_price(uni_df, start, end)
    daily_df = merge_barra_data(price_df, barra_df)
    result_df = calc_forward_returns(daily_df, horizon)
    daily_df = daily_df.merge(result_df, on=['date', 'gvkey'])
    # intra_df = merge_intra_data(daily_df, daybar_df)
    full_df = calc_hl_forecast(daily_df, horizon)
    full_df.to_hdf('all.%s-%s.h5' % (start_s, end_s), 'full_df', mode='w')
    print(full_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="the starting date, formatted as 'YYYYMMdd'", type=str)
    parser.add_argument("--end", help="the end date, formatted as 'YYYYMMdd'", type=str)
    parser.add_argument("--data_dir", help="the directory where raw data folder is stored", type=str, default='.')
    parser.add_argument("--out_dir", help="the directory where new data will be generated", type=str, default='.')
    args = parser.parse_args()
    get_hl(args.start, args.end, args.data_dir, args.out_dir)
