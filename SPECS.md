# SPECS.md — Public API Specification

> Auto-generated function-level specs for test preparation.
> Organized by path, then alphabetical filename.

---

## `alphacalc.py`

No public functions (module re-exports from `calc.py` and `util.py`).

---

## `analyst.py`

### `calc_rtg_daily(daily_df, horizon)`
Calculate daily analyst rating change signals with EWMA smoothing and industry demeaning.
Path: filter_expandable → rolling sum → EWMA → square signal → demean by industry → returns DataFrame with rtg0, rtg0_ma, rtg1_ma

### `rtg_fits(daily_df, horizon, name, middate=None)`
Fit regression coefficients for analyst rating signals across in/out-of-sample.
Path: split by middate → regress_alpha per horizon → extract coef at target → apply to out-of-sample → returns DataFrame with rtg

### `calc_rtg_forecast(daily_df, horizon, middate)`
Main pipeline: signal calculation, regression, and forecast generation.
Path: calc_rtg_daily → calc_forward_returns → merge → rtg_fits → returns DataFrame with rtg

---

## `analyst_badj.py`

### `wavg(group)`
Calculate market-cap weighted average beta-adjusted return.
Path: pbeta * (sum(log_ret * weight) / sum(weight))

### `calc_rtg_daily(daily_df, horizon)`
Calculate beta-adjusted analyst rating signals with multi-lag structure.
Path: filter_expandable → beta-adjust returns → rating signal → demean by industry → create lags → returns DataFrame

### `calc_rtg_intra(intra_df)`
Calculate intraday beta-adjusted rating signals.
Path: filter_expandable → cumulative return / beta → winsorize → demean → returns intra_df with rtgC_B_ma

### `rtg_fits(daily_df, intra_df, horizon, name, middate=None, intercepts=None)`
Fit multi-lag regression for beta-adjusted rating signals with decay weighting.
Path: split by middate → regress per lag → compute residual coefs → apply to out-of-sample → returns DataFrame with rtg

### `calc_rtg_forecast(daily_df, horizon, middate)`
Main pipeline for beta-adjusted analyst rating forecast.
Path: calc_rtg_daily → calc_forward_returns → get_intercept → rtg_fits → returns DataFrame

---

## `badj2_intra.py`

### `wavg(group)` / `wavg2(group)`
Market-cap weighted beta-adjusted return (daily / intraday variants).

### `calc_o2c(daily_df, horizon)`
Calculate daily market-weighted beta-adjusted returns.
Path: filter_expandable → wavg market component → winsorize → demean → create lags → returns DataFrame

### `calc_o2c_intra(intra_df, daily_df)`
Calculate intraday market-weighted beta-adjusted returns.
Path: filter_expandable → wavg2 → winsorize → demean → merge → returns intra_df with o2cC_B_ma

### `o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None)`
Fit intraday regression with 6 hourly bucket coefficients.
Path: split by middate → regress_alpha → extract time-bucket coefs → apply → returns full_df with badj2_i

### `calc_o2c_forecast(daily_df, intra_df, horizon, outsample)`
Main pipeline with Energy/non-Energy sector split.
Path: calc_o2c → calc_o2c_intra → merge → o2c_fits per sector → returns (full_df, outsample_df) with badj2_i

---

## `badj2_multi.py`

### `wavg(group)` / `wavg2(group)`
Market-cap weighted beta-adjusted return (daily / intraday variants).

### `calc_o2c(daily_df, horizon)`
Calculate daily market-weighted beta-adjusted returns with multiple lags.
Path: filter_expandable → wavg → winsorize → demean → create lags → returns DataFrame

### `calc_o2c_intra(intra_df, daily_df)`
Calculate intraday market-weighted beta-adjusted returns.
Path: filter_expandable → wavg2 → winsorize → demean → merge → returns intra_df with o2cC_B_ma

### `o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None)`
Fit regression combining daily lags + intraday.
Path: split by middate → regress daily → compute residual coefs → apply → returns full_df with badj2_m

### `calc_o2c_forecast(daily_df, intra_df, horizon, outsample)`
Main pipeline with sector split.
Path: calc_o2c → calc_o2c_intra → merge → o2c_fits per sector → returns full_df with badj2_m

---

## `badj_both.py`

### `calc_badj_daily(daily_df, horizon)`
Calculate daily beta-adjusted returns (log_ret / pbeta) with lags.
Path: filter_expandable → divide by beta → winsorize → demean → create lags → returns DataFrame

### `calc_badj_intra(intra_df)`
Calculate intraday beta-adjusted returns.
Path: filter_expandable → (overnight + day move) / pbeta → winsorize → demean → returns intra_df with badjC_B_ma

### `badj_fits(daily_df, intra_df, horizon, name, middate=None)`
Fit combined daily+intraday regression.
Path: split → regress lags → compute residual coefs → combine → returns out-of-sample with badj_b

### `calc_badj_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline with sector split.
Path: calc_badj_daily → calc_forward_returns → calc_badj_intra → merge → badj_fits per sector → returns DataFrame with badj_b

---

## `badj_dow_multi.py`

### `calc_o2c(daily_df, horizon)`
Calculate daily beta-adjusted returns (identical to badj_multi.py).
Path: filter_expandable → log_ret / pbeta → winsorize → demean → create lags

### `calc_o2c_intra(intra_df, daily_df)`
Calculate intraday beta-adjusted returns.
Path: filter_expandable → (overnight + intraday) / pbeta → winsorize → demean

### `o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None)`
Fit day-of-week specific regression with separate weekday coefficients.
Path: add dow → split → regress_alpha mode='dow' → extract coefs by dow → apply → returns full_df with badj_m

### `calc_o2c_forecast(daily_df, intra_df, horizon, outsample)`
Main pipeline with sector and day-of-week split.
Path: calc_o2c → calc_o2c_intra → merge → o2c_fits per sector with dow encoding → returns full_df

---

## `badj_intra.py`

### `calc_o2c(daily_df, horizon)`
Calculate daily beta-adjusted returns (reference only, not used in forecast).
Path: filter_expandable → log_ret / pbeta → winsorize → demean → create lags

### `calc_o2c_intra(intra_df, daily_df)`
Calculate intraday beta-adjusted returns.
Path: filter_expandable → (overnight + log(iclose/dopen)) / pbeta → winsorize → demean → merge

### `o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None)`
Fit intraday-only regression with 6 hourly bucket coefficients.
Path: split → regress_alpha intraday → extract 6 time-of-day coefs → apply → returns full_df with badj_i

### `calc_o2c_forecast(daily_df, intra_df, horizon, outsample)`
Main pipeline with sector split (intraday only).
Path: calc_o2c → calc_o2c_intra → merge → o2c_fits per sector → returns (full_df, outsample_df) with badj_i

---

## `badj_multi.py`

### `calc_o2c(daily_df, horizon)`
Calculate daily beta-adjusted returns with multiple lags.
Path: filter_expandable → log_ret / pbeta → winsorize → demean → create lags

### `calc_o2c_intra(intra_df, daily_df)`
Calculate intraday beta-adjusted returns.
Path: filter_expandable → (overnight + intraday) / pbeta → winsorize → demean → merge

### `o2c_fits(daily_df, intra_df, full_df, horizon, name, middate=None)`
Fit regression with daily lags only (intraday coef hardcoded to 0).
Path: split → regress daily lags → compute residual coefs → apply → returns full_df with badj_m

### `calc_o2c_forecast(daily_df, intra_df, horizon, outsample)`
Main pipeline with sector split.
Path: calc_o2c → calc_o2c_intra → merge → o2c_fits per sector → returns (full_df, outsample_df) with badj_m

---

## `badj_rating.py`

### `calc_badj_daily(daily_df, horizon)`
Calculate daily beta-adjusted returns filtered by analyst rating stability.
Path: filter_expandable → log_ret / pbeta → null out where rating != 0 → demean → create lags

### `calc_badj_intra(intra_df)`
Calculate intraday beta-adjusted returns filtered by rolling analyst rating.
Path: filter_expandable → (overnight + intraday) / pbeta → rolling rating filter → demean

### `badj_fits(daily_df, intra_df, horizon, name, middate=None)`
Fit combined daily+intraday regression for rating-filtered signals.
Path: split → regress lags → compute residual coefs → combine → returns out-of-sample with badj_b

### `calc_badj_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline for rating-filtered beta-adjusted forecast.
Path: calc_badj_daily → calc_forward_returns → calc_badj_intra → merge → badj_fits → returns DataFrame

---

## `bd.py`

### `wavg(group)` / `wavg2(group)`
Market-cap weighted beta-adjusted return (daily / intraday).

### `calc_bd_daily(daily_df, horizon)`
Calculate daily beta-adjusted order flow signals with lags.
Path: filter_expandable → wavg beta-adjust → order flow ratio → normalize by sqrt(spread_bps) → winsorize → demean → create lags

### `calc_bd_intra(intra_df)`
Calculate intraday beta-adjusted order flow signals.
Path: filter_expandable → wavg2 → order flow imbalance → winsorize → demean → returns intra_df with bdC_B_ma

### `bd_fits(daily_df, intra_df, horizon, name, middate)`
Fit regression with 6 hourly intraday coefs + daily lags.
Path: split → regress intraday with time-of-day → regress daily lags → combine → returns out-of-sample with bdma

### `calc_bd_forecast(daily_df, intra_df, horizon)`
Main pipeline for beta-adjusted order flow.
Path: calc_bd_daily → calc_forward_returns → calc_bd_intra → merge → bd_fits → returns full_df with bdma

---

## `bd1.py`

### `calc_bd_intra(intra_df)`
Calculate simplified intraday order flow using first-differences.
Path: filter_expandable → diff() hit dollars → ratio → winsorize → demean → merge → returns intra_df with bd1_B_ma

### `bd_fits(daily_df, intra_df, full_df, name)`
Fit simplified regression for order flow signals.
Path: regress_alpha_intra → extract coefs → apply → returns full_df with bd1 and bd1ma

### `calc_bd_forecast(intra_df)`
Main pipeline for simplified bd1 forecast.
Path: calc_bd_intra → merge → bd_fits → returns full_df

---

## `bd_intra.py`

### `wavg(group)` / `wavg2(group)`
Market-cap weighted beta-adjusted return (daily / intraday).

### `calc_bd_intra(intra_df)`
Calculate pure intraday beta-adjusted order flow signals.
Path: filter_expandable → wavg2 → order flow imbalance → winsorize → demean → returns intra_df with bdC_B_ma

### `bd_fits(intra_df, horizon, name, middate)`
Fit intraday regression with 6 hourly bucket coefficients.
Path: split → regress_alpha 'intra_eod' → extract 6 coefs → apply → returns out-of-sample with bdma_i

### `calc_bd_forecast(daily_df, intra_df, horizon)`
Main pipeline for pure intraday order flow.
Path: calc_forward_returns → calc_bd_intra → merge → bd_fits → returns full_df with bdma_i

---

## `bigsim_test.py`

### `pnl_sum(group)`
Calculate cumulative P&L using log returns.
Path: compute cumulative log return diff → multiply by position → sum → return

### Main Script
Load daily data → parse alpha forecasts → optimize portfolio → apply constraints → write results to CSV.

---

## `bsim.py`

### `pnl_sum(group)`
Calculate cumulative P&L using log returns.
Path: compute cumulative log return diff → multiply by position → sum → return

### Main Script
Load data → parse forecasts → load mus → optimize positions via opt.optimize() → apply participation constraints → skip non-positive utility trades (dutil <= 0) → write CSV → send email.

---

## `bsim_weights.py`

### `pnl_sum(group)`
Calculate cumulative P&L using log returns.

### Main Script
Load data → parse forecasts → dynamically adjust weights by 5-day rolling returns (1.2x if positive, 0.8x if negative) → optimize → write CSV → send email.

---

## `bsz.py`

### `wavg(group)` / `wavg2(group)`
Market-cap weighted beta-adjusted return (daily / intraday).

### `calc_bsz_daily(daily_df, horizon)`
Calculate daily bid-ask size imbalance signals with lags.
Path: filter_expandable → (AskSize - BidSize) / (BidSize + AskSize) / sqrt(spread_bps) → winsorize → demean → create lags

### `calc_bsz_intra(intra_df)`
Calculate intraday bid-ask size imbalance signals.
Path: filter_expandable → size imbalance / sqrt(meanSpread) → winsorize → demean → returns intra_df with bszC_B_ma

### `bsz_fits(daily_df, intra_df, horizon, name, middate)`
Fit regression with 6 hourly intraday coefs + daily lags.
Path: split → regress intraday time-of-day → regress daily lags → combine → returns out-of-sample with bsz

### `calc_bsz_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline for bid-ask size imbalance.
Path: calc_bsz_daily → calc_forward_returns → calc_bsz_intra → merge → bsz_fits → returns full_df

---

## `bsz1.py`

### `calc_bsz_daily(intra_df, horizon)`
Calculate daily bid-ask size imbalance from EOD bar data.
Path: unstack → select 16:00 → filter_expandable → size imbalance → winsorize → demean → lags

### `calc_bsz_intra(intra_df)`
Calculate intraday bid-ask size imbalance signals.
Path: filter_expandable → size imbalance → winsorize → demean → merge → returns intra_df with bszC_B_ma

### `bsz_fits(daily_df, intra_df, full_df, horizon, name)`
Fit regression for both raw and market-adjusted signals.
Path: regress_alpha_intra → regress_alpha_daily → extract coefs → returns full_df with bsz and bszma

### `calc_bsz_forecast(intra_df, horizon)`
Main pipeline with Energy/non-Energy sector split.
Path: calc_bsz_daily → calc_bsz_intra → merge → bsz_fits per sector → returns full_df

---

## `c2o.py`

### `wavg(group)` / `wavg2(group)` / `wavg3(group)`
Market-cap weighted beta-adjusted return variants (daily / overnight / current).

### `calc_c2o_daily(daily_df, horizon)`
Calculate daily close-to-open gap signals with beta adjustment.
Path: filter_expandable → beta-adjust → winsorize → demean → create lags

### `calc_c2o_intra(intra_df)`
Calculate intraday close-to-open gap signals with beta adjustment.
Path: filter_expandable → beta-adjust overnight/current → winsorize → demean

### `c2o_fits(daily_df, intra_df, horizon, name, middate)`
Fit c2o alpha model and generate forecasts.
Path: split → regress intraday + daily → extract coefs → apply → returns DataFrame with c2o

### `calc_c2o_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline with sector-specific regressions.
Path: calc_c2o_daily → calc_forward_returns → calc_c2o_intra → merge → c2o_fits per sector → concat

---

## `calc.py`

### `calc_vol_profiles(full_df)`
Calculate intraday volume participation profiles at 15-min intervals.
Path: groupby timeslice → rolling median/std of dollar volume → store in DataFrame columns

### `calc_price_extras(daily_df)`
Calculate derived volatility and volume metrics (ratios).
Path: compute volat_ratio, volume_ratio, volat_move → return DataFrame

### `calc_forward_returns(daily_df, horizon)`
Calculate forward cumulative returns at multiple horizons.
Path: groupby sid → rolling sum of log_ret with forward shift → returns MultiIndex DataFrame

### `winsorize(data, std_level=5)`
Cap outliers at N standard deviations from mean.
Path: calc mean/std → clip values → return Series

### `winsorize_by_date(data)`
Apply winsorization within each date cross-section.
Path: groupby date → apply winsorize() → return Series

### `winsorize_by_ts(data)`
Apply winsorization within each intraday timestamp.
Path: groupby iclose_ts → apply winsorize() → return Series

### `winsorize_by_group(data, group)`
Apply winsorization within custom groups.
Path: groupby variable → apply winsorize() → return Series

### `rolling_ew_corr_pairwise(df, halflife)`
Calculate exponentially-weighted pairwise correlations.
Path: iterate column pairs → ewm().corr() → return dict of dicts

### `push_data(df, col)`
Shift column forward 1 period and merge.
Path: unstack → shift(-1) → restack → merge → return DataFrame with col_n

### `lag_data(daily_df)`
Lag entire DataFrame by 1 period and merge.
Path: unstack → shift(1) → restack → merge → return DataFrame

### `calc_med_price_corr(daily_df)`
Placeholder [NOT IMPLEMENTED].

### `calc_resid_vol(daily_df)`
Calculate Barra residual volatility.
Path: rolling 20-day std of barraResidRet → return Series

### `calc_factor_vol(factor_df)`
Calculate exponentially-weighted factor covariance matrix.
Path: iterate factor pairs → ewm().cov() → return dict of covariance Series

### `create_z_score(daily_df, name)`
Standardize column to z-score within each date cross-section.
Path: groupby date → (x - mean) / std → return DataFrame with _z column

### `calc_factors(daily_df, barraOnly=False)`
Decompose stock returns into factor and residual returns via WLS.
Path: groupby date → factorize() → collect residuals → return (daily_df, factorRets_df)

### `calc_intra_factors(intra_df, barraOnly=False)`
Decompose intraday returns into factor and residual components.
Path: groupby iclose_ts → factorize() → return (intra_df, factorRets_df)

### `factorize(loadings_df, returns_df, weights_df, indwgt)`
Estimate factor returns using WLS regression.
Path: lmfit.minimize(fcn2min) → extract factor returns and residuals → return (dict, array)

### `fcn2min(params, x, data)`
Objective function for lmfit minimization.
Path: compute weighted residuals (x @ params - data) * weights → return array

### `mkt_ret(group)`
Calculate market-cap weighted average return.
Path: (return * weight).sum() / weight.sum() → return float

---

## `dumpall.py`

No public functions (script execution only).

---

## `ebs.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_sal_daily(daily_df, horizon)`
Calculate analyst estimate revision signals with lags.
Path: filter_expandable → std_diff filter → SAL_diff_mean / SAL_median → create lags

### `sal_fits(daily_df, horizon, name, middate=None, intercepts=None)`
Fit separate models for positive/negative estimate revisions.
Path: split by sign → regress_alpha per lag → adjust intercepts → apply → returns DataFrame

### `calc_sal_forecast(daily_df, horizon, middate)`
Main pipeline for analyst estimate-based forecasts.
Path: calc_sal_daily → calc_forward_returns → get_intercept → sal_fits → returns DataFrame

---

## `eps.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_eps_daily(daily_df, horizon)`
Calculate daily earnings surprise signals with distributed lags.
Path: filter_expandable → eps0 = EPS_diff_mean / EPS_median → create lags → returns DataFrame

### `eps_fits(daily_df, horizon, name, middate=None)`
Fit distributed lag regression of EPS surprise against forward returns.
Path: split → regress per horizon → extract coefs → apply → returns DataFrame with eps

### `calc_eps_forecast(daily_df, horizon, middate)`
Main pipeline for EPS-based alpha.
Path: calc_eps_daily → calc_forward_returns → eps_fits → returns DataFrame

---

## `factors.py`

No public functions (script execution only).

---

## `hl.py`

### `calc_hl_daily(full_df, horizon)`
Calculate daily high-low mean reversion signals with lags.
Path: filter_expandable → hl0 = close / sqrt(high * low) → winsorize → demean → create lags → left merge

### `calc_hl_intra(full_df)`
Calculate intraday high-low mean reversion signals.
Path: filter_expandable → hlC = iclose / sqrt(dhigh * dlow) → winsorize → demean → left merge

### `hl_fits(daily_df, intra_df, full_df, horizon, name)`
Fit regressions and generate forecasts (hardcoded hl=4.0).
Path: regress intraday + daily → extract coefs → populate forecast → set hl=4.0 → return full_df

### `calc_hl_forecast(daily_df, intra_df, horizon)`
Main pipeline with sector split.
Path: calc_hl_daily → calc_hl_intra → merge → hl_fits per sector → dump_alpha

---

## `hl_intra.py`

### `calc_hl_intra(full_df)`
Calculate intraday high-low mean reversion signals for real-time trading.
Path: filter_expandable → hlC = iclose / sqrt(dhigh * dlow) → winsorize → demean → left merge

### `hl_fits(daily_df, intra_df, full_df, horizon, name)`
Fit intraday-only regression.
Path: regress hlC_B_ma at horizon=1 → extract coef → apply → return full_df

### `calc_hl_forecast(daily_df, intra_df, horizon)`
Main pipeline with sector split (intraday only).
Path: calc_hl_intra → merge → hl_fits per sector → dump_alpha

---

## `htb.py`

### `calc_htb_daily(daily_df, horizon)`
Calculate hard-to-borrow fee rate signals with lags.
Path: filter_expandable → winsorize fee_rate → create lagged features

### `htb_fits(daily_df, intra_df, horizon, name, middate)`
Fit HTB regression and generate intraday forecasts.
Path: split → regress per horizon → extract coefs → apply to intraday → returns DataFrame with htb

### `calc_htb_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline for HTB forecasts.
Path: calc_htb_daily → calc_forward_returns → merge → htb_fits → returns DataFrame

---

## `load_data_live.py`

### `load_live_file(ifile)`
Load live price data with bid/ask spreads and calculate mid-prices.
Path: read_csv → close_i = (bid + ask) / 2.0 → return DataFrame

---

## `loaddata.py`

### `get_uni(start, end, lookback, uni_size=1400)`
Load and filter stock universe (country/price/volume/cap filters).
Path: read univ/secdata/price CSVs → apply filters → return DataFrame

### `load_barra(uni_df, start, end, cols=None)`
Load Barra risk factors for given universe and date range.
Path: loop dates → read barra CSVs → merge with uni_df → return MultiIndex DataFrame

### `load_prices(uni_df, start, end, cols)`
Load daily prices with returns and volume metrics.
Path: loop dates → read price CSVs → compute mdvp/expandable → lag_data() → return DataFrame

### `load_daybars(uni_df, start, end, cols=None, freq='30Min')`
Load daily bar summaries with intraday OHLCV.
Path: loop dates → read bars.txt.gz → resample to freq → merge → return DataFrame

### `load_bars(uni_df, start, end, cols=None, freq=30)`
Load and aggregate intraday tick bars.
Path: loop dates → read bars.txt.gz → aggregate_bars() → merge → return DataFrame

### `aggregate_bars(bars_df, freq=30)`
Aggregate tick-level bars into time intervals.
Path: loop time buckets → groupby ticker → compute OHLCV/microstructure → return DataFrame

### `load_volume_profile(uni_df, start, end, freq='30Min')`
Load intraday volume profiles (median cumulative volume by time).
Path: loop dates → read profiles → reshape/resample → return DataFrame

### `load_earnings_dates(uni_df, start, end)`
Load upcoming earnings announcement dates.
Path: loop dates → merge earnings → compute daysToEarn → return DataFrame

### `load_past_earnings_dates(uni_df, start, end)`
Load most recent past earnings dates.
Path: loop dates → merge past earnings → compute daysFromEarn → return DataFrame

### `load_locates(uni_df, start, end)`
Load short sale locate availability and borrow costs.
Path: loop dates → merge locate CSVs → negate qty → return DataFrame

### `load_mus(mdir, fcast, start, end)`
Load alpha/mu forecasts from directory.
Path: glob alpha files → filter date/time → read_csv → concat → return DataFrame

### `load_qb_implied_orders(mdir, start, end)`
Load implied order flow from quantbot portfolio files.
Path: glob PORTFOLIO.csv → extract timeslices → compute open_order → concat → return DataFrame

### `load_qb_positions(mdir, start, end)`
Load full intraday position history from quantbot.
Path: glob csv files → read all timestamps → concat → return DataFrame

### `load_qb_eods(mdir, start, end)`
Load end-of-day position and trading summary.
Path: glob EOD.csv → compute dtradenot → concat → return DataFrame

### `load_qb_orders(ofile, date)`
Load order submission data from quantbot orders file.
Path: read csv → filter NEW orders → negate short qty → return DataFrame

### `load_qb_exec(efile, date)`
Load execution/fill data from quantbot executions file.
Path: read csv → negate short qty → return DataFrame

### `load_factor_cache(start, end)`
Load cached factor data from HDF5 files.
Path: glob cache files → filter dates → read HDF5 → combine_first() → return DataFrame

### `load_cache(start, end, cols=None)`
Load cached market data from HDF5 files.
Path: glob cache files → filter dates → read HDF5 → combine_first() → return DataFrame

### `load_all_results(fdir, start, end, cols=None)`
Load alpha results from a single forecast directory.
Path: glob alpha files → filter date/time → read_csv → concat → return DataFrame

### `load_merged_results(fdirs, start, end, cols=None)`
Load and merge alpha results from multiple directories.
Path: loop fdirs → load_all_results() → merge → remove_dup_cols() → return DataFrame

### `transform_barra(barra_df)`
Transform Barra industry classifications from long to wide format.
Path: unstack indname/weights per level → sum → return DataFrame

### `load_ratings_hist(uni_df, start, end, intra=False)`
Load analyst rating history and compute consensus metrics.
Path: query IBES → groupby date/sid → mean/median/std → return DataFrame

### `load_target_hist(uni_df, start, end, intra=False)`
Load analyst price target history and compute consensus.
Path: query IBES → groupby date/sid → mean/median/std → return DataFrame

### `load_estimate_hist(uni_df, start, end, estimate)`
Load analyst estimate history for a specific measure.
Path: query IBES → filter next fiscal period → compute consensus → return DataFrame

### `load_consensus(dtype, uni_df, start, end, freq='30Min')`
Load intraday consensus estimate data from IBES QFS files.
Path: glob estimate files → concat → groupby sid → resample → return DataFrame

---

## `mom_year.py`

### `calc_mom_daily(daily_df, horizon)`
Calculate annual momentum signal with 232-day lag.
Path: filter_expandable → rolling_sum(log_ret, 20) → demean → shift 232 days → returns DataFrame with mom1_ma

### `mom_fits(daily_df, horizon, name, middate)`
Fit momentum regression (starts 252 days in for sufficient history).
Path: split → regress mom1_ma → extract coef → apply → returns DataFrame with mom

### `calc_mom_forecast(daily_df, horizon, middate)`
Main pipeline for momentum forecasts.
Path: calc_mom_daily → calc_forward_returns → mom_fits → returns DataFrame

---

## `new1.py`

### `wavg(group)` / `wavg2(group)`
Market-cap weighted beta-adjusted return (daily / intraday).

### `calc_insd_daily(daily_df, horizon)`
Calculate daily insideness (order book) signals with lags.
Path: filter_expandable → beta-adjust → insideness * sign(return) → winsorize → demean → create lags

### `calc_insd_intra(intra_df)`
Calculate intraday insideness signals.
Path: filter_expandable → beta-adjust → insideness * sign(return) → winsorize → demean

### `insd_fits(daily_df, intra_df, horizon, name, middate)`
Fit with 6 hourly time-of-day bucket coefficients.
Path: split → regress intraday + daily → extract time-bucket coefs → apply → returns DataFrame with insd

### `calc_insd_forecast(daily_df, intra_df, horizon)`
Main pipeline for insideness forecasts.
Path: calc_insd_daily → calc_forward_returns → calc_insd_intra → merge → insd_fits → returns DataFrame

---

## `opt.py`

### `init()`
Initialize global optimization data arrays.
Path: allocate zero arrays for g_positions, g_mu, g_rvar, g_factors, g_fcov, etc.

### `optimize()`
Main portfolio optimization entry point.
Path: getUntradeable() → setup constraints → scipy.optimize.minimize() → compute marginal utilities → printinfo() → return (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2)

### `getUntradeable()`
Partition securities into tradeable/untradeable sets.
Path: loop secs → check if bound range < $10 → return (tradeable, untradeable)

### `setupProblem_scipy(positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, borrowRate, price, lb, ub, Ac, bc, lbexp, ubexp, untradeable_info, sumnot, zero_start)`
Configure scipy.optimize.minimize problem.
Path: setup bounds, LinearConstraint, NonlinearConstraint → return dict

### `objective(target, ...)`
Portfolio utility function to maximize.
Path: calls objective_detail() → return total utility

### `objective_detail(target, ...)`
Objective with detailed component breakdown.
Path: compute tmu, tsrisk, tfrisk, tslip, tcosts → return tuple

### `objective_grad(target, ...)`
Analytical gradient of objective.
Path: sum gradients of mu, risk, slippage, costs → return array

### `slippageFuncAdv(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu)`
Calculate total market impact slippage cost.
Path: compute I (volatility), J (participation) per security → sum(J * |delta|) → return float

### `slippageFunc_grad(target, ...)`
Gradient of slippage function.
Path: compute Id, Jd → multiply by sign(delta) → return array

### `costsFunc(target, positions, brate, price, execFee)`
Calculate total execution and borrow costs.
Path: execFee * sum(|delta_shares|) → return float

### `costsFunc_grad(target, ...)`
Gradient of costs function.
Path: execFee * sign(delta) / price → return array

### `constrain_by_capital(target, ...)`
Constraint: sum(abs(target)) <= max_sumnot.
Path: sum(abs(target)) - max_sumnot → return float

### `constrain_by_capital_grad(target, ...)`
Gradient of capital constraint.
Path: sign(target) → return array

### `constrain_by_trdnot(target, ...)`
Constraint: turnover limit [UNUSED].
Path: sum(abs(delta)) - max_trdnot_hard → return float

### `printinfo(target, ...)`
Print optimization summary.
Path: calculate long/short notional → __printpointinfo() twice → print stats

---

## `osim.py`

### `objective(weights)`
Objective function for forecast weight optimization via simulated order execution.
Path: iterate timestamps → combine forecasts → execute trades → compute Sharpe - diversity penalty → return float

### Main Script
Load data → load forecast positions → merge → set fill prices → scipy.optimize.minimize(objective) → print optimal weights.

---

## `osim2.py`

### Main Script
Fixed-weight simulation loop: load data → set fill prices → fixed weights 0.5 → for each timestamp: merge positions, apply corporate actions, combine forecasts, apply constraints, track P&L → compute Sharpe.

---

## `osim_simple.py`

### `fcn(weights, start, end)`
Variance minimization: compute portfolio variance, return 1/sqrt(variance).

### `sharpe_fcn(weights, start, end)`
Sharpe ratio maximization: annual_return / annual_volatility.

### Main Script
Load returns → rolling 30-day window optimization → evaluate out-of-sample Sharpe → print mean.

---

## `other.py`

### `calc_other_daily(daily_df, horizon)`
Calculate log_ret * volat_ratio signals with winsorization and industry demeaning.
Path: signals → winsorize_by_group → demean → create lags

### `calc_other_intra(intra_df, daily_df)`
Calculate intraday version (overnight + intraday returns * volat_ratio).
Path: winsorize_by_ts → demean → return DataFrame

### `other_fits(daily_df, intra_df, full_df, horizon, name)`
Fit regression combining daily and intraday signals.
Path: regress_alpha_intra → regress_alpha_daily → extract coefs → return full_df with other

### `calc_other_forecast(daily_df, intra_df, horizon)`
Main pipeline.
Path: calc_other_daily → calc_other_intra → merge → other_fits → returns full_df

---

## `other2.py`

### `calc_other_daily(daily_df, horizon)`
Calculate log_ret * insideness signals.
Path: multiply → winsorize_by_group → demean → create lags

### `calc_other_intra(intra_df, daily_df)`
Calculate intraday version using insideness.
Path: winsorize_by_ts → demean → return DataFrame

### `other_fits(daily_df, intra_df, full_df, horizon, name)`
Fit regression and generate forecasts.
Path: regress_alpha_intra → regress_alpha_daily → extract coefs → return full_df

### `calc_other_forecast(daily_df, intra_df, horizon)`
Main pipeline.
Path: calc_other_daily → calc_other_intra → merge → other_fits → returns full_df

---

## `pca.py`

### `calc_pca_daily(daily_df, horizon)`
Calculate daily PCA residuals via rolling 30-day window.
Path: winsorize log returns → for each day fit PCA on standardized window → compute residuals → create lags

### `calc_pca_intra(intra_df)`
Calculate intraday PCA residuals.
Path: daily model cache → augment with current timestamp → fit PCA → residuals → return DataFrame

### `pca_fits(daily_df, intra_df, horizon, name, middate)`
Fit regression with 6 hourly time-of-day coefficients.
Path: regress_alpha intra → extract hourly coefs → regress daily lags → compute incremental coefs → return DataFrame with pca

### `calc_pca_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline for PCA-based alpha.
Path: calc_pca_daily → calc_pca_intra → merge → pca_fits → return DataFrame

---

## `pca_generator.py`

### `calc_pca_intra(intra_df)`
Calculate intraday PCA residuals from correlation structure.
Path: bar-to-bar returns → rolling_corr_pairwise(10) → for each timestamp PCA → residuals → winsorize → demean → returns pcaC_B_ma

---

## `pca_generator_daily.py`

### `calc_pca_daily(daily_df)`
Calculate daily PCA decomposition using exponentially-weighted correlations.
Path: overnight + prev day returns → winsorize → demean → rolling_ew_corr(5-day halflife) → PCA per date → residuals → returns pca0

---

## `prod_eps.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_eps_daily(daily_df, horizon)`
Calculate daily EPS revision signals with lags.
Path: filter_expandable → std_diff filter → EPS_diff_mean / EPS_median → create lags

### `generate_coefs(daily_df, horizon, name, coeffile)`
Fit regression and save coefficients to CSV.
Path: regress_alpha per horizon → extract coef0 → compute incremental coefs → save CSV

### `eps_alpha(daily_df, horizon, name, coeffile)`
Generate forecast using pre-fitted coefficients.
Path: load coefs from CSV → weighted sum of eps signals → return DataFrame with eps

### `calc_eps_forecast(daily_df, horizon, coeffile, fit)`
Main orchestrator (fit mode saves coefs, predict mode generates alpha).
Path: calc_eps_daily → if fit: generate_coefs, else: eps_alpha

---

## `prod_rtg.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_rtg_daily(daily_df, horizon)`
Calculate daily rating revision signals (rating_diff_mean^3).
Path: filter_expandable → std_diff filter → cube signal → create lags

### `generate_coefs(daily_df, horizon, fitfile)`
Fit regression with linear decay and save to CSV.
Path: regress_alpha per horizon → extract coefs with decay → save CSV

### `rtg_alpha(daily_df, horizon, coeffile=None)`
Apply fitted coefficients to generate rating alpha.
Path: load coefs → weighted sum → return DataFrame with rtg

### `calc_rtg_forecast(daily_df, horizon, coeffile, fit)`
Main entry (fit/predict modes).
Path: calc_rtg_daily → if fit: generate_coefs, else: rtg_alpha

---

## `prod_sal.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_sal_daily(daily_df, horizon)`
Calculate daily SAL estimate revision signals.
Path: filter_expandable → std_diff filter → SAL_diff_mean / SAL_median → create lags

### `generate_coefs(daily_df, horizon, name, coeffile, intercepts)`
Fit separate up/down regime models and save to CSV.
Path: split by sign → regress per lag/regime → adjust intercepts → save CSV

### `sal_alpha(daily_df, horizon, name, coeffile)`
Apply regime-specific fitted coefficients.
Path: load coefs → select regime per stock → weighted sum → return DataFrame with sal

### `calc_sal_forecast(daily_df, horizon, coeffile, fit)`
Main entry (fit/predict modes).
Path: calc_sal_daily → if fit: generate_coefs, else: sal_alpha

---

## `prod_tgt.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_tgt_daily(daily_df, horizon)`
Calculate daily price target deviation signals.
Path: filter_expandable → log(target_median / close) → winsorize → demean → create lags

### `generate_coefs(daily_df, horizon, fitfile)`
Fit WLS regression and save coefficients.
Path: regress_alpha per horizon → extract coefs → save CSV

### `tgt_alpha(daily_df, horizon, fitfile)`
Generate forecast using pre-fitted coefficients.
Path: load coefs → weighted sum → return DataFrame with tgt

### `calc_tgt_forecast(daily_df, horizon, coeffile, fit)`
Main orchestrator (fit: 720 days historical, predict: recent data).
Path: calc_tgt_daily → if fit: generate_coefs, else: tgt_alpha

---

## `qhl_both.py`

### `calc_qhl_daily(daily_df, horizon)`
Calculate daily high-low signals with multiple lags.
Path: close / sqrt(qhigh * qlow) → winsorize → demean → create lags

### `calc_qhl_intra(intra_df)`
Calculate intraday high-low signals.
Path: iclose / sqrt(qhigh * qlow) → winsorize → demean → returns qhlC_B_ma

### `qhl_fits(daily_df, intra_df, horizon, name, middate)`
Fit daily regressions and generate combined forecast.
Path: split → regress daily lags → extract coefs → combine daily + intraday → returns qhl_b

### `calc_qhl_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline with Energy/non-Energy split.
Path: calc_qhl_daily → calc_qhl_intra → merge → qhl_fits per sector → concat

---

## `qhl_both_i.py`

### `calc_qhl_daily(daily_df, horizon)` / `calc_qhl_intra(intra_df)`
Same signal calculation as qhl_both.py.

### `qhl_fits(daily_df, intra_df, horizon, name, middate)`
Fit intraday hourly + daily multi-lag coefficients.
Path: regress intra → map to 6 hourly buckets → regress daily lags → combine → returns qhl_b

### `calc_qhl_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline with sector split.

---

## `qhl_intra.py`

### `calc_qhl_intra(intra_df)`
Calculate quote-based intraday high-low signal.
Path: iclose / sqrt(qhigh * qlow) → winsorize → demean → returns qhlC_B_ma

### `qhl_fits(daily_df, intra_df, horizon, name, middate)`
Fit hourly coefficients (intraday only).
Path: split → regress_alpha 'intra_eod' → map to 6 periods → apply → returns qhl_i

### `calc_qhl_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline with sector split.

---

## `qhl_multi.py`

### `calc_qhl_daily(daily_df, horizon)` / `calc_qhl_intra(intra_df, daily_df)`
Daily high-low signals with lags; intraday signal (zero-weighted).

### `qhl_fits(daily_df, intra_df, full_df, horizon, name, middate)`
Fit multi-timeframe incremental coefficients.
Path: split → regress per lag → coef[lag] = coef[horizon] - coef[lag] → combine → returns full_df

### `calc_qhl_forecast(daily_df, intra_df, horizon, outsample)`
Main pipeline with sector split, optional midpoint out-of-sample.

---

## `qsim.py`

### Main Script
30-minute bar simulation: load bar data → compute incremental VWAP → load intraday alpha → mix forecasts → size positions (ALPHA_MULT * forecast, capped at 1% bar volume, max $500K) → for each bar: compute forward returns, apply slippage → aggregate P&L by time/day/month → output Sharpe, charts.

---

## `rating_diff.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_rtg_daily(daily_df, horizon)`
Calculate filtered analyst rating change signals with multi-lag structure.
Path: rating_std.diff() → zero out when std_diff <= 0 → create lags

### `rtg_fits(daily_df, horizon, name, middate)`
Fit multi-lag regression with linear decay weighting.
Path: split → regress per lag → linear decay weights → combine → returns forecast

### `calc_rtg_forecast(daily_df, horizon, middate)`
Main pipeline.
Path: calc_rtg_daily → calc_forward_returns → rtg_fits → returns DataFrame

---

## `rating_diff_updn.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_rtg_daily(daily_df, horizon)`
Calculate quadratic filtered rating signals (sign * diff^2).
Path: rating_std.diff() → zero when std_diff <= 0 → sign * squared → create lags

### `rtg_fits(daily_df, horizon, name, middate, intercepts)`
Fit asymmetric upgrade/downgrade models with intercept adjustments.
Path: split by sign → regress per regime → adjust intercepts → apply regime-specific coefs

### `calc_rtg_forecast(daily_df, horizon, middate)`
Main pipeline with baseline intercept adjustment.
Path: calc_rtg_daily → calc_forward_returns → get_intercept → rtg_fits

---

## `readcsv.py`

No public functions (stdin CSV debug utility).

---

## `regress.py`

### `plot_fit(fits_df, name)`
Visualize regression coefficients across horizons with error bars.
Path: errorbar plot → savefig → return None

### `extract_results(results, indep, horizon)`
Extract key statistics from statsmodels regression result.
Path: get params/bse/tvalues → format dict → return single-row DataFrame

### `get_intercept(daily_df, horizon, name, middate=None)`
Extract regression intercepts across horizons.
Path: loop horizons → regress_alpha_daily() → return dict of intercepts

### `regress_alpha(results_df, indep, horizon, median=False, rtype='daily', intercept=True, start=None, end=None)`
Main regression dispatcher.
Path: dispatch to regress_alpha_daily/intra/dow based on rtype; if median=True, 3-fold validation

### `regress_alpha_daily(daily_df, indep, horizon, intercept=True)`
Cross-sectional daily WLS regression (alpha vs forward returns, weighted by sqrt(mdvp)).
Path: extract cols → dropna → WLS fit → extract_results() → return DataFrame

### `regress_alpha_intra_eod(intra_df, indep)`
Intraday alpha vs end-of-day returns.
Path: loop timeslices → compute day_ret → WLS fit → return DataFrame

### `regress_alpha_intra(intra_df, indep, horizon)`
Intraday forward-looking regression with multi-bar horizon.
Path: loop timeslices → shift log_ret forward → sum into cum_ret → WLS fit → return DataFrame

### `regress_alpha_dow(daily_df, indep, horizon)`
Day-of-week stratified regression.
Path: groupby dow → WLS fit per day → return DataFrame

---

## `rev.py`

### `calc_rev_daily(daily_df, horizon, lag)`
Calculate industry-demeaned rolling return reversal signals.
Path: rolling_sum(log_ret, lag) → demean → shift → returns rev0_ma and rev1_ma

### `rev_fits(daily_df, horizon, name, middate)`
Fit reversal regression.
Path: split → regress rev1_ma → extract coef → apply → returns DataFrame

### `calc_rev_forecast(daily_df, horizon, middate, lag)`
Main pipeline.
Path: calc_rev_daily → calc_forward_returns → rev_fits → returns DataFrame

---

## `rrb.py`

### `calc_rrb_daily(daily_df, horizon)`
Calculate daily Barra residual return signals.
Path: barraResidRet → winsorize → demean → create lags

### `calc_rrb_intra(intra_df)`
Calculate intraday Barra residual signals.
Path: barraResidRetI → winsorize → demean → returns rrbC_B_ma

### `rrb_fits(daily_df, intra_df, horizon, name, middate)`
Fit daily residual regression.
Path: split → regress per lag → compute incremental coefs → combine → returns rrb

### `calc_rrb_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline (excludes Energy sector).
Path: rrb_fits(non-Energy) → returns DataFrame

---

## `slip.py`

No public functions (one-off execution analysis script).

---

## `ssim.py`

### Main Script
Full lifecycle simulation: load data → parse forecasts → combine multi-alpha positions → set fill prices (VWAP/mid) → for each timestamp: merge positions, apply corporate actions (splits/dividends), apply participation constraints, update cash, mark to market → output: total P&L, Sharpe, factor exposure, industry/decile breakdowns, temporal breakdowns, plots.

---

## `target.py`

### `wavg(group)`
Market-cap weighted beta-adjusted return.

### `calc_tgt_daily(daily_df, horizon)`
Calculate analyst price target deviation signals with distributed lags.
Path: log(target_median / close) → winsorize → demean → create lags

### `tgt_fits(daily_df, horizon, name, middate, intercepts)`
Fit distributed lag regression.
Path: split → regress per lag → coef[lag] = coef[horizon] - coef[lag] → combine → returns DataFrame

### `calc_tgt_forecast(daily_df, horizon, middate)`
Main pipeline.
Path: calc_tgt_daily → calc_forward_returns → tgt_fits → returns DataFrame

---

## `util.py`

### `mkdir_p(path)`
Create directory with parents (mkdir -p equivalent).
Path: os.makedirs() → catch EEXIST

### `email(subj, message)`
Send email via local SMTP.
Path: MIMEText → SMTP.sendmail()

### `merge_barra_data(price_df, barra_df)`
Merge Barra factors with prices, lagging by 1 day.
Path: unstack barra → shift(1) → restack → merge → remove_dup_cols()

### `remove_dup_cols(result_df)`
Remove `_dead` suffixed duplicate columns from merge.
Path: loop columns → delete *_dead → return DataFrame

### `merge_intra_eod(daily_df, intra_df)`
Merge EOD (16:00) bar data with daily data.
Path: at_time('16:00') → merge → remove_dup_cols()

### `merge_intra_data(daily_df, intra_df)`
Left-join daily data into every intraday bar.
Path: left merge on (date, sid) → remove_dup_cols() → set index (iclose_ts, sid)

### `filter_expandable(df)`
Filter to expandable universe securities.
Path: dropna(expandable) → filter True → return DataFrame

### `filter_pca(df)`
Filter to large-cap stocks (mkt_cap > $10B) for PCA.
Path: filter mkt_cap > 1e10 → return DataFrame

### `dump_hd5(result_df, name)`
Save DataFrame to HDF5 with date range filename.
Path: to_hdf() with zlib compression

### `dump_all(results_df)`
Export all alpha signals to per-timestamp CSV files.
Path: groupby iclose_ts → to_csv() per group into ./all/

### `dump_alpha(results_df, name)`
Export single alpha signal to per-timestamp CSV files.
Path: groupby iclose_ts → to_csv([sid, name]) per group into ./{name}/

### `dump_prod_alpha(results_df, name, outputfile)`
Export most recent date's alpha to single CSV.
Path: get max date → to_csv()

### `dump_daily_alpha(results_df, name)`
Replicate daily alpha across 26 intraday timestamps (9:30-15:45, 15-min intervals).
Path: groupby date → for each date/time → to_csv()

### `df_dates(df)`
Extract date range string from DataFrame index.
Path: index[0][0] + index[-1][0] → format YYYYMMDD-YYYYMMDD

### `merge_daily_calcs(full_df, result_df)`
Left-merge new daily calculated columns into main DataFrame.
Path: identify new cols → left merge on (date, sid) → restore index

### `merge_intra_calcs(full_df, result_df)`
Left-merge new intraday calculated columns into main DataFrame.
Path: delete date col → identify new cols → left merge on index

### `get_overlapping_cols(df1, df2)`
Get columns in df1 NOT in df2 (name is misleading).
Path: set difference → return list

### `load_merged_results(fdirs, start, end, cols=None)`
Load and merge alpha results from multiple directories.
Path: loop fdirs → load_all_results() → merge → remove_dup_cols()

### `load_all_results(fdir, start, end, cols=None)`
Load alpha CSV files from {fdir}/all/ for date range (10:00-15:30 only).
Path: glob alpha files → filter date/time → read_csv → concat(verify_integrity=True)

---

## `vadj.py`

### `wavg(group)` / `wavg2(group)` / `wavg_ind(group)` / `volmult_i(group)` / `volmult2(group)`
Weighted return calculators and volume normalizers.

### `calc_vadj_daily(daily_df, horizon)`
Calculate volume-adjusted signals with relative volume and beta-adjusted returns.
Path: volmult2() → rv * sign(badjret) → winsorize → demean → create lags

### `calc_vadj_intra(intra_df)`
Calculate intraday volume-adjusted signals.
Path: wavg2() beta adjustment → rv_i * sign(badjret_i) → winsorize → demean → returns vadjC_B_ma

### `vadj_fits(daily_df, intra_df, horizon, name, middate)`
Fit 6 hourly intraday + daily lag coefficients.
Path: regress intra → map to 6 periods → regress daily lags → incremental coefs → returns vadj_b

### `calc_vadj_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline with Energy sector split.
Path: calc_vadj_daily → calc_vadj_intra → merge → vadj_fits per sector → concat

---

## `vadj_intra.py`

### `wavg(group)` / `wavg2(group)` / `wavg_ind(group)`
Weighted return and signal calculators.

### `calc_vadj_intra(intra_df)`
Calculate intraday-only volume-adjusted signals.
Path: overnight + intraday return → wavg2 → rv_i * sign(badjret) → winsorize → demean

### `vadj_fits(daily_df, intra_df, horizon, name, middate)`
Fit intraday-only hourly coefficients.
Path: split → regress 'intra_eod' → map to 6 periods → apply → returns vadj_i

### `calc_vadj_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline with sector split (intraday only).
Path: calc_vadj_intra → merge → vadj_fits per sector → concat

---

## `vadj_multi.py`

### `wavg(group)` / `wavg2(group)` / `wavg_ind(group)`
Weighted return calculators.

### `calc_vadj_daily(daily_df, horizon)`
Calculate simplified daily volume-adjusted signals.
Path: rv = volume / median → badjret = log_ret - beta * mkt → vadj0 = rv * badjret → winsorize → demean → lags

### `calc_vadj_intra(intra_df)`
Calculate intraday signals (present but unused in forecast).

### `vadj_fits(daily_df, intra_df, horizon, name, middate)`
Fit daily-only multi-lag regressions.
Path: split → regress daily lags → incremental coefs → returns vadj_b

### `calc_vadj_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline (daily only) with sector split.

---

## `vadj_old.py`

### `calc_vadj(full_df, horizon)`
Calculate legacy volume-adjusted signals (log volume / beta division).
Path: log(vol/median) * (log_ret / pbeta) → winsorize → demean → lags

### `calc_vadj_intra(full_df)`
Calculate legacy intraday volume-adjusted signals.

### `vadj_fits(daily_df, intra_df, full_df, horizon, name)`
Fit legacy regression using regress_alpha_intra + regress_alpha_daily.

### `calc_vadj_forecast(daily_df, intra_df, horizon)`
Main pipeline.

---

## `vadj_pos.py`

### `wavg(group)` / `wavg2(group)` / `wavg_ind(group)`
Weighted return calculators.

### `calc_vadj_daily(daily_df, horizon)`
Calculate position-sizing signals using sign-based direction.
Path: rv = volume / median → vadj0 = rv * sign(badjret) → winsorize → demean → lags

### `calc_vadj_intra(intra_df)`
Calculate intraday position-sizing signals.
Path: wavg2 → rv_i * sign(badjret) → winsorize → demean

### `vadj_fits(daily_df, intra_df, horizon, name, middate)`
Fit 6 hourly intraday + daily lag coefficients with diagnostic output.
Path: regress intra → map to 6 periods → regress daily lags → incremental coefs → print distribution → returns vadj_b

### `calc_vadj_forecast(daily_df, intra_df, horizon, middate)`
Main pipeline with Energy sector split.
Path: calc_vadj_daily → calc_vadj_intra → merge → vadj_fits per sector → concat

---

# Salamander Module

---

## `salamander/bsim.py`

No public functions (script with argparse + simulation loop).

---

## `salamander/calc.py`

### `calc_vol_profiles(full_df)`
Calculate intraday volume participation profiles at 15-min intervals.
Path: unstack → between_time → rolling → merge

### `calc_price_extras(daily_df)`
Calculate derived volatility and volume ratio metrics.

### `calc_forward_returns(daily_df, horizon)`
Calculate forward cumulative returns at multiple horizons.
Path: groupby gvkey → rolling sum → shift(-h) → returns cum_ret{h}

### `winsorize(data, std_level=5)`
Cap outliers at N standard deviations.

### `winsorize_by_date(data)` / `winsorize_by_ts(data)` / `winsorize_by_group(data, group)`
Apply winsorization within date/timestamp/custom groups.

### `rolling_ew_corr_pairwise(df, halflife)`
Exponentially-weighted pairwise correlations.

### `push_data(df, col)`
Shift column forward 1 period.
Path: unstack → shift(-1) → stack → merge → returns df with col_n

### `lag_data(daily_df)`
Lag entire DataFrame by 1 period.

### `calc_med_price_corr(daily_df)`
Placeholder (not implemented).

### `calc_resid_vol(daily_df)`
Calculate Barra residual volatility.
Path: rolling 20-day var → sqrt

### `calc_factor_vol(factor_df)`
Exponentially-weighted factor covariance matrix.

### `create_z_score(daily_df, name)`
Standardize to z-score within date cross-section.

### `calc_factors(daily_df, barraOnly=False)`
Decompose returns into factor + residual via WLS.
Path: groupby date → factorize() → return (daily_df, factorRets_df)

### `calc_intra_factors(intra_df, barraOnly=False)`
Decompose intraday returns into factor + residual.

### `factorize(loadings_df, returns_df, weights_df, indwgt)`
Estimate factor returns via WLS (lmfit).
Path: minimize(fcn2min) → extract factor returns + residuals

### `fcn2min(params, x, data)`
Objective for lmfit minimization.

### `mkt_ret(group)`
Market-cap weighted average return.

---

## `salamander/change_hl.py`

### `main()`
Fix date index format in HL HDF5 files.

---

## `salamander/change_raw.py`

### `add_mktcap(uni_df, price_df, start, end, out_dir)`
Query and add market cap data to price_df.
Path: SQL query Capital IQ → merge → overwrite CSV

### `add_sedol(uni_df, start, end, out_dir)`
Query and add SEDOL identifiers to uni_df.
Path: SQL query → merge → overwrite CSV

### `main(start_s, end_s, data_dir)`
Process single raw data directory and add missing columns.

---

## `salamander/check_all.py`

No public functions (inspection script).

---

## `salamander/check_hl.py`

No public functions (validation script).

---

## `salamander/gen_alpha.py`

### `main()`
Extract alpha signals from HDF5 and save as CSV.
Path: glob all.*.h5 → filter dates → read → extract hl → save CSV

---

## `salamander/gen_dir.py`

No public functions (directory structure creation script).

---

## `salamander/gen_hl.py`

No public functions (HL generation orchestrator via hl_csv).

---

## `salamander/get_borrow.py`

### `get_borrow(locates_dir)`
Consolidate weekly stock lending files into single CSV.
Path: glob Historical_Avail → read → rename → concat → save borrow.csv

---

## `salamander/hl.py`

### `calc_hl_daily(full_df, horizon)`
Calculate high-low ratio with industry-demeaned lags.
Path: close / sqrt(high * low) → winsorize → demean → lag N → returns hl*_B_ma

### `hl_fits(daily_df, full_df, horizon, name)`
Fit HL coefficients using lagged regression.
Path: regress_alpha_daily lags → compute coef decay → populate hl forecast

### `calc_hl_forecast(daily_df, horizon)`
Main pipeline with sector split.
Path: calc_hl_daily → hl_fits(sector 10) → hl_fits(non-sector)

### `get_hl(start_s, end_s)`
Entry point: load data, calculate, save HDF5.
Path: load uni/barra/price → merge → calc_forward_returns → calc_hl_forecast → to_hdf

---

## `salamander/hl_csv.py`

### `calc_hl_daily(full_df, horizon)`
Calculate high-low ratio with industry-demeaned lags.

### `hl_fits(daily_df, full_df, horizon, name, reg_st, reg_ed, out_dir)`
Fit HL with windowed regression.

### `calc_hl_forecast(daily_df, horizon, reg_st, reg_ed, output_dir)`
Calculate HL with sector fitting, return (full_df, coef_df).

### `six_months_before(date_s)`
Calculate previous 6-month period boundary.

### `get_hl(start_s, end_s, dir)`
Entry point: load rolling CSV data, fit, save.
Path: load 3 periods → merge → calc_hl_forecast → truncate → to_hdf

---

## `salamander/loaddata.py`

### `load_mus(mdir, fcast, start, end)`
Load alpha forecasts from CSV files.
Path: glob alpha files → filter dates → read → concat

### `load_cache(start, end, data_dir, cols=None)`
Load cached market data from HDF5.
Path: glob all.*.h5 → filter dates → read → append

### `load_factor_cache(start, end, data_dir)`
Load cached factor/alpha signals from HDF5.

### `load_locates(uni_df, start, end, locates_dir)`
Load short borrow availability from CSV.
Path: read borrow.csv → filter → merge sedol/symbol → ffill → set_index

---

## `salamander/loaddata_sql.py`

### `get_uni(start, end, lookback, uni_size=1400)`
Build stock universe from Capital IQ + factor databases.
Path: SQL queries → filter price/liquidity/sector/mktcap → exclude biotech (GICS 3520) → rank by mktcap → save CSV

### `load_barra(uni_df, start, end)`
Load Barra risk factors from MySQL.
Path: SQL queries → strip gvkey suffix (last 3 chars) → merge → set_index → save CSV

### `load_price(uni_df, start, end)`
Load daily OHLCV and calculate returns/volatility metrics.
Path: SQL queries → calc ret/log_ret/overnight/volat → save CSV

### `ret(df)` / `log_ret(df)` / `overnight_ret(df)` / `overnight_log_ret(df)` / `today_ret(df)` / `today_log_ret(df)`
Return calculation helpers (simple, log, overnight, intraday variants).

### `med_volume(df, days)` / `volat(df, days)` / `overnight_volat(df, days)` / `today_volat(df, days)`
Rolling median volume and volatility calculators.

---

## `salamander/mktcalendar.py`

No public functions. Exports `TDay` (trading day offset constant).

---

## `salamander/opt.py`

### `init()`
Initialize global optimization arrays.

### `optimize()`
Main portfolio optimization via scipy trust-constr.
Path: getUntradeable() → scipy.optimize.minimize → marginal utilities → return (target, dutil, ...)

### `getUntradeable()`
Partition into tradeable/untradeable (bound range < $10).

### `setupProblem_scipy(...)`
Configure optimization: bounds, linear constraints, nonlinear capital constraint.

### `objective(target, ...)` / `objective_detail(target, ...)` / `objective_grad(target, ...)`
Portfolio utility, detailed breakdown, and analytical gradient.

### `slippageFuncAdv(target, ...)` / `slippageFunc_grad(target, ...)`
Market impact cost and gradient (I=volatility, J=participation model).

### `costsFunc(target, ...)` / `costsFunc_grad(target, ...)`
Execution cost and gradient.

### `constrain_by_capital(target, ...)` / `constrain_by_capital_grad(target, ...)`
Capital constraint: sum(abs(target)) <= max_sumnot.

### `constrain_by_trdnot(target, ...)`
Turnover constraint [unused].

### `printinfo(target, ...)`
Print optimization summary.

---

## `salamander/osim.py`

### `objective(weights)`
Objective for forecast weight optimization via simulated execution.
Path: simulate → compute Sharpe - diversity penalty → return float

---

## `salamander/qsim.py`

No public functions (script-based simulation).

---

## `salamander/regress.py`

### `plot_fit(fits_df, name)`
Visualize regression coefficients with error bars.

### `extract_results(results, indep, horizon)`
Extract params/bse/tvalues from statsmodels result.

### `get_intercept(daily_df, horizon, name, middate=None)`
Extract regression intercepts across horizons (3-fold median).

### `regress_alpha(results_df, indep, horizon, median=False, rtype='daily', intercept=True, start=None, end=None)`
Main regression dispatcher (daily/intra/dow, optional 3-fold validation).

### `regress_alpha_daily(daily_df, indep, horizon, intercept=True)`
Cross-sectional daily WLS (sqrt(mdvp) weights).

### `regress_alpha_intra_eod(intra_df, indep)`
Intraday alpha vs end-of-day returns (per hourly timeslice).

### `regress_alpha_intra(intra_df, indep, horizon)`
Intraday forward regression (multi-bar horizon).

### `regress_alpha_dow(daily_df, indep, horizon)`
Day-of-week stratified regression.

---

## `salamander/show_borrow.py`

### `main()`
View borrow rate history for a specific security (--file, --sedol).

---

## `salamander/show_raw.py`

No public functions (direct execution inspection script).

---

## `salamander/simulation.py`

### `Portfolio.__init__(sec1mean, sec2mean, sec1vol, sec2vol, corr, rebalance_threshold)`
Initialize two-security portfolio with rebalancing threshold.

### `Portfolio.Brownian(periods)`
Generate correlated standard Brownian motion paths.
Path: multivariate normal → cumsum → return array

### `Portfolio.GBM(W, T)`
Generate geometric Brownian motion prices.
Path: Cholesky → drift + diffusion → exp → return array

### `Portfolio.PriceMove(periods)`
Generate complete price path (Brownian + GBM).

### `Portfolio.Simulate(paths, tcost, periods, seed)`
Run Monte Carlo simulation with visualization.

### `Portfolio.Rebalance(pricemovements, tcost, periods)`
Simulate rebalancing and compute transaction costs.
Path: iterate days → update prices → check drift → trade if threshold → return (tradeTotal, costPer, nRebalance, decreaseReturn)

### `Portfolio.reset()`
Reset to initial state.

### `Portfolio.updatePrices(newPrices)` / `Portfolio.updateHoldings()`
Update for new prices / rebalance to 50/50 target.

### `Portfolio.decreaseReturn(pricemovements, tcost, periods)`
Compute annualized return reduction from transaction costs.

### `Portfolio.Tests(paths, tcost, periods, step, seed)`
Test Monte Carlo convergence at step intervals.

### `Portfolio.updateCorr(corr)` / `Portfolio.updateSec1Vol(vol)` / `Portfolio.updateThreshold(t)` / `Portfolio.updateSec1Mean(mean)`
Parameter update methods.

### `Portfolio.solveCorr(...)` / `Portfolio.solveSec1Vol(...)` / `Portfolio.solveSec1Mean(...)` / `Portfolio.solveThreshold(...)`
Sensitivity analysis sweeps (correlation, vol, return, threshold).

### `main()`
CLI dispatcher to Portfolio methods.

---

## `salamander/ssim.py`

No public functions (script-based full lifecycle simulation).

---

## `salamander/util.py`

### `merge_barra_data(price_df, barra_df)`
Merge Barra factors with prices, lagging by 1 day (gvkey-based index).

### `remove_dup_cols(result_df)`
Remove `_dead` suffixed columns.

### `merge_intra_eod(daily_df, intra_df)`
Merge EOD (16:00) bars with daily data.

### `merge_intra_data(daily_df, intra_df)`
Left-join daily into intraday on (date, gvkey).

### `filter_expandable(df)` / `filter_pca(df)`
Universe filters (expandable / large-cap > $10B).

### `dump_hd5(result_df, name)` / `dump_all(results_df)` / `dump_alpha(results_df, name)` / `dump_prod_alpha(results_df, name, outputfile)` / `dump_daily_alpha(results_df, name)`
Export functions (HDF5, per-timestamp CSV, production CSV, daily-to-intraday replication).

### `df_dates(df)`
Extract date range string (YYYYMMDD-YYYYMMDD) from index.

### `merge_daily_calcs(full_df, result_df)` / `merge_intra_calcs(full_df, result_df)`
Left-merge new calculated columns (daily / intraday).

### `get_overlapping_cols(df1, df2)`
Get columns in df1 NOT in df2 (misleading name).

### `load_merged_results(fdirs, start, end, cols=None)` / `load_all_results(fdir, start, end, cols=None)`
Load and merge alpha result CSVs from directories.
