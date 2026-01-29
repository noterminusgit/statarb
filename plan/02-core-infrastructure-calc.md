# Documentation Plan: calc.py

## Priority: HIGH
## Status: COMPLETE
## Estimated Scope: Medium (400 lines)

## Overview

`calc.py` handles core calculations for the trading system:
- Forward return calculations
- Volume profile analysis
- Data winsorization
- Barra factor computations

## Documentation Tasks

### 1. Module-Level Documentation
- [x] Add comprehensive module docstring (already existed)
- [x] Document mathematical formulas used
- [x] Add usage examples with sample data

### 2. Function Documentation
- [x] `calc_forward_returns()` - Forward return calculation with horizons
- [x] `calc_vol_profiles()` - Volume distribution analysis
- [x] `winsorize()` - Outlier handling methodology
- [x] `calc_factors()` / `calc_intra_factors()` - Factor decomposition
- [x] `factorize()` - WLS regression for factor returns
- [x] `fcn2min()` - Optimization objective function
- [x] `calc_price_extras()` - Volatility and volume ratios
- [x] `winsorize_by_date()` / `winsorize_by_ts()` / `winsorize_by_group()`
- [x] `rolling_ew_corr_pairwise()` - Exponentially-weighted correlations
- [x] `push_data()` / `lag_data()` - Time shifting operations
- [x] `calc_resid_vol()` - Barra residual volatility
- [x] `calc_factor_vol()` - Factor covariance matrix
- [x] `create_z_score()` - Cross-sectional standardization
- [x] `mkt_ret()` - Market-cap weighted returns
- [x] `calc_med_price_corr()` - Placeholder documented

### 3. Mathematical Documentation
- [x] Document winsorization methodology and thresholds
- [x] Document forward return horizon options
- [x] Document Barra factor mathematics
- [x] Document WLS regression with market cap weighting
- [x] Document exponentially-weighted covariance calculations

### 4. Input/Output Specifications
- [x] Document expected DataFrame structures
- [x] Document column naming conventions
- [x] Document return value formats
- [x] Document MultiIndex requirements (date, sid) and (iclose_ts, sid)

## Completion Summary

All 20 functions now have comprehensive docstrings covering:
- Purpose and algorithm description
- Args with expected DataFrame columns
- Returns with output format
- Notes on usage, dependencies, and caveats
- Mathematical formulations where relevant
- Cross-references to related functions

## Dependencies
- pandas, numpy (core data structures)
- scipy.stats (statistical functions)
- pandas.stats.moments (ewmcorr, ewmcov - deprecated API)
- lmfit (WLS regression via minimize)
- util (custom utility functions)

## Notes
- Core module used by all 34 alpha strategy files
- Mathematical accuracy is critical for portfolio construction
- Module docstring already comprehensive (existed prior)
