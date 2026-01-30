# Documentation Plan: regress.py

## Priority: HIGH
## Status: COMPLETE
## Estimated Scope: Medium (242 lines)

## Overview

`regress.py` implements regression analysis for alpha fitting:
- Weighted Least Squares (WLS) regression
- Coefficient estimation
- Statistical significance testing

## Documentation Tasks

### 1. Module-Level Documentation
- [x] Module docstring already exists (comprehensive)
- [x] Documents WLS approach and rationale
- [x] Includes mathematical formulation notes

### 2. Function Documentation
- [x] `plot_fit()` - Visualize coefficient decay across horizons
- [x] `extract_results()` - Extract statistics from regression results
- [x] `get_intercept()` - Extract intercepts for time-series analysis
- [x] `regress_alpha()` - Main regression dispatcher with median option
- [x] `regress_alpha_daily()` - Cross-sectional daily regression
- [x] `regress_alpha_intra_eod()` - Intraday EOD prediction regression
- [x] `regress_alpha_intra()` - Intraday forward-looking regression
- [x] `regress_alpha_dow()` - Day-of-week stratified regression

### 3. Mathematical Documentation
- [x] WLS weight specification documented (mdvp^0.5)
- [x] Coefficient interpretation explained in each function
- [x] T-statistics and standard errors extraction documented

### 4. Usage Examples
- [x] Examples added to daily and dow regression docstrings
- [x] Input format expectations documented in each function

## Dependencies
- numpy, scipy
- statsmodels (for WLS regression)

## Completion Notes
- Completed 2026-01-30
- All 8 functions now have comprehensive docstrings
- Module docstring was already comprehensive
- Documented all regression types: daily, intraday, day-of-week
- Explained WLS weighting methodology (mdvp^0.5 for market cap balance)
