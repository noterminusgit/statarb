# Documentation Plan: pca.py

## Priority: HIGH
## Status: COMPLETE
## Estimated Scope: Medium (307 lines)

## Overview

`pca.py` implements PCA decomposition for market-neutral returns:
- Principal component analysis of returns
- Market factor extraction
- Residual return calculation

## Documentation Tasks

### 1. Module-Level Documentation
- [x] Enhance existing docstring with PCA methodology
- [x] Document statistical approach
- [x] Document component interpretation

### 2. Function Documentation
- [x] `calc_pca_daily()` - Daily PCA model fitting with rolling windows
- [x] `calc_pca_intra()` - Intraday PCA decomposition for 30-min bars
- [x] `pca_fits()` - Regression fitting and forecast generation
- [x] `calc_pca_forecast()` - Universe-wide forecast wrapper
- [x] Document all public functions

### 3. Mathematical Documentation
- [x] Document eigenvalue decomposition methodology
- [x] Document variance explained metrics and interpretation
- [x] Document 5 components configuration and rolling window approach
- [x] Document residual calculation (actual - predicted returns)

### 4. Usage Documentation
- [x] Document as alpha signal generator with examples
- [x] Document integration with regression and simulation engines
- [x] Add comprehensive usage examples in all function docstrings

## Dependencies
- numpy, scipy
- sklearn (if used)

## Notes
- Key for market-neutral strategies
- Statistical methodology should be well-documented
