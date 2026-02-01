# Documentation Plan: bsim.py

## Priority: HIGH
## Status: COMPLETE
## Estimated Scope: Medium (388 lines)

## Overview

`bsim.py` is the daily rebalancing backtest engine:
- Main simulation driver for daily strategies
- Portfolio optimization integration
- Performance tracking and reporting

## Documentation Tasks

### 1. Module-Level Documentation
- [x] Enhance existing docstring with detailed workflow
- [x] Document simulation loop architecture
- [x] Add example command-line usage

### 2. CLI Parameters Documentation
- [x] `--start/--end` - Date range specification
- [x] `--fcast` - Forecast specification format (dir:name:multiplier:weight)
- [x] `--kappa` - Risk aversion parameter
- [x] `--maxnot` - Maximum notional constraints
- [x] All other CLI parameters (horizon, mult, vwap, locates, earnings, slipnu, slipbeta, fast, exclude, maxforecast, nonegutil, daily, maxiter, maxdollars)

### 3. Function Documentation
- [x] `pnl_sum()` - Cumulative P&L calculation from log returns
- [x] Main simulation loop - Comprehensive inline comments throughout
- [x] Data loading section - Comments on cache loading and forecast merging
- [x] Optimization section - Comments on optimizer setup and execution
- [x] Position calculation - Comments on participation constraints

### 4. Output Documentation
- [x] Document output CSV columns (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, etc.)
- [x] Document output file naming convention
- [x] Document performance metrics tracked

## Dependencies
- opt.py for optimization
- loaddata.py for data loading
- Strategy files for alpha signals

## Notes
- Primary backtest engine
- Entry point for most backtesting workflows
