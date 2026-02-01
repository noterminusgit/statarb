# Documentation Plan: qsim.py

## Priority: HIGH
## Status: COMPLETE
## Estimated Scope: Medium (414 lines)

## Overview

`qsim.py` is the intraday 30-minute bar simulator:
- Intraday simulation at 30-minute intervals
- Bar-level execution and tracking
- Intraday alpha signal handling

## Documentation Tasks

### 1. Module-Level Documentation
- [x] Enhanced docstring with comprehensive intraday methodology
- [x] Documented 13 time slices per day (9:45-16:00)
- [x] Documented bar data requirements and incremental VWAP calculation
- [x] Added simulation methodology section
- [x] Added example usage and use cases

### 2. CLI Parameters Documentation
- [x] `--fcast` - Intraday forecast specification with format examples
- [x] `--horizon` - Prediction horizon in 30-minute bars
- [x] `--cond` - Conditioning variable for decile stratification
- [x] `--mult` - Alpha multiplier for position sizing
- [x] `--slipbps` - Slippage in basis points
- [x] `--vwap` - VWAP vs. iclose execution flag
- [x] All CLI parameters fully documented

### 3. Code Section Documentation
- [x] Data loading and 30-minute bar processing
- [x] Incremental VWAP calculation formula
- [x] Forecast mixing and combination
- [x] Position sizing with volume constraints (1% of bar)
- [x] Multi-horizon P&L calculation
- [x] Main simulation loop (bar-by-bar processing)
- [x] Decile ranking and stratification
- [x] Delta hedging methodology
- [x] Diagnostics section (forecast analysis, distributions)
- [x] Performance analysis (Sharpe ratios by horizon)
- [x] Stratification breakdowns (time-of-day, day-of-week, month, deciles)

### 4. Intraday-Specific Documentation
- [x] 30-minute bar format (bvolume, bvwap, iclose)
- [x] Market hours handling (9:30-16:00, 13 bars/day)
- [x] Time-of-day performance attribution
- [x] Incremental volume calculation (bvolume_d)
- [x] P&L aggregation buckets explained
- [x] Output files documented

## Dependencies
- loaddata.py (bar data loading)
- Intraday strategy files (qhl_*, badj_intra, etc.)

## Notes
- Requires intraday bar data
- More complex than daily simulation
