# Documentation Plan: qsim.py

## Priority: MEDIUM
## Status: Pending
## Estimated Scope: Medium (414 lines)

## Overview

`qsim.py` is the intraday 30-minute bar simulator:
- Intraday simulation at 30-minute intervals
- Bar-level execution and tracking
- Intraday alpha signal handling

## Documentation Tasks

### 1. Module-Level Documentation
- [ ] Enhance existing docstring with intraday specifics
- [ ] Document time slice methodology
- [ ] Document bar data requirements

### 2. CLI Parameters Documentation
- [ ] `--fcast` - Intraday forecast specification
- [ ] `--horizon` - Prediction horizon (bars)
- [ ] All other CLI parameters

### 3. Function Documentation
- [ ] `simulate_intraday()` - Intraday simulation loop
- [ ] `process_bar()` - Individual bar processing
- [ ] `calc_intraday_returns()` - Return calculation
- [ ] Document all public functions

### 4. Intraday-Specific Documentation
- [ ] Document 30-minute bar format
- [ ] Document market hours handling
- [ ] Document overnight position handling

## Dependencies
- loaddata.py (bar data loading)
- Intraday strategy files (qhl_*, badj_intra, etc.)

## Notes
- Requires intraday bar data
- More complex than daily simulation
