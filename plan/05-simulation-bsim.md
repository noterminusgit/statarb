# Documentation Plan: bsim.py

## Priority: HIGH
## Status: Pending
## Estimated Scope: Medium (388 lines)

## Overview

`bsim.py` is the daily rebalancing backtest engine:
- Main simulation driver for daily strategies
- Portfolio optimization integration
- Performance tracking and reporting

## Documentation Tasks

### 1. Module-Level Documentation
- [ ] Enhance existing docstring with detailed workflow
- [ ] Document simulation loop architecture
- [ ] Add example command-line usage

### 2. CLI Parameters Documentation
- [ ] `--start/--end` - Date range specification
- [ ] `--fcast` - Forecast specification format (name:multiplier:weight)
- [ ] `--kappa` - Risk aversion parameter
- [ ] `--maxnot` - Maximum notional constraints
- [ ] All other CLI parameters

### 3. Function Documentation
- [ ] `run_simulation()` - Main simulation loop
- [ ] `calculate_positions()` - Position calculation
- [ ] `track_performance()` - P&L tracking
- [ ] All helper functions

### 4. Output Documentation
- [ ] Document output metrics (Sharpe, returns, drawdown)
- [ ] Document log file format
- [ ] Document position output format

## Dependencies
- opt.py for optimization
- loaddata.py for data loading
- Strategy files for alpha signals

## Notes
- Primary backtest engine
- Entry point for most backtesting workflows
