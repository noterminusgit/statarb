# Documentation Plan: ssim.py

## Priority: MEDIUM
## Status: Pending
## Estimated Scope: Medium (435 lines)

## Overview

`ssim.py` is the full lifecycle simulation engine:
- Complete position and cash tracking
- Multi-period state management
- Full P&L attribution

## Documentation Tasks

### 1. Module-Level Documentation
- [ ] Enhance existing docstring with lifecycle explanation
- [ ] Document state tracking methodology
- [ ] Document tracking bucket system

### 2. CLI Parameters Documentation
- [ ] Document all CLI parameters
- [ ] Document combined alpha specification

### 3. Function Documentation
- [ ] `run_lifecycle()` - Main lifecycle loop
- [ ] `track_positions()` - Position state tracking
- [ ] `track_cash()` - Cash flow tracking
- [ ] `attribute_pnl()` - P&L attribution
- [ ] Document all public functions

### 4. Lifecycle-Specific Documentation
- [ ] Document position lifecycle states
- [ ] Document cash management
- [ ] Document tracking bucket interpretation

## Dependencies
- All core modules
- Multiple strategy files

## Notes
- Most comprehensive simulator
- Used for full production analysis
