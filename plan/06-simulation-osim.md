# Documentation Plan: osim.py

## Priority: HIGH
## Status: Pending
## Estimated Scope: Medium (298 lines)

## Overview

`osim.py` is the order-level execution simulator:
- Order fill simulation (VWAP, mid, close)
- Execution slippage modeling
- Order-by-order tracking

## Documentation Tasks

### 1. Module-Level Documentation
- [ ] Enhance existing docstring with methodology
- [ ] Document fill strategy options
- [ ] Add execution flow diagram (text-based)

### 2. CLI Parameters Documentation
- [ ] `--fill` - Fill strategy (vwap/mid/close)
- [ ] `--slipbps` - Slippage in basis points
- [ ] All other CLI parameters

### 3. Function Documentation
- [ ] `simulate_fills()` - Fill simulation logic
- [ ] `calculate_slippage()` - Slippage calculation
- [ ] `track_orders()` - Order tracking
- [ ] Document all public functions

### 4. Fill Strategy Documentation
- [ ] VWAP fill methodology
- [ ] Mid-price fill methodology
- [ ] Close price fill methodology
- [ ] Market impact modeling

## Dependencies
- loaddata.py for data
- opt.py for target positions

## Notes
- Used for execution analysis
- Important for realistic slippage estimation
