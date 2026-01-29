# Documentation Plan: Volume-Adjusted Strategies

## Priority: MEDIUM
## Status: Pending
## Estimated Scope: 5 files

## Files Covered
1. `vadj.py` (226 lines) - Volume-adjusted positioning
2. `vadj_multi.py` (204 lines) - Multi-period volume adjustment
3. `vadj_intra.py` - Intraday volume adjustment
4. `vadj_pos.py` (197 lines) - Volume-based position sizing
5. `vadj_old.py` - Legacy version (for reference)

## Overview

Volume-adjusted strategies implement:
- Volume-weighted signal adjustments
- Liquidity-aware position sizing
- Volume profile analysis

## Documentation Tasks

### 1. Strategy Family Overview
- [ ] Document volume adjustment rationale
- [ ] Document liquidity considerations
- [ ] Document position sizing methodology

### 2. Per-File Documentation (for each file)
- [ ] Add module docstring with strategy description
- [ ] Document volume calculation methodology
- [ ] Document adjustment formulas
- [ ] Document parameters and tuning options

### 3. Variant Documentation
- [ ] Document differences from legacy (vadj_old)
- [ ] Document position sizing approach (vadj_pos)
- [ ] Document multi-period lookback (vadj_multi)

### 4. Volume Profile Documentation
- [ ] Document ADV calculation
- [ ] Document volume normalization
- [ ] Document market impact considerations

## Notes
- Important for execution quality
- Interacts with optimization constraints
