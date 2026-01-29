# Documentation Plan: Earnings & Valuation Strategies

## Priority: MEDIUM
## Status: Pending
## Estimated Scope: 3 files

## Files Covered
1. `eps.py` (192 lines) - Earnings surprise strategy
2. `target.py` (181 lines) - Price target miss strategy
3. `prod_tgt.py` (197 lines) - Target production

## Overview

Earnings and valuation strategies implement:
- Post-earnings announcement drift (PEAD)
- Price target deviation signals
- Analyst estimate surprise signals

## Documentation Tasks

### 1. Strategy Family Overview
- [ ] Document earnings surprise methodology
- [ ] Document price target analysis
- [ ] Document academic basis (PEAD literature)

### 2. Per-File Documentation (for each file)
- [ ] Add module docstring with strategy description
- [ ] Document surprise calculation formula
- [ ] Document signal generation logic
- [ ] Document parameters and tuning options

### 3. Data Requirements
- [ ] Document earnings data format
- [ ] Document price target data format
- [ ] Document earnings calendar handling

### 4. Event-Driven Documentation
- [ ] Document earnings window handling
- [ ] Document blackout period logic
- [ ] Document signal decay after events

## Dependencies
- Earnings data from EARNINGS_BASE_DIR
- Analyst estimates from ESTIMATES_BASE_DIR

## Notes
- Event-driven signals require careful timing
- PEAD is well-documented in academic literature
