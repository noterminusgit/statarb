# Documentation Plan: High-Low Mean Reversion Strategies

## Priority: MEDIUM
## Status: Pending
## Estimated Scope: 6 files

## Files Covered
1. `hl.py` - Daily high-low mean reversion
2. `hl_intra.py` - Intraday high-low
3. `qhl_intra.py` - Quote-based intraday high-low
4. `qhl_multi.py` - Multiple timeframe high-low
5. `qhl_both.py` - Combined high-low approach
6. `qhl_both_i.py` - High-low with additional metrics

## Overview

High-low strategies implement mean reversion based on:
- Daily/intraday price ranges
- Deviation from high-low midpoints
- Multi-timeframe signal combination

## Documentation Tasks

### 1. Strategy Family Overview
- [ ] Create overview document explaining HL concept
- [ ] Document signal generation methodology
- [ ] Document expected holding periods

### 2. Per-File Documentation (for each file)
- [ ] Add module docstring with strategy description
- [ ] Document signal calculation formula
- [ ] Document parameters and tuning options
- [ ] Document expected signal characteristics

### 3. Relationship Documentation
- [ ] Document when to use each variant
- [ ] Document timeframe appropriateness
- [ ] Document combination patterns

### 4. Signal Interpretation
- [ ] Document signal range and normalization
- [ ] Document expected turnover
- [ ] Document decay characteristics

## Mathematical Documentation
- [ ] High-low range calculation
- [ ] Mean reversion signal formula
- [ ] Decay/half-life parameters

## Notes
- Core strategy family
- Important to document differences between variants
