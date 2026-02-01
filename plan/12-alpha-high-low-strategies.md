# Documentation Plan: High-Low Mean Reversion Strategies

## Priority: MEDIUM
## Status: In Progress (1/6 complete)
## Estimated Scope: 6 files

## Files Covered
1. `hl.py` - Daily high-low mean reversion [✓ COMPLETE]
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
- [✓] Create overview document explaining HL concept (in hl.py module docstring)
- [✓] Document signal generation methodology
- [✓] Document expected holding periods

### 2. Per-File Documentation
#### hl.py [COMPLETE]
- [✓] Add module docstring with strategy description
- [✓] Document signal calculation formula (hl0 = close/sqrt(high*low))
- [✓] Document parameters and tuning options (horizon, lookback, sector_name)
- [✓] Document expected signal characteristics (range, turnover, decay)
- [✓] Document all 4 functions with detailed docstrings
- [✓] Document usage examples and integration patterns

#### hl_intra.py [PENDING]
- [ ] Add module docstring with strategy description
- [ ] Document signal calculation formula
- [ ] Document parameters and tuning options
- [ ] Document expected signal characteristics

#### qhl_intra.py [PENDING]
- [ ] Add module docstring with strategy description
- [ ] Document signal calculation formula
- [ ] Document parameters and tuning options
- [ ] Document expected signal characteristics

#### qhl_multi.py [PENDING]
- [ ] Add module docstring with strategy description
- [ ] Document signal calculation formula
- [ ] Document parameters and tuning options
- [ ] Document expected signal characteristics

#### qhl_both.py [PENDING]
- [ ] Add module docstring with strategy description
- [ ] Document signal calculation formula
- [ ] Document parameters and tuning options
- [ ] Document expected signal characteristics

#### qhl_both_i.py [PENDING]
- [ ] Add module docstring with strategy description
- [ ] Document signal calculation formula
- [ ] Document parameters and tuning options
- [ ] Document expected signal characteristics

### 3. Relationship Documentation
- [ ] Document when to use each variant
- [ ] Document timeframe appropriateness
- [ ] Document combination patterns

### 4. Signal Interpretation
- [✓] Document signal range and normalization (in hl.py)
- [✓] Document expected turnover (in hl.py)
- [✓] Document decay characteristics (in hl.py)

## Mathematical Documentation
- [✓] High-low range calculation (documented in hl.py)
- [✓] Mean reversion signal formula (documented in hl.py)
- [✓] Decay/half-life parameters (documented via horizon parameter)

## Notes
- Core strategy family
- Important to document differences between variants
