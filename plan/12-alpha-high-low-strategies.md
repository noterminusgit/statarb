# Documentation Plan: High-Low Mean Reversion Strategies

## Priority: MEDIUM
## Status: In Progress (5/6 complete)
## Estimated Scope: 6 files

## Files Covered
1. `hl.py` - Daily high-low mean reversion [✓ COMPLETE]
2. `hl_intra.py` - Intraday high-low [✓ COMPLETE]
3. `qhl_intra.py` - Quote-based intraday high-low [✓ COMPLETE]
4. `qhl_both_i.py` - High-low with time-varying intraday coefficients [✓ COMPLETE]
5. `qhl_multi.py` - Multiple timeframe high-low
6. `qhl_both.py` - Combined high-low approach

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

#### hl_intra.py [COMPLETE]
- [✓] Add module docstring with strategy description
- [✓] Document signal calculation formula (hlC = iclose/sqrt(dhigh*dlow))
- [✓] Document parameters and tuning options (horizon, sector_name)
- [✓] Document expected signal characteristics (range, turnover, mean reversion)
- [✓] Document all 3 functions with detailed docstrings
- [✓] Document key differences from daily hl.py strategy
- [✓] Document intraday-specific aspects (timing, updates, use case)
- [✓] Document bug on line 46 (undefined lag variable)

#### qhl_intra.py [COMPLETE]
- [✓] Add module docstring with strategy description
- [✓] Document signal calculation formula (qhlC = iclose/sqrt(qhigh*qlow))
- [✓] Document parameters and tuning options (horizon, freq, middate, sector_name)
- [✓] Document expected signal characteristics (quote-based vs trade-based)
- [✓] Document all 3 functions with detailed docstrings
- [✓] Document quote-based data distinction (qhigh/qlow vs dhigh/dlow)
- [✓] Document time-of-day coefficient structure (6 hourly periods)
- [✓] Document sector-specific modeling (Energy vs non-Energy)

#### qhl_multi.py [PENDING]
- [ ] Add module docstring with strategy description
- [ ] Document signal calculation formula
- [ ] Document parameters and tuning options
- [ ] Document expected signal characteristics

#### qhl_both.py [COMPLETE]
- [✓] Add module docstring with strategy description
- [✓] Document signal calculation formula
- [✓] Document parameters and tuning options
- [✓] Document expected signal characteristics

- [✓] Document all 4 functions with detailed docstrings
- [✓] Document "both" concept (daily lagged + intraday real-time signals)
- [✓] Document coefficient fitting methodology (regression-based weights)
- [✓] Document sector-specific modeling (Energy vs non-Energy)
- [✓] Document differential coefficient weighting structure
#### qhl_both_i.py [COMPLETE]
- [✓] Add module docstring with strategy description
- [✓] Document signal calculation formula (qhl_b with time-varying intraday coef)
- [✓] Document parameters and tuning options (horizon, middate, freq, sector_name)
- [✓] Document expected signal characteristics (time-of-day specific mean reversion)
- [✓] Document all 4 functions with detailed docstrings
- [✓] Document key difference from qhl_both.py (time-varying vs constant intraday coef)
- [✓] Document time bucket structure (6 hourly periods: 09:30-15:59)
- [✓] Document sector-specific modeling (Energy vs non-Energy separate fits)

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
