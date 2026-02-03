# Documentation Plan: Other Specialized Strategies

## Priority: MEDIUM
## Status: COMPLETED
## Estimated Scope: 7 files

## Files Covered
1. `c2o.py` (217 lines) - Close-to-open gap trading
2. `pca_generator.py` - PCA signal generation
3. `pca_generator_daily.py` - Daily PCA signals
4. `mom_year.py` (171 lines) - Annual momentum
5. `ebs.py` (220 lines) - Equity borrow signals
6. `htb.py` - Hard-to-borrow indicators
7. `rrb.py` - (Unknown - needs investigation)

## Documentation Tasks

### 1. Close-to-Open Strategy (c2o.py)
- [x] Document gap trading methodology
- [x] Document overnight return handling
- [x] Document signal generation

### 2. PCA Generators
- [x] Document PCA signal generation process
- [x] Document daily vs. general approach
- [x] Document integration with pca.py

### 3. Momentum Strategy (mom_year.py)
- [x] Document annual momentum calculation
- [x] Document lookback period (232-day lag, 20-day rolling)
- [x] Document academic basis

### 4. Borrow/Locate Strategies
- [x] Document equity borrow signals (ebs.py - actually analyst estimates)
- [x] Document hard-to-borrow indicators (htb.py - fee rates)
- [x] Document short availability impact

### 5. Residual Return Strategy (rrb.py)
- [x] **INVESTIGATED** Barra factor model residual mean reversion
- [x] Document findings and methodology
- [x] Add comprehensive docstrings

## Dependencies
- LOCATES_BASE_DIR for borrow data
- Price data for gap calculations

## Notes
- Diverse set of specialized strategies
- rrb.py requires investigation
