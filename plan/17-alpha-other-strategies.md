# Documentation Plan: Other Specialized Strategies

## Priority: MEDIUM
## Status: Pending
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
- [ ] Document gap trading methodology
- [ ] Document overnight return handling
- [ ] Document signal generation

### 2. PCA Generators
- [ ] Document PCA signal generation process
- [ ] Document daily vs. general approach
- [ ] Document integration with pca.py

### 3. Momentum Strategy (mom_year.py)
- [ ] Document annual momentum calculation
- [ ] Document lookback period
- [ ] Document academic basis

### 4. Borrow/Locate Strategies
- [ ] Document equity borrow signals (ebs.py)
- [ ] Document hard-to-borrow indicators (htb.py)
- [ ] Document short availability impact

### 5. Unknown Strategy (rrb.py)
- [ ] **INVESTIGATE** purpose and methodology
- [ ] Document findings
- [ ] Add appropriate docstring

## Dependencies
- LOCATES_BASE_DIR for borrow data
- Price data for gap calculations

## Notes
- Diverse set of specialized strategies
- rrb.py requires investigation
