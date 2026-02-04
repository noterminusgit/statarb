# Documentation Plan: Testing & Utility Scripts

## Priority: MEDIUM
## Status: COMPLETE
## Estimated Scope: 8 files

## Files Covered

### Testing Files
1. `bigsim_test.py` (354 lines) - Intraday rebalancing simulator **DOCUMENTED**
2. `osim2.py` (225 lines) - Order sim without weight optimization **DOCUMENTED**
3. `osim_simple.py` (110 lines) - Portfolio weight optimizer **DOCUMENTED**

### Utility Files
4. `dumpall.py` (37 lines) - Data export utility **DOCUMENTED**
5. `readcsv.py` (10 lines) - CSV debugging utility **DOCUMENTED**
6. `slip.py` (22 lines) - Order/execution merger **DOCUMENTED**
7. `factors.py` (11 lines) - Factor visualization tool **DOCUMENTED**
8. `rev.py` (93 lines) - Mean reversion alpha strategy **DOCUMENTED**

## Documentation Tasks

### 1. Testing Documentation

#### bigsim_test.py
- [x] **INVESTIGATE** test coverage and purpose
- [x] Add module docstring
- [x] Document test scenarios
- [x] Document how to run tests
- **Finding**: Despite the name, NOT a test suite - it's an intraday rebalancing simulator

#### osim2.py
- [x] **INVESTIGATE** differences from osim.py
- [x] Add module docstring
- [x] Document when to use this variant
- **Finding**: Same as osim.py but WITHOUT weight optimization - uses fixed weights

#### osim_simple.py
- [x] **INVESTIGATE** simplification details
- [x] Add module docstring
- [x] Document use cases
- **Finding**: Misleading name - this is a portfolio weight optimizer, not a simulator

### 2. Utility Documentation

#### dumpall.py
- [x] Document data export functionality
- [x] Document output format
- [x] Document CLI usage
- **Finding**: Comprehensive data export to HDF5 for all pipeline components

#### readcsv.py
- [x] Document CSV reading interface
- [x] May be a stub - clarify purpose
- **Finding**: Simple debugging utility for viewing CSV in key-value format

#### slip.py
- [x] Document slippage calculation formula
- [x] Document parameters
- [x] Document usage in other modules
- **Finding**: NOT slippage calculations - merges QuantBook order/execution data

#### factors.py
- [x] Document Barra factor enumeration
- [x] Document factor codes
- **Finding**: NOT factor enumeration - plots cumulative factor returns

#### rev.py
- [x] **INVESTIGATE** purpose (reversal strategy?)
- [x] Add appropriate documentation
- **Finding**: Mean reversion alpha strategy with industry demeaning

## Investigation Required
- [x] Determine purpose of osim2 vs osim
- [x] Determine if bigsim_test is actively used
- [x] Clarify rev.py functionality

## Investigation Findings

### Misleading File Names
Several files have misleading names that don't match their actual purpose:

1. **bigsim_test.py**: NOT a test suite
   - Actually: Intraday rebalancing simulator (15-min bars)
   - Should be renamed: isim.py (intraday simulator)

2. **osim_simple.py**: NOT a simplified simulator
   - Actually: Portfolio weight optimizer using OpenOpt
   - Should be renamed: weight_optimizer.py

3. **slip.py**: NOT slippage calculations
   - Actually: QuantBook order/execution merger for historical analysis
   - Appears to be a one-off analysis script from 2012

4. **factors.py**: NOT factor enumeration
   - Actually: Factor visualization/plotting tool
   - Should be renamed: plot_factor.py

### Production vs. Deprecated
- **bigsim_test.py**: Production code for intraday strategies (slow but functional)
- **osim2.py**: Production code for fast backtests with known weights
- **osim_simple.py**: Research tool for offline weight optimization
- **slip.py**: Deprecated - one-off analysis from 2012-02-08
- **rev.py**: Production alpha strategy (mean reversion)

## Notes
- All 8 files now fully documented
- Documentation reveals several naming inconsistencies
- Some files may be deprecated but kept for historical reference
