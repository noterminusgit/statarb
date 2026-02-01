# Documentation Plan: Beta-Adjusted Order Flow Strategies

## Priority: MEDIUM
## Status: In Progress (1/9 complete)
## Estimated Scope: 9 files

## Files Covered
1. `bd.py` (195 lines) - Beta-adjusted order flow ✓ COMPLETE
2. `bd1.py` - Simplified BD variant
3. `bd_intra.py` - Intraday BD
4. `badj_multi.py` - Multi-period beta-adjusted
5. `badj_intra.py` - Intraday beta-adjusted
6. `badj_both.py` - Combined beta-adjusted approach
7. `badj_dow_multi.py` - Day-of-week variant
8. `badj2_multi.py` - Alternative BD implementation
9. `badj2_intra.py` - Alternative intraday BD

## Overview

Beta-adjusted strategies implement:
- Market-neutral order flow signals
- Beta-adjusted returns analysis
- Multi-period signal combination

## Documentation Tasks

### 1. Strategy Family Overview
- [ ] Create overview document explaining beta adjustment
- [ ] Document order flow signal methodology
- [ ] Document market neutrality approach

### 2. Per-File Documentation (for each file)
- [x] bd.py: Add module docstring with strategy description
- [x] bd.py: Document signal calculation formula
- [x] bd.py: Document beta calculation/source
- [x] bd.py: Document parameters and tuning options
- [ ] Remaining 8 files: Add documentation

### 3. Variant Documentation
- [ ] Document `bd` vs `badj` naming distinction
- [ ] Document when to use each variant
- [ ] Document day-of-week effects (dow variant)

### 4. Mathematical Documentation
- [ ] Beta estimation methodology
- [ ] Market return adjustment formula
- [ ] Signal normalization

## Notes
- Largest strategy family (9 files)
- Important to clarify naming conventions
