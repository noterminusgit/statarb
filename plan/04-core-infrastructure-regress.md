# Documentation Plan: regress.py

## Priority: HIGH
## Status: Pending
## Estimated Scope: Medium (242 lines)

## Overview

`regress.py` implements regression analysis for alpha fitting:
- Weighted Least Squares (WLS) regression
- Coefficient estimation
- Statistical significance testing

## Documentation Tasks

### 1. Module-Level Documentation
- [ ] Add module docstring explaining regression methodology
- [ ] Document the WLS approach and rationale
- [ ] Add mathematical formulation

### 2. Function Documentation
- [ ] `fit_wls()` - WLS regression fitting
- [ ] `estimate_coefficients()` - Coefficient estimation
- [ ] `calc_statistics()` - Statistical metric calculation
- [ ] Document all public functions

### 3. Mathematical Documentation
- [ ] Document WLS weight specification
- [ ] Document coefficient interpretation
- [ ] Document t-statistics and p-values

### 4. Usage Examples
- [ ] Add example of fitting alpha factors to returns
- [ ] Document expected input format

## Dependencies
- numpy, scipy
- statsmodels (if used)

## Notes
- Critical for alpha generation
- Statistical accuracy is essential
