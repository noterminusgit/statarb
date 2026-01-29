# Documentation Plan: bsim_weights.py

## Priority: CRITICAL
## Status: Pending
## Estimated Scope: Medium (372 lines)

## Overview

`bsim_weights.py` handles weight optimization for position sizing.
**Currently has NO module docstring - high priority.**

## Documentation Tasks

### 1. Module-Level Documentation (CRITICAL)
- [ ] Add comprehensive module docstring
- [ ] Document relationship to bsim.py
- [ ] Document weight optimization methodology

### 2. Purpose Clarification
- [ ] Investigate and document primary purpose
- [ ] Document when to use this vs. standard bsim.py
- [ ] Document input/output differences

### 3. Function Documentation
- [ ] Document all public functions
- [ ] Document weight calculation logic
- [ ] Document optimization constraints

### 4. Usage Documentation
- [ ] Add CLI usage examples
- [ ] Document typical use cases
- [ ] Document parameter combinations

## Investigation Required
- [ ] Determine if this is a variant of bsim or separate tool
- [ ] Identify differences from opt.py weighting
- [ ] Clarify production usage patterns

## Dependencies
- bsim.py (likely)
- opt.py
- loaddata.py

## Notes
- HIGH PRIORITY: No existing documentation
- May be used in production workflows
