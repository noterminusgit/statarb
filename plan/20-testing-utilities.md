# Documentation Plan: Testing & Utility Scripts

## Priority: MEDIUM
## Status: Pending
## Estimated Scope: 8 files

## Files Covered

### Testing Files
1. `bigsim_test.py` (354 lines) - Integration testing **NO DOCSTRING**
2. `osim2.py` (225 lines) - Alternative order sim **NO DOCSTRING**
3. `osim_simple.py` (110 lines) - Simplified order sim **NO DOCSTRING**

### Utility Files
4. `dumpall.py` (37 lines) - Data export utility
5. `readcsv.py` (10 lines) - CSV reading stub
6. `slip.py` (22 lines) - Slippage calculations
7. `factors.py` (11 lines) - Factor enumeration
8. `rev.py` (93 lines) - Unknown purpose

## Documentation Tasks

### 1. Testing Documentation

#### bigsim_test.py
- [ ] **INVESTIGATE** test coverage and purpose
- [ ] Add module docstring
- [ ] Document test scenarios
- [ ] Document how to run tests

#### osim2.py
- [ ] **INVESTIGATE** differences from osim.py
- [ ] Add module docstring
- [ ] Document when to use this variant

#### osim_simple.py
- [ ] **INVESTIGATE** simplification details
- [ ] Add module docstring
- [ ] Document use cases

### 2. Utility Documentation

#### dumpall.py
- [ ] Document data export functionality
- [ ] Document output format
- [ ] Document CLI usage

#### readcsv.py
- [ ] Document CSV reading interface
- [ ] May be a stub - clarify purpose

#### slip.py
- [ ] Document slippage calculation formula
- [ ] Document parameters
- [ ] Document usage in other modules

#### factors.py
- [ ] Document Barra factor enumeration
- [ ] Document factor codes

#### rev.py
- [ ] **INVESTIGATE** purpose (reversal strategy?)
- [ ] Add appropriate documentation

## Investigation Required
- [ ] Determine purpose of osim2 vs osim
- [ ] Determine if bigsim_test is actively used
- [ ] Clarify rev.py functionality

## Notes
- Several files need investigation before documentation
- Testing infrastructure should be documented for maintenance
