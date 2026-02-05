# Code Quality Improvements

## Priority: MEDIUM
Files: 6 | Estimated effort: 4-6 hours

## Objective
Fix hard-coded paths, improve error handling, and enhance code maintainability in utility scripts.

## Tasks

### Task 1: Fix Hard-Coded Paths (2 files)
**Files:** salamander/change_hl.py, salamander/show_borrow.py

**Current Issues:**
- `change_hl.py`: Hard-coded path `./all/all.20040101-20040630.h5`
- `show_borrow.py`: Hard-coded path `./data/locates/borrow.csv` and SEDOL `2484088`

**Actions:**
- [ ] Add argparse CLI arguments for file paths and parameters
- [ ] Make scripts flexible for different dates/SEDOLs
- [ ] Update docstrings with usage examples
- [ ] Test with different input values

**Subagent Instructions:**
```
Fix hard-coded paths in salamander utility scripts:
1. Read salamander/change_hl.py and salamander/show_borrow.py
2. Replace hard-coded paths with argparse arguments
3. Add --file, --dir, --sedol CLI parameters as appropriate
4. Update module docstrings with new usage examples
5. Ensure backward compatibility (use hard-coded as defaults)
6. Commit: "Add CLI arguments to salamander utility scripts"
7. Push to remote
```

### Task 2: Improve Error Handling (4 files)
**Files:** loaddata.py, calc.py, regress.py, opt.py

**Current Issues:**
- Missing try/except in critical calculation functions
- Silent failures on data quality issues
- Minimal validation of input parameters

**Actions:**
- [ ] Add data quality checks to loaddata.py functions (check for NaN, inf, empty DataFrames)
- [ ] Add parameter validation to calc.py (winsorize limits, valid ranges)
- [ ] Add regression validation in regress.py (check for sufficient observations)
- [ ] Add optimization bounds checking in opt.py (verify feasible constraints)
- [ ] Add informative error messages with context

**Subagent Instructions:**
```
Improve error handling in core modules:
1. Identify critical functions in loaddata.py, calc.py, regress.py, opt.py
2. Add data quality checks (NaN, inf, empty, out-of-range)
3. Add try/except blocks with informative error messages
4. Validate input parameters (ranges, types, non-empty)
5. Use logging for warnings instead of silent failures
6. Focus on user-facing functions and data loading paths
7. Don't break existing behavior - only add safety checks
8. Commit: "Add error handling and validation to core modules"
9. Push to remote
```

## Success Criteria
- [ ] All hard-coded paths replaced with CLI arguments
- [ ] Key functions have input validation
- [ ] Data quality issues raise informative errors
- [ ] All changes tested (at minimum, syntax check)
- [ ] Documentation updated
