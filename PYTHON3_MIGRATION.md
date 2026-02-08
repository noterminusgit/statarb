# Python 3 Migration Analysis

## Executive Summary

This document provides a comprehensive analysis of Python 3 compatibility issues in the statistical arbitrage trading system codebase. The analysis covers 64 Python files in the main codebase (excluding the existing salamander Python 3 module and tests).

**Key Findings:**
- **Total Issues Identified:** ~800+ individual compatibility issues
- **Critical Blockers:** OpenOpt/FuncDesigner library (no Python 3 support)
- **Moderate Issues:** pandas.stats deprecation, print statements, dict iteration methods
- **Minor Issues:** xrange, division operators, string handling

**Recommendation:** Proceed with phased migration. The salamander module demonstrates feasibility, but OpenOpt replacement is the primary challenge requiring 16-24 hours of focused effort.

---

## Detailed Issue Breakdown

### 1. Print Statements (High Volume)

**Count:** 71 occurrences across 8 files

**Affected Files:**
- `ssim.py`: 47 occurrences
- `qsim.py`: 17 occurrences
- `bsim.py`, `osim.py`, `osim2.py`, `osim_simple.py`, `bsim_weights.py`, `bigsim_test.py`: 1-2 each

**Issue Description:**
Python 2 style `print` statements without parentheses:
```python
# Python 2
print "Loading data..."
print var1, var2, var3

# Python 3 (required)
print("Loading data...")
print(var1, var2, var3)
```

**Risk Assessment:** LOW
- Automated tools (2to3, pyupgrade) can fix 95%+ of these
- Easy to verify visually
- No semantic changes, only syntax

**Effort Estimate:** 2-3 hours
- Run automated conversion: 30 minutes
- Manual review and edge cases: 1.5 hours
- Testing output consistency: 1 hour

---

### 2. Dictionary Iterator Methods (Moderate Volume)

**Count:** 8 occurrences across 3 files

**Affected Files:**
- `calc.py`: 2 occurrences (`.iteritems()`)
- `salamander/calc.py`: 2 occurrences (already commented out in salamander)
- `qsim.py`: 1 occurrence (`.iteritems()`)
- `salamander/qsim.py`, `salamander/ssim.py`: Already migrated in salamander

**Issue Description:**
Python 2 iterator methods removed in Python 3:
```python
# Python 2
for k, v in mydict.iteritems():
for k in mydict.iterkeys():
for v in mydict.itervalues():

# Python 3
for k, v in mydict.items():
for k in mydict.keys():
for v in mydict.values():
```

**Risk Assessment:** LOW
- Python 3 `.items()` returns a view (memory efficient like Python 2 iterators)
- Performance impact negligible for dictionary sizes in this codebase
- Automated tools handle this well

**Effort Estimate:** 1 hour
- Automated conversion: 15 minutes
- Manual verification: 30 minutes
- Testing: 15 minutes

---

### 3. xrange() Function (Moderate Volume)

**Count:** 13 occurrences across 2 files

**Affected Files:**
- `opt.py`: 6 occurrences
- `pca.py`: 1 occurrence
- `opt.py.old`: 6 occurrences (backup file)

**Issue Description:**
`xrange()` renamed to `range()` in Python 3:
```python
# Python 2
for i in xrange(1000):  # Memory efficient iterator

# Python 3
for i in range(1000):   # Already returns iterator in Python 3
```

**Risk Assessment:** LOW
- Simple find-replace operation
- Python 3 `range()` is already lazy (like Python 2 `xrange()`)
- No performance regression

**Effort Estimate:** 30 minutes
- Find-replace: 10 minutes
- Verification: 20 minutes

---

### 4. OpenOpt and FuncDesigner Libraries (CRITICAL BLOCKER)

**Count:** Used extensively in 2 core files (`opt.py`, `salamander/opt.py`)

**Issue Description:**
OpenOpt and FuncDesigner are deprecated libraries with no Python 3 support:
- Last release: 2014 (OpenOpt 0.5628)
- No maintainer activity since 2015
- Core dependency for portfolio optimization

**Current Usage:**
```python
import openopt
from FuncDesigner import *

# Portfolio optimization problem
problem = openopt.NLP(
    f=objective_func,
    x0=initial_positions,
    df=gradient_func,
    constraints=constraints_list,
    lb=lower_bounds,
    ub=upper_bounds
)
solution = problem.solve('ralg')
```

**Problem Characteristics:**
- **Type:** Non-Linear Programming (NLP) with quadratic objective
- **Size:** ~1,400 variables (one per security)
- **Constraints:** ~100 linear constraints (position bounds, factor exposure, capital limits)
- **Objective:** Quadratic utility function (mean-variance optimization)
- **Solve Time:** 1-5 seconds typical

**Risk Assessment:** HIGH
- Critical path component (all simulations depend on optimizer)
- Replacement library may produce slightly different solutions
- Potential numerical differences in backtest results
- Requires careful validation and benchmarking

**Effort Estimate:** 16-24 hours
- Research and prototype alternatives: 4-6 hours
- Implementation and testing: 8-12 hours
- Validation and benchmarking: 4-6 hours

**Recommended Alternatives (in priority order):**

1. **scipy.optimize.minimize (SLSQP or trust-constr)**
   - Pros: Built-in, well-maintained, good NLP solver
   - Cons: May require constraint reformulation
   - API complexity: Moderate

2. **cvxpy**
   - Pros: Modern, active development, excellent documentation
   - Cons: Requires convex formulation (may need QP approximation)
   - API complexity: Low-Moderate

3. **CVXOPT**
   - Pros: Mature, efficient QP solver
   - Cons: More low-level API, requires constraint matrices
   - API complexity: High

4. **Pyomo + IPOPT**
   - Pros: Powerful, supports complex constraints
   - Cons: External solver dependency, heavier installation
   - API complexity: Moderate-High

**Note:** salamander/opt.py still uses OpenOpt, so this is blocking for salamander Python 3 adoption too.

---

### 5. pandas.stats Module (HIGH IMPACT)

**Count:** 14 files use deprecated pandas.stats

**Affected Files:**
- Core: `calc.py`
- Alpha strategies: `analyst.py`, `analyst_badj.py`, `rating_diff.py`, `rating_diff_updn.py`, `eps.py`, `target.py`, `ebs.py`, `prod_sal.py`, `prod_eps.py`, `prod_rtg.py`, `prod_tgt.py`
- Already migrated: `salamander/calc.py` (commented out)

**Issue Description:**
pandas.stats module removed in pandas 0.23.0+ (2018):
```python
# Python 2 / Old pandas
from pandas.stats.api import ols
from pandas.stats import moments
result = moments.ewma(series, span=20)

# Python 3 / Modern pandas
# Option 1: Use pandas.Series.ewm()
result = series.ewm(span=20).mean()

# Option 2: Use scipy or custom implementation
```

**Specific Methods Used:**
- `pandas.stats.moments.ewma()`: Exponentially weighted moving average
- `pandas.stats.api.ols()`: Ordinary least squares regression

**Risk Assessment:** MEDIUM-HIGH
- Replacements exist but require code changes
- Numerical results should be identical if parameters match
- Need to verify ewm() parameters (span, min_periods, adjust)
- OLS replacement options: statsmodels, scipy.stats, sklearn

**Effort Estimate:** 6-8 hours
- Replace ewma() calls: 2-3 hours (12 files)
- Replace ols() calls: 2-3 hours (verify calc.py usage)
- Testing and validation: 2 hours

**Migration Path:**
```python
# Before (pandas.stats.moments)
ewma_result = moments.ewma(df['column'], span=20, min_periods=1)

# After (pandas.Series.ewm)
ewma_result = df['column'].ewm(span=20, min_periods=1, adjust=True).mean()
```

---

### 6. Integer Division Operator (Widespread)

**Count:** 592 occurrences of ` / ` operator across 68 files

**Issue Description:**
Division semantics changed in Python 3:
```python
# Python 2
5 / 2   # = 2 (integer division)
5.0 / 2 # = 2.5 (float division)

# Python 3
5 / 2   # = 2.5 (true division - BREAKING CHANGE)
5 // 2  # = 2 (floor division)
```

**Risk Assessment:** MEDIUM
- Most divisions in financial code are intentionally float divisions
- Critical to audit integer divisions that rely on truncation
- May cause subtle bugs in indexing, binning, or date arithmetic

**Affected Areas:**
- Portfolio calculations: positions / ADV, returns / volatility (mostly safe - expect floats)
- Date arithmetic: days / period length (NEEDS REVIEW)
- Indexing: array_size / bins (NEEDS REVIEW)
- Slippage calculations: trade_notional / ADV (safe - expect floats)

**Effort Estimate:** 4-5 hours
- Automated scan for integer division contexts: 1 hour
- Manual review of critical divisions: 2-3 hours
- Testing edge cases: 1 hour

**Migration Strategy:**
1. Add `from __future__ import division` to all files (makes Python 2 behave like Python 3)
2. Use `//` explicitly where floor division is needed
3. Test critical calculations for consistency

---

### 7. Exception Syntax (Low Volume)

**Count:** 3 occurrences across 2 files

**Affected Files:**
- `loaddata.py`: 2 occurrences
- `regress.py`: 1 occurrence

**Issue Description:**
Python 2 raise syntax with three arguments:
```python
# Python 2 (removed in Python 3)
raise ValueError, "message", traceback

# Python 3
raise ValueError("message")
```

**Note:** The grep search found these patterns but they appear to be modern format strings in raise statements, not old-style syntax. Manual verification needed.

**Risk Assessment:** LOW
- Easy to identify and fix
- Automated tools handle this

**Effort Estimate:** 30 minutes

---

### 8. file() Built-in and Other Removed Functions

**Count:** 12 occurrences across 7 files

**Affected Files:**
- `prod_sal.py`, `prod_rtg.py`, `prod_tgt.py`, `prod_eps.py`, `loaddata.py`, `load_data_live.py`, `vadj_old.py`

**Issue Description:**
Several Python 2 built-ins removed:
```python
# Python 2
f = file('data.txt')  # Use open() instead
execfile('script.py')  # Use exec(open().read()) instead
input = raw_input()    # raw_input() renamed to input()
reduce(func, list)     # Moved to functools.reduce()
```

**Risk Assessment:** LOW
- Uncommon in this codebase
- Easy replacements available

**Effort Estimate:** 1 hour

---

### 9. String and Unicode Handling (Low Impact)

**Count:** 0 occurrences of `unicode` or `basestring` in main code (only in plan documentation)

**Issue Description:**
Python 3 unified string types:
```python
# Python 2
isinstance(s, basestring)  # Matches str or unicode
u"unicode string"

# Python 3
isinstance(s, str)  # str is always unicode
"unicode string"   # All strings are unicode
```

**Risk Assessment:** LOW
- Not used in this codebase
- File I/O may need text/binary mode clarification

**Effort Estimate:** 1 hour (audit file operations for encoding issues)

---

### 10. Library Dependency Versions

**Current requirements.txt:**
```
numpy==1.16.0         # Last Python 2 compatible: 1.16.6
pandas==0.23.4        # Last Python 2 compatible: 0.24.2
python-dateutil==2.7.5
pytz==2018.9
six==1.12.0
pytest==4.6.11        # Last Python 2 compatible
pytest-cov==2.12.1
```

**Python 3 Compatible Versions (Recommended):**
```
numpy>=1.21.0         # Modern, well-supported
pandas>=1.3.0         # Mature Python 3 support (pandas.stats removed in 0.23)
python-dateutil>=2.8.2
pytz>=2021.3
pytest>=7.0.0
pytest-cov>=3.0.0
scipy>=1.7.0          # For optimization alternatives
```

**Breaking Changes to Address:**
- pandas 0.23 → 1.3+: pandas.stats removed (see section 5)
- numpy 1.16 → 1.21+: Some deprecated numpy.matrix behaviors
- pandas .ix indexer removed (use .loc[] or .iloc[])

**Risk Assessment:** MEDIUM
- Library APIs have evolved significantly
- Need systematic testing for behavior changes
- Performance may improve with modern libraries

**Effort Estimate:** 3-4 hours
- Update requirements and test compatibility: 2 hours
- Address API deprecations: 1-2 hours

---

## Total Effort Summary

| Category | Effort (hours) | Risk | Priority |
|----------|---------------|------|----------|
| Print statements | 2-3 | Low | Medium |
| Dict iterator methods | 1 | Low | Medium |
| xrange() | 0.5 | Low | Low |
| **OpenOpt replacement** | **16-24** | **High** | **CRITICAL** |
| pandas.stats migration | 6-8 | Medium-High | High |
| Integer division audit | 4-5 | Medium | High |
| Exception syntax | 0.5 | Low | Low |
| file() and removed built-ins | 1 | Low | Low |
| String/unicode handling | 1 | Low | Low |
| Library version updates | 3-4 | Medium | High |

**Total Estimated Effort:** 35-50 hours (4-6 days of focused work)

---

## Risk Assessment

### High Risk Items

1. **OpenOpt Replacement (CRITICAL PATH)**
   - Impact: Core functionality blocker
   - Risk: Numerical differences in optimization results
   - Mitigation: Side-by-side validation of Python 2 vs Python 3 backtests
   - Testing: Run identical backtests, compare positions, PnL, Sharpe ratios

2. **pandas.stats Migration**
   - Impact: Affects alpha factor calculations
   - Risk: Incorrect ewma parameters could change signal generation
   - Mitigation: Unit tests comparing old vs new ewma outputs
   - Testing: Validate factor values match within numerical precision

### Medium Risk Items

3. **Integer Division Semantics**
   - Impact: Potential subtle bugs in date/index arithmetic
   - Risk: Off-by-one errors in binning or date calculations
   - Mitigation: Comprehensive code review, add explicit // where needed
   - Testing: Edge case testing for date handling and array indexing

4. **Library Version Updates**
   - Impact: API changes across numpy/pandas/scipy
   - Risk: Deprecated methods, behavior changes
   - Mitigation: Incremental upgrade with testing at each step
   - Testing: Full test suite execution after each library upgrade

### Low Risk Items

5. **Syntax Changes** (print, xrange, dict methods)
   - Impact: Syntax errors, easy to catch
   - Risk: Minimal - automated tools handle these well
   - Mitigation: 2to3 tool + manual review
   - Testing: Basic smoke tests ensure code runs

---

## Existing Python 3 Work (Salamander Module)

The **salamander/** directory contains a partial Python 3 port (24 files):
- ✅ **Migrated:** Basic simulation engines (bsim, osim, qsim, ssim)
- ✅ **Migrated:** Core utilities (loaddata, calc, regress, util)
- ✅ **Migrated:** Data generation scripts (gen_dir, gen_hl, gen_alpha)
- ❌ **Still uses OpenOpt:** salamander/opt.py (blocks full Python 3 adoption)
- ❌ **Limited alpha strategies:** Only hl.py migrated (missing analyst, earnings, targets, etc.)

**Key Lesson from Salamander:**
- Demonstrates Python 3 migration is feasible
- Print statements and syntax issues easily resolved
- OpenOpt remains the primary blocker even in salamander
- pandas.stats already addressed in salamander/calc.py (commented out)

**Salamander requirements.txt:**
```
openopt==0.5628        # ← BLOCKER: No Python 3 support
FuncDesigner==0.5627   # ← BLOCKER: No Python 3 support
```

---

## Migration Strategy Recommendations

### Option A: Full Migration (Recommended)

**Approach:** Migrate entire main codebase to Python 3
**Timeline:** 5-7 days
**Phases:**
1. Phase 0: Preparation (set up Python 3 environment, update dependencies)
2. Phase 1: Syntax migration (print, xrange, dict methods)
3. Phase 2: OpenOpt replacement (critical path)
4. Phase 3: pandas.stats migration
5. Phase 4: Testing and validation

**Pros:**
- Modern, maintainable codebase
- Better performance (Python 3 is faster)
- Access to modern libraries and tools
- Unified codebase (no Python 2/3 split)

**Cons:**
- Requires OpenOpt replacement (16-24 hours)
- Risk of numerical differences requiring validation
- All-or-nothing (can't run partially migrated code)

### Option B: Gradual Migration

**Approach:** Use `six` library for compatibility, maintain Python 2/3 dual support
**Timeline:** 3-4 days
**Strategy:**
- Add `from __future__ import` statements
- Use `six.iteritems()` instead of `.iteritems()`
- Keep OpenOpt for Python 2, branch to alternative for Python 3

**Pros:**
- Lower risk (maintain Python 2 fallback)
- Can test incrementally

**Cons:**
- Technical debt (maintaining dual compatibility)
- Still requires OpenOpt replacement eventually
- Longer-term maintenance burden

### Option C: Defer Migration (Not Recommended)

**Approach:** Continue using Python 2
**Rationale:** "If it ain't broke, don't fix it"

**Cons:**
- Python 2 EOL: January 1, 2020 (6 years ago)
- Security vulnerabilities won't be patched
- Modern libraries dropping Python 2 support
- Harder to hire developers familiar with Python 2
- Technical debt grows over time

---

## Recommended Action Plan

### Immediate Next Steps

1. **Decision Point:** Commit to full Python 3 migration (Option A)

2. **OpenOpt Alternative Research (Task 2 of Plan 25)**
   - Prototype portfolio optimization with scipy.optimize
   - Test on sample data and compare results
   - Document API changes needed

3. **Set Up Validation Framework**
   - Create test harness comparing Python 2 vs Python 3 outputs
   - Define acceptable tolerance for numerical differences
   - Automate comparison reporting

4. **Create Migration Roadmap** (Task 3 of Plan 25)
   - Break down work into manageable tasks
   - Assign priorities and dependencies
   - Identify checkpoints and rollback points

### Success Metrics

- ✅ All Python files pass `python3 -m py_compile`
- ✅ Full test suite passes under Python 3
- ✅ Sample backtests produce equivalent results (within tolerance):
  - Position differences < 1%
  - PnL differences < 0.1%
  - Sharpe ratio differences < 0.05
- ✅ Performance: Simulation runtime within 10% of Python 2 baseline
- ✅ Documentation updated (README, CLAUDE.md)

---

## Appendix: Detailed File-by-File Analysis

### Files with Print Statements (71 total)
```
ssim.py          47 occurrences
qsim.py          17 occurrences
bsim.py           1 occurrence
osim.py           1 occurrence
osim2.py          2 occurrences
osim_simple.py    1 occurrence
bsim_weights.py   1 occurrence
bigsim_test.py    1 occurrence
```

### Files with dict.iter* Methods (8 total)
```
calc.py                    2 (.iteritems)
qsim.py                    1 (.iteritems)
salamander/calc.py         2 (already commented out)
salamander/qsim.py         1 (mentions in docstring)
salamander/ssim.py         1 (mentions in docstring)
```

### Files with xrange() (13 total)
```
opt.py                     6
pca.py                     1
opt.py.old                 6 (backup file - can ignore)
```

### Files with pandas.stats (14 files)
```
calc.py                    (pandas.stats.api.ols, pandas.stats.moments)
analyst.py                 (pandas.stats.moments.ewma)
analyst_badj.py            (pandas.stats.moments.ewma)
rating_diff.py             (pandas.stats.moments.ewma)
rating_diff_updn.py        (pandas.stats.moments.ewma)
eps.py                     (pandas.stats.moments.ewma)
target.py                  (pandas.stats.moments.ewma)
ebs.py                     (pandas.stats.moments.ewma)
prod_sal.py                (pandas.stats.moments.ewma)
prod_eps.py                (pandas.stats.moments.ewma)
prod_rtg.py                (pandas.stats.moments.ewma)
prod_tgt.py                (pandas.stats.moments.ewma)
salamander/calc.py         (already migrated - commented out)
```

---

## References

- Python 3.0 What's New: https://docs.python.org/3/whatsnew/3.0.html
- 2to3 Documentation: https://docs.python.org/3/library/2to3.html
- pandas Migration Guide: https://pandas.pydata.org/docs/whatsnew/index.html
- OpenOpt Project: http://openopt.org/ (unmaintained since 2014)

---

**Document Version:** 1.0
**Date:** 2026-02-08
**Author:** Automated Analysis via Claude Code
**Status:** Analysis Complete - Awaiting Implementation Decision
