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

## OpenOpt Replacement Research

### Problem Specification

The current opt.py implementation uses OpenOpt NLP solver for portfolio optimization with the following characteristics:

**Problem Type:**
- Non-Linear Programming (NLP) with quadratic objective function
- Mean-variance utility maximization: U = μ·x - κ(σ²·x² + x'Fx) - slippage(Δx) - costs(Δx)
- Quadratic risk terms: specific risk (σ²·x²) and factor risk (x'Fx)
- Nonlinear slippage function: power-law participation cost

**Problem Size:**
- Variables: ~1,400 (one per security in tradeable universe)
- Box constraints: ~1,400 (position bounds per security)
- Linear constraints: ~100 (factor exposure limits from Barra model)
- Nonlinear constraints: 1 (total notional capital constraint)

**Performance Requirements:**
- Current solve time: 1-5 seconds typical
- Called once per day for daily rebalancing (bsim.py)
- Must handle analytical gradients for efficiency

**Current Solver:**
- OpenOpt RALG algorithm (gradient-based NLP)
- Custom termination callback for early stopping
- Supports warm starting from current positions

### Alternative Solutions Comparison

#### 1. scipy.optimize.minimize (RECOMMENDED)

**Overview:**
Built-in Python optimization library with multiple constrained optimization algorithms. Part of SciPy ecosystem with excellent maintenance and community support.

**Suitable Methods:**
- **trust-constr**: Trust-region algorithm for constrained optimization (recommended for large-scale)
- **SLSQP**: Sequential Least Squares Programming (simpler API, adequate for medium-scale)

**Pros:**
- ✅ No additional dependencies (part of SciPy stack)
- ✅ Active development and excellent documentation
- ✅ Handles nonlinear objectives and constraints natively
- ✅ Supports analytical gradients (required for performance)
- ✅ Well-tested, numerically stable algorithms
- ✅ trust-constr specifically designed for large-scale problems (1000+ variables)
- ✅ Similar API to OpenOpt (objective function, gradient, constraints)

**Cons:**
- ⚠️ Constraint specification differs from OpenOpt (requires LinearConstraint/NonlinearConstraint objects)
- ⚠️ No built-in termination callbacks (would need custom wrapper)
- ⚠️ May require constraint reformulation for optimal performance

**Performance:**
- trust-constr: Designed for large-scale problems, expected 1-10 seconds for 1400 variables
- SLSQP: Good for medium-scale, may be slower for 1400+ variables

**Migration Effort:** LOW-MEDIUM
- API translation: 4-6 hours
- Testing and validation: 4-6 hours
- Total: 8-12 hours

**Code Example:**
```python
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

# Objective function (negated for minimization)
def objective_scipy(target):
    return -objective(target, kappa, slip_gamma, slip_nu, positions,
                      mu, rvar, factors, fcov, advp, advpt, vol,
                      mktcap, brate, price, execFee, untradeable_info)

def objective_grad_scipy(target):
    return -objective_grad(target, kappa, slip_gamma, slip_nu, positions,
                           mu, rvar, factors, fcov, advp, advpt, vol,
                           mktcap, brate, price, execFee, untradeable_info)

# Linear constraints (factor exposures)
linear_constraint = LinearConstraint(
    A=numpy.vstack([factors.T, -factors.T]),
    lb=numpy.concatenate([lbexp, -ubexp]),
    ub=numpy.full(2*num_factors, numpy.inf)
)

# Nonlinear constraint (total notional)
def capital_constraint(target):
    return max_sumnot - abs(target).sum()

nonlinear_constraint = NonlinearConstraint(
    fun=lambda x: abs(x).sum(),
    lb=0,
    ub=max_sumnot,
    jac=lambda x: numpy.sign(x)
)

# Solve
result = minimize(
    objective_scipy,
    x0=positions,
    method='trust-constr',
    jac=objective_grad_scipy,
    bounds=[(lb[i], ub[i]) for i in range(len(lb))],
    constraints=[linear_constraint, nonlinear_constraint],
    options={'maxiter': 500, 'verbose': 1}
)

target = result.x
```

**Recommendation Rationale:**
- Best balance of functionality, performance, and ease of migration
- No new dependencies (SciPy already required)
- trust-constr method specifically designed for problems of this scale
- Mature, well-maintained library with strong community support

---

#### 2. cvxpy + OSQP/ECOS

**Overview:**
Modern Python-embedded modeling language for convex optimization. Popular in quantitative finance for portfolio optimization.

**Pros:**
- ✅ High-level, intuitive API (can express problem in ~20 lines)
- ✅ Automatic problem classification and solver selection
- ✅ Excellent documentation and examples for portfolio optimization
- ✅ Active development (used at many quantitative hedge funds)
- ✅ Multiple backend solvers (OSQP, ECOS, SCS, MOSEK, GUROBI)
- ✅ OSQP solver shows excellent performance (benchmarked faster than commercial solvers)
- ✅ Handles up to 2,000 assets efficiently
- ✅ Integration with PyPortfolioOpt for higher-level portfolio operations

**Cons:**
- ⚠️ **Requires convex formulation** - nonlinear slippage function may need approximation
- ⚠️ Power-law slippage term (|Δx|^0.6) is non-convex, would need to:
  - Linearize/approximate the slippage function, OR
  - Use successive convex approximation (SCA) with iterative solving
- ⚠️ Additional dependency (cvxpy package)
- ⚠️ Less control over solver internals compared to scipy

**Performance:**
- OSQP: State-of-the-art QP solver, typically 0.5-5 seconds for 1400 variables
- Warm starting supported (can reuse previous solution)
- Benchmarks show OSQP outperforms ECOS and is competitive with commercial solvers

**Migration Effort:** MEDIUM-HIGH
- Requires reformulating slippage as convex function: 4-8 hours
- API translation: 3-4 hours
- Testing and validation: 4-6 hours
- Total: 11-18 hours

**Code Example (Convex Approximation):**
```python
import cvxpy as cp

# Decision variables
target = cp.Variable(num_secs)

# Convex approximation of slippage (linear or piecewise linear)
# Option 1: First-order Taylor approximation around current positions
delta = target - positions
slippage_linear = cp.sum(slip_gamma * vol * cp.abs(delta) / advp * (mktcap/advp)**slip_delta)

# Option 2: Piecewise linear approximation of power-law term
# (more accurate but more complex)

# Objective components
alpha_return = mu @ target
specific_risk = kappa * cp.quad_form(target, numpy.diag(rvar))
factor_risk = kappa * cp.quad_form(factors @ target, fcov)
execution_costs = execFee * cp.sum(cp.abs(delta) / price)

# Objective function (maximize utility = minimize negative utility)
objective = cp.Maximize(
    alpha_return - specific_risk - factor_risk - slippage_linear - execution_costs
)

# Constraints
constraints = [
    target >= lb,
    target <= ub,
    factors @ target >= lbexp,  # Factor exposure lower bounds
    factors @ target <= ubexp,  # Factor exposure upper bounds
    cp.sum(cp.abs(target)) <= max_sumnot  # Capital constraint
]

# Solve
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP, warm_start=True, verbose=True)

target = target.value
```

**Challenges:**
- Power-law slippage (participation^0.6) is non-convex
- Would need to either:
  1. Linearize: Use first-order approximation (sacrifices accuracy)
  2. Iterate: Successive convex approximation with multiple solves (adds latency)
  3. Simplify: Remove power-law term and use linear slippage only

**Recommendation Rationale:**
- Excellent choice IF slippage function can be adequately approximated
- Modern, well-supported library with strong finance community
- May require trading off some model fidelity for convexity

---

#### 3. CVXOPT QP Solver

**Overview:**
Python library for convex optimization, focusing on quadratic and conic programming. Lower-level API than cvxpy.

**Pros:**
- ✅ Efficient QP solver (mature, well-tested)
- ✅ Direct control over solver parameters
- ✅ Good performance for medium-scale problems

**Cons:**
- ⚠️ **Requires convex formulation** (same limitation as cvxpy)
- ⚠️ Lower-level API - requires manual constraint matrix construction
- ⚠️ Less intuitive than cvxpy or scipy
- ⚠️ Additional dependency
- ⚠️ Nonlinear slippage function not supported (QP only)

**Performance:**
- Good for QP problems, comparable to OSQP
- Not applicable to NLP problems with nonlinear slippage

**Migration Effort:** HIGH
- Reformulate as QP (linearize slippage): 6-8 hours
- Construct constraint matrices manually: 4-6 hours
- Testing and validation: 4-6 hours
- Total: 14-20 hours

**Code Example:**
```python
from cvxopt import matrix, solvers

# Standard QP form: minimize (1/2)x'Px + q'x
# subject to: Gx <= h, Ax = b

# Construct P matrix (quadratic risk terms)
P_specific = 2 * kappa * numpy.diag(rvar)
P_factor = 2 * kappa * factors.T @ fcov @ factors
P = P_specific + P_factor

# Construct q vector (linear terms: -mu + linearized slippage)
q = -mu + slippage_linear_grad

# Inequality constraints Gx <= h
G = numpy.vstack([
    -numpy.eye(num_secs),  # -x <= -lb  =>  x >= lb
    numpy.eye(num_secs),   # x <= ub
    factors.T,             # factor exposures <= ubexp
    -factors.T,            # -factor exposures <= -lbexp
])
h = numpy.concatenate([-lb, ub, ubexp, -lbexp])

# Solve
solution = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
target = numpy.array(solution['x']).flatten()
```

**Recommendation Rationale:**
- Not recommended due to high complexity and same convexity limitations as cvxpy
- cvxpy provides better abstraction with similar backend performance

---

#### 4. PyPortfolioOpt (High-Level Wrapper)

**Overview:**
Specialized portfolio optimization library built on cvxpy. Provides easy-to-use interface for classical portfolio optimization methods.

**Pros:**
- ✅ Extremely simple API for standard portfolio optimization
- ✅ One-liners like `ef.min_volatility()` or `ef.max_sharpe()`
- ✅ Built on cvxpy (leverages OSQP/ECOS solvers)
- ✅ Excellent documentation and examples
- ✅ Active development and community support
- ✅ Supports custom objectives and constraints

**Cons:**
- ⚠️ Abstracts away low-level control needed for custom slippage function
- ⚠️ Designed for standard mean-variance optimization, not custom NLP objectives
- ⚠️ Would require extending/modifying the library for our use case
- ⚠️ Additional dependency

**Use Case:**
- Best for **standard** portfolio optimization (min variance, max Sharpe, efficient frontier)
- Not suitable for custom objective functions with nonlinear slippage costs

**Migration Effort:** NOT APPLICABLE
- Would need to fork/extend library to support custom slippage model
- Easier to use cvxpy or scipy directly

**Recommendation Rationale:**
- Not recommended for this use case (too high-level, inflexible for custom objectives)
- Consider for future refactoring if slippage model is simplified

---

#### 5. Pyomo + IPOPT (Mentioned for Completeness)

**Overview:**
Pyomo is a Python-based optimization modeling language, IPOPT is a powerful nonlinear solver.

**Pros:**
- ✅ Handles complex nonlinear constraints and objectives
- ✅ Powerful solver (IPOPT) for large-scale NLP
- ✅ Supports all features needed (nonlinear slippage, gradients, etc.)

**Cons:**
- ⚠️ External solver dependency (IPOPT requires separate installation)
- ⚠️ Heavier installation footprint
- ⚠️ More complex API than scipy or cvxpy
- ⚠️ Overkill for this problem size

**Migration Effort:** HIGH
- Learning curve: 6-8 hours
- Implementation: 6-8 hours
- Testing: 4-6 hours
- Total: 16-22 hours

**Recommendation Rationale:**
- Not recommended (scipy.optimize is simpler and sufficient)
- Consider only if scipy.optimize proves inadequate

---

### Comparison Summary Table

| Feature | scipy.optimize | cvxpy + OSQP | CVXOPT | PyPortfolioOpt | Pyomo + IPOPT |
|---------|---------------|--------------|--------|----------------|---------------|
| **Problem Type** | NLP, QP | Convex only | QP only | QP only | NLP, QP |
| **Handles Nonlinear Slippage** | ✅ Yes | ⚠️ Needs approximation | ❌ No | ❌ No | ✅ Yes |
| **Problem Size (1400 vars)** | ✅ Excellent (trust-constr) | ✅ Excellent | ✅ Good | ✅ Good | ✅ Good |
| **Analytical Gradients** | ✅ Supported | ✅ Automatic | ✅ Supported | ✅ Automatic | ✅ Supported |
| **API Complexity** | Medium | Low | High | Very Low | High |
| **Dependencies** | Built-in (SciPy) | +cvxpy | +cvxopt | +pypfopt, cvxpy | +pyomo, ipopt |
| **Community Support** | Excellent | Excellent | Good | Good | Good |
| **Maintenance Status** | Active | Active | Active | Active | Active |
| **Migration Effort** | 8-12 hours | 11-18 hours | 14-20 hours | N/A | 16-22 hours |
| **Expected Solve Time** | 1-10 sec | 0.5-5 sec | 1-5 sec | 1-5 sec | 2-10 sec |
| **License** | BSD | Apache 2.0 | GPL | MIT | BSD / EPL |

### Feature-by-Feature Analysis

**Can it handle the optimization problem?**
- **scipy.optimize**: ✅ Yes, natively supports NLP with nonlinear objectives and constraints
- **cvxpy**: ⚠️ Partial, requires convex approximation of power-law slippage
- **CVXOPT**: ❌ No, QP only (cannot handle nonlinear slippage)
- **PyPortfolioOpt**: ❌ No, designed for standard mean-variance optimization
- **Pyomo + IPOPT**: ✅ Yes, full NLP support

**Performance characteristics (speed, scalability):**
- **scipy.optimize**: Excellent for 1400 variables with trust-constr method
- **cvxpy + OSQP**: Best for QP formulation, very fast (OSQP benchmarks show order of magnitude improvements)
- **CVXOPT**: Good for QP, competitive with OSQP
- **PyPortfolioOpt**: Same as cvxpy (uses it as backend)
- **Pyomo + IPOPT**: Good, but external solver adds complexity

**API complexity and migration effort:**
- **scipy.optimize**: Medium complexity, similar to OpenOpt API (8-12 hours)
- **cvxpy**: Low complexity for formulation, but requires slippage approximation (11-18 hours)
- **CVXOPT**: High complexity, manual matrix construction (14-20 hours)
- **PyPortfolioOpt**: Very simple for standard problems, but cannot handle custom slippage (N/A)
- **Pyomo + IPOPT**: High complexity, learning curve (16-22 hours)

**Active maintenance and community support:**
- **scipy.optimize**: ✅ Excellent (part of core SciPy)
- **cvxpy**: ✅ Excellent (1.5k+ stars, active development)
- **CVXOPT**: ✅ Good (mature, stable releases)
- **PyPortfolioOpt**: ✅ Good (2.8k+ stars, active community)
- **Pyomo + IPOPT**: ✅ Good (both actively maintained)

**License compatibility:**
- **scipy.optimize**: ✅ BSD (permissive)
- **cvxpy**: ✅ Apache 2.0 (permissive)
- **CVXOPT**: ⚠️ GPL (copyleft, may require compliance)
- **PyPortfolioOpt**: ✅ MIT (permissive)
- **Pyomo + IPOPT**: ✅ BSD / EPL (permissive)

---

### Recommended Solution: scipy.optimize.minimize (trust-constr)

**Primary Recommendation: scipy.optimize.minimize with trust-constr method**

**Rationale:**

1. **Native NLP Support**: Handles nonlinear slippage function without approximation
   - No need to linearize or approximate the power-law participation cost
   - Preserves model fidelity and optimization accuracy

2. **Optimal for Problem Scale**: trust-constr specifically designed for large-scale constrained optimization
   - Documented as "most appropriate for large-scale problems"
   - Efficient for 1000+ variables (our problem has ~1400)

3. **No Additional Dependencies**: Part of SciPy (already in requirements)
   - Reduces deployment complexity
   - No new libraries to maintain or debug

4. **Similar API to OpenOpt**: Minimal conceptual changes
   - Objective function with gradient
   - Box constraints (bounds)
   - Linear and nonlinear constraints
   - Familiar workflow for debugging and validation

5. **Lowest Migration Risk**:
   - Well-documented, mature library
   - Straightforward API translation
   - Estimated 8-12 hours effort (vs 11-22 hours for alternatives)

6. **Performance**: Expected 1-10 seconds solve time (acceptable for daily rebalancing)

**Implementation Plan:**

1. **Phase 1: Create wrapper functions** (2 hours)
   - Negate objective for minimization (scipy minimizes, OpenOpt maximizes)
   - Construct LinearConstraint objects for factor exposures
   - Construct NonlinearConstraint object for capital constraint

2. **Phase 2: Test on sample data** (3-4 hours)
   - Run side-by-side with OpenOpt on Python 2
   - Compare optimal positions, utility values, convergence
   - Validate numerical accuracy (positions within 1%)

3. **Phase 3: Full integration** (2-3 hours)
   - Replace OpenOpt calls in opt.py
   - Add custom termination logic if needed
   - Update logging/debugging output

4. **Phase 4: Validation** (3-4 hours)
   - Run full backtests (bsim.py) comparing Python 2 vs Python 3
   - Verify PnL, Sharpe ratios, turnover statistics
   - Document any numerical differences and root causes

**Total Estimated Effort: 10-13 hours**

---

### Alternative Recommendation: cvxpy + OSQP (If Slippage Can Be Simplified)

**Secondary Recommendation: cvxpy with OSQP solver**

**When to Consider:**
- If power-law slippage term can be adequately approximated with linear/quadratic functions
- If model simplification is acceptable for Python 3 migration
- If faster solve times (0.5-5 sec) are critical

**Pros of This Approach:**
- Cleaner, more maintainable code (high-level API)
- Excellent performance (OSQP benchmarks show superior speed)
- Better ecosystem integration (PyPortfolioOpt, cvxportfolio)
- Active development and finance community support

**Cons of This Approach:**
- Requires validating that convex approximation preserves optimization quality
- May need successive convex approximation (SCA) loop for accuracy
- Additional dependency (cvxpy)

**Implementation Path:**
1. Test linear approximation of slippage: slippage ≈ c₁·|Δx| + c₂·|Δx|²
2. Compare portfolio quality: optimal positions, realized slippage, PnL
3. If acceptable, proceed with cvxpy migration
4. If not acceptable, fall back to scipy.optimize.minimize

**Effort: 11-18 hours**

---

### Sample Code: Complete scipy.optimize.minimize Implementation

```python
#!/usr/bin/env python3
"""
Portfolio Optimization Module - Python 3 Migration
Using scipy.optimize.minimize (trust-constr method)
"""

import numpy
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

def optimize_scipy():
    """
    Portfolio optimization using scipy.optimize.minimize (trust-constr).

    Replacement for OpenOpt-based optimize() function.
    """
    # Partition tradeable/untradeable securities
    tradeable, untradeable = getUntradeable()

    if len(tradeable) == 0:
        raise ValueError("No tradeable securities found")

    # Extract tradeable subset
    t_num_secs = len(tradeable)
    t_positions = numpy.copy(g_positions[tradeable])
    t_factors = numpy.copy(g_factors[:, tradeable])
    t_lbound = numpy.copy(g_lbound[tradeable])
    t_ubound = numpy.copy(g_ubound[tradeable])
    t_mu = numpy.copy(g_mu[tradeable])
    t_rvar = numpy.copy(g_rvar[tradeable])
    t_advp = numpy.copy(g_advp[tradeable])
    t_advpt = numpy.copy(g_advpt[tradeable])
    t_vol = numpy.copy(g_vol[tradeable])
    t_mktcap = numpy.copy(g_mktcap[tradeable])
    t_borrowRate = numpy.copy(g_borrowRate[tradeable])
    t_price = numpy.copy(g_price[tradeable])

    # Extract untradeable subset
    u_positions = numpy.copy(g_positions[untradeable])
    u_factors = numpy.copy(g_factors[:, untradeable])
    u_mu = numpy.copy(g_mu[untradeable])
    u_rvar = numpy.copy(g_rvar[untradeable])

    # Compute exposure bounds
    exposures = numpy.dot(g_factors, g_positions)
    lbexp = numpy.minimum(exposures, -max_expnot * max_sumnot)
    lbexp = numpy.maximum(lbexp, -max_expnot * max_sumnot * hard_limit)
    ubexp = numpy.maximum(exposures, max_expnot * max_sumnot)
    ubexp = numpy.minimum(ubexp, max_expnot * max_sumnot * hard_limit)

    untradeable_exposures = numpy.dot(u_factors, u_positions)
    lbexp -= untradeable_exposures
    ubexp -= untradeable_exposures

    # Compute capital bound
    sumnot = abs(g_positions).sum()
    sumnot = max(sumnot, max_sumnot)
    sumnot = min(sumnot, max_sumnot * hard_limit)
    sumnot -= abs(u_positions).sum()

    # Position bounds
    lb = numpy.maximum(t_lbound, -max_posnot * max_sumnot)
    ub = numpy.minimum(t_ubound, max_posnot * max_sumnot)
    bounds = [(lb[i], ub[i]) for i in range(t_num_secs)]

    # Untradeable info
    untradeable_mu = numpy.dot(u_mu, u_positions)
    untradeable_rvar = numpy.dot(u_positions * u_rvar, u_positions)
    untradeable_loadings = untradeable_exposures
    untradeable_info = (untradeable_mu, untradeable_rvar, untradeable_loadings)

    # Objective function (negate for minimization)
    def obj_func(x):
        return -objective(x, kappa, slip_gamma, slip_nu, t_positions, t_mu,
                         t_rvar, t_factors, g_fcov, t_advp, t_advpt, t_vol,
                         t_mktcap, t_borrowRate, t_price, execFee, untradeable_info)

    def obj_grad(x):
        return -objective_grad(x, kappa, slip_gamma, slip_nu, t_positions, t_mu,
                              t_rvar, t_factors, g_fcov, t_advp, t_advpt, t_vol,
                              t_mktcap, t_borrowRate, t_price, execFee, untradeable_info)

    # Linear constraints (factor exposures)
    # Ax <= b  where A = [F^T; -F^T], b = [ubexp; -lbexp]
    A_linear = numpy.vstack([t_factors.T, -t_factors.T])
    b_linear = numpy.concatenate([ubexp, -lbexp])
    linear_constraint = LinearConstraint(
        A=A_linear,
        lb=-numpy.inf * numpy.ones(len(b_linear)),
        ub=b_linear
    )

    # Nonlinear constraint (capital)
    def capital_constraint_func(x):
        return abs(x).sum() - sumnot

    def capital_constraint_grad(x):
        return numpy.sign(x)

    nonlinear_constraint = NonlinearConstraint(
        fun=capital_constraint_func,
        lb=-numpy.inf,
        ub=0.0,
        jac=capital_constraint_grad
    )

    # Initial guess
    x0 = t_positions if zero_start == 0 else numpy.zeros(t_num_secs)

    # Solve
    result = minimize(
        fun=obj_func,
        x0=x0,
        method='trust-constr',
        jac=obj_grad,
        bounds=bounds,
        constraints=[linear_constraint, nonlinear_constraint],
        options={
            'maxiter': max_iter,
            'verbose': 2,
            'gtol': 1e-6,
            'xtol': 1e-6,
            'barrier_tol': 1e-6
        }
    )

    # Check convergence
    if not result.success:
        print(f"Optimization failed: {result.message}")
        if zero_start > 0:
            # Retry with current positions as starting point
            result = minimize(
                fun=obj_func,
                x0=t_positions,
                method='trust-constr',
                jac=obj_grad,
                bounds=bounds,
                constraints=[linear_constraint, nonlinear_constraint],
                options={'maxiter': max_iter, 'verbose': 2}
            )

    if not result.success:
        raise Exception(f"Optimization failed: {result.message}")

    # Reconstruct full target array (tradeable + untradeable)
    target = numpy.zeros(num_secs)
    opt_positions = result.x

    opt_index = 0
    tradeable_set = set(tradeable)
    for i in range(num_secs):
        if i in tradeable_set:
            target[i] = opt_positions[opt_index]
            opt_index += 1
        else:
            target[i] = g_positions[i]

    # Compute marginal utilities (same as original optimize())
    g_params = [kappa, slip_gamma, slip_nu, g_positions, g_mu, g_rvar,
                g_factors, g_fcov, g_advp, g_advpt, g_vol, g_mktcap,
                g_borrowRate, g_price, execFee,
                (0.0, 0.0, numpy.zeros(num_factors))]

    dutil = numpy.zeros(num_secs)
    dmu = numpy.zeros(num_secs)
    dsrisk = numpy.zeros(num_secs)
    dfrisk = numpy.zeros(num_secs)
    eslip = numpy.zeros(num_secs)
    costs = numpy.zeros(num_secs)
    dutil2 = numpy.zeros(num_secs)

    for ii in range(num_secs):
        targetwo = target.copy()
        targetwo[ii] = g_positions[ii]

        dutil_o1 = objective_detail(target, *g_params)
        dutil_o2 = objective_detail(targetwo, *g_params)
        dutil[ii] = dutil_o1[0] - dutil_o2[0]
        dmu[ii] = dutil_o1[1] - dutil_o2[1]
        dsrisk[ii] = dutil_o1[2] - dutil_o2[2]
        dfrisk[ii] = dutil_o1[3] - dutil_o2[3]
        eslip[ii] = dutil_o1[4] - dutil_o2[4]
        costs[ii] = dutil_o1[5] - dutil_o2[5]

        positions2 = g_positions.copy()
        positions2[ii] = target[ii]
        dutil2[ii] = objective(positions2, *g_params) - objective(g_positions, *g_params)

    printinfo(target, *g_params)

    return (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2)
```

---

### Next Steps

1. **Decision**: Confirm scipy.optimize.minimize (trust-constr) as the chosen alternative
2. **Prototype**: Implement basic version and test on sample data
3. **Validate**: Compare results with OpenOpt on Python 2
4. **Document**: Update opt.py docstrings and migration notes
5. **Test**: Run full backtests and compare performance metrics

---

## References

### Python 3 Migration
- Python 3.0 What's New: https://docs.python.org/3/whatsnew/3.0.html
- 2to3 Documentation: https://docs.python.org/3/library/2to3.html
- pandas Migration Guide: https://pandas.pydata.org/docs/whatsnew/index.html
- OpenOpt Project: http://openopt.org/ (unmaintained since 2014)

### Optimization Libraries Research

**scipy.optimize:**
- [SciPy minimize() documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [minimize(method='SLSQP') documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
- [minimize(method='trust-constr') documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html)
- [SciPy Optimization Tutorial](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)

**cvxpy:**
- [CVXPY Quadratic Programming Examples](https://cvxpy.readthedocs.io/en/latest/examples/basic/quadratic_program.html)
- [Portfolio Optimization using CVXPY (Medium)](https://medium.com/the-modern-scientist/how-to-select-your-mpf-portfolio-wisely-portfolio-optimization-53c9b86621b2)
- [Basic Portfolio Optimization with CVXPY](https://tirthajyoti.github.io/Notebooks/Portfolio_optimization.html)
- [Comparing SciPy Optimize and CVXPY (Medium)](https://medium.com/@kavya8a/on-comparing-scipy-optimize-and-cvxpy-c721c7d2c219)

**CVXOPT:**
- [Quadratic Optimization with CVXOPT (Towards Data Science)](https://towardsdatascience.com/quadratic-optimization-with-constraints-in-python-using-cvxopt-fc924054a9fc/)
- [CVXOPT QP Solver Tutorial](https://cvxopt.org/examples/tutorial/qp.html)
- [Quadratic Programming in Python](https://scaron.info/blog/quadratic-programming-in-python.html)

**PyPortfolioOpt:**
- [PyPortfolioOpt GitHub](https://github.com/PyPortfolio/PyPortfolioOpt)
- [Mean-Variance Optimization Documentation](https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html)

**OSQP Performance:**
- [OSQP Benchmarks GitHub](https://github.com/osqp/osqp_benchmarks)
- [OSQP: An Operator Splitting Solver for Quadratic Programs (Paper)](https://arxiv.org/pdf/1711.08013)
- [OSQP Official Website](https://osqp.org/)
- [State of Open-Source QP Solvers](https://quantsrus.github.io/post/state_of_convex_quadratic_programming_solvers/)

**General Portfolio Optimization:**
- [Quadratic Programming for Portfolio Optimization Guide](https://mbrenndoerfer.com/writing/quadratic-programming-portfolio-optimization)
- [Portfolio Optimization with SciPy (Towards Data Science)](https://towardsdatascience.com/portfolio-optimization-with-scipy-aa9c02e6b937/)
- [cvxportfolio GitHub](https://github.com/cvxgrp/cvxportfolio)
- [Riskfolio-Lib GitHub](https://github.com/dcajasn/Riskfolio-Lib)
- [skfolio GitHub](https://github.com/skfolio/skfolio)

---

**Document Version:** 1.1
**Date:** 2026-02-08
**Author:** Automated Analysis via Claude Code
**Status:** Task 2 Complete - OpenOpt Alternatives Research Added
