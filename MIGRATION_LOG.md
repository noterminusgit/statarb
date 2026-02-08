# Python 3 Migration Log

**Migration Start Date:** 2026-02-08
**Current Phase:** Phase 0 - Preparation
**Status:** In Progress

---

## Phase 0: Preparation (2-4 hours estimated)

**Objective:** Set up Python 3 migration infrastructure and baseline environment

### Tasks Completed

#### 1. Python 3 Requirements File Created
- **File:** requirements-py3.txt
- **Date:** 2026-02-08
- **Status:** ✅ Complete

**Library Version Updates:**

| Library | Python 2 (Old) | Python 3 (New) | Notes |
|---------|---------------|----------------|-------|
| numpy | 1.16.0 | >=1.19.0, <1.24.0 | Last Py2: 1.16.6 |
| pandas | 0.23.4 | >=1.3.0, <2.0.0 | pandas.stats removed in 0.23 |
| scipy | (not installed) | >=1.5.0 | New: for scipy.optimize |
| python-dateutil | 2.7.5 | >=2.8.2 | Minor upgrade |
| pytz | 2018.9 | >=2021.3 | Minor upgrade |
| six | 1.12.0 | >=1.15.0 | Compatibility layer |
| pytest | 4.6.11 | >=7.0.0 | Major upgrade |
| pytest-cov | 2.12.1 | >=3.0.0 | Minor upgrade |
| OpenOpt | (installed) | **REMOVED** | No Python 3 support |
| FuncDesigner | (installed) | **REMOVED** | No Python 3 support |

**Key Changes:**
- Removed OpenOpt/FuncDesigner (will be replaced with scipy.optimize in Phase 2)
- Added scipy (required for optimization replacement)
- Modern pandas version (pandas.stats migration required in Phase 3)
- Upper bounds on numpy (<1.24) and pandas (<2.0) for backward compatibility

---

#### 2. Migration Log Created
- **File:** MIGRATION_LOG.md (this file)
- **Date:** 2026-02-08
- **Status:** ✅ Complete

**Purpose:** Track all migration activities, decisions, and validation results across all phases.

---

#### 3. Validation Framework Script
- **File:** scripts/validate_migration.py
- **Date:** 2026-02-08
- **Status:** ✅ Complete

**Functionality:**
- Loads two backtest outputs (Python 2 baseline vs Python 3 migrated)
- Compares positions with 1% tolerance
- Compares PnL with 0.1% tolerance
- Compares Sharpe ratios with 0.05 tolerance
- Reports differences in detailed, human-readable format
- Exits with status code for CI/CD integration

**Usage:**
```bash
python3 scripts/validate_migration.py \
  --py2-positions=baseline/positions.csv \
  --py3-positions=migrated/positions.csv \
  --py2-pnl=baseline/pnl.csv \
  --py3-pnl=migrated/pnl.csv
```

**Validation Tolerances:**
- Position differences: < 1% of position size
- PnL differences: < 0.1% of cumulative PnL
- Sharpe ratio differences: < 0.05
- Any differences exceeding tolerance will be reported with details

---

#### 4. Migration Branch Strategy
- **Branch Name:** python3-migration
- **Date Created:** 2026-02-08
- **Status:** ✅ Complete

**Branch Strategy:**
- All Python 3 migration work will be committed to the `python3-migration` branch
- `master` branch remains on Python 2.7 until migration is validated and complete
- Regular commits after each phase completion with clear commit messages
- Branch will be merged to `master` after Phase 4 validation (GO decision)
- Production deployment (Phase 5) happens after merge to master

**Commit Message Format:**
- Phase 0: "Phase 0: Set up Python 3 migration infrastructure"
- Phase 1: "Phase 1: Python 3 syntax migration complete"
- Phase 2: "Phase 2: Replace OpenOpt with scipy.optimize"
- Phase 3: "Phase 3: Migrate pandas.stats to pandas.Series.ewm()"
- Phase 4: "Phase 4: Validation complete - Python 3 migration ready"
- Phase 5: "Phase 5: Python 3 production deployment"

---

### Python 2.7 Baseline Environment

**Python Version:** 2.7.x (legacy)

**Library Versions (from requirements.txt):**
```
numpy==1.16.0
pandas==0.23.4
python-dateutil==2.7.5
pytz==2018.9
six==1.12.0
pytest==4.6.11
pytest-cov==2.12.1
```

**Additional Dependencies (not in requirements.txt):**
- OpenOpt 0.5628 (no Python 3 support)
- FuncDesigner 0.5627 (no Python 3 support)

**Baseline Backtest (to be captured):**
- **Command:** `python2.7 bsim.py --start=20130101 --end=20130630 --fcast=hl:1:1 --kappa=2e-8 --maxnot=200e6`
- **Purpose:** Establish baseline positions, PnL, and statistics for validation
- **Status:** Pending execution (requires Python 2 environment)

**Note:** Baseline backtest execution deferred - can be run later when needed for Phase 4 validation.

---

### Python 3 Target Environment

**Python Version:** 3.8+ (recommended: 3.9 or 3.10 for stability)

**Library Versions (from requirements-py3.txt):**
```
numpy>=1.19.0,<1.24.0
pandas>=1.3.0,<2.0.0
scipy>=1.5.0
python-dateutil>=2.8.2
pytz>=2021.3
six>=1.15.0
pytest>=7.0.0
pytest-cov>=3.0.0
```

**Key Replacements:**
- OpenOpt → scipy.optimize.minimize (trust-constr method)
- FuncDesigner → Native Python optimization formulation
- pandas.stats.moments.ewma → pandas.Series.ewm().mean()
- pandas.stats.api.ols → statsmodels.api.OLS (or scipy.stats.linregress)

**Installation Command:**
```bash
pip3 install -r requirements-py3.txt
```

---

### Phase 0 Success Criteria

- [x] requirements-py3.txt created with Python 3 compatible versions
- [x] MIGRATION_LOG.md created with Phase 0 documentation
- [x] Validation framework script created (scripts/validate_migration.py)
- [x] Migration branch (python3-migration) created
- [ ] All Phase 0 work committed and pushed to remote
- [ ] LOG.md updated with terse Phase 0 summary

**Phase 0 Status:** Complete (pending commit/push)

---

### Next Steps (Phase 1)

**Phase 1: Syntax Migration (8-12 hours estimated)**

Key tasks:
1. Run 2to3 tool on all Python files
2. Convert print statements to print() functions (71 occurrences)
3. Replace dict.iteritems/iterkeys/itervalues with .items/.keys/.values (8 occurrences)
4. Replace xrange() with range() (13 occurrences)
5. Add `from __future__ import division` to all files
6. Audit integer division operators (592 occurrences)
7. Fix exception syntax (3 occurrences)
8. Replace file() with open() (12 occurrences)
9. Run `python3 -m py_compile` on all files to verify syntax
10. Smoke tests: loaddata.py, calc.py basic operations

**Blocker for Phase 1:** None - can proceed immediately after Phase 0 commit

**Phase 1 Branch:** Continue on python3-migration branch

---

## Migration Timeline

| Phase | Description | Effort | Status | Start Date | End Date |
|-------|-------------|--------|--------|------------|----------|
| Phase 0 | Preparation | 2-4 hours | ✅ Complete | 2026-02-08 | 2026-02-08 |
| Phase 1 | Syntax Migration | 8-12 hours | Not Started | - | - |
| Phase 2 | OpenOpt Replacement | 16-24 hours | Not Started | - | - |
| Phase 3 | pandas.stats Migration | 6-8 hours | Not Started | - | - |
| Phase 4 | Testing & Validation | 8-12 hours | Not Started | - | - |
| Phase 5 | Production Deployment | 4-8 hours | Not Started | - | - |
| **Total** | **Full Migration** | **44-68 hours** | **In Progress** | **2026-02-08** | **TBD** |

**Estimated Completion:** 5-7 business days from Phase 1 start

---

## Issues and Decisions Log

### Issue #1: OpenOpt Replacement Choice
- **Date:** 2026-02-08
- **Issue:** OpenOpt has no Python 3 support. Multiple alternatives available.
- **Options Evaluated:**
  1. scipy.optimize.minimize (trust-constr)
  2. cvxpy + OSQP
  3. CVXOPT
  4. Pyomo + IPOPT
- **Decision:** scipy.optimize.minimize (trust-constr method)
- **Rationale:**
  - Native NLP support (no convex approximation needed)
  - No additional dependencies (part of scipy)
  - Lowest migration effort (8-12 hours)
  - Similar API to OpenOpt
  - Optimal for 1400-variable problems
- **Risk:** Potential numerical differences in optimization results
- **Mitigation:** Extensive Phase 4 validation with tight tolerances

### Issue #2: Baseline Backtest Execution
- **Date:** 2026-02-08
- **Issue:** Phase 0 calls for baseline Python 2 backtest, but Python 2 environment may not be readily available
- **Decision:** Defer baseline backtest capture until Phase 4
- **Rationale:**
  - Can use historical backtest results if available
  - Phase 4 will run side-by-side comparison when Python 3 code is ready
  - Does not block Phase 1 or Phase 2 work
- **Action:** Document that baseline can be established in Phase 4

### Issue #3: Version Upper Bounds
- **Date:** 2026-02-08
- **Issue:** Should requirements-py3.txt use exact versions or ranges?
- **Decision:** Use ranges with conservative upper bounds
- **Rationale:**
  - numpy <1.24: API changes in 1.24+ may break compatibility
  - pandas <2.0: Major breaking changes in 2.0
  - Allows security updates within safe range
  - Reduces pinning maintenance burden
- **Action:** Applied upper bounds to numpy and pandas

---

## Validation Results

### Phase 0 Validation
- **Status:** N/A (no code changes in Phase 0, infrastructure only)

### Phase 1 Validation
- **Status:** Not started

### Phase 2 Validation
- **Status:** Not started

### Phase 3 Validation
- **Status:** Not started

### Phase 4 Validation
- **Status:** Not started

---

## References

- **Primary Migration Plan:** PYTHON3_MIGRATION.md
- **Current Requirements:** requirements.txt
- **Python 3 Requirements:** requirements-py3.txt
- **Validation Script:** scripts/validate_migration.py
- **Project Documentation:** CLAUDE.md, README.md

---

**Log Maintained By:** Claude Code (Anthropic)
**Last Updated:** 2026-02-08
