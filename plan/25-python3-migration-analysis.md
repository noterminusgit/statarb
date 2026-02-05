# Python 3 Migration Analysis

## Priority: MEDIUM (Investigation Phase)
Files: Analysis documents | Estimated effort: 6-8 hours

## Objective
Analyze feasibility of Python 3 migration, identify blockers, research OpenOpt alternatives, and create migration roadmap.

## Tasks

### Task 1: Python 3 Compatibility Survey
**Files:** Create PYTHON3_MIGRATION.md

**Analysis Required:**
- Identify Python 2-specific syntax (print statements, dict.iteritems(), etc.)
- Identify deprecated libraries (OpenOpt, FuncDesigner)
- Check NumPy/Pandas version compatibility
- Survey division operator usage (/ vs //)
- Check string/unicode handling

**Subagent Instructions:**
```
Survey Python 3 compatibility issues:
1. Use 2to3 tool to analyze codebase: 2to3 -l *.py > py3_issues.txt
2. Grep for Python 2 patterns:
   - Print statements: grep "print " *.py | wc -l
   - dict.iteritems/iterkeys/itervalues: grep "\.iter" *.py
   - Division operator: analyze critical calculations
   - String types: grep -E "(unicode|basestring)" *.py
3. Check library versions:
   - Current: pandas, numpy, scipy versions
   - Python 3 compatible versions available
4. Create PYTHON3_MIGRATION.md with:
   - Summary of issues found (counts, categories)
   - Risk assessment (high/medium/low)
   - Effort estimation (hours per category)
5. Commit: "Add Python 3 compatibility survey"
6. Push to remote
```

### Task 2: OpenOpt Alternative Research
**Files:** Add to PYTHON3_MIGRATION.md

**Alternatives to evaluate:**
1. **cvxpy** - Convex optimization, active development
2. **scipy.optimize** - Built-in, quadratic programming
3. **CVXOPT** - Open source convex optimization
4. **Commercial solvers** - Gurobi, CPLEX (mention only)

**Research Questions:**
- Can they handle portfolio optimization constraints?
- Performance comparison (speed, iterations)
- API complexity (migration effort)
- License compatibility

**Subagent Instructions:**
```
Research OpenOpt alternatives:
1. Read opt.py to understand current optimization problem:
   - Objective: quadratic utility function (mean-variance)
   - Constraints: position bounds, factor exposure, notional limits
   - Problem size: ~1400 variables, ~100 constraints
2. Research alternatives:
   - cvxpy: Check if it supports quadratic objectives + linear constraints
   - scipy.optimize.minimize: Check if it supports SLSQP or trust-constr
   - CVXOPT: Check QP solver capability
3. Create comparison table in PYTHON3_MIGRATION.md:
   - Features, performance, ease of migration, maintenance status
4. Recommend top choice with rationale
5. Note: Don't implement, just research and document
6. Commit: "Add OpenOpt alternatives research"
7. Push to remote
```

### Task 3: Migration Roadmap
**Files:** Add to PYTHON3_MIGRATION.md, create plan/26-python3-migration-phase1.md (placeholder)

**Roadmap Phases:**
1. **Phase 0: Preparation** (2-4 hours)
   - Set up Python 3 virtual environment
   - Update requirements.txt with Python 3 compatible versions
   - Run 2to3 automated conversion (non-destructive)

2. **Phase 1: Syntax Migration** (8-12 hours)
   - Fix print statements
   - Fix dict.iteritems/iterkeys/itervalues
   - Fix integer division
   - Fix string/bytes handling
   - Test with Python 3 interpreter

3. **Phase 2: OpenOpt Replacement** (16-24 hours)
   - Implement portfolio optimization with chosen alternative
   - Test on sample backtests
   - Performance benchmarking
   - Fallback plan if issues arise

4. **Phase 3: Testing & Validation** (8-12 hours)
   - Run full test suite (if exists)
   - Run sample backtests on Python 2 vs Python 3
   - Compare outputs for consistency
   - Fix any discrepancies

5. **Phase 4: Production Deployment** (4-8 hours)
   - Update documentation
   - Update deployment scripts
   - Monitor production performance

**Subagent Instructions:**
```
Create Python 3 migration roadmap:
1. Read PYTHON3_MIGRATION.md (from previous tasks)
2. Add "Migration Roadmap" section with 5 phases (above)
3. For each phase, list:
   - Objectives
   - Key tasks
   - Effort estimate
   - Risks/blockers
   - Success criteria
4. Add total effort estimate: 38-60 hours (5-7 days)
5. Add risk assessment:
   - High risk: OpenOpt replacement (optimization may differ)
   - Medium risk: Numerical differences (floating point)
   - Low risk: Syntax changes (automated tools)
6. Create placeholder plan/26-python3-migration-phase1.md (empty, for future)
7. Note: This is ANALYSIS only, not implementation
8. Commit: "Add Python 3 migration roadmap"
9. Push to remote
```

## Success Criteria
- [ ] Python 3 compatibility issues catalogued
- [ ] OpenOpt alternatives researched and compared
- [ ] Migration roadmap created with effort estimates
- [ ] Risk assessment completed
- [ ] Recommendation made for migration approach
- [ ] Decision point: Proceed with migration or defer?
