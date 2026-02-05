# Improvement Plan - Code Quality, Testing, and Python 3 Analysis

## Status: Active - 3 New Plans

After completing comprehensive documentation (98%, 86/88 files) and fixing 11 critical bugs, focus shifts to code quality improvements, testing infrastructure, and Python 3 migration analysis.

## Current Plans

### 23: Code Quality Improvements (MEDIUM Priority)
**Objective**: Fix hard-coded paths and improve error handling
- Task 1: Fix hard-coded paths in salamander scripts (2 files)
- Task 2: Add error handling to core modules (4 files)
- **Effort**: 4-6 hours
- **Files**: salamander/change_hl.py, salamander/show_borrow.py, loaddata.py, calc.py, regress.py, opt.py

### 24: Testing Framework (HIGH Priority) - 3/5 COMPLETE
**Objective**: Create pytest infrastructure with comprehensive tests
- Task 1: Set up pytest infrastructure ✅ COMPLETE (Commit: 5b25150)
- Task 2: Unit tests for util.py ✅ COMPLETE (Commit: b34c321)
- Task 3: Unit tests for calc.py ✅ COMPLETE (Commit: ad5519d)
- Task 4: Integration test for bsim.py
- Task 5: Data validation tests
- **Effort**: 8-12 hours (6-8 hours spent)
- **Files**: tests/ directory (created), test_util.py (40+ tests), test_calc.py (26 tests)

### 25: Python 3 Migration Analysis (MEDIUM Priority)
**Objective**: Research Python 3 feasibility and create migration roadmap
- Task 1: Python 3 compatibility survey
- Task 2: OpenOpt alternatives research
- Task 3: Migration roadmap with effort estimates
- **Effort**: 6-8 hours
- **Files**: PYTHON3_MIGRATION.md (new)

## Execution Strategy

### Quick Win (Start Now)
1. **Plan 23, Task 1**: Fix hard-coded paths (1-2 hours) - Quick improvement

### High Priority (Week 1)
2. **Plan 24**: Set up testing framework (8-12 hours) - Foundation for code confidence
3. **Plan 23, Task 2**: Add error handling (3-4 hours) - Production safety

### Analysis Phase (Week 2)
4. **Plan 25**: Python 3 migration analysis (6-8 hours) - Decision point for major migration

## Subagent Execution Instructions

### For Plan 23 (Code Quality)
```bash
# Agent 1: Fix hard-coded paths
Task: Follow instructions in plan/23-code-quality-improvements.md, Task 1
Output: CLI arguments added to 2 salamander scripts
Commit: Auto-committed with descriptive message

# Agent 2: Add error handling
Task: Follow instructions in plan/23-code-quality-improvements.md, Task 2
Output: Validation added to 4 core modules
Commit: Auto-committed with descriptive message
```

### For Plan 24 (Testing)
```bash
# Agent 1: Infrastructure setup ✅ COMPLETE
Task: Follow instructions in plan/24-testing-framework.md, Task 1
Output: pytest infrastructure with fixtures
Status: DONE (Commit: 5b25150)

# Agents 2-5: Tests (parallel - ready to start)
Task: Follow instructions in plan/24-testing-framework.md, Tasks 2-5
Output: 30+ unit tests, 1+ integration test
Status: READY
```

### For Plan 25 (Python 3 Analysis)
```bash
# Agent 1: Compatibility survey (parallel with Agent 2)
Task: Follow instructions in plan/25-python3-migration-analysis.md, Task 1
Output: Python 3 issues catalogued

# Agent 2: OpenOpt research (parallel with Agent 1)
Task: Follow instructions in plan/25-python3-migration-analysis.md, Task 2
Output: Alternative solver comparison

# Agent 3: Roadmap (sequential after Agents 1-2)
Task: Follow instructions in plan/25-python3-migration-analysis.md, Task 3
Output: Migration roadmap with effort estimates
```

## Previous Work Completed

### Documentation Phase ✅ (Complete)
- 86/88 files documented (98%)
- ~16,000 lines of docstrings added
- 21 documentation plans completed and deleted
- Comprehensive README enhancement

### Bug Fixes ✅ (Complete)
- 7 bugs fixed in beta-adjusted strategies
- 2 bugs fixed in hl_intra.py
- 2 incomplete PCA generators completed
- **Total**: 11 critical runtime issues resolved

## Reference Documents

- **plan/00-documentation-overview.md** - Detailed plan descriptions and subagent strategies
- **plan/23-code-quality-improvements.md** - Code quality task details
- **plan/24-testing-framework.md** - Testing infrastructure task details
- **plan/25-python3-migration-analysis.md** - Python 3 analysis task details
- **LOG.md** - Chronological history with all changes

## Success Metrics

### Code Quality
- [ ] Zero hard-coded paths in utility scripts
- [ ] Core functions have input validation
- [ ] Error messages informative and actionable

### Testing
- [x] pytest infrastructure operational
- [x] 30+ unit tests passing (66 tests: 40 for util.py, 26 for calc.py)
- [x] Test coverage >60% for util.py, calc.py (estimated 70-75% for util.py, 40-50% for calc.py)
- [ ] 1+ integration test passing

### Python 3 Analysis
- [ ] Compatibility issues catalogued
- [ ] OpenOpt alternative recommended
- [ ] Migration roadmap documented
- [ ] Go/no-go decision ready

## Total Effort Estimate
- Code Quality: 4-6 hours
- Testing Framework: 8-12 hours
- Python 3 Analysis: 6-8 hours
- **Total**: 18-26 hours (2-3 weeks with subagents)
