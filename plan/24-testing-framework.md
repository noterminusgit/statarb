# Testing Framework

## Priority: HIGH
Files: New test files | Estimated effort: 8-12 hours

## Objective
Create pytest-based testing infrastructure with unit and integration tests for core modules.

## Tasks

### Task 1: Set Up Testing Infrastructure
**Files:** tests/ directory, pytest.ini, conftest.py

**Actions:**
- [ ] Create tests/ directory structure
- [ ] Install pytest, pytest-cov (update requirements.txt)
- [ ] Create pytest.ini configuration
- [ ] Create conftest.py with fixtures for sample data
- [ ] Add .gitignore entries for test cache/coverage files

**Subagent Instructions:**
```
Set up pytest testing infrastructure:
1. Create tests/ directory with __init__.py
2. Create pytest.ini with Python 2.7 compatibility settings
3. Create tests/conftest.py with fixtures:
   - sample_price_df: Simple 10-stock, 5-day price DataFrame
   - sample_returns_df: Returns data for testing
   - sample_universe_df: Mock universe definition
4. Add pytest and pytest-cov to requirements.txt
5. Create tests/README.md with running instructions
6. Add test outputs to .gitignore (*.pyc, __pycache__, .pytest_cache, .coverage, htmlcov/)
7. Commit: "Add pytest testing infrastructure"
8. Push to remote
```

### Task 2: Unit Tests for util.py
**Files:** tests/test_util.py

**Functions to test:**
- `merge_barra_data()` - verify lagging and alignment
- `filter_expandable()` - verify filtering logic
- `filter_tradeable()` - verify price/volume filters
- `winsorize()` - verify clipping at thresholds

**Subagent Instructions:**
```
Create unit tests for util.py:
1. Read util.py to understand function signatures
2. Create tests/test_util.py
3. Write tests for merge_barra_data (verify 1-day lag, no look-ahead)
4. Write tests for filter_expandable/filter_tradeable (verify thresholds)
5. Write tests for winsorize (verify clipping behavior)
6. Use conftest.py fixtures for sample data
7. Aim for 70%+ coverage of tested functions
8. Run: pytest tests/test_util.py -v
9. Commit: "Add unit tests for util.py"
10. Push to remote
```

### Task 3: Unit Tests for calc.py
**Files:** tests/test_calc.py

**Functions to test:**
- `winsorize_by_date()` - verify date-grouped winsorization
- `demean_by_sector()` - verify sector neutralization
- `calc_forward_returns()` - verify return calculation and horizons
- `mkt_ret()` - verify market-cap weighting

**Subagent Instructions:**
```
Create unit tests for calc.py:
1. Read calc.py to understand function signatures
2. Create tests/test_calc.py
3. Write test for winsorize_by_date (check cross-sectional clipping)
4. Write test for demean_by_sector (check mean = 0 within sectors)
5. Write test for calc_forward_returns (check shifting and horizons)
6. Write test for mkt_ret (check weighting formula)
7. Use simple synthetic data with known outputs
8. Run: pytest tests/test_calc.py -v
9. Commit: "Add unit tests for calc.py"
10. Push to remote
```

### Task 4: Integration Test for bsim.py
**Files:** tests/test_bsim_integration.py

**Scenarios to test:**
- End-to-end simulation with synthetic data (5 stocks, 10 days)
- Verify position generation and P&L calculation
- Test with simple forecast (all 1.0, all -1.0, mixed)

**Subagent Instructions:**
```
Create integration test for bsim.py:
1. Read bsim.py to understand simulation flow
2. Create tests/test_bsim_integration.py
3. Create synthetic data fixture:
   - 5 stocks, 10 trading days
   - Simple prices (random walk or linear)
   - Constant volume/market cap
   - Simple forecast signal (e.g., all 1.0)
4. Mock data loading (don't require real data files)
5. Run small simulation end-to-end
6. Verify outputs: positions generated, P&L calculated, no errors
7. Test edge cases: all zero forecasts, constrained optimization
8. Run: pytest tests/test_bsim_integration.py -v
9. Commit: "Add integration test for bsim.py"
10. Push to remote
```

### Task 5: Data Validation Tests
**Files:** tests/test_data_quality.py

**Validations:**
- Check for NaN/inf in loaded data
- Verify date alignment across datasets
- Check for duplicate tickers in universe
- Verify factor exposure sums

**Subagent Instructions:**
```
Create data validation tests:
1. Create tests/test_data_quality.py
2. Write helpers to validate DataFrame quality:
   - check_no_nan_inf(): Verify no NaN or inf values
   - check_date_alignment(): Verify all DataFrames have matching dates
   - check_unique_tickers(): Verify no duplicate tickers per date
3. Write tests that load sample data and validate
4. These tests should PASS on clean data, FAIL on bad data
5. Can be used as smoke tests in production
6. Run: pytest tests/test_data_quality.py -v
7. Commit: "Add data quality validation tests"
8. Push to remote
```

## Success Criteria
- [ ] pytest infrastructure set up and working
- [ ] 30+ unit tests covering util.py, calc.py
- [ ] Integration test for bsim.py passes
- [ ] Data validation tests created
- [ ] All tests passing with pytest
- [ ] Test coverage report generated
