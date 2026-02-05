# Testing Guide for StatArb

This directory contains the pytest-based testing infrastructure for the statistical arbitrage trading system.

## Installation

Install test dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `pytest==4.6.11` - Testing framework (Python 2.7 compatible)
- `pytest-cov==2.12.1` - Coverage reporting plugin

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_util.py
```

### Run specific test function
```bash
pytest tests/test_util.py::test_winsorize
```

### Run tests by marker
```bash
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests
pytest -m "not slow"    # Skip slow tests
```

## Coverage Reports

### Generate coverage report (terminal)
```bash
pytest --cov=. --cov-report=term-missing
```

### Generate HTML coverage report
```bash
pytest --cov=. --cov-report=html
```

Then open `htmlcov/index.html` in a browser.

### Generate coverage for specific module
```bash
pytest --cov=util --cov-report=term-missing tests/test_util.py
```

## Test Structure

```
tests/
├── __init__.py           # Package marker
├── conftest.py           # Shared fixtures
├── README.md             # This file
├── test_util.py          # Unit tests for util.py
├── test_calc.py          # Unit tests for calc.py
├── test_bsim_integration.py  # Integration test for bsim.py
└── test_data_quality.py  # Data validation tests
```

## Available Fixtures

Fixtures are defined in `conftest.py` and automatically available to all tests:

- **sample_price_df**: 10 stocks, 5 days of OHLCV price data
- **sample_returns_df**: Daily returns calculated from prices
- **sample_universe_df**: Mock universe with symbols and sectors
- **sample_barra_df**: Barra factor exposures (beta, momentum, size, etc.)
- **sample_volume_df**: Trading volume and market cap data

## Writing New Tests

### Example unit test
```python
def test_my_function(sample_price_df):
    """Test description."""
    result = my_function(sample_price_df)
    assert result is not None
    assert len(result) > 0
```

### Example with markers
```python
import pytest

@pytest.mark.unit
def test_fast_function():
    """Fast unit test."""
    assert 1 + 1 == 2

@pytest.mark.integration
@pytest.mark.slow
def test_full_simulation(sample_price_df):
    """Slow integration test."""
    # Run full simulation...
    pass
```

## Test Markers

Available markers (defined in `pytest.ini`):
- `@pytest.mark.unit` - Unit test for individual function
- `@pytest.mark.integration` - Integration test for full workflow
- `@pytest.mark.slow` - Test takes >1 second to run
- `@pytest.mark.data` - Test requires data files

## Troubleshooting

### ImportError when running tests
Make sure you're in the project root directory:
```bash
cd /home/dude/statarb
pytest
```

### Python 2.7 compatibility issues
This project uses Python 2.7. Ensure pytest 4.6.11 is installed (last version supporting Python 2.7):
```bash
pip install pytest==4.6.11
```

### Coverage not working
Ensure pytest-cov is installed:
```bash
pip install pytest-cov==2.12.1
```

## CI/CD Integration

To run tests in CI pipeline:

```bash
# Run all tests with coverage
pytest --cov=. --cov-report=xml --cov-report=term

# Check for minimum coverage (e.g., 60%)
pytest --cov=. --cov-fail-under=60
```

## Next Steps

1. Run tests: `pytest -v`
2. Check coverage: `pytest --cov=. --cov-report=html`
3. Write new tests for uncovered modules
4. Aim for >70% coverage on core modules (util.py, calc.py, regress.py, opt.py)
