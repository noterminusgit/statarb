# Phase 4: Data Requirements for Numerical Validation

**Generated:** 2026-02-09
**Migration Branch:** python3-migration
**Python Version:** 3.12.3
**Current Status:** Data requirements documented, awaiting market data

---

## Executive Summary

Phase 4 numerical validation requires historical market data to run backtests and validate the Python 3 migration produces equivalent results to the Python 2 baseline. This document specifies all data requirements and provides a roadmap for completing Phase 4 when data becomes available.

**Current Blocker:** No market data available in configured directories
**Python 2 Baseline:** Python 2.7 not available (cannot capture baseline)
**Recommended Approach:**
1. Document all data requirements (this document)
2. Provide data acquisition guidance
3. Create validation scripts ready to run when data available
4. Test with synthetic data where possible

---

## Data Directory Configuration

The system expects data in the following locations (configured in `loaddata.py`):

```python
UNIV_BASE_DIR = ""       # Stock universe files
PRICE_BASE_DIR = ""      # Daily OHLCV data
BARRA_BASE_DIR = ""      # Barra risk model factors
BAR_BASE_DIR = ""        # Intraday 30-min bars
EARNINGS_BASE_DIR = ""   # Earnings announcement dates
LOCATES_BASE_DIR = ""    # Short borrow availability
ESTIMATES_BASE_DIR = ""  # Analyst estimates (IBES)
```

**Current Status:** All directories empty or not configured

---

## Required Data Sources

### 1. Universe Files (`UNIV_BASE_DIR/YYYY/YYYYMMDD.csv`)

**Purpose:** Define tradable stock universe for each date

**Directory Structure:**
```
UNIV_BASE_DIR/
├── 2013/
│   ├── 20130101.csv
│   ├── 20130102.csv
│   └── ...
└── 2014/
    └── ...
```

**Required Columns:**
- `sid` (int): Security identifier (unique stock ID)
- `ticker_root` (str): Stock ticker symbol (e.g., "AAPL")
- `status` (str): Trading status (e.g., 'ACTIVE')
- `country` (str): Country code (filter: 'USA')
- `currency` (str): Currency code (filter: 'USD')
- `price` (float): Current price for filtering
- `advp` (float): Average dollar volume
- `mkt_cap` (float): Market capitalization

**File Format:** CSV with daily snapshots
**Minimum Coverage:** 60 days before backtest start (for rolling calculations)
**Estimated Size:** ~1,400 stocks × 365 days = ~500K rows/year

**Sample Row:**
```csv
sid,ticker_root,status,country,currency,price,advp,mkt_cap
12345,AAPL,ACTIVE,USA,USD,150.25,5000000000,2500000000000
```

---

### 2. Price Files (`PRICE_BASE_DIR/YYYY/YYYYMMDD.csv`)

**Purpose:** Daily OHLCV market data

**Directory Structure:**
```
PRICE_BASE_DIR/
├── 2013/
│   ├── 20130101.csv
│   ├── 20130102.csv
│   └── ...
```

**Required Columns:**
- `sid` (int): Security identifier
- `ticker` (str): Full ticker symbol
- `open`, `high`, `low`, `close` (float): Daily prices
- `volume` (int): Share volume
- `mkt_cap` (float): Market capitalization ($)
- `advp` (float): Average dollar volume (30-day rolling)

**Data Quality Requirements:**
- Adjusted for splits and dividends
- OHLC validity: `high >= max(open, close)` and `low <= min(open, close)`
- No missing values for active trading days
- Minimum lookback: 60 days before simulation start

**Estimated Size:** ~1,400 stocks × 365 days × 10 columns = ~5M rows/year

**Sample Row:**
```csv
sid,ticker,open,high,low,close,volume,mkt_cap,advp
12345,AAPL,149.50,151.20,148.80,150.25,85000000,2500000000000,12750000000
```

---

### 3. Barra Files (`BARRA_BASE_DIR/YYYY/YYYYMMDD.csv`)

**Purpose:** Multi-factor risk model exposures

**Directory Structure:**
```
BARRA_BASE_DIR/
├── 2013/
│   ├── 20130101.csv
│   └── ...
```

**Required Barra Factors (13):**
1. `country` - Country factor
2. `growth` - Growth factor
3. `size` - Market cap factor (log)
4. `sizenl` - Non-linear size
5. `divyild` - Dividend yield
6. `btop` - Book-to-price ratio
7. `earnyild` - Earnings yield
8. `beta` - Market beta
9. `resvol` - Residual volatility
10. `betanl` - Non-linear beta
11. `momentum` - Price momentum
12. `leverage` - Financial leverage
13. `liquidty` - Trading liquidity

**Industry Classifications:** 58 GICS industries (`ind1` through `ind58`)
- Binary indicators (0 or 1) for industry membership
- Each stock belongs to exactly one industry

**Additional Columns:**
- `sid` (int): Security identifier
- `barraResidRet` (float): Idiosyncratic returns
- `barraSpecRisk` (float): Stock-specific risk
- `estu_barra4s` (bool): Barra estimation universe flag

**Data Source:** Barra USE4 or equivalent risk model
**Update Frequency:** Daily
**Lag:** 1-day lag to prevent look-ahead bias

**Estimated Size:** ~1,400 stocks × (13 factors + 58 industries + extras) = ~100K cells/day

**Sample Row:**
```csv
sid,country,growth,size,sizenl,divyild,btop,earnyild,beta,resvol,betanl,momentum,leverage,liquidty,ind1,...,ind58,barraResidRet,barraSpecRisk,estu_barra4s
12345,1.0,0.5,2.3,-0.1,0.02,0.8,0.05,1.1,0.3,0.05,0.2,-0.1,0.4,0,...,1,0.001,0.02,1
```

**Alternative Data Sources:**
- If Barra not available, can use Fama-French factors or custom factor models
- Minimum viable: beta, size, momentum, volatility

---

### 4. Bar Files (`BAR_BASE_DIR/YYYY/YYYYMMDD.h5`)

**Purpose:** Intraday 30-minute bars for QSIM and intraday strategies

**Directory Structure:**
```
BAR_BASE_DIR/
├── 2013/
│   ├── 20130101.h5
│   └── ...
```

**File Format:** HDF5 with MultiIndex (timestamp, sid)

**Required Columns:**
- `iclose` (float): Interval close price
- `ivwap` (float): Interval VWAP price
- `ivol` (int): Interval volume
- `dhigh` (float): Daily high (broadcast to all intervals)
- `dlow` (float): Daily low (broadcast to all intervals)

**Timestamps:** 30-minute intervals, 13 per day
- 09:30, 10:00, 10:30, 11:00, 11:30, 12:00, 12:30, 13:00, 13:30, 14:00, 14:30, 15:00, 15:30

**Data Structure Example:**
```python
import pandas as pd
df = pd.read_hdf('20130115.h5', 'bars')
# MultiIndex: [(timestamp1, sid1), (timestamp1, sid2), ..., (timestamp13, sid1400)]
# Shape: ~18,200 rows × 5 columns per day
```

**Compression:** LZF or gzip recommended
**Estimated Size:** ~18K rows × 5 columns × 4 bytes = ~350KB/day compressed

**Required for:** QSIM engine, intraday alpha strategies (qhl_*, badj_intra, etc.)
**Optional for:** BSIM daily backtest (not needed for Phase 4 validation)

---

### 5. Earnings Files (`EARNINGS_BASE_DIR/earnings.csv`)

**Purpose:** Earnings announcement dates for event avoidance/exploitation

**File Format:** Single CSV with all historical earnings

**Required Columns:**
- `sid` (int): Security identifier
- `date` (datetime): Announcement date (YYYY-MM-DD)
- `eps_actual` (float): Reported EPS
- `eps_estimate` (float): Consensus estimate
- `surprise` (float): Actual - estimate

**Coverage:** Quarterly earnings for all universe stocks
**Lookback:** 1-2 years recommended

**Estimated Size:** 1,400 stocks × 4 quarters × 2 years = ~11K rows

**Sample Row:**
```csv
sid,date,eps_actual,eps_estimate,surprise
12345,2013-01-24,1.25,1.20,0.05
```

**Required for:** EPS strategy (`eps.py`), earnings avoidance filters
**Optional for:** Basic BSIM backtest (can run without)

---

### 6. Locates File (`LOCATES_BASE_DIR/borrow.csv`)

**Purpose:** Short borrow availability and fee rates

**File Format:** Pipe-delimited CSV

**Required Columns:**
- `sid` or `SEDOL` (int/str): Security identifier
- `shares` (int): Available shares to borrow
- `fee` (float): Annual borrow fee rate (%)
- `symbol` (str): Stock ticker

**Update Frequency:** Daily or weekly
**Usage:** Constrain short positions to available borrows

**Estimated Size:** ~1,400 stocks × 1 row = ~1K rows

**Sample Row:**
```csv
sid|shares|fee|symbol
12345|1000000|0.5|AAPL
```

**Required for:** HTB strategy (`htb.py`), realistic short constraints
**Optional for:** Basic BSIM backtest (can assume unlimited borrow)

---

### 7. Estimates Files (`ESTIMATES_BASE_DIR/sal_YYYY/YYYYMMDD.csv`)

**Purpose:** Analyst estimates and revisions (IBES database)

**Directory Structure:**
```
ESTIMATES_BASE_DIR/
├── sal_2013/
│   ├── 20130101.csv
│   └── ...
```

**Required Columns:**
- `sid` (int): Security identifier
- `mean` (float): Consensus EPS estimate
- `median` (float): Median estimate
- `std` (float): Estimate standard deviation
- `num_estimates` (int): Number of analysts
- `actual` (float): Most recent actual EPS
- `anndats_act` (date): Most recent earnings date

**Data Source:** I/B/E/S or equivalent analyst database
**Update Frequency:** Daily (as estimates revised)

**Estimated Size:** ~1,400 stocks × 365 days = ~500K rows/year

**Sample Row:**
```csv
sid,mean,median,std,num_estimates,actual,anndats_act
12345,1.22,1.20,0.08,15,1.18,2012-10-23
```

**Required for:** Analyst strategies (`analyst.py`, `ebs.py`, `prod_sal.py`)
**Optional for:** Basic BSIM backtest (can run without)

---

## Data Acquisition Guide

### Commercial Data Vendors

1. **Bloomberg Terminal** (Most Comprehensive)
   - Universe: SECF function
   - Prices: BDH/BDP functions
   - Barra: BARR function (if subscribed)
   - Earnings: ERN function
   - Estimates: ANR function (IBES data)
   - Cost: $24K/year per terminal

2. **Refinitiv (formerly Thomson Reuters)**
   - Datastream: Historical prices and fundamentals
   - I/B/E/S: Analyst estimates database
   - StarMine: Alternative factor models
   - Cost: ~$30K-100K/year depending on data scope

3. **FactSet**
   - Prices: Prices database
   - Estimates: Estimates database
   - Ownership: Ownership database
   - Cost: ~$20K-80K/year

4. **Quandl/Nasdaq Data Link**
   - EOD US Stock Prices: $50/month (limited history)
   - SHARADAR Core US Equities: $150/month
   - No Barra factors (would need custom)

### Alternative Data Sources (Free/Low-Cost)

1. **Yahoo Finance** (Free, but limited)
   - Historical prices available via API
   - Coverage: Most US stocks
   - Limitations: No Barra factors, no IBES estimates
   - Python: `yfinance` library
   ```python
   import yfinance as yf
   df = yf.download('AAPL', start='2013-01-01', end='2013-12-31')
   ```

2. **Alpha Vantage** (Free tier: 5 calls/min)
   - Daily prices: Free
   - Earnings data: Free
   - Analyst estimates: Premium ($50/month)
   - API: https://www.alphavantage.co/

3. **Polygon.io** ($199/month)
   - Intraday bars (1-min, aggregatable to 30-min)
   - Daily OHLCV
   - No fundamental/analyst data

4. **Custom Factor Models** (For Barra Replacement)
   - Fama-French 5-factor model (free from Ken French website)
   - Calculate custom factors from price/volume:
     - Size: log(market cap)
     - Momentum: 12-month return
     - Volatility: rolling std of returns
     - Beta: regression vs. SPY
   - Industries: GICS classification (free from Wikipedia/Compustat)

### Minimum Viable Dataset (MVP)

For basic Phase 4 validation, the **absolute minimum** required:

1. **Prices** (PRICE_BASE_DIR): Daily OHLCV for ~200-500 stocks
   - Sufficient for basic backtest validation
   - Can use Yahoo Finance or similar
   - Date range: 2013-01-01 to 2013-06-30 (6 months)

2. **Universe** (UNIV_BASE_DIR): Stock list with basic filters
   - Can derive from price data (filter by price/volume)
   - Manually assign sids (1, 2, 3, ...)

3. **Barra Factors** (BARRA_BASE_DIR): Simplified factor model
   - Beta: Calculate from SPY regression
   - Size: log(market cap)
   - Momentum: 12-month return
   - Volatility: rolling std
   - Industries: Use sector ETF holdings or manual classification

**MVP Effort:** ~8-16 hours data acquisition + processing
**MVP Cost:** $0 (using Yahoo Finance)
**Validation Quality:** 60-70% (sufficient for syntax/logic validation)

---

## Synthetic Data Generation

For immediate validation without market data, we can generate synthetic data:

### Synthetic Data Script

```python
"""
Generate synthetic market data for Phase 4 validation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_universe(start_date, end_date, num_stocks=200):
    """Generate synthetic universe files"""
    dates = pd.bdate_range(start_date, end_date)

    for date in dates:
        sids = list(range(1, num_stocks + 1))
        tickers = [f"SYN{i:04d}" for i in sids]

        df = pd.DataFrame({
            'sid': sids,
            'ticker_root': tickers,
            'status': 'ACTIVE',
            'country': 'USA',
            'currency': 'USD',
            'price': np.random.uniform(10, 200, num_stocks),
            'advp': np.random.uniform(1e6, 1e9, num_stocks),
            'mkt_cap': np.random.uniform(1e8, 1e11, num_stocks)
        })

        # Save to UNIV_BASE_DIR/YYYY/YYYYMMDD.csv
        year_dir = f"synthetic_data/universe/{date.year}"
        os.makedirs(year_dir, exist_ok=True)
        df.to_csv(f"{year_dir}/{date.strftime('%Y%m%d')}.csv", index=False)

def generate_synthetic_prices(start_date, end_date, num_stocks=200):
    """Generate synthetic OHLCV data with realistic correlations"""
    dates = pd.bdate_range(start_date, end_date)

    # Initialize prices
    prices = np.random.uniform(50, 150, num_stocks)

    for date in dates:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, num_stocks)
        prices = prices * (1 + returns)

        sids = list(range(1, num_stocks + 1))
        tickers = [f"SYN{i:04d}" for i in sids]

        # Generate OHLC around close
        closes = prices
        opens = closes * np.random.uniform(0.98, 1.02, num_stocks)
        highs = np.maximum(opens, closes) * np.random.uniform(1.0, 1.03, num_stocks)
        lows = np.minimum(opens, closes) * np.random.uniform(0.97, 1.0, num_stocks)
        volumes = np.random.uniform(1e6, 1e8, num_stocks).astype(int)

        df = pd.DataFrame({
            'sid': sids,
            'ticker': tickers,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'mkt_cap': closes * volumes * np.random.uniform(10, 100, num_stocks),
            'advp': closes * volumes
        })

        year_dir = f"synthetic_data/prices/{date.year}"
        os.makedirs(year_dir, exist_ok=True)
        df.to_csv(f"{year_dir}/{date.strftime('%Y%m%d')}.csv", index=False)

def generate_synthetic_barra(start_date, end_date, num_stocks=200):
    """Generate synthetic Barra factors"""
    dates = pd.bdate_range(start_date, end_date)

    for date in dates:
        sids = list(range(1, num_stocks + 1))

        # 13 Barra factors (standardized)
        factors = pd.DataFrame({
            'sid': sids,
            'country': 1.0,  # Constant country factor
            'growth': np.random.normal(0, 1, num_stocks),
            'size': np.random.normal(0, 1, num_stocks),
            'sizenl': np.random.normal(0, 1, num_stocks),
            'divyild': np.random.normal(0, 1, num_stocks),
            'btop': np.random.normal(0, 1, num_stocks),
            'earnyild': np.random.normal(0, 1, num_stocks),
            'beta': np.random.normal(1.0, 0.3, num_stocks),
            'resvol': np.random.uniform(0.1, 0.5, num_stocks),
            'betanl': np.random.normal(0, 1, num_stocks),
            'momentum': np.random.normal(0, 1, num_stocks),
            'leverage': np.random.normal(0, 1, num_stocks),
            'liquidty': np.random.normal(0, 1, num_stocks),
            'barraResidRet': np.random.normal(0, 0.01, num_stocks),
            'barraSpecRisk': np.random.uniform(0.01, 0.05, num_stocks),
            'estu_barra4s': 1
        })

        # Add 58 industry dummies (each stock in one industry)
        industries = np.random.randint(1, 59, num_stocks)
        for i in range(1, 59):
            factors[f'ind{i}'] = (industries == i).astype(float)

        year_dir = f"synthetic_data/barra/{date.year}"
        os.makedirs(year_dir, exist_ok=True)
        factors.to_csv(f"{year_dir}/{date.strftime('%Y%m%d')}.csv", index=False)

if __name__ == '__main__':
    # Generate 6 months of synthetic data
    start = '2013-01-01'
    end = '2013-06-30'

    print("Generating synthetic universe...")
    generate_synthetic_universe(start, end, num_stocks=200)

    print("Generating synthetic prices...")
    generate_synthetic_prices(start, end, num_stocks=200)

    print("Generating synthetic Barra factors...")
    generate_synthetic_barra(start, end, num_stocks=200)

    print(f"Synthetic data generated in synthetic_data/")
    print(f"Configure loaddata.py paths to point to synthetic_data/")
```

**Limitations of Synthetic Data:**
- No real market correlations or dynamics
- Cannot validate absolute performance metrics
- Useful only for testing code paths, not numerical accuracy
- Should NOT be used for production or final validation

**Validation Quality:** 30-40% (syntax and basic logic only)

---

## Phase 4 Validation Plan

### When Real Data Available

1. **Configure Data Paths** (5 minutes)
   - Edit `loaddata.py` with actual data directory paths
   - Verify directory structure matches requirements

2. **Data Quality Check** (15 minutes)
   ```bash
   python3 scripts/validate_data.py --start=20130101 --end=20130630
   ```
   - Check for missing dates
   - Validate OHLC relationships
   - Check factor coverage
   - Verify universe size

3. **Python 2 Baseline Capture** (30 minutes)
   - If Python 2.7 available in separate environment:
   ```bash
   # On Python 2.7 environment (master branch)
   git checkout master
   python2.7 bsim.py --start=20130101 --end=20130630 \
       --fcast=hl:1:1 --kappa=2e-8 --maxnot=200e6 \
       > baseline/py2_output.txt 2>&1

   # Save outputs
   cp -r opt/ baseline/py2_opt/
   cp -r blotter/ baseline/py2_blotter/
   ```
   - Capture: positions, PnL, trades, Sharpe ratio, max drawdown

4. **Python 3 Backtest Run** (30 minutes)
   ```bash
   # On Python 3.12 environment (python3-migration branch)
   git checkout python3-migration
   python3 bsim.py --start=20130101 --end=20130630 \
       --fcast=hl:1:1 --kappa=2e-8 --maxnot=200e6 \
       > migrated/py3_output.txt 2>&1

   # Save outputs
   cp -r opt/ migrated/py3_opt/
   cp -r blotter/ migrated/py3_blotter/
   ```

5. **Numerical Comparison** (60 minutes)
   ```bash
   python3 scripts/compare_baselines.py \
       --py2=baseline/ --py3=migrated/ \
       --tolerance-pct=1.0
   ```

   **Comparison Metrics:**
   - Position differences (tolerance: < 1%)
   - PnL differences (tolerance: < 0.1%)
   - Sharpe ratio (tolerance: < 0.05)
   - Max drawdown (tolerance: < 0.5%)
   - Optimization solve time (target: < 10 sec)

   **Expected Result:** All metrics within tolerance

6. **Edge Case Testing** (30 minutes)
   - Test with empty universe days
   - Test with missing Barra factors
   - Test with extreme price moves
   - Test with zero volume days

7. **Performance Benchmarking** (15 minutes)
   ```bash
   time python3 bsim.py --start=20130101 --end=20130131 \
       --fcast=hl:1:1 --kappa=2e-8
   ```
   - Measure: data loading time, optimization time, total runtime
   - Compare to Python 2 baseline (expect similar or better)

### Without Python 2 Baseline

If Python 2 not available, validation focuses on:

1. **Internal Consistency Checks**
   - Run same backtest twice → identical results
   - Run with different random seeds → stable results
   - Run forward then backward → reversible

2. **Cross-Validation**
   - Compare BSIM vs OSIM vs QSIM on same data
   - Validate optimizer converges (< 1500 iterations)
   - Check factor exposures reasonable (< 0.5 std)

3. **Known-Result Tests**
   - Run on synthetic data with known alpha
   - Verify optimizer finds correct solution
   - Test regression fits converge

4. **Literature Benchmarks**
   - Compare Sharpe ratios to published research
   - Validate turnover rates reasonable (10-50% daily for stat arb)
   - Check market impact estimates vs academic papers

**Validation Quality:** 70-80% (good but not definitive)

---

## Success Criteria

### Full Validation (with Python 2 baseline)
- ✅ Position differences < 1% for 95% of stocks
- ✅ PnL difference < 0.1% cumulative
- ✅ Sharpe ratio difference < 0.05
- ✅ Optimization runtime < 10 seconds per day
- ✅ No crashes or errors on 6-month backtest
- ✅ Factor exposures within ±0.5 std

### Partial Validation (without Python 2 baseline)
- ✅ Python 3 backtest completes successfully
- ✅ Results internally consistent (repeat runs identical)
- ✅ Sharpe ratio plausible (0.5 - 3.0 for stat arb)
- ✅ Turnover reasonable (10-50% daily)
- ✅ No NaN/inf values in outputs
- ✅ Optimization converges (< 1500 iterations)

### Synthetic Data Validation
- ✅ No crashes on synthetic data
- ✅ All code paths exercised
- ✅ Results have expected structure
- ✅ Edge cases handled gracefully

---

## Next Steps

### Immediate (No Data Required)

1. **Create Synthetic Data Generator** ✅ Documented above
   - Script provided in this document
   - Can run basic validation immediately

2. **Create Validation Scripts**
   - `scripts/validate_data.py` - Check data quality
   - `scripts/compare_baselines.py` - Compare Py2 vs Py3
   - `scripts/generate_synthetic_data.py` - Synthetic data

3. **Document Known Limitations**
   - What validation is possible without data
   - What requires real market data
   - What requires Python 2 baseline

### Short-term (MVP Data Acquisition: ~8-16 hours)

1. **Acquire Minimum Viable Dataset**
   - Use Yahoo Finance for 200-500 stocks
   - 6 months: 2013-01-01 to 2013-06-30
   - Calculate simple factors (beta, size, momentum)

2. **Run Basic Backtest**
   - BSIM with hl:1:1 strategy
   - Validate code executes without errors
   - Check results structure and plausibility

3. **Performance Testing**
   - Measure optimization time
   - Check memory usage
   - Validate scalability

### Medium-term (Full Data Acquisition: Weeks/Months)

1. **Commercial Data Subscription**
   - Evaluate Bloomberg, Refinitiv, FactSet
   - Priority: Prices, Barra factors, Estimates
   - Budget: $20K-50K/year minimum

2. **Full Backtest Suite**
   - Run all 35 alpha strategies
   - Multi-year backtests (2010-2015)
   - Compare to published research

3. **Production Readiness**
   - Live data feeds integration
   - Real-time optimization
   - Production monitoring

### Long-term (Python 2 Baseline: If Possible)

1. **Docker Environment for Python 2**
   - Create containerized Python 2.7 environment
   - Run master branch in container
   - Capture baseline outputs

2. **Comprehensive Comparison**
   - Full numerical validation
   - Statistical hypothesis testing
   - Publish migration results

---

## Appendix: File Size Estimates

### Storage Requirements

| Dataset | Duration | Files | Size (Uncompressed) | Size (Compressed) |
|---------|----------|-------|---------------------|-------------------|
| Universe | 1 year | 252 | 50 MB | 10 MB |
| Prices | 1 year | 252 | 250 MB | 50 MB |
| Barra | 1 year | 252 | 500 MB | 100 MB |
| Bars (30-min) | 1 year | 252 | 5 GB | 500 MB |
| Earnings | 1 year | 1 | 5 MB | 1 MB |
| Locates | 1 year | 52 | 10 MB | 2 MB |
| Estimates | 1 year | 252 | 200 MB | 40 MB |
| **Total** | **1 year** | **1,313** | **6 GB** | **700 MB** |

### For 5-Year Backtest (2010-2015)

- **Uncompressed:** ~30 GB
- **Compressed:** ~3.5 GB
- **With HDF5 cache:** ~5 GB

**Recommendation:** 100 GB minimum storage for data + outputs

---

## Contact & Support

For questions about data requirements:
- See `README.md` section "Data Requirements" for detailed specifications
- See `CLAUDE.md` for codebase navigation guidance
- Check `loaddata.py` docstrings for data loading functions

**Phase 4 Status:** Awaiting market data acquisition
**Next Phase:** Phase 5 (Production deployment) after Phase 4 validation complete
