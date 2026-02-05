# Statistical Arbitrage Trading System

A production-grade statistical arbitrage (stat-arb) trading system that identifies market mispricings through quantitative factor analysis, portfolio optimization, and systematic execution. The system processes historical market data, generates alpha signals from multiple strategies, optimizes portfolio positions considering transaction costs and risk, and backtests trading strategies through multiple simulation engines.

## Overview

This system implements a complete workflow for statistical arbitrage trading:

1. **Data Loading & Preprocessing**: Loads and processes market data from multiple sources
2. **Alpha Generation**: Calculates predictive signals from 20+ trading strategies
3. **Factor Analysis**: Decomposes returns using PCA and Barra risk models
4. **Portfolio Optimization**: Maximizes risk-adjusted returns with realistic constraints
5. **Backtesting**: Simulates execution across multiple engines with transaction cost modeling

The system is designed for daily rebalancing across ~1,400 US equities with sophisticated risk management and execution cost modeling.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Strategies](#strategies)
- [Simulation Engines](#simulation-engines)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Salamander Module](#salamander-module)
- [Performance Metrics](#performance-metrics)

## Features

### Core Capabilities

- **Multi-Source Data Integration**: Daily/intraday prices, Barra factors, analyst estimates, short locates
- **20+ Alpha Strategies**: PCA decomposition, analyst signals, momentum, mean reversion, order flow
- **Advanced Optimization**: NLP solver with factor risk, transaction costs, and participation constraints
- **Multiple Simulation Engines**: Daily (BSIM), order-level (OSIM), intraday (QSIM), full system (SSIM)
- **Risk Management**: Factor exposure limits, position sizing, sector neutrality
- **Realistic Execution Modeling**: Market impact, slippage, borrow costs, VWAP vs. close fills

### Technical Features

- **HDF5 Caching**: Fast data loading with compressed storage
- **Vectorized Operations**: Efficient pandas/numpy operations for large datasets
- **Rolling Window Analysis**: Adaptive factor models with 30-60 day windows
- **Winsorization**: Robust outlier handling at 5-sigma levels
- **Corporate Action Handling**: Automatic adjustment for splits and dividends

## Architecture

### Data Flow

```
Raw Market Data (CSV/SQL)
    ↓
Load & Merge (loaddata.py)
    ↓
Calculate Returns & Features (calc.py)
    ↓
Filter Tradable Universe
    ↓
Generate Alpha Signals (strategy files)
    ↓
Fit Regression Coefficients (regress.py)
    ↓
PCA Decomposition (pca.py) [optional]
    ↓
Portfolio Optimization (opt.py)
    ↓
Simulation Engines (bsim/osim/qsim/ssim)
    ↓
Performance Analysis & Reporting
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Data Loading | `loaddata.py` | Load market data, fundamentals, analyst estimates |
| Calculations | `calc.py` | Forward returns, volume profiles, winsorization |
| Regression | `regress.py` | Fit alpha factors to forward returns (WLS) |
| PCA | `pca.py` | Principal component decomposition |
| Optimization | `opt.py` | Portfolio optimization with OpenOpt NLP |
| Big Sim | `bsim.py` | Daily rebalancing backtest |
| Order Sim | `osim.py` | Order-level execution backtest |
| Quote Sim | `qsim.py` | Intraday 30-min bar backtest |
| System Sim | `ssim.py` | Full lifecycle position tracking |
| Utilities | `util.py` | Helper functions for data merging |

## Installation

### Python Version Requirements

**CRITICAL**: This codebase has **two separate Python environments**:

#### Main Codebase: Python 2.7 (Legacy)
- **All core modules**: loaddata, calc, regress, opt, util
- **All simulation engines**: bsim, osim, qsim, ssim
- **All alpha strategies**: hl, bd, analyst, eps, etc.
- **All production modules**: prod_sal, prod_eps, prod_rtg, prod_tgt
- **Reason**: OpenOpt dependency (not Python 3 compatible)

#### Salamander Module: Python 3.x
- **Location**: `salamander/` directory
- **Purpose**: Simplified, standalone, Python 3 compatible version
- **Use Case**: Modern deployments, easier development
- **Separate Dependencies**: `salamander/requirements.txt`

**Migration Note**: The salamander module provides a migration path to Python 3, with simplified data pipelines and compatible optimization. Consider using salamander for new development.

### Main Codebase Requirements (Python 2.7)

```
Python 2.7.x
numpy==1.16.0
pandas==0.23.4
OpenOpt==0.5628
FuncDesigner==0.5628
statsmodels
scikit-learn
matplotlib
scipy
lmfit
tables (PyTables for HDF5)
mysql-connector-python (optional, for SQL data sources)
```

### Salamander Module Requirements (Python 3.x)

```
Python 3.6+
numpy>=1.19.0
pandas>=1.1.0
scipy>=1.5.0
scikit-learn>=0.23.0
matplotlib>=3.3.0
tables>=3.6.0
lmfit>=1.0.0
```

### Setup Instructions

#### Main Codebase (Python 2.7)

```bash
# Clone the repository
git clone https://github.com/yourusername/statarb.git
cd statarb

# Create Python 2.7 virtual environment (if using virtualenv)
virtualenv -p python2.7 venv27
source venv27/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install OpenOpt (may require manual installation)
pip install openopt funcdesigner

# Optional: Build Cython optimization module
python setup.py build_ext --inplace

# Verify installation
python -c "from loaddata import *; print('Success')"
```

#### Salamander Module (Python 3)

```bash
cd salamander

# Create Python 3 virtual environment
python3 -m venv venv3
source venv3/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "from loaddata import *; print('Success')"
```

### Known Installation Issues

1. **OpenOpt Installation**:
   - OpenOpt is no longer actively maintained
   - May require `--no-deps` flag: `pip install --no-deps openopt`
   - Alternative: Install from source

2. **NumPy/Pandas Compatibility**:
   - Python 2.7 requires specific versions (NumPy 1.16, Pandas 0.23)
   - Newer versions break Python 2.7 compatibility

3. **PyTables (HDF5)**:
   - Required for HDF5 file operations
   - `pip install tables` may require HDF5 C libraries
   - Ubuntu: `sudo apt-get install libhdf5-dev`
   - macOS: `brew install hdf5`

4. **MySQL Connector** (optional):
   - Only needed if using SQL data sources
   - `pip install mysql-connector-python`

## Quick Start

### 1. Prepare Data Directories

Set the base directories in `loaddata.py`:

```python
UNIV_BASE_DIR = "/path/to/universe/"
PRICE_BASE_DIR = "/path/to/prices/"
BARRA_BASE_DIR = "/path/to/barra/"
BAR_BASE_DIR = "/path/to/bars/"
EARNINGS_BASE_DIR = "/path/to/earnings/"
LOCATES_BASE_DIR = "/path/to/locates/"
ESTIMATES_BASE_DIR = "/path/to/estimates/"
```

### 2. Run a Simple Backtest

```bash
# Run BSIM with a single alpha signal
python bsim.py --start=20130101 --end=20130630 \
    --fcast=hl:1:1 \
    --kappa=2e-8 \
    --maxnot=200e6
```

### 3. Combine Multiple Alphas

```bash
# Combine high-low and beta-adjusted signals
python bsim.py --start=20130101 --end=20130630 \
    --fcast=hl:1:0.6,bd:0.8:0.4 \
    --kappa=2e-8
```

## Data Requirements

### Data Directory Configuration

Edit the following constants in `loaddata.py` to point to your data directories:

```python
UNIV_BASE_DIR = "/path/to/universe/"      # Stock universe files
PRICE_BASE_DIR = "/path/to/prices/"       # Daily OHLCV data
BARRA_BASE_DIR = "/path/to/barra/"        # Barra risk model factors
BAR_BASE_DIR = "/path/to/bars/"           # Intraday 30-min bars
EARNINGS_BASE_DIR = "/path/to/earnings/"  # Earnings announcement dates
LOCATES_BASE_DIR = "/path/to/locates/"    # Short borrow availability
ESTIMATES_BASE_DIR = "/path/to/estimates/" # Analyst estimates (IBES)
```

### Required Data Sources

#### 1. Universe Files (`UNIV_BASE_DIR/YYYY/YYYYMMDD.csv`)

**Purpose**: Define tradable stock universe for each date

**Required Columns**:
- `sid` (int): Security identifier
- `ticker_root` (str): Stock ticker symbol
- `status` (str): Trading status (e.g., 'ACTIVE')
- `country` (str): Country code (filter: 'USA')
- `currency` (str): Currency code (filter: 'USD')

**File Format**: CSV with daily snapshots
**File Naming**: `YYYYMMDD.csv` (e.g., `20130115.csv`)
**Directory Structure**: `UNIV_BASE_DIR/YYYY/YYYYMMDD.csv`

#### 2. Price Files (`PRICE_BASE_DIR/YYYY/YYYYMMDD.csv`)

**Purpose**: Daily OHLCV market data

**Required Columns**:
- `sid` (int): Security identifier
- `ticker` (str): Full ticker symbol
- `open`, `high`, `low`, `close` (float): Daily prices
- `volume` (int): Share volume
- `mkt_cap` (float): Market capitalization ($)
- `advp` (float): Average dollar volume (calculated)

**File Format**: CSV with daily snapshots
**Data Quality**: Handle splits, dividends, delistings
**Lookback**: Minimum 60 days for rolling calculations

#### 3. Barra Files (`BARRA_BASE_DIR/YYYY/YYYYMMDD.csv`)

**Purpose**: Multi-factor risk model exposures

**Required Barra Factors** (13):
1. `country` - Country factor
2. `growth` - Growth factor
3. `size` - Market cap factor
4. `sizenl` - Non-linear size
5. `divyild` - Dividend yield
6. `btop` - Book-to-price
7. `earnyild` - Earnings yield
8. `beta` - Market beta
9. `resvol` - Residual volatility
10. `betanl` - Non-linear beta
11. `momentum` - Price momentum
12. `leverage` - Financial leverage
13. `liquidty` - Trading liquidity

**Industry Classifications**: 58 GICS industries (ind1-ind58)

**Additional Columns**:
- `barraResidRet` - Idiosyncratic returns
- `barraSpecRisk` - Stock-specific risk
- Factor covariance matrix (separate file or embedded)

**File Format**: CSV with standardized Barra format
**Update Frequency**: Daily
**Lag**: 1-day lag to prevent look-ahead bias

#### 4. Bar Files (`BAR_BASE_DIR/YYYY/YYYYMMDD.h5`)

**Purpose**: Intraday 30-minute bars for QSIM and intraday strategies

**Required Columns**:
- `iclose` (float): Interval close price
- `ivwap` (float): Interval VWAP
- `ivol` (int): Interval volume
- `dhigh`, `dlow` (float): Daily high/low (broadcast to all intervals)

**File Format**: HDF5 with MultiIndex (timestamp, sid)
**Timestamps**: 30-minute intervals (13 per day: 09:30-16:00)
**Compression**: LZF or gzip for storage efficiency

**Example Structure**:
```python
df = pd.read_hdf('20130115.h5', 'bars')
# MultiIndex: [(timestamp1, sid1), (timestamp1, sid2), ...]
# Columns: iclose, ivwap, ivol, dhigh, dlow
```

#### 5. Earnings Files (`EARNINGS_BASE_DIR/earnings.csv`)

**Purpose**: Earnings announcement dates for event avoidance/exploitation

**Required Columns**:
- `sid` (int): Security identifier
- `date` (datetime): Announcement date
- `eps_actual` (float): Reported EPS
- `eps_estimate` (float): Consensus estimate
- `surprise` (float): Actual - estimate

**File Format**: CSV with all historical earnings
**Coverage**: Quarterly earnings for all universe stocks

#### 6. Locates File (`LOCATES_BASE_DIR/borrow.csv`)

**Purpose**: Short borrow availability and fee rates

**Required Columns**:
- `sid` or `SEDOL` (int/str): Security identifier
- `shares` (int): Available shares to borrow
- `fee` (float): Annual borrow fee rate (%)
- `symbol` (str): Stock ticker

**File Format**: Pipe-delimited CSV
**Update Frequency**: Daily or weekly
**Usage**: Constrain short positions to available borrows

#### 7. Estimates Files (`ESTIMATES_BASE_DIR/sal_*.csv`)

**Purpose**: Analyst estimates and revisions (IBES database)

**Required Columns**:
- `sid` (int): Security identifier
- `mean` (float): Consensus EPS estimate
- `median` (float): Median estimate
- `std` (float): Estimate standard deviation
- `num_estimates` (int): Number of analysts

**File Format**: CSV with date snapshots
**Data Source**: I/B/E/S or equivalent analyst database
**Strategies Using**: `analyst.py`, `ebs.py`, `prod_sal.py`

### Universe Filters

#### Tradable Universe
- **Price Range**: $2.00 - $500.00
- **Min ADV**: $1,000,000 (average dollar volume)
- **Country**: USA
- **Currency**: USD
- **Market Cap**: Top 1,400 stocks (configurable via `uni_size`)

#### Expandable Universe
- **Price Range**: $2.25 - $500.00
- **Min ADV**: $5,000,000
- **Purpose**: Wider universe for alpha generation, narrower for execution

#### Additional Filters
- **Sector Exclusions**: PHARMA industry can be excluded
- **Earnings Avoidance**: Optional N-day window around earnings
- **Locate Requirements**: Short positions require borrow availability
- **Data Quality**: Require non-null prices, volume, Barra factors

### Data Setup Guide

1. **Organize Directory Structure**:
   ```bash
   /data/
   ├── universe/YYYY/YYYYMMDD.csv
   ├── prices/YYYY/YYYYMMDD.csv
   ├── barra/YYYY/YYYYMMDD.csv
   ├── bars/YYYY/YYYYMMDD.h5
   ├── earnings/earnings.csv
   ├── locates/borrow.csv
   └── estimates/sal_YYYY/YYYYMMDD.csv
   ```

2. **Configure Paths**: Edit constants in `loaddata.py`

3. **Validate Data**:
   ```bash
   python readcsv.py  # Check data integrity
   ```

4. **Generate HDF5 Cache** (optional):
   ```python
   # HDF5 cache created automatically on first load
   # Significantly speeds up repeated backtests
   ```

5. **Test Data Loading**:
   ```python
   from loaddata import *
   df = load_prices('20130101', '20130131', lookback=30)
   print(df.info())
   ```

## Usage

### Portfolio Optimization

The optimization module (`opt.py`) maximizes:

```
Utility = Alpha - κ(Specific Risk + Factor Risk) - Slippage - Execution Costs
```

**Key Parameters:**
- `kappa`: Risk aversion (2e-8 to 4.3e-5)
- `max_sumnot`: Max total notional ($50M default)
- `max_posnot`: Max position size (0.48% of capital)
- `slip_nu`: Market impact coefficient (0.14-0.18)

**Constraints:**
- Position limits: ±$40k-$1M per stock
- Capital limits: $4-50M aggregate notional
- Participation: Max 1.5% of ADV
- Factor exposure: Limited Barra factor bets

### Simulation Engines

#### BSIM - Daily Simulation

Most comprehensive daily backtest with optimized positions:

```bash
python bsim.py \
    --start=20130101 \
    --end=20130630 \
    --fcast=hl:1:0.5,bd:0.8:0.3,pca:1.2:0.2 \
    --horizon=3 \
    --kappa=2e-8 \
    --maxnot=200e6 \
    --locates=True \
    --vwap=False
```

**Arguments:**
- `--start/--end`: Date range (YYYYMMDD)
- `--fcast`: Alpha signals (format: `name:multiplier:weight`)
- `--horizon`: Forecast horizon in days
- `--kappa`: Risk aversion parameter
- `--maxnot`: Maximum notional
- `--vwap`: Use VWAP execution (default: close)

#### OSIM - Order Simulation

Order-level backtest with fill strategy analysis:

```bash
python osim.py \
    --start=20130101 \
    --end=20130630 \
    --fill=vwap \
    --slipbps=0.0001 \
    --fcast=alpha_files
```

#### QSIM - Intraday Simulation

30-minute bar simulation for intraday strategies:

```bash
python qsim.py \
    --start=20130101 \
    --end=20130630 \
    --fcast=qhl_intra \
    --horizon=3 \
    --mult=1000 \
    --slipbps=0.0001
```

#### SSIM - System Simulation

Full lifecycle with position and cash tracking:

```bash
python ssim.py \
    --start=20130101 \
    --end=20131231 \
    --fcast=combined_alpha
```

## Alpha Strategies

The system implements **35 alpha strategies** across 7 strategy families, each with multiple variants optimized for different market conditions and time horizons.

### Strategy Families

#### 1. High-Low Mean Reversion (6 strategies)

**Core Concept**: Exploits mean reversion of prices relative to daily high-low geometric midpoint.

**Signal Formula**: `hl0 = close / sqrt(high * low)`

**Variants**:
- `hl.py` - Base strategy with daily + intraday signals, industry-demeaned
- `hl_intra.py` - Intraday-only for high-frequency trading
- `qhl_intra.py` - Quote-level intraday with hourly coefficients
- `qhl_multi.py` - Multi-period daily signals (1-5 day lags)
- `qhl_both.py` - Combined daily + intraday optimization
- `qhl_both_i.py` - Industry-specific variant with sector models

**Characteristics**:
- Holding period: 1-3 days
- Turnover: High (daily rebalancing)
- Industry neutral via demeaning
- Negative coefficients (mean reversion)

**Usage**: `--fcast=hl:1:1` or `--fcast=qhl_intra:1.2:0.5`

#### 2. Beta-Adjusted Order Flow (9 strategies)

**Core Concept**: Two distinct approaches - order flow imbalance and beta-adjusted returns.

**Order Flow Signal**: `bd = (bidqty - askqty) / (bidqty + askqty) / beta`

**Return Signal**: `badj = return / beta`

**Variants**:
- `bd.py` - Base order flow strategy with volume weighting
- `bd1.py` - Simplified daily order flow
- `bd_intra.py` - Intraday order flow (6 hourly periods)
- `badj_multi.py` - Multi-lag beta-adjusted returns
- `badj_intra.py` - Intraday return signals
- `badj_both.py` - Combined daily + intraday returns
- `badj_dow_multi.py` - Day-of-week specific models
- `badj2_multi.py` - Alternative beta adjustment methodology
- `badj2_intra.py` - Alternative intraday implementation

**Characteristics**:
- Exploits market microstructure inefficiencies
- Removes systematic market component via beta
- Separate Energy sector models
- Multiple time horizon variants

**Usage**: `--fcast=bd:0.8:0.4` or `--fcast=badj_multi:1:0.3`

#### 3. Analyst Signals (4 strategies)

**Core Concept**: Fundamental signals from analyst ratings, estimates, and price targets.

**Data Source**: IBES (I/B/E/S) analyst database

**Variants**:
- `analyst.py` - Base analyst rating/estimate changes
- `analyst_badj.py` - Beta-adjusted analyst signals
- `rating_diff.py` - Rating change momentum with cubed amplification
- `rating_diff_updn.py` - Separate up/down revision models

**Characteristics**:
- Low frequency updates (weekly/monthly)
- Fundamental value capture
- Consensus change indicators
- Asymmetric up/down responses

**Usage**: `--fcast=analyst:1.5:0.15`

#### 4. Earnings & Valuation (3 strategies)

**Core Concept**: Event-driven strategies based on earnings surprises and analyst targets.

**Variants**:
- `eps.py` - Post-earnings announcement drift (PEAD)
- `target.py` - Price target deviation signals
- `prod_tgt.py` - Production target strategy with filtering

**Characteristics**:
- Event-driven (quarterly earnings)
- Exploits analyst forecast errors
- Price target revisions
- Earnings surprise magnitude

**Usage**: `--fcast=eps:1:0.1`

#### 5. Volume-Adjusted (5 strategies)

**Core Concept**: Volume-return interaction patterns with liquidity-aware position sizing.

**Signal Formula**: `vadj = (volume/median_volume - 1) * beta_adj_return`

**Variants**:
- `vadj.py` - Full model with daily + intraday signals
- `vadj_multi.py` - Daily-only multi-period version
- `vadj_intra.py` - Intraday-only for short-term trading
- `vadj_pos.py` - Position sizing emphasis with sign-based signals
- `vadj_old.py` - Legacy implementation (deprecated)

**Characteristics**:
- Market-wide volume adjustment
- Industry neutralization
- Hourly coefficient fitting for intraday
- Execution quality focus

**Usage**: `--fcast=vadj:1:0.2`

#### 6. PCA & Residual Strategies (3 strategies)

**Core Concept**: Statistical decomposition to isolate stock-specific noise for mean reversion.

**Variants**:
- `pca_generator.py` - Intraday PCA residual extraction (4 components)
- `pca_generator_daily.py` - Daily PCA with exp-weighted correlation
- `rrb.py` - Barra residual return betting (idiosyncratic returns)

**Characteristics**:
- Market-neutral by construction
- Rolling correlation matrices (10-period)
- Excludes Energy sector
- Multi-day lag combinations

**Usage**: `--fcast=pca:1.2:0.2` or `--fcast=rrb:1:0.15`

#### 7. Specialized Strategies (5 strategies)

**Variants**:
- `c2o.py` - Close-to-open gap trading with intraday timing
- `mom_year.py` - 232-day lagged momentum (annual reversal)
- `ebs.py` - Analyst estimate revision signals (not equity borrow!)
- `htb.py` - Hard-to-borrow fee rate strategy (short squeeze detection)
- `badj_rating.py` - Beta-adjusted rating strategy

**Usage**: `--fcast=c2o:1:0.1,mom:1:0.05`

### Strategy Combination

Combine multiple alphas with optimized weights:

```bash
# Multi-strategy portfolio
python bsim.py \
    --start=20130101 \
    --end=20130630 \
    --fcast=pca:1.0:0.3,hl:1.2:0.25,bd:0.8:0.2,analyst:1.5:0.15,vadj:1:0.1
```

**Forecast Format**: `name:multiplier:weight`
- `name`: Strategy file name (without .py)
- `multiplier`: Scaling factor applied to alpha signal
- `weight`: Portfolio weight (should sum to 1.0)

**Weight Optimization**: Use `bsim_weights.py` for grid search over weight combinations

### Strategy Development Workflow

1. **Develop Alpha**: Create new strategy file with alpha calculation
   - Load data using `loaddata.py` functions
   - Calculate signal (e.g., price ratios, volume patterns, fundamental ratios)
   - Apply transformations (winsorization, industry demeaning)

2. **Fit Coefficients**: Use `regress.py` to fit on in-sample data
   - Weighted least squares regression to forward returns
   - Separate in-sector / ex-sector fits if needed
   - Rolling window or expanding window

3. **Generate Forecasts**: Apply coefficients to out-of-sample period
   - Multiply signals by fitted coefficients
   - Combine multiple lags/timeframes
   - Output to HDF5 or CSV

4. **Optimize**: Run through `opt.py` to get target positions
   - Maximize utility function (alpha - risk - costs)
   - Apply position limits and constraints
   - Factor risk management via Barra model

5. **Backtest**: Simulate with appropriate engine
   - BSIM: Daily strategies with optimization
   - OSIM: Fill strategy comparison
   - QSIM: Intraday strategies on 30-min bars
   - SSIM: Full lifecycle with cash tracking

6. **Analyze**: Evaluate performance metrics
   - Sharpe ratio and information ratio
   - Maximum drawdown
   - Factor exposures (13 Barra factors)
   - Turnover and execution costs

## Simulation Engines

### Comparison

| Engine | Use Case | Granularity | Execution Model |
|--------|----------|-------------|-----------------|
| **BSIM** | Daily strategies | Daily | Optimized positions |
| **OSIM** | Fill analysis | Order-level | VWAP/mid/close fills |
| **QSIM** | Intraday strategies | 30-min bars | Time-of-day analysis |
| **SSIM** | Full system | Daily + intraday | Complete lifecycle |

### Output Metrics

All engines provide:
- **P&L**: Daily and cumulative
- **Sharpe Ratio**: Risk-adjusted returns
- **Drawdown**: Maximum peak-to-trough decline
- **Turnover**: Average daily trading volume
- **Factor Exposures**: Barra factor bets over time
- **Execution Quality**: Realized vs. estimated costs

## Module Dependencies

Understanding the dependency graph helps navigate the codebase and debug issues.

### Core Data Flow

```
loaddata.py (no dependencies)
    ↓
calc.py (imports: loaddata, util)
    ↓
regress.py (imports: loaddata, calc, util)
    ↓
Alpha Strategy Files (imports: loaddata, calc, regress, util)
    ↓
opt.py (imports: loaddata, calc, util)
    ↓
Simulation Engines (imports: loaddata, calc, opt, util, regress)
```

### Dependency Details

#### Level 0: Foundation (No Internal Dependencies)
- `loaddata.py` - Pure data loading, only external libraries
- `util.py` - Helper functions, minimal dependencies

#### Level 1: Calculations (Depends on: loaddata, util)
- `calc.py` - Imports: loaddata, util
  - Functions: `calc_vol`, `calc_forward_rets`, `calc_factors`, `calc_intra_factors`
  - Used by: Nearly all modules

#### Level 2: Analysis (Depends on: loaddata, calc, util)
- `regress.py` - Imports: loaddata, calc, util
  - Functions: `regress_alpha`, `regress_factors`, `regress_daily_multi`
  - Used by: All alpha strategies

- `pca.py` - Imports: loaddata, calc, util
  - Functions: `calc_pca_daily`, `calc_pca_intra`
  - Used by: pca_generator strategies

#### Level 3: Alpha Strategies (Depends on: loaddata, calc, regress, util)
All strategy files import the core stack:
```python
from loaddata import *
from calc import *
from regress import *
from util import *
```

**Strategy Groups**:
- High-Low: `hl.py`, `hl_intra.py`, `qhl_*.py`
- Beta-Adjusted: `bd.py`, `bd1.py`, `bd_intra.py`, `badj_*.py`
- Analyst: `analyst.py`, `analyst_badj.py`, `rating_diff*.py`
- Earnings: `eps.py`, `target.py`
- Volume: `vadj*.py`
- Other: `c2o.py`, `mom_year.py`, `ebs.py`, `htb.py`, `rrb.py`

#### Level 4: Optimization (Depends on: loaddata, calc, util)
- `opt.py` - Imports: loaddata, calc, util
  - Functions: `optimize_cplex`, `optimize_alpha`
  - Used by: All simulation engines
  - External: OpenOpt, FuncDesigner

- `bsim_weights.py` - Imports: loaddata, opt, util
  - Functions: Grid search weight optimization
  - Uses: BSIM engine internally

#### Level 5: Simulation (Depends on: All Above)
- `bsim.py` - Imports: loaddata, calc, regress, opt, util
  - Main simulation orchestrator
  - Loads alpha forecasts from strategy outputs

- `osim.py` - Imports: loaddata, calc, opt, util
  - Order-level execution analysis

- `qsim.py` - Imports: loaddata, calc, opt, util, regress
  - Intraday bar simulation

- `ssim.py` - Imports: loaddata, calc, regress, opt, util
  - Full lifecycle tracking

#### Production Modules (Depends on: loaddata, calc, regress, util)
- `prod_sal.py` - Analyst estimate production pipeline
- `prod_eps.py` - Earnings signal production pipeline
- `prod_rtg.py` - Rating signal production pipeline
- `prod_tgt.py` - Target signal production pipeline

### Import Best Practices

1. **Circular Dependencies**: None identified in current codebase
   - Clean hierarchical structure prevents cycles

2. **Wildcard Imports**: Common pattern `from module import *`
   - Used throughout for convenience
   - Be aware of namespace pollution
   - Key functions documented in each module

3. **Module Loading Order**:
   ```python
   # Correct order for manual imports
   import loaddata
   import util
   import calc
   import regress
   import opt
   # Then strategy or simulation modules
   ```

4. **External Dependencies**:
   - **OpenOpt/FuncDesigner**: Only in `opt.py` and `bsim_weights.py`
   - **scikit-learn**: Used in `calc.py` for PCA
   - **statsmodels**: Used in `regress.py` for WLS
   - **MySQL**: Only in `loaddata.py` if using SQL data sources

### Salamander Module Dependencies

**Simplified Structure** (no cross-dependencies with main codebase):

```
salamander/loaddata.py (standalone)
    ↓
salamander/calc.py
    ↓
salamander/regress.py
    ↓
salamander/opt.py
    ↓
salamander/simulation.py
    ↓
salamander/bsim.py, osim.py, qsim.py, ssim.py
```

**Key Difference**: Salamander uses `simulation.py` as shared library instead of duplicating simulation code.

## Configuration

### Universe Parameters

Edit in `loaddata.py`:

```python
# Tradable universe
t_low_price = 2.0
t_high_price = 500.0
t_min_advp = 1000000.0  # $1M min ADV

# Expandable universe
e_low_price = 2.25
e_high_price = 500.0
e_min_advp = 5000000.0  # $5M min ADV

# Universe size
uni_size = 1400  # Top N by market cap
```

### Optimization Parameters

Edit in `opt.py`:

```python
max_sumnot = 50.0e6      # $50M max notional
max_posnot = 0.0048      # 0.48% max per position
kappa = 4.3e-5           # Risk aversion

# Slippage model
slip_alpha = 1.0         # Base cost
slip_beta = 0.6          # Participation power
slip_delta = 0.25        # Participation coefficient
slip_nu = 0.14           # Market impact
execFee = 0.00015        # 1.5 bps execution fee
```

### Factor Configuration

Edit in `calc.py`:

```python
BARRA_FACTORS = ['country', 'growth', 'size', 'sizenl',
                 'divyild', 'btop', 'earnyild', 'beta',
                 'resvol', 'betanl', 'momentum', 'leverage',
                 'liquidty']

PROP_FACTORS = ['srisk_pct_z', 'rating_mean_z']
```

## Project Structure

### File Inventory

The codebase contains **88 Python files** (~35,000 lines) organized into the following categories:

#### Core Infrastructure (4 files)
- `loaddata.py` (1,135 lines) - Data loading from CSV/SQL, universe filtering, HDF5 caching
- `calc.py` (1,401 lines) - Forward returns, volume profiles, winsorization, Barra factor calculations
- `regress.py` (489 lines) - Weighted least squares regression for alpha factor fitting
- `util.py` (585 lines) - Data merging, filtering, and I/O helper functions

#### Simulation Engines (4 files)
- `bsim.py` (730 lines) - Daily rebalancing backtest with portfolio optimization
- `osim.py` (640 lines) - Order-level execution simulator with fill strategy analysis
- `qsim.py` (531 lines) - Intraday 30-minute bar simulation for high-frequency strategies
- `ssim.py` (585 lines) - Full lifecycle simulator with position and cash tracking

#### Portfolio Optimization (3 files)
- `opt.py` (707 lines) - OpenOpt NLP solver with factor risk and transaction costs
- `bsim_weights.py` (246 lines) - Multi-alpha weight optimization using grid search
- `pca.py` (307 lines) - Principal component decomposition for market-neutral returns

#### Alpha Strategies (35 files, 7 families)

**High-Low Mean Reversion** (6 files):
- `hl.py` (372 lines) - Base high-low strategy with daily + intraday signals
- `hl_intra.py` (183 lines) - Intraday-only variant
- `qhl_intra.py` (183 lines) - Quote-level intraday variant
- `qhl_multi.py` (161 lines) - Multi-period daily signals
- `qhl_both.py` (181 lines) - Combined daily + intraday
- `qhl_both_i.py` (182 lines) - Industry-specific variant

**Beta-Adjusted Order Flow** (9 files):
- `bd.py` (734 lines) - Base beta-adjusted strategy with order imbalance
- `bd1.py` (166 lines) - Simplified daily variant
- `bd_intra.py` (215 lines) - Intraday order flow signals
- `badj_multi.py` (165 lines) - Multi-lag return-based variant
- `badj_intra.py` (139 lines) - Intraday return variant
- `badj_both.py` (180 lines) - Combined daily + intraday returns
- `badj_dow_multi.py` (165 lines) - Day-of-week specific model
- `badj2_multi.py` (165 lines) - Alternative beta adjustment
- `badj2_intra.py` (139 lines) - Alternative intraday variant

**Analyst Signals** (4 files):
- `analyst.py` (313 lines) - Base analyst rating/estimate strategy
- `analyst_badj.py` (306 lines) - Beta-adjusted analyst signals
- `rating_diff.py` (222 lines) - Rating change momentum
- `rating_diff_updn.py` (189 lines) - Separate up/down revisions

**Earnings & Valuation** (3 files):
- `eps.py` (146 lines) - Post-earnings announcement drift (PEAD)
- `target.py` (218 lines) - Analyst price target deviations
- `prod_tgt.py` (252 lines) - Production target strategy

**Volume-Adjusted** (5 files):
- `vadj.py` (226 lines) - Base volume-return interaction strategy
- `vadj_multi.py` (204 lines) - Daily-only multi-period variant
- `vadj_intra.py` (146 lines) - Intraday volume signals
- `vadj_pos.py` (197 lines) - Position sizing emphasis
- `vadj_old.py` (155 lines) - Legacy implementation (deprecated)

**PCA & Residuals** (3 files):
- `pca_generator.py` (80 lines) - Intraday PCA residual extraction
- `pca_generator_daily.py` (81 lines) - Daily PCA with exponential weighting
- `rrb.py` (157 lines) - Barra residual return betting

**Other Specialized** (5 files):
- `c2o.py` (217 lines) - Close-to-open gap trading
- `mom_year.py` (92 lines) - 232-day momentum strategy
- `ebs.py` (221 lines) - Analyst estimate revision signals
- `htb.py` (116 lines) - Hard-to-borrow fee rate strategy
- `badj_rating.py` (Unknown) - Beta-adjusted rating strategy

#### Production Modules (4 files)
- `prod_sal.py` (297 lines) - Production estimate signal generator
- `prod_eps.py` (336 lines) - Production earnings signal generator
- `prod_rtg.py` (311 lines) - Production rating signal generator
- `prod_tgt.py` (252 lines) - Production target signal generator

#### Testing & Utilities (8+ files)
- `bigsim_test.py` - Testing framework for bsim
- `osim_simple.py` - Simplified order simulator
- `osim2.py` - Alternative order simulator
- `readcsv.py` - CSV data validation
- `dumpall.py` - Bulk data export utility
- `factors.py` - Factor analysis utilities
- `slip.py` - Slippage model testing
- `setup.py` - Cython build configuration
- Additional utilities: `new1.py`, `other.py`, `other2.py`, `rev.py`, `bsz.py`, `bsz1.py`, `load_data_live.py`

#### Salamander Module (24 files, Python 3 compatible)

**Core Infrastructure** (5 files):
- `loaddata.py` (287 lines) - CSV-based data loading
- `loaddata_sql.py` (317 lines) - SQL database integration
- `calc.py` (473 lines) - Factor calculations (simplified)
- `regress.py` (165 lines) - Regression fitting
- `util.py` (260 lines) - 18 utility functions

**Simulation Engines** (4 files):
- `bsim.py` (260 lines) - Standalone daily simulator
- `osim.py` (224 lines) - Standalone order simulator
- `qsim.py` (409 lines) - Standalone intraday simulator
- `ssim.py` (271 lines) - Standalone lifecycle simulator
- `simulation.py` (382 lines) - Core simulation library

**Optimization** (1 file):
- `opt.py` (379 lines) - Simplified portfolio optimization

**Workflow Generators** (3 files):
- `gen_dir.py` (17 lines) - Directory structure generator
- `gen_hl.py` (23 lines) - HL signal generator
- `gen_alpha.py` (32 lines) - Alpha file extractor

**Strategy Implementations** (2 files):
- `hl.py` (124 lines) - High-low prototype
- `hl_csv.py` (153 lines) - Production HL with CSV data

**Utilities & Validation** (9 files):
- `change_hl.py` (12 lines) - HDF5 date format converter
- `check_hl.py` (25 lines) - HL signal validator
- `check_all.py` (19 lines) - HDF5 dataset inspector
- `change_raw.py` (118 lines) - Raw data augmentation
- `mktcalendar.py` (20 lines) - US trading calendar
- `get_borrow.py` (18 lines) - Borrow rate aggregator
- `show_borrow.py` (9 lines) - Borrow data inspector
- `show_raw.py` (25 lines) - Raw data inspector
- `README.md` (451 lines) - Comprehensive module documentation

### Directory Structure

```
statarb/
├── README.md                     # Project overview and guide
├── CLAUDE.md                     # AI assistant instructions
├── LOG.md                        # Documentation changelog
├── requirements.txt              # Python 2.7 dependencies
├── setup.py                      # Cython optimization build
│
├── Core Infrastructure/
│   ├── loaddata.py              # Data loading & universe filtering
│   ├── calc.py                  # Returns & factor calculations
│   ├── regress.py               # Alpha coefficient fitting
│   ├── util.py                  # Helper functions
│
├── Simulation Engines/
│   ├── bsim.py                  # Daily rebalancing backtest
│   ├── osim.py                  # Order-level execution
│   ├── qsim.py                  # Intraday 30-min bars
│   ├── ssim.py                  # Full lifecycle tracking
│
├── Portfolio Optimization/
│   ├── opt.py                   # NLP optimizer
│   ├── bsim_weights.py          # Weight optimization
│   ├── pca.py                   # PCA decomposition
│
├── Alpha Strategies/
│   ├── High-Low/                # 6 mean reversion variants
│   ├── Beta-Adjusted/           # 9 order flow variants
│   ├── Analyst/                 # 4 fundamental signal variants
│   ├── Earnings/                # 3 event-driven variants
│   ├── Volume/                  # 5 liquidity-aware variants
│   ├── PCA/                     # 3 residual variants
│   └── Other/                   # 5 specialized strategies
│
├── Production/
│   ├── prod_sal.py              # Estimate signal production
│   ├── prod_eps.py              # Earnings signal production
│   ├── prod_rtg.py              # Rating signal production
│   └── prod_tgt.py              # Target signal production
│
├── Testing & Utilities/         # ~13 testing/validation scripts
│
├── plan/                        # Documentation plans (11 files)
│
└── salamander/                  # Python 3 standalone module
    ├── README.md                # Module documentation
    ├── requirements.txt         # Python 3 dependencies
    ├── Core/                    # 6 infrastructure files
    ├── Simulation/              # 5 engine files
    ├── Generators/              # 3 workflow files
    ├── Strategies/              # 2 HL implementations
    └── Utilities/               # 9 validation scripts
```

## Salamander Module

The `salamander/` directory contains a standalone, simplified version of the system for easier deployment and development.

### Features

- Modular directory structure
- Simplified alpha generation pipeline
- Standalone backtest engine
- Documented workflow in `instructions.txt`

### Usage

```bash
# 1. Create directory structure
python3 salamander/gen_dir.py --dir=/path/to/data

# 2. Generate alpha signals from raw data
python3 salamander/gen_hl.py \
    --start=20100630 \
    --end=20130630 \
    --dir=/path/to/data

# 3. Create alpha signal files
python3 salamander/gen_alpha.py \
    --start=20100630 \
    --end=20130630 \
    --dir=/path/to/data

# 4. Run backtest
python3 salamander/bsim.py \
    --start=20130101 \
    --end=20130630 \
    --dir=/path/to/data \
    --fcast=hl:1:1
```

### Directory Structure

```
data/
├── all/          # Alpha signal files
├── hl/           # High-low strategy files
├── locates/      # Short borrow data (borrow.csv)
├── opt/          # Optimization outputs
├── blotter/      # Trade records
├── raw/          # Raw market data
└── all_graphs/   # Visualization outputs
```

## Production Deployment

### Production Modules

Four dedicated production modules generate alpha signals for live trading:

1. **`prod_sal.py`** - Analyst Estimate Signals
   - Data: I/B/E/S analyst estimates
   - Signal: Estimate revisions and dispersion
   - Frequency: Daily updates
   - Output: `sal` forecast column

2. **`prod_eps.py`** - Earnings Signals
   - Data: Earnings announcements and surprises
   - Signal: Post-earnings announcement drift (PEAD)
   - Frequency: Quarterly (event-driven)
   - Output: `eps` forecast column

3. **`prod_rtg.py`** - Rating Signals
   - Data: Analyst rating changes
   - Signal: Rating revisions with cubed amplification
   - Frequency: Event-driven (rating changes)
   - Output: `rtg` forecast column

4. **`prod_tgt.py`** - Price Target Signals
   - Data: Analyst price targets
   - Signal: Target deviations from current price
   - Frequency: Daily updates
   - Output: `tgt` forecast column

### Production Workflow

```
1. Data Ingestion
   ├── Download IBES estimates → ESTIMATES_BASE_DIR
   ├── Download earnings data → EARNINGS_BASE_DIR
   ├── Download price/volume → PRICE_BASE_DIR
   └── Download Barra factors → BARRA_BASE_DIR

2. Signal Generation (Daily)
   ├── prod_sal.py --start=TODAY --end=TODAY → all/sal.h5
   ├── prod_eps.py --start=TODAY --end=TODAY → all/eps.h5
   ├── prod_rtg.py --start=TODAY --end=TODAY → all/rtg.h5
   └── prod_tgt.py --start=TODAY --end=TODAY → all/tgt.h5

3. Portfolio Optimization
   └── opt.py with combined forecasts → target positions

4. Order Generation
   └── Compare target vs current positions → order list

5. Execution
   └── Send orders to broker/execution system

6. Monitoring
   ├── Track fill prices vs estimates
   ├── Monitor factor exposures
   └── Calculate realized P&L
```

### Environment Configuration

#### Production Settings (`loaddata.py`)

```python
# Production universe (more conservative)
t_low_price = 5.0           # Higher min price
t_high_price = 500.0
t_min_advp = 2000000.0      # Higher liquidity requirement
uni_size = 1000             # Smaller universe (top 1000)

# Expandable universe
e_min_advp = 10000000.0     # Much higher for signal generation
```

#### Optimization Settings (`opt.py`)

```python
# Production risk controls
kappa = 4.3e-5              # Conservative risk aversion
max_sumnot = 50.0e6         # $50M capital
max_posnot = 0.0048         # Max 0.48% per position
max_participation = 0.015   # Max 1.5% of ADV

# Realistic transaction costs
slip_nu = 0.18              # Higher market impact
execFee = 0.00015           # 1.5 bps execution fee
```

#### Monitoring Configuration

```python
# Alert thresholds
MAX_DRAWDOWN = 0.05         # 5% max drawdown
MAX_FACTOR_EXPOSURE = 0.5   # Max factor bet
MAX_INDUSTRY_EXPOSURE = 0.1 # Max 10% in one industry
MIN_SHARPE = 1.5            # Minimum acceptable Sharpe
```

### Production Checklist

#### Daily Operations
- [ ] Verify data freshness (prices, Barra, estimates)
- [ ] Run production signal generators (4 modules)
- [ ] Check signal distributions (no extreme outliers)
- [ ] Run portfolio optimization
- [ ] Review target positions vs. current
- [ ] Validate factor exposures (neutral to Barra factors)
- [ ] Generate order list
- [ ] Review slippage estimates
- [ ] Execute orders
- [ ] Monitor fills and update positions
- [ ] Calculate EOD P&L
- [ ] Archive results and logs

#### Weekly Reviews
- [ ] Analyze Sharpe ratio trend
- [ ] Review factor exposures over time
- [ ] Check alpha decay (are signals still predictive?)
- [ ] Validate transaction cost estimates vs. realized
- [ ] Review largest winners/losers
- [ ] Update universe (corporate actions, delistings)

#### Monthly Tasks
- [ ] Refit regression coefficients (out-of-sample drift)
- [ ] Backtest recent period (validation)
- [ ] Review alpha combination weights
- [ ] Analyze strategy attribution (which alphas contributing?)
- [ ] Update risk model (recalculate factor covariances)

### Production Best Practices

1. **Data Quality**:
   - Validate all data sources before optimization
   - Check for missing values, outliers, stale data
   - Maintain audit trail of data versions

2. **Signal Validation**:
   - Monitor signal distributions (mean, std, extremes)
   - Check for regime changes or structural breaks
   - Compare current vs. historical signal characteristics

3. **Risk Management**:
   - Hard position limits in optimizer (cannot be exceeded)
   - Real-time factor exposure monitoring
   - Circuit breakers for extreme market conditions
   - Diversification across strategy families

4. **Execution Quality**:
   - Compare realized vs. estimated slippage
   - Track implementation shortfall
   - Monitor adverse selection in fills
   - Analyze execution timing (VWAP vs. close)

5. **System Reliability**:
   - Automated data pipeline with fallbacks
   - Redundant optimization runs (validate consistency)
   - Alert system for failures or anomalies
   - Manual review gate before order submission

6. **Performance Attribution**:
   - Decompose P&L by strategy family
   - Track alpha vs. risk vs. costs
   - Identify alpha decay patterns
   - Adjust weights based on recent performance

### Disaster Recovery

1. **Data Loss**:
   - Maintain backups of all historical data
   - Cache critical files (Barra, universe, prices)
   - Document data provider contact info

2. **System Failure**:
   - Manual override process documented
   - Backup optimization environment
   - Position reconciliation procedures

3. **Market Events**:
   - Halt trading triggers (volatility spike, flash crash)
   - Emergency liquidation protocol
   - Risk override procedures

## Performance Metrics

### Key Metrics

The system evaluates strategies using:

- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Information Ratio**: Alpha vs. benchmark volatility
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Turnover**: Average daily trading as % of capital
- **Hit Rate**: Percentage of profitable days
- **Factor Exposures**: Bets on Barra risk factors
- **Participation Rate**: Trading volume vs. ADV

### Risk Management

- **Factor Neutrality**: Limits on Barra factor exposures
- **Sector Limits**: Industry concentration constraints
- **Position Sizing**: Market cap and liquidity-based limits
- **Participation Constraints**: Max 1.5% of ADV to minimize impact
- **Correlation Monitoring**: Rolling 30-day cross-security correlations

## Advanced Topics

### Custom Alpha Development

To create a new alpha signal:

1. Create a new Python file (e.g., `my_alpha.py`)
2. Load data using `loaddata.py` functions
3. Calculate your alpha signal
4. Use `regress.py` to fit coefficients on training data
5. Generate out-of-sample forecasts
6. Save to HDF5 or CSV for simulation engines

Example structure:

```python
from loaddata import *
from calc import *
from regress import *

# Load data
daily_df = load_prices(start, end, lookback)
barra_df = load_barra(start, end, lookback)

# Calculate alpha
daily_df['my_alpha'] = calculate_my_signal(daily_df)

# Fit regression
fits_df = regress_alpha(daily_df, 'my_alpha', horizon=3)

# Generate forecast
forecast_df = apply_coefficients(daily_df, fits_df)

# Save results
dump_alpha(forecast_df, 'my_alpha')
```

### Multi-Factor Combination

Combine multiple alphas with optimized weights:

```bash
python bsim.py \
    --start=20130101 \
    --end=20130630 \
    --fcast=pca:1.0:0.3,hl:1.2:0.25,bd:0.8:0.2,analyst:1.5:0.15,mom:1.0:0.1
```

Weights should sum to 1.0 for proper risk attribution.

### Transaction Cost Analysis

The system models realistic costs:

1. **Execution Fees**: 1.5 bps fixed
2. **Slippage**: Nonlinear function of participation rate
3. **Market Impact**: Based on order size vs. ADV
4. **Borrow Costs**: For short positions
5. **Opportunity Cost**: From delayed fills

Analyze realized vs. estimated costs using OSIM engine.

## Technical Debt & Known Issues

### Critical Issues

1. **Python 2.7 End of Life**:
   - Main codebase stuck on Python 2.7 due to OpenOpt dependency
   - OpenOpt no longer maintained (last update: 2014)
   - **Mitigation**: Salamander module provides Python 3 path
   - **Long-term**: Migrate to cvxpy, scipy.optimize, or commercial solver

2. **Incomplete Implementations** (Now Resolved):
   - ✅ `pca_generator.py` - Residual calculation fixed (2026-02-05)
   - ✅ `pca_generator_daily.py` - Residual extraction enabled (2026-02-05)
   - **Status**: PCA residual strategies now fully functional

3. **Code Bugs** (Now Resolved):
   - ✅ Fixed 7 bugs in beta-adjusted strategies (2026-02-05)
     - Variable naming errors in bd1.py
     - Syntax errors in badj_intra.py
     - Undefined variable references in badj2_multi.py, badj2_intra.py
   - ✅ Fixed 2 bugs in hl_intra.py (2026-02-05)
     - Empty DataFrame overwrite causing KeyError
     - Undefined variable 'lag' causing NameError
   - **Status**: All documented runtime bugs fixed

4. **Misleading Filenames**:
   - `ebs.py` - Actually analyst estimates, not equity borrow
   - **Clarification**: "SAL" = Analyst estimates, not short availability
   - **Action**: Consider renaming to `sal.py` for clarity

### Code Quality Issues

1. **Wildcard Imports**:
   - Extensive use of `from module import *`
   - Makes dependency tracking difficult
   - Potential namespace pollution
   - **Best Practice**: Use explicit imports in new code

2. **Limited Error Handling**:
   - Many functions lack try/except blocks
   - Data validation minimal in some modules
   - Can fail silently on bad data
   - **Improvement**: Add data quality checks and error logging

3. **Documentation Gaps** (Now Resolved):
   - ✅ 78 files documented with comprehensive docstrings
   - ✅ All strategy families documented
   - ✅ All simulation engines documented
   - Remaining: Some utility/test files have minimal docs

4. **Hard-Coded Paths**:
   - Some utility scripts have hard-coded file paths
   - Examples: `salamander/change_hl.py`, `salamander/show_borrow.py`
   - **Fix**: Convert to command-line arguments

### Performance Issues

1. **HDF5 Caching**:
   - First load of data slow (CSV parsing)
   - HDF5 cache significantly speeds up subsequent loads
   - Cache invalidation manual (delete .h5 files)
   - **Improvement**: Automatic cache invalidation based on source file mtime

2. **Large Memory Footprint**:
   - Loading full universe (1,400 stocks) for date range can exceed 8GB RAM
   - Bar data especially memory-intensive
   - **Mitigation**: Process in smaller date chunks

3. **Optimization Speed**:
   - OpenOpt solver can be slow (1,500 iterations)
   - Daily optimization takes 5-30 seconds depending on universe size
   - **Alternative**: Commercial solvers (Gurobi, CPLEX) much faster

### Data Quality Issues

1. **Barra Factor Availability**:
   - Requires proprietary Barra data subscription
   - No public alternative readily available
   - **Workaround**: Can use Fama-French factors or custom factor models

2. **IBES Database**:
   - Analyst data requires expensive IBES subscription
   - Analyst strategies non-functional without it
   - **Alternative**: Use free alternatives (Yahoo Finance estimates, limited)

3. **Corporate Actions**:
   - Split handling implemented but limited testing
   - Dividend adjustments may have edge cases
   - **Testing Needed**: Comprehensive corporate action test suite

### Scalability Limitations

1. **Single-Threaded**:
   - Most modules single-threaded (Python GIL)
   - Parallel processing limited
   - **Improvement**: Multiprocessing for cross-sectional calculations

2. **Universe Size**:
   - Optimized for ~1,400 stocks
   - Larger universes (3,000+) may hit memory/speed limits
   - **Scaling**: Batch processing or distributed computing

3. **Backtest Length**:
   - Multi-year backtests can take hours
   - Output files can exceed 1GB
   - **Optimization**: Incremental backtests, parallel date ranges

### Security & Production Concerns

1. **No Authentication**:
   - No user authentication system
   - No access controls on data directories
   - **Production**: Implement role-based access control

2. **No Audit Trail**:
   - Limited logging of operations
   - No trade audit trail
   - **Production**: Comprehensive logging and audit system required

3. **No Backtesting Safeguards**:
   - Easy to introduce look-ahead bias
   - Data leakage possible in alpha development
   - **Best Practice**: Strict in-sample/out-of-sample discipline

### Unknown Functionality

Several files have unclear purpose and need investigation:

- `new1.py` - Unknown purpose
- `other.py`, `other2.py` - Unclear functionality
- `rev.py` - Likely reversal strategy, undocumented
- `bsz.py`, `bsz1.py` - Unknown (batch size related?)
- `osim2.py` - Alternative osim implementation?
- `badj_rating.py` - Beta-adjusted ratings (incomplete?)
- `factors.py` - Factor analysis utilities?
- `slip.py` - Slippage testing?

**Action Required**: Code review and documentation or removal if obsolete.

## Contributing

This is a research codebase under active documentation. Key areas for improvement:

### High Priority
- **Python 3 Migration**: Replace OpenOpt with modern solver (cvxpy, scipy)
- **Testing Suite**: Add unit tests and integration tests
- **Data Pipeline**: Modernize data ingestion (APIs instead of CSV)
- **Performance**: Parallelize cross-sectional calculations
- **Documentation**: Complete remaining utility scripts

### Medium Priority
- **Additional Alpha Strategies**: Machine learning alpha generation
- **Enhanced Optimization**: Multi-period optimization, convex relaxations
- **Real-Time Data**: Streaming market data integration
- **Execution**: Improved execution modeling (order book dynamics)
- **Monitoring**: Real-time dashboards and alerting

### Low Priority
- **Web Interface**: Dashboard for backtest visualization
- **Cloud Deployment**: Containerization (Docker) and cloud infrastructure
- **Alternative Data**: Incorporate sentiment, satellite, web scraping
- **Risk Models**: Support for alternative factor models (Fama-French, custom)

### Code Style Guidelines

- Follow PEP 8 for new code (legacy code may not comply)
- Add comprehensive docstrings (NumPy style)
- Include usage examples in docstrings
- Use explicit imports instead of wildcards
- Add type hints in Python 3 code (salamander module)
- Write unit tests for new functions
- Document data requirements and assumptions

## License

Apache 2.0

## Contact

For questions and support, please open an issue on GitHub.

---

**Disclaimer**: This system is for research and educational purposes. Use at your own risk. Past performance does not guarantee future results. Trading involves substantial risk of loss.
