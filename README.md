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

### Requirements

- Python 2.7 (legacy codebase)
- NumPy 1.16.0
- Pandas 0.23.4
- OpenOpt 0.5628
- statsmodels
- scikit-learn
- matplotlib
- lmfit
- MySQL connector (optional, for database access)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/statarb.git
cd statarb

# Install dependencies
pip install -r requirements.txt

# For Cython optimization module (optional)
python setup.py build_ext --inplace
```

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

### Required Data Sources

1. **Universe Files** (`UNIV_BASE_DIR/YYYY/YYYYMMDD.csv`)
   - Columns: `sid`, `ticker_root`, `status`, `country`, `currency`

2. **Price Files** (`PRICE_BASE_DIR/YYYY/YYYYMMDD.csv`)
   - Columns: `sid`, `ticker`, `open`, `high`, `low`, `close`, `volume`, `mkt_cap`

3. **Barra Files** (`BARRA_BASE_DIR/YYYY/YYYYMMDD.csv`)
   - Risk factors: beta, momentum, size, volatility, etc. (13 factors)
   - Industry classifications (58 industries)

4. **Bar Files** (`BAR_BASE_DIR/YYYY/YYYYMMDD.h5`)
   - Intraday 30-minute bars with VWAP and volume
   - Format: HDF5 with MultiIndex (timestamp, sid)

5. **Locates File** (`LOCATES_BASE_DIR/borrow.csv`)
   - Short borrow availability and rates

### Universe Filters

- **Price Range**: $2.00 - $500.00
- **Min ADV**: $1M (tradable) / $5M (expandable universe)
- **Country**: USA
- **Currency**: USD
- **Market Cap**: Top 1,400 stocks by default

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

## Strategies

The system includes 20+ alpha strategies in separate files:

### Signal Types

| Category | Files | Description |
|----------|-------|-------------|
| **PCA** | `pca.py` | Market-neutral returns from PCA decomposition |
| **Beta-Adjusted** | `bd.py`, `badj_*.py` | Order flow signals adjusted for beta |
| **High-Low** | `hl.py`, `qhl_*.py` | Intraday high-low mean reversion |
| **Analyst** | `analyst*.py`, `rating_diff.py` | Analyst rating and estimate changes |
| **Momentum** | `mom_year.py` | Annual momentum signals |
| **Volatility** | `vadj_*.py` | Volume-adjusted position models |
| **Overnight** | `c2o.py` | Close-to-open gap trading |
| **Earnings** | `eps.py`, `target.py` | Earnings surprises and target misses |

### Strategy Development Workflow

1. **Develop Alpha**: Create new strategy file with alpha calculation
2. **Fit Coefficients**: Use `regress.py` to fit on in-sample data
3. **Generate Forecasts**: Apply to out-of-sample period
4. **Optimize**: Run through `opt.py` to get target positions
5. **Backtest**: Simulate with appropriate engine (BSIM/OSIM/QSIM/SSIM)
6. **Analyze**: Evaluate Sharpe, drawdown, factor exposures

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

```
statarb/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Cython build configuration
│
├── loaddata.py              # Data loading and preprocessing
├── calc.py                  # Factor calculations
├── regress.py               # Regression analysis
├── pca.py                   # PCA decomposition
├── opt.py                   # Portfolio optimization
├── util.py                  # Utility functions
│
├── bsim.py                  # Daily simulation engine
├── osim.py                  # Order simulation engine
├── qsim.py                  # Intraday simulation engine
├── ssim.py                  # System simulation engine
│
├── bd.py                    # Beta-adjusted order flow
├── hl.py                    # High-low strategy
├── pca.py                   # PCA alpha generation
├── analyst*.py              # Analyst signal strategies
├── rating_diff.py           # Rating change strategy
├── vadj_*.py               # Volume-adjusted strategies
├── mom_year.py              # Momentum strategy
├── eps.py                   # Earnings surprise strategy
├── target.py                # Price target strategy
├── c2o.py                   # Close-to-open strategy
└── ... (additional strategies)
│
└── salamander/              # Standalone module
    ├── instructions.txt     # Salamander usage guide
    ├── requirements.txt     # Salamander dependencies
    ├── gen_dir.py          # Directory structure generator
    ├── gen_hl.py           # Alpha signal generator
    ├── gen_alpha.py        # Alpha file creator
    ├── bsim.py             # Standalone backtest engine
    ├── simulation.py       # Portfolio simulation
    └── ... (supporting files)
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

## Contributing

This is a research codebase. Key areas for improvement:

- Python 3 migration
- Additional alpha strategies
- Enhanced optimization algorithms
- Real-time data integration
- Machine learning alpha generation
- Improved execution modeling

## License

Apache 2.0

## Contact

For questions and support, please open an issue on GitHub.

---

**Disclaimer**: This system is for research and educational purposes. Use at your own risk. Past performance does not guarantee future results. Trading involves substantial risk of loss.
