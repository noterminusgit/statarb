# Salamander Module

## Overview

The **salamander** module is a standalone Python 3 implementation of the statistical arbitrage trading system. It provides a simplified, self-contained backtesting environment that is independent of the main Python 2.7 codebase.

### Key Features

- **Python 3 Compatible**: Fully functional with Python 3.x (tested with Python 3.6+)
- **Standalone Architecture**: Self-contained data handling and processing
- **Simplified Workflow**: Streamlined data generation and simulation pipeline
- **Rapid Prototyping**: Ideal for testing new strategies and parameters quickly
- **Portable**: Minimal dependencies, easy to deploy on different systems

### When to Use Salamander vs Main Codebase

**Use Salamander when:**
- You need Python 3 compatibility
- Running quick backtests with minimal setup
- Prototyping new alpha strategies
- Working with a simplified data pipeline
- Teaching or demonstrating the system

**Use Main Codebase when:**
- Running production-grade backtests
- Need access to all alpha strategies (analyst ratings, earnings, targets, etc.)
- Require full Barra factor models and industry classifications
- Working with complex multi-alpha combinations
- Need detailed performance analytics and reporting

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Setup

1. Install required dependencies:

```bash
cd salamander/
pip install -r requirements.txt
```

2. Create the directory structure:

```bash
python3 gen_dir.py --dir=/path/to/your/workspace
```

This creates the following directory structure under `/path/to/your/workspace/data/`:
- `all/` - Processed market data (HDF5 files)
- `all_graphs/` - Visualization outputs
- `hl/` - High-Low alpha signals
- `locates/` - Borrow/locate data
- `raw/` - Raw market data input
- `blotter/` - Trade execution records
- `opt/` - Optimization results

## Data Preparation Workflow

### Step 1: Prepare Raw Data

Download raw market data for your backtest period plus one year of prior data (needed for historical calculations). Place raw data files in the `raw/` directory created by `gen_dir.py`.

**Important:** Download only fully updated raw data folders (spearmint colored in the shared drive).

### Step 2: Generate 'all' Files

Process raw data into HDF5 format with calculated features:

```bash
python3 gen_hl.py --start=20100630 --end=20130630 --dir=/path/to/your/workspace
```

**Parameters:**
- `--start`: Start date in YYYYMMDD format (must be 1 year before your backtest period)
- `--end`: End date in YYYYMMDD format
- `--dir`: Root directory specified in gen_dir.py

This generates `all.YYYYMMDD-YYYYMMDD.h5` files in the `data/all/` directory containing:
- Price and volume data
- Barra risk factors (growth, size, divyild, btop, momentum)
- High-Low mean reversion signals
- Market metadata (splits, dividends, etc.)

Files are split into 6-month periods (Jan-Jun, Jul-Dec).

### Step 3: Prepare Locates Data

Download `borrow.csv` from the borrow_data folder and place it in the `data/locates/` directory.

**Format:** CSV file with columns: symbol, sedol, date, shares, fee

### Step 4: Generate Alpha Signals

Extract alpha signals from the processed 'all' files:

```bash
python3 gen_alpha.py --start=20130101 --end=20130630 --dir=/path/to/your/workspace
```

This creates `alpha.hl.YYYYMMDD-YYYYMMDD.csv` files in the `data/hl/` directory containing the High-Low mean reversion alpha forecasts.

## Running Backtests

### Basic Simulation (bsim.py)

Daily rebalancing simulation with portfolio optimization:

```bash
python3 bsim.py \
  --start=20130101 \
  --end=20130630 \
  --dir=/path/to/your/workspace \
  --fcast=hl:1:1 \
  --kappa=2e-8 \
  --maxnot=200e6
```

**Key Parameters:**

- `--start`: Backtest start date (YYYYMMDD)
- `--end`: Backtest end date (YYYYMMDD)
- `--dir`: Root directory path
- `--fcast`: Alpha signal specification (format: `name:multiplier:weight`)
- `--kappa`: Risk aversion coefficient (2e-8 to 4.3e-5, lower = more aggressive)
- `--maxnot`: Maximum total notional exposure (default: 200M)
- `--maxdollars`: Maximum position size per stock (default: 1M)
- `--slip_nu`: Market impact coefficient (default: 0.18)
- `--slip_beta`: Market impact power (default: 0.6)
- `--locates`: Use borrow constraints (default: True, set to "None" to disable)
- `--horizon`: Return horizon in days (default: 3)

**Alpha Signal Format:**

Single alpha: `--fcast=hl:1:1` (name:multiplier:weight)
- `hl` = signal name
- `1` = multiplier (scales the signal)
- `1` = weight in combination

Multiple alphas: `--fcast=hl:1:0.6,hl:0.8:0.4` (comma-separated)

**Output:**

The simulation prints:
- Daily P&L and returns
- Annualized return and volatility
- Sharpe ratio
- Position and trade details

Generated files:
- `data/opt/opt.hl.YYYYMMDD_HHMMSS.csv` - Optimization results per timestamp
- `data/blotter/blotter.csv` - Trade execution blotter

### Order Simulation (osim.py)

Order-level execution simulation (not fully implemented in salamander):

```bash
python3 osim.py \
  --start=20130101 \
  --end=20130630 \
  --dir=/path/to/your/workspace \
  --fill=vwap \
  --slipbps=0.0001
```

### Intraday Simulation (qsim.py)

30-minute bar intraday simulation (not fully implemented in salamander):

```bash
python3 qsim.py \
  --start=20130101 \
  --end=20130630 \
  --dir=/path/to/your/workspace \
  --fcast=qhl_intra \
  --horizon=3
```

### Full Lifecycle Simulation (ssim.py)

Complete position and cash tracking simulation (not fully implemented in salamander):

```bash
python3 ssim.py \
  --start=20130101 \
  --end=20131231 \
  --dir=/path/to/your/workspace \
  --fcast=combined_alpha
```

## Data Format Requirements

### Raw Data Files

Place in `data/raw/` directory. Expected format from the shared drive's raw data folders.

### Borrow Data (borrow.csv)

Pipe-delimited CSV in `data/locates/`:

```
symbol|sedol|date|shares|fee
AAPL|2046251|2013-01-03|1000000|0.0025
```

### Alpha Signal Files

Generated by `gen_alpha.py` in `data/hl/`:

```
date,gvkey,hl
2013-01-03,001045,0.00234
2013-01-03,001078,-0.00156
```

Multi-index CSV with columns:
- `date`: Trading date
- `gvkey`: Security identifier
- `hl`: Alpha forecast value

### Processed Data Files (all.*.h5)

HDF5 format in `data/all/`, generated by `gen_hl.py`. Contains DataFrames with multi-index (date, gvkey) and columns:
- Price data: `symbol`, `close`, `open`, `volume`
- Volume metrics: `med_volume_21`, `mdvp`, `volat_21`
- Market data: `mkt_cap`, `split`, `div`, `sedol`
- Barra factors: `growth`, `size`, `divyild`, `btop`, `momentum`
- Industry: `ind1` (numeric industry code)
- Returns: `log_ret`, `overnight_log_ret`
- Alpha signals: `hl` (High-Low mean reversion)

## Architecture

### Core Pipeline

```
Raw Data → gen_hl.py → all.*.h5 files → gen_alpha.py → alpha.*.csv files → bsim.py → Results
```

### Module Organization

**Data Loading & Processing:**
- `loaddata.py` - Load HDF5 and CSV data, apply locates constraints
- `loaddata_sql.py` - SQL database data loading (alternative)
- `calc.py` - Calculate forward returns, volume profiles, winsorization, Barra factors

**Optimization & Strategy:**
- `opt.py` - Portfolio optimization using OpenOpt NLP solver
- `regress.py` - Regression analysis for alpha factors
- `hl.py` - High-Low mean reversion strategy implementation
- `hl_csv.py` - Process raw data into HL signals and Barra factors

**Simulation Engines:**
- `simulation.py` - Base simulation class (appears to be a portfolio rebalancing simulator)
- `bsim.py` - Daily backtesting simulation with optimization
- `osim.py` - Order-level execution simulation
- `qsim.py` - Intraday 30-minute bar simulation
- `ssim.py` - Full system lifecycle simulation

**Data Generation:**
- `gen_dir.py` - Create directory structure
- `gen_hl.py` - Generate all.*.h5 files from raw data
- `gen_alpha.py` - Extract alpha signals to CSV

**Utilities:**
- `util.py` - Helper functions for data merging and manipulation
- `mktcalendar.py` - Market calendar utilities
- `check_hl.py` - Validate HL signal files
- `check_all.py` - Validate all.*.h5 files
- `change_hl.py` - Modify HL parameters
- `change_raw.py` - Transform raw data format
- `show_borrow.py` - Display borrow data
- `show_raw.py` - Display raw data
- `get_borrow.py` - Fetch borrow data

### Key Differences from Main Codebase

**Simplified:**
- Only 5 Barra factors (vs 13 in main)
- No industry classifications (vs 58 in main)
- Limited alpha strategies (HL only by default vs 15+ in main)
- Simplified data loading (no complex caching)
- Fewer configuration options

**Python 3 Compatible:**
- Updated pandas/numpy API calls
- Print statements use function syntax
- Dictionary iteration updated for Python 3

**Standalone:**
- Independent data directory structure
- Self-contained data generation scripts
- No dependency on main codebase modules

**Modified:**
- Universe filtering simplified
- Reduced factor model complexity
- Streamlined optimization parameters

## Configuration

### Universe Parameters (in gen_hl.py / hl_csv.py)

Default filtering criteria:
- Price range: $2 - $500
- Minimum market cap: $1.6B
- Excludes: Pharma industry (ind1 == 3520)
- Top stocks by market cap and liquidity

### Optimization Parameters (in opt.py and bsim.py)

Default settings:
- `kappa`: 2e-8 (risk aversion)
- `max_sumnot`: 200M (total notional)
- `max_expnot`: 0.04 (4% of portfolio per exposure)
- `max_posnot`: 0.0048 (0.48% per position)
- `slip_nu`: 0.18 (market impact coefficient)
- `slip_beta`: 0.6 (market impact power)
- `slip_alpha`: 1.0
- `slip_delta`: 0.25
- `slip_gamma`: 0.3
- `execFee`: 0.00015 (1.5 bps execution fee)

### Barra Factors (in calc.py)

Reduced set compared to main codebase:
- `growth` - Earnings growth
- `size` - Market capitalization
- `divyild` - Dividend yield
- `btop` - Book-to-price ratio
- `momentum` - Price momentum

## Performance Metrics

Backtests output:
- **Total P&L**: Cumulative profit/loss in dollars
- **Daily P&L**: Per-day profit/loss
- **Daily Return**: Daily P&L / total absolute notional
- **Annualized Return**: Geometric return scaled to 252 trading days
- **Annualized Volatility**: Standard deviation of daily returns × sqrt(252)
- **Sharpe Ratio**: Annualized return / annualized volatility (assumes 0% risk-free rate)

## Troubleshooting

### "insufficient 'all...' files"

Run `gen_hl.py` with correct date range covering your backtest period plus 1 year prior.

### "No data for [date]"

Check that:
1. Raw data exists for that date
2. Date falls within a market trading day
3. `all.*.h5` files cover the date range

### Import errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory issues

For large backtests:
- Reduce date range
- Increase system RAM
- Use `--fast=True` to skip some optimization iterations

### Optimization fails to converge

Try adjusting:
- Increase `--maxiter` (default: 1500)
- Adjust `--kappa` (risk aversion)
- Check data quality for NaN values

## Example Workflow

Complete example from scratch:

```bash
# 1. Create directories
python3 gen_dir.py --dir=/home/user/trading

# 2. Place raw data in /home/user/trading/data/raw/

# 3. Generate processed data (need data from 2010 for 2013 backtest)
python3 gen_hl.py --start=20100630 --end=20130630 --dir=/home/user/trading

# 4. Download borrow.csv to /home/user/trading/data/locates/

# 5. Generate alpha signals
python3 gen_alpha.py --start=20130101 --end=20130630 --dir=/home/user/trading

# 6. Run backtest
python3 bsim.py \
  --start=20130101 \
  --end=20130630 \
  --dir=/home/user/trading \
  --fcast=hl:1:1 \
  --kappa=2e-8 \
  --maxnot=200e6

# 7. Results are printed to console and saved to:
#    /home/user/trading/data/opt/opt.hl.*.csv
#    /home/user/trading/data/blotter/blotter.csv
```

## Dependencies

See `requirements.txt` for specific versions. Key dependencies:

- **numpy** (1.14.3) - Numerical computing
- **pandas** (0.23.0) - Data manipulation
- **scipy** (1.1.0) - Scientific computing
- **tables** (3.4.4) - HDF5 file support
- **openopt** (0.5628) - Optimization solver
- **FuncDesigner** (0.5627) - Optimization modeling (required by openopt)
- **lmfit** (0.9.10) - Curve fitting
- **matplotlib** (2.2.2) - Visualization
- **mysql-connector** (2.1.6) - SQL database access (optional)
- **python-dateutil** (2.7.3) - Date parsing

## Notes

- The salamander module is designed for **research and prototyping**, not production trading
- Performance may differ from the main codebase due to simplified models
- Only the HL (High-Low) alpha strategy is fully implemented by default
- For production use, refer to the main Python 2.7 codebase
- Some simulation engines (osim, qsim, ssim) may not be fully functional

## Support

For questions or issues:
1. Review `instructions.txt` for basic usage
2. Check the main codebase documentation in `/CLAUDE.md`
3. Examine the module docstrings in individual `.py` files

## Related Documentation

- `/CLAUDE.md` - Main codebase overview and architecture
- `/salamander/instructions.txt` - Original brief usage instructions
- Main codebase modules for comparison: `bsim.py`, `loaddata.py`, `calc.py`, `opt.py`
