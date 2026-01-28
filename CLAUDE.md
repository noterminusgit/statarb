# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Statistical arbitrage trading system for US equities. Legacy Python 2.7 codebase that implements alpha generation, portfolio optimization, and backtesting across ~1,400 stocks.

## Commands

### Installation
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace  # Optional: Cython optimization
```

### Running Backtests

**BSIM (Daily simulation):**
```bash
python bsim.py --start=20130101 --end=20130630 --fcast=hl:1:1 --kappa=2e-8 --maxnot=200e6
```

**Multi-alpha combination:**
```bash
python bsim.py --start=20130101 --end=20130630 --fcast=hl:1:0.6,bd:0.8:0.4 --kappa=2e-8
```

**OSIM (Order-level):**
```bash
python osim.py --start=20130101 --end=20130630 --fill=vwap --slipbps=0.0001
```

**QSIM (Intraday 30-min bars):**
```bash
python qsim.py --start=20130101 --end=20130630 --fcast=qhl_intra --horizon=3
```

**SSIM (Full lifecycle):**
```bash
python ssim.py --start=20130101 --end=20131231 --fcast=combined_alpha
```

### Salamander Module (Standalone)
```bash
python3 salamander/gen_dir.py --dir=/path/to/data
python3 salamander/gen_hl.py --start=20100630 --end=20130630 --dir=/path/to/data
python3 salamander/gen_alpha.py --start=20100630 --end=20130630 --dir=/path/to/data
python3 salamander/bsim.py --start=20130101 --end=20130630 --dir=/path/to/data --fcast=hl:1:1
```

## Architecture

**Data Flow:**
```
Raw Data → loaddata.py → calc.py → strategy files → regress.py → opt.py → sim engines
```

**Core Pipeline:**
- `loaddata.py`: Data loading from CSV/SQL, universe filtering, HDF5 caching
- `calc.py`: Forward returns, volume profiles, winsorization, Barra factor calculations
- `regress.py`: WLS regression fitting alpha factors to forward returns
- `pca.py`: Principal component decomposition for market-neutral returns
- `opt.py`: Portfolio optimization using OpenOpt NLP solver with factor risk and transaction costs
- `util.py`: Data merging and helper functions

**Simulation Engines:**
- `bsim.py`: Daily rebalancing with optimized positions
- `osim.py`: Order-level execution with VWAP/mid/close fills
- `qsim.py`: Intraday 30-minute bar simulation
- `ssim.py`: Full lifecycle position and cash tracking

**Alpha Strategies** (format: `name:multiplier:weight`):
- `hl.py`, `qhl_*.py`: High-low mean reversion
- `bd.py`, `badj_*.py`: Beta-adjusted order flow
- `pca.py`: PCA decomposition signals
- `analyst*.py`, `rating_diff.py`: Analyst ratings/estimates
- `eps.py`, `target.py`: Earnings and price targets
- `c2o.py`: Close-to-open gaps
- `vadj_*.py`: Volume-adjusted positions

## Key Configuration

**Universe parameters** in `loaddata.py`:
- Price range: $2-$500, Min ADV: $1M tradable / $5M expandable
- Universe size: Top 1,400 by market cap

**Optimization parameters** in `opt.py`:
- `kappa`: Risk aversion (2e-8 to 4.3e-5)
- `max_sumnot`: Max total notional ($50M default)
- `slip_nu`: Market impact coefficient (0.14-0.18)

**Data paths** must be configured in `loaddata.py`: `UNIV_BASE_DIR`, `PRICE_BASE_DIR`, `BARRA_BASE_DIR`, `BAR_BASE_DIR`, `EARNINGS_BASE_DIR`, `LOCATES_BASE_DIR`, `ESTIMATES_BASE_DIR`

## Notes

- Python 2.7 legacy codebase (salamander module supports Python 3)
- Uses OpenOpt/FuncDesigner for optimization (older library)
- Data stored in CSV and HDF5 formats
- Barra risk model with 13 factors and 58 industry classifications
