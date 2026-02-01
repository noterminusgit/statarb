# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Statistical arbitrage trading system for US equities. Legacy Python 2.7 codebase that implements alpha generation, portfolio optimization, and backtesting across ~1,400 stocks.

## Task Execution Strategy

**IMPORTANT: Use subagents for individual tasks. Keep the main session for oversight and coordination.**

When the user requests work on this codebase:

1. **Main session role**: Act as coordinator/overseer only
   - Break down user requests into discrete tasks
   - Spawn subagents for each task
   - Monitor subagent progress and results
   - Synthesize results and report back to user
   - Do NOT perform implementation work directly

2. **Spawn subagents for**:
   - **Code exploration**: Use `Task` tool with `subagent_type=Explore` to understand codebase structure, find implementations, trace dependencies
   - **Implementation work**: Use `Task` tool with `subagent_type=general-purpose` for writing/editing code, fixing bugs, adding features
   - **Running backtests**: Use `Task` tool with `subagent_type=Bash` for executing simulation commands and analyzing outputs
   - **Documentation**: Use `Task` tool with `subagent_type=general-purpose` for documenting modules or generating reports
   - **Analysis**: Use `Task` tool with `subagent_type=general-purpose` for analyzing simulation results, performance metrics, or code patterns

3. **Examples**:
   - User: "Add a new alpha strategy" → Spawn explore agent to understand strategy structure, then spawn implementation agent to write code
   - User: "Run a backtest with these parameters" → Spawn bash agent to execute and monitor the simulation
   - User: "Optimize the regression module" → Spawn explore agent to understand current implementation, then spawn implementation agent for optimization
   - User: "How does the portfolio optimization work?" → Spawn explore agent to trace through opt.py and related modules

4. **Benefits**:
   - Parallel execution of independent tasks
   - Better context management (each agent focused on one task)
   - Main session maintains high-level view
   - Easier to track progress across multiple workstreams

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

