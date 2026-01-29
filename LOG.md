# Claude Code Work Log

## 2026-01-29

### Task: Document loaddata.py (COMPLETE)

**Decision rationale:** Chose loaddata.py over suggested "critical unknown files" because:
- Foundational module - first step in data pipeline, everything depends on it
- Highest complexity (959 lines, 25 functions) = highest documentation impact
- Understanding core infrastructure > investigating possibly-deprecated experiments

**Work completed:**
- Added docstrings to all 25 functions in loaddata.py
- Each docstring includes: purpose, args, returns, file dependencies, notes
- Updated plan/01-core-infrastructure-loaddata.md to COMPLETE status

**Functions documented:**
- `get_uni()`: Universe loading with 6-step filtering
- `load_barra()`: Barra USE4S risk factors
- `load_daybars()`, `load_bars()`, `aggregate_bars()`: Intraday bar handling
- `load_prices()`: Daily OHLC with returns, volatility, volume
- `load_volume_profile()`: 20-day median volume profiles
- `load_earnings_dates()`, `load_past_earnings_dates()`: Earnings calendar
- `load_locates()`: Short borrow availability
- `load_*_results()`, `load_mus()`: Alpha signal loading
- `load_qb_*()`: Quantbot order/position/execution data
- `load_cache()`, `load_factor_cache()`: HDF5 cache access
- `transform_barra()`: Industry weight pivot
- `load_ratings_hist()`, `load_target_hist()`, `load_estimate_hist()`: IBES analyst data
- `load_consensus()`: Real-time consensus estimates

**Key discoveries:**
- Universe filtering: USA/USD only, $2-$500 price, $1M+ ADV, top 1400 by mktcap
- Excludes biotech (group 3520)
- Uses sqlite3 for IBES analyst estimate database
- Quantbot integration for live order/position tracking

---

### Task: Document calc.py (COMPLETE)

**Decision rationale:** Chose calc.py as next priority because:
- Next step in data pipeline: loaddata.py → **calc.py** → strategies → regress.py → opt.py
- Highest dependency: All 34 alpha strategy files depend on calc.py functions
- Already had good module docstring, needed function-level docs

**Work completed:**
- Added comprehensive docstrings to all 20 functions in calc.py
- Each docstring includes: purpose, args, returns, mathematical formulas, dependencies, cross-refs
- Updated plan/02-core-infrastructure-calc.md to COMPLETE status

**Functions documented:**
- `calc_forward_returns()`: N-day forward cumulative returns for alpha backtesting
- `calc_vol_profiles()`: 21-day volume profiles at 15-min intervals (26 timeslices)
- `calc_price_extras()`: Volatility ratios, volume turnover, volatility momentum
- `winsorize()`: Outlier capping at N std devs (default 5σ)
- `winsorize_by_date()`, `winsorize_by_ts()`, `winsorize_by_group()`: Cross-sectional winsorization
- `rolling_ew_corr_pairwise()`: Exponentially-weighted pairwise correlations
- `push_data()`: Forward shift (WARNING: can leak future data)
- `lag_data()`: Backward shift (safe for alpha generation)
- `calc_resid_vol()`: 20-day Barra residual volatility (idiosyncratic risk)
- `calc_factor_vol()`: Exponentially-weighted factor covariance matrix (73x73)
- `create_z_score()`: Cross-sectional standardization (mean=0, std=1)
- `calc_factors()`: Daily factor decomposition via WLS regression (13 Barra + 58 industries + 2 proprietary)
- `calc_intra_factors()`: Intraday factor decomposition using 30-min bars
- `factorize()`: WLS regression using lmfit minimization with market cap weighting
- `fcn2min()`: Objective function for lmfit (cap-weighted residuals)
- `mkt_ret()`: Market-cap weighted average return
- `calc_med_price_corr()`: Placeholder (not implemented)

**Key technical insights:**
- Factor decomposition: 73 factors total (13 Barra + 58 industries + 2 proprietary)
- WLS uses log(market cap) as weights
- Country factor constrained as cap-weighted sum of industry returns
- Halflife=20 days for exponential weighting in covariance calculations
- Uses deprecated pandas.stats.moments API (ewmcorr, ewmcov)
- Forward returns use shift(-N) to look ahead for backtesting
- Winsorization default: 5σ (retains ~99.9999% of normal data)

---

### Task: Investigate and Document Critical Unknown Files (COMPLETE)

**Decision rationale:** Highest priority investigation task:
- 5 files with completely unknown purpose (new1.py, other.py, other2.py, bsz.py, bsz1.py)
- Potential dead code cluttering codebase OR critical undocumented strategies
- Marked CRITICAL in documentation plan
- Must be understood before other documentation work

**Investigation findings:**
1. **new1.py** → Insideness alpha strategy
   - Beta-adjusted returns * order book insideness
   - Daily + intraday with time-of-day coefficients
   - Horizon: 3 days, uses WLS regression

2. **other.py** → Volatility ratio alpha strategy
   - log_ret * volat_ratio (realized volatility)
   - Mean reversion with volatility adjustment
   - Industry-demeaned, horizon: 2 days

3. **other2.py** → Simplified insideness alpha
   - log_ret * insideness (NO beta adjustment)
   - Very similar to new1.py but simpler framework
   - Uses alphacalc instead of direct regress module

4. **bsz.py** → Bid-ask size imbalance alpha
   - (AskSize - BidSize) / (BidSize + AskSize) / sqrt(spread)
   - Beta-adjusted returns, 6 intraday time buckets
   - Sophisticated order book microstructure strategy

5. **bsz1.py** → Simplified bid-ask imbalance
   - Same signal as bsz.py but no beta adjustment
   - Sector-specific analysis (Energy vs non-Energy)
   - Generates both raw and market-adjusted forecasts

**Work completed:**
- Comprehensive module docstrings for all 5 files
- Documented methodology, CLI args, outputs, data requirements
- Updated plan/21-critical-unknown-files.md to COMPLETE
- All tasks checked off in plan file

**Recommendations documented:**
- Rename for clarity: new1.py→insd.py, other.py→volat_ratio.py, etc.
- Consider consolidating other2.py with new1.py (both use insideness)
- ALL files are legitimate strategies - NONE should be deleted
- Need performance analysis before deciding which variants to keep

---

### Task: Document load_data_live.py (COMPLETE)

**Decision rationale:** Most critical undocumented production module:
- CRITICAL priority in plan/18-production-modules.md
- Foundation for all production workflows (live data loading)
- Zero documentation - complete knowledge gap for operational continuity
- Other prod_*.py modules depend on this for live data

**Work completed:**
- Added comprehensive 40+ line module docstring
- Documented load_live_file() function with full context
- Added inline documentation for all configuration constants
- Documented commented-out IBES database functions (25+ line explanation)
- Updated plan/18-production-modules.md progress

**Module purpose:**
- Live/production data loading vs. loaddata.py (backtesting)
- Loads real-time bid/ask price data from CSV
- Calculates mid-price (close_i) for execution pricing
- Infrastructure for IBES analyst data (ratings, targets, estimates) - currently inactive

**Key discoveries:**
- Single active function: load_live_file() - loads CSV, calculates mid-price
- Extensive commented infrastructure for IBES database integration
- UNBIAS=3 hours for production timestamp adjustment
- All base directories empty - requires prod environment configuration
- Commented functions complete and tested, ready for activation

**Production context:**
- Called by prod_sal.py, prod_eps.py, prod_rtg.py for signal generation
- Uses same util.py/calc.py as backtesting but different data sources
- Simpler than loaddata.py - relies on pre-filtered universe
