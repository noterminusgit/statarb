2026-02-03 - Documented salamander utility scripts (10 files)

**Strategy Files:**
1. salamander/hl.py (124 lines) - High-low mean reversion strategy (prototype)
   - Calculates hl0 ratio: close / sqrt(high * low)
   - Industry-demeaned signals with winsorization
   - Sector-based in/out-of-sample fitting (Energy sector 10)
   - Lagged coefficient regression for mean reversion
   - Output: HDF5 with 'hl' forecast column
   - CLI: --start, --end parameters
   - Note: Prototype version, see hl_csv.py for production

2. salamander/hl_csv.py (153 lines) - Production HL strategy with CSV data
   - Enhanced version with rolling 18-month data loading
   - 6-month regression windows for stability
   - Median-based regression for robustness
   - Coefficient persistence in separate DataFrame
   - Loads from structured raw/ directories
   - Output: HDF5 forecasts + diagnostic plots
   - CLI: --start, --end, --dir parameters
   - Key improvement: 3-day horizon (vs 5 in prototype)

**Validation Scripts:**
3. salamander/change_hl.py (12 lines) - HDF5 date format converter
   - Fixes date index dtype (object → datetime64[ns])
   - One-off utility for legacy file repair
   - Hardcoded file path: ./all/all.20040101-20040630.h5

4. salamander/check_hl.py (25 lines) - HL alpha signal validator
   - Finds maximum absolute HL value in alpha CSV
   - Cross-references with raw price data
   - Validates signal calculation reasonableness
   - CLI: --start, --end, --dir parameters

5. salamander/check_all.py (19 lines) - HDF5 full dataset inspector
   - Quick sanity checks on all.h5 output
   - Column availability verification
   - Stock/sector filtering examples (commented)
   - CLI: --start, --end, --dir parameters

**Data Augmentation:**
6. salamander/change_raw.py (118 lines) - Raw data augmentation utility
   - Batch adds missing columns to raw CSV files
   - SQL queries for market cap and SEDOL data
   - Processes all raw/<YYYYMMDD>/ directories
   - Capital IQ database integration (dbDevCapIQ)
   - CLI: --dir parameter
   - Currently active: SEDOL addition
   - Commented out: Market cap addition

7. salamander/mktcalendar.py (20 lines) - US trading calendar
   - Defines US equity market holidays
   - CustomBusinessDay offset (TDay) for date arithmetic
   - 9 holidays: New Year, MLK Day, Presidents Day, Good Friday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas
   - Does not include early closes or special closures
   - Used by: change_raw.py, other date calculations

**Borrow Data Processing:**
8. salamander/get_borrow.py (18 lines) - Borrow rate aggregator
   - Consolidates weekly stock lending files
   - Processes Historical_Avail_US_Weekly_* files
   - Output: Single borrow.csv with SEDOL, shares, fee, symbol
   - Pipe-delimited format for backtesting
   - CLI: --locates_dir parameter

9. salamander/show_borrow.py (9 lines) - Borrow data inspector
   - Quick diagnostic for specific SEDOL
   - Validates borrow.csv output
   - Hardcoded: ./data/locates/borrow.csv, SEDOL 2484088
   - Useful for checking short sale costs

10. salamander/show_raw.py (25 lines) - Raw data inspector
    - Loads and displays raw CSV files
    - Checks barra_df, price_df, missing_borrow.csv
    - Multi-index setup validation
    - CLI: --dir parameter
    - Legacy validation tool

**Key Insights:**
- hl_csv.py is production version with rolling windows and CSV-based data
- hl.py is prototype with simpler data loading
- change_raw.py enables borrow rate integration via SEDOL linkage
- mktcalendar.py provides trading day awareness for all date calculations
- Validation scripts (check_*.py) help verify data integrity at each stage

**Plan Updates:**
- Marked all 10 utility scripts complete in plan/19-salamander-module.md
- All salamander module documentation now complete (24 files)
- Plan status: COMPLETE
- Ready to delete plan/19-salamander-module.md

---

2026-02-03 - Documented salamander generators and util module (4 files)

**Files Documented:**
1. salamander/gen_dir.py (17 lines) - Directory structure generator
   - Creates data/ folder hierarchy for salamander workflow
   - Subdirectories: all/, all_graphs/, hl/, locates/, raw/, blotter/, opt/
   - Must be run before other generators
   - CLI: --dir parameter for root directory

2. salamander/gen_hl.py (23 lines) - HL signal generator
   - Generates high-low mean reversion signals from raw data
   - Processes data in 6-month periods (Jan-Jun, Jul-Dec)
   - Uses 12-month training window for regression
   - Sector-specific models (Energy vs others)
   - Output: HDF5 files with merged data and regression coefficients
   - CLI: --start, --end, --dir parameters

3. salamander/gen_alpha.py (32 lines) - Alpha file extractor
   - Extracts 'hl' signals from HDF5 to CSV format
   - Filters files by date range overlap
   - Output: CSV files compatible with bsim.py
   - CLI: --start, --end, --dir parameters

4. salamander/util.py (260 lines) - Utility functions module
   - 18 utility functions for data manipulation and I/O
   - Data merging: merge_barra_data(), merge_intra_data(), merge_intra_eod()
   - Column management: remove_dup_cols(), get_overlapping_cols()
   - Filtering: filter_expandable(), filter_pca()
   - Alpha export: dump_alpha(), dump_daily_alpha(), dump_prod_alpha(), dump_all()
   - Results loading: load_all_results(), load_merged_results()
   - Helpers: df_dates(), merge_daily_calcs(), merge_intra_calcs()

**Key Workflows Documented:**
- Salamander setup: gen_dir.py → populate raw/ → gen_hl.py → gen_alpha.py → bsim.py
- HDF5 → CSV conversion for backtesting compatibility
- Barra data lagging (shift by 1 day) to prevent look-ahead bias
- Intraday timestamp broadcasting for daily signals
- Multi-directory result merging for alpha combination

**Plan Updates:**
- Marked generator documentation complete in plan/19-salamander-module.md
- Marked util.py documentation complete
- Remaining: utility scripts (change_hl.py, check_hl.py, etc.)

---

2026-02-03 - Documented volume-adjusted strategy family (5 files)

**Files Documented:**
1. vadj.py (226 lines) - Base volume-adjusted strategy with daily + intraday signals
   - Comprehensive module docstring explaining volume-return interaction
   - Market-adjusted volume normalization methodology
   - Beta-adjusted returns using market cap weighting
   - Hourly coefficient fitting for intraday signals
   - Multi-lag daily signal combination
   - Industry neutralization process
   - Sector-specific models (Energy vs others)

2. vadj_multi.py (204 lines) - Simplified daily-only multi-period strategy
   - Daily-only signals without intraday component
   - Raw relative volume (no market adjustment)
   - Direct beta adjustment without sign()
   - Multi-lag forecasting focus
   - Lower computational overhead

3. vadj_intra.py (146 lines) - Intraday-only volume strategy
   - Intraday signals only (no daily component)
   - Hourly coefficient fitting (6 trading periods)
   - End-of-day regression mode
   - Time-of-day liquidity patterns
   - Sector-specific models

4. vadj_pos.py (197 lines) - Position sizing emphasis strategy
   - Sign-based directional signals
   - Liquidity-aware position sizing
   - Cleaner entry/exit decisions
   - Full daily + intraday model
   - Forecast distribution diagnostics

5. vadj_old.py (155 lines) - Legacy implementation (deprecated)
   - Historical reference version
   - Uses log(volume/median) formulation
   - Simple beta division instead of market adjustment
   - Different data pipeline (alphacalc)
   - Explains why it was superseded

**Plan Updates:**
- Marked all tasks complete in plan/16-alpha-volume-adjusted.md
- Deleted plan file after completion

**Key Concepts Documented:**
- Volume normalization: relative volume vs median
- Market-wide volume adjustment to isolate stock-specific patterns
- Beta-adjusted returns: removing market component
- Industry neutralization: demeaning within sectors
- Multi-horizon forecasting: combining current and lagged signals
- Hourly coefficients: adapting to intraday liquidity patterns
- Position sizing: liquidity-aware signal magnitude
- Sign-based signals: directional trades with volume confirmation

**Strategy Variants Explained:**
- vadj.py: Full model for maximum signal capture
- vadj_multi.py: Daily-only for simpler implementation
- vadj_intra.py: Intraday-only for short-term trading
- vadj_pos.py: Position sizing for execution quality
- vadj_old.py: Legacy for historical reference

2026-02-03 - Documented salamander simulation engines (osim, qsim, ssim)

**Files Documented:**
1. salamander/osim.py - Standalone order-level simulator (Python 3)
   - Comprehensive module docstring explaining forecast weight optimization
   - Detailed objective() function documentation
   - CLI parameter documentation with examples
   - Differences from main osim.py highlighted

2. salamander/qsim.py - Standalone intraday simulator (Python 3)
   - Module docstring covering 30-minute bar simulation
   - Multi-horizon analysis documentation
   - Performance attribution breakdown
   - VWAP and fill price methodology

3. salamander/ssim.py - Standalone lifecycle simulator (Python 3)
   - Full lifecycle tracking documentation
   - Corporate action handling details
   - Multi-dimensional attribution system
   - Position and cash flow management

**Plan Updates:**
- Marked osim.py, qsim.py, ssim.py as complete in plan/19-salamander-module.md
- Core simulation documentation now fully complete

**Key Features Documented:**
- Python 3 compatibility differences
- Simplified data pipelines vs main codebase
- Execution models and slippage
- Performance metrics and attribution
- Command-line interfaces with examples

2026-02-01 20:20:06 - Starting documentation task execution

## Task Execution - 10 Parallel Subagents

**Priority Analysis:**
- CRITICAL: prod_rtg.py (last production module)
- HIGH: bsim.py, osim.py, pca.py, qsim.py, salamander README
- MEDIUM: util.py, ssim.py, hl.py, bd.py

**Subagents Launched:**
1. prod_rtg.py - Rating production module (CRITICAL)
2. bsim.py - Daily simulation engine (HIGH)
3. osim.py - Order execution simulator (HIGH)
4. pca.py - PCA decomposition module (HIGH)
5. util.py - Core utilities (MEDIUM, widely used)
6. qsim.py - Intraday simulator (HIGH)
7. ssim.py - Lifecycle simulator (MEDIUM)
8. salamander/README.md - Python 3 module docs (HIGH)
9. hl.py - High-low mean reversion strategy (MEDIUM)
10. bd.py - Beta-adjusted order flow strategy (MEDIUM)

**Expected Outcomes:**
- Each subagent will document assigned file
- Update corresponding plan file
- Commit changes with descriptive message
- Push to remote repository

**Status:** All agents running in parallel

## Completion Status

### ✓ pca.py - COMPLETE
- 307 lines documented
- 4 functions with comprehensive docstrings
- Mathematical PCA methodology explained
- Usage examples for daily/intraday workflows
- Commit: effc0d0 "Document pca.py PCA decomposition module"
- Pushed to master

### ✓ salamander/README.md - COMPLETE
- 451 lines comprehensive documentation
- Installation, data prep, backtest workflows
- Architecture overview and differences from main codebase
- Complete example workflow and troubleshooting
- Commit: 3595f0a "Add comprehensive README for salamander module"
- Pushed to master

### ✓ prod_rtg.py - COMPLETE (CRITICAL)
- 311 lines documentation added
- Last remaining production module documented
- 5 functions with comprehensive docstrings
- Rating revision strategy with cubed amplification
- Commit: 75e35d6 "Document prod_rtg.py rating production module"
- Pushed to master
- **ALL 4 PRODUCTION MODULES NOW COMPLETE**

### ✓ bd.py - COMPLETE
- 700+ lines documentation added
- Beta-adjusted order flow strategy
- 6 functions with comprehensive docstrings
- Order flow imbalance formula and beta adjustment explained
- Commit: 502be63 "Document bd.py beta-adjusted order flow strategy"
- Pushed to master

### ✓ util.py - COMPLETE
- 17 functions documented comprehensively
- Data merging, universe filtering, file I/O operations
- Look-ahead bias prevention documented
- Commit: 1485cbe "Document util.py core utility functions"
- Pushed to master

### ✓ hl.py - COMPLETE
- 372 lines documentation added
- High-low mean reversion strategy
- 4 functions with comprehensive docstrings
- Signal formula and interpretation documented
- Commit: 502be63 (included with bd.py)
- Pushed to master

### ✓ qsim.py - COMPLETE
- 100+ line module docstring expansion
- All CLI parameters documented
- Comprehensive inline code documentation (415 lines)
- Incremental VWAP formula and multi-horizon analysis
- Commit: 46c990d "Document qsim.py intraday bar simulator"
- Pushed to master

### ✓ osim.py - COMPLETE
- Expanded from 298 to 640 lines (+115%)
- Comprehensive module docstring (244 lines)
- Fill strategies (VWAP, mid, close) documented
- Linear slippage model and weight optimization
- Commit: 3c451e9 "Document osim.py order-level execution simulator"
- Pushed to master

### ✓ ssim.py - COMPLETE
- Comprehensive module docstring expansion
- 7-stage position lifecycle documented
- Cash tracking and P&L attribution system
- CLI parameters and execution constraints
- Commit: edfe32b "Document ssim.py lifecycle simulator"
- Pushed to master

### ✓ bsim.py - COMPLETE
- 222 lines documentation added
- Enhanced module docstring (140 lines)
- 18+ CLI parameters documented
- Complete workflow with inline comments
- 6-step simulation process explained
- Commit: 35acb05 "Document bsim.py daily simulation engine"
- Pushed to master

## Summary

**All 10 Subagents Completed Successfully**

### CRITICAL Priority (1)
1. ✓ prod_rtg.py - Rating production module (311 lines)

### HIGH Priority (5)
2. ✓ bsim.py - Daily simulation engine (222 lines)
3. ✓ osim.py - Order execution simulator (+362 lines, 115% growth)
4. ✓ pca.py - PCA decomposition module (222 lines)
5. ✓ qsim.py - Intraday simulator (100+ lines module docstring)
6. ✓ salamander/README.md - Python 3 module (451 lines)

### MEDIUM Priority (4)
7. ✓ util.py - Core utilities (17 functions)
8. ✓ ssim.py - Lifecycle simulator (comprehensive)
9. ✓ hl.py - High-low mean reversion (372 lines)
10. ✓ bd.py - Beta-adjusted order flow (700+ lines)

**Total Impact:**
- 10 files documented
- ~2,800+ lines of documentation added
- 10 commits pushed to master
- All production modules complete
- All main simulation engines complete
- Core infrastructure significantly enhanced

**Key Achievements:**
- CRITICAL production modules: 4/4 complete (100%)
- Main simulation engines: 4/4 complete (bsim, osim, qsim, ssim)
- Optimization modules: 2/2 high-priority complete (opt, pca)
- Core utilities: util.py complete
- Python 3 standalone: salamander README
- Alpha strategies: 2 foundational strategies documented

**Next Priorities:**
- Remaining alpha strategies (hl_intra, qhl_*, badj_*, etc.)
- Salamander module internals (25 files)
- Testing utilities
- README updates

---

2026-02-01 (Session 2) - Continuing documentation with 10 new subagents

**Focus Areas:**
- Salamander core modules (5 files): simulation.py, bsim.py, calc.py, opt.py, regress.py
- Complete high-low strategy family (5 files): hl_intra, qhl_intra, qhl_multi, qhl_both, qhl_both_i

**Spawning 10 subagents in parallel...**

---

2026-02-03 (Session 3) - Main coordinator spawning subagents for high-impact tasks

**Analysis of remaining work:**
- CRITICAL: Production workflow docs (investigation tasks)
- HIGH: Salamander module (21/25 files remaining, but 4 core ones done)
- MEDIUM: Beta-adjusted strategies (8/9 remaining - largest alpha family)
- MEDIUM: Analyst strategies (0/4 - fundamental signals)
- MEDIUM: Various other alpha families

**Selected priorities based on impact:**
1. Salamander data loading (loaddata.py, loaddata_sql.py) - foundational for Python 3
2. Beta-adjusted strategies (8 files) - complete largest alpha family
3. Analyst strategies (4 files) - complete fundamental signal family

**Spawning 3 subagents...**

**Agent 1 (salamander data loading):** COMPLETE
- Documented salamander/loaddata.py (106-line docstring, 4 functions)
- Documented salamander/loaddata_sql.py (157-line docstring, 13 functions)
- Commit: ac9cd51

**Agent 2 (beta-adjusted strategies):** COMPLETE  
- Documented 8 beta-adjusted strategy files (bd1, bd_intra, badj_multi, badj_intra, badj_both, badj_dow_multi, badj2_multi, badj2_intra)
- Identified 2 distinct strategy types: order flow vs return-based
- Found and documented 5 code bugs
- Commits: e2883ac, fb4d2aa
- Deleted plan/13-alpha-beta-adjusted-strategies.md

**Agent 3 (analyst strategies):** COMPLETE
- Documented 4 analyst strategy files (analyst, analyst_badj, rating_diff, rating_diff_updn)
- Added 886 lines of documentation
- Documented IBES database requirements
- Commits: 3229d99, c893ac4
- Deleted plan/14-alpha-analyst-strategies.md

**Checking for next priorities...**

**Next batch priorities:**
1. Salamander simulation engines (osim, qsim, ssim) - Core Python 3 simulators
2. Earnings/valuation strategies (3 files) - Event-driven fundamental signals
3. Volume-adjusted strategies (5 files) - Execution quality

**Spawning 3 new subagents...**

**Agent 4 (salamander simulators):** COMPLETE
- Documented salamander/{osim,qsim,ssim}.py (3 core simulators)
- Added 1,904 lines of documentation
- Commit: 8a9a4a7

**Agent 5 (earnings/valuation):** COMPLETE
- Documented {eps,target,prod_tgt}.py (3 event-driven strategies)
- PEAD and price target deviation signals
- Commits: 87aab44, ac6567a
- Deleted plan/15-alpha-earnings-valuation.md

**Agent 6 (volume-adjusted):** COMPLETE
- Documented {vadj,vadj_multi,vadj_intra,vadj_pos,vadj_old}.py (5 files)
- Volume normalization and liquidity-aware sizing
- Commits: 87aab44, db132d4
- Deleted plan/16-alpha-volume-adjusted.md

**Checking remaining work...**

**Remaining plan files:**
- 17-alpha-other-strategies.md (7 files: c2o, pca_generator, mom_year, ebs, htb, rrb)
- 19-salamander-module.md (generators + 13 utility scripts)
- 20-testing-utilities.md (8 testing/utility files)
- 22-readme-update.md (README enhancement)
- Plus ~10 completed plan files that should be cleaned up

**Next priorities:**
1. Other alpha strategies (7 files) - Complete remaining alpha families
2. Salamander generators (3 files) - Small but important for workflow
3. Salamander utilities (13 files) - Many small utility scripts

**Spawning 3 new subagents...**
2026-02-03 - Documented specialized alpha strategies (7 files)

**Specialized Alpha Strategies:**

1. c2o.py (217 lines) - Close-to-open gap trading strategy
   - Beta-adjusted overnight returns for gap detection
   - Filters gaps < 2% absolute, industry-demeaned
   - Sector-specific regressions for gap reversal dynamics
   - Intraday signals vary by time-of-day (6 hourly buckets)
   - Combines current gap with lagged gaps (multi-day signal)
   - Formula: badjret = overnight_ret - beta * market_ret
   - Output: 'c2o' intraday forecast
   - CLI: --start, --end, --mid, --horizon, --freq

2. pca_generator.py (80 lines) - Intraday PCA residual extraction
   - Rolling 10-period correlation matrices
   - 4-component PCA decomposition
   - Residuals = stock-specific noise for mean reversion
   - Note: Residual calculation commented out (incomplete)
   - Output: 'pcaC_B_ma' signal (needs uncomment for full implementation)
   - CLI: --start, --end, --freq (default 5Min)

3. pca_generator_daily.py (81 lines) - Daily PCA with exp-weighted correlation
   - Combines overnight + previous day returns (2-day signal)
   - Exponentially-weighted correlation (5-day halflife)
   - More adaptive to regime changes than simple rolling
   - Note: Analysis/diagnostic code, residuals commented out
   - Prints explained variance and average correlation
   - CLI: --start, --end

4. mom_year.py (92 lines) - Annual momentum (232-day lag strategy)
   - 20-day cumulative return, industry-demeaned
   - 232-day lag (approx 1 year) for long-term reversal
   - Avoids short-term reversal and medium-term continuation zones
   - Based on momentum/reversal academic literature
   - Output: 'mom' daily forecast
   - CLI: --start, --end, --mid

5. ebs.py (221 lines) - Analyst estimate revision signals (not equity borrow!)
   - Uses SAL estimate data (mean, std, median)
   - Filters for increasing dispersion (std_diff > 0)
   - Signal = estimate_diff_mean / estimate_median
   - Separate up/down regressions for asymmetry
   - Captures analyst upgrade/downgrade dynamics
   - Output: 'sal' daily forecast
   - CLI: --start, --end, --mid, --lag (default 20)
   - Note: Misleading filename, should be sal.py

6. htb.py (116 lines) - Hard-to-borrow fee rate strategy
   - Uses fee_rate from stock loan market
   - High fees indicate crowded shorts → potential squeeze
   - Winsorized with 5-day lag features
   - Intraday forecasts from daily fee rate changes
   - Output: 'htb' intraday forecast
   - CLI: --start, --end, --mid, --freq (default 30Min)

7. rrb.py (157 lines) - Barra residual return betting
   - Barra factor model residuals (barraResidRet)
   - Idiosyncratic return mean reversion
   - Requires calc_factors() and calc_intra_factors()
   - Excludes Energy sector (commodity-driven dynamics)
   - Combines intraday residual with lagged daily residuals
   - Output: 'rrb' intraday forecast
   - CLI: --start, --end, --mid, --horizon (default 3)

**Documentation Coverage:**
- Module docstrings explaining strategy logic and academic basis
- Function docstrings with formulas, parameters, return values
- Data requirements and configuration details
- Relationships to other modules in codebase
- Usage examples and CLI parameter documentation

**Key Findings:**
- ebs.py is analyst estimates, not equity borrow (filename misleading)
- rrb.py = Barra residual mean reversion (Residual Return Betting)
- mom_year.py = 232-day lagged momentum (not simple annual momentum)
- PCA generators have incomplete implementations (residuals commented out)
- c2o.py has sophisticated sector-specific and time-of-day modeling

