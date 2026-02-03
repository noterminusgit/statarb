# Documentation Plan: Salamander Module (Python 3)

## Priority: HIGH
## Status: Pending
## Estimated Scope: 25 files

## Overview

The `salamander/` directory contains a standalone Python 3 implementation
of the core trading system. It provides:
- Simplified, self-contained backtesting
- Python 3 compatibility
- Independent data handling

## Files Requiring Documentation

### Core Simulation Files
1. `salamander/bsim.py` (517 lines) - Standalone backtest engine
2. `salamander/osim.py` (340 lines) - Standalone order sim
3. `salamander/qsim.py` (465 lines) - Standalone intraday sim
4. `salamander/ssim.py` (517 lines) - Standalone system sim
5. `salamander/simulation.py` (503 lines) - Base simulation class

### Data & Calculation Files
6. `salamander/loaddata.py` (117 lines) - Data loading
7. `salamander/loaddata_sql.py` (284 lines) - SQL data loader
8. `salamander/calc.py` (407 lines) - Calculation module ✓
9. `salamander/opt.py` (480 lines) - Optimization module ✓
10. `salamander/regress.py` (250 lines) - Regression module ✓
11. `salamander/util.py` (260 lines) - Utility functions

### Generator Scripts
12. `salamander/gen_dir.py` (17 lines) - Directory generator
13. `salamander/gen_hl.py` (23 lines) - HL signal generator
14. `salamander/gen_alpha.py` (32 lines) - Alpha file generator

### Strategy Files
15. `salamander/hl.py` (124 lines) - HL strategy
16. `salamander/hl_csv.py` (153 lines) - HL CSV processing

### Utility Scripts
17. `salamander/change_hl.py` (12 lines) - HL modification
18. `salamander/check_hl.py` (25 lines) - HL validation
19. `salamander/check_all.py` (19 lines) - Full validation
20. `salamander/change_raw.py` (118 lines) - Raw data transformation
21. `salamander/mktcalendar.py` (20 lines) - Market calendar
22. `salamander/get_borrow.py` (18 lines) - Borrow data fetch
23. `salamander/show_borrow.py` (9 lines) - Borrow display
24. `salamander/show_raw.py` (25 lines) - Raw data display

## Documentation Tasks

### 1. Module-Level Documentation
- [x] Add README.md for salamander directory
- [x] Document Python 3 requirements
- [x] Document standalone usage instructions

### 2. Core Simulation Documentation
- [x] Document simulation.py (portfolio rebalancing Monte Carlo tool)
- [x] Document bsim.py differences from main version
- [x] Document osim.py differences
- [x] Document qsim.py differences
- [x] Document ssim.py differences

### 3. Data Module Documentation
- [x] Document loaddata.py vs main version
- [x] Document loaddata_sql.py SQL integration
- [x] Document data format requirements

### 4. Generator Documentation
- [ ] Document gen_dir.py directory structure
- [ ] Document gen_hl.py signal generation
- [ ] Document gen_alpha.py alpha file format

### 5. Utility Documentation
- [ ] Document each utility script purpose
- [ ] Document CLI usage for each

### 6. Relationship Documentation
- [ ] Document differences from main codebase
- [ ] Document when to use salamander vs main
- [ ] Document data compatibility

## Notes
- Python 3 compatible (unlike main codebase)
- Useful for rapid prototyping
- instructions.txt exists but is minimal
