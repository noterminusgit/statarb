# Documentation Plan: util.py

## Priority: MEDIUM
## Status: Complete
## Estimated Scope: Medium (276 lines)

## Overview

`util.py` provides helper functions and utilities:
- Data merging operations
- File I/O helpers
- Common data transformations

## Documentation Tasks

### 1. Module-Level Documentation
- [x] Add module docstring with utility categories
- [x] Document common usage patterns
- [x] Add quick reference for available utilities

### 2. Function Documentation
- [x] Document all merge functions with examples
- [x] Document file operation helpers
- [x] Document data transformation utilities
- [x] Document date/time helpers

### 3. Examples
- [x] Add doctest examples for key functions
- [x] Document edge cases and error handling

## Dependencies
- pandas, numpy
- os, datetime

## Completion Summary

All functions in util.py have been documented with comprehensive docstrings including:

**Data Merging Functions:**
- `merge_barra_data()`: Lag Barra factors by 1 day to avoid look-ahead bias
- `merge_intra_eod()`: Extract 4 PM bars and merge with daily data
- `merge_intra_data()`: Expand daily data to intraday frequency
- `merge_daily_calcs()`: Add calculated daily features to main pipeline
- `merge_intra_calcs()`: Add calculated intraday features to main pipeline
- `remove_dup_cols()`: Clean up duplicate columns after merges

**Universe Filtering:**
- `filter_expandable()`: Filter to high-liquidity expandable universe
- `filter_pca()`: Filter to large-cap stocks for PCA analysis

**File I/O Operations:**
- `dump_hd5()`: Export to compressed HDF5 with date range filename
- `dump_all()`: Export all intraday signals by timestamp
- `dump_alpha()`: Export single alpha strategy by timestamp
- `dump_prod_alpha()`: Export latest alpha for production trading
- `dump_daily_alpha()`: Replicate daily alpha across intraday timestamps
- `load_all_results()`: Load alpha CSV files from directory for date range
- `load_merged_results()`: Load and merge alpha from multiple directories

**Utilities:**
- `mkdir_p()`: Create directory with parents (like mkdir -p)
- `email()`: Send notification via SMTP
- `df_dates()`: Extract date range string from DataFrame index
- `get_overlapping_cols()`: Get columns unique to first DataFrame

Each function documented with:
- Clear description of purpose and behavior
- Args/Returns with types and descriptions
- Usage examples
- Implementation notes
- Edge case handling

## Notes
- Widely imported across codebase
- Changes affect many downstream modules
- Note: `dump_alpha()` function appears duplicated in source code (lines 378-427)
