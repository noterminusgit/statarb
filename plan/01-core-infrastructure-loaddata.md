# Documentation Plan: loaddata.py

## Priority: HIGH
## Status: Pending
## Estimated Scope: Large (959 lines)

## Overview

`loaddata.py` is the primary data loading module that handles:
- CSV and SQL data ingestion
- Universe filtering and stock selection
- HDF5 caching for performance
- Price, volume, and Barra factor data loading

## Documentation Tasks

### 1. Module-Level Documentation
- [ ] Add comprehensive module docstring explaining purpose and data flow
- [ ] Document configuration requirements (UNIV_BASE_DIR, PRICE_BASE_DIR, etc.)
- [ ] Add usage examples for common loading scenarios

### 2. Function Documentation (20+ functions)
- [ ] `load_univ()` - Universe loading with date filtering
- [ ] `load_prices()` - Price data loading with adjustments
- [ ] `load_volume()` - Volume data loading
- [ ] `load_barra()` - Barra risk factor loading
- [ ] `filter_universe()` - Stock filtering by criteria
- [ ] `cache_to_hdf5()` - HDF5 caching mechanism
- [ ] All remaining public functions

### 3. Configuration Documentation
- [ ] Document universe parameters (price range $2-$500, ADV thresholds)
- [ ] Document data path configuration requirements
- [ ] Document HDF5 cache structure and invalidation

### 4. Type Annotations (Optional Enhancement)
- [ ] Add type hints to function signatures
- [ ] Document DataFrame column specifications

## Dependencies
- pandas, numpy
- HDF5 (pytables or h5py)
- SQL connector (if applicable)

## Notes
- Python 2.7 compatibility required
- Critical module - affects all downstream processing
