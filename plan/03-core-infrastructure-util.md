# Documentation Plan: util.py

## Priority: MEDIUM
## Status: Pending
## Estimated Scope: Medium (276 lines)

## Overview

`util.py` provides helper functions and utilities:
- Data merging operations
- File I/O helpers
- Common data transformations

## Documentation Tasks

### 1. Module-Level Documentation
- [ ] Add module docstring with utility categories
- [ ] Document common usage patterns
- [ ] Add quick reference for available utilities

### 2. Function Documentation
- [ ] Document all merge functions with examples
- [ ] Document file operation helpers
- [ ] Document data transformation utilities
- [ ] Document date/time helpers

### 3. Examples
- [ ] Add doctest examples for key functions
- [ ] Document edge cases and error handling

## Dependencies
- pandas, numpy
- os, datetime

## Notes
- Widely imported across codebase
- Changes affect many downstream modules
