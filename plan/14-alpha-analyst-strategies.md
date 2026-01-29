# Documentation Plan: Analyst Signal Strategies

## Priority: MEDIUM
## Status: Pending
## Estimated Scope: 4 files

## Files Covered
1. `analyst.py` (186 lines) - Analyst ratings
2. `analyst_badj.py` - Beta-adjusted analyst signals
3. `rating_diff.py` - Rating change strategy
4. `rating_diff_updn.py` (192 lines) - Up/down rating analysis

## Overview

Analyst strategies implement signals based on:
- Analyst rating levels
- Rating changes and upgrades/downgrades
- Consensus estimate revisions

## Documentation Tasks

### 1. Strategy Family Overview
- [ ] Create overview document explaining analyst data usage
- [ ] Document data source requirements
- [ ] Document signal generation methodology

### 2. Per-File Documentation (for each file)
- [ ] Add module docstring with strategy description
- [ ] Document rating scale interpretation
- [ ] Document change detection methodology
- [ ] Document parameters and tuning options

### 3. Data Requirements Documentation
- [ ] Document analyst data format
- [ ] Document rating scale conventions
- [ ] Document timestamp handling

### 4. Signal Interpretation
- [ ] Document signal for upgrade vs downgrade
- [ ] Document beta adjustment rationale
- [ ] Document decay parameters

## Dependencies
- Analyst data from ESTIMATES_BASE_DIR
- Rating history data

## Notes
- Requires external analyst data
- Important for fundamental-driven signals
