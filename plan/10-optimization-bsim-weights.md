# Documentation Plan: bsim_weights.py

## Priority: CRITICAL
## Status: COMPLETED
## Estimated Scope: Medium (372 lines)

## Overview

`bsim_weights.py` handles dynamic alpha weight adjustment based on performance.
**Documentation completed with comprehensive module docstring.**

## Documentation Tasks

### 1. Module-Level Documentation (CRITICAL)
- [x] Add comprehensive module docstring
- [x] Document relationship to bsim.py
- [x] Document weight optimization methodology

### 2. Purpose Clarification
- [x] Investigate and document primary purpose
- [x] Document when to use this vs. standard bsim.py
- [x] Document input/output differences

### 3. Function Documentation
- [x] Document all public functions
- [x] Document weight calculation logic
- [x] Document optimization constraints

### 4. Usage Documentation
- [x] Add CLI usage examples
- [x] Document typical use cases
- [x] Document parameter combinations

## Investigation Required
- [x] Determine if this is a variant of bsim or separate tool
- [x] Identify differences from opt.py weighting
- [x] Clarify production usage patterns

## Key Findings

**Purpose**: Dynamic weight optimization backtester that adjusts alpha strategy
weights based on rolling 5-day performance. Unlike standard bsim.py which uses
fixed weights, this creates a momentum-based meta-strategy.

**Weight Adjustment Logic**:
- Reads return history from <alpha_dir>/rets.txt
- If 5-day rolling return > 0: weight *= 1.2 (capped at 0.9)
- If 5-day rolling return <= 0: weight *= 0.8 (floored at 0.1)
- Exception: HTB strategy always uses weight 0.5

**When to Use**:
- Use bsim.py for fixed-weight backtests
- Use bsim_weights.py for adaptive weight allocation
- Useful for meta-strategy optimization and live adaptation

**Output**:
- Saves detailed optimization results to ./opt/ directory
- One CSV per timestamp with positions, utilities, costs
- Does not compute aggregate P&L statistics

## Dependencies
- bsim.py (likely)
- opt.py
- loaddata.py

## Notes
- HIGH PRIORITY: No existing documentation
- May be used in production workflows
