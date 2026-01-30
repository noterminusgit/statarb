# Documentation Plan: opt.py

## Priority: HIGH
## Status: COMPLETE
## Estimated Scope: Large (434 lines)

## Overview

`opt.py` implements portfolio optimization:
- NLP solver integration (OpenOpt)
- Factor risk constraints
- Transaction cost modeling
- Position limit constraints

## Documentation Tasks

### 1. Module-Level Documentation
- [x] Enhance existing docstring with optimization methodology
- [x] Document objective function formulation
- [x] Add mathematical formulation of constraints

### 2. Parameter Documentation
- [x] `kappa` - Risk aversion coefficient (2e-8 to 4.3e-5)
- [x] `max_sumnot` - Maximum total notional ($50M default)
- [x] `slip_nu` - Market impact coefficient (0.14-0.18)
- [x] All constraint parameters

### 3. Function Documentation
- [x] `optimize()` - Main optimization entry point
- [x] `objective()` - Objective function (utility to maximize)
- [x] `objective_detail()` - Objective with component breakdown
- [x] `objective_grad()` - Analytical gradient
- [x] `slippageFuncAdv()` - Market impact slippage cost
- [x] `slippageFunc_grad()` - Slippage gradient
- [x] `costsFunc()` - Execution and borrow costs
- [x] `costsFunc_grad()` - Costs gradient
- [x] `constrain_by_capital()` - Total notional constraint
- [x] `constrain_by_capital_grad()` - Capital constraint gradient
- [x] `constrain_by_trdnot()` - Turnover constraint (unused)
- [x] `setupProblem()` - NLP problem configuration
- [x] `init()` - Global array initialization
- [x] `getUntradeable()` - Tradeable/untradeable partitioning
- [x] `printinfo()` - Optimization summary output
- [x] `__printpointinfo()` - Detailed utility breakdown
- [x] `Terminator` class - Convergence callback

### 4. Mathematical Documentation
- [x] Document objective function mathematics
- [x] Document factor risk model integration
- [x] Document transaction cost model
- [x] Document constraint formulations

### 5. Solver Documentation
- [x] Document OpenOpt/FuncDesigner usage
- [x] Document solver parameters and tuning
- [x] Document convergence criteria

## Dependencies
- OpenOpt, FuncDesigner
- numpy, scipy
- Barra risk model data

## Notes
- Critical for position sizing
- Uses legacy OpenOpt library
- Mathematical accuracy is essential

## Completion Notes
- Completed 2026-01-30
- Module docstring already excellent (comprehensive objective/constraints/parameters)
- Added comprehensive docstrings to all 14 functions and 1 class
- Mathematical formulas documented for all cost/risk components
- Gradient derivations explicit for optimization transparency
- Global variable workflow and init→optimize flow documented
- Per-security marginal utility calculation explained
- Untradeable handling and hard limit buffer documented
