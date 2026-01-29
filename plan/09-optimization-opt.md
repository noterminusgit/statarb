# Documentation Plan: opt.py

## Priority: HIGH
## Status: Pending
## Estimated Scope: Large (434 lines)

## Overview

`opt.py` implements portfolio optimization:
- NLP solver integration (OpenOpt)
- Factor risk constraints
- Transaction cost modeling
- Position limit constraints

## Documentation Tasks

### 1. Module-Level Documentation
- [ ] Enhance existing docstring with optimization methodology
- [ ] Document objective function formulation
- [ ] Add mathematical formulation of constraints

### 2. Parameter Documentation
- [ ] `kappa` - Risk aversion coefficient (2e-8 to 4.3e-5)
- [ ] `max_sumnot` - Maximum total notional ($50M default)
- [ ] `slip_nu` - Market impact coefficient (0.14-0.18)
- [ ] All constraint parameters

### 3. Function Documentation
- [ ] `optimize_portfolio()` - Main optimization entry point
- [ ] `build_objective()` - Objective function construction
- [ ] `build_constraints()` - Constraint specification
- [ ] `solve_nlp()` - NLP solver wrapper
- [ ] Document all public functions

### 4. Mathematical Documentation
- [ ] Document objective function mathematics
- [ ] Document factor risk model integration
- [ ] Document transaction cost model
- [ ] Document constraint formulations

### 5. Solver Documentation
- [ ] Document OpenOpt/FuncDesigner usage
- [ ] Document solver parameters and tuning
- [ ] Document convergence criteria

## Dependencies
- OpenOpt, FuncDesigner
- numpy, scipy
- Barra risk model data

## Notes
- Critical for position sizing
- Uses legacy OpenOpt library
- Mathematical accuracy is essential
