## Documentation Plans

Documentation improvement tasks are tracked in the `plan/` directory. See `plan/00-documentation-overview.md` for the master task list.

### Plan Files by Category

**Core Infrastructure:**
- `plan/01-core-infrastructure-loaddata.md` - Data loading documentation
- `plan/02-core-infrastructure-calc.md` - Calculation module documentation
- `plan/03-core-infrastructure-util.md` - Utility functions documentation
- `plan/04-core-infrastructure-regress.md` - Regression module documentation

**Simulation Engines:**
- `plan/05-simulation-bsim.md` - Daily simulator documentation
- `plan/06-simulation-osim.md` - Order-level simulator documentation
- `plan/07-simulation-qsim.md` - Intraday simulator documentation
- `plan/08-simulation-ssim.md` - Lifecycle simulator documentation

**Optimization:**
- `plan/09-optimization-opt.md` - Portfolio optimization documentation
- `plan/10-optimization-bsim-weights.md` - Weight optimization (CRITICAL)
- `plan/11-optimization-pca.md` - PCA module documentation

**Alpha Strategies:**
- `plan/12-alpha-high-low-strategies.md` - High-low mean reversion (6 files)
- `plan/13-alpha-beta-adjusted-strategies.md` - Beta-adjusted signals (9 files)
- `plan/14-alpha-analyst-strategies.md` - Analyst ratings (4 files)
- `plan/15-alpha-earnings-valuation.md` - Earnings strategies (3 files)
- `plan/16-alpha-volume-adjusted.md` - Volume-adjusted signals (5 files)
- `plan/17-alpha-other-strategies.md` - Specialized strategies (7 files)

**Production & Deployment:**
- `plan/18-production-modules.md` - Production pipeline (CRITICAL)

**Salamander Module:**
- `plan/19-salamander-module.md` - Python 3 standalone module (25 files)

**Testing & Utilities:**
- `plan/20-testing-utilities.md` - Test and utility scripts
- `plan/21-critical-unknown-files.md` - Files needing investigation (CRITICAL)

**Final Documentation:**
- `plan/22-readme-update.md` - README enhancement tasks

### Priority Levels
- **CRITICAL**: Production modules, unknown files, bsim_weights
- **HIGH**: Core infrastructure, main simulators, optimization
- **MEDIUM**: Alpha strategies, utilities, salamander
