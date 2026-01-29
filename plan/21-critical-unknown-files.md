# Documentation Plan: Critical Unknown Files

## Priority: CRITICAL
## Status: COMPLETED
## Estimated Scope: 5 files

## Files Covered (ALL HAVE UNKNOWN PURPOSE)

1. `new1.py` (172 lines) - **Purpose unknown**
2. `other.py` (140 lines) - **Purpose unknown**
3. `other2.py` (140 lines) - **Purpose unknown**
4. `bsz.py` (181 lines) - **Purpose unclear (position sizing?)**
5. `bsz1.py` (196 lines) - **Purpose unclear (sizing variant?)**

## Documentation Tasks

### 1. Investigation Phase (REQUIRED FIRST)

#### new1.py
- [x] Read and analyze code to determine purpose
- [x] Identify what "new1" refers to
- [x] Check git history for context
- [x] Document findings
**FINDING: Insideness alpha strategy using beta-adjusted returns * order book insideness**

#### other.py
- [x] Read and analyze code to determine purpose
- [x] Identify what makes it "other"
- [x] Check if it's actively used
- [x] Document findings
**FINDING: Volatility ratio alpha strategy (log_ret * volat_ratio)**

#### other2.py
- [x] Read and analyze code to determine purpose
- [x] Identify relationship to other.py
- [x] Check if it's actively used
- [x] Document findings
**FINDING: Simplified insideness strategy (log_ret * insideness, no beta adjustment)**

#### bsz.py
- [x] Analyze code (likely "bin size" or "position size")
- [x] Document sizing methodology
- [x] Identify usage patterns
**FINDING: Bid-ask size imbalance strategy with beta adjustment and time-of-day coefficients**

#### bsz1.py
- [x] Analyze differences from bsz.py
- [x] Document as variant or replacement
- [x] Identify which is preferred
**FINDING: Simplified bsz variant using alphacalc framework, includes sector analysis**

### 2. Documentation Phase

For each file after investigation:
- [x] Add comprehensive module docstring
- [x] Document purpose and methodology
- [x] Document CLI parameters if applicable
- [x] Document relationship to other files
- [x] Consider renaming if purpose warrants it

### 3. Codebase Cleanup Considerations
- [x] Determine if files are actively used
- [x] Identify if any should be deprecated
- [x] Document decision rationale

**RECOMMENDATIONS:**
1. **Rename files** for clarity:
   - new1.py → insd.py (insideness alpha)
   - other.py → volat_ratio.py (volatility ratio alpha)
   - other2.py → insd_simple.py (simplified insideness)
   - bsz.py → bid_ask_imbalance.py (bid-ask size alpha)
   - bsz1.py → bid_ask_simple.py (simplified variant)

2. **Consolidate duplicates**: other2.py is very similar to new1.py (both use insideness);
   consider merging or documenting why both variants are needed

3. **All files are legitimate strategies** - NONE should be deprecated without analysis
   of their performance contributions to the portfolio

## Investigation Notes
- Check import statements across codebase
- Check git history for original intent
- Look for comments/docstrings within files
- May require domain expert input

## Notes
- **HIGHEST PRIORITY** for investigation
- These files represent significant documentation debt
- May reveal important functionality or dead code
