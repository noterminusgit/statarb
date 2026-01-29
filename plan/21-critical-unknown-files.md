# Documentation Plan: Critical Unknown Files

## Priority: CRITICAL
## Status: Pending
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
- [ ] Read and analyze code to determine purpose
- [ ] Identify what "new1" refers to
- [ ] Check git history for context
- [ ] Document findings

#### other.py
- [ ] Read and analyze code to determine purpose
- [ ] Identify what makes it "other"
- [ ] Check if it's actively used
- [ ] Document findings

#### other2.py
- [ ] Read and analyze code to determine purpose
- [ ] Identify relationship to other.py
- [ ] Check if it's actively used
- [ ] Document findings

#### bsz.py
- [ ] Analyze code (likely "bin size" or "position size")
- [ ] Document sizing methodology
- [ ] Identify usage patterns

#### bsz1.py
- [ ] Analyze differences from bsz.py
- [ ] Document as variant or replacement
- [ ] Identify which is preferred

### 2. Documentation Phase

For each file after investigation:
- [ ] Add comprehensive module docstring
- [ ] Document purpose and methodology
- [ ] Document CLI parameters if applicable
- [ ] Document relationship to other files
- [ ] Consider renaming if purpose warrants it

### 3. Codebase Cleanup Considerations
- [ ] Determine if files are actively used
- [ ] Identify if any should be deprecated
- [ ] Document decision rationale

## Investigation Notes
- Check import statements across codebase
- Check git history for original intent
- Look for comments/docstrings within files
- May require domain expert input

## Notes
- **HIGHEST PRIORITY** for investigation
- These files represent significant documentation debt
- May reveal important functionality or dead code
