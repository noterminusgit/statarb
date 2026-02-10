# BUGS.md — Known Bugs

> Discovered during SPECS.md verification (February 2026).
> Organized by severity, then alphabetical filename.

---

## Runtime Errors (NameError)

### `bsim.py:490` — undefined `testid`
```python
print(date_group.xs(testid, level=1)[['forecast', 'min_notional', 'max_notional', 'position_last']])
```
`testid` is never defined. Leftover debug code that will crash if reached.

### `new1.py:160` — undefined `middate`
```python
full_df = insd_fits(daily_results_df, intra_results_df, horizon, "", middate)
```
`calc_insd_forecast()` does not accept `middate` as a parameter, but passes it to `insd_fits()`. Variable is only defined in `__main__` block (line 181) and is not in scope here.

### `qhl_multi.py:784-785` — undefined `outsample_df`
```python
dump_alpha(outsample_df, 'qhl_m')
dump_all(outsample_df)
```
`outsample_df` is never assigned. Should be `full_df`, which is the actual return value from `qhl_fits()`.

### `slip.py:81` — typo `merged_Df`
```python
assert merged_Df['sid_ord'] == merged_df['sid_exec']
```
Capital `D` in `merged_Df` — variable was created as `merged_df` (lowercase) on line 75.

---

## Logic Errors

### `prod_sal.py:239` — copy-paste from prod_tgt.py
```python
coef_list.append( { 'name': 'tgt0_ma_coef', 'group': "up", 'coef': coef0 } )
```
Uses `tgt0_ma_coef` (target strategy prefix) instead of `sal0_ma_coef`. Coefficient names won't match when `sal_alpha()` tries to look them up. Same issue on nearby intercept lines.

### `vadj_intra.py:252,257` — swapped sector labels
```python
result1_df = vadj_fits(sector_df, sector_intra_results_df, horizon, "ex", middate)  # Energy sector
...
result2_df = vadj_fits(sector_df, sector_intra_results_df, horizon, "in", middate)  # non-Energy sector
```
Energy sector gets label `"ex"` (ex-sector) and non-Energy gets `"in"` (in-sector). Labels are backwards — Energy should be `"in"`, non-Energy should be `"ex"`.

---

## Deprecated pandas API

### `dumpall.py:113-114` — `.sort()` removed in pandas 0.20+
```python
dump_hd5(intra_df.sort(), "all")
dump_hd5(factor_df.sort(), "all.factors")
```
Should use `.sort_index()`.

### `salamander/regress.py:578` — `pd.rolling_sum()` removed in pandas 1.0+
```python
timeslice_df[retname] = shift_df['log_ret'].groupby(level='sid').apply(lambda x: pd.rolling_sum(x, horizon))
```
Should use `x.rolling(horizon).sum()`.

---

## Debug Code (non-critical)

### `bigsim_test.py:541` — undefined `testid` (commented out)
```python
#pnl_df.xs(testid, level=1).to_csv("debug.csv")
```
Commented out, but `testid` is never defined. Would fail if uncommented.
