#!/usr/bin/env python3
"""
Python 3 Migration Validation Script

Compares backtest outputs between Python 2 baseline and Python 3 migrated versions
to ensure numerical equivalence within acceptable tolerances.

Usage:
    python3 scripts/validate_migration.py \\
        --py2-positions=baseline/positions_20130101_20130630.csv \\
        --py3-positions=migrated/positions_20130101_20130630.csv \\
        --py2-pnl=baseline/pnl.csv \\
        --py3-pnl=migrated/pnl.csv

Tolerances:
    - Position differences: < 1% of position size
    - PnL differences: < 0.1% of cumulative PnL
    - Sharpe ratio differences: < 0.05

Exit Codes:
    0 - All validations passed within tolerance
    1 - Validation failures detected
    2 - Error loading files or invalid arguments
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# Validation tolerances
POSITION_TOLERANCE_PCT = 1.0  # 1% of position size
PNL_TOLERANCE_PCT = 0.1  # 0.1% of cumulative PnL
SHARPE_TOLERANCE = 0.05  # Absolute difference in Sharpe ratio


class ValidationResult:
    """Container for validation results"""

    def __init__(self):
        self.passed = True
        self.messages = []
        self.warnings = []
        self.errors = []

    def add_pass(self, message):
        self.messages.append(f"✓ PASS: {message}")

    def add_warning(self, message):
        self.warnings.append(f"⚠ WARNING: {message}")

    def add_error(self, message):
        self.errors.append(f"✗ FAIL: {message}")
        self.passed = False

    def print_summary(self):
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        if self.messages:
            print("\nPassed Checks:")
            for msg in self.messages:
                print(f"  {msg}")

        if self.warnings:
            print("\nWarnings:")
            for msg in self.warnings:
                print(f"  {msg}")

        if self.errors:
            print("\nFailed Checks:")
            for msg in self.errors:
                print(f"  {msg}")

        print("\n" + "=" * 80)
        if self.passed:
            print("RESULT: ✓ ALL VALIDATIONS PASSED")
        else:
            print("RESULT: ✗ VALIDATION FAILED")
        print("=" * 80 + "\n")

        return self.passed


def load_positions(filepath):
    """Load positions file (CSV format expected)"""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        raise ValueError(f"Error loading positions file {filepath}: {e}")


def load_pnl(filepath):
    """Load PnL file (CSV format expected)"""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        raise ValueError(f"Error loading PnL file {filepath}: {e}")


def validate_positions(py2_positions, py3_positions, result):
    """
    Compare positions between Python 2 and Python 3 versions.
    Tolerance: < 1% of position size
    """
    print("\n" + "-" * 80)
    print("VALIDATING POSITIONS")
    print("-" * 80)

    # Ensure same dimensions
    if py2_positions.shape != py3_positions.shape:
        result.add_error(
            f"Position dimensions mismatch: "
            f"Py2 {py2_positions.shape} vs Py3 {py3_positions.shape}"
        )
        return

    # Align indices and columns
    common_dates = py2_positions.index.intersection(py3_positions.index)
    common_stocks = py2_positions.columns.intersection(py3_positions.columns)

    if len(common_dates) == 0 or len(common_stocks) == 0:
        result.add_error(
            "No common dates or stocks between Python 2 and Python 3 positions"
        )
        return

    py2_aligned = py2_positions.loc[common_dates, common_stocks]
    py3_aligned = py3_positions.loc[common_dates, common_stocks]

    # Calculate differences
    position_diff = py3_aligned - py2_aligned
    position_diff_pct = (position_diff / (py2_aligned.abs() + 1e-10)) * 100

    # Find positions exceeding tolerance
    exceeds_tolerance = position_diff_pct.abs() > POSITION_TOLERANCE_PCT

    # Statistics
    max_diff_pct = position_diff_pct.abs().max().max()
    mean_diff_pct = position_diff_pct.abs().mean().mean()
    num_exceeds = exceeds_tolerance.sum().sum()
    total_positions = (py2_aligned != 0).sum().sum()

    print(f"  Dates compared: {len(common_dates)}")
    print(f"  Stocks compared: {len(common_stocks)}")
    print(f"  Total non-zero positions: {total_positions}")
    print(f"  Max position difference: {max_diff_pct:.4f}%")
    print(f"  Mean position difference: {mean_diff_pct:.4f}%")
    print(f"  Positions exceeding {POSITION_TOLERANCE_PCT}% tolerance: {num_exceeds}")

    if num_exceeds == 0:
        result.add_pass(
            f"All positions within {POSITION_TOLERANCE_PCT}% tolerance "
            f"(max diff: {max_diff_pct:.4f}%)"
        )
    else:
        pct_exceeds = (num_exceeds / total_positions) * 100
        result.add_error(
            f"{num_exceeds} positions ({pct_exceeds:.2f}%) exceed "
            f"{POSITION_TOLERANCE_PCT}% tolerance"
        )

        # Show worst offenders
        print("\n  Top 10 largest position differences:")
        flat_diff = position_diff_pct.abs().stack().sort_values(ascending=False)
        for i, (idx, val) in enumerate(flat_diff.head(10).items()):
            date, stock = idx
            py2_pos = py2_aligned.loc[date, stock]
            py3_pos = py3_aligned.loc[date, stock]
            print(
                f"    {i+1}. {date.strftime('%Y-%m-%d')} {stock}: "
                f"Py2={py2_pos:.2f}, Py3={py3_pos:.2f}, diff={val:.4f}%"
            )


def validate_pnl(py2_pnl, py3_pnl, result):
    """
    Compare PnL between Python 2 and Python 3 versions.
    Tolerance: < 0.1% of cumulative PnL
    """
    print("\n" + "-" * 80)
    print("VALIDATING PNL")
    print("-" * 80)

    # Ensure same dimensions
    if len(py2_pnl) != len(py3_pnl):
        result.add_warning(
            f"PnL length mismatch: Py2 {len(py2_pnl)} vs Py3 {len(py3_pnl)}"
        )

    # Align indices
    common_dates = py2_pnl.index.intersection(py3_pnl.index)
    if len(common_dates) == 0:
        result.add_error("No common dates between Python 2 and Python 3 PnL")
        return

    py2_aligned = py2_pnl.loc[common_dates]
    py3_aligned = py3_pnl.loc[common_dates]

    # Assume PnL has 'pnl' or 'daily_pnl' column
    pnl_col = None
    for col in ['pnl', 'daily_pnl', 'PnL', 'Daily_PnL']:
        if col in py2_aligned.columns:
            pnl_col = col
            break

    if pnl_col is None:
        # If single column, use it
        if len(py2_aligned.columns) == 1:
            pnl_col = py2_aligned.columns[0]
        else:
            result.add_error(
                f"Could not identify PnL column. Available columns: {py2_aligned.columns.tolist()}"
            )
            return

    py2_pnl_series = py2_aligned[pnl_col]
    py3_pnl_series = py3_aligned[pnl_col]

    # Calculate cumulative PnL
    py2_cum_pnl = py2_pnl_series.cumsum()
    py3_cum_pnl = py3_pnl_series.cumsum()

    # Calculate differences
    pnl_diff = py3_pnl_series - py2_pnl_series
    cum_pnl_diff = py3_cum_pnl - py2_cum_pnl

    # Statistics
    total_py2_cum_pnl = py2_cum_pnl.iloc[-1]
    total_py3_cum_pnl = py3_cum_pnl.iloc[-1]
    max_cum_diff = cum_pnl_diff.abs().max()
    max_cum_diff_pct = (max_cum_diff / abs(total_py2_cum_pnl)) * 100 if total_py2_cum_pnl != 0 else 0

    print(f"  Dates compared: {len(common_dates)}")
    print(f"  Python 2 cumulative PnL: ${total_py2_cum_pnl:,.2f}")
    print(f"  Python 3 cumulative PnL: ${total_py3_cum_pnl:,.2f}")
    print(f"  Cumulative PnL difference: ${total_py3_cum_pnl - total_py2_cum_pnl:,.2f}")
    print(f"  Max cumulative PnL diff: ${max_cum_diff:,.2f} ({max_cum_diff_pct:.4f}%)")

    if max_cum_diff_pct < PNL_TOLERANCE_PCT:
        result.add_pass(
            f"PnL within {PNL_TOLERANCE_PCT}% tolerance "
            f"(max diff: {max_cum_diff_pct:.4f}%)"
        )
    else:
        result.add_error(
            f"PnL difference {max_cum_diff_pct:.4f}% exceeds "
            f"{PNL_TOLERANCE_PCT}% tolerance"
        )

    # Calculate Sharpe ratios
    py2_sharpe = calculate_sharpe(py2_pnl_series)
    py3_sharpe = calculate_sharpe(py3_pnl_series)
    sharpe_diff = abs(py3_sharpe - py2_sharpe)

    print(f"  Python 2 Sharpe ratio: {py2_sharpe:.4f}")
    print(f"  Python 3 Sharpe ratio: {py3_sharpe:.4f}")
    print(f"  Sharpe ratio difference: {sharpe_diff:.4f}")

    if sharpe_diff < SHARPE_TOLERANCE:
        result.add_pass(
            f"Sharpe ratio within {SHARPE_TOLERANCE} tolerance "
            f"(diff: {sharpe_diff:.4f})"
        )
    else:
        result.add_error(
            f"Sharpe ratio difference {sharpe_diff:.4f} exceeds "
            f"{SHARPE_TOLERANCE} tolerance"
        )


def calculate_sharpe(pnl_series, annualization_factor=252):
    """Calculate annualized Sharpe ratio from daily PnL series"""
    if len(pnl_series) == 0:
        return 0.0

    mean_pnl = pnl_series.mean()
    std_pnl = pnl_series.std()

    if std_pnl == 0 or np.isnan(std_pnl):
        return 0.0

    sharpe = (mean_pnl / std_pnl) * np.sqrt(annualization_factor)
    return sharpe


def main():
    parser = argparse.ArgumentParser(
        description="Validate Python 2 vs Python 3 migration backtest outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--py2-positions',
        required=True,
        help='Path to Python 2 baseline positions CSV file'
    )
    parser.add_argument(
        '--py3-positions',
        required=True,
        help='Path to Python 3 migrated positions CSV file'
    )
    parser.add_argument(
        '--py2-pnl',
        required=False,
        help='Path to Python 2 baseline PnL CSV file'
    )
    parser.add_argument(
        '--py3-pnl',
        required=False,
        help='Path to Python 3 migrated PnL CSV file'
    )
    parser.add_argument(
        '--position-tolerance',
        type=float,
        default=POSITION_TOLERANCE_PCT,
        help=f'Position difference tolerance percentage (default: {POSITION_TOLERANCE_PCT}%%)'
    )
    parser.add_argument(
        '--pnl-tolerance',
        type=float,
        default=PNL_TOLERANCE_PCT,
        help=f'PnL difference tolerance percentage (default: {PNL_TOLERANCE_PCT}%%)'
    )
    parser.add_argument(
        '--sharpe-tolerance',
        type=float,
        default=SHARPE_TOLERANCE,
        help=f'Sharpe ratio difference tolerance (default: {SHARPE_TOLERANCE})'
    )

    args = parser.parse_args()

    # Update tolerances if provided
    global POSITION_TOLERANCE_PCT, PNL_TOLERANCE_PCT, SHARPE_TOLERANCE
    POSITION_TOLERANCE_PCT = args.position_tolerance
    PNL_TOLERANCE_PCT = args.pnl_tolerance
    SHARPE_TOLERANCE = args.sharpe_tolerance

    result = ValidationResult()

    print("\n" + "=" * 80)
    print("PYTHON 3 MIGRATION VALIDATION")
    print("=" * 80)
    print(f"\nTolerances:")
    print(f"  Position differences: < {POSITION_TOLERANCE_PCT}%")
    print(f"  PnL differences: < {PNL_TOLERANCE_PCT}%")
    print(f"  Sharpe ratio differences: < {SHARPE_TOLERANCE}")

    try:
        # Load and validate positions
        print("\nLoading positions...")
        py2_positions = load_positions(args.py2_positions)
        py3_positions = load_positions(args.py3_positions)
        print(f"  Python 2: {py2_positions.shape[0]} dates, {py2_positions.shape[1]} stocks")
        print(f"  Python 3: {py3_positions.shape[0]} dates, {py3_positions.shape[1]} stocks")

        validate_positions(py2_positions, py3_positions, result)

        # Load and validate PnL if provided
        if args.py2_pnl and args.py3_pnl:
            print("\nLoading PnL...")
            py2_pnl = load_pnl(args.py2_pnl)
            py3_pnl = load_pnl(args.py3_pnl)
            print(f"  Python 2: {len(py2_pnl)} dates")
            print(f"  Python 3: {len(py3_pnl)} dates")

            validate_pnl(py2_pnl, py3_pnl, result)
        else:
            result.add_warning("PnL files not provided, skipping PnL validation")

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        return 2

    # Print summary and return exit code
    passed = result.print_summary()
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
