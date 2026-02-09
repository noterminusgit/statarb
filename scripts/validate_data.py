#!/usr/bin/env python3
"""
Validate market data quality before running backtests

This script checks the quality and completeness of market data files required
for backtesting. It validates:
- Data directory structure
- File existence and coverage
- OHLC relationships
- Factor coverage
- Universe size and composition

Usage:
    python3 scripts/validate_data.py --start=20130101 --end=20130630
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path


class DataValidator:
    def __init__(self, start_date, end_date, verbose=True):
        self.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        self.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def log(self, message, level='INFO'):
        """Log message with level"""
        if self.verbose:
            prefix = {
                'INFO': '  ',
                'WARN': '⚠️ ',
                'ERROR': '❌',
                'OK': '✅'
            }.get(level, '  ')
            print(f"{prefix} {message}")

    def validate_directory_structure(self, base_dir, name):
        """Validate data directory exists and has expected structure"""
        self.log(f"Checking {name} directory: {base_dir}")

        if not base_dir or base_dir == "":
            self.errors.append(f"{name} directory not configured in loaddata.py")
            self.log(f"{name} directory not configured", 'ERROR')
            return False

        if not os.path.exists(base_dir):
            self.errors.append(f"{name} directory does not exist: {base_dir}")
            self.log(f"Directory does not exist: {base_dir}", 'ERROR')
            return False

        self.log(f"{name} directory exists", 'OK')
        return True

    def validate_file_coverage(self, base_dir, name, extension='csv'):
        """Check file coverage for date range"""
        dates = pd.bdate_range(self.start_date, self.end_date)
        missing_dates = []

        for date in dates:
            year_dir = os.path.join(base_dir, str(date.year))
            filename = f"{date.strftime('%Y%m%d')}.{extension}"
            filepath = os.path.join(year_dir, filename)

            if not os.path.exists(filepath):
                missing_dates.append(date)

        if missing_dates:
            self.warnings.append(
                f"{name}: Missing {len(missing_dates)}/{len(dates)} files"
            )
            self.log(
                f"Missing {len(missing_dates)}/{len(dates)} files",
                'WARN'
            )
            if len(missing_dates) <= 10:
                for date in missing_dates[:10]:
                    self.log(f"  Missing: {date.strftime('%Y-%m-%d')}", 'WARN')
            return False
        else:
            self.log(f"All {len(dates)} files present", 'OK')
            return True

    def validate_universe_files(self, base_dir):
        """Validate universe file structure and content"""
        self.log(f"\nValidating Universe Files...")

        if not self.validate_directory_structure(base_dir, "Universe"):
            return False

        if not self.validate_file_coverage(base_dir, "Universe"):
            return False

        # Sample check: validate first file structure
        dates = pd.bdate_range(self.start_date, self.end_date)
        first_date = dates[0]
        filepath = os.path.join(
            base_dir,
            str(first_date.year),
            f"{first_date.strftime('%Y%m%d')}.csv"
        )

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                required_cols = ['sid', 'ticker_root', 'country', 'currency']
                missing_cols = [c for c in required_cols if c not in df.columns]

                if missing_cols:
                    self.errors.append(
                        f"Universe file missing columns: {missing_cols}"
                    )
                    self.log(f"Missing columns: {missing_cols}", 'ERROR')
                    return False

                self.log(f"Sample file has {len(df)} stocks", 'OK')
                self.log(f"Columns: {', '.join(df.columns[:10])}", 'INFO')

                return True

            except Exception as e:
                self.errors.append(f"Error reading universe file: {e}")
                self.log(f"Error reading file: {e}", 'ERROR')
                return False

        return True

    def validate_price_files(self, base_dir):
        """Validate price file structure and OHLC relationships"""
        self.log(f"\nValidating Price Files...")

        if not self.validate_directory_structure(base_dir, "Prices"):
            return False

        if not self.validate_file_coverage(base_dir, "Prices"):
            return False

        # Sample check: validate OHLC relationships
        dates = pd.bdate_range(self.start_date, self.end_date)
        sample_dates = dates[::max(1, len(dates)//5)]  # Sample 5 dates

        ohlc_violations = 0

        for date in sample_dates:
            filepath = os.path.join(
                base_dir,
                str(date.year),
                f"{date.strftime('%Y%m%d')}.csv"
            )

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)

                    required_cols = ['sid', 'open', 'high', 'low', 'close']
                    missing_cols = [c for c in required_cols if c not in df.columns]

                    if missing_cols:
                        self.errors.append(
                            f"Price file missing columns: {missing_cols}"
                        )
                        self.log(f"Missing columns: {missing_cols}", 'ERROR')
                        return False

                    # Check OHLC relationships
                    invalid_high = (df['high'] < df['close']) | (df['high'] < df['open'])
                    invalid_low = (df['low'] > df['close']) | (df['low'] > df['open'])

                    violations = invalid_high.sum() + invalid_low.sum()
                    ohlc_violations += violations

                    if violations > 0:
                        self.warnings.append(
                            f"OHLC violations on {date.strftime('%Y-%m-%d')}: {violations} stocks"
                        )

                except Exception as e:
                    self.errors.append(f"Error reading price file: {e}")
                    self.log(f"Error reading {date.strftime('%Y-%m-%d')}: {e}", 'ERROR')
                    return False

        if ohlc_violations > 0:
            self.log(
                f"Found {ohlc_violations} OHLC violations across sampled dates",
                'WARN'
            )
        else:
            self.log("OHLC relationships valid", 'OK')

        return True

    def validate_barra_files(self, base_dir):
        """Validate Barra factor file structure"""
        self.log(f"\nValidating Barra Files...")

        if not self.validate_directory_structure(base_dir, "Barra"):
            return False

        if not self.validate_file_coverage(base_dir, "Barra"):
            return False

        # Sample check: validate factor coverage
        dates = pd.bdate_range(self.start_date, self.end_date)
        first_date = dates[0]
        filepath = os.path.join(
            base_dir,
            str(first_date.year),
            f"{first_date.strftime('%Y%m%d')}.csv"
        )

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)

                required_factors = [
                    'sid', 'beta', 'size', 'momentum', 'resvol'
                ]
                missing_factors = [f for f in required_factors if f not in df.columns]

                if missing_factors:
                    self.errors.append(
                        f"Barra file missing factors: {missing_factors}"
                    )
                    self.log(f"Missing factors: {missing_factors}", 'ERROR')
                    return False

                # Check for industry dummies
                industry_cols = [c for c in df.columns if c.startswith('ind')]
                self.log(f"Found {len(industry_cols)} industry columns", 'OK')

                self.log(f"Sample file has {len(df)} stocks", 'OK')

                return True

            except Exception as e:
                self.errors.append(f"Error reading Barra file: {e}")
                self.log(f"Error reading file: {e}", 'ERROR')
                return False

        return True

    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*70)
        print("DATA VALIDATION REPORT")
        print("="*70)
        print(f"Start Date:   {self.start_date.strftime('%Y-%m-%d')}")
        print(f"End Date:     {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Trading Days: {len(pd.bdate_range(self.start_date, self.end_date))}")
        print("="*70)

        if not self.errors and not self.warnings:
            print("\n✅ ALL VALIDATIONS PASSED")
            print("\nData is ready for backtesting.")
            return 0
        else:
            if self.errors:
                print(f"\n❌ ERRORS: {len(self.errors)}")
                for error in self.errors:
                    print(f"  - {error}")

            if self.warnings:
                print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
                for warning in self.warnings:
                    print(f"  - {warning}")

            if self.errors:
                print("\n❌ VALIDATION FAILED")
                print("\nFix errors before running backtests.")
                return 1
            else:
                print("\n⚠️  VALIDATION PASSED WITH WARNINGS")
                print("\nData may be usable, but review warnings.")
                return 0


def main():
    parser = argparse.ArgumentParser(
        description='Validate market data quality',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--start', required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('--end', required=True, help='End date (YYYYMMDD)')
    parser.add_argument('--univ', help='Universe directory (default: from loaddata.py)')
    parser.add_argument('--prices', help='Prices directory (default: from loaddata.py)')
    parser.add_argument('--barra', help='Barra directory (default: from loaddata.py)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Import loaddata to get default paths
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        import loaddata
        univ_dir = args.univ or loaddata.UNIV_BASE_DIR
        prices_dir = args.prices or loaddata.PRICE_BASE_DIR
        barra_dir = args.barra or loaddata.BARRA_BASE_DIR
    except ImportError:
        print("❌ Error: Could not import loaddata.py")
        return 1

    validator = DataValidator(args.start, args.end, verbose=not args.quiet)

    # Run validations
    validator.validate_universe_files(univ_dir)
    validator.validate_price_files(prices_dir)
    validator.validate_barra_files(barra_dir)

    # Generate report
    return validator.generate_report()


if __name__ == '__main__':
    exit(main())
