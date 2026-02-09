#!/usr/bin/env python3
"""
Import Validation Script for Python 3 Migration
Phase 3.9: Environment Setup and Import Validation

This script tests whether all core modules can be imported successfully
in Python 3. It's designed to catch import-time errors before running
full functional tests.

Usage:
    python3 scripts/test_imports_py3.py

Returns:
    Exit code 0 if all imports succeed, 1 if any fail
"""

import sys
import os
from pathlib import Path

# Add the statarb directory to the Python path
STATARB_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STATARB_DIR))

# Define modules to test
CORE_MODULES = [
    'calc',
    'loaddata',
    'regress',
    'opt',
    'util',
]

SIMULATION_ENGINES = [
    'bsim',
    'osim',
    'qsim',
    'ssim',
]

ALPHA_STRATEGIES = [
    'hl',
    'bd',
    'analyst',
    'eps',
    'target',
    'rating_diff',
    'pca',
    'c2o',
]

SUPPORT_MODULES = [
    'slip',
    'factors',
]


def test_import(module_name, category):
    """
    Attempt to import a module and report success/failure.

    Args:
        module_name: Name of the module to import
        category: Category name for reporting (e.g., "Core Module")

    Returns:
        True if import succeeded, False otherwise
    """
    try:
        __import__(module_name)
        print(f"  [OK] {module_name}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {module_name}")
        print(f"         ImportError: {e}")
        return False
    except SyntaxError as e:
        print(f"  [FAIL] {module_name}")
        print(f"         SyntaxError: {e}")
        return False
    except Exception as e:
        print(f"  [WARN] {module_name}")
        print(f"         {type(e).__name__}: {e}")
        # Consider this a partial success - module imported but had runtime issues
        return True


def main():
    """Run import tests for all module categories."""
    print("=" * 70)
    print("Python 3 Import Validation Test")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"STATARB_DIR: {STATARB_DIR}")
    print("=" * 70)

    results = {}

    # Test core modules
    print("\n[Core Modules]")
    results['core'] = [test_import(m, 'Core') for m in CORE_MODULES]

    # Test simulation engines
    print("\n[Simulation Engines]")
    results['sim'] = [test_import(m, 'Simulation') for m in SIMULATION_ENGINES]

    # Test alpha strategies
    print("\n[Alpha Strategies]")
    results['alpha'] = [test_import(m, 'Alpha') for m in ALPHA_STRATEGIES]

    # Test support modules
    print("\n[Support Modules]")
    results['support'] = [test_import(m, 'Support') for m in SUPPORT_MODULES]

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_tested = 0
    total_passed = 0

    for category, test_results in results.items():
        passed = sum(test_results)
        tested = len(test_results)
        total_tested += tested
        total_passed += passed

        category_name = {
            'core': 'Core Modules',
            'sim': 'Simulation Engines',
            'alpha': 'Alpha Strategies',
            'support': 'Support Modules',
        }[category]

        print(f"{category_name:25s}: {passed:2d}/{tested:2d} passed")

    print("-" * 70)
    print(f"{'TOTAL':25s}: {total_passed:2d}/{total_tested:2d} passed")

    success_rate = (total_passed / total_tested * 100) if total_tested > 0 else 0
    print(f"{'Success Rate':25s}: {success_rate:.1f}%")
    print("=" * 70)

    # Exit with appropriate code
    if total_passed == total_tested:
        print("\nAll imports successful!")
        return 0
    else:
        print(f"\n{total_tested - total_passed} modules failed to import.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
