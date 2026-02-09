#!/usr/bin/env python3
"""
Alpha Calculation Utilities

This module provides convenience imports for commonly used functions
in alpha strategy development. It re-exports functions from calc.py
and util.py to provide a simple import interface for strategy modules.

Usage:
    from alphacalc import *

This was historically used in many alpha strategy files as a convenience
layer. In Python 3, we maintain this for backward compatibility.
"""

from __future__ import division, print_function

# Import calculation functions from calc.py
from calc import (
    winsorize,
    winsorize_by_date,
    winsorize_by_ts,
    winsorize_by_group,
)

# Import utility functions from util.py
from util import (
    filter_expandable,
    remove_dup_cols,
)

# Make everything available with "from alphacalc import *"
__all__ = [
    'winsorize',
    'winsorize_by_date',
    'winsorize_by_ts',
    'winsorize_by_group',
    'filter_expandable',
    'remove_dup_cols',
]
