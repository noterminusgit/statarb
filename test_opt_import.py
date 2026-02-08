#!/usr/bin/env python3
"""
Simple test to validate opt.py imports and basic structure after scipy migration.
"""

from __future__ import division, print_function

import sys
import numpy as np

print("Testing opt.py imports and basic structure...")

try:
    import opt
    print("✓ opt.py imported successfully")
except ImportError as e:
    print("✗ Failed to import opt.py: {}".format(e))
    sys.exit(1)

# Check that key functions exist
required_functions = [
    'objective',
    'objective_grad',
    'objective_detail',
    'slippageFuncAdv',
    'slippageFunc_grad',
    'costsFunc',
    'costsFunc_grad',
    'optimize',
    'init',
    'getUntradeable',
    'setupProblem_scipy',
    'printinfo'
]

missing = []
for func_name in required_functions:
    if not hasattr(opt, func_name):
        missing.append(func_name)
        print("✗ Missing function: {}".format(func_name))
    else:
        print("✓ Found function: {}".format(func_name))

if missing:
    print("\nERROR: Missing {} required functions".format(len(missing)))
    sys.exit(1)

# Check that global parameters exist
required_globals = [
    'kappa',
    'max_sumnot',
    'max_expnot',
    'max_posnot',
    'slip_alpha',
    'slip_beta',
    'slip_delta',
    'slip_gamma',
    'slip_nu',
    'execFee'
]

missing_globals = []
for global_name in required_globals:
    if not hasattr(opt, global_name):
        missing_globals.append(global_name)
        print("✗ Missing global: {}".format(global_name))
    else:
        print("✓ Found global: {} = {}".format(global_name, getattr(opt, global_name)))

if missing_globals:
    print("\nERROR: Missing {} required globals".format(len(missing_globals)))
    sys.exit(1)

# Test simple objective function call with dummy data
print("\nTesting objective function with dummy data...")
try:
    num_secs = 10
    target = np.random.randn(num_secs) * 1000
    positions = np.random.randn(num_secs) * 1000
    mu = np.random.randn(num_secs) * 0.001
    rvar = np.abs(np.random.randn(num_secs)) * 0.0001
    factors = np.random.randn(5, num_secs) * 0.1
    fcov = np.eye(5) * 0.01
    advp = np.abs(np.random.randn(num_secs)) * 1e6
    advpt = advp * 0.5
    vol = np.abs(np.random.randn(num_secs)) * 0.02
    mktcap = np.abs(np.random.randn(num_secs)) * 1e9
    brate = np.random.randn(num_secs) * 0.001
    price = np.abs(np.random.randn(num_secs)) * 50
    untradeable_info = (0.0, 0.0, np.zeros(5))

    obj_value = opt.objective(
        target, opt.kappa, opt.slip_gamma, opt.slip_nu, positions,
        mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price,
        opt.execFee, untradeable_info
    )
    print("✓ objective() returned: {:.2f}".format(obj_value))

    grad = opt.objective_grad(
        target, opt.kappa, opt.slip_gamma, opt.slip_nu, positions,
        mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price,
        opt.execFee, untradeable_info
    )
    print("✓ objective_grad() returned array of shape: {}".format(grad.shape))

except Exception as e:
    print("✗ Error testing objective function: {}".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS: All basic tests passed!")
print("="*60)
print("\nopt.py has been successfully migrated to scipy.optimize.")
print("The module can be imported and key functions are callable.")
print("\nNext steps:")
print("1. Run integration tests with actual market data")
print("2. Compare optimization results with Python 2 baseline")
print("3. Validate numerical equivalence (positions within 1%)")
