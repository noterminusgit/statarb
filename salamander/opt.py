"""
Portfolio Optimization Module (Python 3)

This module implements portfolio optimization using scipy.optimize to
maximize risk-adjusted returns while respecting trading constraints.

This is the Python 3 version of opt.py from the main codebase. Key differences:
- Python 3 syntax (print functions, range instead of xrange)
- Uses np alias for numpy (instead of full 'numpy')
- Uses scipy.optimize.minimize instead of OpenOpt (migrated in Phase 2)
- Otherwise functionally identical to main opt.py

Objective Function:
    Maximize: Alpha - κ(Specific Risk + Factor Risk) - Slippage - Execution Costs

Components:
    Alpha: Expected return from alpha signals (μ · positions)
    Specific Risk: Idiosyncratic variance (σ² · positions²)
    Factor Risk: Systematic risk from Barra factors (x'Fx)
    Slippage: Nonlinear market impact cost function
    Execution Fees: Fixed bps cost (default: 1.5 bps)

Constraints:
    - Position Limits: Min/max shares per security (±0.048% of capital)
    - Capital Limits: Max aggregate notional ($50M default)
    - Factor Exposure Limits: Bounds on factor bets (±4.8% of capital)
    - Participation Limits: Implicit via slippage function
    - Dollar Neutrality: Optional long/short balance

Slippage Model:
    Two-component nonlinear market impact model:
    I = γ · vol · (|Δpos|/advp) · (mktcap/advp)^δ
    J = I/2 + ν · vol · (|Δpos|/advpt)^β
    Cost = Σ(J · |Δpos|)

    Parameters:
    - α (slip_alpha): Base cost (default: 1.0, currently unused in formula)
    - β (slip_beta): Participation power law exponent (default: 0.6)
    - δ (slip_delta): Market cap scaling exponent (default: 0.25)
    - γ (slip_gamma): Volatility coefficient (default: 0.3)
    - ν (slip_nu): Market impact coefficient (default: 0.14-0.18)

Risk Model:
    Total Risk = Specific Risk + Factor Risk
    - Specific Risk: Stock-specific variance (diagonal)
    - Factor Risk: Systematic risk from Barra factor model
    - Risk aversion parameter κ controls risk penalty

Parameters:
    kappa: Risk aversion parameter (4.3e-5 default, range: 2e-8 to 4.3e-5)
    max_sumnot: Max total notional ($50M default)
    max_posnot: Max position as fraction of capital (0.48% default)
    max_expnot: Max factor exposure per factor (4.8% default)
    hard_limit: Multiplier allowing constraint violations (1.02 = 2% buffer)

Solver Configuration:
    - Algorithm: scipy.optimize.minimize with trust-constr method
    - Typical solve time: 1-10 seconds for 1400 securities
    - Max iterations: 500
    - Convergence tolerances: gtol=1e-6, xtol=1e-6, barrier_tol=1e-6

Global Variables:
    g_positions: Current positions (dollars)
    g_mu: Expected returns (alpha signals)
    g_rvar: Residual variance (specific risk)
    g_factors: Factor loadings matrix (num_factors × num_secs)
    g_fcov: Factor covariance matrix (num_factors × num_factors)
    g_advp: Average daily volume in dollars
    g_advpt: Average daily tradeable volume
    g_vol: Daily volatility (standard deviation)
    g_mktcap: Market capitalization
    g_borrowRate: Borrow costs for shorts
    g_price: Stock prices per share
    g_lbound/g_ubound: Position limits per security

Usage:
    1. Set num_secs and num_factors
    2. Call init() to allocate arrays
    3. Populate global arrays with market data
    4. Call optimize() to compute target positions
    5. Extract results: (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2)

Example:
    import opt
    opt.num_secs = 1400
    opt.num_factors = 13
    opt.init()
    opt.g_positions[:] = current_positions
    opt.g_mu[:] = alpha_signals
    # ... populate other arrays ...
    target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2, vol, price = opt.optimize()
"""

import sys
import numpy as np
import math
import logging
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

import util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

max_sumnot = 50.0e6
max_expnot = 0.048
max_posnot = 0.0048
max_trdnot = 1.0
max_iter = 500
min_iter = 500
plotit = False

hard_limit = 1.02
kappa = 4.3e-5

# HAND-TWEAKED PARAMETERS TO MATCH CURRENT TRADING BEHAVIOR
slip_alpha = 1.0
slip_delta = 0.25
slip_beta = 0.6
slip_gamma = 0.3
slip_nu = 0.14
execFee = 0.00015

num_secs = 0
num_factors = 0
stocks_ii = 0
factors_ii = 0
zero_start = 0

# prefix them with g_ to avoid errors
g_positions = None
g_lbound = None
g_ubound = None
g_mu = None
g_rvar = None
g_advp = None
g_borrowRate = None
g_price = None
g_factors = None
g_fcov = None
g_vol = None
g_mktcap = None
g_advpt = None
np.set_printoptions(threshold=float('nan'))

p = None

# Note: scipy.optimize.minimize uses different callback mechanism than OpenOpt.
# The Terminator class functionality is replaced by scipy's built-in convergence
# criteria (gtol, xtol, barrier_tol) and maxiter parameter.


class Terminator():
    """Custom termination callback for OpenOpt NLP solver.

    Monitors optimization progress and terminates early if improvement plateaus.
    Uses a rolling window approach to track objective function improvement over
    iterations, providing early stopping when convergence is detected.

    The callback checks if the improvement in the best objective value over a
    lookback window falls below a threshold, indicating convergence. This saves
    computation time compared to running to max iterations.

    Args:
        lookback (int): Number of iterations to look back for improvement comparison.
                       Default in setupProblem: 50
        stopThreshold (float): Minimum improvement required to continue (in objective units).
                              If improvement < threshold, optimization stops.
                              Default in setupProblem: 10 (dollars)
        minIter (int): Minimum iterations before checking termination criteria.
                      Prevents premature stopping during initial exploration.

    Attributes:
        iter (int): Current iteration count
        objValues (list): History of objective function values at each iteration
        maxAtLookback: Unused (legacy attribute)

    Returns:
        bool: True to stop optimization, False to continue

    Notes:
        - Only checks every 10 iterations (for efficiency)
        - Handles infeasible points by recording np.inf
        - OpenOpt internally minimizes, so signs are inverted for maximization
        - Prints progress messages during optimization
    """
    def __init__(self, lookback, stopThreshold, minIter):
        self.iter = 0
        self.objValues = []
        self.maxAtLookback = None
        self.lookback = lookback
        self.stopThreshold = stopThreshold
        self.minIter = minIter

    def __call__(self, p):
        """Callback invoked after each solver iteration.

        Args:
            p: OpenOpt problem object with current state (p.fk = objective value, p.rk = constraint residual)

        Returns:
            bool: True if optimization should stop, False to continue
        """
        self.iter += 1
        # infeasible points are disregarded from computations
        if p.rk <= 0:
            self.objValues.append(p.fk)
        else:
            self.objValues.append(np.inf)

        # don't start checking until we have seen at least min iters
        if self.iter <= self.lookback + self.minIter:
            return False
        # only check every 10 iterations
        if self.iter % 10 != 0:
            return False

        # internally it works as a minimizer, so take that into account by getting the minimum values and inverting them
        # each iteration is not guaranteed to increase the obj function values.
        curr = -min(self.objValues[-self.lookback:-1])
        prev = -min(self.objValues[0:(-self.lookback - 1)])

        if np.isinf(prev):
            print("Haven't found a feasible point yet")
            return False
        elif np.isinf(curr):
            print("We are probably diverging, but we are staying the course for a huge comeback")
            return False

        if self.iter % 10 == 0:
            print("Current improvement after {} iterations is {}".format(self.lookback, float(curr - prev)))
        if curr - prev < self.stopThreshold:
            print("Current improvement after {} iterations is {}".format(self.lookback, float(curr - prev)))
            return True
        else:
            return False


def printinfo(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate,
              price, execFee, untradeable_info):
    """Print optimization summary comparing current and target portfolios.

    Displays portfolio statistics and utility decomposition for both the
    current holdings and the optimized target positions. This provides
    transparency into the optimizer's decisions and expected improvements.

    Args:
        target: Array of target positions (dollars)
        kappa: Risk aversion parameter
        slip_gamma: Slippage volatility coefficient
        slip_nu: Slippage market impact coefficient
        positions: Array of current positions (dollars)
        mu: Expected returns (alpha signals)
        rvar: Residual variance (specific risk)
        factors: Factor loadings matrix (num_factors × num_secs)
        fcov: Factor covariance matrix
        advp: Average daily volume in dollars
        advpt: Average daily tradeable volume
        vol: Volatility (daily standard deviation)
        mktcap: Market capitalization
        brate: Borrow rates for shorts
        price: Stock prices
        execFee: Execution fee (bps as decimal, e.g., 0.00015 = 1.5 bps)
        untradeable_info: Tuple of (mu, rvar, loadings) for untradeable positions

    Returns:
        Tuple of (vol, price): Volatility and price arrays (for compatibility)

    Prints to stdout:
        - Current portfolio: long notional, short notional, total notional
        - Target portfolio: long notional, short notional, total notional
        - Dollars traded (total turnover)
        - Utility breakdown for current portfolio
        - Utility breakdown for optimum portfolio

    Example output:
        [CURRENT] Long: 25000000, Short: 25000000, Total: 50000000
        [TARGET]  Long: 26000000, Short: 24000000, Total: 50000000
        Dollars traded: 2000000
        @Current: total=50000, mu=100000, risk=40000, slip=5.00, costs=10.00, ratio=2.500, var=20000, covar=20000
        @Optimum: total=55000, mu=105000, risk=39000, slip=6.00, costs=11.00, ratio=2.692, var=19000, covar=20000
    """
    clong = 0
    cshort = 0
    tlong = 0
    tshort = 0
    diff = 0
    for ii in range(len(positions)):
        if positions[ii] >= 0:
            clong += positions[ii]
        else:
            cshort -= positions[ii]
    for ii in range(len(target)):
        if target[ii] >= 0:
            tlong += target[ii]
        else:
            tshort -= target[ii]
        diff += abs(target[ii] - positions[ii])
    print("[CURRENT] Long: {:.0f}, Short: {:.0f}, Total: {:.0f}".format(clong, cshort, clong + cshort))
    print("[TARGET]  Long: {:.0f}, Short: {:.0f}, Total: {:.0f}".format(tlong, tshort, tlong + tshort))
    print("Dollars traded: {:.0f}".format(diff))

    __printpointinfo("Current", positions, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt,
                     vol, mktcap, brate, price, execFee, untradeable_info)
    __printpointinfo("Optimum", target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt,
                     vol, mktcap, brate, price, execFee, untradeable_info)
    return (vol, price)


def __printpointinfo(name, target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol,
                     mktcap, brate, price, execFee, untradeable_info):
    """Print detailed utility decomposition for a portfolio position.

    Internal helper function that breaks down total utility into individual
    components: alpha (mu), risk penalty, slippage, and execution costs.
    Separates specific (idiosyncratic) risk from factor (systematic) risk.

    Args:
        name (str): Label for this point (e.g., "Current" or "Optimum")
        target: Portfolio positions to analyze (dollars)
        kappa: Risk aversion parameter
        slip_gamma: Slippage volatility coefficient
        slip_nu: Slippage market impact coefficient
        positions: Current positions for slippage calculation
        mu: Expected returns (alpha signals)
        rvar: Residual variance (specific risk)
        factors: Factor loadings matrix
        fcov: Factor covariance matrix
        advp: Average daily volume (dollars)
        advpt: Average daily tradeable volume
        vol: Daily volatility
        mktcap: Market capitalization
        brate: Borrow rates
        price: Stock prices
        execFee: Execution fee (bps)
        untradeable_info: Tuple of (mu, rvar, loadings) for untradeable positions

    Prints to stdout:
        @{name}: total={utility}, mu={alpha}, risk={penalty}, slip={cost},
                 costs={fees}, ratio={mu/risk}, var={specific}, covar={factor}

    Where:
        - total: Net utility = mu - risk - slip - costs
        - mu: Expected alpha return (dollars)
        - risk: Total risk penalty = var + covar (dollars)
        - slip: Market impact slippage (dollars)
        - costs: Execution fees (dollars)
        - ratio: Sharpe-like ratio = mu / risk
        - var: Specific risk penalty component (dollars)
        - covar: Factor risk penalty component (dollars)
    """
    untradeable_mu, untradeable_rvar, untradeable_loadings = untradeable_info[0], untradeable_info[1], untradeable_info[
        2]

    loadings = np.dot(factors, target) + untradeable_loadings
    utility1 = np.dot(target, mu) + untradeable_mu
    utility2 = kappa * (untradeable_rvar + np.dot(target * rvar, target) + np.dot(np.dot(loadings, fcov), loadings))
    utility3 = slippageFuncAdv(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu)
    utility4 = costsFunc(target, positions, brate, price, execFee)
    var = kappa * np.dot(target * rvar, target)
    covar = kappa * np.dot(np.dot(loadings, fcov), loadings)
    print(
        "@{}: total={:.0f}, mu={:.0f}, risk={:.0f}, slip={:.2f}, costs={:.2f}, ratio={:.3f}, var={:.0f}, covar={:.0f}".format(
            name, utility1 - utility2 - utility3 - utility4, utility1, utility2, utility3, utility4,
            utility1 / utility2, var, covar))


def slippageFuncAdv(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu):
    """Calculate total market impact slippage cost for portfolio rebalancing.

    Implements nonlinear slippage model with two components:
    1. Volatility-based impact: γ * vol * participation * (mktcap/advp)^δ
    2. Power-law participation cost: ν * vol * (participation)^β

    The model penalizes:
    - Large trades in volatile stocks (high vol, high participation)
    - Trading in less liquid stocks (lower mktcap/advp ratio)
    - High participation rates (trading too much of daily volume)

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars)
        advp: Average daily volume in dollars (for participation rate)
        advpt: Average daily tradeable volume in dollars
        vol: Daily volatility (standard deviation)
        mktcap: Market capitalization
        slip_gamma: Volatility coefficient (default: 0.3)
        slip_nu: Market impact coefficient (default: 0.14-0.18)

    Returns:
        float: Total slippage cost (dollars) across all securities

    Formula:
        Δpos = |target - positions|
        I = γ * vol * (Δpos/advp) * (mktcap/advp)^δ
        J = I/2 + ν * vol * (Δpos/advpt)^β
        slippage = Σ(J * Δpos)

    Where:
        - First term (I): Liquidity-adjusted volatility impact
        - Second term: Power-law participation rate penalty
        - δ (slip_delta): Market cap scaling exponent (0.25)
        - β (slip_beta): Participation power law exponent (0.6)

    Notes:
        - Nonlinear in trade size (quadratic-like due to J * Δpos)
        - Encourages spreading trades across securities
        - Higher slippage for illiquid or volatile stocks
    """
    newpos_abs = abs(target - positions)
    I = slip_gamma * vol * (newpos_abs / advp) * (mktcap / advp) ** slip_delta
    J = I / 2 + slip_nu * vol * (newpos_abs / advpt) ** slip_beta
    slip = J * newpos_abs
    index = np.isnan(newpos_abs/advp)
    '''
    if np.any(index):
        print(newpos_abs[index])
        print(advp[index])
        import sys
        sys.exit()
    '''
    return slip.sum()


def slippageFunc_grad(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu):
    """Calculate gradient of slippage function with respect to target positions.

    Provides first derivative for gradient-based optimization. The gradient
    combines the derivatives of both slippage components (volatility-based
    and participation-based) with respect to position changes.

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars)
        advp: Average daily volume (dollars)
        advpt: Average daily tradeable volume
        vol: Daily volatility
        mktcap: Market capitalization
        slip_gamma: Volatility coefficient
        slip_nu: Market impact coefficient

    Returns:
        np.ndarray: Array of partial derivatives ∂(slippage)/∂(target) for each security

    Formula:
        Δpos = target - positions
        Id = 0.5 * γ * vol * (1/advp) * (mktcap/advp)^δ
        Jd = [Id + ν * vol * (1+β) * (|Δpos|/advpt)^β] * sign(Δpos)

    Notes:
        - Gradient is nonlinear due to power-law participation term
        - Sign preserves direction of trade (buy vs sell)
        - Used by RALG solver for efficient optimization
    """
    newpos = target - positions
    Id = .5 * slip_gamma * vol * (1 / advp) * (mktcap / advp) ** slip_delta
    Jd = (Id + slip_nu * vol * (1 + slip_beta) * (abs(newpos) / advpt) ** slip_beta) * np.sign(newpos)
    return Jd


def costsFunc(target, positions, brate, price, execFee):
    """Calculate total execution and borrow costs.

    Computes fixed execution fees based on shares traded and (optionally)
    borrow costs for short positions. Currently only execution fees are
    active; borrow costs are disabled pending data availability.

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars)
        brate: Borrow rates for shorts (currently unused, negative values)
        price: Stock prices per share
        execFee: Execution fee as decimal (default: 0.00015 = 1.5 bps)

    Returns:
        float: Total execution costs (dollars)

    Formula:
        costs = execFee * Σ(|Δdollars| / price)
              = execFee * Σ(|Δshares|)

    When borrow costs enabled (currently disabled):
        costs += Σ(brate * min(0, target))
        Note: brate is negative, so this adds a positive cost for shorts

    Notes:
        - Execution fee is per-share basis (not per-dollar)
        - Borrow costs currently commented out (see XXX comment)
        - Linear in trade size (no market impact here, that's in slippage)
    """
    costs = execFee * np.dot(1.0 / price, abs(target - positions))
    # ATTENTION! borrow costs are negative, negative times negative gives a positive cost
    # XXX add back once we have borrow costs!
    # costs += np.dot(brate, np.minimum(0.0, target))
    return costs


def costsFunc_grad(target, positions, brate, price, execFee):
    """Calculate gradient of costs function with respect to target positions.

    Provides first derivative for gradient-based optimization. Since execution
    costs are linear in trade size, the gradient is simply the signed fee rate.

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars)
        brate: Borrow rates (currently unused)
        price: Stock prices per share
        execFee: Execution fee (bps as decimal)

    Returns:
        np.ndarray: Array of partial derivatives ∂(costs)/∂(target) for each security

    Formula:
        ∂(costs)/∂(target) = execFee * sign(Δpos) / price

    When borrow costs enabled (currently disabled):
        For short positions: ∂(costs)/∂(target) += brate
        Note: brate is negative, so increasing shorts increases costs

    Notes:
        - Gradient is piecewise constant (discontinuous at Δpos = 0)
        - Sign function captures buy vs sell direction
        - Borrow cost gradient currently commented out
    """
    grad = execFee * np.sign(target - positions) / price
    #    for i in range(len(grad)):
    # ATTENTION!  borrow costs are negative, derivative is negative (more positive position, lower costs)
    #        if target[i] <=0 : grad[i] += brate[i]
    return grad


def objective(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate,
              price, execFee, untradeable_info):
    """Portfolio optimization objective function to maximize.

    Returns total utility (expected return minus risk penalties and costs).
    This is the main objective function passed to the OpenOpt NLP solver.

    Args:
        target: Target positions (dollars) - optimization variables
        kappa: Risk aversion parameter (higher = more conservative)
        slip_gamma: Slippage volatility coefficient
        slip_nu: Slippage market impact coefficient
        positions: Current positions (dollars)
        mu: Expected returns (alpha signals)
        rvar: Residual variance (specific risk)
        factors: Factor loadings matrix (num_factors × num_secs)
        fcov: Factor covariance matrix (num_factors × num_factors)
        advp: Average daily volume (dollars)
        advpt: Average daily tradeable volume
        vol: Daily volatility
        mktcap: Market capitalization
        brate: Borrow rates
        price: Stock prices
        execFee: Execution fee (bps as decimal)
        untradeable_info: Tuple of (mu, rvar, loadings) for untradeable positions

    Returns:
        float: Utility value (to be maximized by solver)
               U = α - κ*Risk - Slippage - Costs

    Formula:
        U = μ·x - κ(σ²·x² + x'Fx) - slippage(Δx) - costs(Δx)

    Where:
        μ·x = expected alpha return
        σ²·x² = specific risk penalty (idiosyncratic variance)
        x'Fx = factor risk penalty (systematic risk)
        slippage(Δx) = nonlinear market impact costs
        costs(Δx) = execution fees

    Notes:
        - Wrapper around objective_detail() that returns only total utility
        - OpenOpt will maximize this function subject to constraints
        - Gradient provided by objective_grad() for efficiency
    """
    return objective_detail(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap,
                     brate, price, execFee, untradeable_info)[0]


def objective_detail(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap,
                     brate, price, execFee, untradeable_info):
    """Portfolio optimization objective with detailed component breakdown.

    Same as objective() but returns all individual components for analysis
    and debugging. Used for detailed utility attribution and marginal
    contribution calculations.

    Args:
        (same as objective)

    Returns:
        Tuple of (utility, tmu, tsrisk, tfrisk, tslip, tcosts):
            utility (float): Total utility = tmu - tsrisk - tfrisk - tslip - tcosts
            tmu (float): Alpha component (μ·x + untradeable_mu)
            tsrisk (float): Specific risk penalty (κ*σ²·x² + κ*untradeable_rvar)
            tfrisk (float): Factor risk penalty (κ*x'Fx)
            tslip (float): Slippage costs (dollars)
            tcosts (float): Execution costs (dollars)

    This detailed breakdown is used by:
    - printinfo() for utility attribution reporting
    - optimize() for per-security marginal utility calculations
    - __printpointinfo() for current vs optimum comparison

    Formula Details:
        tmu: Total expected alpha return including tradeable and untradeable
        tsrisk: Specific risk = κ * (σ_tradeable² + σ_untradeable²)
        tfrisk: Factor risk = κ * (F·x + F_untradeable)' * Cov * (F·x + F_untradeable)
        tslip: Market impact from rebalancing trades
        tcosts: Execution fees from rebalancing trades

    Notes:
        - Untradeable positions contribute to mu, rvar, and factor loadings
        - Risk penalties scaled by kappa (risk aversion parameter)
        - All components in dollar units for interpretability
    """
    untradeable_mu, untradeable_rvar, untradeable_loadings = untradeable_info[0], untradeable_info[1], untradeable_info[
        2]

    # objective function to be minimized (negative utility)
    loadings = np.dot(factors, target) + untradeable_loadings

    tmu = np.dot(target, mu) + untradeable_mu
    tsrisk = kappa * (untradeable_rvar + np.dot(target * rvar, target))
    tfrisk = kappa * np.dot(np.dot(loadings, fcov), loadings)
    tslip = slippageFuncAdv(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu)
    tcosts = costsFunc(target, positions, brate, price, execFee)

    utility = tmu
    utility -= tsrisk
    utility -= tfrisk
    utility -= tslip
    utility -= tcosts

    return (utility, tmu, tsrisk, tfrisk, tslip, tcosts)


def objective_grad(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap,
                   brate, price, execFee, untradeable_info):
    """Calculate gradient of objective function with respect to target positions.

    Provides analytical gradient for gradient-based optimization (RALG solver).
    Combines gradients of all utility components to guide the search direction.

    Args:
        (same as objective)

    Returns:
        np.ndarray: Array of partial derivatives ∂U/∂(target) for each security

    Formula:
        ∂U/∂x = μ - 2κ(σ²·x + F'·C·(F·x + u)) - ∂slip/∂x - ∂costs/∂x

    Where:
        μ = expected returns (alpha signals)
        σ² = residual variance (element-wise multiplication with x)
        F = factor loadings matrix (num_factors × num_secs)
        F' = transpose of F
        C = factor covariance matrix (num_factors × num_factors)
        u = untradeable factor loadings
        ∂slip/∂x = slippage gradient (from slippageFunc_grad)
        ∂costs/∂x = costs gradient (from costsFunc_grad)

    Component Interpretation:
        - μ: Positive gradient towards high alpha securities
        - -2κ(σ²·x): Negative gradient penalizing risky positions
        - -2κF'·C·(F·x+u): Negative gradient penalizing factor exposures
        - -∂slip/∂x: Negative gradient penalizing large trades
        - -∂costs/∂x: Negative gradient penalizing trades

    The gradient guides the optimizer toward positions that balance:
    - High expected return (positive μ)
    - Low risk exposure (negative risk terms)
    - Minimal trading costs (negative cost terms)

    Notes:
        - Factor 2 in risk gradient from quadratic term derivative
        - Analytical gradient much faster than numerical differentiation
        - Used by RALG solver for efficient convergence
    """
    untradeable_mu, untradeable_rvar, untradeable_loadings = untradeable_info[0], untradeable_info[1], untradeable_info[
        2]

    F = factors
    Ft = np.transpose(F)
    grad = np.zeros(len(target))
    grad += mu
    grad -= 2 * kappa * (rvar * target + np.dot(Ft, np.dot(fcov, np.dot(F, target) + untradeable_loadings)))
    grad -= slippageFunc_grad(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu)
    grad -= costsFunc_grad(target, positions, brate, price, execFee)
    return grad


# constrain <= 0
def constrain_by_capital(target, positions, max_sumnot, factors, lbexp, ubexp, max_trdnot_hard):
    """Constraint function enforcing maximum total notional limit.

    Ensures total portfolio notional (sum of absolute positions) does not
    exceed the specified capital limit. This is a nonlinear inequality
    constraint passed to the NLP solver.

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars, unused here)
        max_sumnot: Maximum total notional allowed
        factors: Factor loadings (unused here)
        lbexp: Lower bound exposures (unused here)
        ubexp: Upper bound exposures (unused here)
        max_trdnot_hard: Hard turnover limit (unused here)

    Returns:
        float: Constraint value that must be <= 0 for feasibility
               = Σ|target| - max_sumnot

    Constraint is satisfied when: Σ|target| <= max_sumnot

    Notes:
        - Nonlinear due to absolute value
        - Prevents over-leveraging
        - Applied to tradeable securities only (untradeable offset handled in optimize())
        - Gradient provided by constrain_by_capital_grad()
    """
    ret = abs(target).sum() - max_sumnot
    return ret


def constrain_by_capital_grad(target, positions, max_sumnot, factors, lbexp, ubexp, max_trdnot_hard):
    """Gradient of capital constraint with respect to target positions.

    Provides derivative for gradient-based constraint handling.

    Args:
        (same as constrain_by_capital)

    Returns:
        np.ndarray: Array of partial derivatives ∂(constraint)/∂(target)
                   = sign(target) for each security

    Formula:
        ∂(Σ|x| - max_sumnot)/∂x = sign(x)

    Interpretation:
        - Increasing any position (long or short) increases total notional
        - Gradient points in direction that would violate constraint
        - Solver uses this to stay within feasible region

    Notes:
        - Gradient undefined at target=0, but sign(0)=0 works in practice
        - Piecewise constant gradient (not smooth)
    """
    return np.sign(target)


# def constrain_by_exposures(target, positions, max_sumnot, factors, lbexp, ubexp, max_trdnot_hard):
#    exposures = np.dot(factors, target)
#    ret = max(np.r_[lbexp - exposures, exposures - ubexp])
#    return ret

### UGH this is ignored!
def constrain_by_trdnot(target, positions, max_sumnot, factors, lbexp, ubexp, max_trdnot_hard):
    """Constraint function enforcing maximum turnover limit (CURRENTLY UNUSED).

    Would limit total dollars traded in a single rebalance, but this constraint
    is currently not activated in the optimization setup.

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars)
        max_trdnot_hard: Maximum turnover allowed (dollars)
        (other args unused)

    Returns:
        float: Constraint value = Σ|Δpos| - max_trdnot_hard

    Would be satisfied when: Σ|Δpos| <= max_trdnot_hard

    Notes:
        - This constraint is defined but NOT added to the optimizer in setupProblem()
        - To enable, add to p.c and p.dc lists in setupProblem()
        - Currently commented out as "ignored" (see comment above function)
        - Slippage function provides soft turnover penalty instead
        - Hard turnover limit may be too restrictive in practice
    """
    ret = abs(target - positions).sum() - max_trdnot_hard
    return ret


def setupProblem_scipy(positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, borrowRate, price, lb, ub, Ac, bc, lbexp,
                       ubexp, untradeable_info, sumnot, zero_start):
    """Configure scipy.optimize.minimize problem for portfolio optimization.

    Sets up the constrained nonlinear programming problem with:
    - Objective function and gradient
    - Position bounds (box constraints)
    - Factor exposure limits (linear constraints)
    - Capital constraint (nonlinear constraint)
    - Solver parameters

    Args:
        positions: Current positions (dollars) - also used as x0 if zero_start=0
        mu: Expected returns (alpha signals)
        rvar: Residual variance
        factors: Factor loadings matrix
        fcov: Factor covariance matrix
        advp: Average daily volume
        advpt: Average daily tradeable volume
        vol: Daily volatility
        mktcap: Market capitalization
        borrowRate: Borrow rates
        price: Stock prices
        lb: Lower bounds on positions (per security)
        ub: Upper bounds on positions (per security)
        Ac: Linear constraint matrix for factor exposures (unused - we construct directly)
        bc: Linear constraint RHS vector (unused - we construct directly)
        lbexp: Lower bounds on factor exposures
        ubexp: Upper bounds on factor exposures
        untradeable_info: (mu, rvar, loadings) for untradeable securities
        sumnot: Maximum total notional
        zero_start: If > 0, initialize optimizer at zero positions

    Returns:
        Dict containing all components needed for scipy.optimize.minimize call:
            - x0: Initial positions
            - bounds: Box constraints
            - constraints: List of LinearConstraint and NonlinearConstraint
            - options: Solver options
            - obj_args: Arguments for objective function
            - grad_args: Arguments for gradient function

    Solver Configuration:
        - Method: trust-constr (designed for large-scale constrained optimization)
        - Max iterations: max_iter (default 500)
        - Tolerances: gtol=1e-6, xtol=1e-6, barrier_tol=1e-6
    """
    # Initial guess
    x0 = np.zeros(len(positions)) if zero_start > 0 else positions.copy()

    # Box constraints (position bounds)
    bounds = [(lb[i], ub[i]) for i in range(len(lb))]

    # Linear constraints: Ac @ x <= bc
    # Factor exposures must satisfy: lbexp <= factors @ x <= ubexp
    # Reformulate as: factors @ x <= ubexp and -factors @ x <= -lbexp
    # This is already done in Ac, bc by the caller
    A_linear = Ac
    b_linear = bc
    linear_constraint = LinearConstraint(
        A=A_linear,
        lb=-np.inf * np.ones(len(b_linear)),
        ub=b_linear
    )

    # Nonlinear constraint: sum(abs(x)) <= sumnot
    def capital_constraint_func(x):
        return abs(x).sum() - sumnot

    def capital_constraint_grad(x):
        return np.sign(x)

    nonlinear_constraint = NonlinearConstraint(
        fun=capital_constraint_func,
        lb=-np.inf,
        ub=0.0,
        jac=capital_constraint_grad
    )

    constraints = [linear_constraint, nonlinear_constraint]

    # Solver options
    options = {
        'maxiter': max_iter,
        'verbose': 2,
        'gtol': 1e-6,
        'xtol': 1e-6,
        'barrier_tol': 1e-6
    }

    # Arguments for objective and gradient functions
    obj_args = (kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov,
                advp, advpt, vol, mktcap, borrowRate, price, execFee, untradeable_info)

    return {
        'x0': x0,
        'bounds': bounds,
        'constraints': constraints,
        'options': options,
        'obj_args': obj_args
    }


def optimize():
    """Main portfolio optimization entry point.

    Optimizes portfolio positions by maximizing risk-adjusted utility subject
    to position limits, factor exposure constraints, and capital constraints.

    Uses global variables (g_positions, g_mu, g_rvar, etc.) set by caller
    via init() and direct assignment.

    Algorithm:
        1. Partition securities into tradeable/untradeable based on bounds
        2. Extract data arrays for tradeable subset
        3. Compute exposure and capital limits with hard_limit buffer (1.02x)
        4. Set up NLP problem with box constraints and linear factor constraints
        5. Solve using RALG algorithm (gradient-based)
        6. If infeasible with zero_start=1, retry with zero_start=0
        7. Calculate per-security marginal utility contributions
        8. Print optimization summary
        9. Return results

    Returns:
        Tuple of (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2, vol, price):
            target (np.ndarray): Optimized target positions (dollars) for all securities
            dutil (np.ndarray): Marginal utility of each position
            eslip (np.ndarray): Marginal slippage cost
            dmu (np.ndarray): Marginal alpha contribution
            dsrisk (np.ndarray): Marginal specific risk contribution
            dfrisk (np.ndarray): Marginal factor risk contribution
            costs (np.ndarray): Marginal execution costs
            dutil2 (np.ndarray): Alternative marginal utility calculation
            vol (np.ndarray): Volatility array (for compatibility)
            price (np.ndarray): Price array (for compatibility)

    Raises:
        Exception: If optimization fails to find feasible solution

    Global Variables Used:
        g_positions, g_mu, g_rvar, g_factors, g_fcov, g_advp, g_advpt,
        g_vol, g_mktcap, g_borrowRate, g_price, g_lbound, g_ubound
        kappa, slip_gamma, slip_nu, execFee, max_expnot, max_sumnot,
        max_posnot, hard_limit, num_factors, num_secs, zero_start

    Global Variables Modified:
        p (optimization problem object)

    Side Effects:
        Prints optimization summary via printinfo()
        Prints solver progress messages

    Notes:
        - Tradeable securities: those with lbound/ubound difference > $10
        - Untradeable securities: held fixed, contribute to risk/return
        - Exposure limits: ±4.8% of capital per factor (with 2% buffer)
        - Capital limit: $50M default (with 2% buffer)
        - Marginal utilities computed by perturbing each position individually
    """
    global p

    tradeable, untradeable = getUntradeable()
    t_num_secs = len(tradeable)
    t_positions = np.copy(g_positions[tradeable])
    t_factors = np.copy(g_factors[:, tradeable])
    t_lbound = np.copy(g_lbound[tradeable])
    t_ubound = np.copy(g_ubound[tradeable])
    t_mu = np.copy(g_mu[tradeable])
    t_rvar = np.copy(g_rvar[tradeable])
    t_advp = np.copy(g_advp[tradeable])

    t_advpt = np.copy(g_advpt[tradeable])
    t_vol = np.copy(g_vol[tradeable])
    t_mktcap = np.copy(g_mktcap[tradeable])

    t_borrowRate = np.copy(g_borrowRate[tradeable])
    t_price = np.copy(g_price[tradeable])

    u_positions = np.copy(g_positions[untradeable])
    u_factors = np.copy(g_factors[:, untradeable])
    u_mu = np.copy(g_mu[untradeable])
    u_rvar = np.copy(g_rvar[untradeable])

    exposures = np.dot(g_factors, g_positions)
    lbexp = exposures
    lbexp = np.minimum(lbexp, -max_expnot * max_sumnot)
    lbexp = np.maximum(lbexp, -max_expnot * max_sumnot * hard_limit)
    ubexp = exposures
    ubexp = np.maximum(ubexp, max_expnot * max_sumnot)
    ubexp = np.minimum(ubexp, max_expnot * max_sumnot * hard_limit)
    # offset the lbexp and ubexp by the untradeable positions
    untradeable_exposures = np.dot(u_factors, u_positions)
    lbexp -= untradeable_exposures
    ubexp -= untradeable_exposures

    sumnot = abs(g_positions).sum()
    sumnot = max(sumnot, max_sumnot)
    sumnot = min(sumnot, max_sumnot * hard_limit)
    # offset sumnot by the untradeable positions
    sumnot -= abs(u_positions).sum()

    lb = np.maximum(t_lbound, -max_posnot * max_sumnot)
    ub = np.minimum(t_ubound, max_posnot * max_sumnot)

    # exposure constraints
    Ac = np.zeros((2 * num_factors, t_num_secs))
    bc = np.zeros(2 * num_factors)
    for i in range(num_factors):
        for j in range(t_num_secs):
            Ac[i, j] = t_factors[i, j]
            Ac[num_factors + i, j] = -t_factors[i, j]
        bc[i] = ubexp[i]
        bc[num_factors + i] = -lbexp[i]

    untradeable_mu = np.dot(u_mu, u_positions)
    untradeable_rvar = np.dot(u_positions * u_rvar, u_positions)
    untradeable_loadings = untradeable_exposures
    untradeable_info = (untradeable_mu, untradeable_rvar, untradeable_loadings)

    # Validate sumnot constraint
    if sumnot <= 0:
        raise ValueError("Capital constraint sumnot must be positive, got: {}".format(sumnot))

    # Objective function wrapper for scipy (negated for minimization)
    def obj_scipy(x):
        return -objective(x, kappa, slip_gamma, slip_nu, t_positions, t_mu, t_rvar,
                         t_factors, g_fcov, t_advp, t_advpt, t_vol, t_mktcap,
                         t_borrowRate, t_price, execFee, untradeable_info)

    def grad_scipy(x):
        return -objective_grad(x, kappa, slip_gamma, slip_nu, t_positions, t_mu, t_rvar,
                              t_factors, g_fcov, t_advp, t_advpt, t_vol, t_mktcap,
                              t_borrowRate, t_price, execFee, untradeable_info)

    try:
        problem_setup = setupProblem_scipy(t_positions, t_mu, t_rvar, t_factors, g_fcov,
                                          t_advp, t_advpt, t_vol, t_mktcap, t_borrowRate,
                                          t_price, lb, ub, Ac, bc, lbexp, ubexp,
                                          untradeable_info, sumnot, zero_start)

        # Solve with scipy.optimize.minimize
        result = minimize(
            fun=obj_scipy,
            x0=problem_setup['x0'],
            method='trust-constr',
            jac=grad_scipy,
            bounds=problem_setup['bounds'],
            constraints=problem_setup['constraints'],
            options=problem_setup['options']
        )
    except Exception as e:
        raise ValueError("Optimization setup or solve failed: {}".format(str(e)))

    # Retry with current positions as starting point if failed and zero_start was used
    if not result.success and zero_start > 0:
        logging.warning("Optimization failed with zero_start, retrying with current positions")
        problem_setup = setupProblem_scipy(t_positions, t_mu, t_rvar, t_factors, g_fcov,
                                          t_advp, t_advpt, t_vol, t_mktcap, t_borrowRate,
                                          t_price, lb, ub, Ac, bc, lbexp, ubexp,
                                          untradeable_info, sumnot, 0)

        result = minimize(
            fun=obj_scipy,
            x0=problem_setup['x0'],
            method='trust-constr',
            jac=grad_scipy,
            bounds=problem_setup['bounds'],
            constraints=problem_setup['constraints'],
            options=problem_setup['options']
        )

    target = np.zeros(num_secs)
    g_params = [kappa, slip_gamma, slip_nu, g_positions, g_mu, g_rvar, g_factors, g_fcov, g_advp, g_advpt, g_vol,
                g_mktcap, g_borrowRate, g_price, execFee, (0.0, 0.0, np.zeros_like(untradeable_loadings))]

    if not result.success:
        logging.error("Optimization failed: {}".format(result.message))
        print(objective_detail(target, *g_params))
        raise Exception("Optimization failed: {}".format(result.message))

    logging.info("Optimization succeeded: {} iterations, final objective = {:.2f}".format(
        result.nit, -result.fun))

    # Extract optimized positions
    opt = result.x
    #    print("SEAN: " + str(result.x))
    #    print(str(result.fun))
    targetIndex = 0
    optIndex = 0
    tradeable = set(tradeable)
    while targetIndex < num_secs:
        if targetIndex in tradeable:
            target[targetIndex] = opt[optIndex]
            optIndex += 1
        else:
            target[targetIndex] = g_positions[targetIndex]
        targetIndex += 1

    dutil = np.zeros(len(target))
    dutil2 = np.zeros(len(target))
    dmu = np.zeros(len(target))
    dsrisk = np.zeros(len(target))
    dfrisk = np.zeros(len(target))
    eslip = np.zeros(len(target))
    costs = np.zeros(len(target))
    for ii in range(len(target)):
        targetwo = target.copy()
        targetwo[ii] = g_positions[ii]

        dutil_o1 = objective_detail(target, *g_params)
        dutil_o2 = objective_detail(targetwo, *g_params)
        dutil[ii] = dutil_o1[0] - dutil_o2[0]
        dmu[ii] = dutil_o1[1] - dutil_o2[1]
        dsrisk[ii] = dutil_o1[2] - dutil_o2[2]
        dfrisk[ii] = dutil_o1[3] - dutil_o2[3]
        eslip[ii] = dutil_o1[4] - dutil_o2[4]
        costs[ii] = dutil_o1[5] - dutil_o2[5]

        trade = target[ii] - g_positions[ii]

        positions2 = g_positions.copy()
        positions2[ii] = target[ii]
        dutil2[ii] = objective(positions2, *g_params) - objective(g_positions, *g_params)

    (vol, price) = printinfo(target, *g_params) # to change

    return (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2, vol, price)


def init():
    """Initialize global optimization data arrays.

    Allocates zero-initialized numpy arrays for all optimization inputs.
    Must be called once after setting num_secs and num_factors, and before
    the first call to optimize().

    The caller should populate these arrays with actual data before calling
    optimize():
        g_positions: Current portfolio positions
        g_lbound/g_ubound: Position limits per security
        g_mu: Expected returns (alpha signals)
        g_rvar: Residual variance (specific risk)
        g_advp/g_advpt: Average daily volume
        g_vol: Volatility
        g_mktcap: Market capitalization
        g_borrowRate: Borrow costs
        g_price: Stock prices
        g_factors: Factor loadings matrix
        g_fcov: Factor covariance matrix

    Global Variables Modified:
        All g_* arrays (positions, mu, rvar, factors, etc.)

    Depends On:
        num_secs: Number of securities (must be set before calling)
        num_factors: Number of risk factors (must be set before calling)

    Example:
        import opt
        opt.num_secs = 1400
        opt.num_factors = 13
        opt.init()
        opt.g_mu[:] = alpha_signals
        # ... populate other arrays ...
        target, dutil, ... = opt.optimize()
    """
    global num_secs, num_factors, g_positions, g_lbound, g_ubound, g_mu, g_rvar, g_advp, g_advpt, g_vol, g_mktcap, g_borrowRate, g_price, g_factors, g_fcov

    g_positions = np.zeros(num_secs)
    g_lbound = np.zeros(num_secs)
    g_ubound = np.zeros(num_secs)
    g_mu = np.zeros(num_secs)
    g_rvar = np.zeros(num_secs)
    g_advp = np.zeros(num_secs)
    g_advpt = np.zeros(num_secs)
    g_vol = np.zeros(num_secs)
    g_mktcap = np.zeros(num_secs)
    g_borrowRate = np.zeros(num_secs)
    g_price = np.zeros(num_secs)
    g_factors = np.zeros((num_factors, num_secs))
    g_fcov = np.zeros((num_factors, num_factors))
    return


def getUntradeable():
    """Partition securities into tradeable and untradeable sets.

    Securities are marked untradeable when their position bounds are too tight
    (within $10 of each other), indicating they cannot be meaningfully adjusted.
    This typically occurs when:
    - Security is restricted (e.g., no short locates available)
    - Position is locked for other reasons
    - Bounds are set equal to force a specific position

    Returns:
        Tuple of (tradeable, untradeable):
            tradeable (list): Indices of securities that can be optimized
            untradeable (list): Indices of securities held at fixed positions

    Uses Global Variables:
        g_lbound: Lower bounds on positions (dollars)
        g_ubound: Upper bounds on positions (dollars)
        num_secs: Total number of securities

    The optimizer only optimizes over tradeable securities, treating
    untradeable positions as fixed contributions to portfolio risk and return.
    Untradeable positions still contribute to:
    - Alpha (via untradeable_mu)
    - Specific risk (via untradeable_rvar)
    - Factor risk (via untradeable_loadings)
    - Capital constraint (offset from max_sumnot)
    - Exposure constraints (offset from lbexp/ubexp)

    Notes:
        - Threshold of $10 is somewhat arbitrary but works in practice
        - Too tight threshold may exclude illiquid securities
        - Too loose threshold may include effectively fixed positions
    """
    untradeable = []
    tradeable = []
    for ii in range(num_secs):
        if abs(g_lbound[ii] - g_ubound[ii]) < 10:
            untradeable.append(ii)
        else:
            tradeable.append(ii)

    return tradeable, untradeable
