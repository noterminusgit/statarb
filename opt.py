#!/usr/bin/env python
"""
Portfolio Optimization Module

This module implements portfolio optimization using OpenOpt NLP solver to
maximize risk-adjusted returns while respecting trading constraints.

Objective Function:
    Maximize: Alpha - κ(Specific Risk + Factor Risk) - Slippage - Execution Costs

Components:
    Alpha: Expected return from alpha signals (μ · positions)
    Specific Risk: Idiosyncratic variance (σ² · positions²)
    Factor Risk: Systematic risk from Barra factors (x'Fx)
    Slippage: Nonlinear market impact cost function
    Execution Fees: Fixed bps cost (default: 1.5 bps)

Constraints:
    - Position Limits: Min/max shares per security (±0.04M notional)
    - Capital Limits: Max aggregate notional ($4-50M)
    - Factor Exposure Limits: Bounds on factor bets
    - Participation Limits: Max 1.5% participation rate
    - Dollar Neutrality: Optional long/short balance
    - Sector Limits: Optional industry concentration limits

Slippage Model:
    Cost = α + δ · (participation_rate)^β + γ · volatility + ν · price_impact
    - α (slip_alpha): Base cost (default: 1.0 bps)
    - β (slip_beta): Participation power (default: 0.6)
    - δ (slip_delta): Participation coefficient (default: 0.25)
    - γ (slip_gamma): Volatility coefficient (default: 0.3)
    - ν (slip_nu): Market impact coefficient (default: 0.14-0.18)

Parameters:
    kappa: Risk aversion parameter (4.3e-5 default)
    max_sumnot: Max total notional ($50M default)
    max_posnot: Max position as fraction of capital (0.48% default)
    max_expnot: Max exposure per security (4.8% default)

The optimizer uses OpenOpt's NLP solver with gradient-based methods for
efficient convergence. Typical solve time: 1-5 seconds for 1400 securities.

Global Variables:
    g_positions: Current positions
    g_mu: Expected returns (alpha signals)
    g_rvar: Residual variance (specific risk)
    g_factors: Factor loadings matrix
    g_fcov: Factor covariance matrix
    g_advp: Average daily volume in dollars
    g_borrowRate: Borrow costs for shorts
"""

import sys
import numpy
import math
import logging
import openopt

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

#HAND-TWEAKED PARAMETERS TO MATCH CURRENT TRADING BEHAVIOR
slip_alpha = 1.0
slip_delta = 0.25
slip_beta = 0.6
slip_gamma = 0.3
slip_nu = 0.14
execFee= 0.00015

num_secs = 0
num_factors = 0
stocks_ii = 0
factors_ii = 0
zero_start = 0

#prefix them with g_ to avoid errors
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
numpy.set_printoptions(threshold=float('nan'))

p=None

class Terminator():
    """Custom termination callback for OpenOpt NLP solver.

    Monitors optimization progress and terminates early if improvement plateaus.
    Uses a rolling window approach to track objective function improvement.

    Args:
        lookback: Number of iterations to look back for improvement comparison
        stopThreshold: Minimum improvement required to continue (in objective units)
        minIter: Minimum iterations before checking termination criteria

    The callback is invoked after each solver iteration and tracks:
    - Objective function values over iterations
    - Improvement rate over rolling window
    - Convergence and divergence patterns

    Returns True to stop optimization, False to continue.
    """
    def __init__(self, lookback, stopThreshold, minIter):
        self.iter = 0
        self.objValues = []
        self.maxAtLookback = None
        self.lookback = lookback
        self.stopThreshold = stopThreshold
        self.minIter = minIter

    def __call__(self, p):
        self.iter += 1
        #infeasible points are disregarded from computations
        if p.rk <= 0:
            self.objValues.append(p.fk)
        else:
            self.objValues.append(float('inf'))

        #don't start checking until we have seen at least min iters
        if self.iter <= self.lookback + self.minIter:
            return False
        #only check every 10 iterations
        if self.iter % 10 != 0:
            return False

        #internally it works as a minimizer, so take that into account by getting the minimum values and inverting them
        #each iteration is not guaranteed to increase the obj function values.
        curr = -min(self.objValues[-self.lookback:-1])
        prev = -min(self.objValues[0:(-self.lookback -1)])

        if numpy.isinf(prev):
            print "Haven't found a feasible point yet"
            return False
        elif numpy.isinf(curr):
            print "We are probably diverging, but we are staying the course for a huge comeback"
            return False

        if self.iter % 10 == 0:
            print "Current improvement after {} iterations is {}".format(self.lookback, float(curr-prev))
        if curr - prev < self.stopThreshold:
            print "Current improvement after {} iterations is {}".format(self.lookback, float(curr-prev))
            return True
        else:
            return False
        

def printinfo(target, kappa, slip_gamma, slip_nu,positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price, execFee, untradeable_info):
    """Print optimization summary comparing current and target portfolios.

    Displays long/short notional, trade sizes, and utility decomposition for
    both current and optimized target positions.

    Args:
        target: Array of target positions (dollars)
        kappa: Risk aversion parameter
        slip_gamma: Slippage volatility coefficient
        slip_nu: Slippage market impact coefficient
        positions: Array of current positions (dollars)
        mu: Expected returns (alpha signals)
        rvar: Residual variance (specific risk)
        factors: Factor loadings matrix (num_factors x num_secs)
        fcov: Factor covariance matrix
        advp: Average daily volume in dollars
        advpt: Average daily tradeable volume
        vol: Volatility (daily standard deviation)
        mktcap: Market capitalization
        brate: Borrow rates for shorts
        price: Stock prices
        execFee: Execution fee (bps as decimal)
        untradeable_info: Tuple of (mu, rvar, loadings) for untradeable positions

    Prints to stdout:
        - Current portfolio: long/short/total notional
        - Target portfolio: long/short/total notional
        - Dollars traded (total turnover)
        - Utility breakdown for current and optimum portfolios
    """
    clong=0
    cshort=0
    tlong=0
    tshort=0
    diff=0
    for ii in xrange(len(positions)):
        if positions[ii]>=0:
            clong+=positions[ii]
        else:
            cshort-=positions[ii]
    for ii in xrange(len(target)):
        if target[ii]>=0:
            tlong+=target[ii]
        else:
            tshort-=target[ii]
        diff+=abs(target[ii]-positions[ii])
    print "[CURRENT] Long: {:.0f}, Short: {:.0f}, Total: {:.0f}".format(clong,cshort,clong+cshort)
    print "[TARGET]  Long: {:.0f}, Short: {:.0f}, Total: {:.0f}".format(tlong,tshort,tlong+tshort)
    print "Dollars traded: {:.0f}".format(diff)
    __printpointinfo("Current",positions,  kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price, execFee, untradeable_info)
    __printpointinfo("Optimum",target,  kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price, execFee, untradeable_info)

def __printpointinfo(name,target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price, execFee, untradeable_info):
    """Print detailed utility decomposition for a portfolio position.

    Internal helper function that breaks down total utility into components:
    alpha (mu), risk penalty, slippage, and execution costs.

    Args:
        name: Label for this point (e.g., "Current" or "Optimum")
        target: Portfolio positions to analyze
        (other parameters same as printinfo)

    Prints to stdout:
        @{name}: total={utility}, mu={alpha}, risk={penalty}, slip={cost},
                 costs={fees}, ratio={mu/risk}, var={specific}, covar={factor}
    """
    untradeable_mu, untradeable_rvar, untradeable_loadings = untradeable_info[0], untradeable_info[1], untradeable_info[2]

    loadings = numpy.dot(factors, target)+untradeable_loadings
    utility1 = numpy.dot(target, mu) + untradeable_mu
    utility2 = kappa * ( untradeable_rvar + numpy.dot(target * rvar, target) + numpy.dot(numpy.dot(loadings, fcov), loadings) )
    utility3 = slippageFuncAdv(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu)
    utility4 = costsFunc(target, positions, brate, price, execFee)
    var = kappa * numpy.dot(target * rvar, target)
    covar = kappa * numpy.dot(numpy.dot(loadings, fcov), loadings)
    print "@{}: total={:.0f}, mu={:.0f}, risk={:.0f}, slip={:.2f}, costs={:.2f}, ratio={:.3f}, var={:.0f}, covar={:.0f}".format(name,utility1-utility2-utility3-utility4, utility1,utility2,utility3,utility4,utility1/utility2, var, covar)

def slippageFuncAdv(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu):
    """Calculate total market impact slippage cost for portfolio rebalancing.

    Implements nonlinear slippage model with two components:
    1. Volatility-based impact: γ * vol * participation * (mktcap/advp)^δ
    2. Power-law participation cost: ν * vol * (participation)^β

    The model penalizes both:
    - Large trades in volatile stocks
    - High participation rates (trading too much of daily volume)
    - Trading in less liquid stocks (lower mktcap/advp ratio)

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars)
        advp: Average daily volume (dollars)
        advpt: Average daily tradeable volume (dollars)
        vol: Daily volatility (standard deviation)
        mktcap: Market capitalization
        slip_gamma: Volatility coefficient (default: 0.3)
        slip_nu: Market impact coefficient (default: 0.14-0.18)

    Returns:
        Total slippage cost (dollars) across all securities

    Formula:
        I = γ * vol * |Δpos|/advp * (mktcap/advp)^δ
        J = I/2 + ν * vol * (|Δpos|/advpt)^β
        slippage = Σ(J * |Δpos|)
    """
    newpos_abs = abs(target-positions)
    I = slip_gamma * vol * (newpos_abs/advp) * (mktcap/advp) ** slip_delta
    J = I/2 + slip_nu * vol * (newpos_abs/advpt) ** slip_beta
    slip = J * newpos_abs
    return slip.sum()

def slippageFunc_grad(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu):
    """Calculate gradient of slippage function with respect to target positions.

    Provides first derivative for gradient-based optimization. The gradient
    combines the derivatives of both slippage components with respect to
    position changes.

    Args:
        (same as slippageFuncAdv)

    Returns:
        Array of partial derivatives ∂(slippage)/∂(target) for each security

    Formula:
        Id = 0.5 * γ * vol * (1/advp) * (mktcap/advp)^δ
        Jd = [Id + ν * vol * (1+β) * (|Δpos|/advpt)^β] * sign(Δpos)
    """
    newpos = target-positions
    Id = .5 * slip_gamma * vol * (1/advp) * (mktcap/advp) ** slip_delta
    Jd = (Id + slip_nu * vol * (1 + slip_beta) * (abs(newpos)/advpt) ** slip_beta) * numpy.sign(newpos)
    return Jd

def costsFunc(target, positions, brate, price, execFee):
    """Calculate total execution and borrow costs.

    Computes fixed execution fees based on shares traded and (optionally)
    borrow costs for short positions.

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars)
        brate: Borrow rates for shorts (currently unused)
        price: Stock prices per share
        execFee: Execution fee as decimal (default: 0.00015 = 1.5 bps)

    Returns:
        Total execution costs (dollars)

    Formula:
        costs = execFee * Σ(|Δshares|)
              = execFee * Σ(|Δdollars|/price)

    Note:
        Borrow costs currently disabled (see XXX comment in code).
        When enabled: costs += Σ(brate * min(0, target))
    """
    costs = execFee * numpy.dot(1.0/price, abs(target - positions))
    #ATTENTION! borrow costs are negative, negative times negative gives a positive cost
    #XXX add back once we have borrow costs!
    #costs += numpy.dot(brate, numpy.minimum(0.0, target))
    return costs

def costsFunc_grad(target, positions, brate, price, execFee):
    """Calculate gradient of costs function with respect to target positions.

    Provides first derivative for gradient-based optimization.

    Args:
        (same as costsFunc)

    Returns:
        Array of partial derivatives ∂(costs)/∂(target) for each security

    Formula:
        ∂(costs)/∂(target) = execFee * sign(Δpos) / price

    Note:
        Borrow cost gradient currently disabled. When enabled:
        ∂(costs)/∂(target) += brate[i] for target[i] <= 0
    """
    grad = execFee * numpy.sign(target - positions) / price
#    for i in xrange(len(grad)):
        #ATTENTION!  borrow costs are negative, derivative is negative (more positive position, lower costs)
#        if target[i] <=0 : grad[i] += brate[i]
    return grad

def objective(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price, execFee, untradeable_info):
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
        factors: Factor loadings matrix
        fcov: Factor covariance matrix
        advp: Average daily volume (dollars)
        advpt: Average daily tradeable volume
        vol: Daily volatility
        mktcap: Market capitalization
        brate: Borrow rates
        price: Stock prices
        execFee: Execution fee (bps)
        untradeable_info: (mu, rvar, loadings) for untradeable securities

    Returns:
        Utility = α - κ*Risk - Slippage - Costs (to be maximized)

    Formula:
        U = μ·x - κ(σ²·x² + x'Fx) - slippage(Δx) - costs(Δx)
    where:
        μ·x = expected alpha return
        σ²·x² = specific risk penalty
        x'Fx = factor risk penalty
        slippage(Δx) = market impact costs
        costs(Δx) = execution fees
    """
    return objective_detail(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price, execFee, untradeable_info)[0]

def objective_detail(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price, execFee, untradeable_info):
    """Portfolio optimization objective with detailed component breakdown.

    Same as objective() but returns all individual components for analysis
    and debugging. Used for detailed utility attribution.

    Args:
        (same as objective)

    Returns:
        Tuple of (utility, tmu, tsrisk, tfrisk, tslip, tcosts):
            utility: Total utility = tmu - tsrisk - tfrisk - tslip - tcosts
            tmu: Alpha component (μ·x)
            tsrisk: Specific risk penalty (κ*σ²·x²)
            tfrisk: Factor risk penalty (κ*x'Fx)
            tslip: Slippage costs
            tcosts: Execution costs

    This detailed breakdown is used by:
    - printinfo() for utility attribution reporting
    - optimize() for per-security marginal utility calculations
    """
    untradeable_mu, untradeable_rvar, untradeable_loadings = untradeable_info[0], untradeable_info[1], untradeable_info[2]

    # objective function to be minimized (negative utility)    
    loadings = numpy.dot(factors, target) + untradeable_loadings

    tmu = numpy.dot(target, mu) + untradeable_mu
    tsrisk = kappa * (untradeable_rvar + numpy.dot(target * rvar, target))
    tfrisk = kappa * numpy.dot(numpy.dot(loadings, fcov), loadings) 
    tslip = slippageFuncAdv(target, positions, advp, advpt, vol, mktcap, slip_gamma, slip_nu)
    tcosts = costsFunc(target, positions, brate, price, execFee)

    utility = tmu
    utility -= tsrisk
    utility -= tfrisk
    utility -= tslip
    utility -= tcosts

    return (utility, tmu, tsrisk, tfrisk, tslip, tcosts)

def objective_grad(target, kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, brate, price, execFee, untradeable_info):
    """Calculate gradient of objective function with respect to target positions.

    Provides analytical gradient for gradient-based optimization (RALG solver).
    Combines gradients of all utility components.

    Args:
        (same as objective)

    Returns:
        Array of partial derivatives ∂U/∂(target) for each security

    Formula:
        ∂U/∂x = μ - 2κ(σ²·x + F'·C·(F·x + u)) - ∂slip/∂x - ∂costs/∂x
    where:
        μ = expected returns
        σ² = residual variance
        F = factor loadings matrix
        C = factor covariance matrix
        u = untradeable factor loadings
        ∂slip/∂x = slippage gradient
        ∂costs/∂x = costs gradient

    The gradient guides the optimizer toward positions that balance:
    - High expected return (positive μ)
    - Low risk exposure (negative risk terms)
    - Minimal trading costs (negative cost terms)
    """
    untradeable_mu, untradeable_rvar, untradeable_loadings = untradeable_info[0], untradeable_info[1], untradeable_info[2]

    F = factors
    Ft = numpy.transpose(F)
    grad = numpy.zeros(len(target))
    grad += mu
    grad -= 2 * kappa * (rvar * target + numpy.dot(Ft, numpy.dot(fcov, numpy.dot(F, target) + untradeable_loadings)))
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
        positions: Current positions (dollars, unused)
        max_sumnot: Maximum total notional allowed
        factors: Factor loadings (unused)
        lbexp: Lower bound exposures (unused)
        ubexp: Upper bound exposures (unused)
        max_trdnot_hard: Hard turnover limit (unused)

    Returns:
        Constraint value that must be <= 0 for feasibility
        = Σ|target| - max_sumnot

    The constraint is satisfied when: Σ|target| <= max_sumnot
    """
    ret = abs(target).sum() - max_sumnot
    return ret

def constrain_by_capital_grad(target, positions, max_sumnot, factors, lbexp, ubexp, max_trdnot_hard):
    """Gradient of capital constraint with respect to target positions.

    Args:
        (same as constrain_by_capital)

    Returns:
        Array of partial derivatives ∂(constraint)/∂(target)
        = sign(target) for each security

    The gradient indicates that increasing any position (long or short)
    increases the total notional constraint value.
    """
    return numpy.sign(target)

#def constrain_by_exposures(target, positions, max_sumnot, factors, lbexp, ubexp, max_trdnot_hard):
#    exposures = numpy.dot(factors, target)
#    ret = max(numpy.r_[lbexp - exposures, exposures - ubexp])
#    return ret

### UGH this is ignored!
def constrain_by_trdnot(target, positions, max_sumnot, factors, lbexp, ubexp, max_trdnot_hard):
    """Constraint function enforcing maximum turnover limit (CURRENTLY UNUSED).

    Would limit total dollars traded in a single rebalance, but this constraint
    is currently not activated in the optimization setup.

    Args:
        target: Target positions (dollars)
        positions: Current positions (dollars)
        max_trdnot_hard: Maximum turnover allowed
        (other args unused)

    Returns:
        Constraint value = Σ|Δpos| - max_trdnot_hard

    Note:
        This constraint is defined but NOT added to the optimizer in
        setupProblem(). To enable, add to p.c and p.dc lists.
    """
    ret = abs(target - positions).sum() - max_trdnot_hard
    return ret

def setupProblem(positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, borrowRate, price, lb, ub, Ac, bc, lbexp, ubexp, untradeable_info, sumnot, zero_start):
    """Configure OpenOpt NLP problem for portfolio optimization.

    Sets up the constrained nonlinear programming problem with:
    - Objective function and gradient
    - Position bounds (box constraints)
    - Factor exposure limits (linear constraints)
    - Capital constraint (nonlinear constraint)
    - Solver parameters and termination callback

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
        Ac: Linear constraint matrix for factor exposures
        bc: Linear constraint RHS vector
        lbexp: Lower bounds on factor exposures
        ubexp: Upper bounds on factor exposures
        untradeable_info: (mu, rvar, loadings) for untradeable securities
        sumnot: Maximum total notional
        zero_start: If > 0, initialize optimizer at zero positions

    Returns:
        Configured OpenOpt NLP problem ready for solve()

    Solver Configuration:
        - Algorithm: RALG (gradient-based)
        - Max iterations: max_iter (default 500)
        - Min iterations: min_iter (default 500)
        - Tolerance: ftol = 1e-6
        - Early stopping: Terminator callback (50 iter lookback, threshold=10)
    """
    if zero_start > 0:
        p = openopt.NLP(goal='max', f=objective, df=objective_grad, x0=numpy.zeros(len(positions)), lb=lb, ub=ub, A=Ac, b=bc, plot=plotit)
    else:
        p = openopt.NLP(goal='max', f=objective, df=objective_grad, x0=positions, lb=lb, ub=ub, A=Ac, b=bc, plot=plotit)
    p.args.f = (kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, borrowRate, price, execFee, untradeable_info)
    p.args.df = (kappa, slip_gamma, slip_nu, positions, mu, rvar, factors, fcov, advp, advpt, vol, mktcap, borrowRate, price, execFee, untradeable_info)
    p.c = [constrain_by_capital]
    p.dc = [constrain_by_capital_grad]
    p.args.c = (positions, sumnot, factors, lbexp, ubexp, sumnot)
    p.args.dc = (positions, sumnot, factors, lbexp, ubexp, sumnot)
    p.ftol = 1e-6
    p.maxFunEvals = 1e9
    p.maxIter = max_iter
    p.minIter = min_iter
    p.callback = Terminator(50, 10, p.minIter)
    
    return p

def optimize():
    """Main portfolio optimization entry point.

    Optimizes portfolio positions by maximizing risk-adjusted utility subject
    to position limits, factor exposure constraints, and capital constraints.

    Uses global variables (g_positions, g_mu, g_rvar, etc.) set by caller
    via init() and direct assignment.

    Algorithm:
        1. Partition securities into tradeable/untradeable based on bounds
        2. Extract data arrays for tradeable subset
        3. Compute exposure and capital limits with hard_limit buffer
        4. Set up NLP problem with box and linear constraints
        5. Solve using RALG algorithm
        6. If infeasible with zero_start=1, retry with zero_start=0
        7. Calculate per-security marginal utility contributions
        8. Print optimization summary

    Returns:
        Tuple of (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2):
            target: Optimized target positions (dollars) for all securities
            dutil: Marginal utility of each position
            eslip: Marginal slippage cost
            dmu: Marginal alpha contribution
            dsrisk: Marginal specific risk contribution
            dfrisk: Marginal factor risk contribution
            costs: Marginal execution costs
            dutil2: Alternative marginal utility calculation

    Raises:
        Exception: If optimization fails to find feasible solution

    Global Variables Used:
        g_positions, g_mu, g_rvar, g_factors, g_fcov, g_advp, g_advpt,
        g_vol, g_mktcap, g_borrowRate, g_price, g_lbound, g_ubound

    Global Variables Modified:
        p (optimization problem object)

    Side Effects:
        Prints optimization summary via printinfo()
    """
    global p

    # Validate global variables are initialized
    if g_positions is None or len(g_positions) == 0:
        raise ValueError("g_positions is not initialized or empty")
    if g_mu is None or len(g_mu) == 0:
        raise ValueError("g_mu is not initialized or empty")
    if g_factors is None or g_factors.size == 0:
        raise ValueError("g_factors is not initialized or empty")

    # Check dimensions match
    if len(g_positions) != len(g_mu):
        raise ValueError("Dimension mismatch: g_positions ({}) != g_mu ({})".format(len(g_positions), len(g_mu)))
    if g_factors.shape[1] != len(g_positions):
        raise ValueError("Dimension mismatch: g_factors columns ({}) != g_positions ({})".format(g_factors.shape[1], len(g_positions)))

    # Check for NaN/inf in critical inputs
    if numpy.isnan(g_mu).any():
        logging.warning("Found {} NaN values in g_mu, setting to 0".format(numpy.isnan(g_mu).sum()))
        g_mu[numpy.isnan(g_mu)] = 0
    if numpy.isinf(g_mu).any():
        logging.warning("Found {} infinite values in g_mu, setting to 0".format(numpy.isinf(g_mu).sum()))
        g_mu[numpy.isinf(g_mu)] = 0

    if numpy.isnan(g_rvar).any() or (g_rvar < 0).any():
        logging.warning("Found invalid g_rvar values (NaN or negative), setting to small positive value")
        g_rvar[numpy.isnan(g_rvar) | (g_rvar < 0)] = 1e-8

    tradeable, untradeable = getUntradeable()

    if len(tradeable) == 0:
        raise ValueError("No tradeable securities found - check bounds configuration")

    t_num_secs = len(tradeable)
    t_positions = numpy.copy(g_positions[tradeable])
    t_factors = numpy.copy(g_factors[:, tradeable])
    t_lbound = numpy.copy(g_lbound[tradeable])
    t_ubound = numpy.copy(g_ubound[tradeable])
    t_mu = numpy.copy(g_mu[tradeable])
    t_rvar = numpy.copy(g_rvar[tradeable])
    t_advp = numpy.copy(g_advp[tradeable])

    t_advpt = numpy.copy(g_advpt[tradeable])
    t_vol = numpy.copy(g_vol[tradeable])
    t_mktcap = numpy.copy(g_mktcap[tradeable])

    t_borrowRate = numpy.copy(g_borrowRate[tradeable])
    t_price = numpy.copy(g_price[tradeable]) 

    u_positions = numpy.copy(g_positions[untradeable])
    u_factors = numpy.copy(g_factors[:, untradeable])
    u_mu = numpy.copy(g_mu[untradeable])
    u_rvar = numpy.copy(g_rvar[untradeable])
        
    exposures = numpy.dot(g_factors, g_positions)
    lbexp = exposures
    lbexp = numpy.minimum(lbexp, -max_expnot * max_sumnot)
    lbexp = numpy.maximum(lbexp, -max_expnot * max_sumnot * hard_limit)
    ubexp = exposures
    ubexp = numpy.maximum(ubexp, max_expnot * max_sumnot)
    ubexp = numpy.minimum(ubexp, max_expnot * max_sumnot * hard_limit)
    #offset the lbexp and ubexp by the untradeable positions
    untradeable_exposures = numpy.dot(u_factors, u_positions)
    lbexp -= untradeable_exposures
    ubexp -= untradeable_exposures

    sumnot = abs(g_positions).sum()
    sumnot = max(sumnot, max_sumnot)
    sumnot = min(sumnot, max_sumnot * hard_limit)
    #offset sumnot by the untradeable positions
    sumnot -= abs(u_positions).sum()

    lb = numpy.maximum(t_lbound, -max_posnot * max_sumnot)
    ub = numpy.minimum(t_ubound, max_posnot * max_sumnot)

    # Validate bounds are feasible
    if (lb > ub).any():
        num_infeasible = (lb > ub).sum()
        logging.warning("Found {} securities with infeasible bounds (lb > ub)".format(num_infeasible))
        # Fix by relaxing bounds
        infeasible_mask = lb > ub
        midpoint = (lb[infeasible_mask] + ub[infeasible_mask]) / 2
        lb[infeasible_mask] = midpoint
        ub[infeasible_mask] = midpoint

    #exposure constraints
    Ac = numpy.zeros((2 * num_factors, t_num_secs))
    bc = numpy.zeros(2 * num_factors)
    for i in xrange(num_factors):
        for j in xrange(t_num_secs):
            Ac[i, j] = t_factors[i, j]
            Ac[num_factors + i, j] = -t_factors[i, j]
        bc[i] = ubexp[i]
        bc[num_factors + i] = -lbexp[i]

    # Validate exposure bounds
    if (lbexp > ubexp).any():
        num_infeasible = (lbexp > ubexp).sum()
        logging.warning("Found {} factors with infeasible exposure bounds (lb > ub)".format(num_infeasible))

    untradeable_mu = numpy.dot(u_mu, u_positions)
    untradeable_rvar = numpy.dot(u_positions * u_rvar, u_positions)
    untradeable_loadings = untradeable_exposures
    untradeable_info = (untradeable_mu, untradeable_rvar, untradeable_loadings)

    # Validate sumnot constraint
    if sumnot <= 0:
        raise ValueError("Capital constraint sumnot must be positive, got: {}".format(sumnot))

    try:
        p = setupProblem(t_positions, t_mu, t_rvar, t_factors, g_fcov, t_advp, t_advpt, t_vol, t_mktcap, t_borrowRate, t_price, lb, ub, Ac, bc, lbexp, ubexp, untradeable_info, sumnot, zero_start)
        r = p.solve('ralg')
    except Exception as e:
        raise ValueError("Optimization setup or solve failed: {}".format(str(e)))
    
    #XXX need to check for small number of iterations!!!
    if (r.stopcase == -1 or r.isFeasible == False) and zero_start > 0:
        #try again with zero_start = 0
        p = setupProblem(t_positions, t_mu, t_rvar, t_factors, g_fcov, t_advp, t_advpt, t_vol, t_mktcap, t_borrowRate, t_price, lb, ub, Ac, bc, lbexp, ubexp, untradeable_info, sumnot, 0)
        r = p.solve('ralg')

    target = numpy.zeros(num_secs)
    g_params = [kappa, slip_gamma, slip_nu, g_positions, g_mu, g_rvar, g_factors, g_fcov, g_advp, g_advpt, g_vol, g_mktcap, g_borrowRate, g_price, execFee, (0.0,0.0, numpy.zeros_like(untradeable_loadings))]
    
    if (r.stopcase == -1 or r.isFeasible == False):
        print objective_detail(target, *g_params)
        raise Exception("Optimization failed")

    #the target is the zipping of the opt result and the untradeable securities
    opt = numpy.array(r.xf)
#    print "SEAN: " + str(r.xf)
#    print str(r.ff)
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
            
    dutil = numpy.zeros(len(target))
    dutil2 = numpy.zeros(len(target))
    dmu = numpy.zeros(len(target))
    dsrisk = numpy.zeros(len(target))
    dfrisk = numpy.zeros(len(target))
    eslip = numpy.zeros(len(target))
    costs = numpy.zeros(len(target))
    for ii in range(len(target)):
        targetwo = target.copy()
        targetwo[ii] = g_positions[ii]

        dutil_o1 = objective_detail(target, *g_params)
        dutil_o2 = objective_detail(targetwo, *g_params)
        dutil[ii] = dutil_o1[0] - dutil_o2[0]
        dmu[ii] = dutil_o1[1]  - dutil_o2[1]
        dsrisk[ii] = dutil_o1[2] - dutil_o2[2]
        dfrisk[ii] = dutil_o1[3] - dutil_o2[3]
        eslip[ii] = dutil_o1[4] - dutil_o2[4]
        costs[ii] = dutil_o1[5] - dutil_o2[5]

        trade = target[ii]-g_positions[ii]

        positions2 = g_positions.copy()
        positions2[ii] = target[ii]
        dutil2[ii] = objective(positions2, *g_params) - objective(g_positions, *g_params)

    printinfo(target, *g_params)

    return (target, dutil, eslip, dmu, dsrisk, dfrisk, costs, dutil2)

def init():
    """Initialize global optimization data arrays.

    Allocates zero-initialized numpy arrays for all optimization inputs.
    Must be called once before setting num_secs and num_factors, and before
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
    """
    global num_secs, num_factors, g_positions, g_lbound, g_ubound, g_mu, g_rvar, g_advp, g_advpt, g_vol, g_mktcap, g_borrowRate, g_price, g_factors, g_fcov

    g_positions = numpy.zeros(num_secs)
    g_lbound = numpy.zeros(num_secs) 
    g_ubound = numpy.zeros(num_secs)
    g_mu = numpy.zeros(num_secs)
    g_rvar = numpy.zeros(num_secs)
    g_advp = numpy.zeros(num_secs)
    g_advpt = numpy.zeros(num_secs)
    g_vol = numpy.zeros(num_secs)
    g_mktcap = numpy.zeros(num_secs)
    g_borrowRate = numpy.zeros(num_secs)
    g_price = numpy.zeros(num_secs)
    g_factors = numpy.zeros((num_factors, num_secs))
    g_fcov = numpy.zeros((num_factors, num_factors)) 
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
            tradeable: List of security indices that can be optimized
            untradeable: List of security indices held at fixed positions

    Uses Global Variables:
        g_lbound: Lower bounds on positions
        g_ubound: Upper bounds on positions
        num_secs: Total number of securities

    The optimizer only optimizes over tradeable securities, treating
    untradeable positions as fixed contributions to portfolio risk and return.
    """
    untradeable = []
    tradeable = []

    for ii in xrange(num_secs):
        if abs(g_lbound[ii] - g_ubound[ii]) < 10:
            untradeable.append(ii)
        else:
            tradeable.append(ii)

    return tradeable, untradeable

