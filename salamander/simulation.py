"""Portfolio Rebalancing Simulation with Monte Carlo Analysis

This module implements a Monte Carlo simulation framework for analyzing transaction
costs in rebalanced portfolios. It models a two-security portfolio that periodically
rebalances to maintain target weights, tracking the impact of transaction costs on
returns under different market conditions.

The simulation uses geometric Brownian motion (GBM) to model correlated asset price
movements and analyzes how various parameters affect rebalancing frequency and costs:
- Asset volatilities and correlation
- Expected returns
- Rebalancing threshold
- Transaction cost per trade

Key capabilities:
- Monte Carlo price path generation with correlated Brownian motion
- Threshold-based rebalancing logic
- Transaction cost impact analysis
- Sensitivity analysis for key parameters (correlation, volatility, threshold)
- Convergence testing for Monte Carlo estimates
- Visualization of price paths and parameter relationships

Usage:
    # Run basic simulation with default parameters
    python simulation.py --simulate=True --paths=100 --periods=252

    # Analyze correlation impact on transaction costs
    python simulation.py --solveCorr=True --paths=500

    # Test convergence of transaction cost estimates
    python simulation.py --convergence_test=True --paths=1000 --step=50

Command-line Arguments:
    --sec1vol: Annualized volatility of security 1 (default: 0.4)
    --sec2vol: Annualized volatility of security 2 (default: 0.3)
    --corr: Correlation between securities (default: 0.8)
    --sec1mean: Annualized return of security 1 (default: 0.05)
    --sec2mean: Annualized return of security 2 (default: 0.1)
    --paths: Number of Monte Carlo iterations (default: 500)
    --periods: Number of trading days to simulate (default: 252)
    --tcost: Transaction cost per dollar traded (default: 0.001, i.e. 0.1%)
    --rebalance_threshold: Minimum weight divergence to trigger rebalance (default: 0.01)
    --seed: Random seed for reproducibility (default: 5)
    --simulate: Run full simulation with visualization (default: False)
    --convergence_test: Test Monte Carlo convergence (default: False)
    --solveCorr: Analyze transaction cost vs correlation (default: False)
    --solveVol: Analyze transaction cost vs volatility (default: False)
    --solveReturn: Analyze transaction cost vs expected return (default: False)
    --solveThreshold: Analyze transaction cost vs rebalance threshold (default: False)

Example:
    # Analyze impact of rebalance threshold on transaction costs
    python simulation.py --solveThreshold=True --sec1vol=0.3 --sec2vol=0.2 --corr=0.6

    # Run simulation with low correlation and high volatility
    python simulation.py --simulate=True --corr=0.2 --sec1vol=0.5 --paths=200

Mathematical Model:
    Asset prices follow geometric Brownian motion:
        dS_i = μ_i S_i dt + σ_i S_i dW_i

    where W_1 and W_2 are correlated Brownian motions with correlation ρ.

    Portfolio rebalances when |w_i - w_target_i| > threshold for any asset,
    incurring transaction costs proportional to trade size.

    Transaction cost impact = (Pre-cost return) - (Post-cost return)

    where costs = sum of |trade dollars| × tcost rate

Notes:
    - This is a standalone educational/research tool, not used by other salamander modules
    - Designed for analyzing optimal rebalancing policies under transaction costs
    - Results are sensitive to random seed; use convergence testing for robust estimates
    - Generates PNG plots for visualization of results
"""

import pandas as pd
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import linear_model


class Portfolio(object):
    """Two-security portfolio with threshold-based rebalancing.

    Models a portfolio containing two securities with target 50/50 weights that
    rebalances when weights drift beyond a specified threshold. Tracks transaction
    costs and their impact on portfolio returns under correlated asset price movements.

    The portfolio uses geometric Brownian motion to simulate correlated price paths
    and maintains detailed records of rebalancing events, trade sizes, and costs.

    Attributes:
        numberOfStocks (int): Number of securities in portfolio (always 2)
        initprices (np.array): Initial prices for both securities [$5, $5]
        prices (np.array): Current prices for both securities
        initholdings (list): Initial share holdings [10, 10]
        holdings (list): Current share holdings
        inittotal (float): Initial portfolio value ($100)
        total (float): Current portfolio value
        initweightings (list): Target weights [0.5, 0.5]
        weightings (list): Current weights based on market values
        means (np.array): Annualized expected returns for both securities
        corr (float): Correlation coefficient between asset returns
        sec1vol (float): Annualized volatility of security 1
        sec2vol (float): Annualized volatility of security 2
        dailymeans (np.array): Daily expected returns (annualized / 252)
        dailysec1vol (float): Daily volatility of security 1 (annualized / sqrt(252))
        dailysec2vol (float): Daily volatility of security 2 (annualized / sqrt(252))
        dailycovmat (np.array): 2x2 daily covariance matrix for returns
        rebalance_threshold (float): Minimum weight divergence to trigger rebalance

    Example:
        >>> portfolio = Portfolio(sec1mean=0.05, sec2mean=0.10,
        ...                       sec1vol=0.40, sec2vol=0.30,
        ...                       corr=0.80, rebalance_threshold=0.01)
        >>> portfolio.Simulate(paths=100, tcost=0.001, periods=252, seed=42)
    """

    def __init__(self, sec1mean, sec2mean, sec1vol, sec2vol, corr, rebalance_threshold):
        """Initialize portfolio with asset parameters and rebalancing threshold.

        Args:
            sec1mean (float): Annualized expected return for security 1
            sec2mean (float): Annualized expected return for security 2
            sec1vol (float): Annualized volatility for security 1
            sec2vol (float): Annualized volatility for security 2
            corr (float): Correlation coefficient between asset returns (0 to 1)
            rebalance_threshold (float): Minimum weight divergence to trigger rebalance
                (e.g., 0.01 means rebalance when any weight drifts 1% from target)

        Notes:
            - Initial portfolio value is fixed at $100 with 50/50 weights
            - Each security starts at $5 with 10 shares
            - Converts annualized parameters to daily for simulation:
                * Daily returns = annual / 252
                * Daily volatility = annual / sqrt(252)
            - Constructs daily covariance matrix from volatilities and correlation
        """
        self.numberOfStocks = 2
        self.initprices = np.asarray([5, 5])
        self.prices = self.initprices
        self.initholdings = [10, 10]
        self.holdings = self.initholdings
        self.inittotal = 100
        self.total = self.inittotal
        self.initweightings = [.5, .5]
        self.weightings = self.initweightings
        # input
        self.means = np.asarray([sec1mean, sec2mean])
        self.corr = corr
        self.sec1vol = sec1vol
        self.sec2vol = sec2vol
        self.dailymeans = self.means / 252
        self.dailysec1vol = self.sec1vol / np.sqrt(252)
        self.dailysec2vol = self.sec2vol / np.sqrt(252)
        dailycov = self.dailysec1vol * self.dailysec2vol * self.corr
        self.dailycovmat = np.asarray([[self.dailysec1vol ** 2, dailycov], [dailycov, self.dailysec2vol ** 2]])
        self.rebalance_threshold = rebalance_threshold

    def Brownian(self, periods):
        """Generate correlated standard Brownian motion paths.

        Creates two correlated standard Brownian motion paths for the specified number
        of periods. These paths are used as the random component in geometric Brownian
        motion for asset price simulation.

        Args:
            periods (int): Number of time steps to simulate (typically trading days)

        Returns:
            np.array: Array of shape (periods+1, 2) containing Brownian motion values
                starting at [0, 0] and evolving as W ~ N(0, t)

        Notes:
            - Generates independent standard normal increments for each period
            - Cumulative sum produces Brownian motion: W(t) = sum of increments
            - W(0) = [0, 0] by construction (inserted at beginning)
            - Independence allows later correlation to be added via Cholesky decomposition
            - Time step dt = 1 (daily)
        """
        dt = 1
        # standard brownian increment = multivariate_normal distribution * sqrt of dt
        b = np.random.multivariate_normal((0., 0.), ((1., 0.), (0., 1.)), int(periods)) * np.sqrt(dt)
        # standard brownian motion for two variables ~ N(0,t)
        W = np.cumsum(b, axis=0)
        W = np.insert(W, 0, (0., 0.), axis=0)
        W = np.asarray(W)
        return W

    def GBM(self, W, T):
        """Generate correlated geometric Brownian motion price paths.

        Simulates asset price evolution using geometric Brownian motion (GBM) with
        correlated returns. Implements the analytical solution to the stochastic
        differential equation dS = μS dt + σS dW.

        Args:
            W (np.array): Standard Brownian motion paths from Brownian(), shape (T+1, 2)
            T (int): Number of time periods to simulate

        Returns:
            np.array: Simulated price paths for both securities, shape (T+1, 2)
                S[t, i] = price of security i at time t

        Mathematical Formula:
            S(t) = S(0) × exp((μ - σ²/2)t + σW(t))

            where the σW(t) terms are correlated via Cholesky decomposition:
            - L = Cholesky(Σ) where Σ is the covariance matrix
            - Correlated diffusion = L × W (independent Brownian motion)

        Notes:
            - Drift adjustment: (μ - σ²/2) ensures E[S(t)] = S(0)e^(μt)
            - Cholesky decomposition induces correlation between asset returns
            - Initial prices are self.initprices (default: $5 for both)
            - Returns log-normal distributed prices
        """
        S = []
        # divide time axis from 0 to 1 into T pieces,
        t = np.linspace(0, T, T + 1)
        L = np.linalg.cholesky(self.dailycovmat)
        var = self.dailycovmat.diagonal()
        for i in range(T + 1):
            drift = (self.dailymeans - (0.5 * var)) * t[i]
            diffusion = np.dot(L, W[i])
            S.append(self.initprices * np.exp(drift + diffusion))
        S = np.asarray(S)
        return S

    def PriceMove(self, periods):
        """Generate complete price path for both securities.

        Convenience method that combines Brownian motion generation and geometric
        Brownian motion to produce correlated price paths.

        Args:
            periods (int): Number of trading periods to simulate

        Returns:
            np.array: Price paths for both securities, shape (periods+1, 2)

        Notes:
            - Combines Brownian() and GBM() in single call
            - Each call uses current random state (affected by seed)
        """
        W = self.Brownian(periods)
        return self.GBM(W, periods)

    def Simulate(self, paths, tcost, periods, seed):
        """Run Monte Carlo simulation of portfolio rebalancing with visualization.

        Simulates multiple price paths, rebalancing the portfolio when weights drift
        beyond threshold. Tracks rebalancing frequency, trade sizes, and transaction
        cost impact on returns. Generates visualization of all simulated price paths.

        Args:
            paths (int): Number of Monte Carlo paths to simulate
            tcost (float): Transaction cost per dollar traded (e.g., 0.001 = 0.1%)
            periods (int): Number of trading days per path (typically 252 for 1 year)
            seed (int): Random seed for reproducibility

        Prints:
            - Rebalancing details for each path (via Rebalance method)
            - Average number of rebalances across all paths
            - Average total dollars traded
            - Average transaction cost as % of portfolio value
            - Average return reduction due to transaction costs (annualized %)

        Side Effects:
            - Sets numpy random seed
            - Creates and saves "simulate.png" plot showing all price paths
            - Resets portfolio state after each path

        Notes:
            - All paths plotted on same figure (can become crowded with many paths)
            - Portfolio resets to initial state between paths
            - Transaction cost impact measured as difference in annualized returns
        """
        cost = 0
        trade = 0
        nRebalance = 0
        decreaseReturn = 0
        fig, ax = plt.subplots(nrows=1, ncols=1)
        np.random.seed(seed)
        for i in range(paths):
            pricemovements = self.PriceMove(periods)
            print("path %d: " % (i + 1))
            tradePath, costPath, nRebalancePath, decreaseReturnPath = self.Rebalance(pricemovements, tcost, periods)
            cost += costPath
            trade += tradePath
            nRebalance += nRebalancePath
            decreaseReturn += decreaseReturnPath
            t = np.linspace(0, periods, periods + 1)
            image, = ax.plot(t, pricemovements[:, 0], label="stock1")
            image, = ax.plot(t, pricemovements[:, 1], label="stock2", ls='--')
            plt.ylabel('stock price, $')
            plt.xlabel('time, day')
            plt.title('correlated brownian simulation')
            plt.draw()
            fig.savefig("simulate.png")
        averageRebalance = nRebalance / paths
        averageDollarTraded = trade / paths
        averageTcost = cost / paths
        averageDecreaseReturn = decreaseReturn / paths
        print(
            "average number of rebalances: %.3f\naverage dollars traded: %.3f$\naverage transaction cost as percentage of book value: %.3f%%\nexpected transaction costs: %.3f%%"
            % (averageRebalance, averageDollarTraded, averageTcost * 100, averageDecreaseReturn * 100))

    def Rebalance(self, pricemovements, tcost, periods):
        """Simulate rebalancing along a price path and compute transaction costs.

        Steps through a price path day by day, updating portfolio values and checking
        if rebalancing is needed. When weights drift beyond threshold, rebalances back
        to 50/50 target weights and records transaction costs.

        Args:
            pricemovements (np.array): Simulated price paths, shape (periods+1, 2)
            tcost (float): Transaction cost per dollar traded
            periods (int): Number of trading days in the price path

        Returns:
            tuple: (tradeTotal, costTotalPer, nRebalance, decreaseReturn)
                - tradeTotal (float): Total dollars traded across all rebalances
                - costTotalPer (float): Total transaction cost as % of final portfolio value
                - nRebalance (int): Number of times portfolio was rebalanced
                - decreaseReturn (float): Annualized return reduction due to transaction costs

        Prints:
            DataFrame showing each rebalancing event with:
            - Current prices when rebalance occurred
            - Size of trade in dollars
            - Transaction cost incurred

        Rebalancing Logic:
            - Check weights after each day's price movement
            - Rebalance when max(|w_i - w_target_i|) >= threshold
            - Trade size = sum of absolute dollar changes to reach target weights
            - Transaction cost = trade size × tcost rate

        Notes:
            - Resets portfolio to initial state after computing metrics
            - Annualizes returns assuming 252 trading days per year
            - Transaction costs computed as: dollars traded × cost rate
            - Return impact = (pre-cost return) - (post-cost return)
        """
        trades = []
        priceSpread = []
        costs = []
        nRebalance = 0
        # len(pricemovements) = periods + 1
        for i in range(1, periods + 1):
            newPrices = pricemovements[i]
            # update prices, dollar value, and weightings of a portfolio each time prices change
            self.updatePrices(newPrices)
            difference = np.subtract(self.weightings, self.initweightings)
            # max returns a (positive) percentage difference between the actual weigntings and the desired weightings
            if max(difference) >= self.rebalance_threshold:
                # change the holdings so that the actual weightings are as desired
                self.updateHoldings()
                # difference in weightings * total = change of the amount of dollar invested in two stocks
                trade = np.sum(np.absolute(difference * self.total))
                trades.append(trade)
                costs.append(trade * tcost)
                priceSpread.append(np.round(self.prices, 2))
                nRebalance += 1
        # pandaframe
        data = {"price spread, $": priceSpread,
                "size of the trade, $": trades,
                "transaction cost, $": costs}
        df = pd.DataFrame(data=data, index=range(1, nRebalance + 1))
        df.index.name = "#rebalancing"
        print(df)
        # return metrics
        tradeTotal = sum(trades)
        costTotal = tradeTotal * tcost
        annualizedPeriods = periods / 252
        annualizedReturn = (self.total / self.inittotal) ** (1 / annualizedPeriods) - 1
        postcost = ((self.total - costTotal) / self.inittotal) ** (1 / annualizedPeriods) - 1
        decreaseReturn = annualizedReturn - postcost
        costTotalPer = costTotal / self.total
        # set parameters back to initial value
        self.reset()
        return tradeTotal, costTotalPer, nRebalance, decreaseReturn

    def reset(self):
        """Reset portfolio to initial state.

        Restores portfolio to starting conditions with original prices, holdings,
        weights, and total value. Called between Monte Carlo paths to ensure
        independence of simulations.

        Side Effects:
            - Sets weightings back to [0.5, 0.5]
            - Sets holdings back to [10, 10]
            - Sets prices back to [$5, $5]
            - Sets total value back to $100
        """
        self.weightings = self.initweightings
        self.holdings = self.initholdings
        self.prices = self.initprices
        self.total = self.inittotal

    def updatePrices(self, newPrices):
        """Update portfolio given new asset prices (holdings unchanged).

        Recalculates total portfolio value and asset weights based on new prices
        while keeping share holdings constant. This reflects passive price drift
        before any rebalancing decision.

        Args:
            newPrices (np.array): New prices for both securities

        Side Effects:
            - Updates self.prices to newPrices
            - Recalculates self.total as sum of (holdings × prices)
            - Recalculates self.weightings as (holdings × prices) / total
              for each security

        Notes:
            - Called every period to mark-to-market the portfolio
            - Weight drift occurs naturally as prices change
            - Does not change holdings (no trading)
        """
        self.prices = newPrices
        # dot product of the number of shares and price per share
        self.total = np.dot(self.holdings, newPrices)
        # the weight of stocks after stock prices change = (number of share * price of stock per share)/total amount of asset
        self.weightings = [holding * price / self.total for price, holding in zip(self.prices, self.holdings)]

    def updateHoldings(self):
        """Rebalance portfolio to target weights by adjusting holdings.

        Trades securities to restore target 50/50 weights at current prices.
        Calculates new share holdings that achieve target weights given current
        prices and total portfolio value.

        Side Effects:
            - Updates self.holdings to achieve target weights
            - Recalculates self.weightings (should equal initweightings)

        Trading Logic:
            New holdings for security i:
                h_i = (total_value × target_weight_i) / price_i

            This ensures: (h_i × price_i) / total_value = target_weight_i

        Notes:
            - Called when weights drift beyond rebalance threshold
            - Does not account for transaction costs (those tracked separately)
            - Assumes fractional shares allowed
            - Weightings recalculated to verify (should be [0.5, 0.5])
        """
        self.holdings = [self.total * initWeight / price for initWeight, price in zip(self.initweightings, self.prices)]
        self.weightings = [price * holding / self.total for holding, price in zip(self.holdings, self.prices)]

    def decreaseReturn(self, pricemovements, tcost, periods):
        """Compute annualized return reduction due to transaction costs.

        Simulates rebalancing along a price path and calculates the difference
        between pre-cost and post-cost annualized returns. Used for sensitivity
        analysis and convergence testing.

        Args:
            pricemovements (np.array): Simulated price paths, shape (periods+1, 2)
            tcost (float): Transaction cost per dollar traded
            periods (int): Number of trading days

        Returns:
            float: Annualized return reduction due to transaction costs (e.g., 0.02 = 2%)

        Calculation:
            1. Simulate rebalancing, accumulating total transaction costs
            2. Compute pre-cost annualized return from final portfolio value
            3. Compute post-cost annualized return = (final value - costs)^(1/years) - 1
            4. Return difference: pre-cost return - post-cost return

        Notes:
            - Does not print detailed rebalancing information (unlike Rebalance method)
            - Resets portfolio state after calculation
            - Annualizes using 252 trading days per year
            - Used by Tests() and solve*() methods for parameter sweeps
        """
        costTotal = 0
        for i in range(1, len(pricemovements)):
            newPrices = pricemovements[i]
            self.updatePrices(newPrices)
            difference = np.subtract(self.weightings, self.initweightings)
            if max(difference) >= self.rebalance_threshold:
                self.updateHoldings()
                trade = np.sum(np.absolute(difference * self.total))
                costTotal += trade * tcost
        annualizedPeriods = periods / 252
        annualizedReturn = (self.total / self.inittotal) ** (1 / annualizedPeriods) - 1
        postcost = ((self.total - costTotal) / self.inittotal) ** (1 / annualizedPeriods) - 1
        decreaseReturn = annualizedReturn - postcost
        self.reset()
        return decreaseReturn

    def Tests(self, paths, tcost, periods, step, seed):
        """Test convergence of Monte Carlo transaction cost estimates.

        Runs many Monte Carlo paths and tracks how the sample mean transaction cost
        converges. Plots cumulative average vs number of paths to visualize
        convergence behavior and assess required number of iterations.

        Args:
            paths (int): Total number of Monte Carlo paths to run
            tcost (float): Transaction cost per dollar traded
            periods (int): Number of trading days per path
            step (int): Record cumulative average every 'step' paths
            seed (int): Random seed for reproducibility

        Prints:
            Final average transaction cost as percentage of return

        Side Effects:
            - Sets numpy random seed
            - Creates and saves plot: "convergence test (seed=<seed>).png"
            - Plot shows sample mean transaction cost vs number of paths
            - Helps determine if enough paths used for stable estimates

        Notes:
            - Cumulative average computed at multiples of 'step'
            - Smaller step gives smoother convergence plot but more computation
            - Well-converged estimates show flattening curve
            - Different seeds may have different convergence rates
        """
        meanDecrease = []
        totalDecrease = 0
        fig, ax = plt.subplots(nrows=1, ncols=1)
        np.random.seed(seed)
        for i in range(1, paths + 1):
            pricemovements = self.PriceMove(periods)
            decreaseReturn = self.decreaseReturn(pricemovements, tcost, periods)
            totalDecrease += decreaseReturn * 100
            if (i % step == 0):
                meanDecrease.append(totalDecrease / i)
        print("when seed = %d, paths = %d, the average transaction cost is: %f%%" % (seed, paths, meanDecrease[-1]))
        t = np.linspace(1, paths, len(meanDecrease))
        image, = ax.plot(t, meanDecrease)
        plt.ylabel('sample mean transaction cost (%)')
        plt.xlabel('number of paths')
        plt.title('convergence test (seed = %d)' % (seed))
        plt.draw()
        fig.savefig("convergence test (seed=%d).png" % (seed))

    def updateCorr(self, corr):
        """Update correlation coefficient and rebuild covariance matrix.

        Args:
            corr (float): New correlation coefficient between securities (0 to 1)

        Side Effects:
            - Rebuilds self.dailycovmat with new correlation
            - Preserves existing volatilities

        Notes:
            - Used by solveCorr() to sweep correlation parameter
            - Does not update self.corr attribute (parameter sweeps are temporary)
        """
        dailycov = self.dailysec1vol * self.dailysec2vol * corr
        self.dailycovmat = np.asarray([[self.dailysec1vol ** 2, dailycov], [dailycov, self.dailysec2vol ** 2]])

    def updateSec1Vol(self, sec1vol):
        """Update security 1 volatility and rebuild covariance matrix.

        Args:
            sec1vol (float): New annualized volatility for security 1

        Side Effects:
            - Updates self.sec1vol and self.dailysec1vol
            - Rebuilds self.dailycovmat with new volatility
            - Preserves existing correlation and security 2 volatility

        Notes:
            - Converts annualized volatility to daily: vol / sqrt(252)
            - Used by solveSec1Vol() to sweep volatility parameter
        """
        self.sec1vol = sec1vol
        self.dailysec1vol = sec1vol / np.sqrt(252)
        dailycov = self.dailysec1vol * self.dailysec2vol * self.corr
        self.dailycovmat = np.asarray([[self.dailysec1vol ** 2, dailycov], [dailycov, self.dailysec2vol ** 2]])

    def updateThreshold(self, threshold):
        """Update rebalancing threshold.

        Args:
            threshold (float): New minimum weight divergence to trigger rebalance

        Side Effects:
            - Updates self.rebalance_threshold

        Notes:
            - Used by solveThreshold() to sweep threshold parameter
            - Higher thresholds mean less frequent rebalancing
        """
        self.rebalance_threshold = threshold

    def updateSec1Mean(self, sec1mean):
        """Update security 1 expected return.

        Args:
            sec1mean (float): New annualized expected return for security 1

        Side Effects:
            - Updates self.means[0]
            - Recalculates self.dailymeans (annualized / 252)

        Notes:
            - Used by solveSec1Mean() to sweep expected return parameter
            - Security 2 return unchanged
        """
        self.means[0] = sec1mean
        self.dailymeans = self.means / 252

    def solveCorr(self, paths, tcost, periods, seed):
        """Analyze transaction cost sensitivity to correlation coefficient.

        Sweeps correlation from 0 to 1 in 11 steps, running Monte Carlo simulations
        at each value to estimate average transaction cost impact. Fits linear
        regression to quantify relationship.

        Args:
            paths (int): Number of Monte Carlo paths per correlation value
            tcost (float): Transaction cost per dollar traded
            periods (int): Number of trading days per path
            seed (int): Random seed for reproducibility

        Prints:
            - Parameter settings (seed, volatilities, return, threshold)
            - Linear regression coefficients (slope, intercept)

        Side Effects:
            - Creates and saves plot: "corr-tcost graph.png"
            - Plot shows transaction cost (%) vs correlation coefficient
            - Temporarily modifies correlation via updateCorr()

        Analysis:
            - Higher correlation typically reduces rebalancing frequency
            - Assets moving together means weights stay more stable
            - Linear fit quantifies sensitivity: d(tcost)/d(corr)

        Notes:
            - Tests 11 correlation values: 0.0, 0.1, ..., 1.0
            - Each point is average of 'paths' Monte Carlo runs
            - Portfolio parameters reset between correlation values
        """
        start = 0
        end = 1
        x = np.linspace(0, 1, 11)
        y = []
        for i in range(len(x)):
            totalDecrease = 0
            self.updateCorr(x[i])
            np.random.seed(seed)
            for i in range(paths):
                pricemovements = self.PriceMove(periods)
                decreaseReturn = self.decreaseReturn(pricemovements, tcost, periods)
                totalDecrease += decreaseReturn * 100
            meanDecrease = np.round(totalDecrease / paths, 1)
            y.append(meanDecrease)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image, = ax.plot(x, y)
        plt.ylabel('transaction cost (%)')
        plt.xlabel('correlation coefficient')
        plt.title('corr - tcost graph')
        plt.draw()
        fig.savefig('corr-tcost graph')
        print(
            'corr-tcost:\nseed=%d\nsec1vol=%.2f\nsec2vol=%.2f\ncorr=%.2f-%.2f\nsec1mean=%.2f\nsec2mean=%.2f\nthreshold=%.2f'
            % (seed, self.sec1vol, self.sec2vol, start, end, self.means[0], self.means[1], self.rebalance_threshold))
        print('coeff:', np.polyfit(x, y, 1))
        '''reg = linear_model.Lasso(alpha = 0.1)
        reg.fit(x,y)
        print('lasso coeff:',reg.coef_)
        print('lasso intercept',reg.intercept_)'''

    def solveSec1Vol(self, paths, tcost, periods, seed):
        """Analyze transaction cost sensitivity to security 1 volatility.

        Sweeps security 1 annualized volatility from 1% to 51% in 11 steps, running
        Monte Carlo simulations at each value to estimate average transaction cost
        impact. Fits linear regression to quantify relationship.

        Args:
            paths (int): Number of Monte Carlo paths per volatility value
            tcost (float): Transaction cost per dollar traded
            periods (int): Number of trading days per path
            seed (int): Random seed for reproducibility

        Prints:
            - Parameter settings (seed, vol range, correlation, returns, threshold)
            - Linear regression coefficients (slope, intercept)

        Side Effects:
            - Creates and saves plot: "sec1vol-tcost graph.png"
            - Plot shows transaction cost (%) vs security 1 volatility
            - Temporarily modifies sec1vol via updateSec1Vol()

        Analysis:
            - Higher volatility increases rebalancing frequency
            - More price movement causes weights to drift faster
            - Linear fit quantifies sensitivity: d(tcost)/d(vol)

        Notes:
            - Tests 11 volatility values: 0.01, 0.06, 0.11, ..., 0.51
            - Each point is average of 'paths' Monte Carlo runs
            - Security 2 volatility held constant
        """
        start = .01
        end = .51
        x = np.linspace(start, end, 11)
        y = []
        for i in range(len(x)):
            totalDecrease = 0
            self.updateSec1Vol(x[i])
            np.random.seed(seed)
            for i in range(paths):
                pricemovements = self.PriceMove(periods)
                decreaseReturn = self.decreaseReturn(pricemovements, tcost, periods)
                totalDecrease += decreaseReturn * 100
            meanDecrease = np.round(totalDecrease / paths, 1)
            y.append(meanDecrease)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image, = ax.plot(x, y)
        plt.ylabel('transaction cost (%)')
        plt.xlabel('security 1 volatility')
        plt.title('sec1vol - tcost graph')
        plt.draw()
        fig.savefig('sec1vol-tcost graph')
        print(
            'sec1vol_tcost:\nseed=%d\nsec1vol=%.2f-%.2f\nsec2vol=%.2f\ncorr=%.2f\nsec1mean=%.2f\nsec2mean=%.2f\nthreshold=%.2f'
            % (seed, start, end, self.sec2vol, self.corr, self.means[0], self.means[1], self.rebalance_threshold))
        print("coeff:", np.polyfit(x, y, 1))

    def solveSec1Mean(self, paths, tcost, periods, seed):
        """Analyze transaction cost sensitivity to security 1 expected return.

        Sweeps security 1 annualized expected return from 0% to 50% in 11 steps,
        running Monte Carlo simulations at each value to estimate average transaction
        cost impact. Fits linear regression to quantify relationship.

        Args:
            paths (int): Number of Monte Carlo paths per return value (overridden to 500)
            tcost (float): Transaction cost per dollar traded
            periods (int): Number of trading days per path
            seed (int): Random seed for reproducibility

        Prints:
            - Parameter settings (seed, volatilities, correlation, return range, threshold)
            - Linear regression coefficients (slope, intercept)

        Side Effects:
            - Creates and saves plot: "sec1mean-tcost graph.png"
            - Plot shows transaction cost (%) vs security 1 expected return
            - Temporarily modifies sec1mean via updateSec1Mean()

        Analysis:
            - Expected return may affect rebalancing indirectly via drift
            - Higher return differential creates more weight imbalance over time
            - Linear fit quantifies sensitivity: d(tcost)/d(return)

        Notes:
            - Tests 11 return values: 0.0, 0.05, 0.10, ..., 0.50
            - Each point is average of 500 Monte Carlo runs (hardcoded)
            - Security 2 return held constant
            - Input 'paths' parameter is overridden to 500 in the code
        """
        start = 0
        end = .5
        x = np.linspace(0, .5, 11)
        paths = 500
        y = []
        for i in range(len(x)):
            totalDecrease = 0
            self.updateSec1Mean(x[i])
            np.random.seed(seed)
            for i in range(paths):
                pricemovements = self.PriceMove(periods)
                decreaseReturn = self.decreaseReturn(pricemovements, tcost, periods)
                totalDecrease += decreaseReturn * 100
            meanDecrease = np.round(totalDecrease / paths, 1)
            y.append(meanDecrease)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image, = ax.plot(x, y)
        plt.ylabel('transaction cost (%)')
        plt.xlabel('security 1 return')
        plt.title('sec1mean - tcost graph')
        plt.draw()
        fig.savefig('sec1mean-tcost graph')
        print(
            'sec1mean-tcost:\nseed=%d\nsec1vol=%.2f\nsec2vol=%.2f\ncorr=%.2f\nsec1mean=%.2f-%.2f\nsec2mean=%.2f\nthreshold=%.2f'
            % (seed, self.sec1vol, self.sec2vol, self.corr, start, end, self.means[1], self.rebalance_threshold))
        print('coef:', np.polyfit(x, y, 1))

    def solveThreshold(self, paths, tcost, periods, seed):
        """Analyze transaction cost sensitivity to rebalancing threshold.

        Sweeps rebalancing threshold from 1% to 10% in 11 steps, running Monte Carlo
        simulations at each value to estimate average transaction cost impact. Fits
        linear regression to quantify relationship.

        Args:
            paths (int): Number of Monte Carlo paths per threshold value
            tcost (float): Transaction cost per dollar traded
            periods (int): Number of trading days per path
            seed (int): Random seed for reproducibility

        Prints:
            - Parameter settings (seed, volatilities, correlation, returns, threshold range)
            - Linear regression coefficients (slope, intercept)

        Side Effects:
            - Creates and saves plot: "threshold-tcost graph.png"
            - Plot shows transaction cost (%) vs rebalance threshold (%)
            - Temporarily modifies threshold via updateThreshold()

        Analysis:
            - Higher threshold reduces rebalancing frequency
            - Portfolio allowed to drift more before rebalancing
            - Trade-off: fewer rebalances vs larger tracking error
            - Linear fit quantifies sensitivity: d(tcost)/d(threshold)

        Notes:
            - Tests 11 threshold values: 1%, 1.9%, 2.8%, ..., 10%
            - Each point is average of 'paths' Monte Carlo runs
            - Optimal threshold balances transaction costs vs tracking error
        """
        start = 1
        end = 10
        x = np.linspace(1, 10, 11)
        y = []
        for i in range(len(x)):
            totalDecrease = 0
            self.updateThreshold(x[i] / 100)
            np.random.seed(seed)
            for i in range(paths):
                pricemovements = self.PriceMove(periods)
                decreaseReturn = self.decreaseReturn(pricemovements, tcost, periods)
                totalDecrease += decreaseReturn * 100
            meanDecrease = np.round(totalDecrease / paths, 1)
            y.append(meanDecrease)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image, = ax.plot(x, y)
        plt.ylabel('transaction cost (%)')
        plt.xlabel('rebalance threshold (%)')
        plt.title('threshold - tcost graph')
        plt.draw()
        fig.savefig("threshold-tcost graph")
        print(
            "threshold-tcost:\nseed=%d\nsec1vol=%.2f\nsec2vol=%.2f\ncorr=%.2f\nsec1mean=%.2f\nsec2mean=%.2f\nthreshold=%.2f-%.2f"
            % (seed, self.sec1vol, self.sec2vol, self.corr, self.means[0], self.means[1], start, end))
        print('coef:', np.polyfit(x, y, 1))


def main():
    """Command-line interface for portfolio rebalancing simulation.

    Parses command-line arguments and dispatches to appropriate simulation method
    based on flags. Supports basic simulation, convergence testing, and parameter
    sensitivity analysis.

    Command-line Arguments:
        Portfolio parameters:
            --sec1vol: Annualized volatility of security 1 (default: 0.4)
            --sec2vol: Annualized volatility of security 2 (default: 0.3)
            --corr: Correlation between securities (default: 0.8)
            --sec1mean: Annualized return of security 1 (default: 0.05)
            --sec2mean: Annualized return of security 2 (default: 0.1)

        Simulation parameters:
            --paths: Number of Monte Carlo paths (default: 500)
            --periods: Number of trading days (default: 252)
            --tcost: Transaction cost per dollar traded (default: 0.1, i.e. 10%)
            --rebalance_threshold: Weight divergence to trigger rebalance (default: 0.01)
            --seed: Random seed (default: 5)

        Analysis modes (mutually exclusive):
            --simulate: Run full simulation with visualization
            --convergence_test: Test Monte Carlo convergence
            --solveCorr: Analyze correlation sensitivity
            --solveVol: Analyze volatility sensitivity
            --solveReturn: Analyze return sensitivity
            --solveThreshold: Analyze threshold sensitivity

        Convergence test parameter:
            --step: Sample convergence every N paths (default: 10)

    Examples:
        python simulation.py --simulate=True --paths=100
        python simulation.py --convergence_test=True --step=50
        python simulation.py --solveCorr=True --sec1vol=0.3
        python simulation.py --solveThreshold=True --corr=0.5

    Notes:
        - Only one analysis mode should be True at a time
        - Default tcost=0.1 is very high (10%); realistic values are ~0.001
        - Results saved as PNG files in current directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec1vol", help="annualized volatility of security 1", type=float, default=.4)
    parser.add_argument("--sec2vol", help="annualized volatility of security 2", type=float, default=.3)
    parser.add_argument("--corr", help="correlation between security 1 and 2", type=float, default=.8)
    parser.add_argument("--sec1mean", help="annualized return of security 1", type=float, default=.05)
    parser.add_argument("--sec2mean", help="annualized return of security 2", type=float, default=.1)
    parser.add_argument("--paths", help="number of monte carlo iterations", type=int, default=500)
    parser.add_argument("--periods", help="number of days", type=int, default=252)
    parser.add_argument("--tcost", help="transaction cost per trade", type=float, default=.1)
    parser.add_argument("--rebalance_threshold", help="the minimal divergence that causes rebalance", type=float,
                        default=.01)
    parser.add_argument("--seed", help="set seed for the simulation", type=int, default=5)
    parser.add_argument("--simulate",
                        help="plot price movements of two stocks and print information about their transaction costs",
                        type=bool, default=False)
    parser.add_argument("--convergence_test", help="test convergence of transaction cost", type=bool, default=False)
    parser.add_argument("--step", help="set the step for convergence test", type=int, default=10)
    parser.add_argument("--solveCorr", help="solve transaction cost with respect to correlation coefficient", type=bool,
                        default=False)
    parser.add_argument("--solveVol", help="solve transaction cost with respect to the volatity of a security",
                        type=bool, default=False)
    parser.add_argument("--solveReturn", help="solve transaction cost with respect to the return of a security",
                        type=bool, default=False)
    parser.add_argument("--solveThreshold", help="solve transaction cost with respect to the rebalance threshold",
                        type=bool, default=False)
    args = parser.parse_args()
    portfolio = Portfolio(args.sec1mean, args.sec2mean,
                          args.sec1vol, args.sec2vol, args.corr, args.rebalance_threshold)
    if args.simulate == True:
        portfolio.Simulate(args.paths, args.tcost, args.periods, args.seed)
    elif args.convergence_test == True:
        portfolio.Tests(args.paths, args.tcost, args.periods, args.step, args.seed)
    elif args.solveCorr == True:
        portfolio.solveCorr(args.paths, args.tcost, args.periods, args.seed)
    elif args.solveVol == True:
        portfolio.solveSec1Vol(args.paths, args.tcost, args.periods, args.seed)
    elif args.solveThreshold == True:
        portfolio.solveThreshold(args.paths, args.tcost, args.periods, args.seed)
    elif args.solveReturn == True:
        portfolio.solveSec1Mean(args.paths, args.tcost, args.periods, args.seed)


main()
