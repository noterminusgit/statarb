"""
Pytest fixtures for statarb testing.

This module provides shared fixtures for testing the statistical arbitrage
codebase. Fixtures generate synthetic data that matches the structure of
real market data used by the system.

Available Fixtures:
    sample_price_df: Simple 10-stock, 5-day price DataFrame
    sample_returns_df: Returns data for testing
    sample_universe_df: Mock universe definition
    sample_barra_df: Barra factor exposures
    sample_volume_df: Trading volume data
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_price_df():
    """
    Create a simple 10-stock, 5-day price DataFrame.

    Returns:
        DataFrame with MultiIndex (date, sid) and columns:
            - close: Closing prices
            - open: Opening prices
            - high: High prices
            - low: Low prices
            - volume: Share volume
            - dollars: Dollar volume (close * volume)

    Example:
        >>> def test_something(sample_price_df):
        ...     assert len(sample_price_df) == 50  # 10 stocks * 5 days
    """
    dates = pd.date_range(start='2013-01-02', periods=5, freq='D')
    sids = range(1000, 1010)  # 10 stocks with sid 1000-1009

    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

    # Generate synthetic price data
    np.random.seed(42)
    n_rows = len(index)

    close_prices = 50.0 + np.random.randn(n_rows) * 5.0
    close_prices = np.maximum(close_prices, 10.0)  # Floor at $10

    df = pd.DataFrame({
        'close': close_prices,
        'open': close_prices * (1.0 + np.random.randn(n_rows) * 0.01),
        'high': close_prices * (1.0 + np.abs(np.random.randn(n_rows)) * 0.02),
        'low': close_prices * (1.0 - np.abs(np.random.randn(n_rows)) * .02),
        'volume': np.random.randint(100000, 1000000, n_rows),
    }, index=index)

    df['dollars'] = df['close'] * df['volume']

    return df


@pytest.fixture
def sample_returns_df(sample_price_df):
    """
    Create returns DataFrame from price data.

    Args:
        sample_price_df: Price fixture (injected by pytest)

    Returns:
        DataFrame with MultiIndex (date, sid) and columns:
            - ret: Daily returns (close-to-close)
            - o2c: Open-to-close returns

    Example:
        >>> def test_returns(sample_returns_df):
        ...     assert 'ret' in sample_returns_df.columns
    """
    df = sample_price_df.copy()

    # Calculate returns within each sid group
    df['ret'] = df.groupby(level='sid')['close'].pct_change()
    df['o2c'] = (df['close'] - df['open']) / df['open']

    return df[['ret', 'o2c']]


@pytest.fixture
def sample_universe_df():
    """
    Create mock universe definition DataFrame.

    Returns:
        DataFrame with index=sid and columns:
            - symbol: Stock ticker symbol
            - sector_name: Sector classification
            - mkt_cap: Market capitalization

    Example:
        >>> def test_universe(sample_universe_df):
        ...     assert len(sample_universe_df) == 10
    """
    sids = range(1000, 1010)
    symbols = ['STOCK{}'.format(i) for i in range(10)]
    sectors = ['Technology', 'Financials', 'Healthcare', 'Energy', 'Industrials',
               'Technology', 'Financials', 'Consumer', 'Utilities', 'Materials']

    df = pd.DataFrame({
        'symbol': symbols,
        'sector_name': sectors,
        'mkt_cap': np.random.uniform(1e9, 100e9, 10),
    }, index=sids)

    df.index.name = 'sid'

    return df


@pytest.fixture
def sample_barra_df():
    """
    Create Barra factor exposure DataFrame.

    Returns:
        DataFrame with MultiIndex (date, sid) and columns:
            - beta: Market beta
            - momentum: 12-month momentum
            - size: Log market cap
            - volatility: Historical volatility
            - ind_xxx: Industry dummy variables

    Example:
        >>> def test_barra(sample_barra_df):
        ...     assert 'beta' in sample_barra_df.columns
    """
    dates = pd.date_range(start='2013-01-02', periods=5, freq='D')
    sids = range(1000, 1010)

    index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

    np.random.seed(42)
    n_rows = len(index)

    df = pd.DataFrame({
        'beta': 1.0 + np.random.randn(n_rows) * 0.3,
        'momentum': np.random.randn(n_rows) * 0.2,
        'size': np.random.uniform(8, 12, n_rows),  # Log market cap
        'volatility': np.random.uniform(0.15, 0.45, n_rows),
        'value': np.random.randn(n_rows) * 0.1,
        'growth': np.random.randn(n_rows) * 0.1,
    }, index=index)

    # Add industry dummies (one-hot encoded)
    industries = ['ind_tech', 'ind_finance', 'ind_health', 'ind_energy']
    for ind in industries:
        df[ind] = np.random.choice([0, 1], n_rows, p=[0.7, 0.3])

    return df


@pytest.fixture
def sample_volume_df():
    """
    Create trading volume profile DataFrame.

    Returns:
        DataFrame with MultiIndex (date, sid) and columns:
            - adv: Average daily volume (20-day)
            - advp: Average daily dollar volume
            - mkt_cap: Market capitalization

    Example:
        >>> def test_volume(sample_volume_df):
        ...     assert (sample_volume_df['advp'] > 0).all()
    """
    dates = pd.date_range(start='2013-01-02', periods=5, freq='D')
    sids = range(1000, 1010)

    index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

    np.random.seed(42)
    n_rows = len(index)

    adv = np.random.uniform(500000, 5000000, n_rows)
    price = np.random.uniform(20, 100, n_rows)

    df = pd.DataFrame({
        'adv': adv,
        'advp': adv * price,
        'mkt_cap': np.random.uniform(1e9, 50e9, n_rows),
    }, index=index)

    return df
