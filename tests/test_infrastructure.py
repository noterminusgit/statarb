"""
Infrastructure smoke tests.

Verifies that the pytest testing infrastructure is set up correctly
and that fixtures are working as expected.
"""

import pytest
import numpy as np
import pandas as pd


def test_pytest_working():
    """Verify pytest is working."""
    assert True


def test_sample_price_df_fixture(sample_price_df):
    """Verify sample_price_df fixture works correctly."""
    # Check structure
    assert isinstance(sample_price_df, pd.DataFrame)
    assert len(sample_price_df) == 50  # 10 stocks * 5 days

    # Check index
    assert sample_price_df.index.names == ['date', 'sid']

    # Check columns
    expected_cols = ['close', 'open', 'high', 'low', 'volume', 'dollars']
    for col in expected_cols:
        assert col in sample_price_df.columns

    # Check data validity
    assert (sample_price_df['close'] > 0).all()
    assert (sample_price_df['volume'] > 0).all()
    assert (sample_price_df['dollars'] > 0).all()
    assert (sample_price_df['high'] >= sample_price_df['close']).all()
    assert (sample_price_df['low'] <= sample_price_df['close']).all()


def test_sample_returns_df_fixture(sample_returns_df):
    """Verify sample_returns_df fixture works correctly."""
    assert isinstance(sample_returns_df, pd.DataFrame)
    assert 'ret' in sample_returns_df.columns
    assert 'o2c' in sample_returns_df.columns
    assert sample_returns_df.index.names == ['date', 'sid']


def test_sample_universe_df_fixture(sample_universe_df):
    """Verify sample_universe_df fixture works correctly."""
    assert isinstance(sample_universe_df, pd.DataFrame)
    assert len(sample_universe_df) == 10
    assert sample_universe_df.index.name == 'sid'

    expected_cols = ['symbol', 'sector_name', 'mkt_cap']
    for col in expected_cols:
        assert col in sample_universe_df.columns

    assert (sample_universe_df['mkt_cap'] > 0).all()


def test_sample_barra_df_fixture(sample_barra_df):
    """Verify sample_barra_df fixture works correctly."""
    assert isinstance(sample_barra_df, pd.DataFrame)
    assert len(sample_barra_df) == 50  # 10 stocks * 5 days
    assert sample_barra_df.index.names == ['date', 'sid']

    # Check key factor columns
    expected_factors = ['beta', 'momentum', 'size', 'volatility']
    for factor in expected_factors:
        assert factor in sample_barra_df.columns


def test_sample_volume_df_fixture(sample_volume_df):
    """Verify sample_volume_df fixture works correctly."""
    assert isinstance(sample_volume_df, pd.DataFrame)
    assert len(sample_volume_df) == 50
    assert sample_volume_df.index.names == ['date', 'sid']

    expected_cols = ['adv', 'advp', 'mkt_cap']
    for col in expected_cols:
        assert col in sample_volume_df.columns

    assert (sample_volume_df['adv'] > 0).all()
    assert (sample_volume_df['advp'] > 0).all()
    assert (sample_volume_df['mkt_cap'] > 0).all()
