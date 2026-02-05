"""
Unit tests for calc.py module.

This module tests calculation functions for alpha factors, forward returns,
and statistical transformations used in the statistical arbitrage system.

Critical tests:
    - winsorize_by_date: Verifies cross-sectional outlier clipping
    - calc_forward_returns: Verifies forward-looking return calculations
    - mkt_ret: Verifies market-cap weighted return calculations
    - winsorize: Verifies statistical outlier trimming
    - create_z_score: Verifies cross-sectional standardization
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import functions from calc module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import calc


class TestWinsorize:
    """Test suite for winsorize function."""

    def test_winsorize_basic(self):
        """Test basic winsorization with default std_level=5."""
        # Create data with outliers
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier

        result = calc.winsorize(data, std_level=2)

        # Check that outlier was clipped
        assert result.max() < 100, "Outlier should be clipped"

        # Check that normal values unchanged
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 2.0

    def test_winsorize_symmetric_clipping(self):
        """Test that both upper and lower outliers are clipped."""
        # Create symmetric outliers
        data = pd.Series([-100, -1, 0, 1, 2, 100])

        result = calc.winsorize(data, std_level=2)

        # Both outliers should be clipped
        assert result.iloc[0] > -100, "Lower outlier should be clipped"
        assert result.iloc[-1] < 100, "Upper outlier should be clipped"

        # Check symmetry around mean
        mean = data.mean()
        std = data.std()
        upper_threshold = mean + 2 * std
        lower_threshold = mean - 2 * std

        assert result.max() <= upper_threshold + 1e-10, "Max should be at upper threshold"
        assert result.min() >= lower_threshold - 1e-10, "Min should be at lower threshold"

    def test_winsorize_no_outliers(self):
        """Test winsorization when no outliers present."""
        # Create normal data
        np.random.seed(42)
        data = pd.Series(np.random.randn(100))

        result = calc.winsorize(data, std_level=5)

        # With std_level=5, almost no clipping should occur for normal data
        # Check that most values are unchanged
        assert np.abs(result - data).max() < 1e-10 or len(data[np.abs(data - data.mean()) > 5 * data.std()]) > 0

    def test_winsorize_exact_threshold(self):
        """Test exact threshold calculation."""
        # Create data where we know the threshold
        data = pd.Series([0, 0, 0, 0, 10])  # mean=2, std~=4.47

        result = calc.winsorize(data, std_level=1)

        mean = data.mean()
        std = data.std()

        # Value at exactly threshold should be at threshold
        assert result.max() <= mean + std + 1e-10
        assert result.min() >= mean - std - 1e-10

    def test_winsorize_all_same_values(self):
        """Test winsorization when all values are identical."""
        data = pd.Series([5.0, 5.0, 5.0, 5.0])

        result = calc.winsorize(data, std_level=2)

        # All values should remain unchanged (std=0)
        assert (result == 5.0).all()


class TestWinsorizeByDate:
    """Test suite for winsorize_by_date function."""

    def test_winsorize_by_date_basic(self):
        """Test cross-sectional winsorization by date."""
        # Create data with two dates, outliers on each date
        dates = pd.date_range(start='2013-01-02', periods=2, freq='D')
        sids = range(1000, 1005)  # 5 stocks

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # First date: normal values with one outlier
        # Second date: normal values with different outlier
        values = [1, 2, 3, 4, 100,  # Date 1: 100 is outlier
                  5, 6, 7, 8, 200]   # Date 2: 200 is outlier

        data = pd.Series(values, index=index)

        result = calc.winsorize_by_date(data)

        # Check that outliers were clipped within each date
        date1_data = result.xs(dates[0], level='date')
        date2_data = result.xs(dates[1], level='date')

        assert date1_data.max() < 100, "Date 1 outlier should be clipped"
        assert date2_data.max() < 200, "Date 2 outlier should be clipped"

    def test_winsorize_by_date_independence(self):
        """Test that dates are winsorized independently."""
        # Create data where one date has high values, other has low values
        dates = pd.date_range(start='2013-01-02', periods=2, freq='D')
        sids = range(1000, 1010)  # 10 stocks

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Date 1: values around 100
        # Date 2: values around 1
        np.random.seed(42)
        values = list(100 + np.random.randn(10) * 2) + list(1 + np.random.randn(10) * 0.1)

        data = pd.Series(values, index=index)

        result = calc.winsorize_by_date(data)

        # Each date should be winsorized relative to its own distribution
        date1_result = result.xs(dates[0], level='date')
        date2_result = result.xs(dates[1], level='date')

        # Date 1 should still have values around 100
        assert date1_result.mean() > 90
        # Date 2 should still have values around 1
        assert date2_result.mean() < 10

    def test_winsorize_by_date_preserves_index(self):
        """Test that MultiIndex structure is preserved."""
        dates = pd.date_range(start='2013-01-02', periods=3, freq='D')
        sids = range(1000, 1005)

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])
        data = pd.Series(np.random.randn(15), index=index)

        result = calc.winsorize_by_date(data)

        # Check index preserved
        assert result.index.names == ['date', 'sid']
        assert len(result) == len(data)
        assert result.index.equals(data.index)


class TestCalcForwardReturns:
    """Test suite for calc_forward_returns function."""

    def test_calc_forward_returns_basic(self):
        """Test basic forward return calculation."""
        # Create simple log returns
        dates = pd.date_range(start='2013-01-02', periods=5, freq='D')
        sids = [1000]

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Log returns: 0.01 each day
        log_returns = pd.Series([0.01] * 5, index=index)
        daily_df = pd.DataFrame({'log_ret': log_returns})

        result = calc.calc_forward_returns(daily_df, horizon=3)

        # Check columns created
        assert 'cum_ret1' in result.columns
        assert 'cum_ret2' in result.columns
        assert 'cum_ret3' in result.columns

        # Check that forward returns are calculated
        # First row should have cum_ret1 = second day's return (0.01)
        assert np.isclose(result['cum_ret1'].iloc[0], 0.01, atol=1e-10)
        # First row should have cum_ret2 = sum of next 2 days (0.02)
        assert np.isclose(result['cum_ret2'].iloc[0], 0.02, atol=1e-10)
        # First row should have cum_ret3 = sum of next 3 days (0.03)
        assert np.isclose(result['cum_ret3'].iloc[0], 0.03, atol=1e-10)

    def test_calc_forward_returns_multiple_stocks(self):
        """Test forward returns for multiple stocks independently."""
        dates = pd.date_range(start='2013-01-02', periods=5, freq='D')
        sids = [1000, 1001]

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Stock 1000: returns of 0.01
        # Stock 1001: returns of 0.02
        log_returns = [0.01] * 5 + [0.02] * 5
        daily_df = pd.DataFrame({'log_ret': log_returns}, index=index)

        result = calc.calc_forward_returns(daily_df, horizon=2)

        # Check stock 1000
        stock_1000 = result.xs(1000, level='sid')
        assert np.isclose(stock_1000['cum_ret1'].iloc[0], 0.01, atol=1e-10)
        assert np.isclose(stock_1000['cum_ret2'].iloc[0], 0.02, atol=1e-10)

        # Check stock 1001
        stock_1001 = result.xs(1001, level='sid')
        assert np.isclose(stock_1001['cum_ret1'].iloc[0], 0.02, atol=1e-10)
        assert np.isclose(stock_1001['cum_ret2'].iloc[0], 0.04, atol=1e-10)

    def test_calc_forward_returns_end_of_series(self):
        """Test that last dates have NaN forward returns (no future data)."""
        dates = pd.date_range(start='2013-01-02', periods=5, freq='D')
        sids = [1000]

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])
        log_returns = pd.Series([0.01] * 5, index=index)
        daily_df = pd.DataFrame({'log_ret': log_returns})

        result = calc.calc_forward_returns(daily_df, horizon=2)

        # Last date should have NaN (no future data)
        assert pd.isna(result['cum_ret1'].iloc[-1])
        assert pd.isna(result['cum_ret2'].iloc[-1])

        # Second to last should have NaN for cum_ret2 (needs 2 future days)
        assert pd.isna(result['cum_ret2'].iloc[-2])

    def test_calc_forward_returns_horizon_1(self):
        """Test horizon=1 produces only 1-day forward returns."""
        dates = pd.date_range(start='2013-01-02', periods=3, freq='D')
        sids = [1000]

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])
        log_returns = pd.Series([0.01, 0.02, 0.03], index=index)
        daily_df = pd.DataFrame({'log_ret': log_returns})

        result = calc.calc_forward_returns(daily_df, horizon=1)

        # Only cum_ret1 should exist
        assert 'cum_ret1' in result.columns
        assert 'cum_ret2' not in result.columns

        # First date should have second date's return
        assert np.isclose(result['cum_ret1'].iloc[0], 0.02, atol=1e-10)

    def test_calc_forward_returns_varying_returns(self):
        """Test with varying returns to verify cumulative calculation."""
        dates = pd.date_range(start='2013-01-02', periods=6, freq='D')
        sids = [1000]

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Returns: 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
        log_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], index=index)
        daily_df = pd.DataFrame({'log_ret': log_returns})

        result = calc.calc_forward_returns(daily_df, horizon=3)

        # First date (index 0): future returns are 0.02, 0.03, 0.04
        assert np.isclose(result['cum_ret1'].iloc[0], 0.02, atol=1e-10)
        assert np.isclose(result['cum_ret2'].iloc[0], 0.02 + 0.03, atol=1e-10)
        assert np.isclose(result['cum_ret3'].iloc[0], 0.02 + 0.03 + 0.04, atol=1e-10)

        # Second date (index 1): future returns are 0.03, 0.04, 0.05
        assert np.isclose(result['cum_ret1'].iloc[1], 0.03, atol=1e-10)
        assert np.isclose(result['cum_ret2'].iloc[1], 0.03 + 0.04, atol=1e-10)
        assert np.isclose(result['cum_ret3'].iloc[1], 0.03 + 0.04 + 0.05, atol=1e-10)


class TestMktRet:
    """Test suite for mkt_ret function."""

    def test_mkt_ret_basic(self):
        """Test basic market-cap weighted return calculation."""
        # Create simple data
        data = pd.DataFrame({
            'cum_ret1': [0.01, 0.02, 0.03],
            'mkt_cap': [100, 200, 300]
        })

        result = calc.mkt_ret(data)

        # Calculate expected: (0.01*0.1 + 0.02*0.2 + 0.03*0.3) / (0.1+0.2+0.3)
        # Weights are mkt_cap / 1e6
        weights = np.array([100, 200, 300]) / 1e6
        expected = (0.01 * weights[0] + 0.02 * weights[1] + 0.03 * weights[2]) / weights.sum()

        assert np.isclose(result, expected, atol=1e-10)

    def test_mkt_ret_equal_weights(self):
        """Test with equal market caps (should be simple average)."""
        data = pd.DataFrame({
            'cum_ret1': [0.01, 0.02, 0.03],
            'mkt_cap': [1e6, 1e6, 1e6]  # Equal weights
        })

        result = calc.mkt_ret(data)

        # With equal weights, should be simple average
        expected = (0.01 + 0.02 + 0.03) / 3.0

        assert np.isclose(result, expected, atol=1e-10)

    def test_mkt_ret_single_stock_dominates(self):
        """Test when one stock dominates market cap."""
        data = pd.DataFrame({
            'cum_ret1': [0.01, 0.10],  # Second stock has 10% return
            'mkt_cap': [1e6, 99e6]     # Second stock is 99% of market
        })

        result = calc.mkt_ret(data)

        # Result should be very close to 0.10 (dominated by large cap stock)
        assert result > 0.09
        assert result < 0.11

    def test_mkt_ret_negative_returns(self):
        """Test with negative returns."""
        data = pd.DataFrame({
            'cum_ret1': [-0.05, 0.05],
            'mkt_cap': [1e6, 1e6]  # Equal weights
        })

        result = calc.mkt_ret(data)

        # Should be zero (equal weights, opposite returns)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_mkt_ret_groupby_pattern(self):
        """Test typical usage pattern with groupby."""
        # Create multi-date data
        dates = pd.date_range(start='2013-01-02', periods=2, freq='D')
        sids = [1000, 1001]

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Date 1: stocks return 0.01, 0.02 with equal caps
        # Date 2: stocks return 0.03, 0.04 with equal caps
        data = pd.DataFrame({
            'cum_ret1': [0.01, 0.02, 0.03, 0.04],
            'mkt_cap': [1e6, 1e6, 1e6, 1e6]
        }, index=index)

        # Apply mkt_ret to each date
        result = data.groupby(level='date').apply(calc.mkt_ret)

        # Date 1 should average to 0.015
        assert np.isclose(result.iloc[0], 0.015, atol=1e-10)
        # Date 2 should average to 0.035
        assert np.isclose(result.iloc[1], 0.035, atol=1e-10)


class TestCreateZScore:
    """Test suite for create_z_score function."""

    def test_create_z_score_basic(self):
        """Test basic z-score standardization."""
        # Create data with known mean and std
        dates = pd.date_range(start='2013-01-02', periods=1, freq='D')
        sids = range(1000, 1005)

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Values: 0, 1, 2, 3, 4 (mean=2, std=sqrt(2))
        data = pd.DataFrame({
            'test_col': [0, 1, 2, 3, 4],
            'gdate': [dates[0]] * 5
        }, index=index)

        result = calc.create_z_score(data, 'test_col')

        # Check that z-score column was created
        assert 'test_col_z' in result.columns

        # Check that mean is approximately 0
        assert np.isclose(result['test_col_z'].mean(), 0.0, atol=1e-10)

        # Check that std is approximately 1
        assert np.isclose(result['test_col_z'].std(), 1.0, atol=1e-10)

    def test_create_z_score_by_date(self):
        """Test that z-scores are calculated within each date."""
        # Create two dates with different distributions
        dates = pd.date_range(start='2013-01-02', periods=2, freq='D')
        sids = range(1000, 1005)

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Date 1: values 0-4 (mean=2)
        # Date 2: values 10-14 (mean=12)
        data = pd.DataFrame({
            'test_col': list(range(5)) + list(range(10, 15)),
            'gdate': [dates[0]] * 5 + [dates[1]] * 5
        }, index=index)

        result = calc.create_z_score(data, 'test_col')

        # Each date should have mean ~0, std ~1 separately
        date1_z = result.loc[result['gdate'] == dates[0], 'test_col_z']
        date2_z = result.loc[result['gdate'] == dates[1], 'test_col_z']

        assert np.isclose(date1_z.mean(), 0.0, atol=1e-10)
        assert np.isclose(date2_z.mean(), 0.0, atol=1e-10)
        assert np.isclose(date1_z.std(), 1.0, atol=1e-10)
        assert np.isclose(date2_z.std(), 1.0, atol=1e-10)

    def test_create_z_score_preserves_ranking(self):
        """Test that z-score transformation preserves ranking."""
        dates = pd.date_range(start='2013-01-02', periods=1, freq='D')
        sids = range(1000, 1006)

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Random values
        np.random.seed(42)
        values = np.random.randn(6)

        data = pd.DataFrame({
            'test_col': values,
            'gdate': [dates[0]] * 6
        }, index=index)

        result = calc.create_z_score(data, 'test_col')

        # Check that ranking is preserved
        original_rank = data['test_col'].rank()
        zscore_rank = result['test_col_z'].rank()

        assert (original_rank == zscore_rank).all()

    def test_create_z_score_extreme_values(self):
        """Test z-score calculation with extreme values."""
        dates = pd.date_range(start='2013-01-02', periods=1, freq='D')
        sids = range(1000, 1005)

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        # Values with one extreme outlier
        data = pd.DataFrame({
            'test_col': [1, 1, 1, 1, 100],
            'gdate': [dates[0]] * 5
        }, index=index)

        result = calc.create_z_score(data, 'test_col')

        # The outlier should have high positive z-score
        assert result['test_col_z'].iloc[-1] > 1.0

        # Other values should have negative z-scores
        assert (result['test_col_z'].iloc[:-1] < 0).all()


class TestCalcPriceExtras:
    """Test suite for calc_price_extras function."""

    def test_calc_price_extras_basic(self):
        """Test basic calculation of volatility and volume ratios."""
        dates = pd.date_range(start='2013-01-02', periods=5, freq='D')
        sids = [1000]

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        data = pd.DataFrame({
            'volat_21': [0.20, 0.22, 0.21, 0.23, 0.24],
            'volat_60': [0.20, 0.20, 0.20, 0.20, 0.20],
            'tradable_volume': [100000] * 5,
            'comp_volume': [200000] * 5,
            'shares_out': [1000000] * 5
        }, index=index)

        result = calc.calc_price_extras(data)

        # Check columns created
        assert 'volat_ratio' in result.columns
        assert 'volume_ratio' in result.columns
        assert 'volat_move' in result.columns

        # Check volatility ratio
        assert np.isclose(result['volat_ratio'].iloc[0], 1.0, atol=1e-10)
        assert np.isclose(result['volat_ratio'].iloc[1], 1.1, atol=1e-10)

        # Check volume ratio (should use comp_volume in final calculation)
        assert np.isclose(result['volume_ratio'].iloc[0], 0.5, atol=1e-10)

    def test_calc_price_extras_volat_move(self):
        """Test volatility move calculation (day-over-day change)."""
        dates = pd.date_range(start='2013-01-02', periods=5, freq='D')
        sids = [1000]

        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        data = pd.DataFrame({
            'volat_21': [0.20, 0.22, 0.21, 0.23, 0.24],
            'volat_60': [0.20] * 5,
            'tradable_volume': [100000] * 5,
            'comp_volume': [200000] * 5,
            'shares_out': [1000000] * 5
        }, index=index)

        result = calc.calc_price_extras(data)

        # First value should be NaN (no prior day)
        assert pd.isna(result['volat_move'].iloc[0])

        # Second value should be 0.02 (0.22 - 0.20)
        assert np.isclose(result['volat_move'].iloc[1], 0.02, atol=1e-10)

        # Third value should be -0.01 (0.21 - 0.22)
        assert np.isclose(result['volat_move'].iloc[2], -0.01, atol=1e-10)


class TestWinsorizeByGroup:
    """Test suite for winsorize_by_group function."""

    def test_winsorize_by_group_basic(self):
        """Test winsorization within groups."""
        # Create data with two groups
        groups = ['A', 'A', 'A', 'B', 'B', 'B']
        values = [1, 2, 100, 5, 6, 200]  # Outlier in each group

        data = pd.Series(values)
        group = pd.Series(groups)

        result = calc.winsorize_by_group(data, group)

        # Check that outliers were clipped within each group
        # Group A: outlier 100 should be clipped
        assert result.iloc[2] < 100
        # Group B: outlier 200 should be clipped
        assert result.iloc[5] < 200

    def test_winsorize_by_group_independence(self):
        """Test that groups are winsorized independently."""
        # Create data where groups have different scales
        groups = ['A'] * 10 + ['B'] * 10
        values = list(np.random.randn(10) * 100) + list(np.random.randn(10) * 1)

        data = pd.Series(values)
        group = pd.Series(groups)

        result = calc.winsorize_by_group(data, group)

        # Each group should maintain its scale
        group_a = result.iloc[:10]
        group_b = result.iloc[10:]

        # Group A should have values around 100
        assert group_a.mean() > 50 or group_a.mean() < -50
        # Group B should have values around 1
        assert abs(group_b.mean()) < 10
