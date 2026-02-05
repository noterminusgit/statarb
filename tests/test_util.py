"""
Unit tests for util.py module.

This module tests utility functions used throughout the statistical arbitrage
system, with focus on data merging, filtering, and temporal alignment.

Critical tests:
    - merge_barra_data: Verifies 1-day lag to prevent look-ahead bias
    - filter_expandable/filter_pca: Verifies universe filtering logic
    - merge functions: Verifies data alignment and duplicate handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import functions from util module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import util


class TestMergeBarraData:
    """Test suite for merge_barra_data function."""

    def test_merge_barra_data_basic(self, sample_price_df, sample_barra_df):
        """Test basic merge functionality."""
        result = util.merge_barra_data(sample_price_df, sample_barra_df)

        # Check that merge happened
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check that both price and barra columns are present
        assert 'close' in result.columns
        assert 'beta' in result.columns

        # Check index structure
        assert result.index.names == ['date', 'sid']

    def test_merge_barra_data_one_day_lag(self, sample_price_df, sample_barra_df):
        """
        CRITICAL: Verify Barra data is lagged by 1 day (no look-ahead bias).

        Barra data for date T should not be available until end of day T,
        so we use T-1 data for trading decisions on day T.
        """
        # Get unique dates
        dates = sample_price_df.index.get_level_values('date').unique()

        # For testing, we need at least 2 dates
        assert len(dates) >= 2

        # Create simple test data where we can track the lag
        test_price = sample_price_df.copy()
        test_barra = sample_barra_df.copy()

        # Set distinct values in barra data to track the lag
        # For each date, set beta = date's day of month
        for date in dates:
            date_mask = test_barra.index.get_level_values('date') == date
            test_barra.loc[date_mask, 'beta'] = float(date.day)

        # Merge with lagging
        result = util.merge_barra_data(test_price, test_barra)

        # Check the lag: Barra data on date T should have beta from date T-1
        # First date should have NaN (no prior data)
        first_date = dates[0]
        first_date_mask = result.index.get_level_values('date') == first_date
        assert result.loc[first_date_mask, 'beta'].isna().all(), \
            "First date should have NaN Barra data (no prior day)"

        # Second date should have beta from first date
        if len(dates) >= 2:
            second_date = dates[1]
            second_date_mask = result.index.get_level_values('date') == second_date
            expected_beta = float(first_date.day)
            actual_beta = result.loc[second_date_mask, 'beta'].iloc[0]
            assert actual_beta == expected_beta, \
                f"Barra data should be lagged by 1 day: expected {expected_beta}, got {actual_beta}"

    def test_merge_barra_data_no_duplicate_columns(self, sample_price_df, sample_barra_df):
        """Verify no '_dead' suffix columns remain after merge."""
        result = util.merge_barra_data(sample_price_df, sample_barra_df)

        # Check that no columns end with '_dead'
        dead_cols = [col for col in result.columns if col.endswith('_dead')]
        assert len(dead_cols) == 0, f"Found duplicate '_dead' columns: {dead_cols}"

    def test_merge_barra_data_empty_barra(self, sample_price_df):
        """Test merge with empty Barra DataFrame."""
        # Create empty Barra DataFrame with correct structure
        empty_barra = pd.DataFrame(
            columns=['beta', 'momentum'],
            index=pd.MultiIndex.from_tuples([], names=['date', 'sid'])
        )

        result = util.merge_barra_data(sample_price_df, empty_barra)

        # Should still have price data
        assert len(result) == len(sample_price_df)
        assert 'close' in result.columns


class TestRemoveDupCols:
    """Test suite for remove_dup_cols function."""

    def test_remove_dup_cols_basic(self):
        """Test removal of columns ending with '_dead'."""
        df = pd.DataFrame({
            'close': [1, 2, 3],
            'close_dead': [4, 5, 6],
            'volume': [100, 200, 300],
            'volume_dead': [400, 500, 600],
            'beta': [1.0, 1.1, 1.2]
        })

        result = util.remove_dup_cols(df)

        # Check that '_dead' columns are removed
        assert 'close' in result.columns
        assert 'close_dead' not in result.columns
        assert 'volume' in result.columns
        assert 'volume_dead' not in result.columns
        assert 'beta' in result.columns

        # Check that data is preserved
        assert len(result) == 3
        assert list(result['close']) == [1, 2, 3]

    def test_remove_dup_cols_no_dead_columns(self):
        """Test with DataFrame that has no '_dead' columns."""
        df = pd.DataFrame({
            'close': [1, 2, 3],
            'volume': [100, 200, 300]
        })

        result = util.remove_dup_cols(df)

        # Should return unchanged
        assert list(result.columns) == ['close', 'volume']
        assert len(result) == 3

    def test_remove_dup_cols_preserves_index(self):
        """Test that MultiIndex is preserved."""
        dates = pd.date_range('2013-01-01', periods=3)
        sids = [1000, 1001, 1002]
        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        df = pd.DataFrame({
            'close': range(9),
            'close_dead': range(9, 18)
        }, index=index[:9])

        result = util.remove_dup_cols(df)

        assert result.index.names == ['date', 'sid']
        assert len(result) == 9


class TestFilterExpandable:
    """Test suite for filter_expandable function."""

    def test_filter_expandable_basic(self):
        """Test basic filtering of expandable universe."""
        # Create test DataFrame with expandable column
        df = pd.DataFrame({
            'sid': [1000, 1001, 1002, 1003, 1004],
            'close': [50, 60, 70, 80, 90],
            'expandable': [True, True, False, True, False]
        })

        result = util.filter_expandable(df)

        # Should keep only expandable=True rows
        assert len(result) == 3
        assert all(result['expandable'] == True)
        assert list(result['sid']) == [1000, 1001, 1003]

    def test_filter_expandable_removes_nan(self):
        """Test that rows with NaN expandable are removed."""
        df = pd.DataFrame({
            'sid': [1000, 1001, 1002, 1003],
            'close': [50, 60, 70, 80],
            'expandable': [True, np.nan, False, True]
        })

        result = util.filter_expandable(df)

        # Should keep only non-NaN and True
        assert len(result) == 2
        assert list(result['sid']) == [1000, 1003]

    def test_filter_expandable_all_false(self):
        """Test with all expandable=False."""
        df = pd.DataFrame({
            'sid': [1000, 1001],
            'close': [50, 60],
            'expandable': [False, False]
        })

        result = util.filter_expandable(df)

        # Should return empty DataFrame
        assert len(result) == 0

    def test_filter_expandable_preserves_columns(self):
        """Test that all columns are preserved."""
        df = pd.DataFrame({
            'sid': [1000, 1001, 1002],
            'close': [50, 60, 70],
            'volume': [100000, 200000, 300000],
            'expandable': [True, False, True]
        })

        result = util.filter_expandable(df)

        # Check columns are preserved
        assert 'close' in result.columns
        assert 'volume' in result.columns
        assert 'expandable' in result.columns


class TestFilterPCA:
    """Test suite for filter_pca function."""

    def test_filter_pca_basic(self):
        """Test basic filtering by market cap threshold."""
        # Create test DataFrame with varying market caps
        df = pd.DataFrame({
            'sid': [1000, 1001, 1002, 1003, 1004],
            'close': [50, 60, 70, 80, 90],
            'mkt_cap': [5e9, 15e9, 8e9, 25e9, 12e9]  # Mix above/below 10B threshold
        })

        result = util.filter_pca(df)

        # Should keep only mkt_cap > 10B
        assert len(result) == 3
        assert all(result['mkt_cap'] > 1e10)
        assert list(result['sid']) == [1001, 1003, 1004]

    def test_filter_pca_threshold_boundary(self):
        """Test exact boundary condition at 10B market cap."""
        df = pd.DataFrame({
            'sid': [1000, 1001, 1002],
            'close': [50, 60, 70],
            'mkt_cap': [9.9e9, 1e10, 10.1e9]  # Just below, at, just above threshold
        })

        result = util.filter_pca(df)

        # Should exclude exactly 10B (> not >=)
        assert len(result) == 1
        assert result.iloc[0]['sid'] == 1002
        assert result.iloc[0]['mkt_cap'] > 1e10

    def test_filter_pca_all_below_threshold(self):
        """Test with all stocks below market cap threshold."""
        df = pd.DataFrame({
            'sid': [1000, 1001],
            'close': [50, 60],
            'mkt_cap': [5e9, 8e9]  # All below 10B
        })

        result = util.filter_pca(df)

        # Should return empty DataFrame
        assert len(result) == 0

    def test_filter_pca_with_nan_mkt_cap(self):
        """Test behavior with NaN market cap values."""
        df = pd.DataFrame({
            'sid': [1000, 1001, 1002],
            'close': [50, 60, 70],
            'mkt_cap': [15e9, np.nan, 20e9]
        })

        result = util.filter_pca(df)

        # NaN comparison should filter out NaN row
        assert len(result) == 2
        assert list(result['sid']) == [1000, 1002]


class TestGetOverlappingCols:
    """Test suite for get_overlapping_cols function."""

    def test_get_overlapping_cols_basic(self):
        """Test basic column set difference."""
        df1 = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4],
            'c': [5, 6]
        })

        df2 = pd.DataFrame({
            'b': [7, 8],
            'd': [9, 10]
        })

        result = util.get_overlapping_cols(df1, df2)

        # Should return columns in df1 but not in df2
        assert set(result) == {'a', 'c'}

    def test_get_overlapping_cols_no_overlap(self):
        """Test when DataFrames have completely different columns."""
        df1 = pd.DataFrame({'a': [1], 'b': [2]})
        df2 = pd.DataFrame({'c': [3], 'd': [4]})

        result = util.get_overlapping_cols(df1, df2)

        # All df1 columns should be returned
        assert set(result) == {'a', 'b'}

    def test_get_overlapping_cols_complete_overlap(self):
        """Test when all df1 columns are in df2."""
        df1 = pd.DataFrame({'a': [1], 'b': [2]})
        df2 = pd.DataFrame({'a': [3], 'b': [4], 'c': [5]})

        result = util.get_overlapping_cols(df1, df2)

        # No unique columns in df1
        assert len(result) == 0

    def test_get_overlapping_cols_empty_df2(self):
        """Test with empty second DataFrame."""
        df1 = pd.DataFrame({'a': [1], 'b': [2]})
        df2 = pd.DataFrame()

        result = util.get_overlapping_cols(df1, df2)

        # All df1 columns should be returned
        assert set(result) == {'a', 'b'}


class TestMergeDailyCalcs:
    """Test suite for merge_daily_calcs function."""

    def test_merge_daily_calcs_basic(self):
        """Test basic merge of daily calculations."""
        # Create base DataFrame
        dates = pd.date_range('2013-01-01', periods=3)
        sids = [1000, 1001]
        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        full_df = pd.DataFrame({
            'close': range(6),
            'volume': range(100, 106)
        }, index=index)

        # Create result DataFrame with new column
        result_df = pd.DataFrame({
            'close': range(6),  # Overlapping column
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # New column
        }, index=index)

        merged = util.merge_daily_calcs(full_df, result_df)

        # Check that new column is added
        assert 'alpha' in merged.columns
        assert 'close' in merged.columns
        assert 'volume' in merged.columns

        # Check index is preserved
        assert merged.index.names == ['date', 'sid']
        assert len(merged) == 6

    def test_merge_daily_calcs_preserves_all_rows(self):
        """Test that left join preserves all rows from full_df."""
        dates = pd.date_range('2013-01-01', periods=2)
        sids = [1000, 1001]
        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        full_df = pd.DataFrame({
            'close': range(4)
        }, index=index)

        # result_df has fewer rows
        partial_index = pd.MultiIndex.from_tuples(
            [(dates[0], 1000), (dates[0], 1001)],
            names=['date', 'sid']
        )
        result_df = pd.DataFrame({
            'alpha': [0.1, 0.2]
        }, index=partial_index)

        merged = util.merge_daily_calcs(full_df, result_df)

        # Should preserve all 4 rows from full_df
        assert len(merged) == 4

        # Alpha should have NaN for missing rows
        assert merged['alpha'].isna().sum() == 2

    def test_merge_daily_calcs_no_duplicate_columns(self):
        """Test that duplicate columns are not created."""
        dates = pd.date_range('2013-01-01', periods=2)
        sids = [1000, 1001]
        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        full_df = pd.DataFrame({
            'close': range(4),
            'volume': range(100, 104)
        }, index=index)

        result_df = pd.DataFrame({
            'close': range(4),
            'volume': range(100, 104),
            'alpha': [0.1, 0.2, 0.3, 0.4]
        }, index=index)

        merged = util.merge_daily_calcs(full_df, result_df)

        # Check no duplicate columns
        assert 'close_dead' not in merged.columns
        assert 'volume_dead' not in merged.columns

        # Original columns should exist once
        assert merged.columns.tolist().count('close') == 1
        assert merged.columns.tolist().count('volume') == 1


class TestMergeIntraCalcs:
    """Test suite for merge_intra_calcs function."""

    def test_merge_intra_calcs_basic(self):
        """Test basic merge of intraday calculations."""
        # Create intraday index
        timestamps = pd.date_range('2013-01-01 09:30', periods=3, freq='30min')
        sids = [1000, 1001]
        index = pd.MultiIndex.from_product([timestamps, sids], names=['iclose_ts', 'sid'])

        full_df = pd.DataFrame({
            'iclose': range(6),
            'ivol': range(100, 106)
        }, index=index)

        # Create result DataFrame with date column and new alpha
        result_df = pd.DataFrame({
            'date': pd.Timestamp('2013-01-01'),
            'bar_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        }, index=index)

        merged = util.merge_intra_calcs(full_df, result_df)

        # Check that new column is added but date is removed
        assert 'bar_alpha' in merged.columns
        assert 'date' not in merged.columns  # Date should be deleted
        assert 'iclose' in merged.columns

        # Check index is preserved
        assert merged.index.names == ['iclose_ts', 'sid']

    def test_merge_intra_calcs_removes_date_column(self):
        """Test that date column is explicitly removed from result_df."""
        timestamps = pd.date_range('2013-01-01 09:30', periods=2, freq='30min')
        sids = [1000]
        index = pd.MultiIndex.from_product([timestamps, sids], names=['iclose_ts', 'sid'])

        full_df = pd.DataFrame({
            'iclose': range(2)
        }, index=index)

        result_df = pd.DataFrame({
            'date': pd.Timestamp('2013-01-01'),
            'alpha': [0.1, 0.2]
        }, index=index)

        merged = util.merge_intra_calcs(full_df, result_df)

        # Date column should be removed to avoid NaT issues
        assert 'date' not in merged.columns

    def test_merge_intra_calcs_preserves_index(self):
        """Test that intraday MultiIndex is preserved."""
        timestamps = pd.date_range('2013-01-01 09:30', periods=3, freq='30min')
        sids = [1000, 1001]
        index = pd.MultiIndex.from_product([timestamps, sids], names=['iclose_ts', 'sid'])

        full_df = pd.DataFrame({'iclose': range(6)}, index=index)
        result_df = pd.DataFrame({
            'date': pd.Timestamp('2013-01-01'),
            'alpha': range(6)
        }, index=index)

        merged = util.merge_intra_calcs(full_df, result_df)

        # Check index structure
        assert merged.index.names == ['iclose_ts', 'sid']
        assert len(merged) == 6


class TestDfDates:
    """Test suite for df_dates function."""

    def test_df_dates_basic(self):
        """Test date range string extraction."""
        dates = pd.date_range('2013-01-15', periods=3)
        sids = [1000, 1001]
        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        df = pd.DataFrame({'close': range(6)}, index=index)

        result = util.df_dates(df)

        # Should return "YYYYMMDD-YYYYMMDD" format
        assert result == "20130115-20130117"

    def test_df_dates_single_date(self):
        """Test with single date."""
        date = pd.Timestamp('2013-06-30')
        sids = [1000, 1001]
        index = pd.MultiIndex.from_product([[date], sids], names=['date', 'sid'])

        df = pd.DataFrame({'close': [1, 2]}, index=index)

        result = util.df_dates(df)

        # Start and end should be the same
        assert result == "20130630-20130630"

    def test_df_dates_year_boundary(self):
        """Test date formatting across year boundary."""
        dates = [pd.Timestamp('2012-12-31'), pd.Timestamp('2013-01-02')]
        sids = [1000]
        index = pd.MultiIndex.from_product([dates, sids], names=['date', 'sid'])

        df = pd.DataFrame({'close': [1, 2]}, index=index)

        result = util.df_dates(df)

        assert result == "20121231-20130102"


class TestMkdirP:
    """Test suite for mkdir_p function."""

    def test_mkdir_p_creates_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "test_dir"
        util.mkdir_p(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_mkdir_p_creates_nested_directories(self, tmp_path):
        """Test creating nested directories (like mkdir -p)."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        util.mkdir_p(str(nested_dir))

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_mkdir_p_existing_directory(self, tmp_path):
        """Test that existing directory doesn't raise error."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        # Should not raise exception
        util.mkdir_p(str(existing_dir))

        assert existing_dir.exists()

    def test_mkdir_p_file_exists_raises_error(self, tmp_path):
        """Test that trying to create directory where file exists raises error."""
        # Create a file
        file_path = tmp_path / "testfile.txt"
        file_path.write_text("test")

        # Trying to create directory with same name should fail
        with pytest.raises(Exception) as exc_info:
            util.mkdir_p(str(file_path))

        assert "Could not create" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_merge_barra_data_with_nan_values(self, sample_price_df):
        """Test merge_barra_data handles NaN values properly."""
        # Create Barra data with some NaN values
        barra_df = sample_price_df.copy()[['close']]
        barra_df.columns = ['beta']
        barra_df.iloc[::2] = np.nan  # Set every other row to NaN

        result = util.merge_barra_data(sample_price_df, barra_df)

        # Should complete without error
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_filter_expandable_empty_dataframe(self):
        """Test filter_expandable with empty DataFrame."""
        df = pd.DataFrame(columns=['sid', 'close', 'expandable'])

        result = util.filter_expandable(df)

        # Should return empty DataFrame without error
        assert len(result) == 0
        assert 'expandable' in result.columns

    def test_filter_pca_empty_dataframe(self):
        """Test filter_pca with empty DataFrame."""
        df = pd.DataFrame(columns=['sid', 'close', 'mkt_cap'])

        result = util.filter_pca(df)

        # Should return empty DataFrame without error
        assert len(result) == 0
        assert 'mkt_cap' in result.columns

    def test_remove_dup_cols_empty_dataframe(self):
        """Test remove_dup_cols with empty DataFrame."""
        df = pd.DataFrame(columns=['close', 'close_dead'])

        result = util.remove_dup_cols(df)

        # Should remove _dead column even from empty DataFrame
        assert 'close' in result.columns
        assert 'close_dead' not in result.columns
        assert len(result) == 0
