"""Tests for transformation classes."""

import pytest
import pandas as pd
import numpy as np
from dataprep.transforms import (
    Normalizer,
    DropMissing,
    FillMissing,
    FeatureSelector,
    OneHotEncoder,
    OutlierRemover,
    CustomTransform
)


class TestNormalizer:
    """Test cases for Normalizer."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
        })

    def test_standard_normalization(self):
        """Test standard normalization."""
        normalizer = Normalizer(method='standard')
        result = normalizer.fit_transform(self.data)

        # Check mean is 0 and std is 1
        assert np.allclose(result.mean(), 0, atol=1e-10)
        assert np.allclose(result.std(), 1, atol=1e-10)

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        normalizer = Normalizer(method='minmax')
        result = normalizer.fit_transform(self.data)

        # Check values are between 0 and 1
        assert result.min().min() >= 0
        assert result.max().max() <= 1


class TestDropMissing:
    """Test cases for DropMissing."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': [10, None, None, 40, 50],
            'c': [100, 200, 300, 400, 500],
        })

    def test_drop_rows(self):
        """Test dropping rows with missing values."""
        transform = DropMissing(axis=0)
        result = transform.fit_transform(self.data)

        assert len(result) == 3  # Only 3 rows have no missing values
        assert result.isnull().sum().sum() == 0

    def test_drop_with_threshold(self):
        """Test dropping with threshold."""
        transform = DropMissing(axis=0, threshold=0.5)
        result = transform.fit_transform(self.data)

        # Rows with less than 50% non-missing values are dropped
        assert result.isnull().sum().sum() <= self.data.isnull().sum().sum()


class TestFillMissing:
    """Test cases for FillMissing."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': [10, None, 30, 40, 50],
        })

    def test_fill_with_mean(self):
        """Test filling with mean."""
        transform = FillMissing(strategy='mean')
        result = transform.fit_transform(self.data)

        assert result.isnull().sum().sum() == 0
        # Check that missing value in 'a' is filled with mean
        assert result.loc[2, 'a'] == self.data['a'].mean()

    def test_fill_with_constant(self):
        """Test filling with constant."""
        transform = FillMissing(strategy='constant', value=999)
        result = transform.fit_transform(self.data)

        assert result.isnull().sum().sum() == 0
        assert result.loc[2, 'a'] == 999


class TestFeatureSelector:
    """Test cases for FeatureSelector."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })

    def test_select_features(self):
        """Test selecting features."""
        transform = FeatureSelector(features=['a', 'c'])
        result = transform.fit_transform(self.data)

        assert list(result.columns) == ['a', 'c']
        assert len(result) == len(self.data)

    def test_drop_features(self):
        """Test dropping features."""
        transform = FeatureSelector(features=['b'], drop=True)
        result = transform.fit_transform(self.data)

        assert list(result.columns) == ['a', 'c']


class TestOneHotEncoder:
    """Test cases for OneHotEncoder."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'value': [1, 2, 3, 4],
        })

    def test_one_hot_encoding(self):
        """Test one-hot encoding."""
        transform = OneHotEncoder(columns=['category'])
        result = transform.fit_transform(self.data)

        # Original column should be removed
        assert 'category' not in result.columns
        # New columns should be created
        assert any('category_' in col for col in result.columns)
        # Value column should still exist
        assert 'value' in result.columns


class TestOutlierRemover:
    """Test cases for OutlierRemover."""

    def setup_method(self):
        """Set up test data with outliers."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'b': [10, 20, 30, 40, 50, 60],
        })

    def test_remove_outliers_iqr(self):
        """Test removing outliers with IQR method."""
        transform = OutlierRemover(method='iqr', threshold=1.5)
        result = transform.fit_transform(self.data)

        # The row with value 100 should be removed
        assert len(result) < len(self.data)
        assert 100 not in result['a'].values


class TestCustomTransform:
    """Test cases for CustomTransform."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })

    def test_custom_transform(self):
        """Test custom transformation."""
        def double_values(df):
            return df * 2

        transform = CustomTransform(double_values, name='DoubleValues')
        result = transform.fit_transform(self.data)

        assert (result == self.data * 2).all().all()

    def test_custom_transform_with_fit(self):
        """Test custom transformation with fit function."""
        def fit_func(df):
            return df.mean()

        def transform_func(df):
            return df - df.mean()

        transform = CustomTransform(transform_func, fit_func=fit_func)
        result = transform.fit_transform(self.data)

        # Check that result is centered around 0
        assert np.allclose(result.mean(), 0, atol=1e-10)
