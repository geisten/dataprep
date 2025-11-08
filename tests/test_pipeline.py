"""Tests for the Pipeline class."""

import pytest
import pandas as pd
import numpy as np
from dataprep import Pipeline
from dataprep.sources import DataFrameSource
from dataprep.transforms import Normalizer, DropMissing, FeatureSelector
from dataprep.sinks import DataFrameSink


class TestPipeline:
    """Test cases for Pipeline."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500],
        })

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        pipeline = Pipeline()
        assert len(pipeline) == 0

        pipeline = Pipeline([Normalizer()])
        assert len(pipeline) == 1

    def test_add_transform(self):
        """Test adding transformations to pipeline."""
        pipeline = Pipeline()
        pipeline.add_transform(Normalizer())
        assert len(pipeline) == 1

        pipeline.add_transform(DropMissing())
        assert len(pipeline) == 2

    def test_fit_transform(self):
        """Test fitting and transforming data."""
        pipeline = Pipeline([
            Normalizer(method='standard')
        ])

        result = pipeline.fit_transform(self.data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.data.shape

        # Check that data is normalized (mean ~0, std ~1)
        assert np.allclose(result.mean(), 0, atol=1e-10)
        assert np.allclose(result.std(), 1, atol=1e-10)

    def test_run_with_source_sink(self):
        """Test running pipeline with source and sink."""
        pipeline = Pipeline([
            FeatureSelector(features=['a', 'b'])
        ])

        source = DataFrameSource(self.data)
        sink = DataFrameSink()

        pipeline.run(source, sink)

        result = sink.get_data()
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['a', 'b']
        assert len(result) == len(self.data)

    def test_pipeline_metadata(self):
        """Test pipeline metadata."""
        pipeline = Pipeline([
            Normalizer(),
            DropMissing()
        ])

        metadata = pipeline.get_metadata()
        assert metadata['num_transforms'] == 2
        assert metadata['fitted'] == False

        pipeline.fit(self.data)
        metadata = pipeline.get_metadata()
        assert metadata['fitted'] == True

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        pipeline = Pipeline([
            Normalizer(),
            DropMissing()
        ])

        repr_str = repr(pipeline)
        assert 'Normalizer' in repr_str
        assert 'DropMissing' in repr_str
