"""Tests for data sources and sinks."""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from dataprep.sources import CSVSource, JSONSource, DataFrameSource
from dataprep.sinks import CSVSink, JSONSink, DataFrameSink


class TestCSVSource:
    """Test cases for CSVSource."""

    def setup_method(self):
        """Set up test data and temporary file."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def teardown_method(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)

    def test_read_csv(self):
        """Test reading CSV file."""
        source = CSVSource(self.temp_file.name)
        result = source.read()

        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.data.shape
        assert list(result.columns) == list(self.data.columns)

    def test_validate(self):
        """Test validation."""
        source = CSVSource(self.temp_file.name)
        assert source.validate() == True

        source = CSVSource('nonexistent.csv')
        assert source.validate() == False


class TestJSONSource:
    """Test cases for JSONSource."""

    def setup_method(self):
        """Set up test data and temporary file."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.data.to_json(self.temp_file.name, orient='records')
        self.temp_file.close()

    def teardown_method(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)

    def test_read_json(self):
        """Test reading JSON file."""
        source = JSONSource(self.temp_file.name, orient='records')
        result = source.read()

        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.data.shape


class TestDataFrameSource:
    """Test cases for DataFrameSource."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })

    def test_read_dataframe(self):
        """Test reading from DataFrame."""
        source = DataFrameSource(self.data)
        result = source.read()

        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.data.shape
        # Should be a copy, not the same object
        assert result is not self.data

    def test_validate(self):
        """Test validation."""
        source = DataFrameSource(self.data)
        assert source.validate() == True


class TestCSVSink:
    """Test cases for CSVSink."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.temp_file.close()

    def teardown_method(self):
        """Clean up temporary file."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_write_csv(self):
        """Test writing to CSV file."""
        sink = CSVSink(self.temp_file.name)
        sink.write(self.data)

        # Read back and verify
        result = pd.read_csv(self.temp_file.name)
        assert result.shape == self.data.shape
        assert list(result.columns) == list(self.data.columns)


class TestJSONSink:
    """Test cases for JSONSink."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()

    def teardown_method(self):
        """Clean up temporary file."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_write_json(self):
        """Test writing to JSON file."""
        sink = JSONSink(self.temp_file.name, orient='records')
        sink.write(self.data)

        # Read back and verify
        result = pd.read_json(self.temp_file.name, orient='records')
        assert result.shape == self.data.shape


class TestDataFrameSink:
    """Test cases for DataFrameSink."""

    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })

    def test_write_and_get_data(self):
        """Test writing and retrieving data."""
        sink = DataFrameSink()
        sink.write(self.data)

        result = sink.get_data()
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.data.shape
        # Should be a copy
        assert result is not self.data

    def test_get_data_before_write(self):
        """Test getting data before writing."""
        sink = DataFrameSink()

        with pytest.raises(ValueError):
            sink.get_data()
