"""Base classes for data sources, sinks, and transformations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class DataSource(ABC):
    """Abstract base class for data sources.

    Data sources are responsible for reading data from various sources
    like files, databases, APIs, etc.
    """

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Read data from the source.

        Returns:
            pd.DataFrame: The data read from the source.
        """
        pass

    def validate(self) -> bool:
        """Validate that the source is accessible and valid.

        Returns:
            bool: True if the source is valid, False otherwise.
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        return {}


class DataSink(ABC):
    """Abstract base class for data sinks.

    Data sinks are responsible for writing processed data to various
    destinations like files, databases, etc.
    """

    @abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """Write data to the sink.

        Args:
            data: The data to write.
        """
        pass

    def validate(self) -> bool:
        """Validate that the sink is accessible and valid.

        Returns:
            bool: True if the sink is valid, False otherwise.
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data sink.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        return {}


class Transform(ABC):
    """Abstract base class for data transformations.

    Transformations modify the data in some way, such as normalization,
    feature engineering, cleaning, etc.
    """

    def __init__(self):
        """Initialize the transformation."""
        self._fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'Transform':
        """Fit the transformation to the data.

        This method should learn any parameters needed for the transformation.
        For stateless transformations, this can simply return self.

        Args:
            data: The data to fit the transformation to.

        Returns:
            Transform: The fitted transformation (self).
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformation and transform the data in one step.

        Args:
            data: The data to fit and transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        return self.fit(data).transform(data)

    def is_fitted(self) -> bool:
        """Check if the transformation has been fitted.

        Returns:
            bool: True if fitted, False otherwise.
        """
        return self._fitted

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {}

    def set_params(self, **params: Any) -> 'Transform':
        """Set parameters of the transformation.

        Args:
            **params: Parameters to set.

        Returns:
            Transform: The transformation (self).
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
