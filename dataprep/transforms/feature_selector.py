"""Feature selection transformation."""

from typing import Any, Dict, List, Optional
import pandas as pd

from dataprep.core.base import Transform


class FeatureSelector(Transform):
    """Select or drop specific features from the dataset.

    Args:
        features: List of features to select.
        drop: If True, drop the specified features instead of selecting them (default: False).
    """

    def __init__(self, features: List[str], drop: bool = False):
        """Initialize FeatureSelector transformation."""
        super().__init__()
        self.features = features
        self.drop = drop

    def fit(self, data: pd.DataFrame) -> 'FeatureSelector':
        """Fit the transformation (no-op for FeatureSelector).

        Args:
            data: The data to fit.

        Returns:
            FeatureSelector: The fitted transformation (self).
        """
        # Validate that features exist in the data
        missing_features = set(self.features) - set(data.columns)
        if missing_features and not self.drop:
            raise ValueError(f"Features not found in data: {missing_features}")

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select or drop features from the data.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        if self.drop:
            # Drop specified features
            existing_features = [f for f in self.features if f in data.columns]
            return data.drop(columns=existing_features)
        else:
            # Select specified features
            existing_features = [f for f in self.features if f in data.columns]
            return data[existing_features]

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {
            'features': self.features,
            'drop': self.drop,
        }
