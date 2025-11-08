"""Label encoding transformation."""

from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

from dataprep.core.base import Transform


class LabelEncoder(Transform):
    """Encode categorical features as integers.

    Args:
        columns: List of columns to encode (default: None = all object columns).
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """Initialize LabelEncoder transformation."""
        super().__init__()
        self.columns = columns
        self._encoders: Dict[str, SklearnLabelEncoder] = {}
        self._columns_to_encode: List[str] = []

    def fit(self, data: pd.DataFrame) -> 'LabelEncoder':
        """Fit the encoder by learning categories.

        Args:
            data: The data to fit.

        Returns:
            LabelEncoder: The fitted transformation (self).
        """
        # Determine which columns to encode
        if self.columns is not None:
            self._columns_to_encode = self.columns
        else:
            self._columns_to_encode = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Fit label encoder for each column
        for col in self._columns_to_encode:
            if col in data.columns:
                encoder = SklearnLabelEncoder()
                encoder.fit(data[col].astype(str))
                self._encoders[col] = encoder

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Label encode the data.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        result = data.copy()

        for col, encoder in self._encoders.items():
            if col in result.columns:
                # Handle unseen labels by setting them to -1
                result[col] = result[col].astype(str)
                result[col] = result[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )

        return result

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {
            'columns': self.columns,
        }
