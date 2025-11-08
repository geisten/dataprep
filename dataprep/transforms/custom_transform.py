"""Custom transformation using user-defined functions."""

from typing import Any, Callable, Dict, Optional
import pandas as pd

from dataprep.core.base import Transform


class CustomTransform(Transform):
    """Apply a custom transformation function to the data.

    This allows users to easily extend the pipeline with their own
    transformation logic without creating a new class.

    Args:
        transform_func: Function that takes a DataFrame and returns a transformed DataFrame.
        fit_func: Optional function for fitting (default: None).
        name: Optional name for the transformation (default: 'CustomTransform').
    """

    def __init__(
        self,
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        fit_func: Optional[Callable[[pd.DataFrame], Any]] = None,
        name: str = 'CustomTransform'
    ):
        """Initialize CustomTransform."""
        super().__init__()
        self.transform_func = transform_func
        self.fit_func = fit_func
        self.name = name
        self._fit_result: Any = None

    def fit(self, data: pd.DataFrame) -> 'CustomTransform':
        """Fit the transformation using the custom fit function.

        Args:
            data: The data to fit.

        Returns:
            CustomTransform: The fitted transformation (self).
        """
        if self.fit_func is not None:
            self._fit_result = self.fit_func(data)

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the custom transform function.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        return self.transform_func(data)

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {
            'name': self.name,
        }

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            str: String representation.
        """
        return self.name
