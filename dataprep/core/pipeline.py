"""Pipeline implementation for data preprocessing."""

from typing import List, Optional, Dict, Any
import pandas as pd
import yaml
from pathlib import Path

from dataprep.core.base import DataSource, DataSink, Transform


class Pipeline:
    """Data preprocessing pipeline.

    A pipeline chains together multiple transformations and handles
    the flow of data from source to sink.
    """

    def __init__(self, transforms: Optional[List[Transform]] = None):
        """Initialize the pipeline.

        Args:
            transforms: List of transformations to apply.
        """
        self.transforms = transforms or []
        self._fitted = False
        self._metadata: Dict[str, Any] = {}

    def add_transform(self, transform: Transform) -> 'Pipeline':
        """Add a transformation to the pipeline.

        Args:
            transform: The transformation to add.

        Returns:
            Pipeline: The pipeline (self) for method chaining.
        """
        self.transforms.append(transform)
        return self

    def fit(self, data: pd.DataFrame) -> 'Pipeline':
        """Fit all transformations in the pipeline.

        Args:
            data: The data to fit the transformations to.

        Returns:
            Pipeline: The fitted pipeline (self).
        """
        current_data = data.copy()

        for i, transform in enumerate(self.transforms):
            print(f"Fitting transform {i + 1}/{len(self.transforms)}: {transform.__class__.__name__}")
            transform.fit(current_data)
            current_data = transform.transform(current_data)

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted pipeline.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.

        Raises:
            RuntimeError: If the pipeline has not been fitted.
        """
        if not self._fitted and self.transforms:
            raise RuntimeError("Pipeline must be fitted before transform. Call fit() first.")

        current_data = data.copy()

        for i, transform in enumerate(self.transforms):
            print(f"Applying transform {i + 1}/{len(self.transforms)}: {transform.__class__.__name__}")
            current_data = transform.transform(current_data)

        return current_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data in one step.

        Args:
            data: The data to fit and transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        return self.fit(data).transform(data)

    def run(
        self,
        source: DataSource,
        sink: DataSink,
        fit: bool = True,
    ) -> None:
        """Run the complete pipeline from source to sink.

        Args:
            source: The data source to read from.
            sink: The data sink to write to.
            fit: Whether to fit the transformations (default: True).
        """
        print(f"Reading data from {source.__class__.__name__}...")
        data = source.read()
        print(f"Read {len(data)} rows")

        if fit:
            print("Fitting and transforming data...")
            transformed_data = self.fit_transform(data)
        else:
            print("Transforming data...")
            transformed_data = self.transform(data)

        print(f"Writing {len(transformed_data)} rows to {sink.__class__.__name__}...")
        sink.write(transformed_data)
        print("Pipeline completed successfully!")

    def execute(
        self,
        source: Optional[DataSource] = None,
        sink: Optional[DataSink] = None,
    ) -> pd.DataFrame:
        """Execute the pipeline and optionally return the result.

        Args:
            source: Optional data source. If provided, data will be read from it.
            sink: Optional data sink. If provided, data will be written to it.

        Returns:
            pd.DataFrame: The transformed data.
        """
        if source is not None:
            data = source.read()
            result = self.fit_transform(data)

            if sink is not None:
                sink.write(result)

            return result
        else:
            raise ValueError("Source must be provided for execute()")

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Pipeline':
        """Create a pipeline from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Pipeline: The configured pipeline.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Import modules for dynamic loading
        import importlib

        # Build transformations
        transforms = []
        for transform_config in config.get('pipeline', {}).get('transforms', []):
            transform_type = transform_config['type']
            params = transform_config.get('params', {})

            # Try to load from dataprep.transforms
            try:
                module = importlib.import_module('dataprep.transforms')
                transform_class = getattr(module, transform_type)
                transforms.append(transform_class(**params))
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not load transform {transform_type}: {e}")

        pipeline = cls(transforms)

        # Store source and sink configuration for later use
        pipeline._source_config = config.get('source')
        pipeline._sink_config = config.get('sink')

        return pipeline

    def save_config(self, path: str) -> None:
        """Save the pipeline configuration to a YAML file.

        Args:
            path: Path to save the configuration to.
        """
        config = {
            'pipeline': {
                'transforms': [
                    {
                        'type': transform.__class__.__name__,
                        'params': transform.get_params()
                    }
                    for transform in self.transforms
                ]
            }
        }

        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the pipeline.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        return {
            'num_transforms': len(self.transforms),
            'fitted': self._fitted,
            'transforms': [t.__class__.__name__ for t in self.transforms],
            **self._metadata
        }

    def __len__(self) -> int:
        """Get the number of transformations in the pipeline.

        Returns:
            int: Number of transformations.
        """
        return len(self.transforms)

    def __repr__(self) -> str:
        """Get string representation of the pipeline.

        Returns:
            str: String representation.
        """
        transforms_str = ', '.join(t.__class__.__name__ for t in self.transforms)
        return f"Pipeline([{transforms_str}])"
