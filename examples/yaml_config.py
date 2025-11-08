"""Example of using YAML configuration."""

import pandas as pd
from dataprep import Pipeline
from dataprep.sources import DataFrameSource, CSVSource
from dataprep.sinks import DataFrameSink, CSVSink
import yaml

# First, let's create a sample configuration file
config = """
pipeline:
  transforms:
    - type: DropMissing
      params:
        threshold: 0.5

    - type: FeatureSelector
      params:
        features: ['age', 'income', 'score']

    - type: Normalizer
      params:
        method: standard
"""

# Save configuration
with open('/tmp/pipeline_config.yaml', 'w') as f:
    f.write(config)

print("Configuration file created:")
print(config)

# Create sample data and save as CSV
data = pd.DataFrame({
    'age': [25, 30, None, 35, 40, 28, 32],
    'income': [50000, 60000, 55000, None, 80000, 58000, 62000],
    'score': [0.8, 0.6, 0.9, 0.7, 0.95, 0.85, 0.75],
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
})

data.to_csv('/tmp/input_data.csv', index=False)

# Load pipeline from YAML
pipeline = Pipeline.from_yaml('/tmp/pipeline_config.yaml')

print(f"\nLoaded pipeline: {pipeline}")

# Execute pipeline
source = CSVSource('/tmp/input_data.csv')
sink = CSVSink('/tmp/output_data.csv')

pipeline.run(source, sink)

# Load and display results
result = pd.read_csv('/tmp/output_data.csv')
print("\nProcessed data:")
print(result)

# You can also save a pipeline configuration
pipeline.save_config('/tmp/saved_pipeline_config.yaml')
print("\nPipeline configuration saved to: /tmp/saved_pipeline_config.yaml")
