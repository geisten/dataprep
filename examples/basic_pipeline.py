"""Basic pipeline example."""

import pandas as pd
from dataprep import Pipeline
from dataprep.sources import DataFrameSource
from dataprep.transforms import DropMissing, Normalizer, FeatureSelector
from dataprep.sinks import DataFrameSink

# Create sample data
data = pd.DataFrame({
    'age': [25, 30, None, 35, 40, 28, 32],
    'income': [50000, 60000, 55000, None, 80000, 58000, 62000],
    'score': [0.8, 0.6, 0.9, 0.7, 0.95, 0.85, 0.75],
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
})

print("Original data:")
print(data)
print(f"\nShape: {data.shape}")

# Create pipeline
pipeline = Pipeline([
    DropMissing(threshold=0.5),  # Drop rows with >50% missing values
    FeatureSelector(features=['age', 'income', 'score']),  # Select numeric features
    Normalizer(method='standard'),  # Normalize to zero mean, unit variance
])

# Create source and sink
source = DataFrameSource(data)
sink = DataFrameSink()

# Run pipeline
pipeline.run(source, sink)

# Get results
result = sink.get_data()
print("\nProcessed data:")
print(result)
print(f"\nShape: {result.shape}")
