"""Advanced pipeline example with multiple transformations."""

import pandas as pd
import numpy as np
from dataprep import Pipeline
from dataprep.sources import DataFrameSource
from dataprep.transforms import (
    DropMissing,
    FillMissing,
    OutlierRemover,
    OneHotEncoder,
    Normalizer,
    FeatureSelector
)
from dataprep.sinks import DataFrameSink

# Create sample data with various issues
np.random.seed(42)
n_samples = 100

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'experience': np.random.randint(0, 40, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], n_samples),
    'performance_score': np.random.uniform(0, 1, n_samples),
})

# Add some missing values
data.loc[np.random.choice(data.index, 10), 'age'] = None
data.loc[np.random.choice(data.index, 15), 'income'] = None
data.loc[np.random.choice(data.index, 5), 'experience'] = None

# Add some outliers
data.loc[np.random.choice(data.index, 3), 'income'] = 1000000

print("Original data:")
print(data.head(10))
print(f"\nShape: {data.shape}")
print(f"\nMissing values:\n{data.isnull().sum()}")
print(f"\nData types:\n{data.dtypes}")

# Create comprehensive pipeline
pipeline = Pipeline([
    # Step 1: Handle missing values
    FillMissing(strategy='median', columns=['age', 'income', 'experience']),

    # Step 2: Remove outliers from numeric columns
    OutlierRemover(method='iqr', threshold=1.5, columns=['income', 'age']),

    # Step 3: Encode categorical variables
    OneHotEncoder(columns=['education', 'department'], drop_first=True),

    # Step 4: Normalize numeric features
    Normalizer(method='standard'),
])

# Create source and sink
source = DataFrameSource(data)
sink = DataFrameSink()

# Run pipeline
print("\n" + "="*50)
print("Running pipeline...")
print("="*50)
pipeline.run(source, sink)

# Get results
result = sink.get_data()
print("\nProcessed data:")
print(result.head(10))
print(f"\nShape: {result.shape}")
print(f"\nColumns: {list(result.columns)}")
print(f"\nMissing values:\n{result.isnull().sum().sum()}")

# Show statistics
print("\nNumeric columns statistics:")
print(result.describe())

# Show pipeline metadata
print("\nPipeline metadata:")
print(pipeline.get_metadata())
