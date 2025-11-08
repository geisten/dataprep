"""Example of using custom transformations."""

import pandas as pd
from dataprep import Pipeline
from dataprep.sources import DataFrameSource
from dataprep.transforms import CustomTransform
from dataprep.sinks import DataFrameSink

# Create sample data
data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000],
})

print("Original data:")
print(data)

# Define custom transformation functions
def add_age_group(df):
    """Add an age group column."""
    df = df.copy()
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 30, 100],
        labels=['young', 'mid', 'senior']
    )
    return df

def calculate_hourly_rate(df):
    """Calculate approximate hourly rate."""
    df = df.copy()
    # Assuming 2000 working hours per year
    df['hourly_rate'] = df['salary'] / 2000
    return df

def uppercase_names(df):
    """Convert names to uppercase."""
    df = df.copy()
    df['name'] = df['name'].str.upper()
    return df

# Create pipeline with custom transformations
pipeline = Pipeline([
    CustomTransform(add_age_group, name='AddAgeGroup'),
    CustomTransform(calculate_hourly_rate, name='CalculateHourlyRate'),
    CustomTransform(uppercase_names, name='UppercaseNames'),
])

# Run pipeline
source = DataFrameSource(data)
sink = DataFrameSink()
pipeline.run(source, sink)

# Get results
result = sink.get_data()
print("\nProcessed data:")
print(result)
