# DataPrep - Data Preprocessing Library

Eine flexible und erweiterbare Python-Bibliothek zur Datenvorverarbeitung für Machine Learning Training-Pipelines.

## Features

- **Modulare Architektur**: Erweiterbare Pipeline mit Datenquellen, Transformationen und Datensenken
- **Flexibel**: Unterstützt verschiedene Datenformate (CSV, JSON, Parquet, Datenbanken, etc.)
- **Erweiterbar**: Einfaches Hinzufügen eigener Datenquellen, Transformationen und Senken
- **Type-Safe**: Vollständig typisiert mit Python Type Hints
- **Pipeline-basiert**: Deklarative Konfiguration von Preprocessing-Pipelines

## Installation

```bash
pip install -e .
```

Mit allen optionalen Abhängigkeiten:
```bash
pip install -e ".[all]"
```

Für Entwicklung:
```bash
pip install -e ".[dev]"
```

## Schnellstart

```python
from dataprep import Pipeline
from dataprep.sources import CSVSource
from dataprep.transforms import Normalizer, DropMissing, FeatureSelector
from dataprep.sinks import ParquetSink

# Pipeline erstellen
pipeline = Pipeline([
    DropMissing(threshold=0.8),
    FeatureSelector(features=['feature1', 'feature2', 'feature3']),
    Normalizer(method='standard'),
])

# Datenquelle definieren
source = CSVSource('data/input.csv')

# Datensenke definieren
sink = ParquetSink('data/output.parquet')

# Pipeline ausführen
pipeline.run(source, sink)
```

## Architektur

### Datenquellen (Sources)

Datenquellen sind verantwortlich für das Einlesen von Daten:

```python
from dataprep.sources import CSVSource, JSONSource, ParquetSource

# CSV-Datei
source = CSVSource('data.csv', delimiter=';')

# JSON-Datei
source = JSONSource('data.json', orient='records')

# Parquet-Datei
source = ParquetSource('data.parquet')
```

### Transformationen (Transforms)

Transformationen verarbeiten die Daten:

```python
from dataprep.transforms import (
    Normalizer,
    DropMissing,
    FeatureSelector,
    OneHotEncoder,
    CustomTransform
)

# Normalisierung
transform = Normalizer(method='standard')  # oder 'minmax'

# Fehlende Werte entfernen
transform = DropMissing(threshold=0.5)

# Features auswählen
transform = FeatureSelector(features=['age', 'income'])

# One-Hot Encoding
transform = OneHotEncoder(columns=['category'])
```

### Datensenken (Sinks)

Datensenken speichern die verarbeiteten Daten:

```python
from dataprep.sinks import CSVSink, ParquetSink, JSONSink

# CSV-Ausgabe
sink = CSVSink('output.csv')

# Parquet-Ausgabe
sink = ParquetSink('output.parquet', compression='snappy')

# JSON-Ausgabe
sink = JSONSink('output.json', orient='records')
```

## Eigene Komponenten erstellen

### Eigene Transformation

```python
from dataprep.core import Transform
import pandas as pd

class MyCustomTransform(Transform):
    def __init__(self, my_param: str):
        self.my_param = my_param

    def fit(self, data: pd.DataFrame) -> 'MyCustomTransform':
        # Optional: Parameter aus Daten lernen
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ihre Transformation hier
        return data
```

### Eigene Datenquelle

```python
from dataprep.core import DataSource
import pandas as pd

class MyCustomSource(DataSource):
    def __init__(self, path: str):
        self.path = path

    def read(self) -> pd.DataFrame:
        # Ihre Logik zum Einlesen
        return pd.DataFrame()
```

### Eigene Datensenke

```python
from dataprep.core import DataSink
import pandas as pd

class MyCustomSink(DataSink):
    def __init__(self, path: str):
        self.path = path

    def write(self, data: pd.DataFrame) -> None:
        # Ihre Logik zum Schreiben
        pass
```

## Pipeline-Konfiguration mit YAML

```yaml
pipeline:
  transforms:
    - type: DropMissing
      params:
        threshold: 0.8

    - type: FeatureSelector
      params:
        features: [feature1, feature2, feature3]

    - type: Normalizer
      params:
        method: standard

source:
  type: CSVSource
  params:
    path: data/input.csv
    delimiter: ','

sink:
  type: ParquetSink
  params:
    path: data/output.parquet
    compression: snappy
```

Ausführen:
```python
from dataprep import Pipeline

pipeline = Pipeline.from_yaml('config.yaml')
pipeline.execute()
```

## Beispiele

Weitere Beispiele finden Sie im `examples/` Verzeichnis:

- `examples/basic_pipeline.py` - Einfache Pipeline
- `examples/custom_transform.py` - Eigene Transformation
- `examples/yaml_config.py` - YAML-Konfiguration
- `examples/advanced_pipeline.py` - Fortgeschrittene Features

## Tests

```bash
pytest tests/
```

Mit Coverage:
```bash
pytest --cov=dataprep tests/
```

## Lizenz

MIT License
