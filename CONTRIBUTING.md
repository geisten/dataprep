# Beiträge zur DataPrep Library

Vielen Dank für Ihr Interesse, zur DataPrep Library beizutragen!

## Entwicklungsumgebung einrichten

1. Repository klonen:
```bash
git clone <repository-url>
cd dataprep
```

2. Virtuelle Umgebung erstellen:
```bash
python -m venv venv
source venv/bin/activate  # Auf Windows: venv\Scripts\activate
```

3. Entwicklungsabhängigkeiten installieren:
```bash
pip install -e ".[dev]"
# oder
pip install -r requirements-dev.txt
```

## Tests ausführen

```bash
pytest tests/
```

Mit Coverage:
```bash
pytest --cov=dataprep tests/
```

## Code-Stil

Wir verwenden Black für Code-Formatierung:
```bash
black dataprep/ tests/ examples/
```

Und Flake8 für Linting:
```bash
flake8 dataprep/ tests/
```

## Neue Komponenten hinzufügen

### Neue Datenquelle

1. Erstellen Sie eine neue Datei in `dataprep/sources/`
2. Erben Sie von `DataSource` und implementieren Sie die `read()` Methode
3. Fügen Sie Tests in `tests/test_sources_sinks.py` hinzu
4. Exportieren Sie die Klasse in `dataprep/sources/__init__.py`

### Neue Transformation

1. Erstellen Sie eine neue Datei in `dataprep/transforms/`
2. Erben Sie von `Transform` und implementieren Sie `fit()` und `transform()`
3. Fügen Sie Tests in `tests/test_transforms.py` hinzu
4. Exportieren Sie die Klasse in `dataprep/transforms/__init__.py`

### Neue Datensenke

1. Erstellen Sie eine neue Datei in `dataprep/sinks/`
2. Erben Sie von `DataSink` und implementieren Sie die `write()` Methode
3. Fügen Sie Tests in `tests/test_sources_sinks.py` hinzu
4. Exportieren Sie die Klasse in `dataprep/sinks/__init__.py`

## Pull Requests

1. Erstellen Sie einen Feature-Branch
2. Implementieren Sie Ihre Änderungen mit Tests
3. Stellen Sie sicher, dass alle Tests bestehen
4. Erstellen Sie einen Pull Request mit einer klaren Beschreibung

## Fragen?

Öffnen Sie ein Issue auf GitHub für Fragen oder Diskussionen.
