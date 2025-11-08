# Beispiele

Dieser Ordner enthält Beispiele für die Verwendung der LLM Pre-Training Data Pipeline.

## Verfügbare Beispiele

### 1. Text-Pipeline (`text_pipeline_example.py`)

Zeigt die Verwendung der Text-Pipeline für:
- HTML-zu-Text Extraktion
- Qualitätsfilterung
- Deduplizierung (Soft Dedup)

```bash
python text_pipeline_example.py
```

### 2. Multimodal-Pipeline (`multimodal_example.py`)

Demonstriert:
- Bild-Verarbeitung
- Bild-Text Paare
- Video-Konzepte

```bash
python multimodal_example.py
```

### 3. Vollständige Pipeline (`complete_pipeline_example.py`)

Zeigt die Verwendung für:
- Alle Datentypen (Auto-Detection)
- Markdown-Verarbeitung
- Zweiphasen-Ansatz (Phi-3)

```bash
python complete_pipeline_example.py
```

## Eigene Daten verarbeiten

### Text-Daten

```python
from dataprep import TextPipeline

pipeline = TextPipeline(
    deduplication='soft',
    quality_filter=True,
    toxicity_filter=True
)

# Einzelne Datei
result = pipeline.process('document.html')

# Ganzes Verzeichnis
results = pipeline.process('data/html_files/')

# Deduplizierung
deduplicated = pipeline.deduplicate(results)
```

### Multimodale Daten

```python
from dataprep import MultimodalPipeline

pipeline = MultimodalPipeline(
    image_size=224,
    video_fps=1
)

# Bild
image_result = pipeline.process_image('photo.jpg')

# Video
video_result = pipeline.process_video('clip.mp4')

# Bild-Text Paar
pair_result = pipeline.process_image_text('photo.jpg', 'A description')
```

### Auto-Detection

```python
from dataprep import CompletePipeline

pipeline = CompletePipeline()

# Automatische Format-Erkennung
result = pipeline.process('document.pdf')  # PDF
result = pipeline.process('article.md')     # Markdown
result = pipeline.process('paper.tex')      # LaTeX
result = pipeline.process('image.png')      # Bild
```

## Konfiguration

Siehe `config/default_config.yaml` für alle Optionen.

Eigene Konfiguration verwenden:

```python
from dataprep import TextPipeline

pipeline = TextPipeline(config='my_config.yaml')
```
