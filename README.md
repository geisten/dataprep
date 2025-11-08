# LLM Pre-Training Data Pipeline

Eine moderne, umfassende Datenaufbereitungs-Pipeline für das Pre-Training von Large Language Models (LLMs), basierend auf neuesten Erkenntnissen von Phi-3/4, LLaMA 3 und multimodalen Architekturen.

## Features

### 📝 Text-Verarbeitung
- **HTML-zu-Text Extraktion**: Erhält Formeln (TeX/MathML), Code-Blöcke, Tabellen
- **Deduplizierung**: Exact, Fuzzy und Soft Deduplication
- **Qualitätsfilterung**: Modellbasierte und heuristische Filter
- **Toxizitätsfilter**: Automatische Erkennung problematischer Inhalte

### 📄 Strukturierte Formate
- **LaTeX/TeX**: Multi-File TeX-Quellen, Formelerhaltung
- **Markdown**: Strukturerhaltende Verarbeitung
- **PDF**: Textextraktion mit Layout-Erhalt

### 🎨 Multimodal
- **Bilder**: JPEG, PNG mit Resize und Tokenisierung
- **Videos**: Frame-Extraktion, temporale Pooling
- **Vision-Text Alignment**: Für multimodale LLMs

### 🎯 Zweiphasen-Ansatz (Phi-3 Methodik)
- **Phase 1**: Web-Daten für allgemeines Wissen
- **Phase 2**: Gefilterte Web-Daten + synthetische Daten für Reasoning

## Installation

```bash
pip install -e .
```

## Schnellstart

### Text-Pipeline

```python
from dataprep import TextPipeline

pipeline = TextPipeline(
    deduplication='soft',
    quality_filter=True,
    toxicity_filter=True
)

# Verarbeite HTML/Text
processed = pipeline.process("input.html")
```

### Multimodal-Pipeline

```python
from dataprep import MultimodalPipeline

pipeline = MultimodalPipeline(
    image_size=224,
    video_fps=1
)

# Verarbeite Bild-Text Paare
result = pipeline.process_image_text("image.jpg", "caption.txt")
```

## Architektur

Die Pipeline folgt dem State-of-the-Art Design moderner LLM-Trainingspipelines:

1. **Extraktion**: Format-spezifische Parser (HTML, PDF, LaTeX, etc.)
2. **Reinigung**: Unicode-Normalisierung, Sprachtrennung
3. **Deduplizierung**: Mehrschichtige Deduplizierung
4. **Qualitätsfilter**: Heuristische + modellbasierte Filter
5. **Tokenisierung**: Optimiert für LLM-Training
6. **Export**: JSONL-Format für Training

## Konfiguration

Siehe `config/default_config.yaml` für alle Optionen.

## Basiert auf

- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)
- [NVIDIA NeMo Curator](https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/)
- [SoftDedup](https://arxiv.org/abs/2409.05816)

## Lizenz

MIT
