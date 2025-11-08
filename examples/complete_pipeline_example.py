"""
Beispiel: Vollständige Pipeline

Zeigt wie man die CompleteePipeline für alle Datentypen verwendet.
"""

from pathlib import Path
from dataprep import CompletePipeline

def main():
    print("Vollständige LLM Pre-Training Data Pipeline")
    print("=" * 80)
    print()

    # Erstelle Pipeline
    pipeline = CompletePipeline()

    print("Pipeline initialisiert")
    print()
    print("Unterstützte Formate:")
    print("  Text:       .txt, .html, .htm")
    print("  Markdown:   .md, .markdown")
    print("  LaTeX:      .tex, .latex")
    print("  PDF:        .pdf")
    print("  Bilder:     .jpg, .jpeg, .png, .webp")
    print("  Videos:     .mp4, .avi, .webm, .mov")
    print()

    # Beispiel: Markdown-Verarbeitung
    print("=" * 80)
    print("Markdown-Verarbeitung")
    print("=" * 80)
    print()

    markdown_content = """---
title: Deep Learning Tutorial
author: AI Researcher
---

# Deep Learning Fundamentals

Deep learning is a subset of machine learning.

## Neural Networks

A neural network consists of layers:

- Input layer
- Hidden layers
- Output layer

### Code Example

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

## Mathematical Formulation

The loss function: $L = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$

## References

- [PyTorch](https://pytorch.org)
- [TensorFlow](https://tensorflow.org)
"""

    # Speichere temporär
    temp_md = Path("example.md")
    temp_md.write_text(markdown_content)

    # Verarbeite
    result = pipeline.process(temp_md)

    print(f"Typ: {result.get('type')}")
    print(f"Quelle: {result.get('source')}")
    print()

    if "metadata" in result:
        print("Metadaten:")
        for key, value in result["metadata"].items():
            print(f"  {key}: {value}")
        print()

    if "structure" in result:
        print("Struktur:")
        structure = result["structure"]
        print(f"  Headers: {len(structure.get('headers', []))}")
        print(f"  Paragraphs: {structure.get('num_paragraphs', 0)}")
        print(f"  Code Blocks: {result.get('num_code_blocks', 0)}")
        print(f"  Links: {result.get('num_links', 0)}")
        print()

    if "headers" in result.get("structure", {}):
        print("Header-Struktur:")
        for header in result["structure"]["headers"][:5]:
            indent = "  " * header["level"]
            print(f"{indent}- {header['text']}")
        print()

    if "code_blocks" in result:
        print(f"Code-Blöcke gefunden: {len(result['code_blocks'])}")
        for i, block in enumerate(result["code_blocks"][:2], 1):
            print(f"  Block {i}: {block['language']}")
            print(f"    Zeilen: {len(block['code'].split(chr(10)))}")
        print()

    # Aufräumen
    temp_md.unlink()

    # Zweiphasen-Ansatz (Phi-3 Methodik)
    print("\n" + "=" * 80)
    print("Zweiphasen Pre-Training Ansatz (Phi-3)")
    print("=" * 80)
    print()

    print("Phase 1: Allgemeines Wissen")
    print("  - Web-Daten (HTML-Extraktion)")
    print("  - Qualitätsfilterung (heuristisch)")
    print("  - Deduplizierung (exact + fuzzy)")
    print("  → Ziel: Breites Sprachwissen")
    print()

    print("Phase 2: Spezialisierung & Reasoning")
    print("  - Gefilterte Web-Daten (höhere Qualitätsschwelle)")
    print("  - Strukturierte Daten (Markdown, LaTeX, Code)")
    print("  - Synthetische Daten (optional)")
    print("  - Soft Deduplizierung (erhält Information)")
    print("  → Ziel: Reasoning & spezielle Fähigkeiten")
    print()

    print("Qualitätsmetriken:")
    print("  ✓ Durchschnittliche Wortlänge")
    print("  ✓ Alphabetische/Numerische Ratio")
    print("  ✓ Repetitions-Erkennung")
    print("  ✓ Toxizitäts-Filtering")
    print("  ✓ URL/Symbol Ratio")
    print()

    print("Multimodale Erweiterung:")
    print("  ✓ Bild-Text Paare (Vision-Language)")
    print("  ✓ Video-Frame Extraktion")
    print("  ✓ Temporale Pooling")
    print()


if __name__ == "__main__":
    main()
