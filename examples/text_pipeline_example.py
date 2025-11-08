"""
Beispiel: Text-Pipeline für LLM Pre-Training

Zeigt wie man die Text-Pipeline für HTML/Text-Daten verwendet.
"""

from pathlib import Path
from dataprep import TextPipeline

def main():
    # Erstelle Pipeline mit Soft Deduplication
    pipeline = TextPipeline(
        deduplication="soft",  # soft, fuzzy, oder exact
        quality_filter=True,
        toxicity_filter=False,  # Deaktiviert da Modell-Download erforderlich
    )

    print("Text-Pipeline initialisiert")
    print(f"Deduplication: soft")
    print(f"Quality Filter: aktiviert")
    print()

    # Beispiel 1: Verarbeite HTML-Text
    html_example = """
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Deep Learning</h1>
        <p>Deep learning is a subset of machine learning that uses neural networks.</p>

        <h2>Code Example</h2>
        <pre><code class="language-python">
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
        </code></pre>

        <h2>Math Formula</h2>
        <p>The loss is computed as: <span class="tex">L = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2</span></p>
    </body>
    </html>
    """

    # Speichere temporär
    temp_file = Path("temp_example.html")
    temp_file.write_text(html_example)

    # Verarbeite
    result = pipeline.process(temp_file)

    print("Ergebnis:")
    print(f"Quelle: {result['source']}")
    print(f"Gefiltert: {result['filtered']}")
    print(f"Text-Länge: {result.get('length', 0)} Zeichen")
    print()
    print("Extrahierter Text:")
    print("-" * 80)
    print(result.get("text", "")[:500])
    print("-" * 80)
    print()

    if "quality_scores" in result:
        print("Qualitäts-Scores:")
        for key, value in result["quality_scores"].items():
            print(f"  {key}: {value:.3f}")
    print()

    # Aufräumen
    temp_file.unlink()

    # Beispiel 2: Deduplizierung
    print("\n" + "=" * 80)
    print("Deduplizierungs-Beispiel")
    print("=" * 80 + "\n")

    documents = [
        {"text": "This is the first document about machine learning."},
        {"text": "This is the second document about deep learning."},
        {"text": "This is the first document about machine learning."},  # Exaktes Duplikat
        {"text": "This is the first document about machine learning!"},  # Sehr ähnlich
        {"text": "Completely different content here."},
    ]

    print(f"Original: {len(documents)} Dokumente")

    deduplicated = pipeline.deduplicate(documents)

    print(f"Nach Deduplizierung: {len(deduplicated)} Dokumente")
    print()

    # Zeige Weights (bei Soft Dedup)
    for i, doc in enumerate(deduplicated):
        weight = doc.get("weight", 1.0)
        text_preview = doc["text"][:50]
        print(f"Doc {i+1}: weight={weight:.2f} | {text_preview}...")


if __name__ == "__main__":
    main()
