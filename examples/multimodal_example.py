"""
Beispiel: Multimodal-Pipeline für Vision-Language Models

Zeigt wie man Bilder und Videos für multimodale LLM Pre-Training aufbereitet.
"""

from pathlib import Path
import numpy as np
from dataprep import MultimodalPipeline

def create_sample_image(path: Path, size=(224, 224)):
    """Erstelle ein Beispiel-Bild"""
    try:
        from PIL import Image

        # Erstelle ein einfaches Gradientenbild
        img_array = np.zeros((*size, 3), dtype=np.uint8)

        for i in range(size[0]):
            for j in range(size[1]):
                img_array[i, j] = [i % 256, j % 256, (i+j) % 256]

        img = Image.fromarray(img_array)
        img.save(path)

        return True
    except ImportError:
        print("PIL nicht verfügbar - überspringe Bild-Beispiel")
        return False


def main():
    print("Multimodal-Pipeline Beispiel")
    print("=" * 80)
    print()

    # Erstelle Pipeline
    pipeline = MultimodalPipeline(
        image_size=224,
        video_fps=1,
    )

    print("Pipeline initialisiert")
    print(f"Bild-Größe: {pipeline.image_size}")
    print(f"Video FPS: {pipeline.video_fps}")
    print()

    # Beispiel 1: Bild-Verarbeitung
    print("=" * 80)
    print("Bild-Verarbeitung")
    print("=" * 80)
    print()

    sample_image = Path("sample_image.jpg")

    if create_sample_image(sample_image):
        result = pipeline.process_image(sample_image)

        print(f"Quelle: {result.get('source')}")
        print(f"Original-Größe: {result.get('original_size')}")
        print(f"Verarbeitete Größe: {result.get('processed_size')}")
        print(f"Array-Shape: {result.get('shape')}")
        print(f"Qualitäts-Score: {result.get('quality_score', 0):.3f}")
        print()

        # Aufräumen
        sample_image.unlink()
    else:
        print("Überspringe Bild-Beispiel (PIL nicht verfügbar)")

    # Beispiel 2: Bild-Text Paar
    print("\n" + "=" * 80)
    print("Bild-Text Paar (für Vision-Language Training)")
    print("=" * 80)
    print()

    sample_image2 = Path("sample_image2.jpg")

    if create_sample_image(sample_image2):
        caption = "A beautiful gradient image showing various colors from red to blue."

        result = pipeline.process_image_text(sample_image2, caption)

        print(f"Quelle: {result.get('source')}")
        print(f"Text: {result.get('text')}")
        print(f"Bild-Shape: {result['image'].get('shape')}")
        print()

        # Aufräumen
        sample_image2.unlink()

    # Video-Beispiel (nur Struktur zeigen, kein echtes Video)
    print("\n" + "=" * 80)
    print("Video-Verarbeitung (Konzept)")
    print("=" * 80)
    print()

    print("Video-Processing extrahiert:")
    print("  - Frames mit konfigurierbarer FPS")
    print("  - Temporale Pooling (average, max, attention)")
    print("  - Frame-Normalisierung")
    print()
    print("Verwendung:")
    print("  result = pipeline.process_video('video.mp4')")
    print("  frames = result['frames']  # Liste von Frame-Arrays")
    print("  pooled = result['pooled_representation']  # Temporale Aggregation")
    print()


if __name__ == "__main__":
    main()
