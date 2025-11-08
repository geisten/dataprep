"""
Bild-Prozessor für multimodale LLM Pre-Training

Basierend auf modernen Vision-Language Model Ansätzen:
- Resize/Normalisierung
- Tokenisierung (kontinuierlich oder diskret)
- Vision-Text Alignment
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python nicht verfügbar - eingeschränkte Bildverarbeitung")


class ImageProcessor:
    """
    Bild-Prozessor für multimodale LLMs

    Features:
    - Adaptive oder feste Größenanpassung
    - Normalisierung (ImageNet-Statistiken)
    - Qualitätsprüfung
    """

    def __init__(self, config: Optional[Dict] = None, target_size: int = 224):
        """
        Args:
            config: Konfiguration
            target_size: Zielgröße für Bilder
        """
        self.config = config or {}
        self.target_size = self.config.get("target_size", target_size)
        self.max_size = self.config.get("max_size", 512)
        self.resize_mode = self.config.get("resize_mode", "adaptive")
        self.normalize = self.config.get("normalize", True)

        # ImageNet Statistiken (Standard für Vision Models)
        self.mean = self.config.get("mean", [0.485, 0.456, 0.406])
        self.std = self.config.get("std", [0.229, 0.224, 0.225])

        self.supported_formats = self.config.get(
            "formats", ["jpg", "jpeg", "png", "webp"]
        )

    def process(self, image_path: Path) -> Dict[str, Any]:
        """
        Verarbeite Bild

        Args:
            image_path: Pfad zum Bild

        Returns:
            Dict mit verarbeitetem Bild und Metadaten
        """
        result = {
            "source": str(image_path),
            "type": "image",
        }

        # Prüfe Format
        suffix = image_path.suffix.lower().lstrip(".")
        if suffix not in self.supported_formats:
            result["error"] = f"Nicht unterstütztes Format: {suffix}"
            return result

        try:
            # Lade Bild
            image = Image.open(image_path)

            # Konvertiere zu RGB falls nötig
            if image.mode != "RGB":
                image = image.convert("RGB")

            result["original_size"] = image.size
            result["original_mode"] = image.mode

            # Resize
            if self.resize_mode == "fixed":
                resized = self._resize_fixed(image)
            elif self.resize_mode == "adaptive":
                resized = self._resize_adaptive(image)
            else:
                resized = image

            result["processed_size"] = resized.size

            # Zu NumPy Array
            image_array = np.array(resized).astype(np.float32)

            # Normalisierung
            if self.normalize:
                image_array = self._normalize(image_array)

            result["image_array"] = image_array
            result["shape"] = image_array.shape

            # Qualitätsprüfung
            quality_score = self._assess_quality(image_array)
            result["quality_score"] = quality_score

        except Exception as e:
            logger.error(f"Fehler bei Bildverarbeitung: {e}")
            result["error"] = str(e)

        return result

    def _resize_fixed(self, image: Image.Image) -> Image.Image:
        """Resize zu fester Größe"""
        return image.resize((self.target_size, self.target_size), Image.LANCZOS)

    def _resize_adaptive(self, image: Image.Image) -> Image.Image:
        """
        Adaptive Größenanpassung (erhält Aspect Ratio)

        Skaliert so dass die kürzere Seite = target_size
        """
        width, height = image.size

        # Finde kürzere Seite
        if width < height:
            new_width = self.target_size
            new_height = int(height * (self.target_size / width))
        else:
            new_height = self.target_size
            new_width = int(width * (self.target_size / height))

        # Begrenze auf max_size
        if new_width > self.max_size or new_height > self.max_size:
            scale = min(self.max_size / new_width, self.max_size / new_height)
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)

        return image.resize((new_width, new_height), Image.LANCZOS)

    def _normalize(self, image_array: np.ndarray) -> np.ndarray:
        """
        Normalisiere Bild (ImageNet-Statistiken)

        Args:
            image_array: Bild als NumPy Array (H, W, C), Werte 0-255

        Returns:
            Normalisiertes Array
        """
        # Skaliere zu [0, 1]
        image_array = image_array / 255.0

        # Normalisiere mit mean/std
        mean = np.array(self.mean).reshape(1, 1, 3)
        std = np.array(self.std).reshape(1, 1, 3)

        image_array = (image_array - mean) / std

        return image_array

    def _assess_quality(self, image_array: np.ndarray) -> float:
        """
        Bewerte Bildqualität

        Simple Heuristiken:
        - Varianz (zu niedrig = wahrscheinlich blank/einfarbig)
        - Blur-Detektion (Laplacian Varianz)

        Returns:
            Quality score (0.0 - 1.0)
        """
        try:
            # Varianz-Check
            variance = np.var(image_array)

            # Niedrige Varianz = niedrige Qualität
            if variance < 0.01:
                return 0.3

            # Blur-Detektion (erfordert OpenCV)
            if CV2_AVAILABLE:
                # Konvertiere zurück zu 0-255
                img_uint8 = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)

                # Laplacian Varianz
                laplacian_var = cv2.Laplacian(img_uint8, cv2.CV_64F).var()

                # Normalisiere (typische Werte: 0-1000)
                blur_score = min(laplacian_var / 100, 1.0)

                return (blur_score + min(variance * 10, 1.0)) / 2

            # Fallback: nur Varianz
            return min(variance * 10, 1.0)

        except Exception as e:
            logger.warning(f"Qualitätsbewertung fehlgeschlagen: {e}")
            return 0.5  # Neutral
