"""
Video-Prozessor für multimodale LLM Pre-Training

Basierend auf modernen Video-Language Model Ansätzen:
- Frame-Extraktion
- Temporale Pooling
- Vision-Encoder Features
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python nicht verfügbar - Video-Verarbeitung deaktiviert")

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    logger.warning("av nicht verfügbar - verwende OpenCV für Video")


class VideoProcessor:
    """
    Video-Prozessor für multimodale LLMs

    Features:
    - Frame-Extraktion (uniform sampling)
    - Temporale Pooling (average, max, attention)
    - Resize/Normalisierung
    """

    def __init__(self, config: Optional[Dict] = None, fps: int = 1):
        """
        Args:
            config: Konfiguration
            fps: Frames pro Sekunde zu extrahieren
        """
        if not CV2_AVAILABLE and not AV_AVAILABLE:
            raise ImportError("opencv-python oder av erforderlich für Video-Verarbeitung")

        self.config = config or {}
        self.fps = self.config.get("fps", fps)
        self.max_frames = self.config.get("max_frames", 100)
        self.resize = self.config.get("resize", 224)
        self.temporal_pooling = self.config.get("temporal_pooling", "average")

        self.supported_formats = self.config.get(
            "formats", ["mp4", "avi", "webm", "mov"]
        )

    def process(self, video_path: Path) -> Dict[str, Any]:
        """
        Verarbeite Video

        Args:
            video_path: Pfad zum Video

        Returns:
            Dict mit extrahierten Frames und Metadaten
        """
        result = {
            "source": str(video_path),
            "type": "video",
        }

        # Prüfe Format
        suffix = video_path.suffix.lower().lstrip(".")
        if suffix not in self.supported_formats:
            result["error"] = f"Nicht unterstütztes Format: {suffix}"
            return result

        try:
            # Extrahiere Frames
            if AV_AVAILABLE:
                frames, metadata = self._extract_frames_av(video_path)
            else:
                frames, metadata = self._extract_frames_cv2(video_path)

            result.update(metadata)
            result["frames"] = frames
            result["num_frames"] = len(frames)

            # Temporale Pooling
            if len(frames) > 0:
                pooled = self._temporal_pooling(frames)
                result["pooled_representation"] = pooled

        except Exception as e:
            logger.error(f"Fehler bei Video-Verarbeitung: {e}")
            result["error"] = str(e)

        return result

    def _extract_frames_cv2(self, video_path: Path) -> tuple[List[np.ndarray], Dict]:
        """Extrahiere Frames mit OpenCV"""
        cap = cv2.VideoCapture(str(video_path))

        # Metadaten
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        metadata = {
            "total_frames": total_frames,
            "fps": video_fps,
            "width": width,
            "height": height,
            "duration": duration,
        }

        # Berechne Frame-Intervall
        frame_interval = int(video_fps / self.fps) if self.fps > 0 else 1
        frame_interval = max(frame_interval, 1)

        frames = []
        frame_idx = 0

        while len(frames) < self.max_frames:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Konvertiere BGR zu RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize
                if self.resize:
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (self.resize, self.resize),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

                # Normalisiere zu [0, 1]
                frame_normalized = frame_rgb.astype(np.float32) / 255.0

                frames.append(frame_normalized)

            frame_idx += 1

        cap.release()

        return frames, metadata

    def _extract_frames_av(self, video_path: Path) -> tuple[List[np.ndarray], Dict]:
        """Extrahiere Frames mit PyAV (effizienter)"""
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        # Metadaten
        metadata = {
            "total_frames": stream.frames,
            "fps": float(stream.average_rate),
            "width": stream.width,
            "height": stream.height,
            "duration": float(stream.duration * stream.time_base) if stream.duration else 0,
        }

        # Berechne Frame-Intervall
        frame_interval = int(metadata["fps"] / self.fps) if self.fps > 0 else 1
        frame_interval = max(frame_interval, 1)

        frames = []
        frame_idx = 0

        for frame in container.decode(video=0):
            if frame_idx % frame_interval == 0 and len(frames) < self.max_frames:
                # Konvertiere zu NumPy Array
                img = frame.to_ndarray(format="rgb24")

                # Resize
                if self.resize:
                    import cv2
                    img = cv2.resize(
                        img,
                        (self.resize, self.resize),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

                # Normalisiere
                img_normalized = img.astype(np.float32) / 255.0

                frames.append(img_normalized)

            frame_idx += 1

            if len(frames) >= self.max_frames:
                break

        container.close()

        return frames, metadata

    def _temporal_pooling(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Temporale Pooling über Frames

        Args:
            frames: Liste von Frame-Arrays

        Returns:
            Gepoolte Representation
        """
        if not frames:
            return np.array([])

        frames_array = np.stack(frames, axis=0)  # (T, H, W, C)

        if self.temporal_pooling == "average":
            # Average pooling über Zeit
            return np.mean(frames_array, axis=0)

        elif self.temporal_pooling == "max":
            # Max pooling über Zeit
            return np.max(frames_array, axis=0)

        elif self.temporal_pooling == "attention":
            # Einfache Attention-basierte Pooling
            # Berechne "Wichtigkeit" jedes Frames (Varianz als Proxy)
            frame_importance = np.array([np.var(f) for f in frames])

            # Softmax
            frame_importance = np.exp(frame_importance - np.max(frame_importance))
            frame_importance = frame_importance / np.sum(frame_importance)

            # Gewichtete Summe
            weighted = np.sum(
                frames_array * frame_importance.reshape(-1, 1, 1, 1),
                axis=0,
            )

            return weighted

        else:
            # Fallback: average
            return np.mean(frames_array, axis=0)
