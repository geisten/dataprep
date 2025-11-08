"""
Qualitäts-Scorer für LLM Pre-Training Daten

Implementiert modellbasierte Qualitätsbewertung:
- Toxizitäts-Scoring
- Komplexitäts-Scoring (optional)
- Educational-Value Scoring (optional)
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    logger.warning("detoxify nicht verfügbar - Toxizitäts-Scoring deaktiviert")


class QualityScorer:
    """
    Modellbasierter Qualitäts-Scorer

    Features:
    - Toxizitäts-Scoring (Detoxify)
    - Erweiterbar für weitere Metriken
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration
        """
        self.config = config or {}
        toxicity_config = self.config.get("toxicity", {})

        self.toxicity_enabled = toxicity_config.get("enabled", True)
        self.max_toxicity = toxicity_config.get("max_toxicity", 0.5)
        self.check_categories = toxicity_config.get(
            "check_categories",
            ["toxicity", "severe_toxicity", "obscene", "threat", "insult"],
        )

        # Lazy loading des Toxizitäts-Modells
        self._toxicity_model = None

    @property
    def toxicity_model(self):
        """Lazy loading des Detoxify-Modells"""
        if self._toxicity_model is None and DETOXIFY_AVAILABLE:
            logger.info("Lade Toxizitäts-Modell (Detoxify)...")
            self._toxicity_model = Detoxify("original")
        return self._toxicity_model

    def score_toxicity(self, text: str) -> Dict[str, float]:
        """
        Bewerte Toxizität eines Texts

        Args:
            text: Zu bewertender Text

        Returns:
            Dict mit Toxizitäts-Scores für verschiedene Kategorien
        """
        if not self.toxicity_enabled or not DETOXIFY_AVAILABLE:
            return {}

        if not text:
            return {}

        try:
            # Detoxify kann nur max 512 Tokens verarbeiten
            # Verwende ersten Teil des Texts
            text_sample = text[:2048]

            scores = self.toxicity_model.predict(text_sample)

            # Konvertiere zu Python floats
            return {k: float(v) for k, v in scores.items()}

        except Exception as e:
            logger.error(f"Toxizitäts-Scoring fehlgeschlagen: {e}")
            return {}

    def is_toxic(self, text: str) -> bool:
        """
        Prüfe ob Text zu toxisch ist

        Args:
            text: Zu prüfender Text

        Returns:
            True wenn Text zu toxisch
        """
        scores = self.score_toxicity(text)

        if not scores:
            return False

        # Prüfe alle konfigurierten Kategorien
        for category in self.check_categories:
            if category in scores and scores[category] > self.max_toxicity:
                logger.debug(f"Text zu toxisch ({category}={scores[category]})")
                return True

        return False

    def score_quality(self, text: str) -> Dict[str, float]:
        """
        Umfassende Qualitätsbewertung

        Args:
            text: Zu bewertender Text

        Returns:
            Dict mit verschiedenen Qualitäts-Metriken
        """
        scores = {}

        # Toxizität
        if self.toxicity_enabled:
            toxicity_scores = self.score_toxicity(text)
            scores.update({f"toxicity_{k}": v for k, v in toxicity_scores.items()})

        # Hier können weitere Metriken hinzugefügt werden:
        # - Komplexität (z.B. Flesch Reading Ease)
        # - Educational Value (z.B. mit spezialisiertem Modell)
        # - Coherence
        # - etc.

        return scores
