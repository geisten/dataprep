"""
Qualitätsfilter für LLM Pre-Training Daten

Implementiert heuristische Filter basierend auf:
- Character-Level Statistiken
- Wort-Level Statistiken
- Struktur-Metriken
- Repetitions-Erkennung
"""

import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Heuristischer Qualitätsfilter

    Prüft verschiedene Qualitätskriterien wie:
    - Durchschnittliche Wortlänge
    - Ratio von alphabetischen/numerischen/Symbol-Zeichen
    - Groß-/Kleinschreibung
    - Repetitionen
    - Zeilen-/Absatz-Anzahl
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration mit heuristic-Sektion
        """
        from typing import Optional  # Import hier für Typ-Hint

        heuristic_config = (config or {}).get("heuristic", {})

        # Wort-Level
        self.min_avg_word_length = heuristic_config.get("min_avg_word_length", 3)
        self.max_avg_word_length = heuristic_config.get("max_avg_word_length", 10)

        # Character-Level
        self.min_alpha_ratio = heuristic_config.get("min_alpha_ratio", 0.8)
        self.max_symbol_ratio = heuristic_config.get("max_symbol_ratio", 0.1)
        self.max_digit_ratio = heuristic_config.get("max_digit_ratio", 0.2)
        self.max_uppercase_ratio = heuristic_config.get("max_uppercase_ratio", 0.3)

        # Struktur
        self.min_lines = heuristic_config.get("min_lines", 3)
        self.max_url_ratio = heuristic_config.get("max_url_ratio", 0.2)

        # Repetitionen
        self.max_line_repetition = heuristic_config.get("max_line_repetition", 0.3)
        self.max_paragraph_repetition = heuristic_config.get("max_paragraph_repetition", 0.3)

    def filter(self, text: str) -> Tuple[bool, Dict[str, float]]:
        """
        Prüfe ob Text Qualitätskriterien erfüllt

        Args:
            text: Zu prüfender Text

        Returns:
            Tuple von (passed: bool, scores: dict)
        """
        if not text:
            return False, {}

        scores = {}

        # 1. Wort-Level Metriken
        words = text.split()
        if not words:
            return False, {}

        avg_word_length = sum(len(w) for w in words) / len(words)
        scores["avg_word_length"] = avg_word_length

        if avg_word_length < self.min_avg_word_length:
            logger.debug(f"Zu kurze durchschnittliche Wortlänge: {avg_word_length}")
            return False, scores

        if avg_word_length > self.max_avg_word_length:
            logger.debug(f"Zu lange durchschnittliche Wortlänge: {avg_word_length}")
            return False, scores

        # 2. Character-Level Metriken
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        upper_chars = sum(1 for c in text if c.isupper())

        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        digit_ratio = digit_chars / total_chars if total_chars > 0 else 0
        upper_ratio = upper_chars / alpha_chars if alpha_chars > 0 else 0

        scores["alpha_ratio"] = alpha_ratio
        scores["digit_ratio"] = digit_ratio
        scores["upper_ratio"] = upper_ratio

        if alpha_ratio < self.min_alpha_ratio:
            logger.debug(f"Zu wenig alphabetische Zeichen: {alpha_ratio}")
            return False, scores

        if digit_ratio > self.max_digit_ratio:
            logger.debug(f"Zu viele Ziffern: {digit_ratio}")
            return False, scores

        if upper_ratio > self.max_uppercase_ratio:
            logger.debug(f"Zu viele Großbuchstaben: {upper_ratio}")
            return False, scores

        # 3. Symbol-Ratio
        symbol_chars = sum(
            1 for c in text if not c.isalnum() and not c.isspace()
        )
        symbol_ratio = symbol_chars / total_chars if total_chars > 0 else 0
        scores["symbol_ratio"] = symbol_ratio

        if symbol_ratio > self.max_symbol_ratio:
            logger.debug(f"Zu viele Symbole: {symbol_ratio}")
            return False, scores

        # 4. Struktur-Metriken
        lines = text.split("\n")
        num_lines = len([l for l in lines if l.strip()])
        scores["num_lines"] = num_lines

        if num_lines < self.min_lines:
            logger.debug(f"Zu wenige Zeilen: {num_lines}")
            return False, scores

        # 5. URL-Ratio
        url_pattern = r"https?://\S+"
        urls = re.findall(url_pattern, text)
        url_chars = sum(len(url) for url in urls)
        url_ratio = url_chars / total_chars if total_chars > 0 else 0
        scores["url_ratio"] = url_ratio

        if url_ratio > self.max_url_ratio:
            logger.debug(f"Zu viele URLs: {url_ratio}")
            return False, scores

        # 6. Repetitionen
        line_rep_ratio = self._calculate_repetition_ratio(lines)
        scores["line_repetition"] = line_rep_ratio

        if line_rep_ratio > self.max_line_repetition:
            logger.debug(f"Zu viele wiederholte Zeilen: {line_rep_ratio}")
            return False, scores

        paragraphs = text.split("\n\n")
        para_rep_ratio = self._calculate_repetition_ratio(paragraphs)
        scores["paragraph_repetition"] = para_rep_ratio

        if para_rep_ratio > self.max_paragraph_repetition:
            logger.debug(f"Zu viele wiederholte Absätze: {para_rep_ratio}")
            return False, scores

        # Alle Tests bestanden
        return True, scores

    def _calculate_repetition_ratio(self, items: list) -> float:
        """
        Berechne Ratio von duplizierten Items

        Args:
            items: Liste von Strings (Zeilen oder Absätze)

        Returns:
            Ratio von Duplikaten (0.0 - 1.0)
        """
        if not items:
            return 0.0

        # Normalisiere (strip whitespace)
        normalized = [item.strip() for item in items if item.strip()]

        if not normalized:
            return 0.0

        # Zähle unique vs total
        unique_count = len(set(normalized))
        total_count = len(normalized)

        # Repetition ratio = (total - unique) / total
        return (total_count - unique_count) / total_count if total_count > 0 else 0.0
