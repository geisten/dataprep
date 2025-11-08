"""
Text-Reinigung und Normalisierung

Basierend auf Standard-LLM-Preprocessing Best Practices
"""

import re
import logging
import unicodedata
from typing import Dict, Optional

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect nicht verfügbar - Spracherkennung deaktiviert")

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Text-Reiniger

    Features:
    - Unicode-Normalisierung
    - Encoding-Fixes
    - Control-Character Entfernung
    - Längenfilterung
    - Sprachfilterung
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration
        """
        self.config = config or {}
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.fix_encoding = self.config.get("fix_encoding", True)
        self.remove_control_chars = self.config.get("remove_control_chars", True)
        self.min_length = self.config.get("min_length", 100)
        self.max_length = self.config.get("max_length", 1000000)

    def clean(self, text: str) -> str:
        """
        Reinige Text

        Args:
            text: Roher Text

        Returns:
            Gereinigter Text
        """
        if not text:
            return ""

        # 1. Unicode-Normalisierung
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        # 2. Encoding-Fixes
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # 3. Control-Characters entfernen
        if self.remove_control_chars:
            text = self._remove_control_chars(text)

        # 4. Whitespace-Normalisierung
        text = self._normalize_whitespace(text)

        # 5. Längenprüfung
        if len(text) < self.min_length or len(text) > self.max_length:
            logger.debug(f"Text zu kurz/lang: {len(text)} Zeichen")
            return ""

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalisiere Unicode (NFC)"""
        return unicodedata.normalize("NFC", text)

    def _fix_encoding(self, text: str) -> str:
        """
        Fixe häufige Encoding-Probleme

        Z.B. falsch dekodiertes UTF-8
        """
        # Häufige mojibake patterns
        replacements = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "â€"": "—",
            "â€"": "–",
            "Ã©": "é",
            "Ã¨": "è",
            "Ã¼": "ü",
            "Ã¶": "ö",
            "Ã¤": "ä",
            "ÃŸ": "ß",
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        return text

    def _remove_control_chars(self, text: str) -> str:
        """Entferne Control-Characters (außer \n, \t, \r)"""
        # Behalte nur printable characters + newline/tab
        return "".join(
            char
            for char in text
            if char in ["\n", "\t", "\r"] or not unicodedata.category(char).startswith("C")
        )

    def _normalize_whitespace(self, text: str) -> str:
        """Normalisiere Whitespace"""
        # Ersetze verschiedene Whitespace-Chars durch normale Spaces
        text = re.sub(r"[\xa0\u1680\u2000-\u200b\u202f\u205f\u3000]", " ", text)

        # Entferne trailing whitespace pro Zeile
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # Reduziere multiple spaces zu einem
        text = re.sub(r" +", " ", text)

        # Reduziere multiple newlines zu max 2
        text = re.sub(r"\n\n+", "\n\n", text)

        return text.strip()

    def detect_language(self, text: str) -> Optional[str]:
        """
        Erkenne Sprache des Texts

        Args:
            text: Text

        Returns:
            ISO 639-1 Sprachcode (z.B. 'de', 'en') oder None
        """
        if not LANGDETECT_AVAILABLE:
            return None

        try:
            return langdetect.detect(text)
        except Exception as e:
            logger.debug(f"Spracherkennung fehlgeschlagen: {e}")
            return None

    def is_language_acceptable(self, text: str, keep_languages: list) -> bool:
        """
        Prüfe ob Text in einer akzeptierten Sprache ist

        Args:
            text: Text
            keep_languages: Liste von ISO 639-1 Sprachcodes

        Returns:
            True wenn Sprache akzeptabel
        """
        if not LANGDETECT_AVAILABLE or not keep_languages:
            return True

        lang = self.detect_language(text)
        if lang is None:
            return True  # Bei Fehler akzeptieren

        return lang in keep_languages
