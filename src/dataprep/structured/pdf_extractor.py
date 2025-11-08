"""
PDF Extraktor

Extrahiert Text und Struktur aus PDF-Dateien.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber nicht verfügbar")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 nicht verfügbar")


class PDFExtractor:
    """
    PDF Extraktor

    Extrahiert Text aus PDFs mit Layout-Erhaltung (optional).
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration
        """
        self.config = config or {}
        self.extract_layout = self.config.get("extract_layout", True)
        self.ocr_fallback = self.config.get("ocr_fallback", False)
        self.min_quality_score = self.config.get("min_quality_score", 0.7)

    def process(self, file_path: Path) -> Dict[str, Any]:
        """
        Verarbeite PDF-Datei

        Args:
            file_path: Pfad zur PDF-Datei

        Returns:
            Dict mit extrahiertem Content
        """
        result = {
            "source": str(file_path),
            "type": "pdf",
        }

        # Versuche pdfplumber (besser für Layout)
        if PDFPLUMBER_AVAILABLE and self.extract_layout:
            try:
                return self._extract_with_pdfplumber(file_path, result)
            except Exception as e:
                logger.warning(f"pdfplumber fehlgeschlagen: {e}, versuche PyPDF2")

        # Fallback: PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                return self._extract_with_pypdf2(file_path, result)
            except Exception as e:
                logger.error(f"PyPDF2 fehlgeschlagen: {e}")
                result["error"] = str(e)
                return result

        result["error"] = "Keine PDF-Bibliothek verfügbar"
        return result

    def _extract_with_pdfplumber(self, file_path: Path, result: Dict) -> Dict:
        """Extrahiere mit pdfplumber (erhält Layout)"""
        with pdfplumber.open(file_path) as pdf:
            result["num_pages"] = len(pdf.pages)

            pages_text = []
            tables = []

            for i, page in enumerate(pdf.pages):
                # Extrahiere Text
                text = page.extract_text()
                if text:
                    pages_text.append(text)

                # Extrahiere Tabellen
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        tables.append({
                            "page": i + 1,
                            "data": table,
                        })

            result["text"] = "\n\n".join(pages_text)
            result["tables"] = tables
            result["num_tables"] = len(tables)

        return result

    def _extract_with_pypdf2(self, file_path: Path, result: Dict) -> Dict:
        """Extrahiere mit PyPDF2 (einfacher)"""
        with open(file_path, "rb") as f:
            pdf = PdfReader(f)

            result["num_pages"] = len(pdf.pages)

            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

            result["text"] = "\n\n".join(pages_text)

            # Metadaten
            if pdf.metadata:
                result["metadata"] = {
                    "title": pdf.metadata.get("/Title", ""),
                    "author": pdf.metadata.get("/Author", ""),
                    "subject": pdf.metadata.get("/Subject", ""),
                    "creator": pdf.metadata.get("/Creator", ""),
                }

        return result
