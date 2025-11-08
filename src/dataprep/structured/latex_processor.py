"""
LaTeX Prozessor

Verarbeitet LaTeX-Dateien für LLM Pre-Training:
- Multi-File TeX-Quellen (\\input, \\include)
- Erhält mathematische Formeln
- Optional: nur Text-Extraktion
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from pylatexenc.latex2text import LatexNodes2Text
    PYLATEXENC_AVAILABLE = True
except ImportError:
    PYLATEXENC_AVAILABLE = False
    logger.warning("pylatexenc nicht verfügbar - LaTeX-Konvertierung eingeschränkt")


class LaTeXProcessor:
    """
    LaTeX Prozessor

    Verarbeitet LaTeX-Dateien, optional mit Text-Only Extraktion.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration
        """
        self.config = config or {}
        self.multi_file_support = self.config.get("multi_file_support", True)
        self.preserve_math = self.config.get("preserve_math", True)
        self.extract_text_only = self.config.get("extract_text_only", False)

    def process(self, file_path: Path) -> Dict[str, Any]:
        """
        Verarbeite LaTeX-Datei

        Args:
            file_path: Pfad zur LaTeX-Datei

        Returns:
            Dict mit verarbeitetem Content
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Multi-File Support
        if self.multi_file_support:
            content = self._resolve_includes(file_path, content)

        result = {
            "source": str(file_path),
            "type": "latex",
            "raw_content": content,
        }

        # Text-Extraktion
        if self.extract_text_only:
            text = self._extract_text(content)
            result["text"] = text
        else:
            result["content"] = content

        # Extrahiere Metadaten
        result["metadata"] = self._extract_metadata(content)

        # Extrahiere mathematische Formeln
        if self.preserve_math:
            formulas = self._extract_formulas(content)
            result["formulas"] = formulas
            result["num_formulas"] = len(formulas)

        return result

    def _resolve_includes(self, base_path: Path, content: str) -> str:
        """Resolv \\input und \\include Befehle"""
        base_dir = base_path.parent

        # \input{file} oder \include{file}
        pattern = r"\\(?:input|include)\{([^}]+)\}"

        def replace_include(match):
            filename = match.group(1)

            # Füge .tex hinzu falls keine Extension
            if not filename.endswith(".tex"):
                filename += ".tex"

            include_path = base_dir / filename

            if include_path.exists():
                try:
                    with open(include_path, "r", encoding="utf-8", errors="ignore") as f:
                        included_content = f.read()

                    # Rekursiv resolven
                    return self._resolve_includes(include_path, included_content)
                except Exception as e:
                    logger.warning(f"Konnte {include_path} nicht laden: {e}")
                    return match.group(0)
            else:
                logger.warning(f"Include-Datei nicht gefunden: {include_path}")
                return match.group(0)

        return re.sub(pattern, replace_include, content)

    def _extract_text(self, content: str) -> str:
        """Extrahiere nur Text (ohne LaTeX-Befehle)"""
        if PYLATEXENC_AVAILABLE:
            try:
                converter = LatexNodes2Text()
                return converter.latex_to_text(content)
            except Exception as e:
                logger.error(f"LaTeX-zu-Text Konvertierung fehlgeschlagen: {e}")

        # Fallback: Einfache Regex-basierte Extraktion
        # Entferne Kommentare
        content = re.sub(r"%.*$", "", content, flags=re.MULTILINE)

        # Entferne Preamble (bis \begin{document})
        match = re.search(r"\\begin\{document\}", content)
        if match:
            content = content[match.end():]

        # Entferne \end{document} und danach
        match = re.search(r"\\end\{document\}", content)
        if match:
            content = content[:match.start()]

        # Entferne LaTeX-Befehle (grob)
        content = re.sub(r"\\[a-zA-Z]+(\[.*?\])?\{.*?\}", "", content)
        content = re.sub(r"\\[a-zA-Z]+", "", content)

        # Entferne {}, [], etc.
        content = re.sub(r"[{}[\]]", "", content)

        return content.strip()

    def _extract_metadata(self, content: str) -> Dict[str, str]:
        """Extrahiere LaTeX-Metadaten (Titel, Autor, etc.)"""
        metadata = {}

        # \title{...}
        match = re.search(r"\\title\{([^}]+)\}", content)
        if match:
            metadata["title"] = match.group(1)

        # \author{...}
        match = re.search(r"\\author\{([^}]+)\}", content)
        if match:
            metadata["author"] = match.group(1)

        # \date{...}
        match = re.search(r"\\date\{([^}]+)\}", content)
        if match:
            metadata["date"] = match.group(1)

        return metadata

    def _extract_formulas(self, content: str) -> List[Dict[str, str]]:
        """Extrahiere mathematische Formeln"""
        formulas = []

        # Display math: $$...$$, \[...\], \begin{equation}...\end{equation}
        patterns = [
            (r"\$\$(.*?)\$\$", "display"),
            (r"\\\[(.*?)\\\]", "display"),
            (r"\\begin\{equation\}(.*?)\\end\{equation\}", "equation"),
            (r"\\begin\{align\}(.*?)\\end\{align\}", "align"),
        ]

        for pattern, formula_type in patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                formulas.append({
                    "type": formula_type,
                    "content": match.group(1).strip(),
                })

        # Inline math: $...$
        inline_matches = re.finditer(r"\$([^\$]+)\$", content)
        for match in inline_matches:
            formulas.append({
                "type": "inline",
                "content": match.group(1).strip(),
            })

        return formulas
