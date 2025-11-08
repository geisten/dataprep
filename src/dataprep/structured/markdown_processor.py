"""
Markdown Prozessor

Verarbeitet Markdown-Dateien für LLM Pre-Training:
- Erhält Struktur (Headers, Listen, etc.)
- Extrahiert Code-Blöcke separat
- Erhält Links und Referenzen
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """
    Markdown Prozessor

    Verarbeitet Markdown-Dateien während die Struktur erhalten bleibt.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration
        """
        self.config = config or {}
        self.preserve_structure = self.config.get("preserve_structure", True)
        self.extract_code_blocks = self.config.get("extract_code_blocks", True)
        self.preserve_links = self.config.get("preserve_links", True)

    def process(self, file_path: Path) -> Dict[str, Any]:
        """
        Verarbeite Markdown-Datei

        Args:
            file_path: Pfad zur Markdown-Datei

        Returns:
            Dict mit verarbeitetem Content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        result = {
            "source": str(file_path),
            "type": "markdown",
            "content": content,
        }

        # Extrahiere Metadaten
        result["metadata"] = self._extract_metadata(content)

        # Extrahiere Code-Blöcke
        if self.extract_code_blocks:
            code_blocks = self._extract_code_blocks(content)
            result["code_blocks"] = code_blocks
            result["num_code_blocks"] = len(code_blocks)

        # Extrahiere Links
        if self.preserve_links:
            links = self._extract_links(content)
            result["links"] = links
            result["num_links"] = len(links)

        # Strukturanalyse
        result["structure"] = self._analyze_structure(content)

        return result

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extrahiere YAML Front Matter (falls vorhanden)"""
        metadata = {}

        # YAML Front Matter: ---\n...\n---
        yaml_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(yaml_pattern, content, re.DOTALL)

        if match:
            yaml_content = match.group(1)
            # Einfaches Parsing (ohne YAML-Library)
            for line in yaml_content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

        return metadata

    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extrahiere Code-Blöcke"""
        code_blocks = []

        # Fenced code blocks: ```lang\ncode\n```
        pattern = r"```(\w*)\n(.*?)\n```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            lang = match.group(1) or "plain"
            code = match.group(2)

            code_blocks.append({
                "language": lang,
                "code": code,
            })

        return code_blocks

    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extrahiere Links"""
        links = []

        # Markdown links: [text](url)
        pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        matches = re.finditer(pattern, content)

        for match in matches:
            text = match.group(1)
            url = match.group(2)

            links.append({
                "text": text,
                "url": url,
            })

        return links

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analysiere Markdown-Struktur"""
        structure = {
            "headers": [],
            "num_paragraphs": 0,
            "num_lists": 0,
            "num_blockquotes": 0,
        }

        lines = content.split("\n")

        # Headers
        for line in lines:
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("#").strip()
                structure["headers"].append({
                    "level": level,
                    "text": text,
                })

        # Paragraphs (grobe Schätzung)
        paragraphs = content.split("\n\n")
        structure["num_paragraphs"] = len([p for p in paragraphs if p.strip()])

        # Lists
        list_pattern = r"^[\*\-\+]\s+|\d+\.\s+"
        structure["num_lists"] = sum(
            1 for line in lines if re.match(list_pattern, line.strip())
        )

        # Blockquotes
        structure["num_blockquotes"] = sum(
            1 for line in lines if line.strip().startswith(">")
        )

        return structure
