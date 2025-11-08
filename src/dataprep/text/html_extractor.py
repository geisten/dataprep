"""
HTML-zu-Text Extraktor

Basierend auf Phi-4 Ansatz:
- Erhält fragile Inhalte (TeX/MathML Formeln, Code-Blöcke, Tabellen)
- Entfernt Boilerplate und Werbung
- Erhält Thread-Struktur in Foren
"""

import re
import logging
from typing import Dict, Optional
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)


class HTMLExtractor:
    """
    Fortgeschrittener HTML-zu-Text Extraktor

    Features:
    - Erhält mathematische Formeln (TeX, MathML)
    - Erhält Code-Blöcke mit Formatierung
    - Erhält Tabellen-Struktur
    - Entfernt Boilerplate/Ads
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration (preserve_formulas, preserve_code, etc.)
        """
        self.config = config or {}
        self.preserve_formulas = self.config.get("preserve_formulas", True)
        self.preserve_code = self.config.get("preserve_code", True)
        self.preserve_tables = self.config.get("preserve_tables", True)
        self.remove_boilerplate = self.config.get("remove_boilerplate", True)
        self.remove_ads = self.config.get("remove_ads", True)

        # Bekannte Boilerplate/Ad-Klassen und IDs
        self.boilerplate_patterns = [
            r"^nav",
            r"menu",
            r"sidebar",
            r"footer",
            r"header",
            r"cookie",
            r"advertisement",
            r"ad[-_]",
            r"sponsor",
            r"social",
            r"share",
            r"comment-form",
            r"widget",
        ]

    def extract(self, html: str) -> str:
        """
        Extrahiere Text aus HTML

        Args:
            html: HTML-String

        Returns:
            Extrahierter Text
        """
        soup = BeautifulSoup(html, "lxml")

        # 1. Entferne Kommentare
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 2. Entferne Script/Style Tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # 3. Entferne Boilerplate
        if self.remove_boilerplate:
            self._remove_boilerplate(soup)

        # 4. Verarbeite spezielle Elemente
        if self.preserve_formulas:
            self._preserve_formulas(soup)

        if self.preserve_code:
            self._preserve_code_blocks(soup)

        if self.preserve_tables:
            self._preserve_tables(soup)

        # 5. Extrahiere Text
        text = soup.get_text(separator="\n", strip=True)

        # 6. Cleanup
        text = self._cleanup_text(text)

        return text

    def _remove_boilerplate(self, soup: BeautifulSoup):
        """Entferne Boilerplate-Elemente"""
        for pattern in self.boilerplate_patterns:
            # Nach Klasse
            for tag in soup.find_all(class_=re.compile(pattern, re.I)):
                tag.decompose()

            # Nach ID
            for tag in soup.find_all(id=re.compile(pattern, re.I)):
                tag.decompose()

    def _preserve_formulas(self, soup: BeautifulSoup):
        """
        Erhält mathematische Formeln (TeX, MathML)

        Konvertiert zu Markdown-kompatiblem Format
        """
        # MathML
        for math_tag in soup.find_all("math"):
            # Versuche LaTeX-Attribut zu finden
            latex = math_tag.get("alttext") or math_tag.get("data-latex")
            if latex:
                math_tag.string = f"$${latex}$$"
            else:
                # Fallback: Behalte MathML-Inhalt
                pass

        # Inline TeX (oft in spans mit class="tex" oder ähnlich)
        for tag in soup.find_all(class_=re.compile(r"tex|math|formula", re.I)):
            text = tag.get_text()
            if text and not text.startswith("$"):
                tag.string = f"${text}$"

        # Display TeX (oft in divs)
        for tag in soup.find_all("div", class_=re.compile(r"display.*math", re.I)):
            text = tag.get_text()
            if text and not text.startswith("$$"):
                tag.string = f"$${text}$$"

    def _preserve_code_blocks(self, soup: BeautifulSoup):
        """
        Erhält Code-Blöcke

        Konvertiert zu Markdown Code-Blöcken
        """
        # <pre><code> Blöcke
        for pre in soup.find_all("pre"):
            code = pre.find("code")
            if code:
                lang = ""
                # Versuche Sprache zu extrahieren
                classes = code.get("class", [])
                for cls in classes:
                    if cls.startswith("language-"):
                        lang = cls.replace("language-", "")
                        break
                    elif cls.startswith("lang-"):
                        lang = cls.replace("lang-", "")
                        break

                code_text = code.get_text()
                pre.string = f"\n```{lang}\n{code_text}\n```\n"
            else:
                # Nur <pre> ohne <code>
                pre_text = pre.get_text()
                pre.string = f"\n```\n{pre_text}\n```\n"

        # Inline code (<code> ohne <pre>)
        for code in soup.find_all("code"):
            if not code.find_parent("pre"):
                code_text = code.get_text()
                code.string = f"`{code_text}`"

    def _preserve_tables(self, soup: BeautifulSoup):
        """
        Erhält Tabellen-Struktur

        Konvertiert zu Markdown-Tabellen (einfache Version)
        """
        for table in soup.find_all("table"):
            markdown_table = self._table_to_markdown(table)
            if markdown_table:
                table.string = f"\n{markdown_table}\n"

    def _table_to_markdown(self, table) -> Optional[str]:
        """Konvertiere HTML-Tabelle zu Markdown"""
        rows = []

        # Header
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
                if headers:
                    rows.append("| " + " | ".join(headers) + " |")
                    rows.append("|" + "|".join(["---"] * len(headers)) + "|")

        # Body
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr", recursive=False):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows) if rows else None

    def _cleanup_text(self, text: str) -> str:
        """Bereinige extrahierten Text"""
        # Entferne mehrfache Leerzeilen
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Entferne trailing/leading whitespace
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # Entferne zu viele Leerzeichen
        text = re.sub(r" +", " ", text)

        return text.strip()
