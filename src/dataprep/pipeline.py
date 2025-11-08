"""
Hauptpipeline für LLM Pre-Training Datenaufbereitung

Implementiert den zweiphasigen Ansatz von Phi-3:
- Phase 1: Web-Daten für allgemeines Wissen
- Phase 2: Gefilterte Web-Daten + synthetische Daten für Reasoning
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
from tqdm import tqdm

from .text.html_extractor import HTMLExtractor
from .text.cleaner import TextCleaner
from .text.deduplicator import ExactDeduplicator, FuzzyDeduplicator, SoftDeduplicator
from .quality.filter import QualityFilter
from .quality.scorer import QualityScorer

logger = logging.getLogger(__name__)


class BasePipeline:
    """Basis-Klasse für alle Pipelines"""

    def __init__(self, config: Optional[Union[str, Dict]] = None):
        """
        Args:
            config: Pfad zur Konfigurationsdatei oder Dict mit Konfiguration
        """
        if config is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, str):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        self._setup_logging()

    def _setup_logging(self):
        """Konfiguriere Logging"""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        if log_config.get("file"):
            log_path = Path(log_config["file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)


class TextPipeline(BasePipeline):
    """
    Pipeline für Text-Datenverarbeitung

    Implementiert:
    - HTML-zu-Text Extraktion (mit Formel/Code/Tabellen-Erhalt)
    - Text-Reinigung und Normalisierung
    - Deduplizierung (Exact, Fuzzy, Soft)
    - Qualitätsfilterung
    - Toxizitätsfilterung
    """

    def __init__(
        self,
        config: Optional[Union[str, Dict]] = None,
        deduplication: str = "soft",
        quality_filter: bool = True,
        toxicity_filter: bool = True,
    ):
        """
        Args:
            config: Konfiguration
            deduplication: Art der Deduplizierung ('exact', 'fuzzy', 'soft')
            quality_filter: Ob Qualitätsfilter angewendet werden soll
            toxicity_filter: Ob Toxizitätsfilter angewendet werden soll
        """
        super().__init__(config)

        # Initialisiere Komponenten
        self.html_extractor = HTMLExtractor(self.config.get("text", {}).get("html", {}))
        self.cleaner = TextCleaner(self.config.get("text", {}).get("cleaning", {}))

        # Deduplizierung
        self.deduplication_mode = deduplication
        self.deduplicators = {}
        if deduplication in ["exact", "soft"]:
            self.deduplicators["exact"] = ExactDeduplicator(
                self.config.get("deduplication", {}).get("exact", {})
            )
        if deduplication in ["fuzzy", "soft"]:
            self.deduplicators["fuzzy"] = FuzzyDeduplicator(
                self.config.get("deduplication", {}).get("fuzzy", {})
            )
        if deduplication == "soft":
            self.deduplicators["soft"] = SoftDeduplicator(
                self.config.get("deduplication", {}).get("soft", {})
            )

        # Filter
        self.use_quality_filter = quality_filter
        self.use_toxicity_filter = toxicity_filter

        if quality_filter:
            self.quality_filter = QualityFilter(self.config.get("quality", {}))

        if toxicity_filter:
            self.quality_scorer = QualityScorer(self.config.get("quality", {}))

        logger.info(f"TextPipeline initialisiert (deduplication={deduplication})")

    def process(
        self,
        source: Union[str, Path, List[str]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Verarbeite Text-Daten

        Args:
            source: Eingabedatei, Verzeichnis oder Liste von Dateien
            output_dir: Ausgabeverzeichnis (optional)

        Returns:
            Verarbeitete Daten
        """
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.is_file():
                return self._process_file(source_path, output_dir)
            elif source_path.is_dir():
                return self._process_directory(source_path, output_dir)
        elif isinstance(source, list):
            return [self._process_file(Path(f), output_dir) for f in source]

        raise ValueError(f"Ungültige Quelle: {source}")

    def _process_file(
        self, file_path: Path, output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Verarbeite eine einzelne Datei"""
        logger.info(f"Verarbeite: {file_path}")

        # 1. Extraktion (HTML -> Text)
        if file_path.suffix in [".html", ".htm"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text = self.html_extractor.extract(f.read())
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # 2. Reinigung
        text = self.cleaner.clean(text)

        # 3. Qualitätsfilter
        if self.use_quality_filter:
            passed, scores = self.quality_filter.filter(text)
            if not passed:
                logger.debug(f"Dokument nicht bestanden: {file_path}")
                return {
                    "source": str(file_path),
                    "filtered": True,
                    "reason": "quality",
                    "scores": scores,
                }

        # 4. Toxizitätsfilter
        if self.use_toxicity_filter:
            toxicity_scores = self.quality_scorer.score_toxicity(text)
            max_toxicity = self.config.get("quality", {}).get("toxicity", {}).get("max_toxicity", 0.5)
            if toxicity_scores.get("toxicity", 0) > max_toxicity:
                logger.debug(f"Dokument zu toxisch: {file_path}")
                return {
                    "source": str(file_path),
                    "filtered": True,
                    "reason": "toxicity",
                    "toxicity_scores": toxicity_scores,
                }

        result = {
            "source": str(file_path),
            "text": text,
            "filtered": False,
            "length": len(text),
        }

        if self.use_quality_filter:
            result["quality_scores"] = scores

        if self.use_toxicity_filter:
            result["toxicity_scores"] = toxicity_scores

        return result

    def _process_directory(
        self, dir_path: Path, output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Verarbeite alle Dateien in einem Verzeichnis"""
        files = list(dir_path.rglob("*"))
        files = [f for f in files if f.is_file()]

        results = []
        for file_path in tqdm(files, desc="Verarbeite Dateien"):
            try:
                result = self._process_file(file_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Fehler bei {file_path}: {e}")
                results.append({
                    "source": str(file_path),
                    "error": str(e),
                })

        return results

    def deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dedupliziere eine Liste von Dokumenten

        Args:
            documents: Liste von Dokumenten mit 'text' Feld

        Returns:
            Deduplizierte Dokumente
        """
        if not self.deduplicators:
            return documents

        logger.info(f"Dedupliziere {len(documents)} Dokumente (Methode: {self.deduplication_mode})")

        if self.deduplication_mode == "exact":
            return self.deduplicators["exact"].deduplicate(documents)
        elif self.deduplication_mode == "fuzzy":
            return self.deduplicators["fuzzy"].deduplicate(documents)
        elif self.deduplication_mode == "soft":
            return self.deduplicators["soft"].deduplicate(documents)

        return documents


class MultimodalPipeline(BasePipeline):
    """
    Pipeline für multimodale Datenverarbeitung

    Unterstützt:
    - Bilder (JPEG, PNG, WebP)
    - Videos (MP4, AVI, WebM)
    - Vision-Text Alignment
    """

    def __init__(
        self,
        config: Optional[Union[str, Dict]] = None,
        image_size: int = 224,
        video_fps: int = 1,
    ):
        """
        Args:
            config: Konfiguration
            image_size: Zielgröße für Bilder
            video_fps: Frames pro Sekunde für Video-Extraktion
        """
        super().__init__(config)

        self.image_size = image_size
        self.video_fps = video_fps

        # Lade Module lazy (nur wenn benötigt)
        self._image_processor = None
        self._video_processor = None

        logger.info(f"MultimodalPipeline initialisiert (image_size={image_size}, video_fps={video_fps})")

    @property
    def image_processor(self):
        """Lazy loading des Image Processors"""
        if self._image_processor is None:
            from .multimodal.image_processor import ImageProcessor
            self._image_processor = ImageProcessor(
                self.config.get("multimodal", {}).get("image", {}),
                target_size=self.image_size,
            )
        return self._image_processor

    @property
    def video_processor(self):
        """Lazy loading des Video Processors"""
        if self._video_processor is None:
            from .multimodal.video_processor import VideoProcessor
            self._video_processor = VideoProcessor(
                self.config.get("multimodal", {}).get("video", {}),
                fps=self.video_fps,
            )
        return self._video_processor

    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Verarbeite ein Bild"""
        return self.image_processor.process(image_path)

    def process_video(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Verarbeite ein Video"""
        return self.video_processor.process(video_path)

    def process_image_text(
        self,
        image_path: Union[str, Path],
        text: str,
    ) -> Dict[str, Any]:
        """
        Verarbeite ein Bild-Text Paar

        Args:
            image_path: Pfad zum Bild
            text: Zugehöriger Text/Caption

        Returns:
            Verarbeitetes Bild-Text Paar
        """
        image_result = self.process_image(image_path)

        return {
            "image": image_result,
            "text": text,
            "source": str(image_path),
        }


class CompletePipeline(BasePipeline):
    """
    Vollständige Pipeline für alle Datentypen

    Kombiniert:
    - Text-Pipeline
    - Multimodal-Pipeline
    - Strukturierte Formate (Markdown, LaTeX, PDF)
    """

    def __init__(self, config: Optional[Union[str, Dict]] = None):
        super().__init__(config)

        self.text_pipeline = TextPipeline(config)
        self.multimodal_pipeline = MultimodalPipeline(config)

        # Lazy loading für strukturierte Formate
        self._markdown_processor = None
        self._latex_processor = None
        self._pdf_extractor = None

        logger.info("CompletePipeline initialisiert")

    def process(
        self,
        source: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Verarbeite beliebige Datenquelle (auto-detect)

        Args:
            source: Eingabedatei
            output_dir: Ausgabeverzeichnis

        Returns:
            Verarbeitete Daten
        """
        source_path = Path(source)
        suffix = source_path.suffix.lower()

        # Text/HTML
        if suffix in [".txt", ".html", ".htm"]:
            return self.text_pipeline.process(source_path, output_dir)

        # Markdown
        if suffix in [".md", ".markdown"]:
            return self._process_markdown(source_path)

        # LaTeX
        if suffix in [".tex", ".latex"]:
            return self._process_latex(source_path)

        # PDF
        if suffix == ".pdf":
            return self._process_pdf(source_path)

        # Bilder
        if suffix in [".jpg", ".jpeg", ".png", ".webp"]:
            return self.multimodal_pipeline.process_image(source_path)

        # Videos
        if suffix in [".mp4", ".avi", ".webm", ".mov"]:
            return self.multimodal_pipeline.process_video(source_path)

        raise ValueError(f"Nicht unterstütztes Format: {suffix}")

    def _process_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Verarbeite Markdown"""
        if self._markdown_processor is None:
            from .structured.markdown_processor import MarkdownProcessor
            self._markdown_processor = MarkdownProcessor(
                self.config.get("structured", {}).get("markdown", {})
            )
        return self._markdown_processor.process(file_path)

    def _process_latex(self, file_path: Path) -> Dict[str, Any]:
        """Verarbeite LaTeX"""
        if self._latex_processor is None:
            from .structured.latex_processor import LaTeXProcessor
            self._latex_processor = LaTeXProcessor(
                self.config.get("structured", {}).get("latex", {})
            )
        return self._latex_processor.process(file_path)

    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Verarbeite PDF"""
        if self._pdf_extractor is None:
            from .structured.pdf_extractor import PDFExtractor
            self._pdf_extractor = PDFExtractor(
                self.config.get("structured", {}).get("pdf", {})
            )
        return self._pdf_extractor.process(file_path)
