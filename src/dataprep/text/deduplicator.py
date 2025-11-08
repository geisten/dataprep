"""
Deduplizierung für LLM Pre-Training Daten

Implementiert drei Ansätze:
1. Exact Deduplication: Exakte String-Matches (sharded)
2. Fuzzy Deduplication: MinHash LSH für ähnliche Dokumente
3. Soft Deduplication: Reweighting statt Löschen (SoftDedup Paper)
"""

import hashlib
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    logging.warning("xxhash nicht verfügbar - verwende hashlib")

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    logging.warning("datasketch nicht verfügbar - Fuzzy Deduplication deaktiviert")

logger = logging.getLogger(__name__)


class ExactDeduplicator:
    """
    Exact Deduplication mit Sharding

    Verwendet in GneissWeb (10T tokens) und Phi-3
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration
        """
        self.config = config or {}
        self.sharded = self.config.get("sharded", True)
        self.seen_hashes: Set[str] = set()

    def deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Entferne exakte Duplikate

        Args:
            documents: Liste von Dokumenten mit 'text' Feld

        Returns:
            Deduplizierte Dokumente
        """
        unique_docs = []
        duplicates = 0

        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue

            doc_hash = self._hash_text(text)

            if doc_hash not in self.seen_hashes:
                self.seen_hashes.add(doc_hash)
                unique_docs.append(doc)
            else:
                duplicates += 1

        logger.info(
            f"Exact deduplication: {len(documents)} -> {len(unique_docs)} "
            f"({duplicates} Duplikate entfernt)"
        )

        return unique_docs

    def _hash_text(self, text: str) -> str:
        """Erstelle Hash für Text"""
        if XXHASH_AVAILABLE:
            return xxhash.xxh64(text.encode()).hexdigest()
        else:
            return hashlib.sha256(text.encode()).hexdigest()

    def reset(self):
        """Lösche gespeicherte Hashes"""
        self.seen_hashes.clear()


class FuzzyDeduplicator:
    """
    Fuzzy Deduplication mit MinHash LSH

    Findet ähnliche (nicht nur exakte) Dokumente
    Verwendet in Nemotron-CC und modernen Pipelines
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration
        """
        if not DATASKETCH_AVAILABLE:
            raise ImportError("datasketch erforderlich für Fuzzy Deduplication")

        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.8)
        self.ngram_size = self.config.get("ngram_size", 5)
        self.num_perm = self.config.get("num_perm", 128)

        # MinHash LSH Index
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.doc_minhashes = {}

    def deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Entferne ähnliche Dokumente

        Args:
            documents: Liste von Dokumenten mit 'text' Feld

        Returns:
            Deduplizierte Dokumente
        """
        unique_docs = []
        duplicates = 0

        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            if not text:
                continue

            # Erstelle MinHash
            minhash = self._create_minhash(text)

            # Prüfe auf ähnliche Dokumente
            similar = self.lsh.query(minhash)

            if not similar:
                # Kein ähnliches Dokument gefunden -> behalten
                doc_id = f"doc_{i}"
                self.lsh.insert(doc_id, minhash)
                self.doc_minhashes[doc_id] = minhash
                unique_docs.append(doc)
            else:
                duplicates += 1

        logger.info(
            f"Fuzzy deduplication (threshold={self.threshold}): "
            f"{len(documents)} -> {len(unique_docs)} ({duplicates} ähnliche entfernt)"
        )

        return unique_docs

    def _create_minhash(self, text: str) -> MinHash:
        """Erstelle MinHash für Text"""
        minhash = MinHash(num_perm=self.num_perm)

        # N-gram Shingling
        tokens = self._get_ngrams(text, self.ngram_size)

        for token in tokens:
            minhash.update(token.encode())

        return minhash

    def _get_ngrams(self, text: str, n: int) -> List[str]:
        """Erstelle character-level N-grams"""
        # Normalisiere Text
        text = text.lower()
        text = "".join(text.split())  # Entferne Whitespace

        # Erstelle N-grams
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def reset(self):
        """Lösche LSH Index"""
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.doc_minhashes.clear()


class SoftDeduplicator:
    """
    Soft Deduplication (SoftDedup Paper, 2024)

    Statt Duplikate zu löschen, werden sie reweighted.
    Vorteile:
    - Erhält Information
    - Reduziert Redundanz
    - 26% weniger Training Steps bei gleicher Perplexity
    - ~1.8% bessere Downstream Accuracy

    Paper: "Improving Pretraining Data Using Perplexity Correlations"
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Konfiguration
        """
        if not DATASKETCH_AVAILABLE:
            raise ImportError("datasketch erforderlich für Soft Deduplication")

        self.config = config or {}
        self.reweight_factor = self.config.get("reweight_factor", 0.5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.85)
        self.num_perm = self.config.get("num_perm", 128)

        # Für Clustering ähnlicher Dokumente
        self.clusters = defaultdict(list)
        self.doc_to_cluster = {}

    def deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Soft Deduplication - reweightet statt zu löschen

        Args:
            documents: Liste von Dokumenten mit 'text' Feld

        Returns:
            Dokumente mit 'weight' Feld (0.0 - 1.0)
        """
        # 1. Erstelle MinHash für alle Dokumente
        doc_minhashes = []
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            if text:
                minhash = self._create_minhash(text)
                doc_minhashes.append((i, minhash))

        # 2. Finde Cluster ähnlicher Dokumente
        self._cluster_documents(doc_minhashes)

        # 3. Berechne Weights basierend auf Cluster-Größe
        weighted_docs = []
        for i, doc in enumerate(documents):
            cluster_id = self.doc_to_cluster.get(i)

            if cluster_id is None:
                # Dokument nicht geclustert (unique)
                weight = 1.0
            else:
                # Dokument in Cluster - reweight basierend auf Cluster-Größe
                cluster_size = len(self.clusters[cluster_id])
                # Je größer das Cluster, desto niedriger das Weight
                weight = max(self.reweight_factor, 1.0 / cluster_size)

            doc_with_weight = doc.copy()
            doc_with_weight["weight"] = weight
            weighted_docs.append(doc_with_weight)

        # Statistiken
        unique_count = sum(1 for d in weighted_docs if d["weight"] == 1.0)
        reweighted_count = len(weighted_docs) - unique_count

        logger.info(
            f"Soft deduplication: {len(documents)} Dokumente, "
            f"{unique_count} unique, {reweighted_count} reweighted, "
            f"{len(self.clusters)} Cluster"
        )

        return weighted_docs

    def _create_minhash(self, text: str) -> MinHash:
        """Erstelle MinHash für Text"""
        minhash = MinHash(num_perm=self.num_perm)

        # Character-level 5-grams
        text = text.lower()
        text = "".join(text.split())

        for i in range(len(text) - 4):
            ngram = text[i : i + 5]
            minhash.update(ngram.encode())

        return minhash

    def _cluster_documents(self, doc_minhashes: List[Tuple[int, MinHash]]):
        """Clustere ähnliche Dokumente"""
        # Verwende LSH für effizientes Clustering
        lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=self.num_perm)

        cluster_counter = 0

        for doc_idx, minhash in doc_minhashes:
            # Finde ähnliche Dokumente
            similar = lsh.query(minhash)

            if similar:
                # Füge zu existierendem Cluster hinzu
                # Nimm das erste gefundene ähnliche Dokument
                similar_doc_id = similar[0]
                cluster_id = self.doc_to_cluster.get(similar_doc_id)

                if cluster_id is not None:
                    self.clusters[cluster_id].append(doc_idx)
                    self.doc_to_cluster[doc_idx] = cluster_id
            else:
                # Erstelle neues Cluster
                cluster_id = f"cluster_{cluster_counter}"
                cluster_counter += 1

                self.clusters[cluster_id].append(doc_idx)
                self.doc_to_cluster[doc_idx] = cluster_id

            # Füge zu LSH hinzu
            lsh.insert(doc_idx, minhash)

    def reset(self):
        """Lösche Cluster-Information"""
        self.clusters.clear()
        self.doc_to_cluster.clear()


def deduplicate_documents(
    documents: List[Dict[str, Any]],
    method: str = "soft",
    config: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience-Funktion für Deduplizierung

    Args:
        documents: Liste von Dokumenten
        method: 'exact', 'fuzzy', oder 'soft'
        config: Konfiguration

    Returns:
        Deduplizierte Dokumente
    """
    if method == "exact":
        deduplicator = ExactDeduplicator(config)
    elif method == "fuzzy":
        deduplicator = FuzzyDeduplicator(config)
    elif method == "soft":
        deduplicator = SoftDeduplicator(config)
    else:
        raise ValueError(f"Unbekannte Methode: {method}")

    return deduplicator.deduplicate(documents)
