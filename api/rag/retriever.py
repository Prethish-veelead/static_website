import logging
import pickle
from typing import Any, Dict, List

import faiss
import numpy as np

from rag.embeddings import get_embeddings
from rag.paths import INDEX_FILE, METADATA_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """Singleton retriever for FAISS index."""

    def __init__(self):
        self.index = None
        self.metadata = []
        self.embeddings = None
        self._load_index()

    def _get_embeddings(self):
        """Get embeddings based on provider."""
        if self.embeddings is not None:
            return self.embeddings

        self.embeddings = get_embeddings()
        return self.embeddings

    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            if INDEX_FILE.exists() and METADATA_FILE.exists():
                self.index = faiss.read_index(str(INDEX_FILE))
                with open(METADATA_FILE, "rb") as file_obj:
                    self.metadata = pickle.load(file_obj)
                logger.info("Index loaded with %s embeddings", self.index.ntotal)
            else:
                logger.warning("Index files not found at %s", INDEX_FILE)
        except Exception as exc:
            logger.error("Error loading index: %s", exc)

    def is_index_loaded(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def get_chunk_count(self) -> int:
        return self.index.ntotal if self.index else 0

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for query."""
        if not self.is_index_loaded():
            raise RuntimeError("Run POST /api/ingest first to create the index")

        try:
            embeddings = self._get_embeddings()
            query_embedding = embeddings.embed_query(query)
            query_array = np.array([query_embedding]).astype("float32")

            _, indices = self.index.search(query_array, top_k)

            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.metadata):
                    chunk_data = self.metadata[idx].copy()
                    results.append(chunk_data)

            logger.info("Retrieved %s chunks for query", len(results))
            return results
        except Exception as exc:
            logger.error("Error during retrieval: %s", exc)
            raise
