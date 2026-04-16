import logging
import pickle
from typing import Any, Dict, List

import faiss
import fitz
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag.embeddings import get_embeddings
from rag.paths import INDEX_DIR, INDEX_FILE, METADATA_FILE, PDF_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf() -> List[Dict[str, Any]]:
    """Extract text from PDF page by page."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    logger.info("Extracting text from %s", PDF_PATH)
    pages_data = []

    try:
        doc = fitz.open(PDF_PATH)
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages_data.append(
                    {
                        "text": text,
                        "page": page_num,
                        "source": PDF_PATH.name,
                    }
                )
        doc.close()
        logger.info("Extracted %s pages", len(pages_data))
        return pages_data
    except Exception as exc:
        logger.error("Error extracting text from PDF: %s", exc)
        raise


def chunk_pages(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk pages using RecursiveCharacterTextSplitter."""
    logger.info("Chunking pages...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for page_data in pages_data:
        text = page_data["text"]
        page_num = page_data["page"]
        source = page_data["source"]

        split_texts = splitter.split_text(text)
        for chunk_idx, chunk_text in enumerate(split_texts):
            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_num,
                    "source": source,
                    "chunk_id": f"{page_num}_{chunk_idx}",
                }
            )

    logger.info("Created %s chunks", len(chunks))
    return chunks


def create_faiss_index(chunks: List[Dict[str, Any]]) -> tuple:
    """Create FAISS index from chunks."""
    logger.info("Creating FAISS index...")

    embeddings = get_embeddings()
    texts = [chunk["text"] for chunk in chunks]

    logger.info("Generating embeddings for %s chunks...", len(texts))
    embeddings_list = embeddings.embed_documents(texts)
    embeddings_array = np.array(embeddings_list).astype("float32")

    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    logger.info("FAISS index created with %s embeddings", index.ntotal)
    return index, chunks


def save_index(index: faiss.IndexFlatL2, metadata: List[Dict[str, Any]]):
    """Save FAISS index and metadata."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    logger.info("Index saved to %s", INDEX_FILE)

    with open(METADATA_FILE, "wb") as file_obj:
        pickle.dump(metadata, file_obj)
    logger.info("Metadata saved to %s", METADATA_FILE)


def ingest_pdf() -> int:
    """Main ingestion pipeline."""
    try:
        pages = extract_text_from_pdf()
        chunks = chunk_pages(pages)
        index, chunk_metadata = create_faiss_index(chunks)
        save_index(index, chunk_metadata)

        logger.info("Ingestion complete: %s chunks indexed", len(chunks))
        return len(chunks)
    except Exception as exc:
        logger.error("Ingest pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    ingest_pdf()
