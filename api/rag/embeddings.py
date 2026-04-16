import logging
import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class AzureOpenAIEmbeddingClient:
    """Small adapter with the embed_* methods expected by the RAG pipeline."""

    def __init__(self) -> None:
        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_KEY", api_key),
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", deployment),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing Azure OpenAI embedding configuration: {', '.join(missing)}"
            )

        self.deployment = deployment
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        batch_size = 128

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.deployment,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)

        return embeddings


def get_embeddings():
    """Get embeddings based on provider."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    if provider == "openai":
        logger.info("Using OpenAI embeddings (text-embedding-3-small)")
        return OpenAIEmbeddings(model="text-embedding-3-small")

    if provider == "azure_openai":
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        logger.info("Using Azure OpenAI embeddings deployment '%s'", deployment)
        return AzureOpenAIEmbeddingClient()

    logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
