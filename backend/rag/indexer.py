"""
Chroma vector-store indexing for health documents.

Rebuilds the collection from scratch on every call so the index stays
consistent with the database — avoids stale or duplicated embeddings.

Public API:
  index_health_data(db, embeddings=None) -> int
  get_vectorstore(embeddings=None) -> Chroma
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from backend.config import settings
from backend.rag.documents import load_all_documents


def get_vectorstore(embeddings=None):
    """Return the persisted Chroma collection (read-only helper)."""
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    if embeddings is None:
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
    return Chroma(
        collection_name="health_data",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def index_health_data(db: Session, embeddings=None) -> int:
    """
    Rebuild the Chroma index from all health data in the DB.

    Drops the existing collection before re-indexing so there are no stale
    embeddings. Returns the number of documents indexed.
    """
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    docs = load_all_documents(db)
    if not docs:
        return 0

    if embeddings is None:
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    # Drop and recreate to prevent embedding accumulation across re-ingestions
    vs = Chroma(
        collection_name="health_data",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    vs.delete_collection()

    vs = Chroma(
        collection_name="health_data",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    ids = [f"doc_{i}" for i in range(len(docs))]
    vs.add_documents(docs, ids=ids)

    return len(docs)
