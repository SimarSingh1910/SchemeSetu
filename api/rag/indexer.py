"""Build the Chroma index from the corpus, and build it on demand if missing."""
from __future__ import annotations

import logging
from pathlib import Path

from . import store
from .chunk import CORPUS_RAW, MANIFEST_PATH, SCHEMES_PATH, chunk_corpus
from .embed import DEFAULT_BATCH_SIZE, DEFAULT_PAUSE, TASK_DOCUMENT, embed_texts

logger = logging.getLogger(__name__)


def build_index(
    corpus_dir: Path = CORPUS_RAW,
    manifest_path: Path = MANIFEST_PATH,
    schemes_path: Path = SCHEMES_PATH,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pause: float = DEFAULT_PAUSE,
    client=None,
    embed=embed_texts,
) -> int:
    """Chunk, embed and store the whole corpus. Returns the number of chunks indexed.

    `embed` is injectable so the pipeline can run without network access.
    """
    chunks = list(chunk_corpus(corpus_dir, manifest_path, schemes_path))
    if not chunks:
        logger.warning("No chunks produced; nothing to index.")
        return 0
    logger.info("Embedding %d chunks ...", len(chunks))
    embeddings = embed([c.text for c in chunks], task_type=TASK_DOCUMENT,
                       batch_size=batch_size, pause=pause)

    client = client or store.get_client()
    collection = store.build_collection(client, reset=True)
    store.add_chunks(collection, chunks, embeddings)
    return collection.count()


def ensure_index(client=None, **kwargs) -> bool:
    """Build the index if it is missing or empty. Returns True if a build ran."""
    client = client or store.get_client()
    if store.index_is_ready(client):
        return False
    logger.info("No index found; building it from the corpus.")
    build_index(client=client, **kwargs)
    return True
