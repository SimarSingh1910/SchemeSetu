"""Build the persistent Chroma index from the PDFs in corpus/raw/.

Run from the api/ directory:
    python scripts/build_index.py

Needs GEMINI_API_KEY (api/.env or the environment). The collection is rebuilt
from scratch on every run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the `rag` package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import store
from rag.chunk import CORPUS_RAW, MANIFEST_PATH, SCHEMES_PATH
from rag.embed import DEFAULT_BATCH_SIZE, DEFAULT_PAUSE
from rag.indexer import build_index

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Build the SchemeSetu Chroma index.")
    p.add_argument("--corpus", type=Path, default=CORPUS_RAW)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--schemes", type=Path, default=SCHEMES_PATH)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--pause", type=float, default=DEFAULT_PAUSE,
                   help="Seconds to wait between embedding batches.")
    args = p.parse_args()

    count = build_index(args.corpus, args.manifest, args.schemes,
                        batch_size=args.batch_size, pause=args.pause)
    print(f"Indexed {count} chunks into {store.INDEX_DIR}")
