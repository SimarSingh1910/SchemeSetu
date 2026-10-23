import chromadb

from rag import store
from rag.indexer import build_index, ensure_index
from rag.retrieve import search


def fake_embed(texts, *, task_type, batch_size, pause):
    # Deterministic 8-d vectors; enough to exercise storage and filtering.
    return [[(hash((t, i)) % 97) / 97 + 0.01 for i in range(8)] for t in texts]


def test_ensure_index_builds_once(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    assert not store.index_is_ready(client)

    assert ensure_index(client=client, embed=fake_embed, pause=0) is True
    assert store.index_is_ready(client)
    assert ensure_index(client=client, embed=fake_embed, pause=0) is False


def test_search_filters_by_scheme(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    n = build_index(client=client, embed=fake_embed, pause=0)
    coll = store.get_collection(client)
    assert coll.count() == n

    hits = search(coll, fake_embed(["x"], task_type="q", batch_size=1, pause=0)[0],
                  k=5, scheme="mudra")
    assert hits
    assert {h.scheme for h in hits} == {"mudra"}
    assert {h.source_doc for h in hits} == {"mudra_pmmy.pdf"}
