import json
from types import SimpleNamespace

import chromadb

from rag import store
from rag.generate import NOT_FOUND_MESSAGE, generate_answer
from rag.indexer import build_index
from rag.retrieve import RetrievedChunk, search


def chunk(doc, page, score=0.8, scheme="startup_india", text="Some text."):
    return RetrievedChunk(chunk_id=f"{doc}_p{page}", text=text, score=score,
                          distance=1 - score, source_doc=doc, source_url="",
                          page=page, retrieved_on="", section="", scheme=scheme)


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.calls = 0
        self.models = self
        self._payload, self._error = payload, error

    def generate_content(self, **kwargs):
        self.calls += 1
        if self._error:
            raise self._error
        return SimpleNamespace(text=json.dumps(self._payload))


CHUNKS = [chunk("sisfs_guidelines.pdf", 3), chunk("ffs_2_guidelines.pdf", 7),
          chunk("sisfs_guidelines.pdf", 4)]


def test_groups_by_pdf_with_pages():
    client = FakeClient({"summary": "Seed funding exists.", "sections": [
        {"source_doc": "sisfs_guidelines.pdf", "points": ["Up to Rs 20 lakh grant."]},
        {"source_doc": "ffs_2_guidelines.pdf", "points": ["Funds invest via AIFs."]},
        {"source_doc": "invented.pdf", "points": ["Should be dropped."]},
    ]})
    ans = generate_answer("seed funding?", CHUNKS, client=client)
    assert ans.found and ans.mode == "generated"
    assert [s.source_doc for s in ans.sources] == ["sisfs_guidelines.pdf", "ffs_2_guidelines.pdf"]
    assert ans.sources[0].pages == [3, 4]
    assert ans.sources[1].pages == [7]


def test_not_found_skips_llm():
    client = FakeClient({"summary": "x", "sections": []})
    for chunks in ([], [chunk("a.pdf", 1, score=0.2)]):
        ans = generate_answer("weather today?", chunks, client=client)
        assert not ans.found
        assert ans.summary == NOT_FOUND_MESSAGE
        assert ans.sources == []
    assert client.calls == 0


def test_falls_back_to_extractive_on_error():
    client = FakeClient(error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    ans = generate_answer("seed funding?", CHUNKS, scheme="startup_india", client=client)
    assert ans.found and ans.mode == "extractive"
    assert ans.scheme == "startup_india"
    docs = {s.source_doc: s for s in ans.sources}
    assert docs["sisfs_guidelines.pdf"].pages == [3, 4]
    assert docs["sisfs_guidelines.pdf"].points


def test_scheme_filter_limits_sources(tmp_path):
    def fake_embed(texts, *, task_type, batch_size, pause):
        return [[(hash((t, i)) % 97) / 97 + 0.01 for i in range(8)] for t in texts]

    client = chromadb.PersistentClient(path=str(tmp_path))
    build_index(client=client, embed=fake_embed, pause=0)
    coll = store.get_collection(client)
    hits = search(coll, fake_embed(["q"], task_type="q", batch_size=1, pause=0)[0],
                  k=8, scheme="standup_india")
    ans = generate_answer("q", hits, scheme="standup_india", client=FakeClient(error=ValueError()),
                          min_score=-1)
    assert ans.sources
    assert {s.scheme for s in ans.sources} == {"standup_india"}
