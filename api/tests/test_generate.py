import json
from types import SimpleNamespace

import chromadb
from google.genai import errors

from rag import generate, store
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


def test_pages_come_from_cited_excerpts():
    client = FakeClient({"summary": "Seed grant.", "sections": [
        {"source_doc": "sisfs_guidelines.pdf", "points": ["Up to Rs 20 lakh."], "excerpts": [3]},
    ]})
    ans = generate_answer("seed funding?", CHUNKS, client=client)
    assert ans.sources[0].pages == [4]  # only excerpt [3] (page 4), not both sisfs pages


def test_bad_citations_fall_back_to_all_doc_pages():
    client = FakeClient({"summary": "Seed grant.", "sections": [
        # [2] is the ffs PDF and [9] doesn't exist, so neither can supply a sisfs page.
        {"source_doc": "sisfs_guidelines.pdf", "points": ["Up to Rs 20 lakh."], "excerpts": [2, 9]},
    ]})
    ans = generate_answer("seed funding?", CHUNKS, client=client)
    assert ans.sources[0].pages == [3, 4]


def test_echoed_header_still_matches_source_doc():
    client = FakeClient({"summary": "Seed grant.", "sections": [
        {"source_doc": "SISFS_Guidelines.pdf (page 3)", "points": ["Up to Rs 20 lakh."]},
        {"source_doc": "the fund of funds document", "points": ["Via AIFs."], "excerpts": [2]},
    ]})
    ans = generate_answer("seed funding?", CHUNKS, client=client)
    assert ans.mode == "generated"
    assert [s.source_doc for s in ans.sources] == ["sisfs_guidelines.pdf", "ffs_2_guidelines.pdf"]
    assert ans.sources[1].pages == [7]


def test_model_saying_no_answer_is_not_found():
    client = FakeClient({"summary": "The excerpts do not cover this.", "sections": []})
    ans = generate_answer("seed funding?", CHUNKS, client=client)
    assert not ans.found
    assert ans.summary == NOT_FOUND_MESSAGE


def test_default_threshold_rejects_near_domain_noise():
    # Off-topic questions score ~0.56-0.63 against this corpus with gemini-embedding-001.
    client = FakeClient({"summary": "x", "sections": []})
    ans = generate_answer("home loan from SBI?", [chunk("a.pdf", 1, score=0.63)], client=client)
    assert not ans.found
    assert client.calls == 0


class FlakyClient(FakeClient):
    """Fails with the given status codes in order, then returns the payload."""

    def __init__(self, payload, codes):
        super().__init__(payload)
        self.models_used, self._codes = [], list(codes)

    def generate_content(self, **kwargs):
        self.models_used.append(kwargs["model"])
        if self._codes:
            code = self._codes.pop(0)
            raise errors.APIError(code, {"error": {"code": code, "message": "x", "status": "x"}})
        return super().generate_content(**kwargs)


PAYLOAD = {"summary": "Seed grant.", "sections": [
    {"source_doc": "sisfs_guidelines.pdf", "points": ["Up to Rs 20 lakh."], "excerpts": [1]}]}


def test_retries_then_falls_back_to_next_model(monkeypatch):
    monkeypatch.setattr(generate, "_sleep_for", lambda attempt: None)
    monkeypatch.setattr(generate, "GEN_MODELS", ["primary", "backup"])
    client = FlakyClient(PAYLOAD, codes=[503, 429])
    ans = generate_answer("seed funding?", CHUNKS, client=client)
    assert ans.mode == "generated"
    assert client.models_used == ["primary", "primary", "backup"]


def test_non_retryable_error_goes_extractive(monkeypatch):
    monkeypatch.setattr(generate, "_sleep_for", lambda attempt: None)
    monkeypatch.setattr(generate, "GEN_MODELS", ["primary", "backup"])
    client = FlakyClient(PAYLOAD, codes=[404])
    ans = generate_answer("seed funding?", CHUNKS, client=client)
    assert ans.mode == "extractive"
    assert client.models_used == ["primary"]


def test_sections_for_the_same_pdf_are_merged():
    client = FakeClient({"summary": "Seed grant.", "sections": [
        {"source_doc": "sisfs_guidelines.pdf", "points": ["Grant up to Rs 20 lakh."], "excerpts": [1]},
        {"source_doc": "ffs_2_guidelines.pdf", "points": ["Via AIFs."], "excerpts": [2]},
        {"source_doc": "sisfs_guidelines.pdf", "points": ["Debt up to Rs 50 lakh."], "excerpts": [3]},
    ]})
    ans = generate_answer("seed funding?", CHUNKS, client=client)
    assert [s.source_doc for s in ans.sources] == ["sisfs_guidelines.pdf", "ffs_2_guidelines.pdf"]
    assert ans.sources[0].points == ["Grant up to Rs 20 lakh.", "Debt up to Rs 50 lakh."]
    assert ans.sources[0].pages == [3, 4]
