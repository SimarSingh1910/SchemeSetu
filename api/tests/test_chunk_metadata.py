import json

import pytest

from rag.chunk import CORPUS_RAW, SCHEMES_PATH, chunk_corpus, chunk_text, load_schemes
from rag.ingest import clean_text


@pytest.fixture(scope="module")
def chunks():
    return list(chunk_corpus())


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("regis-\ntration  of\n\n\n\nfirms") == "registration of\n\nfirms"


def test_chunk_text_windows_overlap():
    words = [f"w{i}" for i in range(1000)]
    parts = chunk_text(" ".join(words), chunk_tokens=130, overlap_tokens=13)
    assert len(parts) > 1
    assert all(len(p.split()) <= 100 for p in parts)
    assert parts[0].split()[-10:] == parts[1].split()[:10]


def test_every_pdf_is_mapped_to_a_scheme():
    mapped = set(load_schemes())
    on_disk = {p.name for p in CORPUS_RAW.glob("*.pdf")}
    assert mapped == on_disk


def test_schemes_file_has_no_duplicate_docs():
    raw = json.loads(SCHEMES_PATH.read_text(encoding="utf-8"))
    docs = [d for info in raw.values() for d in info["docs"]]
    assert len(docs) == len(set(docs))


def test_chunks_carry_scheme_source_and_page(chunks):
    assert chunks
    scheme_of = load_schemes()
    for c in chunks:
        assert c.source_doc.endswith(".pdf")
        assert c.page >= 1
        assert c.scheme == scheme_of[c.source_doc]
        meta = c.metadata()
        assert meta["scheme"] and meta["source_doc"] and meta["source_url"]
        assert meta["page"] == c.page


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_all_schemes_present_in_chunks(chunks):
    assert {c.scheme for c in chunks} == {"startup_india", "mudra", "msme", "standup_india"}
