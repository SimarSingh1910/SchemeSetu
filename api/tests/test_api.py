import pytest
from fastapi.testclient import TestClient

import main
from rag.schema import Answer, SourceSection

client = TestClient(main.app)


def fake_answer(question="q", scheme=None):
    return Answer(question=question, scheme=scheme, summary="Short answer.",
                  sources=[SourceSection(source_doc="mudra_pmmy.pdf", scheme="mudra",
                                         pages=[2], points=["Loans up to 10 lakh."])])


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake(question, scheme=None, k=6):
        calls.append((question, scheme, k))
        return fake_answer(question, scheme)

    monkeypatch.setattr(main, "answer_question", fake)
    return calls


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_schemes_lists_four_with_docs():
    data = client.get("/api/schemes").json()
    assert {s["id"] for s in data} == {"startup_india", "mudra", "msme", "standup_india"}
    mudra = next(s for s in data if s["id"] == "mudra")
    assert mudra["docs"] == ["mudra_pmmy.pdf"]


def test_query_returns_structured_answer(captured):
    r = client.post("/api/query", json={"question": "  What is Shishu?  ", "scheme": "mudra"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"][0]["source_doc"] == "mudra_pmmy.pdf"
    assert body["found"] is True
    assert captured == [("What is Shishu?", "mudra", 6)]


def test_scheme_defaults_to_all(captured):
    client.post("/api/query", json={"question": "Who can apply?"})
    client.post("/api/query", json={"question": "Who can apply?", "scheme": "all", "k": 3})
    assert captured[0][1] is None
    assert captured[1] == ("Who can apply?", None, 3)


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   "},
    {"question": "x" * 501},
    {"question": "ok", "k": 0},
    {"question": "ok", "k": 50},
    {},
])
def test_invalid_input_is_422(captured, payload):
    r = client.post("/api/query", json=payload)
    assert r.status_code == 422
    assert captured == []


def test_validation_message_is_readable(captured):
    r = client.post("/api/query", json={"question": ""})
    assert "must not be empty" in r.json()["detail"][0]["msg"]


def test_unknown_scheme_is_422(captured):
    r = client.post("/api/query", json={"question": "hi", "scheme": "nope"})
    assert r.status_code == 422
    assert "nope" in r.json()["detail"]


def test_empty_index_is_503(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("Index is empty. Run scripts/build_index.py first.")

    monkeypatch.setattr(main, "answer_question", boom)
    r = client.post("/api/query", json={"question": "hi"})
    assert r.status_code == 503


def test_unexpected_error_is_500(monkeypatch):
    def boom(*a, **kw):
        raise ValueError("secret detail")

    monkeypatch.setattr(main, "answer_question", boom)
    r = client.post("/api/query", json={"question": "hi"})
    assert r.status_code == 500
    assert "secret detail" not in r.text


def test_cors_allows_local_dev_origin():
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
