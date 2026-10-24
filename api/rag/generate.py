"""Grounded generation: answer only from retrieved chunks, grouped by source PDF.

If nothing relevant is retrieved the LLM is not called. Transient Gemini errors
(429/5xx) are retried briefly, then the next model in GEN_MODELS is tried; if
generation still fails (quota, network, bad output) we fall back to an
extractive answer built from the top excerpts.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from google.genai import errors, types

from .embed import _RETRY_CODES, _sleep_for, get_client
from .retrieve import RetrievedChunk, retrieve
from .schema import Answer, LLMAnswer, SourceSection

logger = logging.getLogger(__name__)

# Tried in order; a model that keeps returning 429/5xx hands over to the next.
# (gemini-2.5-flash now 404s for new API keys: "no longer available to new users".)
GEN_MODELS = [m.strip() for m in
              os.getenv("GEMINI_GEN_MODELS", "gemini-3.8-flash,gemini-3.5-flash").split(",")
              if m.strip()]
DEFAULT_K = 6
# Cosine similarity of the best hit below which we treat the question as out of scope.
# With gemini-embedding-001, unrelated questions still score ~0.56-0.63 against this
# corpus while in-scope ones score ~0.72+, so 0.5 let everything through.
MIN_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.67"))
MAX_EXCERPT_CHARS = 600
# Attempts per model. A user is waiting, so retry far less than bulk embedding does.
GEN_ATTEMPTS = 2

NOT_FOUND_MESSAGE = (
    "This was not found in the scheme documents. Try rephrasing the question "
    "or selecting a different scheme."
)

PROMPT = """You answer questions about Indian government startup and MSME schemes.
Use ONLY the numbered excerpts below. Do not use outside knowledge. If the
excerpts answer only part of the question, answer that part and say in the
summary what the documents do not cover. Return no sections only if nothing in
the excerpts is relevant to the question.

Return JSON with:
- summary: a short plain-language answer (2-3 sentences).
- sections: one entry per source PDF actually used, with source_doc exactly as
  given in the excerpt header, points as short, self-contained bullet strings,
  and excerpts as the [n] numbers of the excerpts those points came from.

Question: {question}

Excerpts:
{context}
"""


def _page_list(chunks: list[RetrievedChunk]) -> list[int]:
    pages = set()
    for c in chunks:
        try:
            pages.add(int(c.page))
        except (TypeError, ValueError):
            pass
    return sorted(pages)


def _group_by_doc(chunks: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
    groups: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:  # chunks arrive best-first, so groups keep that order
        groups.setdefault(c.source_doc, []).append(c)
    return groups


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] source_doc: {c.source_doc} (page {c.page})\n{' '.join(c.text.split())}"
        for i, c in enumerate(chunks, start=1)
    )


def not_found(question: str, scheme: Optional[str]) -> Answer:
    return Answer(question=question, scheme=scheme, summary=NOT_FOUND_MESSAGE,
                  sources=[], found=False, mode="extractive")


def extractive_answer(question: str, scheme: Optional[str],
                      chunks: list[RetrievedChunk]) -> Answer:
    """Top excerpts grouped by PDF, used when generation is unavailable."""
    sections = []
    for doc, group in _group_by_doc(chunks).items():
        sections.append(SourceSection(
            source_doc=doc,
            scheme=group[0].scheme,
            pages=_page_list(group),
            points=[" ".join(c.text.split())[:MAX_EXCERPT_CHARS] for c in group[:2]],
        ))
    return Answer(
        question=question, scheme=scheme,
        summary="Showing the most relevant excerpts from the scheme documents.",
        sources=sections, found=True, mode="extractive",
    )


def _cited_pages(sec_doc: str, excerpts: list[int], chunks: list[RetrievedChunk]) -> list[int]:
    """Pages of the cited excerpts that belong to sec_doc; ignores bad numbers."""
    cited = [chunks[n - 1] for n in excerpts
             if isinstance(n, int) and 1 <= n <= len(chunks)
             and chunks[n - 1].source_doc == sec_doc]
    return _page_list(cited)


def _generate(client, model: str, contents: str):
    """One model, retrying 429/5xx a couple of times."""
    for attempt in range(GEN_ATTEMPTS):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=LLMAnswer,
                ),
            )
        except errors.APIError as exc:
            if getattr(exc, "code", None) in _RETRY_CODES and attempt < GEN_ATTEMPTS - 1:
                _sleep_for(attempt)
                continue
            raise


def _norm_doc(name: str) -> str:
    """'MUDRA_PMMY.pdf (page 3)' -> 'mudra_pmmy.pdf'; the model sometimes echoes the header."""
    name = re.sub(r"\s*\(page[^)]*\)\s*$", "", name.strip(), flags=re.I).lower()
    return name if name.endswith(".pdf") else name + ".pdf"


def _resolve_doc(sec_doc: str, excerpts: list[int],
                 chunks: list[RetrievedChunk]) -> Optional[str]:
    """Map the model's source_doc to a retrieved doc, or to the doc its cited excerpts share."""
    by_norm = {_norm_doc(c.source_doc): c.source_doc for c in chunks}
    if _norm_doc(sec_doc) in by_norm:
        return by_norm[_norm_doc(sec_doc)]
    cited = {chunks[n - 1].source_doc for n in excerpts
             if isinstance(n, int) and 1 <= n <= len(chunks)}
    return cited.pop() if len(cited) == 1 else None


def _call_llm(question: str, chunks: list[RetrievedChunk], client=None) -> LLMAnswer:
    client = client or get_client()
    contents = PROMPT.format(question=question, context=_format_context(chunks))
    for i, model in enumerate(GEN_MODELS):
        try:
            resp = _generate(client, model, contents)
            break
        except errors.APIError as exc:
            if getattr(exc, "code", None) in _RETRY_CODES and i < len(GEN_MODELS) - 1:
                logger.warning("%s unavailable (%s); trying %s.", model, exc.code, GEN_MODELS[i + 1])
                continue
            raise
    return LLMAnswer.model_validate_json(resp.text)


def generate_answer(question: str, chunks: list[RetrievedChunk],
                    scheme: Optional[str] = None, client=None,
                    min_score: float = MIN_SCORE) -> Answer:
    """Build an Answer from already-retrieved chunks."""
    if not chunks or max(c.score for c in chunks) < min_score:
        return not_found(question, scheme)

    try:
        llm = _call_llm(question, chunks, client)
    except Exception as exc:  # quota (429), network, malformed output
        logger.warning("Generation failed (%s); returning extractive answer.", exc)
        return extractive_answer(question, scheme, chunks)

    if not llm.sections:  # the model judged that the excerpts don't answer it
        return not_found(question, scheme)

    groups = _group_by_doc(chunks)
    sections: dict[str, SourceSection] = {}  # one per PDF, even if the model splits it
    for sec in llm.sections:
        doc = _resolve_doc(sec.source_doc, sec.excerpts, chunks)
        points = [p.strip() for p in sec.points if p.strip()]
        if not doc:
            logger.warning("Dropping section for unknown source_doc %r.", sec.source_doc)
            continue  # ignore sources the model made up
        if not points:
            continue
        group = groups[doc]
        # Prefer the pages the model cited; fall back to every retrieved page of the doc.
        pages = _cited_pages(doc, sec.excerpts, chunks) or _page_list(group)
        if doc in sections:
            merged = sections[doc]
            merged.points += points
            merged.pages = sorted(set(merged.pages) | set(pages))
        else:
            sections[doc] = SourceSection(source_doc=doc, scheme=group[0].scheme,
                                          pages=pages, points=points)
    if not sections:
        return extractive_answer(question, scheme, chunks)
    return Answer(question=question, scheme=scheme, summary=llm.summary.strip(),
                  sources=list(sections.values()), found=True, mode="generated")


def answer_question(question: str, scheme: Optional[str] = None, k: int = DEFAULT_K,
                    client=None) -> Answer:
    """Retrieve (optionally within one scheme) and generate a grounded answer."""
    chunks = retrieve(question, k, scheme=scheme)
    return generate_answer(question, chunks, scheme, client)
