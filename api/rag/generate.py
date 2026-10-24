"""Grounded generation: answer only from retrieved chunks, grouped by source PDF.

If nothing relevant is retrieved the LLM is not called. If generation fails
(quota, network, bad output) we fall back to an extractive answer built from
the top excerpts.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from google.genai import types

from .embed import get_client
from .retrieve import RetrievedChunk, retrieve
from .schema import Answer, LLMAnswer, SourceSection

logger = logging.getLogger(__name__)

GEN_MODEL = "gemini-2.5-flash"
DEFAULT_K = 6
# Cosine similarity of the best hit below which we treat the question as out of scope.
MIN_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.5"))
MAX_EXCERPT_CHARS = 600

NOT_FOUND_MESSAGE = (
    "This was not found in the scheme documents. Try rephrasing the question "
    "or selecting a different scheme."
)

PROMPT = """You answer questions about Indian government startup and MSME schemes.
Use ONLY the numbered excerpts below. Do not use outside knowledge. If the
excerpts do not answer the question, say so in the summary and return no sections.

Return JSON with:
- summary: a short plain-language answer (2-3 sentences).
- sections: one entry per source PDF actually used, with source_doc exactly as
  given in the excerpt header and points as short, self-contained bullet strings.

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


def _call_llm(question: str, chunks: list[RetrievedChunk], client=None) -> LLMAnswer:
    client = client or get_client()
    resp = client.models.generate_content(
        model=GEN_MODEL,
        contents=PROMPT.format(question=question, context=_format_context(chunks)),
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=LLMAnswer,
        ),
    )
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

    groups = _group_by_doc(chunks)
    sections = []
    for sec in llm.sections:
        group = groups.get(sec.source_doc)
        points = [p.strip() for p in sec.points if p.strip()]
        if not group or not points:
            continue  # ignore sources the model made up
        sections.append(SourceSection(source_doc=sec.source_doc, scheme=group[0].scheme,
                                      pages=_page_list(group), points=points))
    if not sections:
        return extractive_answer(question, scheme, chunks)
    return Answer(question=question, scheme=scheme, summary=llm.summary.strip(),
                  sources=sections, found=True, mode="generated")


def answer_question(question: str, scheme: Optional[str] = None, k: int = DEFAULT_K,
                    client=None) -> Answer:
    """Retrieve (optionally within one scheme) and generate a grounded answer."""
    chunks = retrieve(question, k, scheme=scheme)
    return generate_answer(question, chunks, scheme, client)
