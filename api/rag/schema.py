"""Pydantic models for the structured answer payload."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SourceSection(BaseModel):
    """Points taken from one source PDF."""

    source_doc: str
    scheme: str = ""
    pages: list[int] = Field(default_factory=list)
    points: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    scheme: Optional[str] = None
    summary: str = ""
    sources: list[SourceSection] = Field(default_factory=list)
    found: bool = True
    mode: Literal["generated", "extractive"] = "generated"


class LLMSection(BaseModel):
    """Shape requested from the model; pages are filled in from the retrieved chunks."""

    source_doc: str
    points: list[str]


class LLMAnswer(BaseModel):
    summary: str
    sections: list[LLMSection]
