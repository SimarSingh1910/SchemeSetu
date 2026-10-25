"""FastAPI app: scheme listing and grounded question answering under /api."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv(Path(__file__).parent / ".env")

from rag.generate import DEFAULT_K, answer_question  # noqa: E402
from rag.schema import Answer  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMES_PATH = Path(__file__).parent / "corpus" / "schemes.json"
MAX_QUESTION_CHARS = 500
MAX_K = 10

app = FastAPI(title="SchemeSetu API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
router = APIRouter(prefix="/api")


def load_schemes() -> dict[str, dict]:
    with open(SCHEMES_PATH, encoding="utf-8") as f:
        return json.load(f)


class QueryRequest(BaseModel):
    question: str
    scheme: Optional[str] = "all"
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)

    @field_validator("question")
    @classmethod
    def check_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty.")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
        return v


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/schemes")
def schemes() -> list[dict]:
    return [{"id": sid, "name": s["name"], "docs": s["docs"]}
            for sid, s in load_schemes().items()]


@router.post("/query", response_model=Answer)
def query(req: QueryRequest) -> Answer:
    scheme = None if not req.scheme or req.scheme == "all" else req.scheme
    if scheme and scheme not in load_schemes():
        raise HTTPException(status_code=422, detail=f"Unknown scheme '{scheme}'.")
    try:
        return answer_question(req.question, scheme, req.k)
    except RuntimeError as exc:  # e.g. empty index
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Something went wrong answering that question.")


app.include_router(router)
