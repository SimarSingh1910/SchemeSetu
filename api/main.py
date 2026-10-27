"""FastAPI app: scheme listing and grounded question answering under /api."""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

load_dotenv(Path(__file__).parent / ".env")

from rag.generate import DEFAULT_K, answer_question  # noqa: E402
from rag.schema import Answer  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMES_PATH = Path(__file__).parent / "corpus" / "schemes.json"
MAX_QUESTION_CHARS = 500
MAX_K = 10
WEB_DIST = Path(os.getenv("WEB_DIST", Path(__file__).resolve().parent.parent / "web" / "dist"))
LOADING_MESSAGE = "The document index is still loading. Please try again in a minute."

_building = threading.Event()


def _build_index_if_missing() -> None:
    from rag import store
    from rag.indexer import ensure_index

    try:
        if store.index_is_ready(store.get_client()):
            return
        _building.set()
        ensure_index()
    except Exception:
        logger.exception("Background index build failed")
    finally:
        _building.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_build_index_if_missing, daemon=True).start()
    yield


app = FastAPI(title="SchemeSetu API", lifespan=lifespan)
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
    if _building.is_set():
        raise HTTPException(status_code=503, detail=LOADING_MESSAGE)
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

if (WEB_DIST / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found.")
        target = (WEB_DIST / path).resolve()
        if path and target.is_file() and WEB_DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(WEB_DIST / "index.html")
