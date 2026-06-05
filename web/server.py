"""FastAPI server for the RAG web UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.llm.generate import CloudThenMockLLMClient
from rag.pipeline import query_pipeline
from rag.indexing.cache import warm_index_cache
from rag.indexing.index import migrate_numpy_index_to_faiss
from rag.profiles.query import (
    DEFAULT_PROFILES_PATH,
    OFFICIAL_INDEX_DIR,
    PROFILE_LABELS,
    PROFILE_NAMES,
    load_profiles,
    run_profile_query,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
OFFICIAL_INDEX = ROOT / OFFICIAL_INDEX_DIR

load_dotenv(ROOT / ".env")

_LLM = CloudThenMockLLMClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _index_ready(OFFICIAL_INDEX):
        try:
            migrate_numpy_index_to_faiss(OFFICIAL_INDEX)
        except Exception:
            pass
        warm_index_cache(OFFICIAL_INDEX)
    yield


app = FastAPI(title="DRAG RAG", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def _static_no_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    mode: Literal["domain", "baseline"] = "domain"
    profile: str = "freshness_accuracy"
    top_k: int | None = Field(default=None, ge=1, le=20)
    retrieval_k: int | None = Field(default=None, ge=1, le=50)


def _index_ready(index_dir: str | Path) -> bool:
    p = Path(index_dir)
    return (p / "meta.json").exists() and (p / "chunks.json").exists()


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_dir": OFFICIAL_INDEX_DIR,
        "corpus": ["aws", "docker", "google_drive"],
        "index_ready": _index_ready(OFFICIAL_INDEX),
        "llm_mode": "auto",
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    profiles = load_profiles(ROOT / DEFAULT_PROFILES_PATH)
    return {
        "index_dir": OFFICIAL_INDEX_DIR,
        "corpus": ["aws", "docker", "google_drive"],
        "llm_mode": "auto",
        "profiles": [
            {
                "id": name,
                "label": PROFILE_LABELS.get(name, name),
                "top_k": profiles[name].get("top_k"),
                "retrieval_k": profiles[name].get("retrieval_k"),
                "policy": profiles[name].get("policy", []),
                "index_ready": _index_ready(OFFICIAL_INDEX),
            }
            for name in PROFILE_NAMES
            if name in profiles
        ],
        "modes": ["domain", "baseline"],
    }


@app.post("/api/query")
def run_query(body: QueryRequest) -> dict[str, Any]:
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty")

    if not _index_ready(OFFICIAL_INDEX):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Index not ready at {OFFICIAL_INDEX_DIR}. "
                "Run: python run_ingest.py --docs-dir data/docs --index-dir data/index/official"
            ),
        )

    try:
        if body.mode == "baseline":
            result = query_pipeline(
                query=q,
                index_dir=OFFICIAL_INDEX_DIR,
                top_k=body.top_k or 3,
                retrieval_k=body.retrieval_k,
                log_path=ROOT / "runs/logs.jsonl",
                llm_client=_LLM,
            )
            return {
                "mode": "baseline",
                "llm_mode_requested": "auto",
                "index_dir": OFFICIAL_INDEX_DIR,
                **result,
            }

        if body.profile not in PROFILE_NAMES:
            raise HTTPException(status_code=400, detail=f"Unknown profile: {body.profile}")

        profiles = load_profiles(ROOT / DEFAULT_PROFILES_PATH)
        result = run_profile_query(
            query=q,
            profile_name=body.profile,
            profiles=profiles,
            llm_name="auto",
            log_path=ROOT / "runs/domain_logs.jsonl",
            skip_external_sync=True,
            top_k=body.top_k,
            retrieval_k=body.retrieval_k,
        )
        return {
            "mode": "domain",
            "llm_mode_requested": "auto",
            "index_dir": OFFICIAL_INDEX_DIR,
            **result,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
