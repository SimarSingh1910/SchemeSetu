# SchemeSetu

A grounded Retrieval-Augmented Generation (RAG) assistant that answers questions about
Indian government **startup & MSME schemes**. Every answer is grounded only in an indexed
corpus of official scheme documents and returns citations back to the exact source chunks.

The project ships with an **evaluation harness** that measures retrieval quality
(Hit-Rate@k, MRR) and answer quality (RAGAS faithfulness, answer relevance) with numbers.

## Corpus (bounded)
1. Startup India (DPIIT recognition, 80-IAC, Angel Tax/Sec 56, Fund of Funds, SISFS)
2. MUDRA / PMMY (Shishu / Kishore / Tarun)
3. MSME (Udyam registration, classification thresholds, CGTMSE)
4. Stand-Up India (SC/ST & women entrepreneurs)

## Stack
Python 3.11+ · FastAPI · Google Gemini (Flash) via `google-genai` · Chroma · pypdf/pdfplumber
· React + TypeScript + Vite + Tailwind.

## Layout
```
api/
  main.py            # FastAPI app: /api/health, /api/schemes, /api/query
  rag/               # ingest, chunk, embed, store, indexer, retrieve, generate, schema
  eval/              # retrieval_eval, answer_eval, eval_set.json
  scripts/           # build_index.py
  corpus/raw/        # official source PDFs (human-supplied)
  corpus/schemes.json  # scheme -> PDFs mapping
  tests/             # pytest (offline)
  data/index/        # persisted Chroma store (gitignored)
web/                 # React + Vite + Tailwind frontend
```

## Setup
```bash
cd api
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements-dev.txt   # requirements.txt for deploy only
cp .env.example .env          # then set GEMINI_API_KEY
```

Run the API (after building the index with `python scripts/build_index.py`):
```bash
uvicorn main:app --reload
pytest
```
Set `CORS_ORIGINS` (comma-separated) to allow a deployed frontend origin.

Run the frontend (proxies `/api` to the API on port 8000):
```bash
cd web
npm install
npm run dev      # npm run build for a production bundle
```
