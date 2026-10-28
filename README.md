# SchemeSetu

SchemeSetu answers questions about Indian government startup and MSME schemes. You type a
question, optionally pick a scheme, and get an answer built only from the official PDFs in
`api/corpus/raw/`, grouped by the document each point came from.

## Features

- Ask a question in plain language and get a short summary plus supporting points.
- Filter by scheme, or search across all of them.
- Answers are grouped by source PDF, with the page numbers each point was taken from.
- Questions the documents don't cover return "not found" instead of a guess.
- If Gemini is unavailable or out of quota, the app falls back to showing the most relevant
  excerpts directly, with a small note saying so.

## Schemes and documents

| Scheme | PDFs |
| --- | --- |
| Startup India | `startup_india_notification_2019.pdf`, `sisfs_guidelines.pdf`, `ffs_2_guidelines.pdf` |
| MUDRA | `mudra_pmmy.pdf` |
| MSME | `msme_udyam_notification.pdf`, `cgtmse_scheme_document.pdf` |
| Stand-Up India | `standup_india_faq.pdf`, `standup_india_factsheet.pdf` |

The mapping lives in `api/corpus/schemes.json`.

## Tech stack

Python 3.11, FastAPI, Google Gemini (`google-genai`) for embeddings and generation, Chroma for
the vector store, pypdf for extraction. Frontend: React, TypeScript, Vite and Tailwind.

## Run locally

You need a Gemini API key from [Google AI Studio](https://aistudio.google.com/).

Backend:

```bash
cd api
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
cp .env.example .env               # then set GEMINI_API_KEY
uvicorn main:app --reload
```

On first start the index is built in the background from the PDFs (paced for the free-tier
rate limits, so it takes a few minutes). Until it finishes, `/api/query` replies that the
index is still loading. You can also build it up front with `python scripts/build_index.py`.
Run the tests with `pytest`.

Frontend (proxies `/api` to port 8000):

```bash
cd web
npm install
npm run dev
```

## Run with Docker

The root `Dockerfile` builds the frontend and serves it from FastAPI on `$PORT`:

```bash
docker build -t schemesetu .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key schemesetu
```

Then open http://localhost:8000.

## Deploy to Render

The repo includes a `render.yaml` blueprint for a single free-tier web service.

1. Go to [render.com](https://render.com) and choose **New** -> **Blueprint**.
2. Pick this repository.
3. Paste your `GEMINI_API_KEY` when prompted.
4. Click **Deploy**.
5. Open the `https://<name>.onrender.com` link once the deploy is live.

Free services sleep when idle and the disk is not persistent, so after a cold start the index
is rebuilt in the background. Questions asked in the first few minutes get a "still loading"
message. Set `CORS_ORIGINS` only if the frontend is hosted on a different origin.

## Project layout

```
api/
  main.py            FastAPI app (/api/health, /api/schemes, /api/query) and static frontend
  rag/               ingest, chunk, embed, store, indexer, retrieve, generate
  corpus/            source PDFs, scheme mapping, manifest
  eval/              retrieval and answer evaluation scripts
  tests/             offline pytest suite
web/                 React frontend
render.yaml          Render blueprint
Dockerfile
```
