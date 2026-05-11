# TrainSmartAI

A personal health intelligence dashboard built on Apple Health export data. Inspired by Whoop — TrainSmartAI ingests your full health export, computes a daily readiness score, surfaces correlation-based insights, generates rule-based recommendations, and lets you ask freeform questions about your data through a local LLM-powered chat assistant.

The entire AI stack runs on your machine. No OpenAI key, no cloud calls, no per-query cost.

---

## What it does

| Capability | How it works |
|---|---|
| **Ingest** Apple Health `export.xml` | Streaming XML parser handles multi-GB files; idempotent re-upload via `(record_type, source, start, end)` unique constraint |
| **Daily features** | Aggregates raw records into one row per calendar day: steps, avg/resting HR, sleep duration + window, workout count/min/calories, active energy |
| **Readiness score** (0–100) | Weighted formula across sleep (40%), HR vs personal baseline (30%), training load (20%), and bedtime consistency (10%). Missing components get weight redistributed |
| **Weekly summaries** | Monday-anchored rollups: avg steps/sleep/HR, sleep consistency score, total workout minutes, workout days, avg readiness |
| **Insights** | Correlation-based plain-English observations — sleep vs activity (Pearson r), RHR trend (polyfit), sleep trend, workout pattern, post-workout RHR, weekend sleep, steps trend |
| **Recommendations** | 10 rules across recovery / workout / sleep / habit categories with priority tiers (1=urgent, 2=worth doing, 3=nudge). Each recommendation stores its reasoning |
| **AI chat** | LangChain RAG chain over indexed summaries — retrieval (Chroma) + Ollama embeddings + Ollama LLM. Multi-turn conversation history persisted to Postgres |

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 + Alembic |
| Data processing | pandas, NumPy |
| RAG | LangChain + Chroma (vector store) + Ollama (embeddings + LLM) |
| Frontend | Streamlit + Plotly *(in progress)* |
| Testing | pytest + httpx + FastAPI TestClient |

---

## Architecture

```
Apple Health export.xml
         │
         ▼
   POST /upload  ──────────────────────────────┐
         │                                     │
   ┌─────▼────────────────────┐                │
   │  Ingestion Pipeline      │                │
   │  parse → clean → upsert  │                │  pipeline runs the
   └─────────────┬────────────┘                │  full chain on every
                 │                             │  upload, end to end
          health_records                       │
                 │                             │
   ┌─────────────▼──────────────┐              │
   │  Daily Feature Aggregation │              │
   └─────────────┬──────────────┘              │
                 │                             │
          daily_features                       │
                 │                             │
   ┌─────────────┴──────────────┐              │
   │                            │              │
 Readiness Score      Weekly Summaries          │
                 │                              │
                 ▼                              │
            Insights ──┐                        │
                       ▼                        │
              Recommendations ──────────────────┘
                       │
                       ▼
            (separate manual step)
                       │
                       ▼
              RAG index build  ──→  Chroma
                                       │
                                       ▼
                                  POST /chat
                                       │
                       Ollama LLM + retrieved context
                                       │
                                       ▼
                                  natural-language
                                  health coaching
```

---

## Prerequisites

- macOS / Linux
- Python 3.12+
- Docker (for Postgres)
- [Ollama](https://ollama.com) (for the local LLM)

---

## Setup

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/dsharm9148/TrainSmartAI.git
cd TrainSmartAI
pip install -r requirements.txt
```

### 2. Start Postgres

```bash
docker-compose up -d postgres
```

This spins up Postgres 16 on `localhost:5432` with credentials `postgres / postgres` and the database `trainsmart`. A `pgadmin` UI is also available at `http://localhost:5050` (login `admin@trainsmart.local / admin`).

### 3. Create a dedicated test database

The test suite uses a separate `trainsmart_test` database so tests can drop tables freely without touching your live data.

```bash
docker exec -it $(docker ps -qf "name=postgres") psql -U postgres -c "CREATE DATABASE trainsmart_test;"
```

### 4. Run migrations

```bash
alembic upgrade head
```

### 5. Install Ollama and pull models

```bash
brew install ollama          # macOS — see ollama.com for other platforms
brew services start ollama   # starts the daemon on http://localhost:11434

ollama pull nomic-embed-text # embeddings model (~270 MB)
ollama pull llama3.2:3b      # chat model (~2 GB, fast on M-series Mac)
```

You can substitute other Ollama models — `qwen2.5:7b`, `mistral:7b`, etc. Override defaults via `.env`:

```env
OLLAMA_CHAT_MODEL=qwen2.5:7b
```

### 6. Configure environment

```bash
cp .env.example .env
```

The defaults in `.env.example` work as-is for local development.

### 7. Start the API

```bash
uvicorn backend.api.main:app --reload
```

Interactive docs at **http://localhost:8000/docs**.

---

## Trying it out end to end

### Option A — synthetic demo data (no Apple Watch needed)

Generate 90 days of realistic synthetic data with a progressive fitness trend, post-workout fatigue effects, and 5 anomaly days simulating illness/travel:

```bash
python data/synthetic/generate_sample.py
# → writes data/synthetic/demo_export.xml
```

### Option B — your real export

On your iPhone: Health app → profile (top right) → Export All Health Data → AirDrop the resulting zip to your Mac → unzip → use the `export.xml` inside.

### Upload

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/synthetic/demo_export.xml"
```

The pipeline runs end-to-end and returns counts:

```json
{
  "parsed": 887, "cleaned": 887, "inserted": 542, "filtered": 0,
  "days_computed": 90, "weeks_computed": 14,
  "message": "Ingested 542 new records across 90 days.",
  "by_type": { "HKQuantityTypeIdentifierStepCount": 90, ... }
}
```

After upload, the following are auto-computed: daily features, readiness scores, weekly summaries, insights, recommendations.

### Build the RAG index

The index is **not** automatically rebuilt on upload (it can take a minute on a large dataset). Trigger it manually:

```bash
python -c "from backend.db.session import SessionLocal; from backend.rag.indexer import index_health_data; print(index_health_data(SessionLocal()), 'docs indexed')"
```

You should see output like `542 docs indexed`. The index is persisted at `./chroma_db/`.

### Chat with your data

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How did I sleep last week, and how did that affect my readiness?"}'
```

First call is slow (~5–15 s on M-series Mac) due to Ollama model warm-up. Subsequent calls are fast (~1–3 s). The response includes a `session_id` you can pass back on follow-up turns:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What should I do about it tomorrow?", "session_id": "<uuid from previous response>"}'
```

Or use the Swagger UI at http://localhost:8000/docs — much easier for ad-hoc experimentation.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + DB reachability check |
| `POST` | `/upload` | Ingest Apple Health `export.xml` — runs the full pipeline |
| `GET` | `/daily` | Daily features — `?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD` |
| `GET` | `/weekly` | Weekly summaries — `?start_date=&end_date=` |
| `GET` | `/readiness` | Readiness scores with component breakdowns |
| `POST` | `/readiness/recompute` | Force-rebuild readiness scores |
| `GET` | `/insights` | Generated insights — filterable by `insight_type` |
| `POST` | `/insights/generate` | Force-regenerate insights |
| `GET` | `/recommendations` | Rule-based recommendations — filter by `for_date`, `category` |
| `POST` | `/recommendations/generate` | Force-regenerate recommendations |
| `POST` | `/chat` | Multi-turn chat — body `{message, session_id?}` |
| `GET` | `/chat/sessions` | List chat sessions, newest first |
| `GET` | `/chat/sessions/{id}/messages` | Full transcript for a session |

Full interactive docs at `http://localhost:8000/docs`.

---

## Running the test suite

```bash
pytest tests/ -v
```

258 tests covering ingestion, daily features, readiness scoring, insights, recommendations, weekly summaries, RAG (document builders, indexer, chain), chat (sessions, history, persistence), and HTTP routes. The tests use Postgres (`trainsmart_test`) for DB integration and stub out the LLM via `FakeListChatModel` + a fake embeddings class — they run in under 3 seconds end-to-end and make zero network calls.

Run a specific module:

```bash
pytest tests/test_recommendations.py -v
pytest tests/test_chat.py -v
```

---

## Project structure

```
backend/
├── api/                # FastAPI app, routes, Pydantic schemas
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── ingestion.py
│       ├── dashboard.py
│       ├── readiness.py
│       ├── insights.py
│       ├── recommendations.py
│       └── chat.py
├── analytics/          # Insights engine + weekly summary computation
├── clustering/         # K-means day-type labeling (Day 17)
├── db/                 # SQLAlchemy models, Alembic migrations, session
├── ingestion/          # XML parser, cleaner, bulk loader, pipeline
├── preprocessing/      # Daily feature aggregation
├── rag/                # LangChain chain, Chroma indexer, document builders
├── recommendations/    # Rule engine
└── scoring/            # Readiness score

data/synthetic/         # Demo data generator
frontend/               # Streamlit pages (in progress)
tests/                  # pytest suite (258 tests)
```

---

## Configuration reference

All settings live in `backend/config.py` and can be overridden via environment variables in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/trainsmart` | Postgres connection string |
| `ENVIRONMENT` | `development` | App environment label |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embeddings model name |
| `OLLAMA_CHAT_MODEL` | `llama3.2:3b` | Chat model name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store persistence path |
| `READINESS_WEIGHT_SLEEP` | `0.40` | Readiness component weight |
| `READINESS_WEIGHT_HR` | `0.30` | Readiness component weight |
| `READINESS_WEIGHT_LOAD` | `0.20` | Readiness component weight |
| `READINESS_WEIGHT_CONSISTENCY` | `0.10` | Readiness component weight |

---

## Common issues

**Tests fail with `database "trainsmart_test" does not exist`.** Create it: `docker exec -it $(docker ps -qf "name=postgres") psql -U postgres -c "CREATE DATABASE trainsmart_test;"`.

**`/chat` returns 500.** Most often this means either (a) Ollama isn't running — check with `curl http://localhost:11434/api/tags`, or (b) one of the models isn't pulled — verify with `ollama list`, or (c) the RAG index is empty — rebuild with the indexer command above.

**First `/chat` call hangs for 10+ seconds.** Expected — Ollama loads the model into memory on first use. Subsequent calls are fast.

**Want to use a different LLM?** Pull any Ollama-supported model (`ollama pull qwen2.5:7b`) and set `OLLAMA_CHAT_MODEL=qwen2.5:7b` in your `.env`. Larger models give better answers but use more memory.

---

## Roadmap

- [x] Days 1–5: Scaffold, models, ingestion pipeline, daily feature aggregation
- [x] Day 6: FastAPI routes (upload, daily, weekly)
- [x] Day 7: Synthetic data generator
- [x] Day 8: Readiness score with component breakdowns
- [x] Day 9: Correlation-based insights engine
- [x] Day 10: Rule-based recommendation engine
- [x] Day 11: Weekly summary computation
- [x] Day 12: LangChain RAG chain + Chroma indexing
- [x] Day 13: Chat API with persistent conversation history
- [ ] Days 14–16: Streamlit dashboard pages (Upload, Trends, Readiness, Insights, Chat)
- [ ] Day 17: K-means day-type clustering
- [ ] Day 18: Deployment polish (Render API, Streamlit Cloud frontend)
