# TrainSmartAI

A personal health intelligence dashboard built on Apple Health export data.
Inspired by Whoop — tracks readiness, sleep, activity, and workouts, then surfaces insights and recommendations through a conversational AI assistant.

---

## Features

- **Upload** your Apple Health `export.xml` — streaming parser handles multi-GB files
- **Daily & weekly dashboards** — steps, heart rate, sleep, workout metrics
- **Readiness score** — weighted formula across sleep quality, resting HR, training load, and consistency
- **Insights engine** — correlation-based plain-English observations (e.g. "Your RHR rises 4 bpm after back-to-back workout days")
- **Recommendations** — rule-based, personalized suggestions by category (recovery, sleep, workout, habit)
- **AI chat assistant** — LangChain RAG chain over your summaries, conversation history persisted in Postgres
- **Day-type clustering** — K-means labels your days (high sleep + active, low step + recovery, etc.)

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Database | PostgreSQL + SQLAlchemy 2.0 + Alembic |
| Data processing | Pandas, NumPy |
| ML | scikit-learn (K-means) |
| RAG | LangChain + Chroma + OpenAI |
| Frontend | Streamlit + Plotly |
| Testing | pytest + httpx |
| CI | GitHub Actions |

---

## Architecture

```
Apple Health export.xml
        │
        ▼
  POST /upload
        │
   ┌────▼──────────────────────────┐
   │  Ingestion Pipeline           │
   │  parse → clean → bulk upsert │
   └────────────────┬──────────────┘
                    │
             health_records
                    │
        ┌───────────▼────────────┐
        │  Feature Aggregation   │
        │  (steps, HR, sleep,    │
        │   workouts → 1 row/day)│
        └───────────┬────────────┘
                    │
            daily_features
                    │
        ┌───────────┴──────────────┐
        │                          │
  Readiness Score           Weekly Summaries
  (sleep 40%, HR 30%,      (rollups + consistency
   load 20%, consistency    scores)
   10%)
        │                          │
        └─────────────┬────────────┘
                      │
               Insights Engine
               (correlations →
                plain English)
                      │
             Recommendations
             (rule-based, by
              category + priority)
                      │
               RAG / Chat
               (LangChain +
                Chroma over
                summaries)
```

---

## Quick Start

**Prerequisites:** Docker, Python 3.12+

```bash
# 1. Clone and install dependencies
git clone https://github.com/dsharm9148/TrainSmartAI.git
cd TrainSmartAI
pip install -r requirements.txt

# 2. Start Postgres
docker-compose up -d postgres

# 3. Create the schema
alembic upgrade head

# 4. Copy environment config
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Start the API
uvicorn backend.api.main:app --reload

# 6. Start the frontend (separate terminal)
streamlit run frontend/app.py
```

API docs available at `http://localhost:8000/docs`.

---

## Demo Data

No Apple Watch required — generate 90 days of realistic synthetic data:

```bash
python data/synthetic/generate_sample.py
# → writes data/synthetic/demo_export.xml
```

The generator produces:
- Progressive fitness trend (steps +15%, resting HR -5 bpm over 90 days)
- Post-workout fatigue effects (elevated RHR and shorter sleep the next day)
- 5 anomaly days simulating illness or travel (low steps, high HR, poor sleep)
- ~3 workouts/week across running, cycling, and walking

Upload `demo_export.xml` via the dashboard Upload page or the API:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/synthetic/demo_export.xml"
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/upload` | Ingest Apple Health export.xml |
| `GET` | `/daily` | Daily features (`?start_date=&end_date=`) |
| `GET` | `/weekly` | Weekly summaries (`?start_date=&end_date=`) |

Full interactive docs: `http://localhost:8000/docs`

---

## Running Tests

```bash
# Run all tests (requires Postgres running)
pytest tests/ -v

# Run a specific module
pytest tests/test_routes.py -v
```

77 tests across ingestion, pipeline, daily features, and API routes.

---

## Project Structure

```
backend/
├── api/           # FastAPI app, routes, Pydantic schemas
├── db/            # SQLAlchemy models, Alembic migrations, session
├── ingestion/     # XML parser, cleaner, bulk loader
├── preprocessing/ # Daily feature aggregation
├── scoring/       # Readiness score formula
├── analytics/     # Insights and weekly summaries
├── recommendations/
├── clustering/    # K-means day-type labeling
└── rag/           # LangChain chain, Chroma indexing

data/synthetic/    # Demo data generator
frontend/          # Streamlit pages
tests/             # pytest suite
```

---

## Roadmap

- [x] Days 1–5: Scaffold, models, ingestion pipeline, daily features
- [x] Day 6: FastAPI routes (upload, daily, weekly)
- [x] Day 7: Synthetic data generator, README
- [ ] Day 8: Readiness score
- [ ] Day 9: Insights engine
- [ ] Day 10: Recommendation engine
- [ ] Day 11: Weekly summaries
- [ ] Days 12–13: LangChain RAG + chat API
- [ ] Days 14–16: Streamlit dashboard pages
- [ ] Day 17: K-means clustering
- [ ] Day 18: Deployment (Render + Streamlit Cloud)
