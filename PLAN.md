# TrainSmartAI — Project Plan

> I loved using Whoop, but after leaving the company I didn't want to keep paying for the subscription. I still had an old Apple Watch, so I built my own "Whoop-like" dashboard using Apple Health export data. I added an LLM-powered assistant to help me plan runs, workouts, and habits, and I created a simple recovery/readiness score to bring back the gamified experience I liked most.

---

## Table of Contents

1. [Folder Structure](#folder-structure)
2. [Architecture Decisions](#architecture-decisions)
3. [Database Schema](#database-schema)
4. [Week 1 Tasks](#week-1-tasks)
5. [Full 5-Week Roadmap](#full-5-week-roadmap)
6. [Tradeoffs and Scope Cuts](#tradeoffs-and-scope-cuts)

---

## Folder Structure

```
TrainSmartAI/
├── backend/
│   ├── api/
│   │   ├── main.py                  # FastAPI app entrypoint
│   │   └── routes/
│   │       ├── ingestion.py         # /upload, /ingest
│   │       ├── dashboard.py         # /daily, /weekly
│   │       ├── insights.py          # /insights
│   │       ├── recommendations.py   # /recommendations
│   │       ├── readiness.py         # /readiness
│   │       ├── clusters.py          # /clusters
│   │       └── chat.py              # /chat
│   ├── ingestion/
│   │   ├── parser.py                # Apple Health XML → raw dicts
│   │   ├── cleaner.py               # normalize units, filter bad values
│   │   └── loader.py                # upsert cleaned records into Postgres
│   ├── preprocessing/
│   │   ├── daily_features.py        # aggregate raw records → one row/day
│   │   └── weekly_features.py       # aggregate daily → one row/week
│   ├── feature_engineering/
│   │   ├── sleep_features.py        # duration, consistency, bedtime windows
│   │   ├── hr_features.py           # resting HR trend, 7-day rolling avg
│   │   ├── activity_features.py     # steps, active energy, rolling load
│   │   └── workout_features.py      # frequency, volume, recent strain
│   ├── scoring/
│   │   └── readiness.py             # weighted formula → 0-100 score
│   ├── analytics/
│   │   ├── insights.py              # correlation-based plain-English insights
│   │   └── correlations.py          # sleep vs HR, sleep vs steps, etc.
│   ├── recommendations/
│   │   └── engine.py                # rule-based personalized recs
│   ├── clustering/
│   │   └── day_types.py             # K-means on daily features
│   ├── rag/
│   │   ├── summary_indexer.py       # embed daily/weekly summaries → Chroma
│   │   ├── retriever.py             # query Chroma for relevant context
│   │   ├── chain.py                 # LangChain RAG chain
│   │   └── prompts.py               # system prompt templates
│   ├── db/
│   │   ├── models.py                # SQLAlchemy ORM models
│   │   ├── session.py               # DB connection + session factory
│   │   └── migrations/              # Alembic migration files
│   ├── utils/
│   │   ├── date_utils.py            # timezone handling, week boundaries
│   │   └── unit_converter.py        # Apple Health unit normalization
│   └── config.py                    # env vars, settings via pydantic-settings
│
├── frontend/
│   ├── Home.py                      # Streamlit entrypoint
│   └── pages/
│       ├── 1_Upload.py
│       ├── 2_Dashboard.py
│       ├── 3_Trends.py
│       ├── 4_Insights.py
│       ├── 5_Recommendations.py
│       ├── 6_Readiness.py
│       ├── 7_Assistant.py
│       └── 8_Day_Types.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_scoring.py
│   ├── test_recommendations.py
│   └── fixtures/
│       ├── sample_export.xml        # minimal synthetic Apple Health XML
│       └── conftest.py
│
├── data/
│   └── synthetic/
│       └── generate_sample.py       # script to produce demo-safe fake data
│
├── docker-compose.yml               # Postgres + pgAdmin only
├── .env.example
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml                   # lint + test on push
└── README.md
```

---

## Architecture Decisions

### FastAPI over Flask
Auto-generates OpenAPI docs (impressive in demos), native async support, and Pydantic validation — all visible skills to an interviewer. Flask would work but signals less.

**Supports:** software engineering, API design

### PostgreSQL over SQLite
SQLite would be simpler, but Postgres signals production thinking. The time-series health data benefits from real date functions. SQLAlchemy ORM + Alembic migrations is a standard data engineering pattern worth showing.

**Supports:** data engineering

### Streamlit over React
React would double the scope. Streamlit lets you ship a working, demo-ready UI in hours per page instead of days. The tradeoff is explainable in interviews: "I chose Streamlit to keep the project shippable solo in 4 weeks; I'd swap it for Next.js in a production context."

**Supports:** software engineering (scoping decisions)

### LangChain + Chroma (local)
LangChain is the standard orchestration layer on most LLM job descriptions right now. Chroma running locally (persisted to disk) keeps the RAG pipeline self-contained with no cloud dependency. Summaries are embedded — not raw records — which is an important design decision to explain.

**Supports:** ML/AI engineering, LLM applications

### Rule-based recommendations, not a trained model
A trained model would require labeled data that doesn't exist here. Rule-based logic grounded in user history is more defensible, faster to ship, and equally impressive when explained well. Frame it as "personalized heuristics derived from user history."

**Supports:** ML applications, data engineering

### Docker Compose scope: Postgres only
Containerizing just Postgres gives a reproducible database environment and lets you say "containerized" on your resume without adding weeks of ops complexity.

**Supports:** software engineering, DevOps basics

---

## Database Schema

```sql
-- Raw records: one row per Apple Health data point
-- Supports re-ingestion via unique constraint + upsert
CREATE TABLE health_records (
    id            SERIAL PRIMARY KEY,
    record_type   VARCHAR(120) NOT NULL,  -- e.g. HKQuantityTypeIdentifierStepCount
    source_name   VARCHAR(100),
    value         FLOAT,
    unit          VARCHAR(50),
    start_date    TIMESTAMPTZ NOT NULL,
    end_date      TIMESTAMPTZ NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (record_type, source_name, start_date, end_date)
);

-- Daily features: one row per calendar day
-- Built by preprocessing layer from health_records
CREATE TABLE daily_features (
    id                   SERIAL PRIMARY KEY,
    date                 DATE UNIQUE NOT NULL,
    steps                INTEGER,
    avg_heart_rate       FLOAT,
    resting_heart_rate   FLOAT,
    sleep_duration_hrs   FLOAT,
    sleep_start          TIMESTAMPTZ,
    sleep_end            TIMESTAMPTZ,
    workout_count        INTEGER,
    workout_minutes      FLOAT,
    workout_calories     FLOAT,
    active_energy_kcal   FLOAT,
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Weekly summaries: one row per week (Monday–Sunday)
CREATE TABLE weekly_summaries (
    id                       SERIAL PRIMARY KEY,
    week_start               DATE UNIQUE NOT NULL,
    week_end                 DATE NOT NULL,
    avg_daily_steps          FLOAT,
    avg_sleep_hrs            FLOAT,
    sleep_consistency_score  FLOAT,  -- lower std dev = more consistent
    avg_resting_hr           FLOAT,
    total_workout_minutes    FLOAT,
    workout_days             INTEGER,
    avg_readiness_score      FLOAT,
    updated_at               TIMESTAMPTZ DEFAULT NOW()
);

-- Readiness scores: one row per day
-- Formula components stored separately for explainability
CREATE TABLE readiness_scores (
    id                    SERIAL PRIMARY KEY,
    date                  DATE UNIQUE NOT NULL,
    score                 FLOAT NOT NULL,     -- 0-100
    sleep_score           FLOAT,              -- component: sleep duration + consistency
    hr_score              FLOAT,              -- component: resting HR vs personal baseline
    load_score            FLOAT,              -- component: recent workout load (7-day)
    consistency_score     FLOAT,              -- component: habit consistency
    explanation           TEXT,               -- human-readable breakdown
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

-- Insights: generated plain-English observations
CREATE TABLE insights (
    id            SERIAL PRIMARY KEY,
    generated_for DATE NOT NULL,
    insight_type  VARCHAR(60),   -- 'sleep_activity', 'hr_trend', 'workout_pattern', 'weekly'
    text          TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Recommendations: personalized, rule-based
CREATE TABLE recommendations (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    category    VARCHAR(50),    -- 'recovery', 'workout', 'sleep', 'habit'
    priority    SMALLINT,       -- 1=high, 2=medium, 3=low
    text        TEXT NOT NULL,
    reasoning   TEXT,           -- why this rec was triggered (interview-friendly)
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Day-type clusters from K-means
CREATE TABLE cluster_assignments (
    id            SERIAL PRIMARY KEY,
    date          DATE UNIQUE NOT NULL,
    cluster_id    SMALLINT,
    cluster_label VARCHAR(100),  -- e.g. 'high sleep, active day'
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Chat sessions: groups a conversation
CREATE TABLE chat_sessions (
    id          SERIAL PRIMARY KEY,
    session_id  UUID DEFAULT gen_random_uuid() UNIQUE,
    started_at  TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW()
);

-- Chat messages: individual turns
CREATE TABLE chat_messages (
    id          SERIAL PRIMARY KEY,
    session_id  UUID REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for the most common query patterns
CREATE INDEX idx_health_records_type_date ON health_records (record_type, start_date);
CREATE INDEX idx_daily_features_date      ON daily_features (date);
CREATE INDEX idx_readiness_date           ON readiness_scores (date);
CREATE INDEX idx_chat_messages_session    ON chat_messages (session_id, created_at);
```

### Schema design notes

- `health_records` has a composite unique key — makes re-ingestion safe (upsert on conflict)
- Readiness score components stored as separate columns for explainable formula breakdown
- `recommendations.reasoning` stores why each rule fired — shows explainability thinking
- Chat tables are simple but complete enough to support session-based history retrieval for RAG

---

## Week 1 Tasks

**Goal:** Working data pipeline from raw XML → clean daily features in Postgres, with a running FastAPI server.

### Day 1 — Project scaffold
- [ ] Initialize repo, create folder structure, `requirements.txt`, `.env.example`
- [ ] Set up virtual environment
- [ ] Docker Compose with Postgres (+ optional pgAdmin)
- [ ] `config.py` using `pydantic-settings` to load env vars
- [ ] Confirm Postgres is reachable

### Day 2 — Database layer
- [ ] Write all SQLAlchemy models in `db/models.py`
- [ ] Set up Alembic, generate and run initial migration
- [ ] Write `db/session.py` with a session factory
- [ ] Confirm tables exist via pgAdmin or psql

### Day 3 — Apple Health XML parser
- [ ] Write `ingestion/parser.py` using `xml.etree.ElementTree` with `iterparse` (streaming, handles large files)
- [ ] Filter to 4 record types: steps, heart rate, sleep, workouts
- [ ] Output: list of raw dicts with `record_type`, `source_name`, `value`, `unit`, `start_date`, `end_date`
- [ ] Handle missing/malformed values without crashing
- [ ] Write `tests/fixtures/sample_export.xml` — minimal synthetic XML with ~30 days of fake data
- [ ] Write `tests/test_ingestion.py` with parse tests

### Day 4 — Cleaner and loader
- [ ] Write `ingestion/cleaner.py`: normalize units, parse Apple's timestamp format, filter impossible values (HR < 20 or > 250, sleep > 16 hrs, steps per interval > 50k)
- [ ] Write `ingestion/loader.py`: bulk upsert to `health_records` using `ON CONFLICT DO NOTHING`
- [ ] Wire up end-to-end: parse → clean → load

### Day 5 — Daily features
- [ ] Write `preprocessing/daily_features.py`
- [ ] For each date: aggregate steps (sum), resting HR (min HR in early-morning window or Apple's dedicated type), sleep (sum of records ending before 10am), workout count and minutes
- [ ] Write to `daily_features` table
- [ ] Handle days with partial or missing data — store `None`, not `0`

### Day 6 — First FastAPI routes
- [ ] `api/main.py` with FastAPI app, CORS, lifespan DB init
- [ ] `routes/ingestion.py`: `POST /upload` — accepts XML file, runs full pipeline, returns record count
- [ ] `routes/dashboard.py`: `GET /daily?start=&end=` — returns daily features as JSON
- [ ] Test the upload endpoint with synthetic XML via curl

### Day 7 — Buffer and CI
- [ ] Add `data/synthetic/generate_sample.py` — generates 90-day synthetic dataset for demos
- [ ] Wire up GitHub Actions CI: `ruff` lint + `pytest` on push
- [ ] Write the first README section: what the project does and how to run locally

---

## Full 5-Week Roadmap

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Foundation | XML parser, DB schema, daily features pipeline, FastAPI skeleton |
| 2 | Analytics + Scoring | Readiness score, correlation insights, recommendation engine, weekly summaries |
| 3 | RAG / Assistant | LangChain chain, Chroma indexing of summaries, chat API endpoints, conversation history |
| 4 | Frontend | All Streamlit pages wired to backend, charts, score card, assistant chat UI |
| 5 | Polish + Clustering + Deploy | K-means day types, synthetic demo data, README, deployment, optional LangSmith eval |

---

## Tradeoffs and Scope Cuts

| Decision | Rationale |
|----------|-----------|
| No Alembic autogenerate in CI | Run migrations manually locally; CI autogenerate adds complexity without benefit |
| Sync SQLAlchemy (not async) | Single-user app; async adds boilerplate without a real concurrency problem to solve |
| Chroma persisted to disk, not a server | Zero ops overhead; persists and restores automatically on restart |
| Rule-based recommendations only | No labeled training data exists; rules grounded in user history are more defensible and faster to ship |
| K-means clustering only | Simpler, explainable, fixed cluster count you can name and describe in a demo |
| LangSmith eval is stretch-goal | Core RAG pipeline first; only add eval dashboard in Week 5 if ahead of schedule |
| Streamlit, not React | Cuts frontend scope in half; tradeoff is known and explainable |
| Simple RAG chain, no agent loop | More predictable, easier to demo, less failure surface than a multi-step agent |
| No connection pooling | PgBouncer is overkill for a single-user portfolio project |
| Docker Compose for Postgres only | Keeps "containerized" on the resume without adding weeks of ops work |

### One scope risk to watch

Apple Health XML exports can be very large (multi-GB for several years of data). The parser must use `iterparse` for streaming rather than loading the full tree into memory. This is a real engineering decision worth explaining in interviews and documenting in the README.

---

## Stack Summary

| Layer | Technology |
|-------|------------|
| Backend API | FastAPI + Pydantic |
| Database | PostgreSQL + SQLAlchemy + Alembic |
| Data pipeline | Python (stdlib `xml`, `pandas`) |
| Scoring / analytics | Python (descriptive stats, rolling averages) |
| Clustering | scikit-learn K-means |
| LLM orchestration | LangChain |
| Vector store | Chroma (local, disk-persisted) |
| LLM provider | OpenAI-compatible API |
| Frontend | Streamlit |
| Testing | pytest |
| CI | GitHub Actions (ruff + pytest) |
| Deployment | Render (backend) + Streamlit Community Cloud (frontend) |
