from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.dashboard import router as dashboard_router
from backend.api.routes.ingestion import router as ingestion_router
from backend.db.session import check_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not check_connection():
        raise RuntimeError(
            "Cannot connect to database. Is Postgres running? "
            "Try: docker-compose up -d postgres"
        )
    print("[trainsmart] Database connection OK")
    yield


app = FastAPI(
    title="TrainSmartAI",
    description="Personal health intelligence dashboard powered by Apple Health data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this if deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router)
app.include_router(dashboard_router)


@app.get("/health", tags=["meta"])
def health_check():
    """Liveness check — confirms the API is running and DB is reachable."""
    db_ok = check_connection()
    return {"status": "ok", "database": "connected" if db_ok else "unreachable"}
