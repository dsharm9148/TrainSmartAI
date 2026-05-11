import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use a dedicated test database so the live `trainsmart` DB is never touched.
# The test engine creates all tables at session start and drops them at the end,
# which would destroy live data if pointed at the main DB.
TEST_DB_URL = "postgresql://postgres:postgres@localhost:5432/trainsmart_test"

os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ.setdefault("OPENAI_API_KEY", "test-key")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def db_engine():
    """Create tables once per test session in trainsmart_test, drop when done."""
    # Import after env var is set so engine uses the test DB URL
    from backend.db.session import Base
    import backend.db.models  # noqa: F401 — registers all models with Base

    engine = create_engine(TEST_DB_URL)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """
    Provide a clean DB session for each test.
    Rolls back all changes after the test so tests stay isolated.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
