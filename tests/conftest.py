import os
from pathlib import Path

import pytest

# Ensure test env vars are set before any imports that load config
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/trainsmart_test")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
