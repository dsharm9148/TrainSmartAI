from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/trainsmart"
    environment: str = "development"

    # Readiness score weights (must sum to 1.0)
    readiness_weight_sleep: float = 0.40
    readiness_weight_hr: float = 0.30
    readiness_weight_load: float = 0.20
    readiness_weight_consistency: float = 0.10

    # Chroma
    chroma_persist_dir: str = "./chroma_db"

    # LLM (Ollama — runs locally, no API key required)
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3.2:3b"


settings = Settings()
