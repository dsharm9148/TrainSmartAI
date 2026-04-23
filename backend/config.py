from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/trainsmart"
    openai_api_key: str = ""
    environment: str = "development"

    # Readiness score weights (must sum to 1.0)
    readiness_weight_sleep: float = 0.40
    readiness_weight_hr: float = 0.30
    readiness_weight_load: float = 0.20
    readiness_weight_consistency: float = 0.10

    # Chroma
    chroma_persist_dir: str = "./chroma_db"


settings = Settings()
