from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FPL_BOT_",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://fpl:fpl@localhost:5432/fpl_bot"
    raw_data_dir: Path = Path("data/raw")
    user_agent: str = "fpl-bot/0.1 (research; non-commercial)"
    request_timeout_seconds: float = 30.0

    # Optional. Required only for `fpl-bot ingest oddsapi`. Free tier at
    # https://the-odds-api.com (500 req/mo).
    odds_api_key: str | None = None

    @property
    def raw_dir(self) -> Path:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        return self.raw_data_dir


settings = Settings()
