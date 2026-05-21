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

    # Your FPL entry/team id. Set FPL_BOT_TEAM_ID in .env so live commands
    # can omit --team-id (and so it never lives in tracked files / history).
    team_id: int | None = None

    # Optional. Required only for `fpl-bot ingest oddsapi`. Free tier at
    # https://the-odds-api.com (500 req/mo).
    odds_api_key: str | None = None

    # Optional. Full Cookie header copied from an authenticated
    # fantasy.premierleague.com browser session. Enables exact current-squad
    # `my-team/{team_id}` ingest: purchase prices, sell prices, bank and FT.
    fpl_cookie: str | None = None

    # Optional local overrides for exact live state when authenticated FPL
    # ingest is unavailable.
    live_state_overrides_path: Path = Path("configs/live_state_overrides.yaml")

    # The historical GW-end price model failed its gates. Keep it disabled by
    # default; v2 snapshot features can be validated before re-enabling.
    enable_shelved_price_predictor: bool = False

    @property
    def raw_dir(self) -> Path:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        return self.raw_data_dir


settings = Settings()
