from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FPL_BOT_",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://fpl:fpl@localhost:5432/fpl_bot"

    # The season being played, as an FPL season_id (26 = 2026-27). Single
    # source of truth for CLI `--season-id` defaults so a new season needs one
    # edit here (or FPL_BOT_CURRENT_SEASON_ID in .env) rather than a hunt
    # through command signatures — a stale default silently writes this
    # season's data into last season's rows.
    current_season_id: int = 26

    # Seasons the prediction models train on, comma-separated season_ids.
    # Append the current season once it has ~5+ played GWs (see
    # docs/RUNBOOK_live.md); before that its rolling features are too thin.
    train_seasons: str = "19,20,21,22,23,24,25"

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

    # Phase 8: weight given to anytime-goalscorer market rates when blending
    # with the model's per-90 goal rate. Deliberately 0.0 — the feature has no
    # backtest (no free historical player props), so it ships inert and logs a
    # model-vs-market shadow comparison every gameweek. Raise it only once that
    # log shows the market rate is better calibrated against actual goals; see
    # docs/design/phase8_player_prop_odds.md.
    player_prop_market_weight: float = 0.0

    @property
    def raw_dir(self) -> Path:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        return self.raw_data_dir

    @property
    def train_season_ids(self) -> list[int]:
        return [int(s) for s in self.train_seasons.split(",") if s.strip()]


settings = Settings()
