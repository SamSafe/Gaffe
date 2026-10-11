from fpl_bot.config import Settings


def test_blank_optional_environment_values_use_defaults(monkeypatch):
    for key in ("TEAM_ID", "ODDS_API_KEY", "FPL_ACCESS_TOKEN", "FPL_COOKIE"):
        monkeypatch.setenv(f"FPL_BOT_{key}", "")
    config = Settings(_env_file=None)
    assert config.team_id is None
    assert config.odds_api_key is None
    assert config.fpl_access_token is None
    assert config.fpl_cookie is None


def test_explicit_team_id_from_environment(monkeypatch):
    monkeypatch.setenv("FPL_BOT_TEAM_ID", "718959")
    assert Settings(_env_file=None).team_id == 718959
