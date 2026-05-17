"""Phase 6 post-GW retrospective.

After a GW closes, pull the actual outcomes (minutes + points per player),
apply the same auto-sub scorer the backtest uses, and compute realized
points for the recommendation we wrote pre-deadline. Append to
`actuals.json` in the same per-GW directory.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from sqlalchemy import select

from fpl_bot.db.models import DimFixture, FactPlayerMatch
from fpl_bot.db.session import session_scope
from fpl_bot.optim.scorer import ScorerInputs, score_gw
from fpl_bot.optim.state import GwDecisions

OUTPUT_ROOT = Path("data/live/recommendations")


def _per_player_actuals_for_gw(
    season_id: int, gameweek: int
) -> tuple[dict[int, int], dict[int, int]]:
    """Returns (actual_pts_by_pid, actual_minutes_by_pid) for the GW.

    Sums across DGW fixtures. Uses latest recorded_at per (player, fixture).
    """
    with session_scope() as s:
        from sqlalchemy import func as sa_func
        latest = (
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.fixture_id,
                sa_func.max(FactPlayerMatch.recorded_at).label("max_rec"),
            )
            .group_by(FactPlayerMatch.player_id, FactPlayerMatch.fixture_id)
            .subquery("fpm_latest")
        )
        rows = s.execute(
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.total_points,
                FactPlayerMatch.minutes,
                DimFixture.gameweek,
            )
            .join(
                latest,
                (latest.c.player_id == FactPlayerMatch.player_id)
                & (latest.c.fixture_id == FactPlayerMatch.fixture_id)
                & (latest.c.max_rec == FactPlayerMatch.recorded_at),
            )
            .join(DimFixture, DimFixture.fixture_id == FactPlayerMatch.fixture_id)
            .where(DimFixture.season_id == season_id)
            .where(DimFixture.gameweek == gameweek)
        ).all()

    pts: dict[int, int] = {}
    mins: dict[int, int] = {}
    for r in rows:
        pid = int(r.player_id)
        pts[pid] = pts.get(pid, 0) + int(r.total_points or 0)
        mins[pid] = mins.get(pid, 0) + int(r.minutes or 0)
    return pts, mins


def compute_retrospective(
    *,
    season_id: int,
    gameweek: int,
    team_id: int,
    output_dir: Path | None = None,
) -> Path:
    """Read the prior recommendation.json, fetch actuals, score, write actuals.json.

    Returns the path to actuals.json.
    """
    if output_dir is None:
        output_dir = OUTPUT_ROOT / f"season_{season_id}" / f"gw_{gameweek}"
    rec_path = output_dir / "recommendation.json"
    if not rec_path.exists():
        raise FileNotFoundError(f"No recommendation at {rec_path}")

    rec = json.loads(rec_path.read_text())
    d = rec["decisions"]

    actual_pts, actual_minutes = _per_player_actuals_for_gw(season_id, gameweek)

    # Build a positions dict for the scorer
    from fpl_bot.db import pit
    positions = (
        pit.all_player_positions()
        .to_pandas()
        .set_index("player_id")["position_code"]
        .to_dict()
    )

    decisions = GwDecisions(
        gameweek=d["gameweek"],
        squad=frozenset(d["squad"]),
        starting_xi=frozenset(d["starting_xi"]),
        captain=d["captain"],
        vice=d["vice"],
        transferred_in=frozenset(d["transferred_in"]),
        transferred_out=frozenset(d["transferred_out"]),
        chip_played=d["chip_played"],
        hits=d["hits"],
        objective_value=d["objective_value"],
    )

    # Use the chip-aware scorer with bench order by predicted xPts. We don't
    # have the original pred_dict here, so fall back to actual_pts as a
    # proxy (the bench order matters only for auto-sub; using actuals is
    # an OK retroactive approximation).
    bench_order_xpts = {p: float(actual_pts.get(p, 0)) for p in decisions.squad}

    scorer_out = score_gw(
        ScorerInputs(
            decisions=decisions,
            actual_pts={p: actual_pts.get(p, 0) for p in decisions.squad},
            actual_minutes={p: actual_minutes.get(p, 0) for p in decisions.squad},
            positions={p: positions.get(p, "MID") for p in decisions.squad},
            bench_order_xpts=bench_order_xpts,
        )
    )

    actuals_path = output_dir / "actuals.json"
    actuals_path.write_text(
        json.dumps(
            {
                "computed_at": dt.datetime.now(dt.UTC).isoformat(),
                "season_id": season_id,
                "gameweek": gameweek,
                "team_id": team_id,
                "gw_points": scorer_out.gw_points,
                "auto_subs": list(scorer_out.auto_subs),
                "used_vice": scorer_out.used_vice,
                "captain_final": scorer_out.captain_final,
                "predicted_obj_value": decisions.objective_value,
            },
            indent=2,
        )
    )
    return actuals_path
