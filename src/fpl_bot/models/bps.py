"""BPS / Bonus Points simulator (Phase 2.4).

Combines Phase 2.1-2.3 outputs with per-event simulation; ranks within-fixture
to assign 3/2/1 bonus. Per Phase 0 §4.5: simulate the mechanism, do NOT
regress on historical bonus.

v1 simplifications (per design round-1 review):
  - Independent player goal sampling. Sum across teammates won't match the
    sampled team_score on every iteration. Acknowledged as bonus-concentration
    distortion; PRIORITY V2 candidate is Multinomial(team_score, normalized_λ).
  - Independent player minutes sampling (no 5-sub cap); v2 candidate.
  - 4-bucket minutes (0/30/70/90 midpoints). p_full from Phase 2.1 split into
    p_70 + p_90 via per-position α = P(<90 | 60+) fitted from training data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from fpl_bot.db.event_source import EventSource
from fpl_bot.models.xpts import (
    HAUL_THRESHOLDS,
    XPTS_HIST_BINS,
    XPTS_HIST_MAX,
    XPTS_HIST_MIN,
    score_fpl_points,
)

# ── BPS rule table (FPL 2024/25) ──────────────────────────────────────────────
# Per-event integer BPS contribution. Positions: GKP, DEF, MID, FWD.
# References: FPL official BPS rules + community-verified breakdowns.
# Note: "saves" rule is +2 BPS per 3 saves (rounded down); see _bps_from_saves.

BPS_60_MIN = 6  # +6 BPS for an appearance ≥ 60 min

BPS_GOAL_BY_POSITION: dict[str, int] = {
    "GKP": 24,
    "DEF": 24,
    "MID": 18,
    "FWD": 12,
}

BPS_ASSIST = 9
BPS_CLEAN_SHEET_GK_DEF = 12  # +12 BPS for GK/DEF on a 90+ min appearance with a CS
BPS_PEN_SAVE = 9
BPS_PEN_MISS = -6
BPS_YELLOW = -3
BPS_RED = -9
BPS_OWN_GOAL = -6
# Goals conceded: -1 BPS per goal conceded for GK/DEF (90+ min only).


def _bps_from_saves(saves: int) -> int:
    """+2 BPS per 3 saves (integer division)."""
    if saves <= 0:
        return 0
    return 2 * (saves // 3)


def score_bps_known_events(
    *,
    position: str,
    minutes: int,
    goals: int,
    assists: int,
    team_clean_sheet: bool,
    team_goals_conceded: int,
    saves: int = 0,
    yellow_cards: int = 0,
    red_cards: int = 0,
    own_goals: int = 0,
    penalties_scored: int = 0,
    penalties_missed_or_saved: int = 0,
) -> int:
    """BPS contribution from the events we either model directly or measure."""
    bps = 0
    if minutes >= 60:
        bps += BPS_60_MIN
    if goals > 0:
        bps += goals * BPS_GOAL_BY_POSITION.get(position, 12)
    if assists > 0:
        bps += assists * BPS_ASSIST
    if position in ("GKP", "DEF") and minutes >= 60 and team_clean_sheet:
        bps += BPS_CLEAN_SHEET_GK_DEF
    if position in ("GKP", "DEF") and minutes >= 90 and team_goals_conceded > 0:
        bps -= team_goals_conceded
    if saves > 0 and position == "GKP":
        bps += _bps_from_saves(saves)
    bps += penalties_missed_or_saved * BPS_PEN_MISS
    bps -= yellow_cards * 3
    bps -= red_cards * 9
    bps -= own_goals * 6
    return bps


# ── Bonus assignment ──────────────────────────────────────────────────────────


def assign_bonus_within_fixture(
    bps_array: np.ndarray, player_ids: np.ndarray
) -> np.ndarray:
    """Given per-player simulated BPS for one fixture, return per-player bonus
    points (3/2/1/0). FPL's tie-break rules:
      - Bonus 3 to the highest BPS (multiple players if tied for top → all 3)
      - Bonus 2 to next highest (multiple if tied → all 2; remaining bonus pts
        absorbed by the tie)
      - Bonus 1 to the third (multiple if tied → all 1)

    For v1 we implement the "split tied bonuses" version:
      - All players tied at rank 1 get 3.
      - If rank 1 has k players, ranks 2 & 3 are skipped — they get nothing
        (FPL absorbs the conflict at the top).
    """
    bonus = np.zeros(len(bps_array), dtype=np.int8)
    if len(bps_array) == 0:
        return bonus

    sorted_unique_bps = np.sort(np.unique(bps_array))[::-1]
    if len(sorted_unique_bps) >= 1:
        top = sorted_unique_bps[0]
        bonus[bps_array == top] = 3
    if len(sorted_unique_bps) >= 2:
        second = sorted_unique_bps[1]
        bonus[bps_array == second] = 2
    if len(sorted_unique_bps) >= 3:
        third = sorted_unique_bps[2]
        bonus[bps_array == third] = 1
    return bonus


# ── Minutes-bucket sampling ───────────────────────────────────────────────────


def split_p_full_by_position(
    train_df: pl.DataFrame,
) -> dict[str, float]:
    """Compute α = P(60-89 minutes | 60+ minutes) per position from training data.

    Used to split the Phase 2.1 minutes model's p_full into p_70 + p_90.
    train_df needs columns: position_code, minutes.
    """
    alphas: dict[str, float] = {}
    for pos in ("GKP", "DEF", "MID", "FWD"):
        sub = train_df.filter(
            (pl.col("position_code") == pos) & (pl.col("minutes") >= 60)
        )
        if sub.is_empty():
            alphas[pos] = 0.4  # league-typical fallback
            continue
        n_60_89 = sub.filter(pl.col("minutes") < 90).height
        n_total = sub.height
        alphas[pos] = float(n_60_89) / float(n_total) if n_total > 0 else 0.4
    return alphas


def sample_minutes_bucket(
    p_zero: float,
    p_short: float,
    p_full: float,
    alpha_60_89: float,
    rng: np.random.Generator,
) -> int:
    """Sample one of {0, 30, 70, 90} per the design's bucket midpoints.

    Splits p_full → (p_70, p_90) using the position-conditional alpha.
    """
    p_full = max(0.0, p_full)
    p_70 = p_full * alpha_60_89
    p_90 = p_full * (1.0 - alpha_60_89)
    probs = np.array([p_zero, p_short, p_70, p_90], dtype=np.float64)
    s = probs.sum()
    if s <= 0:
        # degenerate; default to "0 minutes"
        return 0
    probs /= s
    idx = int(rng.choice(4, p=probs))
    return MINUTES_BUCKET_MIDPOINTS_LIST[idx]


MINUTES_BUCKET_MIDPOINTS_LIST: list[int] = [0, 30, 70, 90]


# ── Core simulator ────────────────────────────────────────────────────────────


@dataclass
class FixtureInputs:
    """Per-fixture model predictions feeding the simulator."""

    fixture_id: int
    season_id: int
    gameweek: int
    home_team_id: int
    away_team_id: int
    home_team_lambda: float  # Dixon-Coles lambda for home team's goals
    away_team_lambda: float
    home_market_cs_prob: float
    away_market_cs_prob: float
    # Per-player rows; columns expected:
    #   player_id, position_code, team_id (= home or away), is_home,
    #   p_minutes_zero, p_minutes_short, p_minutes_full,
    #   lambda_goals_per_90, lambda_assists_per_90,
    #   saves_rate_per_90, yc_rate_per_90, rc_rate_per_90,
    #   is_penalty_taker
    players: pl.DataFrame
    alphas_by_position: dict[str, float]


@dataclass
class BPSSimulator:
    event_source: EventSource
    n_iterations: int = 500
    seed: int = 42
    pen_per_match_lambda: float = 0.27  # league-typical penalty rate
    pen_conversion: float = 0.78
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def simulate_fixture(
        self,
        inputs: FixtureInputs,
        *,
        return_raw_xpts_samples: bool = False,
    ) -> pl.DataFrame | tuple[pl.DataFrame, np.ndarray]:
        """Per-player bonus + xPts distribution for one fixture.

        Output DataFrame columns:
          player_id, p_bonus_0..p_bonus_3, expected_bonus,
          e_xpts, var_xpts, p_xpts_ge_2, p_xpts_ge_6, p_xpts_ge_10, p_xpts_ge_15,
          xpts_pmf (list[float] of length 31; bin 0 = -5 pts, bin 30 = +25 pts).

        If `return_raw_xpts_samples=True`, returns a tuple
        (DataFrame, np.int16 array of shape (n_players, n_iterations)).
        """
        players = inputs.players
        n_players = players.height
        if n_players == 0:
            empty = pl.DataFrame()
            if return_raw_xpts_samples:
                return empty, np.zeros((0, 0), dtype=np.int16)
            return empty

        player_ids = players["player_id"].to_numpy()
        positions = players["position_code"].to_list()
        is_home = players["is_home"].to_numpy()
        p_zeros = players["p_minutes_zero"].to_numpy()
        p_shorts = players["p_minutes_short"].to_numpy()
        p_fulls = players["p_minutes_full"].to_numpy()
        lambda_g = players["lambda_goals_per_90"].to_numpy()
        lambda_a = players["lambda_assists_per_90"].to_numpy()
        saves_rate = players["saves_rate_per_90"].to_numpy()
        yc_rate = players["yc_rate_per_90"].to_numpy()
        rc_rate = players["rc_rate_per_90"].to_numpy()
        is_pen_taker = players["is_penalty_taker"].to_numpy()

        bonus_counts = np.zeros((n_players, 4), dtype=np.int64)
        xpts_samples = np.zeros((n_players, self.n_iterations), dtype=np.int16)

        # Per-iteration scratch buffers (reset each iter)
        minutes_buf = np.zeros(n_players, dtype=np.int16)
        goals_buf = np.zeros(n_players, dtype=np.int16)
        assists_buf = np.zeros(n_players, dtype=np.int16)
        team_cs_buf = np.zeros(n_players, dtype=np.bool_)
        team_gc_buf = np.zeros(n_players, dtype=np.int16)
        saves_buf = np.zeros(n_players, dtype=np.int16)
        yc_buf = np.zeros(n_players, dtype=np.int8)
        rc_buf = np.zeros(n_players, dtype=np.int8)
        pens_missed_buf = np.zeros(n_players, dtype=np.int8)

        for s in range(self.n_iterations):
            h_score = int(self._rng.poisson(inputs.home_team_lambda))
            a_score = int(self._rng.poisson(inputs.away_team_lambda))
            home_cs = (a_score == 0)
            away_cs = (h_score == 0)
            home_pens = int(self._rng.poisson(self.pen_per_match_lambda / 2))
            away_pens = int(self._rng.poisson(self.pen_per_match_lambda / 2))

            bps_per_player = np.zeros(n_players, dtype=np.float64)
            minutes_buf.fill(0)
            goals_buf.fill(0)
            assists_buf.fill(0)
            team_cs_buf.fill(False)
            team_gc_buf.fill(0)
            saves_buf.fill(0)
            yc_buf.fill(0)
            rc_buf.fill(0)
            pens_missed_buf.fill(0)

            for i in range(n_players):
                pos = positions[i]
                player_home = bool(is_home[i])
                team_cs = home_cs if player_home else away_cs
                team_gc = a_score if player_home else h_score
                pens_for_team = home_pens if player_home else away_pens

                minutes = sample_minutes_bucket(
                    p_zeros[i],
                    p_shorts[i],
                    p_fulls[i],
                    inputs.alphas_by_position.get(pos, 0.4),
                    self._rng,
                )
                minutes_buf[i] = minutes
                team_cs_buf[i] = team_cs
                team_gc_buf[i] = team_gc
                if minutes <= 0:
                    continue

                minutes_factor = minutes / 90.0

                goals = int(self._rng.poisson(max(0.0, lambda_g[i]) * minutes_factor))
                assists = int(
                    self._rng.poisson(max(0.0, lambda_a[i]) * minutes_factor)
                )

                saves = int(
                    self._rng.poisson(max(0.0, saves_rate[i]) * minutes_factor)
                ) if pos == "GKP" else 0
                yc_drawn = self._rng.random() < (
                    max(0.0, yc_rate[i]) * minutes_factor
                )
                rc_drawn = self._rng.random() < (
                    max(0.0, rc_rate[i]) * minutes_factor
                )

                pens_scored = 0
                pens_missed = 0
                if bool(is_pen_taker[i]) and pens_for_team > 0:
                    for _ in range(pens_for_team):
                        if self._rng.random() < self.pen_conversion:
                            pens_scored += 1
                        else:
                            pens_missed += 1
                goals += pens_scored

                goals_buf[i] = goals
                assists_buf[i] = assists
                saves_buf[i] = saves
                yc_buf[i] = int(yc_drawn)
                rc_buf[i] = int(rc_drawn)
                pens_missed_buf[i] = pens_missed

                bps_known = score_bps_known_events(
                    position=pos,
                    minutes=minutes,
                    goals=goals,
                    assists=assists,
                    team_clean_sheet=team_cs,
                    team_goals_conceded=team_gc,
                    saves=saves,
                    yellow_cards=int(yc_drawn),
                    red_cards=int(rc_drawn),
                    penalties_missed_or_saved=pens_missed,
                )
                bps_residual = self.event_source.simulate_unmodeled_bps(
                    pos, minutes, self._rng
                )
                bps_per_player[i] = bps_known + bps_residual

            # Bonus depends on rank — must come after all per-player BPS
            bonus_array = assign_bonus_within_fixture(bps_per_player, player_ids)

            # FPL points use the same draws + the just-computed bonus
            for i in range(n_players):
                if minutes_buf[i] <= 0:
                    xpts = 0
                else:
                    xpts = score_fpl_points(
                        position=positions[i],
                        minutes=int(minutes_buf[i]),
                        goals=int(goals_buf[i]),
                        assists=int(assists_buf[i]),
                        team_clean_sheet=bool(team_cs_buf[i]),
                        team_goals_conceded=int(team_gc_buf[i]),
                        saves=int(saves_buf[i]),
                        yellow_cards=int(yc_buf[i]),
                        red_cards=int(rc_buf[i]),
                        penalties_missed_or_saved=int(pens_missed_buf[i]),
                        bonus=int(bonus_array[i]),
                    )
                # Clip to histogram range; rare outliers stack at edge
                if xpts < XPTS_HIST_MIN:
                    xpts = XPTS_HIST_MIN
                elif xpts > XPTS_HIST_MAX:
                    xpts = XPTS_HIST_MAX
                xpts_samples[i, s] = xpts
                bonus_counts[i, int(bonus_array[i])] += 1

        # Aggregate
        n = self.n_iterations
        e_xpts = xpts_samples.mean(axis=1)
        var_xpts = xpts_samples.var(axis=1, ddof=1) if n > 1 else np.zeros(n_players)
        tail_probs: dict[int, np.ndarray] = {
            t: (xpts_samples >= t).mean(axis=1) for t in HAUL_THRESHOLDS
        }
        # PMF: bins indexed 0=XPTS_HIST_MIN ... XPTS_HIST_BINS-1=XPTS_HIST_MAX
        pmf_matrix = np.zeros((n_players, XPTS_HIST_BINS), dtype=np.float64)
        for i in range(n_players):
            counts = np.bincount(
                xpts_samples[i] - XPTS_HIST_MIN, minlength=XPTS_HIST_BINS
            )[:XPTS_HIST_BINS]
            pmf_matrix[i] = counts / n

        rows = []
        for i in range(n_players):
            p = bonus_counts[i] / n
            rows.append(
                {
                    "player_id": int(player_ids[i]),
                    "p_bonus_0": float(p[0]),
                    "p_bonus_1": float(p[1]),
                    "p_bonus_2": float(p[2]),
                    "p_bonus_3": float(p[3]),
                    "expected_bonus": float(
                        0 * p[0] + 1 * p[1] + 2 * p[2] + 3 * p[3]
                    ),
                    "e_xpts": float(e_xpts[i]),
                    "var_xpts": float(var_xpts[i]),
                    "p_xpts_ge_2": float(tail_probs[2][i]),
                    "p_xpts_ge_6": float(tail_probs[6][i]),
                    "p_xpts_ge_10": float(tail_probs[10][i]),
                    "p_xpts_ge_15": float(tail_probs[15][i]),
                    "xpts_pmf": pmf_matrix[i].tolist(),
                }
            )

        out_df = pl.DataFrame(rows)
        if return_raw_xpts_samples:
            return out_df, xpts_samples
        return out_df


# ── Residual-fitting helpers ──────────────────────────────────────────────────


def fit_residual_dataset(
    train_player_match: pl.DataFrame,
    positions: pl.DataFrame,
) -> pl.DataFrame:
    """Build the residual training DataFrame for EmpiricalResidualEventSource.fit.

    Inputs:
      train_player_match: per-(player, fixture) actual events from
        pit.all_player_match_with_kickoff. Required cols: player_id, minutes,
        goals, assists, clean_sheet (player-level FPL eligibility),
        goals_conceded, was_home, bonus, bps. Plus team-level via aggregation.
      positions: per-player position_code from pit.all_player_positions.

    Returns DataFrame with: player_id, fixture_id, position_code, minutes,
    actual_bps, simulated_bps_known, residual_bps. Only rows with minutes >= 1
    are included (bench players have no BPS to model).
    """
    df = train_player_match.filter(pl.col("minutes") >= 1).filter(
        pl.col("bps").is_not_null()
    )
    df = df.join(positions, on="player_id", how="left").drop_nulls("position_code")

    # Team-level cs and goals_conceded: aggregate per (fixture_id, team) from
    # was_home + home_team_id/away_team_id, then derive cs = (goals_conceded == 0).
    df = df.with_columns(
        pl.when(pl.col("was_home"))
        .then(pl.col("home_team_id"))
        .otherwise(pl.col("away_team_id"))
        .alias("team_id_resolved"),
    )
    team_gc = (
        df.group_by(["team_id_resolved", "fixture_id"])
        .agg(pl.col("goals_conceded").max().alias("team_goals_conceded"))
        .with_columns((pl.col("team_goals_conceded") == 0).alias("team_clean_sheet"))
    )
    df = df.join(
        team_gc,
        on=["team_id_resolved", "fixture_id"],
        how="left",
    )

    # Compute simulated BPS from actual events
    sim_bps = []
    for row in df.iter_rows(named=True):
        sim_bps.append(
            score_bps_known_events(
                position=row["position_code"],
                minutes=row["minutes"] or 0,
                goals=row["goals"] or 0,
                assists=row["assists"] or 0,
                team_clean_sheet=bool(row["team_clean_sheet"]),
                team_goals_conceded=row["team_goals_conceded"] or 0,
                saves=row.get("saves") or 0,
                yellow_cards=row.get("yellow_cards") or 0,
                red_cards=row.get("red_cards") or 0,
            )
        )
    df = df.with_columns(pl.Series(name="simulated_bps_known", values=sim_bps))
    df = df.with_columns(
        (pl.col("bps").cast(pl.Int64) - pl.col("simulated_bps_known"))
        .alias("residual_bps")
    )
    return df.select(
        "player_id",
        "fixture_id",
        "position_code",
        "minutes",
        pl.col("bps").alias("actual_bps"),
        "simulated_bps_known",
        "residual_bps",
    )
