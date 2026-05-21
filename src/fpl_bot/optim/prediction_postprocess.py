"""Shared prediction post-processing for backtest and live recommend.

The trained xPts stack emits raw per-(player, GW) expectations. Several
production adjustments are deliberately outside that model:
  - early-season prior blend when rolling features are sparse;
  - per-position or per-player calibration experiments;
  - DefCon additive points for seasons where FPL exposes the stat.

Keeping these in one module prevents the live path from drifting away from
the validated backtest path.
"""
from __future__ import annotations

from pathlib import Path

from fpl_bot.db import pit

DEFCON_PER_POSITION_SHRINKAGE = {"DEF": 0.4, "MID": 1.0, "FWD": 1.0}
EARLY_GW_PRIOR_BLEND = {1: 0.7, 2: 0.5, 3: 0.3}


def apply_prediction_postprocessing(
    pred_by_pgw: dict[tuple[int, int], float],
    *,
    season_id: int,
    train_seasons: list[int],
    position_calibration: dict[str, float] | None = None,
    defcon_shrinkage: float | None = None,
    defcon_per_position_shrinkage: dict[str, float] | None = DEFCON_PER_POSITION_SHRINKAGE,
    extend_defcon_future: bool = False,
    fwd_calibration: bool = False,
    cache_dir: Path = Path("data/cache/xpts_predictions"),
) -> dict[tuple[int, int], float]:
    """Return a post-processed copy of raw xPts predictions.

    `extend_defcon_future=False` preserves historical backtest behavior:
    DefCon is added only to player/GW rows observed in the DefCon source.
    Live can set it to True to project DefCon rates into upcoming GWs after
    the latest observed appearance.
    """
    out = dict(pred_by_pgw)
    if not out:
        return out

    _apply_early_gw_prior(out, season_id=season_id, train_seasons=train_seasons)
    _apply_position_calibration(out, position_calibration)
    _apply_defcon(
        out,
        season_id=season_id,
        defcon_shrinkage=defcon_shrinkage,
        defcon_per_position_shrinkage=defcon_per_position_shrinkage,
        extend_defcon_future=extend_defcon_future,
    )
    if fwd_calibration:
        _apply_fwd_calibration(out, season_id=season_id, cache_dir=cache_dir)
    return out


def _apply_early_gw_prior(
    pred_by_pgw: dict[tuple[int, int], float],
    *,
    season_id: int,
    train_seasons: list[int],
) -> None:
    prior_train_seasons = [s for s in train_seasons if s != season_id]
    if not prior_train_seasons:
        return
    prior_pts = pit.player_actual_pts_last_n_gws(max(prior_train_seasons), n=5)
    if not prior_pts:
        return
    for (pid, gw), value in list(pred_by_pgw.items()):
        blend_w = EARLY_GW_PRIOR_BLEND.get(gw)
        if blend_w is None or pid not in prior_pts:
            continue
        pred_by_pgw[(pid, gw)] = blend_w * prior_pts[pid] + (1 - blend_w) * value


def _apply_position_calibration(
    pred_by_pgw: dict[tuple[int, int], float],
    position_calibration: dict[str, float] | None,
) -> None:
    if not position_calibration:
        return
    positions_lookup = (
        pit.all_player_positions()
        .to_pandas()
        .set_index("player_id")["position_code"]
        .to_dict()
    )
    for (pid, gw), value in list(pred_by_pgw.items()):
        pos = positions_lookup.get(pid)
        if pos and pos in position_calibration:
            pred_by_pgw[(pid, gw)] = value * position_calibration[pos]


def _apply_defcon(
    pred_by_pgw: dict[tuple[int, int], float],
    *,
    season_id: int,
    defcon_shrinkage: float | None,
    defcon_per_position_shrinkage: dict[str, float] | None,
    extend_defcon_future: bool,
) -> None:
    if defcon_shrinkage is None and defcon_per_position_shrinkage is None:
        return
    # The shrinkage constants were tuned only for the first DefCon season
    # currently present in the historical database. Future seasons should be
    # explicitly retuned before enabling this adjustment in backtests.
    if season_id != 25:
        return
    from fpl_bot.eval.defcon_adjustment import compute_defcon_adjustments

    target_gws = (
        sorted({gw for (_, gw) in pred_by_pgw if gw > 0})
        if extend_defcon_future
        else None
    )
    if defcon_per_position_shrinkage is not None:
        defcon = compute_defcon_adjustments(
            test_season=season_id,
            target_gws=target_gws,
            per_position_shrinkage=defcon_per_position_shrinkage,
        )
        for (pid, gw), value in list(pred_by_pgw.items()):
            adj = defcon.get((pid, gw))
            if adj is not None:
                pred_by_pgw[(pid, gw)] = value + adj
        return

    defcon = compute_defcon_adjustments(test_season=season_id, target_gws=target_gws)
    for (pid, gw), value in list(pred_by_pgw.items()):
        adj = defcon.get((pid, gw))
        if adj is not None:
            pred_by_pgw[(pid, gw)] = value + defcon_shrinkage * adj


def _apply_fwd_calibration(
    pred_by_pgw: dict[tuple[int, int], float],
    *,
    season_id: int,
    cache_dir: Path,
) -> None:
    from fpl_bot.models.fwd_calibration import fit_fwd_calibrator

    calibrator = fit_fwd_calibrator(season_id, cache_dir=cache_dir)
    if calibrator is None:
        return
    positions_lookup = (
        pit.all_player_positions()
        .to_pandas()
        .set_index("player_id")["position_code"]
        .to_dict()
    )
    for (pid, gw), value in list(pred_by_pgw.items()):
        if positions_lookup.get(pid) == "FWD":
            pred_by_pgw[(pid, gw)] = calibrator.transform(value)
