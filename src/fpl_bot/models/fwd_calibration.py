"""Phase 7 model #1b — post-hoc FWD xPts calibration.

The cross-fold diagnostic (Phase 7) showed the joint xPts model
systematically under-predicts elite FWDs: top-decile bias −0.57 to
−0.95. The `finishing_skill_last_10` feature didn't fix it (LightGBM
under-weights it vs xG). The structural cause is that the Poisson-rate-
on-xG model predicts ≈xG, but clinical finishers beat xG.

A monotonic (isotonic) recalibration directly corrects the mapping:
fit `actual = f(predicted)` on out-of-fold FWD (pred, actual) pairs,
where f is non-decreasing. The fit naturally bends the curve UP at the
high end where actuals exceed predictions — exactly the elite-FWD
region — while leaving the well-calibrated low/mid range alone.

PIT-correctness: the calibrator for a test season is fit ONLY on other
folds' held-out predictions (never the test season itself). For 25/26
that means folds 21-24's (pred, actual) FWD pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from fpl_bot.db import pit

# Default fold caches used to source out-of-fold FWD pairs.
_FOLD_CACHES = {
    21: "season_21_train_19_20.parquet",
    22: "season_22_train_19_20_21.parquet",
    23: "season_23_train_19_20_21_22.parquet",
    24: "season_24_train_19_20_21_22_23.parquet",
    25: "season_25_train_19_20_21_22_23_24.parquet",
}


@dataclass
class FwdCalibrator:
    """Wraps an isotonic mapping for FWD predicted→calibrated xPts."""

    iso: IsotonicRegression
    n_pairs: int

    def transform(self, e_xpts: float) -> float:
        if e_xpts is None:
            return e_xpts
        return float(self.iso.predict([e_xpts])[0])

    def transform_array(self, arr: np.ndarray) -> np.ndarray:
        return self.iso.predict(arr)


def _fwd_pairs_from_cache(
    cache_path: Path, season_id: int, fwd_player_ids: set[int]
) -> list[tuple[float, float]]:
    """Aggregate a fold cache to per-(player, gw) FWD (pred, actual) pairs."""
    if not cache_path.exists():
        return []
    df = pl.read_parquet(cache_path)
    if df.is_empty() or "total_points" not in df.columns:
        return []
    fid_to_gw = pit.fixture_gameweeks_for_season(season_id)
    df = df.with_columns(
        pl.col("fixture_id").replace_strict(fid_to_gw, default=0).alias("gameweek")
    ).filter(pl.col("gameweek") > 0)
    df = df.filter(pl.col("player_id").is_in(list(fwd_player_ids)))
    if df.is_empty():
        return []
    agg = df.group_by(["player_id", "gameweek"]).agg(
        pl.col("e_xpts").sum().alias("pred"),
        pl.col("total_points").sum().alias("actual"),
    )
    return [
        (float(r["pred"]), float(r["actual"]))
        for r in agg.iter_rows(named=True)
    ]


def fit_fwd_calibrator(
    test_season: int,
    cache_dir: Path = Path("data/cache/xpts_predictions"),
) -> FwdCalibrator | None:
    """Fit an isotonic FWD calibrator on all folds EXCEPT `test_season`.

    Returns None if there aren't enough out-of-fold FWD pairs (the caller
    then skips calibration).
    """
    positions = (
        pit.all_player_positions()
        .to_pandas()
        .set_index("player_id")["position_code"]
        .to_dict()
    )
    fwd_ids = {pid for pid, pos in positions.items() if pos == "FWD"}

    pairs: list[tuple[float, float]] = []
    for season_id, fname in _FOLD_CACHES.items():
        if season_id == test_season:
            continue
        pairs.extend(
            _fwd_pairs_from_cache(cache_dir / fname, season_id, fwd_ids)
        )
    if len(pairs) < 200:
        return None

    preds = np.array([p for p, _ in pairs], dtype=np.float64)
    actuals = np.array([a for _, a in pairs], dtype=np.float64)
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(preds, actuals)
    return FwdCalibrator(iso=iso, n_pairs=len(pairs))
