"""Current market inputs for live transfer feasibility."""
from __future__ import annotations


def apply_current_market(
    teams: dict[int, int],
    prices: dict[tuple[int, int], int],
    snapshots: dict[int, tuple[int, int]],
    gameweeks: list[int],
) -> None:
    """Latest bootstrap supersedes played-match clubs and prices in the horizon."""
    for pid, (team, price) in snapshots.items():
        teams[pid] = team
        for gw in gameweeks:
            prices[pid, gw] = price


def selling_price(current: int, purchase: int) -> int:
    """FPL credits half the profit, rounded down to a price increment."""
    return min(current, purchase + max(0, current - purchase) // 2)
