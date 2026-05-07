"""BacktestState — tracks squad / bank / FT / chips through the rolling backtest.

State transitions are explicit (not implicit in the MILP). Each per-GW solve
takes a state, returns decisions, the harness then applies actual fixture
outcomes to compute the next state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Chip slot identifiers
CHIPS_FIRST_HALF = ("WC1", "FH1")
CHIPS_SECOND_HALF = ("WC2", "FH2")
CHIPS_ANYTIME = ("BB", "TC")
ALL_CHIP_SLOTS = (*CHIPS_FIRST_HALF, *CHIPS_SECOND_HALF, *CHIPS_ANYTIME)

# Per-FPL rules: WC1/FH1 in GW 1-19, WC2/FH2 in GW 20-38
SECOND_HALF_FIRST_GW = 20

INITIAL_BUDGET_TENTHS = 1000  # £100.0m


@dataclass(frozen=True)
class BacktestState:
    """Snapshot of the FPL team's state entering a gameweek."""

    season_id: int
    gameweek: int  # the GW we're about to solve for
    squad: frozenset[int] = frozenset()  # 15 stable player_ids
    bank: int = INITIAL_BUDGET_TENTHS  # tenths of millions
    free_transfers: int = 1
    chips_used: frozenset[str] = frozenset()
    # Per-player cost basis (the price we paid). Used for sell-price computation
    # on transfers out. Empty on cold start.
    cost_basis: dict[int, int] = field(default_factory=dict)

    @classmethod
    def cold_start(cls, season_id: int) -> BacktestState:
        return cls(
            season_id=season_id,
            gameweek=1,
            squad=frozenset(),
            bank=INITIAL_BUDGET_TENTHS,
            free_transfers=1,
            chips_used=frozenset(),
            cost_basis={},
        )

    def chips_available_for_gw(self, gw: int) -> tuple[str, ...]:
        """Which chip slots are eligible to be played in GW `gw`?"""
        out: list[str] = []
        first_half = gw < SECOND_HALF_FIRST_GW
        for slot in ALL_CHIP_SLOTS:
            if slot in self.chips_used:
                continue
            if slot in CHIPS_FIRST_HALF and not first_half:
                continue
            if slot in CHIPS_SECOND_HALF and first_half:
                continue
            out.append(slot)
        return tuple(out)


@dataclass(frozen=True)
class GwDecisions:
    """One gameweek's decisions emerging from the MILP solve."""

    gameweek: int
    squad: frozenset[int]
    starting_xi: frozenset[int]
    captain: int | None
    vice: int | None
    transferred_in: frozenset[int]
    transferred_out: frozenset[int]
    chip_played: str | None  # one of WC1/WC2/FH1/FH2/BB/TC, or None
    hits: int  # transfers above free count, ×4 cost
    objective_value: float


def apply_gw_outcomes(
    state: BacktestState,
    decisions: GwDecisions,
    actual_prices: dict[int, dict[str, int]],
) -> BacktestState:
    """Transition: apply transfers + chip + bank/FT evolution.

    Currently uses the same buy/sell price for transfers (static-price v1
    assumption per Phase 3 design §1). Sell price = cost_basis price for
    transferred-out players when we have a basis; else current buy price.
    """
    transfers_out = decisions.transferred_out
    transfers_in = decisions.transferred_in

    # Money in. v1.0 static-price assumption: both buys and sells use the
    # CURRENT price passed in actual_prices. This matches what the MILP
    # solved against (it sees buy=sell=current_price). Cost basis is still
    # tracked for forward compatibility with Phase 3.5 where dynamic prices
    # will reintroduce sell-vs-buy spread.
    bank_after = state.bank
    new_cost_basis = dict(state.cost_basis)
    for p in transfers_out:
        # Static v1: sell at current price (== what MILP saw as sell_price)
        sell_price = actual_prices.get(p, {}).get("sell")
        if sell_price is None:
            sell_price = new_cost_basis.get(p, 0)
        bank_after += sell_price
        new_cost_basis.pop(p, None)
    for p in transfers_in:
        buy_price = actual_prices.get(p, {}).get("buy", 0)
        bank_after -= buy_price
        new_cost_basis[p] = buy_price

    # Chips
    chips_after = set(state.chips_used)
    if decisions.chip_played:
        chips_after.add(decisions.chip_played)

    # FT evolution: WC and FH refund the FT count to 1 the week they're played
    # (and don't consume FT this week — the MILP enforces hits=0 under WC). For
    # next-GW carry-over: ft_next = min(5, ft_now + 1 - transfers_used_this_gw),
    # with WC/FH waiving consumption.
    transfers_used = len(transfers_in)
    chip_waives_ft = decisions.chip_played in ("WC1", "WC2", "FH1", "FH2")
    consumed = 0 if chip_waives_ft else min(transfers_used, state.free_transfers)
    ft_next = min(5, state.free_transfers + 1 - consumed)

    # FH reverts squad next GW; v1 marks chip used but leaves squad as-is —
    # next-GW MILP must force the squad back to pre-FH. v1 simplification:
    # we model FH as a one-week roster swap inside the MILP horizon only;
    # if the rolling solve plays FH at horizon-tip, the next solve uses the
    # state's squad which we keep frozen at pre-FH for that case.
    # (Phase 5 chip-DP handles cross-horizon FH coordination.)

    return BacktestState(
        season_id=state.season_id,
        gameweek=state.gameweek + 1,
        squad=decisions.squad,
        bank=bank_after,
        free_transfers=ft_next,
        chips_used=frozenset(chips_after),
        cost_basis=new_cost_basis,
    )
