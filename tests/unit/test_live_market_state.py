import inspect

import pytest

from fpl_bot.live.market_state import apply_current_market, selling_price


def test_live_market_supersedes_old_club_and_price_without_rewriting_history():
    teams = {1: 10}
    prices = {(1, 3): 70, (1, 4): 70}
    apply_current_market(teams, prices, {1: (20, 73), 2: (30, 45)}, [4, 5])
    assert teams == {1: 20, 2: 30}
    assert prices == {(1, 3): 70, (1, 4): 73, (1, 5): 73, (2, 4): 45, (2, 5): 45}


@pytest.mark.parametrize(('current', 'purchase', 'expected'), [(73, 70, 71), (71, 70, 70), (69, 70, 69), (70, 70, 70)])
def test_sale_profit_rounds_down(current, purchase, expected):
    assert selling_price(current, purchase) == expected


def test_live_defaults_maximize_points():
    from fpl_bot.cli.main import live_recommend
    from fpl_bot.live.recommend import generate_recommendation

    for fn in (live_recommend, generate_recommendation):
        assert inspect.signature(fn).parameters['rho'].default == 0.0
