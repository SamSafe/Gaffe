from fpl_bot.optim.milp import _choose_vice_captain


def test_vice_uses_highest_expected_points_among_remaining_starters():
    predictions = {(1, 7): 8.0, (2, 7): 3.0, (3, 7): 6.0, (4, 7): 9.0}
    assert _choose_vice_captain(frozenset({1, 2, 3}), 1, predictions, 7) == 3


def test_vice_ties_are_deterministic():
    assert _choose_vice_captain(frozenset({1, 2, 3}), 1, {}, 7) == 2
