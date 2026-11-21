"""Build the season role snapshots offline from manually recorded match observations."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs"


def render(as_of: date = date.max) -> str:
    matches = defaultdict(set)
    with (CONFIG / "set_piece_observations.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            role = row["role"]
            if role not in {"penalty", "direct_fk", "corner"} or not row["source"].startswith("https://"):
                raise ValueError(f"Invalid observation: {row}")
            if day <= as_of:
                matches[int(row["season"]), row["team"], role, row["player"]].add(day)

    candidates = defaultdict(list)
    for (season, team, role, player), days in matches.items():
        candidates[season, team, role].append((-len(days), -max(days).toordinal(), player))

    seasons = {}
    for (season, team, role), players in sorted(candidates.items()):
        names = [name for _, _, name in sorted(players)]
        team_roles = seasons.setdefault(season, {}).setdefault(team, {})
        team_roles[role] = names[0] if role == "penalty" else names
    return "# Generated from set_piece_observations.csv by scripts/gen_set_piece_takers.py\n" + yaml.safe_dump(
        seasons, allow_unicode=True, sort_keys=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.max)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = render(args.as_of)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
