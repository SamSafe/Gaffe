"""Subprocess worker: read LP file, solve with highspy, write solution JSON.

Spawned per MILP solve by `optim.milp.solve_milp` to fully isolate HiGHS
C++ state from the parent process. HiGHS (both via `appsi_highs` and via
`highspy` direct) accumulates internal state across solves in one Python
process and eventually causes heap corruption (`malloc()` / `free()`
errors). A fresh subprocess per solve sidesteps the issue entirely.

Invocation:
    python -m fpl_bot.optim._highspy_worker <lp_file> <out_json>
        --time-limit 120 --mip-rel-gap 0.01

Output JSON schema:
    {
      "status": "ok" | "no_feasible_solution",
      "termination": "optimal" | "feasible" | "<HighsModelStatus name>",
      "objective": float | null,
      "col_values": { "col_name": float, ... }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lp_file", type=Path)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.01)
    args = parser.parse_args()

    import highspy

    h = highspy.Highs()
    h.silent()
    read_status = h.readModel(str(args.lp_file))
    if read_status != highspy.HighsStatus.kOk:
        out = {
            "status": "no_feasible_solution",
            "termination": f"read_failed_{read_status}",
            "objective": None,
            "col_values": {},
        }
        args.out_json.write_text(json.dumps(out))
        return 0

    h.setOptionValue("time_limit", float(args.time_limit))
    h.setOptionValue("mip_rel_gap", float(args.mip_rel_gap))
    h.run()

    model_status = h.getModelStatus()
    info_h = h.getInfo()
    sol = h.getSolution()
    obj_val = info_h.objective_function_value

    ok_feasible = model_status in (
        highspy.HighsModelStatus.kOptimal,
        highspy.HighsModelStatus.kTimeLimit,
        highspy.HighsModelStatus.kInterrupt,
        highspy.HighsModelStatus.kIterationLimit,
        highspy.HighsModelStatus.kSolutionLimit,
    )
    has_incumbent = (
        ok_feasible and obj_val is not None and len(sol.col_value) > 0
    )

    if not has_incumbent:
        out = {
            "status": "no_feasible_solution",
            "termination": str(model_status).replace(
                "HighsModelStatus.", ""
            ),
            "objective": None,
            "col_values": {},
        }
        args.out_json.write_text(json.dumps(out))
        return 0

    # Collect col_name → value
    n_cols = h.getNumCol()
    col_values: dict[str, float] = {}
    for i in range(n_cols):
        _, name = h.getColName(i)
        col_values[name] = float(sol.col_value[i])

    is_optimal = model_status == highspy.HighsModelStatus.kOptimal
    out = {
        "status": "ok",
        "termination": "optimal" if is_optimal else "feasible",
        "objective": float(obj_val),
        "col_values": col_values,
    }
    args.out_json.write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
