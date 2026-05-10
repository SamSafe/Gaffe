"""Phase 4 SAA prep: dump per-iteration raw FPL points for all walk-forward
folds (21-24). Caches to `data/cache/xpts_raw_samples/`.
"""
from __future__ import annotations

import subprocess

FOLDS = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]


def main() -> None:
    for fold in FOLDS:
        train_str = ", ".join(str(s) for s in fold["train"])
        print(f"\n=== fold test={fold['test']} (train=[{train_str}]) ===", flush=True)
        # Subprocess per fold to dodge the HiGHS state leak / training memory growth.
        code = f"""
from fpl_bot.eval.xpts_eval import run_fold_with_raw_samples
path = run_fold_with_raw_samples(test_season={fold['test']}, train_seasons=[{train_str}], n_iterations=200, seed=42)
print(f'wrote: {{path}}')
"""
        result = subprocess.run(
            ["uv", "run", "python", "-u", "-c", code],
            capture_output=False,
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"FOLD {fold['test']} FAILED: rc={result.returncode}", flush=True)


if __name__ == "__main__":
    main()
