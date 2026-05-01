"""Static-import leakage audit (§3.4 / §7.6).

Confirms by AST walk that no module under guarded packages imports raw DB
plumbing — only the sanctioned PIT layer (`fpl_bot.db.pit`).

Runs without a live database; safe in any CI environment.
"""
from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = PROJECT_ROOT / "src"

GUARDED_PACKAGES = [
    "fpl_bot.features",
    "fpl_bot.models",
    "fpl_bot.scenarios",
]

# These imports are forbidden inside guarded packages.
# `fpl_bot.db.pit` is the only sanctioned read path.
FORBIDDEN_IMPORT_ROOTS = {
    "fpl_bot.db.models",
    "fpl_bot.db.session",
    "sqlalchemy",
    "psycopg",
}


def _iter_python_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    yield from (p for p in root.rglob("*.py") if p.is_file())


def _is_empty_module(path: pathlib.Path) -> bool:
    text = path.read_text().strip()
    if not text:
        return True
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return all(isinstance(n, (ast.Expr, ast.Pass)) for n in tree.body)


def _imports_in(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _is_forbidden(import_name: str) -> bool:
    return any(
        import_name == root or import_name.startswith(root + ".")
        for root in FORBIDDEN_IMPORT_ROOTS
    )


@pytest.mark.leakage
@pytest.mark.parametrize("package", GUARDED_PACKAGES)
def test_no_raw_db_imports_in_guarded_package(package: str) -> None:
    """Modules under {features,models,scenarios} must use PIT, not raw DB."""
    pkg_path = SRC / package.replace(".", "/")
    if not pkg_path.exists():
        pytest.skip(f"{package} not yet present (Phase 2+)")

    violations: list[tuple[str, str]] = []
    for py_file in _iter_python_files(pkg_path):
        if _is_empty_module(py_file):
            continue
        for imp in _imports_in(py_file):
            if _is_forbidden(imp):
                violations.append((str(py_file.relative_to(PROJECT_ROOT)), imp))

    assert not violations, (
        f"{package} must read DB only via fpl_bot.db.pit; found forbidden imports: "
        f"{violations}"
    )


@pytest.mark.leakage
def test_pit_module_is_importable_without_db() -> None:
    """The PIT module's public surface must import cleanly even without a live DB."""
    import fpl_bot.db.pit as pit_module

    expected = {
        "player_status_as_of",
        "squad_market_as_of",
        "player_match_history",
        "market_xg_for_fixture",
        "eo_as_of",
        "match_event_history",
        "bps_rules_for_season",
        "penalty_taker_as_of",
    }
    missing = expected - set(dir(pit_module))
    assert not missing, f"PIT API missing functions: {missing}"
