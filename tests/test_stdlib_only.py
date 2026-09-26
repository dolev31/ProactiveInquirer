"""pinq must import nothing outside the standard library and nothing first-party.

This is checked by AST rather than by importing, so it catches a dependency that happens to
be installed in the developer's venv but would be absent in a clean install.
"""

import ast
import pathlib
import sys

import pytest

PINQ = pathlib.Path(__file__).resolve().parents[1] / "src" / "pinq"
FIRST_PARTY = {"pinq_adapters", "pinq_expt", "pi_eval", "pi_run", "pinq_train"}


def _roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module.split(".")[0])
    return out


@pytest.mark.parametrize("path", sorted(PINQ.rglob("*.py")), ids=lambda p: p.name)
def test_pinq_module_imports_only_stdlib(path):
    for root in _roots(path):
        assert root not in FIRST_PARTY, f"{path.name} imports first-party {root}"
        assert root in sys.stdlib_module_names, f"{path.name} imports third-party {root}"
