"""pinq — the detachable core of ProactiveInquirer.

STDLIB ONLY. This package must remain installable next to any pinned dependency set
(notably torch's), so it takes no third-party dependency and imports no sibling package.
Enforced by two import-linter contracts and by tests/test_stdlib_only.py.
"""

__version__ = "0.1.0"
