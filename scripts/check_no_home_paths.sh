#!/usr/bin/env bash
# A committed absolute home path is the single most common way a research repo becomes
# unreproducible for anyone but its author. Fail loudly rather than discover it at release.
set -euo pipefail
cd "$(dirname "$0")/.."
if grep -rInE '/Users/[a-z0-9_.-]+/' \
     --include='*.py' --include='*.toml' --include='*.yaml' --include='*.yml' \
     --include='*.md' --include='*.cfg' --include='*.sh' --include='*.tex' \
     --exclude-dir='.venv*' --exclude-dir=.git --exclude-dir=data --exclude-dir=runs \
     . ; then
  echo "ERROR: absolute home paths found in committed files (see above)" >&2
  exit 1
fi
echo "check_no_home_paths: clean"

# The convlog corpus is built from other people's machines, so "no absolute home path" is a
# DATA property there, not just a source-tree one: the scrubber rewrites /Users/<x>/ to ~/
# and this is the independent check that it did. Skipped silently when the suite is absent,
# because most checkouts will never build it.
if [ -d data/corpora/convlog ]; then
  if grep -rInE '/(Users|home)/[a-z0-9_.-]+/' data/corpora/convlog ; then
    echo "ERROR: unscrubbed home paths in data/corpora/convlog (see above)" >&2
    exit 1
  fi
  echo "check_no_home_paths: convlog corpus clean"
fi
