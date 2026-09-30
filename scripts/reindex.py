"""Convenience wrapper: `python scripts/reindex.py [args]` == `python -m app.cli reindex [args]`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["reindex", *sys.argv[1:]]))
