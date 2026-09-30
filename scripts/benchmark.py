"""Convenience wrapper: `python scripts/benchmark.py [args]` == `python -m app.cli evaluate [args]`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
