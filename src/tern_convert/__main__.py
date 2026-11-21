"""Entry point for 'python -m tern_convert'."""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())