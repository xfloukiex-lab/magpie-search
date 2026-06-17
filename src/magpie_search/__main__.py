"""Entry point: `python -m magpie_search <subcommand>` (or installed: `magpie_search <subcommand>`)."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
