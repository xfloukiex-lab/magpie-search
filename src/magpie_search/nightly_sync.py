"""nightly_sync — compatibility entry point for `magpie_search.backup`.

Older Windows Task Scheduler / cron entries call `python -m magpie_search.nightly_sync`;
they keep working by routing through this thin shim. New automation should
target `python -m magpie_search.backup` directly.

All configuration is read from env vars / `~/.magpie-search/backup.env` — see
`magpie_search.backup` and the README for keys. Nothing operator-specific lives
in this module anymore.
"""
from __future__ import annotations

from . import backup as _backup


def main(argv: list[str] | None = None) -> int:
    return _backup.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
