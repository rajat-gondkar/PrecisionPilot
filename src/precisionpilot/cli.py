"""Command-line interface: ``python -m precisionpilot <command> [options]``.

Works identically on macOS, Linux and Windows (PowerShell / cmd).
Heavy modules are imported lazily inside each command.
"""

from __future__ import annotations

import argparse
import sys


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--set", dest="overrides", nargs="*", default=[],
                   metavar="KEY=VALUE", help="Override config values, e.g. train.epochs=2")


def _cfg(args):
    from precisionpilot.config import load_config

    return load_config(args.config, args.overrides)


# ---------------------------------------------------------------- commands
def cmd_env_check(args) -> int:
    from precisionpilot import envcheck

    envcheck.run(verbose=True)
    return 0


COMMANDS: dict[str, tuple] = {
    "env-check": (cmd_env_check, "Print library/GPU versions and compatibility warnings"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precisionpilot", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (fn, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        _add_common(p)
        extra = getattr(fn, "add_args", None)
        if extra:
            extra(p)
        p.set_defaults(func=fn)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
