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


def cmd_make_splits(args) -> int:
    from precisionpilot.data import build_data

    cfg = _cfg(args)
    data = build_data(cfg)
    print("splits:", data.splits.sizes(), "->", cfg.resolve(cfg.data.splits_file))
    return 0


def cmd_train(args) -> int:
    from precisionpilot.data import build_data
    from precisionpilot.train import train
    from precisionpilot.utils import get_device

    cfg = _cfg(args)
    device = get_device(args.device)
    data = build_data(cfg, fake=args.fake)
    out = args.out or cfg.path("checkpoint")
    print(f"training on {device}; splits {data.splits.sizes()}")
    train(cfg, data, device, out, pretrained=not args.no_pretrained,
          max_steps_per_epoch=args.max_steps)
    return 0


cmd_train.add_args = lambda p: (
    p.add_argument("--device", default="auto"),
    p.add_argument("--out", default=None, help="checkpoint path (default: paths.checkpoint)"),
    p.add_argument("--fake", action="store_true", help="synthetic data (smoke test)"),
    p.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke)"),
    p.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch"),
)


def cmd_evaluate(args) -> int:
    from precisionpilot.data import build_data
    from precisionpilot.evaluate import baseline_accuracies
    from precisionpilot.model import load_model
    from precisionpilot.utils import get_device, gpu_slug, save_json

    cfg = _cfg(args)
    device = get_device(args.device)
    data = build_data(cfg, fake=args.fake)
    model, meta = load_model(cfg, device, allow_random=args.fake)
    res = baseline_accuracies(model, data, device, splits=tuple(args.splits),
                              batch_size=cfg.data.eval_batch_size,
                              image_size=cfg.data.image_size)
    res["device"] = str(device)
    for k, v in res.items():
        print(f"{k:>14}: {v:.2f}" if isinstance(v, float) else f"{k:>14}: {v}")
    if not args.fake:
        out = cfg.path("results_dir") / "baselines" / f"baselines_{gpu_slug()}.json"
        save_json(res, out)
        print("saved ->", out)
    return 0


cmd_evaluate.add_args = lambda p: (
    p.add_argument("--device", default="auto"),
    p.add_argument("--splits", nargs="+", default=["val", "test"]),
    p.add_argument("--fake", action="store_true", help="synthetic data + random weights"),
)


COMMANDS: dict[str, tuple] = {
    "env-check": (cmd_env_check, "Print library/GPU versions and compatibility warnings"),
    "make-splits": (cmd_make_splits, "Download CIFAR-10 and create/verify data/splits.json"),
    "train": (cmd_train, "Fine-tune ResNet-18 on CIFAR-10 (run on Kaggle GPU)"),
    "evaluate": (cmd_evaluate, "FP32 and all-FP16 top-1 accuracy baselines"),
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
