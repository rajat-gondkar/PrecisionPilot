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
    p.add_argument("--threads", type=int, default=None,
                   help="CPU threads for PyTorch (default: 2 when no CUDA GPU is present)")


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


# ------------------------------------------------------------ experiment stages
def _ws(args, need_model: bool = True):
    from precisionpilot.experiments.workspace import make_workspace

    return make_workspace(_cfg(args), args.device, fake=getattr(args, "fake", False),
                          need_model=need_model)


def _add_device(p):
    p.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    p.add_argument("--fake", action="store_true", help="synthetic data + random weights")


def cmd_run_accuracy(args) -> int:
    from precisionpilot.experiments import accuracy as acc
    from precisionpilot.experiments.pipeline import run_accuracy

    ws = _ws(args)
    if args.only == "reference":
        acc.reference_accuracies(ws)
    elif args.only == "table":
        acc.build_accuracy_table(ws)
    elif args.only == "sensitivity":
        acc.run_sensitivity(ws)
    else:
        run_accuracy(ws)
    return 0


cmd_run_accuracy.add_args = lambda p: (
    _add_device(p),
    p.add_argument("--only", choices=["reference", "table", "sensitivity"], default=None,
                   help="run a single accuracy stage (default: all three)"),
)


def _add_backend(p):
    p.add_argument("--backend", default=None, help="mock | torch | trt (default: config)")
    p.add_argument("--gpu", default=None, help="results folder name (default: from device)")
    p.add_argument("--profile", default="mock-ampere", help="mock backend profile")


def _backend(args, cfg):
    from precisionpilot.experiments.pipeline import make_backend

    name = args.backend or cfg.latency.backend
    extra = {"profile": args.profile} if name == "mock" else {}
    return make_backend(cfg, name, **extra)


def cmd_build_lut(args) -> int:
    from precisionpilot.experiments.workspace import resolve_gpu_slug
    from precisionpilot.latency.lut import build_lut

    ws = _ws(args)
    backend = _backend(args, ws.cfg)
    lut = build_lut(ws.mm, backend, ws.cfg.latency.batch_sizes, ws.image_size)
    path = lut.save(ws.gpu_dir(resolve_gpu_slug(backend.device_name, args.gpu)) / "lut.csv")
    print("saved", path)
    return 0


cmd_build_lut.add_args = lambda p: (_add_device(p), _add_backend(p))


def cmd_run_gpu(args) -> int:
    from precisionpilot.experiments.pipeline import run_gpu

    ws = _ws(args)
    run_gpu(ws, _backend(args, ws.cfg), args.gpu, skip_lut=args.skip_lut,
            skip_verify=args.skip_verify)
    return 0


cmd_run_gpu.add_args = lambda p: (
    _add_device(p), _add_backend(p),
    p.add_argument("--skip-lut", action="store_true", help="reuse an existing lut.csv"),
    p.add_argument("--skip-verify", action="store_true", help="no full-model measurements"),
)


def cmd_report(args) -> int:
    from precisionpilot.analysis.report import build_report

    cfg = _cfg(args)
    figs = build_report(cfg.path("results_dir"), cfg.budget.main_pp)
    print(f"{len(figs)} figures")
    return 0


def cmd_pipeline(args) -> int:
    from precisionpilot.experiments.pipeline import smoke

    if not args.smoke:
        print("Full runs are split by machine; see README 'Running the experiments'.\n"
              "Use --smoke for the end-to-end check on synthetic data.")
        return 1
    smoke(args.overrides, threads=args.threads or 2)
    return 0


cmd_pipeline.add_args = lambda p: p.add_argument("--smoke", action="store_true")


def cmd_export_onnx(args) -> int:
    from precisionpilot.export.onnx_export import count_qdq, export_config
    from precisionpilot.precision import str_to_cfg

    ws = _ws(args)
    path = export_config(ws.mm, str_to_cfg(args.cfg_bits), args.batch_size, ws.image_size,
                         args.out)
    print("saved", path, count_qdq(path))
    return 0


cmd_export_onnx.add_args = lambda p: (
    _add_device(p),
    p.add_argument("--cfg-bits", required=True, help="10-bit string, e.g. 0111111110"),
    p.add_argument("--batch-size", type=int, default=1),
    p.add_argument("--out", default="exports"),
)


COMMANDS: dict[str, tuple] = {
    "env-check": (cmd_env_check, "Print library/GPU versions and compatibility warnings"),
    "make-splits": (cmd_make_splits, "Download CIFAR-10 and create/verify data/splits.json"),
    "train": (cmd_train, "Fine-tune ResNet-18 on CIFAR-10 (run on Kaggle GPU)"),
    "evaluate": (cmd_evaluate, "FP32 and all-FP16 top-1 accuracy baselines"),
    "run-accuracy": (cmd_run_accuracy, "Reference accuracies, 1,024-config table, sensitivity"),
    "build-lut": (cmd_build_lut, "Per-unit latency lookup table on this GPU"),
    "run-gpu": (cmd_run_gpu, "LUT + all searches + verification + final test on this GPU"),
    "report": (cmd_report, "Regenerate all figures and summary tables"),
    "pipeline": (cmd_pipeline, "End-to-end smoke run (--smoke) on synthetic data"),
    "export-onnx": (cmd_export_onnx, "Export one configuration to ONNX with Q/DQ"),
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
    _limit_cpu(args.threads)
    return int(args.func(args) or 0)


def _limit_cpu(threads: int | None) -> None:
    """Keep local (non-CUDA) runs from saturating every CPU core."""
    import torch

    from precisionpilot.utils import limit_threads

    if threads is not None:
        limit_threads(threads)
    elif not torch.cuda.is_available():
        limit_threads(2)


if __name__ == "__main__":
    sys.exit(main())
