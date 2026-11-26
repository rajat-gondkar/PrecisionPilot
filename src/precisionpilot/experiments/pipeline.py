"""Orchestration of the experiment stages.

Stages and where they normally run:

=================  ==========================================  ======================
stage              what                                        where
=================  ==========================================  ======================
run_accuracy       reference accuracies, 1,024-config table,   Kaggle T4 (any GPU)
                   sensitivity scan (Experiment 1)
run_gpu            latency LUT, search suite, verification,    every benchmark GPU
                   final test of chosen configs                (3050 Ti, T4)
report             figures + tables                            anywhere (CPU)
=================  ==========================================  ======================

``smoke`` runs all of them on synthetic data, random weights, 16-px inputs and the mock
latency backend (two mock GPU profiles) — a full end-to-end check in about a minute.
"""

from __future__ import annotations

import pandas as pd

from precisionpilot.experiments import accuracy as acc_exp
from precisionpilot.experiments.search_suite import chosen_configs, run_search_suite
from precisionpilot.experiments.verify import run_backend_accuracy, run_verify
from precisionpilot.experiments.workspace import make_workspace, resolve_gpu_slug
from precisionpilot.latency.backends.base import get_backend
from precisionpilot.latency.lut import LatencyTable, build_lut
from precisionpilot.search.common import TableAccuracy


def run_accuracy(ws, log=print) -> dict:
    ref = acc_exp.reference_accuracies(ws, log)
    acc_exp.build_accuracy_table(ws, log)
    acc_exp.run_sensitivity(ws, log=log)
    return ref


def make_backend(cfg, name: str | None = None, **kw):
    name = name or cfg.latency.backend
    kwargs = dict(warmup=cfg.latency.warmup, reps=cfg.latency.reps,
                  cooldown_s=cfg.latency.cooldown_s)
    if name in {"trt", "tensorrt"}:
        kwargs["cache_dir"] = cfg.path("engine_cache")
    if name == "mock":
        kwargs["cooldown_s"] = 0.0
        kwargs["warmup"] = 0
    kwargs.update(kw)
    return get_backend(name, **kwargs)


def run_gpu(ws, backend, gpu: str | None = None, skip_lut: bool = False,
            skip_verify: bool = False, log=print) -> dict:
    cfg = ws.cfg
    gpu = resolve_gpu_slug(backend.device_name, gpu)
    gdir = ws.gpu_dir(gpu)
    bss = list(cfg.latency.batch_sizes)

    lut_path = gdir / "lut.csv"
    if skip_lut and lut_path.exists():
        lut = LatencyTable.load(lut_path)
    else:
        lut = build_lut(ws.mm, backend, bss, ws.image_size, log)
        lut.save(lut_path)
        log(f"saved {lut_path}")

    ref = acc_exp.load_reference(ws)
    table = TableAccuracy.from_csv(ws.accuracy_dir / "accuracy_table.csv")
    results, curve = run_search_suite(table, ref["fp32_val_subset"], lut, gpu, bss,
                                      list(cfg.budget.sweep_pp), cfg.search.random_seeds,
                                      cfg.search.epsilon, ref.get("sec_per_eval", 0.0),
                                      ws.mm.n_units, log)
    results.to_csv(gdir / "search_results.csv", index=False)
    curve.to_csv(gdir / "random_curve.csv", index=False)
    log(f"saved {gdir / 'search_results.csv'}")

    out = {"gpu": gpu, "results": results}
    if not skip_verify:
        verify = run_verify(ws, backend, lut, results, bss, log)
        verify.to_csv(gdir / "verify.csv", index=False)
        log(f"saved {gdir / 'verify.csv'}")
        out["verify"] = verify

    # held-out test accuracy of everything worth reporting (simulated INT8) ...
    pairs = []
    for bs in bss:
        pairs += chosen_configs(results, bs)
    uniq = list(dict(pairs).items())
    acc_exp.final_test(ws, [c for c, _ in uniq], [lbl for _, lbl in uniq], log)
    # ... and of the real engines, if the backend can run them (TensorRT)
    if not skip_verify:
        eng = run_backend_accuracy(ws, backend, uniq, batch_size=max(bss), log=log)
        if eng is not None:
            eng.to_csv(gdir / "engine_accuracy.csv", index=False)
    return out


def smoke(cfg_overrides: list[str] | None = None, log=print, threads: int = 2) -> list:
    """End-to-end check on tiny synthetic data; under ~30 s on 2 CPU threads."""
    from precisionpilot.analysis.report import build_report
    from precisionpilot.config import load_config
    from precisionpilot.utils import limit_threads

    limit_threads(threads)
    overrides = ["paths.results_dir=results/smoke", "data.image_size=16",
                 "data.eval_batch_size=64", "latency.reps=10", "search.random_seeds=3",
                 "latency.batch_sizes=[1,32]", "quant.calib_batches=1",
                 *(cfg_overrides or [])]
    cfg = load_config(overrides=overrides)
    ws = make_workspace(cfg, "cpu", fake=True, fake_size="tiny")
    run_accuracy(ws, log)
    for profile in ("mock-ampere", "mock-turing"):
        run_gpu(ws, make_backend(cfg, "mock", profile=profile), log=log)
    figs = build_report(ws.results, cfg.budget.main_pp, log)
    log(f"smoke pipeline OK: {len(figs)} figures in {ws.results / 'figures'}")
    return figs


__all__ = ["run_accuracy", "run_gpu", "smoke", "make_workspace", "make_backend", "pd"]
