"""Regenerate every figure and summary table from the result CSVs (``precisionpilot report``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from precisionpilot.analysis import plots
from precisionpilot.analysis.tables import summary_table, to_markdown
from precisionpilot.latency.lut import LatencyTable
from precisionpilot.utils import ensure_dir, load_json


def _gpu_dirs(results: Path) -> list[Path]:
    return sorted(p.parent for p in results.glob("*/search_results.csv"))


def build_report(results: Path, main_budget: float, log=print,
                 extra_budgets: Sequence[float] = ()) -> list[Path]:
    """Figures for the main budget plus every supplementary budget (``extra_budgets``)."""
    results = Path(results)
    figs = ensure_dir(results / "figures")
    tabs = ensure_dir(results / "tables")
    out: list[Path] = []
    acc_dir = results / "accuracy"

    sens_path = acc_dir / "sensitivity.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        out.append(plots.plot_sensitivity(sens, figs / "fig1_sensitivity.png"))
        (tabs / "sensitivity.md").write_text(to_markdown(
            sens[["unit", "name", "acc", "drop_vs_fp16", "drop_vs_ref"]]))

    ref = load_json(acc_dir / "reference.json") if (acc_dir / "reference.json").exists() else {}
    acc_table = None
    if (acc_dir / "accuracy_table.csv").exists():
        acc_table = pd.read_csv(acc_dir / "accuracy_table.csv", dtype={"config": str})

    all_results, all_verify = [], []
    for gdir in _gpu_dirs(results):
        gpu = gdir.name
        res = pd.read_csv(gdir / "search_results.csv", dtype={"config": str})
        res["gpu"] = gpu
        all_results.append(res)
        curve = pd.read_csv(gdir / "random_curve.csv")
        lut = LatencyTable.load(gdir / "lut.csv")
        present = set(res.budget_pp.round(6))
        budgets = [b for b in dict.fromkeys([main_budget, *extra_budgets])
                   if round(b, 6) in present]
        for bs in sorted(res.batch_size.unique()):
            for b in budgets:
                out.append(plots.plot_cost_vs_quality(
                    res, curve, figs / f"fig2_cost_vs_quality_{gpu}_bs{bs}_b{b:g}.png", bs, b))
                if acc_table is not None and "fp32_val_subset" in ref:
                    out.append(plots.plot_scatter(
                        acc_table, lut.latency_fn(bs), ref["fp32_val_subset"], res,
                        figs / f"fig3_acc_vs_latency_{gpu}_bs{bs}_b{b:g}.png", bs, b))
        out.append(plots.plot_config_heatmap(res, figs / f"fig4_heatmap_{gpu}.png",
                                             title=f"Greedy selection per batch / budget — {gpu}"))
        out.append(plots.plot_config_heatmap(res, figs / f"fig4_heatmap_optimum_{gpu}.png",
                                             method="exhaustive",
                                             title=f"Exhaustive optimum per batch / budget — {gpu}"))
        if (gdir / "verify.csv").exists():
            v = pd.read_csv(gdir / "verify.csv", dtype={"config": str})
            v["gpu"] = gpu
            all_verify.append(v)

    if all_results:
        res = pd.concat(all_results, ignore_index=True)
        out.append(plots.plot_speedups(res, figs / "fig6_speedups.png", main_budget))
        for b in extra_budgets:
            if round(b, 6) in set(res.budget_pp.round(6)) and b != main_budget:
                out.append(plots.plot_speedups(res, figs / f"fig6_speedups_b{b:g}.png", b))
        if res.gpu.nunique() > 1:
            main = res[res.budget_pp == main_budget]
            out.append(plots.plot_config_heatmap(main, figs / "fig5_heatmap_gpus.png",
                                                 title=f"Greedy selection across GPUs "
                                                       f"(budget {main_budget:g} pp)"))
        summ = summary_table(res)
        summ.to_csv(tabs / "summary.csv", index=False)
        (tabs / "summary.md").write_text(to_markdown(summ))
    if all_verify:
        v = pd.concat(all_verify, ignore_index=True)
        out.append(plots.plot_lut_error(v, figs / "fig7_lut_error.png"))
        cols = [c for c in ["gpu", "batch_size", "label", "config", "estimated_ms",
                            "measured_median_ms", "measured_p95_ms", "throughput_ips",
                            "error_pct", "engine_mb", "weights_mb"] if c in v.columns]
        (tabs / "verify.md").write_text(to_markdown(v[cols]))
    if (acc_dir / "final_test.csv").exists():
        (tabs / "final_test.md").write_text(to_markdown(
            pd.read_csv(acc_dir / "final_test.csv", dtype={"config": str})))
    for p in out:
        log(f"figure -> {p}")
    return out
