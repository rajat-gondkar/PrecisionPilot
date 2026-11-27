import pandas as pd
import pytest
import torch

from precisionpilot.analysis.tables import summary_table, to_markdown
from precisionpilot.cli import main
from precisionpilot.config import load_config
from precisionpilot.experiments.accuracy import exhaustive_accuracy_table
from precisionpilot.experiments.search_suite import chosen_configs, run_search_suite
from precisionpilot.experiments.workspace import make_workspace
from precisionpilot.latency.backends.mock import MockBackend
from precisionpilot.latency.lut import build_lut
from precisionpilot.precision import all_fp16, all_int8, cfg_to_str, int_to_cfg
from precisionpilot.search.common import ModelAccuracy, TableAccuracy

TINY = ["data.image_size=16", "data.eval_batch_size=8", "quant.calib_batches=1"]


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    out = tmp_path_factory.mktemp("res")
    cfg = load_config(overrides=[*TINY, f"paths.results_dir={out}"])
    return make_workspace(cfg, "cpu", fake=True, fake_size="tiny")


@pytest.fixture(scope="module")
def table(ws):
    x, y = ws.data.get("val_subset")
    return exhaustive_accuracy_table(ws.mm, x, y, "cpu", batch_size=8, image_size=16,
                                     log=lambda *_: None)


def test_prefix_tree_table_matches_direct_evaluation(ws, table):
    assert len(table) == 1024 and table["config"].is_unique
    x, y = ws.data.get("val_subset")
    direct = ModelAccuracy(ws.mm, x, y, "cpu", batch_size=8, image_size=16)
    lookup = TableAccuracy(dict(zip(table["config"], table["val_acc"])))
    for i in (0, 1, 2, 341, 512, 777, 1023):
        cfg = int_to_cfg(i)
        assert lookup(cfg) == pytest.approx(direct(cfg), abs=1e-9), cfg_to_str(cfg)


def test_search_suite_and_chosen_configs(ws, table):
    lut = build_lut(ws.mm, MockBackend(reps=5), [1, 8], 16, log=lambda *_: None)
    res, curve = run_search_suite(TableAccuracy(dict(zip(table.config, table.val_acc))),
                                  float(table.val_acc.max()), lut, "mock", [1, 8], [0.2, 1.0],
                                  random_seeds=2, sec_per_eval=0.5, log=lambda *_: None,
                                  supplementary=[0.2])
    kinds = res.groupby("budget_pp").budget_kind.unique().map(list).to_dict()
    assert kinds == {0.2: ["supplementary"], 1.0: ["pre-registered"]}
    assert set(curve.budget_kind) == {"supplementary", "pre-registered"}
    methods = set(res.method)
    assert {"all-fp16", "all-int8", "greedy-skip", "greedy-stop", "greedy-skip+ls",
            "greedy-dyn", "random", "exhaustive"} <= methods
    ex = res[res.method == "exhaustive"]
    assert (ex.gap_to_opt_pct.abs() < 1e-9).all() and (ex.n_evals == 1024).all()
    feas = res[res.feasible & res.method.isin(["greedy-skip", "greedy-skip+ls", "random"])]
    assert (feas.gap_to_opt_pct >= -1e-9).all()  # nothing beats the optimum
    assert (res.est_search_s == res.n_evals * 0.5).all()
    assert set(curve.n_evals) >= {5, 1024}
    assert (curve[curve.n_evals == 1024].gap_mean_pct.abs() < 1e-9).all()

    pairs = chosen_configs(res, 8)
    configs = [c for c, _ in pairs]
    assert len(configs) == len(set(configs))
    assert cfg_to_str(all_fp16()) in configs and cfg_to_str(all_int8()) in configs

    summ = summary_table(res)
    assert len(summ) == 2 * 2 * 8  # bs x budget x methods (random aggregated)
    assert set(summ.budget_kind) == {"supplementary", "pre-registered"}
    assert (summ[summ.method == "random"].config == "(varies)").all()
    md = to_markdown(summ.head(3))
    assert md.count("\n") == 5 and "|" in md


def test_smoke_pipeline_end_to_end(smoke_results):
    tmp_path, figs = smoke_results
    assert len(figs) >= 10 and all(p.exists() for p in figs)
    for f in ["accuracy/reference.json", "accuracy/accuracy_table.csv",
              "accuracy/sensitivity.csv", "accuracy/final_test.csv",
              "mockampere/lut.csv", "mockampere/search_results.csv",
              "mockturing/verify.csv", "tables/summary.md", "figures/fig5_heatmap_gpus.png",
              "mockampere/ctx_train.csv", "mockampere/ctx_model.csv",
              "tables/memory.md", "tables/latency_model_error.md", "tables/latency_model_comparison.md",
              "figures/fig8_memory_mockampere.png", "figures/fig9_latency_models_mockampere.png"]:
        assert (tmp_path / f).exists(), f
    res = pd.read_csv(tmp_path / "mockampere" / "search_results.csv", dtype={"config": str})
    assert res.config.str.len().eq(10).all()
    assert set(res.latency_model) == {"additive", "context"}
    assert "greedy-dyn" in set(res.method) and res.model_mb.notna().all()
    mem = pd.read_csv(tmp_path / "tables" / "memory.csv")
    assert (mem[mem.name == "all-INT8"].model_mb_saving_pct.round() == 50).all()
    assert torch.get_num_threads() == 2


def test_cli_pipeline_requires_smoke_flag():
    assert main(["pipeline"]) == 1


def test_report_tolerates_results_from_before_context_model_and_sizes(smoke_results, tmp_path):
    """Result files produced by older versions (no latency_model / model_mb / ctx columns)
    must still produce a report instead of crashing."""
    import shutil

    from precisionpilot.analysis.report import build_report

    src, _ = smoke_results
    old = tmp_path / "old"
    shutil.copytree(src, old, ignore=shutil.ignore_patterns("figures", "tables"))
    for gdir in old.glob("mock*"):
        res = pd.read_csv(gdir / "search_results.csv", dtype={"config": str})
        res = res[res.latency_model == "additive"].drop(columns=["latency_model", "model_mb",
                                                                 "size_saving_pct"])
        res.to_csv(gdir / "search_results.csv", index=False)
        curve = pd.read_csv(gdir / "random_curve.csv")
        curve[curve.latency_model == "additive"].drop(columns="latency_model").to_csv(
            gdir / "random_curve.csv", index=False)
        ver = pd.read_csv(gdir / "verify.csv", dtype={"config": str})
        ver.drop(columns=[c for c in ver.columns if c.startswith("ctx_")]).to_csv(
            gdir / "verify.csv", index=False)
        (gdir / "ctx_model.csv").unlink()
        lut = pd.read_csv(gdir / "lut.csv")
        lut.drop(columns=["params"]).to_csv(gdir / "lut.csv", index=False)
    figs = build_report(old, 1.0, log=lambda *_: None, extra_budgets=[0.2])
    assert len(figs) >= 8
