"""Demo web app: FastAPI backend serving precomputed results + live predictions.

Run:  python -m precisionpilot demo            (http://127.0.0.1:8000)

* ``/api/meta``      GPUs, batch sizes, units, reference accuracies, sensitivity
* ``/api/select``    greedy selection for (gpu, batch size, any budget) — computed live from the
                     exhaustive accuracy table + that GPU's latency table, with the optimum and
                     baselines for comparison
* ``/api/samples``   CIFAR-10 test images
* ``/api/predict``   top-k predictions of FP32 vs a mixed-precision configuration
* ``/api/upload``    same, for an uploaded image
"""

import base64
import io
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from precisionpilot.data import CIFAR10_CLASSES
from precisionpilot.latency.context import ContextLatencyTable
from precisionpilot.latency.lut import LatencyTable
from precisionpilot.model import UNIT_NAMES
from precisionpilot.precision import FP16, INT8, all_fp16, all_int8, str_to_cfg
from precisionpilot.search.common import Evaluator, TableAccuracy
from precisionpilot.search.exhaustive import exhaustive_search
from precisionpilot.search.greedy import greedy_dynamic, greedy_search
from precisionpilot.utils import load_json

STATIC = Path(__file__).parent / "static"


class DemoState:
    """Loads result files once; answers selection queries; lazily loads the model."""

    def __init__(self, cfg, results_dir: Path, fake: bool = False):
        self.cfg = cfg
        self.results = Path(results_dir)
        self.fake = fake
        acc = self.results / "accuracy"
        self.reference = load_json(acc / "reference.json") if (acc / "reference.json").exists() \
            else {}
        self.table = None
        if (acc / "accuracy_table.csv").exists():
            self.table = TableAccuracy.from_csv(acc / "accuracy_table.csv")
        self.sensitivity = pd.read_csv(acc / "sensitivity.csv") \
            if (acc / "sensitivity.csv").exists() else None
        self.final_test = pd.read_csv(acc / "final_test.csv", dtype={"config": str}) \
            if (acc / "final_test.csv").exists() else None
        self.luts: dict[str, LatencyTable] = {}
        self.ctx: dict[str, ContextLatencyTable] = {}
        self.verify: dict[str, pd.DataFrame] = {}
        for lut_path in sorted(self.results.glob("*/lut.csv")):
            gpu = lut_path.parent.name
            self.luts[gpu] = LatencyTable.load(lut_path)
            ctx_path = lut_path.parent / "ctx_model.csv"
            if ctx_path.exists():
                self.ctx[gpu] = ContextLatencyTable.load(ctx_path)
            v = lut_path.parent / "verify.csv"
            if v.exists():
                self.verify[gpu] = pd.read_csv(v, dtype={"config": str})
        self._model_lock = threading.Lock()
        self._mm = None
        self._data = None

    # ---------------------------------------------------------------- metadata
    @property
    def ready(self) -> bool:
        return self.table is not None and bool(self.luts) and "fp32_val_subset" in self.reference

    def meta(self) -> dict:
        sens = []
        if self.sensitivity is not None:
            sens = [{"unit": int(r.unit), "name": r.name, "drop": float(r.drop_vs_fp16)}
                    for r in self.sensitivity.itertuples()]
        return {
            "ready": self.ready,
            "results_dir": str(self.results),
            "gpus": [{"id": g, "label": _gpu_label(g, lut.gpu), "batch_sizes": lut.batch_sizes,
                      "backend": lut.backend, "has_context_model": g in self.ctx,
                      "has_model_size": "params" in lut.df.columns}
                     for g, lut in self.luts.items()],
            "units": list(UNIT_NAMES),
            "budgets": list(self.cfg.budget.sweep_pp),
            "main_budget": self.cfg.budget.main_pp,
            "reference": {k: v for k, v in self.reference.items()
                          if isinstance(v, (int, float))},
            "sensitivity": sens,
            "classes": list(CIFAR10_CLASSES),
        }

    # ---------------------------------------------------------------- selection
    def select(self, gpu: str, batch_size: int, budget: float,
               latency_model: str | None = None) -> dict:
        """``latency_model``: "context" (default when fitted) or "additive"."""
        if latency_model is None:
            latency_model = "context" if gpu in self.ctx else "additive"
        if latency_model == "context" and gpu not in self.ctx:
            raise KeyError(f"no context-aware latency model for {gpu}")
        return _select_cached(self, gpu, int(batch_size), round(float(budget), 2), latency_model)

    def _measured(self, gpu: str, batch_size: int, config: str) -> dict | None:
        v = self.verify.get(gpu)
        if v is None:
            return None
        row = v[(v.batch_size == batch_size) & (v.config == config)]
        if not len(row):
            return None
        r = row.iloc[0]
        out = {"median_ms": float(r.measured_median_ms), "p95_ms": float(r.measured_p95_ms),
               "throughput_ips": float(r.throughput_ips)}
        for col in ("engine_mb", "activation_memory_mb", "gpu_footprint_mb"):
            if col in r.index and pd.notna(r[col]):
                out[col] = float(r[col])
        return out

    def _test_acc(self, config: str) -> float | None:
        if self.final_test is None:
            return None
        row = self.final_test[self.final_test.config == config]
        return float(row.test_acc.iloc[0]) if len(row) else None

    def _describe(self, name, res, gpu, bs, fp16_lat, opt_lat) -> dict:
        lut = self.luts[gpu]
        size = lut.model_mb(res.cfg) if "params" in lut.df.columns else None
        fp16_size = lut.model_mb(all_fp16()) if size is not None else None
        return {
            "model_mb": size,
            "size_saving_pct": (100.0 * (1 - size / fp16_size) if size is not None else None),
            "method": name, "config": res.config, "val_acc": res.val_acc,
            "drop_pp": res.drop_pp, "feasible": res.feasible, "n_evals": res.n_evals,
            "est_latency_ms": res.latency, "speedup": fp16_lat / res.latency,
            "gap_pct": 100.0 * (res.latency - opt_lat) / opt_lat,
            "measured": self._measured(gpu, bs, res.config),
            "test_acc": self._test_acc(res.config),
        }

    # ---------------------------------------------------------------- predictions
    def _ensure_model(self):
        with self._model_lock:
            if self._mm is None:
                from precisionpilot.data import build_data
                from precisionpilot.model import load_model
                from precisionpilot.precision import build_mixed_model
                from precisionpilot.utils import get_device

                device = get_device()
                self._data = build_data(self.cfg, fake=self.fake, fake_size="tiny")
                model, _ = load_model(self.cfg, "cpu", allow_random=self.fake)
                calib_x, _ = self._data.get("calib")
                self._mm = build_mixed_model(model, self.cfg, calib_x, device, max_batches=2)
                self._device = device
        return self._mm

    def samples(self, n: int = 12) -> list[dict]:
        self._ensure_model()
        x, y = self._data.get("test")
        picked, seen = [], {}
        for i, label in enumerate(y):  # first images of each class, round-robin
            if seen.get(int(label), 0) < (n + 9) // 10:
                picked.append(i)
                seen[int(label)] = seen.get(int(label), 0) + 1
            if len(picked) >= n:
                break
        return [{"id": int(i), "label": CIFAR10_CLASSES[int(y[i])], "png": _png_b64(x[i])}
                for i in picked]

    def predict_array(self, img: np.ndarray, config: str, top_k: int = 3) -> dict:
        import torch

        from precisionpilot.data import preprocess

        mm = self._ensure_model()
        xb = torch.from_numpy(img[None]).permute(0, 3, 1, 2)
        with self._model_lock, torch.inference_mode():
            inp = preprocess(xb, self._device, self.cfg.data.image_size)
            p32 = torch.softmax(mm.fp32_forward(inp).float(), 1)[0].cpu().numpy()
            mm.set_config(str_to_cfg(config))
            pmx = torch.softmax(mm(inp).float(), 1)[0].cpu().numpy()
        return {"fp32": _topk(p32, top_k), "mixed": _topk(pmx, top_k), "config": config,
                "agree": int(p32.argmax()) == int(pmx.argmax())}

    def predict_sample(self, sample_id: int, config: str) -> dict:
        self._ensure_model()
        x, y = self._data.get("test")
        out = self.predict_array(x[sample_id], config)
        out["label"] = CIFAR10_CLASSES[int(y[sample_id])]
        return out


@lru_cache(maxsize=512)
def _select_cached(state: DemoState, gpu: str, bs: int, budget: float,
                   latency_model: str = "additive") -> dict:
    if not state.ready:
        raise RuntimeError("results not available")
    lut = state.luts[gpu]
    lat = (state.ctx[gpu] if latency_model == "context" else lut).latency_fn(bs)
    ref = state.reference["fp32_val_subset"]

    def ev():
        return Evaluator(state.table, lat, ref, budget)

    ex, _ = exhaustive_search(ev())
    eps = state.cfg.search.epsilon
    greedy = (greedy_dynamic(ev(), epsilon=eps) if latency_model == "context"
              else greedy_search(ev(), epsilon=eps))
    fp16_lat, opt = lat(all_fp16()), ex.latency
    from precisionpilot.search.common import make_result

    rows = [state._describe("greedy", greedy, gpu, bs, fp16_lat, opt),
            state._describe("optimum", ex, gpu, bs, fp16_lat, opt),
            state._describe("all-fp16", make_result("all-fp16", ev(), all_fp16()), gpu, bs,
                            fp16_lat, opt),
            state._describe("all-int8", make_result("all-int8", ev(), all_int8()), gpu, bs,
                            fp16_lat, opt)]
    units = [{"unit": i, "name": n,
              "precision": "int8" if greedy.config[i] == "1" else "fp16",
              "fp16_ms": lut.unit_ms(bs, i, FP16), "int8_ms": lut.unit_ms(bs, i, INT8)}
             for i, n in enumerate(UNIT_NAMES)]
    return {"gpu": gpu, "batch_size": bs, "budget": budget, "reference_acc": ref,
            "latency_model": latency_model, "methods": rows, "units": units, "space": 1024}


def _gpu_label(slug: str, name: str) -> str:
    return {"rtx3050ti": "RTX 3050 Ti", "t4": "Tesla T4", "mockampere": "Mock Ampere",
            "mockturing": "Mock Turing"}.get(slug, name or slug)


def _topk(p: np.ndarray, k: int) -> list[dict]:
    idx = np.argsort(-p)[:k]
    return [{"label": CIFAR10_CLASSES[int(i)], "prob": float(p[i])} for i in idx]


def _png_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_bytes_to_cifar(data: bytes) -> np.ndarray:
    """Centre-crop to a square and resize to 32x32 RGB (the model's native input)."""
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((32, 32), Image.BICUBIC), dtype=np.uint8)


def create_app(state: DemoState):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="PrecisionPilot demo")

    def _check_cfg(config: str) -> str:
        if len(config) != len(UNIT_NAMES) or set(config) - {"0", "1"}:
            raise HTTPException(400, "config must be a 10-character 0/1 string")
        return config

    @app.get("/api/meta")
    def meta():
        return state.meta()

    @app.get("/api/select")
    def select(gpu: str, batch_size: int, budget: float, latency_model: str | None = None):
        if not state.ready:
            raise HTTPException(409, "No results yet: run the accuracy and GPU stages first.")
        if gpu not in state.luts:
            raise HTTPException(404, f"unknown gpu {gpu}")
        if batch_size not in state.luts[gpu].batch_sizes:
            raise HTTPException(404, f"batch size {batch_size} not measured on {gpu}")
        if not 0 <= budget <= 10:
            raise HTTPException(400, "budget must be between 0 and 10 pp")
        if latency_model not in (None, "context", "additive"):
            raise HTTPException(400, "latency_model must be 'context' or 'additive'")
        try:
            return state.select(gpu, batch_size, budget, latency_model)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e

    @app.get("/api/samples")
    def samples(n: int = 12):
        try:
            return state.samples(max(1, min(n, 30)))
        except FileNotFoundError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/predict")
    def predict(sample_id: int, config: str):
        try:
            return state.predict_sample(sample_id, _check_cfg(config))
        except FileNotFoundError as e:
            raise HTTPException(409, str(e)) from e
        except IndexError as e:
            raise HTTPException(404, "unknown sample") from e

    @app.post("/api/upload")
    async def upload(config: str, file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(413, "image too large (max 8 MB)")
        try:
            img = image_bytes_to_cifar(data)
        except Exception as e:  # noqa: BLE001 - any decode failure is a client error
            raise HTTPException(400, "could not read image") from e
        out = state.predict_array(img, _check_cfg(config))
        out["png"] = _png_b64(img)
        return out

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(cfg, results_dir: Path, host: str = "127.0.0.1", port: int = 8000,
          fake: bool = False) -> None:
    import uvicorn

    app = create_app(DemoState(cfg, results_dir, fake=fake))
    print(f"PrecisionPilot demo on http://{host}:{port}  (results: {results_dir})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
