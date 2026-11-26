import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from precisionpilot.config import load_config  # noqa: E402
from precisionpilot.demo.server import DemoState, create_app, image_bytes_to_cifar  # noqa: E402


@pytest.fixture(scope="module")
def client(smoke_results):
    results, _ = smoke_results
    cfg = load_config(overrides=["data.image_size=16", "data.eval_batch_size=32",
                                 "quant.calib_batches=1"])
    return TestClient(create_app(DemoState(cfg, results, fake=True)))


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "PrecisionPilot" in r.text
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_meta(client):
    m = client.get("/api/meta").json()
    assert m["ready"] is True
    assert {g["id"] for g in m["gpus"]} == {"mockampere", "mockturing"}
    assert len(m["units"]) == 10 and len(m["sensitivity"]) == 10
    assert m["main_budget"] == 1.0


def test_select_any_budget(client):
    r = client.get("/api/select", params={"gpu": "mockampere", "batch_size": 32, "budget": 1.3})
    assert r.status_code == 200
    sel = r.json()
    methods = {m["method"]: m for m in sel["methods"]}
    assert set(methods) == {"greedy", "optimum", "all-fp16", "all-int8"}
    assert methods["optimum"]["n_evals"] == 1024
    assert methods["greedy"]["feasible"]
    assert methods["greedy"]["gap_pct"] >= -1e-9
    assert methods["all-fp16"]["speedup"] == pytest.approx(1.0)
    assert len(sel["units"]) == 10
    assert all(u["precision"] in {"fp16", "int8"} for u in sel["units"])
    # larger budget never yields a slower greedy result than a zero budget's all-FP16
    zero = client.get("/api/select", params={"gpu": "mockampere", "batch_size": 32,
                                             "budget": 0}).json()
    assert {m["method"]: m for m in zero["methods"]}["greedy"]["drop_pp"] <= 1e-9


def test_select_errors(client):
    assert client.get("/api/select", params={"gpu": "nope", "batch_size": 1,
                                             "budget": 1}).status_code == 404
    assert client.get("/api/select", params={"gpu": "mockampere", "batch_size": 7,
                                             "budget": 1}).status_code == 404
    assert client.get("/api/select", params={"gpu": "mockampere", "batch_size": 1,
                                             "budget": 99}).status_code == 400


def test_samples_and_predict(client):
    samples = client.get("/api/samples", params={"n": 4}).json()
    assert len(samples) == 4 and samples[0]["png"].startswith("data:image/png;base64,")
    r = client.get("/api/predict", params={"sample_id": samples[0]["id"],
                                           "config": "0111111110"})
    assert r.status_code == 200
    out = r.json()
    assert len(out["fp32"]) == 3 and len(out["mixed"]) == 3
    assert abs(sum(p["prob"] for p in out["fp32"])) <= 1.0 + 1e-6
    assert client.get("/api/predict", params={"sample_id": 0,
                                              "config": "012"}).status_code == 400


def test_upload(client):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((50, 80, 3), 200, dtype=np.uint8)).save(buf, format="PNG")
    r = client.post("/api/upload", params={"config": "0000000000"},
                    files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and r.json()["agree"] is True
    bad = client.post("/api/upload", params={"config": "0000000000"},
                      files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400


def test_image_preprocessing_is_square_32():
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((40, 100, 3), dtype=np.uint8)).save(buf, format="JPEG")
    assert image_bytes_to_cifar(buf.getvalue()).shape == (32, 32, 3)


def test_not_ready_state(tmp_path):
    cfg = load_config()
    c = TestClient(create_app(DemoState(cfg, tmp_path)))
    assert c.get("/api/meta").json()["ready"] is False
    assert c.get("/api/select", params={"gpu": "x", "batch_size": 1,
                                        "budget": 1}).status_code == 409
