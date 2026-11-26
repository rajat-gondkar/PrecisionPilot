import numpy as np
import pytest
import torch

from precisionpilot.data import (
    IMAGENET_MEAN,
    CifarData,
    iterate_batches,
    load_or_create_splits,
    make_fake_cifar10,
    make_splits,
    preprocess,
)


@pytest.fixture
def arrays():
    return make_fake_cifar10(n_train=1000, n_test=600, seed=0)


def _splits(arrays, seed=7):
    return make_splits(arrays.y_train, arrays.y_test, seed, n_calib=100, n_spare=100,
                       n_val=300, n_val_subset=100, n_test=300)


def test_split_sizes(arrays):
    s = _splits(arrays)
    assert s.sizes() == {"finetune": 800, "calib": 100, "spare": 100, "val": 300,
                         "val_subset": 100, "test": 300}


def test_splits_disjoint_and_cover(arrays):
    s = _splits(arrays)
    tr = [set(s.indices[k]) for k in ("finetune", "calib", "spare")]
    assert not (tr[0] & tr[1]) and not (tr[0] & tr[2]) and not (tr[1] & tr[2])
    assert set().union(*tr) == set(range(1000))
    assert not set(s.indices["val"]) & set(s.indices["test"])
    assert set(s.indices["val_subset"]) <= set(s.indices["val"])


def test_splits_deterministic(arrays):
    assert _splits(arrays, 3).indices == _splits(arrays, 3).indices
    assert _splits(arrays, 3).indices != _splits(arrays, 4).indices


def test_splits_stratified(arrays):
    s = _splits(arrays)
    counts = np.bincount(arrays.y_train[s["calib"]], minlength=10)
    assert counts.min() == counts.max() == 10


def test_splits_persist_and_reload(arrays, tmp_path):
    path = tmp_path / "splits.json"
    a = load_or_create_splits(path, arrays, 1, 100, 100, 300, 100, 300)
    assert path.exists()
    b = load_or_create_splits(path, arrays, 1, 100, 100, 300, 100, 300)
    assert a.indices == b.indices
    c = load_or_create_splits(path, arrays, 2, 100, 100, 300, 100, 300)  # seed change
    assert c.seed == 2


def test_cifardata_get_uses_right_source(arrays):
    s = _splits(arrays)
    d = CifarData(arrays, s)
    x, y = d.get("test")
    assert np.array_equal(y, arrays.y_test[s["test"]])
    x, y = d.get("calib", limit=10)
    assert x.shape == (10, 32, 32, 3)


def test_iterate_batches_shapes():
    x = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    y = np.arange(10)
    batches = list(iterate_batches(x, y, 4))
    assert [b[0].shape[0] for b in batches] == [4, 4, 2]
    assert batches[0][0].shape == (4, 3, 32, 32)
    assert len(list(iterate_batches(x, y, 4, drop_last=True))) == 2


def test_preprocess_resize_and_normalise():
    xb = torch.full((2, 3, 32, 32), 255, dtype=torch.uint8)
    out = preprocess(xb, "cpu", image_size=224)
    assert out.shape == (2, 3, 224, 224) and out.dtype == torch.float32
    expected = (1 - IMAGENET_MEAN[0]) / 0.229
    assert torch.allclose(out[:, 0], torch.full_like(out[:, 0], expected), atol=1e-5)
    assert preprocess(xb, "cpu", 32, dtype=torch.float16).dtype == torch.float16


def test_augment_preserves_shape_and_is_seeded():
    xb = torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)
    a = preprocess(xb, "cpu", 32, augment=True, generator=torch.Generator().manual_seed(0))
    b = preprocess(xb, "cpu", 32, augment=True, generator=torch.Generator().manual_seed(0))
    assert a.shape == (4, 3, 32, 32) and torch.equal(a, b)
