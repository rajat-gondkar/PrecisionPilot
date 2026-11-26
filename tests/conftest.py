import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)


def _has_trt() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs CUDA GPU")
    skip_trt = pytest.mark.skip(reason="needs TensorRT + CUDA")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
        if "trt" in item.keywords and not (torch.cuda.is_available() and _has_trt()):
            item.add_marker(skip_trt)
