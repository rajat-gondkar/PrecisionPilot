import pytest
import torch


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
