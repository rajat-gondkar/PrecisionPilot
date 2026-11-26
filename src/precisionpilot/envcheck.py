"""Environment check: report PyTorch / CUDA / cuDNN / TensorRT versions and compatibility.

Version mismatches between PyTorch, CUDA and TensorRT are the most common time sink
(outline §12), so this is the first thing to run on every machine.
"""

from __future__ import annotations

import importlib
import platform
import sys


def _try_version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    return getattr(mod, "__version__", "unknown")


def collect() -> dict:
    import torch

    info: dict = {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": _try_version("torchvision"),
        "onnx": _try_version("onnx"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "mps_available": bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()),
        "gpu": None,
        "compute_capability": None,
        "tensorrt": _try_version("tensorrt"),
        "cuda_python": _try_version("cuda"),
    }
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{major}.{minor}"
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def diagnose(info: dict) -> list[tuple[str, str]]:
    """Return a list of (level, message) where level is OK / WARN / INFO."""
    out: list[tuple[str, str]] = []
    if info["cuda_available"]:
        out.append(("OK", f"CUDA GPU: {info['gpu']} (sm_{info['compute_capability']})"))
        cc = float(info["compute_capability"])
        if cc < 7.0:
            out.append(("WARN", "Compute capability < 7.0: no INT8 Tensor Cores; "
                                "INT8 speedups unlikely."))
        if info["tensorrt"] is None:
            out.append(("WARN", "TensorRT not installed: `pip install -r requirements/gpu.txt` "
                                "(latency backend 'trt' unavailable)."))
        else:
            trt_major = int(str(info["tensorrt"]).split(".")[0])
            if trt_major < 10:
                out.append(("WARN", "TensorRT < 10 detected; this project targets TRT 10.x "
                                    "(explicit Q/DQ quantization)."))
            else:
                out.append(("OK", f"TensorRT {info['tensorrt']}"))
            torch_cuda_major = str(info["torch_cuda_build"] or "0").split(".")[0]
            if torch_cuda_major != "12":
                out.append(("WARN", f"PyTorch built for CUDA {info['torch_cuda_build']}, but "
                                    "tensorrt-cu12 expects CUDA 12.x."))
        if info["cuda_python"] is None and info["tensorrt"] is not None:
            out.append(("INFO", "cuda-python not installed; TRT backend will fall back to "
                                "torch-managed CUDA buffers (fine)."))
    else:
        if info["mps_available"]:
            out.append(("INFO", "No CUDA. Apple MPS available: accuracy work runs, latency uses "
                                "the 'mock' backend."))
        else:
            out.append(("INFO", "No CUDA GPU: CPU-only. Use latency backend 'mock'."))
    if info["onnx"] is None:
        out.append(("WARN", "onnx not installed (needed for TensorRT export)."))
    return out


def run(verbose: bool = True) -> dict:
    info = collect()
    if verbose:
        print("PrecisionPilot environment check")
        print("-" * 40)
        for k, v in info.items():
            print(f"{k:>20}: {v}")
        print("-" * 40)
        for level, msg in diagnose(info):
            print(f"[{level:4}] {msg}")
    return info
