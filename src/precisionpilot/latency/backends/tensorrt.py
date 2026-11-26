"""TensorRT 10.x backend: real FP16 / INT8 latency via explicit (Q/DQ) quantization.

Pipeline per measurement:
  MixedPrecisionModel --(export_unit / export_config)--> ONNX with Q/DQ in INT8 parts
  --(TensorRT builder, FP16 + INT8 flags)--> engine (cached on disk) --> timed with CUDA events.

Buffers are allocated with PyTorch on the GPU, so no pycuda / cuda-python is required.
Engines are cached under ``.engine_cache/`` keyed by ONNX bytes, TensorRT version and GPU name.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch

from precisionpilot.latency.backends.base import LatencyBackend
from precisionpilot.latency.timing import TimingStats, cooldown, time_cuda


def _trt():
    try:
        import tensorrt as trt
    except ImportError as e:  # pragma: no cover - depends on machine
        raise RuntimeError("TensorRT not installed: pip install -r requirements/gpu.txt") from e
    return trt


_TRT_TO_TORCH = None


def _torch_dtype(trt, dt):
    global _TRT_TO_TORCH
    if _TRT_TO_TORCH is None:
        _TRT_TO_TORCH = {trt.float32: torch.float32, trt.float16: torch.float16,
                         trt.int8: torch.int8, trt.int32: torch.int32, trt.bool: torch.bool}
        if hasattr(trt, "int64"):
            _TRT_TO_TORCH[trt.int64] = torch.int64
    return _TRT_TO_TORCH[dt]


class TrtEngine:
    """A deserialized engine + execution context with torch-allocated I/O buffers."""

    def __init__(self, engine_bytes: bytes):
        trt = _trt()
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.size_bytes = len(engine_bytes)
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _torch_dtype(trt, self.engine.get_tensor_dtype(name))
            buf = torch.zeros(shape, dtype=dtype, device="cuda")
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = buf
            else:
                self.outputs[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())

    @property
    def device_memory_bytes(self) -> int:
        e = self.engine
        for attr in ("device_memory_size_v2", "device_memory_size"):
            if hasattr(e, attr):
                return int(getattr(e, attr))
        return 0

    def run(self) -> None:
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        inp = next(iter(self.inputs.values()))
        inp.copy_(x.to(inp.dtype))
        self.run()
        torch.cuda.current_stream().synchronize()
        return next(iter(self.outputs.values())).clone()


def build_engine_bytes(onnx_path: str | Path, fp16: bool = True, int8: bool | None = None,
                       io_fp16: bool = True, workspace_mb: int = 1024) -> bytes:
    """Build a serialized engine from ONNX.

    ``int8=None`` (default) enables the INT8 builder flag only if the graph has Q/DQ nodes.
    With the flag on but no Q/DQ nodes and no calibrator, TensorRT 10 fails with
    "Calibration failure occurred with no scaling factors detected".
    ``io_fp16`` makes the network inputs/outputs FP16 (linear), so a unit engine does not pay
    FP32<->FP16 casts at its boundaries; this matches the FP16 tensors that flow between units.
    """
    from precisionpilot.export.onnx_export import has_qdq

    trt = _trt()
    if int8 is None:
        int8 = has_qdq(onnx_path)
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt, "NetworkDefinitionCreationFlag") and \
            hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") and \
            int(trt.__version__.split(".")[0]) < 10:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if io_fp16 and fp16:
        linear = 1 << int(trt.TensorFormat.LINEAR)
        for i in range(network.num_inputs):
            t = network.get_input(i)
            t.dtype = trt.float16
            t.allowed_formats = linear
        for i in range(network.num_outputs):
            t = network.get_output(i)
            t.dtype = trt.float16
            t.allowed_formats = linear
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    return bytes(serialized)


class TensorRTBackend(LatencyBackend):
    name = "trt"

    def __init__(self, warmup: int = 20, reps: int = 100, cooldown_s: float = 0.0,
                 cache_dir: str | Path = ".engine_cache", workspace_mb: int = 1024,
                 io_fp16: bool = True):
        super().__init__(warmup, reps, cooldown_s)
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend needs a CUDA GPU")
        self.trt = _trt()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_mb = workspace_mb
        self.io_fp16 = io_fp16
        self._io_confirmed = False   # True once one FP16-I/O build has succeeded
        self._onnx_dir = Path(tempfile.mkdtemp(prefix="pp_onnx_"))

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(0)

    @property
    def notes(self) -> str:
        return f"trt={self.trt.__version__} io={'fp16' if self.io_fp16 else 'fp32'}"

    # -- engines -------------------------------------------------------------
    def _engine_for(self, onnx_path: Path) -> TrtEngine:
        def cache_file() -> Path:
            key = hashlib.sha1(onnx_path.read_bytes() + self.trt.__version__.encode()
                               + self.device_name.encode()
                               + f"io{int(self.io_fp16)}".encode()).hexdigest()[:20]
            return self.cache_dir / f"{onnx_path.stem}_{key}.engine"

        cached = cache_file()
        if cached.exists():
            return TrtEngine(cached.read_bytes())
        try:
            data = build_engine_bytes(onnx_path, io_fp16=self.io_fp16,
                                      workspace_mb=self.workspace_mb)
        except RuntimeError:
            if not self.io_fp16 or self._io_confirmed:
                raise
            # First build ever and FP16 I/O was rejected: use FP32 I/O for the whole run
            # (consistently), and say so (recorded in the LUT's `notes` column).
            print("[trt] FP16 network I/O not accepted; falling back to FP32 I/O "
                  "(unit latencies will include boundary casts)")
            self.io_fp16 = False
            cached = cache_file()
            data = build_engine_bytes(onnx_path, io_fp16=False, workspace_mb=self.workspace_mb)
        else:
            self._io_confirmed = True
        cached.write_bytes(data)
        return TrtEngine(data)

    def engine_for_config(self, mm, cfg, batch_size, image_size) -> TrtEngine:
        from precisionpilot.export.onnx_export import export_config

        return self._engine_for(export_config(mm, cfg, batch_size, image_size, self._onnx_dir))

    def engine_for_unit(self, mm, index, precision, input_shape) -> TrtEngine:
        from precisionpilot.export.onnx_export import export_unit

        return self._engine_for(export_unit(mm, index, precision, input_shape, self._onnx_dir))

    # -- timing --------------------------------------------------------------
    def _time(self, eng: TrtEngine) -> TimingStats:
        for buf in eng.inputs.values():
            if buf.is_floating_point():
                buf.normal_()
        stats = time_cuda(eng.run, self.warmup, self.reps)
        cooldown(self.cooldown_s)
        return stats

    def measure_unit(self, mm, index, precision, input_shape) -> TimingStats:
        return self._time(self.engine_for_unit(mm, index, precision, input_shape))

    def measure_config(self, mm, cfg, batch_size, image_size) -> TimingStats:
        return self._time(self.engine_for_config(mm, cfg, batch_size, image_size))

    def model_info(self, mm, cfg, batch_size, image_size) -> dict:
        eng = self.engine_for_config(mm, cfg, batch_size, image_size)
        return {"engine_mb": eng.size_bytes / 1e6,
                "activation_memory_mb": eng.device_memory_bytes / 1e6}

    # -- real accuracy -------------------------------------------------------
    @torch.inference_mode()
    def accuracy(self, mm, cfg, x: np.ndarray, y: np.ndarray, batch_size: int,
                 image_size: int) -> float:
        """Top-1 of the *real* TensorRT engine (static batch; the tail batch is padded)."""
        from precisionpilot.data import iterate_batches, preprocess

        eng = self.engine_for_config(mm, cfg, batch_size, image_size)
        correct = 0
        for xb, yb in iterate_batches(x, y, batch_size):
            n = len(yb)
            inp = preprocess(xb, "cuda", image_size)
            if n < batch_size:
                inp = torch.cat([inp, inp.new_zeros(batch_size - n, *inp.shape[1:])])
            logits = eng.infer(inp)[:n]
            correct += int((logits.float().argmax(1).cpu() == yb).sum())
        return 100.0 * correct / max(1, len(y))
