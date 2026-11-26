# PrecisionPilot

**Sensitivity-guided FP16/INT8 mixed-precision selection for GPU inference.**

Given a pretrained ResNet-18, a GPU, a batch size and an accuracy budget, PrecisionPilot picks
which of the model's 10 units run in INT8 and which stay in FP16 — minimising latency while
keeping the top-1 drop vs FP32 within budget — using a handful of evaluations instead of
brute-forcing all 2^10 configurations.

Full technical documentation (method, experiments, results, limitations): **[DOCUMENTATION.md](DOCUMENTATION.md)**.

---

## Where things run

| Machine | Used for | Latency backend |
|---|---|---|
| macOS (Apple Silicon / Intel) | development, tests, accuracy simulation (CPU/MPS), demo | `mock` |
| Windows laptop, RTX 3050 Ti | **primary latency benchmark** (TensorRT) | `trt` |
| Kaggle notebook, Tesla T4 | fine-tuning, exhaustive accuracy table, second-GPU benchmark | `trt` |

Every command is a Python entry point (`python -m precisionpilot ...`), so the same commands
work in zsh, bash, PowerShell and cmd.

## Setup

Python 3.10–3.12.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,demo]"
python -m precisionpilot env-check
pytest
```

### Windows (PowerShell) — with NVIDIA GPU
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# 1) CUDA 12 build of PyTorch (must be installed before the package)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 2) the package + TensorRT
pip install -e ".[dev,demo,gpu]"
python -m precisionpilot env-check
pytest
```
`env-check` must report a CUDA 12 PyTorch build and TensorRT 10.x with no `WARN` lines before
any benchmarking. If `Activate.ps1` is blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Laptop benchmarking hygiene: plug in, select the Windows *Best performance* power mode,
close other GPU apps (browsers, games, overlays).

### Kaggle
Use the notebooks in [`notebooks/`](notebooks/) (Settings → Accelerator → **GPU T4**, Internet **on**).
Each notebook clones this repository and installs it with `pip install -e .`.

## Commands

```text
python -m precisionpilot env-check        # versions + compatibility warnings
```
More commands are added as the project phases land; `python -m precisionpilot --help` lists them.

Any config value can be overridden from the CLI, e.g.
`python -m precisionpilot <cmd> --set train.epochs=2 latency.backend=trt`.

## Repository layout

```
configs/            YAML configuration (accuracy budget is fixed here)
src/precisionpilot/ library + CLI
notebooks/          Kaggle notebooks (training, T4 benchmark)
demo/               lightweight web demo (FastAPI + vanilla JS)
tests/              pytest suite (CPU; GPU/TRT tests auto-skip)
results/            small result tables/figures committed to git
```
