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
python -m precisionpilot make-splits      # download CIFAR-10, verify data/splits.json
python -m precisionpilot train            # fine-tune ResNet-18 (run on Kaggle GPU)
python -m precisionpilot evaluate         # FP32 / all-FP16 baselines (needs checkpoint)
python -m precisionpilot run-accuracy     # reference accs, 1,024-config table, sensitivity
python -m precisionpilot build-lut        # per-unit latency table on this GPU
python -m precisionpilot run-gpu          # LUT + all searches + verification + final test
python -m precisionpilot report           # regenerate figures and tables from results/
python -m precisionpilot pipeline --smoke # end-to-end check on synthetic data (~20 s)
python -m precisionpilot export-onnx --cfg-bits 0111111110   # one config -> ONNX (Q/DQ)
```
On machines without a CUDA GPU, the CLI limits PyTorch to 2 CPU threads by default
(override with `--threads N`).

## Running the experiments

| Step | Where | Command / notebook | Produces |
|---|---|---|---|
| 1. Fine-tune | Kaggle T4 | `notebooks/kaggle_01_finetune.ipynb` | `checkpoints/resnet18_cifar10.pt` |
| 2. Accuracy + T4 latency | Kaggle T4 (add step 1's output as input) | `notebooks/kaggle_02_t4_benchmark.ipynb` | `results/accuracy/*`, `results/t4/*` (download the zip, unzip in the repo root) |
| 3. RTX 3050 Ti latency | Windows laptop | see below | `results/rtx3050ti/*` |
| 4. Figures and tables | anywhere | `python -m precisionpilot report` | `results/figures/`, `results/tables/` |

Step 3 on Windows (PowerShell), after `git pull` and copying the checkpoint into `checkpoints\`:
```powershell
.\.venv\Scripts\Activate.ps1
python -m precisionpilot env-check           # must show CUDA 12 + TensorRT 10, no WARN
python -m precisionpilot make-splits
python -m precisionpilot run-gpu --device cuda --backend trt
python -m precisionpilot report
```
Step 3 needs `results/accuracy/` from step 2 (commit it, or copy it over).
Expect roughly 1 hour, most of it spent building TensorRT engines; they are cached in
`.engine_cache\`, so a rerun is quick.
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
