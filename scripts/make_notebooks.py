"""Generate the Kaggle notebooks in notebooks/ (keeps them reviewable as plain Python).

Run:  python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/rajat-gondkar/PrecisionPilot.git"
OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def setup(base: str) -> str:
    return f"""
import os, subprocess, sys
os.makedirs("{base}", exist_ok=True)
os.chdir("{base}")
if not os.path.exists("PrecisionPilot"):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
os.chdir("{base}/PrecisionPilot")
subprocess.run(["git", "pull", "--ff-only"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
"""


FINETUNE = notebook([
    md("""
# PrecisionPilot — 01 · Fine-tune ResNet-18 on CIFAR-10 (224 px)

**Before running:** *Settings → Accelerator → GPU T4* and *Settings → Internet → On*.
Then *Run All*. The run takes roughly 10 minutes on a T4.

What it does:
1. clones the PrecisionPilot repo and installs it,
2. checks the environment,
3. downloads CIFAR-10 and loads the committed data splits (`data/splits.json`),
4. fine-tunes the ImageNet-pretrained ResNet-18 (AMP, SGD + cosine, 8 epochs),
5. reports FP32 / all-FP16 top-1 on the validation and test splits,
6. copies `resnet18_cifar10.pt` (+ `.json` training log) to `/kaggle/working/` for download.

**After it finishes:** open the *Output* panel, download `resnet18_cifar10.pt` and
`resnet18_cifar10.json`, and place both in `checkpoints/` of your local repo
(Mac and Windows). Also download `baselines_t4.json` into `results/baselines/`.
"""),
    code("!nvidia-smi"),
    code(setup("/kaggle/working")),
    code("!python -m precisionpilot env-check"),
    md("### Data splits\nThe split file is committed in the repo, so every machine uses "
       "identical indices. This cell downloads CIFAR-10 and verifies the splits."),
    code("!python -m precisionpilot make-splits"),
    md("### Fine-tune\nOverride any setting with `--set`, e.g. `--set train.epochs=10`."),
    code("!python -m precisionpilot train --device cuda"),
    md("### Baselines (FP32 and all-FP16)"),
    code("!python -m precisionpilot evaluate --device cuda"),
    code("""
import shutil, glob
for f in ["checkpoints/resnet18_cifar10.pt", "checkpoints/resnet18_cifar10.json",
          *glob.glob("results/baselines/*.json")]:
    shutil.copy(f, "/kaggle/working/")
    print("ready for download:", "/kaggle/working/" + f.split("/")[-1])
"""),
])


BENCHMARK = notebook([
    md("""
# PrecisionPilot — 02 · T4 accuracy table + TensorRT benchmark

## Run it unattended (so you can sleep)
1. *Settings → Accelerator → GPU T4* (x1 or x2) and *Internet → On*.
2. Right panel → *Add Input → Datasets* → your dataset containing `resnet18_cifar10.pt`
   (one-time: kaggle.com → *Datasets* → *New Dataset* → upload
   `checkpoints/resnet18_cifar10.pt`).
3. Click **Save Version → Save & Run All (Commit)** (not plain *Run All*). It then runs in the
   background on Kaggle's servers: you can close the browser or sleep. A committed run may take
   up to 12 hours; this one should take roughly 2–3 hours (it builds ~250 TensorRT engines: the
   per-unit table, the context-aware latency model's fitting set, and the verification set). The finished version and its output
   files are kept permanently.
4. When you are back: open the notebook → *Versions* (or the *Output* tab of the latest
   version) → download **`precisionpilot_results_t4.zip`** and unzip it into the repo root.

This notebook never stops at the first failure: every stage runs, the results are zipped after
each stage, and `status.json` records which stages succeeded. Whatever finished is in the zip.

## What it does
1. installs PrecisionPilot + TensorRT 10 and checks versions (work happens in `/tmp`, so only the
   zip and logs are saved as output),
2. **hardware-independent accuracy** (`run-accuracy`): FP32 / FP16 / INT8 references, accuracy of
   all 1,024 configurations on the validation subset, per-unit sensitivity,
3. **T4 latency study** (`run-gpu --backend trt`): per-unit lookup table for batch 1/8/32, the
   context-aware latency model (fitted on full-engine measurements), all search methods x budgets
   under both latency models, full-engine verification incl. measured GPU memory, real TensorRT
   test accuracy,
4. builds the figures and tables.
"""),
    code("!nvidia-smi"),
    code(setup("/tmp/pp")),
    code("""
import glob, json, os, shutil, subprocess, sys, time

STATUS = {}

def run(cmd, label):
    \"\"\"Run a shell command, stream its output, never raise.\"\"\"
    print(f"\\n===== {label} =====", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    STATUS[label] = {"exit_code": rc, "seconds": round(time.time() - t0)}
    print(f"===== {label}: exit {rc} after {time.time() - t0:.0f}s =====", flush=True)
    save_outputs()
    return rc

def save_outputs():
    \"\"\"Zip everything produced so far into /kaggle/working (overwrites the previous zip).\"\"\"
    if os.path.isdir("results"):
        shutil.make_archive("/kaggle/working/precisionpilot_results_t4", "zip", ".", "results")
    with open("/kaggle/working/status.json", "w") as f:
        json.dump(STATUS, f, indent=2)
"""),
    code("""
run(f"{sys.executable} -m pip install -q 'tensorrt-cu12>=10.0,<11'", "install tensorrt")
run(f"{sys.executable} -m precisionpilot env-check", "env-check")
"""),
    code("""
os.makedirs("checkpoints", exist_ok=True)
found = sorted(glob.glob("/kaggle/input/**/resnet18_cifar10.pt", recursive=True))
assert found, ("resnet18_cifar10.pt not found under /kaggle/input. Add your checkpoint "
               "Dataset via the right panel: Add Input -> Datasets.")
shutil.copy(found[0], "checkpoints/resnet18_cifar10.pt")
js = glob.glob(os.path.join(os.path.dirname(found[0]), "resnet18_cifar10.json"))
if js:
    shutil.copy(js[0], "checkpoints/")
print("checkpoint:", found[0])
"""),
    code("""
PY = sys.executable
run(f"{PY} -m precisionpilot make-splits", "make-splits")
run(f"{PY} -m precisionpilot evaluate --device cuda", "evaluate (FP32 / FP16 baselines)")
"""),
    md("### Stage 1 — hardware-independent accuracy (reference, 1,024-config table, sensitivity)"),
    code("""run(f"{PY} -m precisionpilot run-accuracy --device cuda", "run-accuracy")"""),
    md("### Stage 2 — T4 latency study with TensorRT\n"
       "Engines are cached in `/tmp` (not saved as output)."),
    code("""
run(f"{PY} -m precisionpilot run-gpu --device cuda --backend trt "
    "--set paths.engine_cache=/tmp/engine_cache", "run-gpu (TensorRT)")
"""),
    md("### Stage 3 — figures and tables"),
    code("""run(f"{PY} -m precisionpilot report", "report")"""),
    code("""
save_outputs()
print(json.dumps(STATUS, indent=2))
failed = [k for k, v in STATUS.items() if v["exit_code"] != 0]
print("\\nFAILED STAGES:", failed if failed else "none")
print("Download: precisionpilot_results_t4.zip (+ status.json)")
"""),
])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "kaggle_01_finetune.ipynb").write_text(json.dumps(FINETUNE, indent=1))
    (OUT / "kaggle_02_t4_benchmark.ipynb").write_text(json.dumps(BENCHMARK, indent=1))
    print("wrote notebooks to", OUT)


if __name__ == "__main__":
    main()
