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


SETUP = f"""
import os, subprocess, sys
os.chdir("/kaggle/working")
if not os.path.exists("PrecisionPilot"):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
os.chdir("/kaggle/working/PrecisionPilot")
subprocess.run(["git", "pull", "--ff-only"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
"""

FINETUNE = notebook([
    md("""
# PrecisionPilot — 01 · Fine-tune ResNet-18 on CIFAR-10 (224 px)

**Before running:** *Settings → Accelerator → GPU T4 x1* and *Settings → Internet → On*.
Then *Run All*. The run takes roughly 15–25 minutes on a T4.

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
    code(SETUP),
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

**Before running:**
1. *Settings → Accelerator → GPU T4* (x1 or x2; only GPU 0 is used) and *Internet → On*.
2. *Add Input* → **Notebook Output** → pick your run of `01 · Fine-tune` (it contains
   `resnet18_cifar10.pt`). Alternatively upload the checkpoint as a Kaggle Dataset.
3. *Run All*. Expected time: roughly 1–2 hours (mostly TensorRT engine builds and the
   1,024-config accuracy table).

What it does:
1. installs PrecisionPilot + TensorRT 10 and checks versions,
2. **hardware-independent accuracy** (`run-accuracy`): FP32 / FP16 / INT8 references,
   accuracy of all 1,024 configurations on the validation subset, per-unit sensitivity,
3. **T4 latency study** (`run-gpu --backend trt`): per-unit lookup table for batch 1/8/32,
   all search methods x budgets, full-engine verification, real TensorRT test accuracy,
4. builds the figures and zips `results/` for download.

**After it finishes:** download `precisionpilot_results_t4.zip` and unzip it into the repo
root on your Mac (it fills `results/accuracy/`, `results/t4/`, `results/figures/`,
`results/tables/`). The accuracy files are needed by the Windows benchmark run.
"""),
    code("!nvidia-smi"),
    code(SETUP),
    code("!pip install -q \"tensorrt-cu12>=10.0,<11\""),
    code("!python -m precisionpilot env-check"),
    code("""
import glob, os, shutil
os.makedirs("checkpoints", exist_ok=True)
found = sorted(glob.glob("/kaggle/input/**/resnet18_cifar10.pt", recursive=True))
assert found, "Add the fine-tuning notebook output (or a dataset with the checkpoint) as input"
shutil.copy(found[0], "checkpoints/resnet18_cifar10.pt")
js = glob.glob(os.path.join(os.path.dirname(found[0]), "resnet18_cifar10.json"))
if js:
    shutil.copy(js[0], "checkpoints/")
print("checkpoint:", found[0])
"""),
    code("!python -m precisionpilot make-splits"),
    code("!python -m precisionpilot evaluate --device cuda"),
    md("### Hardware-independent accuracy (reference, 1,024-config table, sensitivity)"),
    code("!python -m precisionpilot run-accuracy --device cuda"),
    md("### T4 latency study with TensorRT"),
    code("!python -m precisionpilot run-gpu --device cuda --backend trt"),
    code("!python -m precisionpilot report"),
    code("""
import shutil
shutil.make_archive("/kaggle/working/precisionpilot_results_t4", "zip", ".", "results")
print("download: /kaggle/working/precisionpilot_results_t4.zip")
"""),
])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "kaggle_01_finetune.ipynb").write_text(json.dumps(FINETUNE, indent=1))
    (OUT / "kaggle_02_t4_benchmark.ipynb").write_text(json.dumps(BENCHMARK, indent=1))
    print("wrote notebooks to", OUT)


if __name__ == "__main__":
    main()
