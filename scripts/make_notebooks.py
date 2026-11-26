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


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "kaggle_01_finetune.ipynb").write_text(json.dumps(FINETUNE, indent=1))
    print("wrote", OUT / "kaggle_01_finetune.ipynb")


if __name__ == "__main__":
    main()
