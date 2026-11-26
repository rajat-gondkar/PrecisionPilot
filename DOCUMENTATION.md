# PrecisionPilot — Project Documentation

*Living technical documentation and report source. Updated at the end of every project phase.*

**Status:** Phases 0–4 implemented. GPU runs (Kaggle fine-tuning, TensorRT benchmarks) are pending.

---

## Table of contents
1. [Introduction](#1-introduction)
2. [Problem statement and claim](#2-problem-statement-and-claim)
3. [Related work and novelty positioning](#3-related-work-and-novelty-positioning)
4. [System overview](#4-system-overview)
5. [Method](#5-method)
6. [Experimental setup](#6-experimental-setup)
7. [Experiments and results](#7-experiments-and-results)
8. [Limitations](#8-limitations)
9. [Reproduction guide](#9-reproduction-guide)
10. [Testing](#10-testing)
11. [Project log](#11-project-log)
12. [References](#12-references)

---

## 1. Introduction

Neural networks can execute in several numeric formats. FP32 is the accuracy reference; FP16
halves memory traffic and uses Tensor Cores; INT8 maps values to 8-bit integers through a scale
(and optional zero point) and is usually the smallest and often the fastest format. Lower
precision is not uniformly harmful: some layers (typically the first and the last) are fragile,
while others tolerate INT8 with negligible accuracy loss. Whether INT8 is *faster* also depends
on the hardware, the operator, the tensor sizes and the batch size.

**PrecisionPilot** is a lightweight framework that decides, per unit of a pretrained network,
whether that unit runs in INT8 or FP16, so that inference latency is minimised subject to a
user-defined accuracy-loss budget. The decision is made by a **sensitivity-guided greedy search**
that needs on the order of tens of accuracy evaluations, compared with 2^N for exhaustive search.

## 2. Problem statement and claim

- Running everything in high precision wastes memory and time.
- Running everything in INT8 can hurt accuracy badly.
- Exhaustively testing per-layer combinations grows as 2^N.
- The best configuration depends on batch size and GPU.

**Central question.** Can a cheap, sensitivity-guided search find mixed-precision
configurations that stay within an accuracy budget, with far fewer evaluations than brute-force
or random search, while adapting to batch size and hardware?

**Claim tested.** *A sensitivity-guided search finds mixed-precision configurations within the
accuracy budget using far fewer evaluations than brute-force or random search, and the best
configuration depends on batch size and GPU.*

### Scope
| Included | Excluded |
|---|---|
| ResNet-18 (ImageNet-pretrained, fine-tuned on CIFAR-10 at 224 px) | Quantization-aware training |
| Two precisions per unit: FP16 / INT8 (FP32 = accuracy reference) | Runtime/dynamic precision switching |
| Post-training quantization only | Energy measurement |
| Offline selection per (model, GPU, batch size) | Large models (ResNet-50+, transformers) |
| Batch sizes 1, 8, 32; GPUs RTX 3050 Ti (main) and Tesla T4 | Many-tool comparisons |

## 3. Related work and novelty positioning

*Literature check performed at project start (Phase 0), as recommended by the outline.*

### 3.1 Quantization fundamentals
- **Jacob et al. (2018)** introduced the integer-arithmetic-only inference scheme
  (scale + zero point, quantized training) used by most deployment stacks.
- **Krishnamoorthi (2018)** is the standard practical whitepaper: per-channel weight
  quantization, post-training quantization (PTQ) vs quantization-aware training (QAT), and the
  observation that per-channel symmetric weights + per-tensor activations keep CNNs close to FP32.
- **Wu et al. (NVIDIA, 2020)**, *Integer Quantization for Deep Learning Inference: Principles
  and Empirical Evaluation*, give the TensorRT-oriented recipe (symmetric, per-channel weights,
  calibrated activations: max / entropy / percentile) and use **partial quantization**: a
  per-layer sensitivity analysis after which the most sensitive layers are left in higher
  precision until the accuracy target is met. **This is the closest prior work to PrecisionPilot**
  and the reason our contribution is framed around *latency-aware* scoring, search cost and
  hardware/batch awareness rather than sensitivity analysis itself.

### 3.2 Mixed-precision search
| Work | Sensitivity / search signal | Cost of search | Hardware in the loop? |
|---|---|---|---|
| DNAS (Wu et al., 2018) | Differentiable NAS over bit-widths | Training a supernet | Indirect (cost proxy) |
| HAQ (Wang et al., CVPR 2019) | Reinforcement learning agent | Many episodes, fine-tuning | **Yes**, latency/energy from hardware simulator or measurement |
| HAWQ (Dong et al., ICCV 2019) | Top Hessian eigenvalue per block | Hessian computation | No |
| HAWQ-V2 (Dong et al., NeurIPS 2020) | Average Hessian trace; Pareto-frontier selection | Hessian trace estimation | No (model-size constraint) |
| HAWQ-V3 (Yao et al., ICML 2021) | Hessian + integer linear programming (ILP) | ILP solve | **Yes**, measured TVM latency on T4 |
| ZeroQ (Cai et al., CVPR 2020) | Data-free, distilled data; Pareto frontier from per-layer sensitivity | Cheap | No (size constraint) |
| BRECQ (Li et al., ICLR 2021) | Block reconstruction PTQ; genetic algorithm for mixed precision | Reconstruction + GA | Uses a hardware cost model |
| Tool-based accuracy-aware tuning (e.g. Intel Neural Compressor, NVIDIA TensorRT Model Optimizer / Polygraphy) | Per-op fallback to higher precision until an accuracy criterion is met | Moderate | Partially |

### 3.3 What is (and is not) new here
FP32/FP16/INT8 comparison with a threshold selector is already common in deployment tools.
Per-layer sensitivity analysis and mixed-precision search are well studied (HAWQ, HAQ, ZeroQ,
NVIDIA partial quantization). PrecisionPilot does **not** claim a new field or a new idea.

The defensible contribution is a **lightweight, practical, accuracy-constrained method** with:
1. **Latency-aware scoring** — units are ranked by *measured latency saved per accuracy point
   lost* on the target GPU and batch size, not by sensitivity alone (unlike partial
   quantization or HAWQ-V1/V2);
2. **Low and reported search cost** — number of evaluations and wall-clock time vs random search
   at an equal evaluation budget and vs exhaustive search;
3. **Measured gap to the exhaustive optimum** — the 10-unit granularity makes the full 2^10
   space enumerable, so optimality is measured instead of assumed;
4. **Batch-size and GPU awareness** — the same procedure on two GPUs and three batch sizes,
   showing whether and how the selected configuration shifts.

> Safe wording: *"A lightweight, accuracy-constrained mixed-precision selection framework using
> sensitivity-guided search, evaluated against random and exhaustive search across batch sizes
> and GPUs."*

## 4. System overview

*(Filled in as modules land.)*

```
pretrained ResNet-18 ──► 10-unit view ──► calibration (INT8 ranges)
                                   │
          ┌────────────────────────┴───────────────────────┐
  accuracy oracle (fake-quant, val split)     latency oracle (per-unit LUT, per GPU × batch)
          └────────────────────────┬───────────────────────┘
                     sensitivity scan ──► scoring ──► greedy (+ local search)
                                   │
            chosen config ──► held-out test accuracy + real TensorRT latency
```

## 5. Method

### 5.1 Model and unit grouping (`src/precisionpilot/model.py`)
The model is the torchvision ResNet-18 with ImageNet weights, with its FC layer replaced by a
10-class head and fine-tuned on CIFAR-10 upscaled to 224×224. `UnitResNet` views it as **10
units**, each of which receives one precision:

| # | Unit | Contents | Input shape (batch 1) |
|---|---|---|---|
| 0 | `stem` | conv1 7×7/2 + BN + ReLU + maxpool | 3×224×224 |
| 1–2 | `layer1.0`, `layer1.1` | BasicBlock, 64 ch | 64×56×56 |
| 3–4 | `layer2.0`, `layer2.1` | BasicBlock, 128 ch (2.0 has a downsample) | 64×56×56 / 128×28×28 |
| 5–6 | `layer3.0`, `layer3.1` | BasicBlock, 256 ch (3.0 has a downsample) | 128×28×28 / 256×14×14 |
| 7–8 | `layer4.0`, `layer4.1` | BasicBlock, 512 ch (4.0 has a downsample) | 256×14×14 / 512×7×7 |
| 9 | `head` | global avgpool + FC 512→10 | 512×7×7 |

A configuration is a 10-bit string, written unit 0 first (`0` = FP16, `1` = INT8). For
example, `0111111110` keeps the stem and head in FP16. That gives 2^10 = 1,024
configurations, few enough to enumerate exhaustively as ground truth.

### 5.2 Fine-tuning (`src/precisionpilot/train.py`)
- SGD with Nesterov momentum 0.9, lr 0.01, weight decay 5e-4, label smoothing 0.1 and batch
  size 128.
- 5% linear warm-up, then a cosine decay, for 8 epochs. AMP (FP16 autocast + GradScaler) is
  used on CUDA.
- Augmentation: random crop with 4 px reflect padding and horizontal flip at 32×32, then a
  bilinear upscale to 224 and ImageNet normalisation.
- Augmentation and the upscale run **on the GPU, per batch**. The dataset sits in memory as
  uint8 arrays, so no DataLoader worker processes are needed. That avoids Windows `spawn`
  problems and the CPU resizing bottleneck.
- Training is monitored on the `spare` split, which comes from CIFAR train. Validation and test
  never influence training decisions.
- Training runs on a Kaggle T4 (`notebooks/kaggle_01_finetune.ipynb`).

### 5.3 Simulated INT8 quantization (`quant/`, `precision.py`)
- **BN folding.** Every Conv+BN pair is folded into a single conv with a bias before
  quantization, as TensorRT does. The folded network matches the original to within ~1e-6.
- **Weights.** Per output channel, symmetric, zero point 0, with scale
  `s_c = max|W_c| / 127`. The scales are precomputed once, so the ONNX export carries
  *constant* Q/DQ scales; TensorRT rejects scales computed inside the graph.
- **Activations.** Per tensor, symmetric, with `s = amax / 127`. `amax` is calibrated on
  1,000 calibration images, using `percentile` (the default: mean over batches of the
  99.99th percentile of |x|) or `minmax`.
- **Integer range.** [-128, 127], via `torch.fake_quantize_per_{tensor,channel}_affine`.
  These ops export to ONNX `QuantizeLinear`/`DequantizeLinear` with an int8 zero point.
- **Quantizer placement.** Quantizers sit on the inputs of every conv/linear, plus one on the
  residual branch of each BasicBlock so TensorRT can fuse the add into the INT8 conv.
  Quantizers that see the same tensor with the same scale are merged by the exporter.
- **Calibration is configuration-independent.** Ranges are measured on the FP32 graph, as
  TensorRT calibration does, so a single calibration serves all 1,024 configurations.
- **Mixed-precision execution.** `MixedPrecisionModel` pre-builds an FP16 copy and a
  fake-INT8 copy of every unit, so switching configuration costs nothing.
  - An FP16 unit runs with half-precision weights and activations.
  - An INT8 unit takes an FP32 input, runs fake-quantized math and emits an FP16 output.
    This mirrors TensorRT, where INT8 kernels accumulate in INT32 and hand FP16 tensors to
    their FP16 neighbours.
- **CPU-only FP16 emulation.** CPU half-precision convolutions are very slow, so CPU runs
  (smoke runs and tests) use FP32 compute with FP16-rounded weights and unit outputs.
  CUDA and MPS run native FP16.

### 5.4 ONNX export for TensorRT (`export/onnx_export.py`)
`export_config` writes the full model for any configuration: plain FP32 graph for the FP16
units, with Q/DQ pairs around the INT8 units. `export_unit` writes a single unit at a fixed
input shape. Building with TensorRT's FP16 and INT8 flags then gives exactly the requested
per-unit precisions (*explicit quantization*, the TensorRT 10 path).

### 5.5 Latency measurement (`latency/`)
- **Timing protocol.** Warm-up runs, then N timed runs with one CUDA-event pair per run. The
  median and p95 are reported, plus throughput (images/s at the median), with an optional
  cooldown between measurements.
- **Backends:**
  - `trt`: TensorRT 10. Engines are built from the ONNX exports and cached in
    `.engine_cache/`, with buffers allocated by PyTorch.
  - `torch`: CUDA FP16 sanity check. Its INT8 timings are only the simulation and are never
    reported as INT8 latency.
  - `mock`: deterministic analytic model for machines without CUDA, with Ampere-like and
    Turing-like profiles. It models FLOPs/throughput, launch overhead, INT8 fixed costs, a
    fusion discount and a reformat cost at FP16↔INT8 boundaries, so the additive estimate
    has a realistic, non-zero error.
- **Lookup table (`lut.py`).** Each unit is measured alone in FP16 and INT8 for each batch
  size, and a config's latency is estimated as the sum of its units' median latencies:
  `L̂(c) = Σ_u t_u(c_u, b)`. `verify_configs` measures selected configurations end to end and
  reports `error % = (L̂ − L) / L`.

### 5.6 Search (`search/`)
All methods share an `Evaluator`:
- It measures accuracy on validation data only.
- It counts **distinct** accuracy evaluations; repeats are cached and not charged.
- It takes latency from the lookup table, which costs no evaluation.
- It checks the budget: `drop = acc_FP32 − acc(c) ≤ budget`.

The methods:
1. **Sensitivity scan** (`sensitivity.py`). Evaluate all-FP16 and each one-unit-INT8
   configuration: 1 + 10 = 11 evaluations.
2. **Scoring.** `score_u = Δlatency_u / max(Δacc_u, ε)` with ε = 0.01 pp. Units whose INT8
   version is not faster get −∞ and are never flipped. Ties break by larger saving, then by
   unit index.
3. **Greedy** (`greedy.py`, the proposed method). Start from all-FP16 and flip units in score
   order. A flip that breaks the budget is backed off, then the search either continues with
   the next unit (`greedy-skip`, the main variant) or stops (`greedy-stop`). The cost is at
   most 11 + 10 evaluations.
4. **Local search** (`local_search.py`, optional). Neighbours of the current config are single
   additions and pairwise swaps. Only neighbours with lower estimated latency are evaluated,
   fastest first, and the first feasible improvement is accepted. The result is never worse
   than the start.
5. **Random search** (`random_search.py`). Distinct uniformly random configurations, given the
   **same number of evaluations** as greedy; it returns the fastest feasible one, or falls
   back to all-FP16.
6. **Exhaustive search** (`exhaustive.py`). All 1,024 configurations; this is the ground-truth
   optimum. Also provides `gap_to_optimum` and the Pareto front.

The chosen configuration's held-out test accuracy is measured afterwards and never feeds back
into selection.

## 6. Experimental setup

### 6.1 Fixed rules (decided before any results)
| Rule | Value | Where |
|---|---|---|
| Main accuracy budget | **1.0 percentage point** top-1 drop vs FP32, on the validation split | `configs/default.yaml → budget.main_pp` |
| Budget sweep | 0.5, 1.0, 2.0 pp | `budget.sweep_pp` |
| Batch sizes | 1, 8, 32 | `latency.batch_sizes` |
| Timing | 20 warm-up runs, 100 timed runs, median + p95 reported, 1 s cooldown | `latency.*` |
| Random seed | 42 | `seed` |
| Selection data | validation split only; the test split is used only for final reporting | — |

### 6.2 Data splits
The splits are stratified by class, use seed 42, and are stored in `data/splits.json`, which
is committed so that every machine uses identical indices (`src/precisionpilot/data.py`).

| Split | Source | Size | Used for |
|---|---|---|---|
| `finetune` | CIFAR-10 train | 48,000 | fine-tuning |
| `calib` | CIFAR-10 train | 1,000 (100/class) | INT8 activation calibration |
| `spare` | CIFAR-10 train | 1,000 | monitoring during fine-tuning |
| `val` | CIFAR-10 test | 5,000 | sensitivity scan (Experiment 1) |
| `val_subset` | first 1,000 of `val` (stratified) | 1,000 | all searches + the exhaustive accuracy table |
| `test` | CIFAR-10 test | 5,000 | **final reported numbers only** |

`val` and `test` are disjoint, and `calib`/`spare` are excluded from fine-tuning.

### 6.3 Hardware and software
*(Exact versions recorded from `env-check` output once GPU runs happen.)*

| Machine | GPU | Role |
|---|---|---|
| Windows laptop | NVIDIA RTX 3050 Ti Laptop, 4 GB, Ampere (sm_86) | Primary latency benchmark |
| Kaggle | NVIDIA Tesla T4, 16 GB, Turing (sm_75) | Training, exhaustive accuracy table, second GPU |
| MacBook | Apple Silicon (MPS) | Development, tests, demo |

## 7. Experiments and results

| # | Experiment | What it shows | Status |
|---|---|---|---|
| 1 | Per-unit sensitivity scan | Which units are fragile | pending |
| 2 | Search cost vs result quality | Evaluations vs final latency (main chart) | pending |
| 3 | Accuracy budget sweep (0.5/1/2 pp) | Config change with budget | pending |
| 4 | Batch-size variation (1/8/32) | Config change with batch size | pending |
| 5 | Second GPU (T4) | Hardware dependence | pending |
| 6 | Latency estimation accuracy | LUT vs real TensorRT | pending |

## 8. Limitations

*(Collected throughout the project.)*
- Accuracy is measured with **simulated** INT8. The real TensorRT kernels can differ slightly,
  for example in rounding and fused-epilogue behaviour. TensorRT test accuracy of the chosen
  configurations is reported alongside the simulated numbers to quantify this.
- The latency of a configuration is **estimated** additively from per-unit measurements. Real
  engines fuse across unit boundaries and pay reformat costs at precision changes, so the
  estimate's error is measured and reported (Experiment 6).
- On machines without CUDA (the Mac), latency comes only from the `mock` analytic backend.
  Those numbers check that the code works and are never reported as results.

## 9. Reproduction guide

See [README.md](README.md) for environment setup on macOS, Windows and Kaggle. Always run
`python -m precisionpilot env-check` first.

## 10. Testing

- Framework: `pytest`, CPU-only by default. Tests marked `gpu` / `trt` are skipped
  automatically when CUDA / TensorRT is unavailable.
- CI: GitHub Actions on Ubuntu, Windows and macOS (`.github/workflows/tests.yml`), also running
  `ruff` lint.

- Local runs are limited to **2 CPU threads** (`tests/conftest.py`, `utils.limit_threads`).

| Test file | Covers |
|---|---|
| `tests/test_config_utils.py` | config loading, overrides, unknown-key rejection, path resolution, seeding, device selection, JSON IO, `env-check` CLI |
| `tests/test_data.py` | split sizes, disjointness and coverage, determinism, stratification, persistence; batching; preprocessing (resize, normalisation, dtype); seeded augmentation |
| `tests/test_model_train.py` | 10-unit view equals torchvision forward and shares parameters; per-unit input shapes; accuracy function; FP16 copy; end-to-end smoke training + checkpoint reload + baselines |
| `tests/test_quant.py` | config encoding; fake-quant error ≤ scale/2, idempotence, grid and clipping; per-channel scales; minmax/percentile calibration; exact BN folding; mixed-model calibration, FP16/INT8 error ordering, config switching; ONNX Q/DQ placement and constant scales |
| `tests/test_sensitivity.py` | distinct-evaluation counting, sensitivity scan cost (N+1) and drops, scoring order, exclusion of units that are slower in INT8 |
| `tests/test_latency.py` | timing statistics, mock backend determinism and profiles, LUT build / estimate / save / load, verification error; `gpu`/`trt`-marked real-hardware tests (TensorRT engine build, engine cache, TRT vs simulated accuracy) |
| `tests/test_search.py` | 25 random synthetic problems with interacting units: greedy is always feasible, exhaustive equals brute force, local search never makes things worse; greedy optimal on additive problems; skip vs stop; fallbacks; random-search reproducibility and evaluation budget; neighbourhood; gap and Pareto front |

## 11. Project log

| Phase | Content | Status |
|---|---|---|
| 0 | Scaffold, packaging, config, CLI, env-check, CI, literature check | done |
| 1 | Data splits, model, FP32/FP16 baselines, Kaggle fine-tuning | done (training run pending on Kaggle) |
| 2 | Simulated INT8, calibration, sensitivity scan, ONNX Q/DQ export | done |
| 3 | Latency harness, backends, lookup table, TensorRT verification | done (TensorRT runs pending on GPU machines) |
| 4 | Greedy / random / exhaustive / local search | done |
| 5 | Experiments 1–6, Kaggle T4 benchmark | — |
| 6 | Plots and summary tables | — |
| 7 | Web demo | — |
| 8 | Final documentation | — |

## 12. References

1. B. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018.
2. R. Krishnamoorthi. *Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper.* arXiv:1806.08342, 2018.
3. H. Wu, P. Judd, X. Zhang, M. Isaev, P. Micikevicius. *Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation.* arXiv:2004.09602, 2020.
4. B. Wu et al. *Mixed Precision Quantization of ConvNets via Differentiable Neural Architecture Search.* arXiv:1812.00090, 2018.
5. K. Wang, Z. Liu, Y. Lin, J. Lin, S. Han. *HAQ: Hardware-Aware Automated Quantization with Mixed Precision.* CVPR 2019.
6. Z. Dong, Z. Yao, A. Gholami, M. Mahoney, K. Keutzer. *HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision.* ICCV 2019.
7. Z. Dong et al. *HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks.* NeurIPS 2020.
8. Z. Yao et al. *HAWQ-V3: Dyadic Neural Network Quantization.* ICML 2021.
9. Y. Cai et al. *ZeroQ: A Novel Zero Shot Quantization Framework.* CVPR 2020.
10. Y. Li et al. *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction.* ICLR 2021.
11. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
12. A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* Tech. report, 2009 (CIFAR-10).
13. NVIDIA. *TensorRT Developer Guide — Working with Quantized Types (explicit quantization, Q/DQ).*
