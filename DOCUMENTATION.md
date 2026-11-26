# PrecisionPilot — Project Documentation

*Living technical documentation and report source. Updated at the end of every project phase.*

**Status:** Phase 0 complete (scaffold, environment, literature check).

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

*(Detailed in Phases 1–4.)* Summary of the planned design:
- **Model grouping:** stem, 8 residual BasicBlocks, head → 10 units → 1,024 configurations.
- **Accuracy:** PyTorch simulated (fake) INT8 quantization per unit; weights per-channel
  symmetric, activations per-tensor calibrated.
- **Latency:** per-unit TensorRT latency in FP16 and INT8 per (GPU, batch size) → lookup
  table; additive estimate verified against full-model TensorRT engines.
- **Search:** sensitivity scan → score = latency saved / accuracy lost → greedy flips → final
  test check → optional pairwise-swap local search.

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
*(Finalised in Phase 1.)* CIFAR-10 train (50k) → fine-tuning / calibration (1k) / spare (1k);
CIFAR-10 test (10k) → validation (5k; 1k search subset) / test (5k).

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

## 9. Reproduction guide

See [README.md](README.md) for environment setup on macOS, Windows and Kaggle. Always run
`python -m precisionpilot env-check` first.

## 10. Testing

- Framework: `pytest`, CPU-only by default. Tests marked `gpu` / `trt` are skipped
  automatically when CUDA / TensorRT is unavailable.
- CI: GitHub Actions on Ubuntu, Windows and macOS (`.github/workflows/tests.yml`), also running
  `ruff` lint.

| Test file | Covers |
|---|---|
| `tests/test_config_utils.py` | config loading, overrides, unknown-key rejection, path resolution, seeding, device selection, JSON IO, `env-check` CLI |

## 11. Project log

| Phase | Content | Status |
|---|---|---|
| 0 | Scaffold, packaging, config, CLI, env-check, CI, literature check | done |
| 1 | Data splits, model, FP32/FP16 baselines, Kaggle fine-tuning | — |
| 2 | Simulated INT8, calibration, sensitivity scan, ONNX Q/DQ export | — |
| 3 | Latency harness, backends, lookup table, TensorRT verification | — |
| 4 | Greedy / random / exhaustive / local search | — |
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
