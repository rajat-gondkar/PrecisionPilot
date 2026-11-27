# PrecisionPilot — Project Documentation

*Living technical documentation and report source. Updated at the end of every project phase.*

**Status:** Phases 0–7 implemented. Fine-tuning (§7.4), the accuracy stage (§7.5) and a first full Tesla T4 TensorRT study (§7.6, additive latency model only) are done. The context-aware latency model (§5.5) and the memory study (§7.7) are implemented; their T4 re-run and the RTX 3050 Ti run are pending.

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
    `.engine_cache/`, with buffers allocated by PyTorch. Build details that matter:
    - The FP16 builder flag is always on. The **INT8 flag is set only when the graph contains
      Q/DQ nodes**: with the flag on but no Q/DQ nodes and no calibrator, TensorRT 10 refuses to
      build ("Calibration failure occurred with no scaling factors detected"). This was found
      on the first Kaggle run and fixed.
    - Engine inputs and outputs are **FP16 (linear)**, matching the FP16 tensors that flow
      between units in the simulation, so a unit engine does not pay FP32↔FP16 casts at its
      boundaries (which would inflate the summed per-unit latencies). If TensorRT rejects FP16
      I/O on the very first build, the backend falls back to FP32 I/O for the whole run and
      records it in the lookup table's `notes` column (`trt=<version> io=fp16|fp32`).
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

#### 5.5.0 Memory metrics
Four memory numbers are reported for every configuration:
- `model_mb`: **weights**, computed exactly as 2 bytes per parameter for FP16 units and 1 byte for
  INT8 units (BatchNorm folded; per-channel INT8 scales add about 0.1% and are ignored). It is
  hardware independent and is stored with every search result.
- `engine_mb`: size of the serialized TensorRT engine (weights plus engine metadata).
- `activation_memory_mb`: TensorRT's per-execution-context scratch memory for activations,
  which scales with batch size.
- `gpu_footprint_mb`: **measured** drop in free GPU memory after deserializing the engine,
  creating its context, allocating the I/O buffers and running it once, taken with
  `torch.cuda.mem_get_info`. This is what the model occupies on the device on top of the CUDA context.

#### 5.5.1 Context-aware latency model (`latency/context.py`)
The additive lookup table assumes a unit costs the same in a full engine as alone. The first T4 run
showed it does not (§7.6): a unit timed alone pays an FP16↔INT8 conversion at both ends, while in a
real engine neighbouring INT8 units keep the tensor in INT8 and share one conversion. The table
therefore overestimates INT8-heavy configurations by up to +107%.

The context-aware model keeps the table's structure and adds boundary costs. With virtual FP16
neighbours before the first and after the last unit, for a configuration `c` (`c_u = 1` for INT8):

`L(c) = Σ_u [ (1 − c_u)·f_u + c_u·g_u ] + Σ_k [ c_(k−1) ≠ c_k ]·h_k`

where `f_u`, `g_u` are the unit's costs in FP16 / INT8 *inside a full engine* and `h_k`
(k = 0 … N) is the cost of an FP16↔INT8 boundary at position k. A run of INT8 units pays
conversions only at its two ends. This is a linear model with 3N + 1 = 31 parameters per batch size,
fitted by **ridge regression** to full-engine latencies:
- Fitting set per batch size (56 configurations with the default 20 random ones): all-FP16, all-INT8,
  every single-unit flip in both directions, every INT8 prefix and suffix, and 20 seeded random
  configurations.
- Ridge prior: the isolated-unit table (`f`, `g`) with zero boundary costs, so scarce data degrades
  gracefully to the additive table.
- It models latency only. Accuracy still comes from real evaluations.
- Quality is judged on **held-out** configurations: verification configurations that were not in the
  fitting set are reported separately (`ctx_trained` column in `verify.csv`).
- On the mock backend (a synthetic world with the same kind of boundary effects) the held-out mean
  error falls from 2.0% to 0.2% (`tests/test_context.py`). Real numbers come from the T4 re-run.

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
3b. **Dynamic greedy** (`greedy.py: greedy_dynamic`, the method used with the context-aware model).
   The static greedy ranks units once by their *isolated* saving, which goes stale when a unit's
   benefit depends on its neighbours (an INT8 stem only pays off once `layer1.0` is INT8 too). The
   dynamic variant re-scores every FP16 unit at each step by its **marginal** latency gain in the
   *current* configuration, read from the latency model (free, no evaluation), divided by its
   accuracy cost from the sensitivity scan: `gain_u / max(drop_u, ε)`. The best unit is tried and
   kept if the budget still holds, otherwise banned. If no single flip gains anything it looks one
   step ahead at adjacent **pairs** (a run of two INT8 units pays the conversion once, not twice).
   It stops when no unit or pair has a positive gain. With an additive latency model the order is
   identical to the static greedy (checked by tests).
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
| Budget sweep (pre-registered) | 0.5, 1.0, 2.0 pp | `budget.sweep_pp` |
| Supplementary budgets (**post-hoc**) | 0.1, 0.2, 0.3 pp | `budget.extra_pp` |
| Batch sizes | 1, 8, 32 | `latency.batch_sizes` |
| Timing | 20 warm-up runs, 100 timed runs, median + p95 reported, 1 s cooldown | `latency.*` |
| Random seed | 42 | `seed` |
| Selection data | validation split only; the test split is used only for final reporting | — |

**Why there are supplementary budgets.** The pre-registered budgets and the main budget were set
before any result existed and are unchanged. After the first accuracy run showed that *every*
configuration is within budget at 1 pp and 2 pp (§7.5), the accuracy constraint cannot influence
the choice there. Three tighter budgets (0.1, 0.2, 0.3 pp) were therefore **added afterwards**, so
the constrained search can actually be tested. They are tagged `budget_kind = supplementary` in
every results table and are always reported separately from the pre-registered ones; nothing
pre-registered was removed or changed.

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

### 7.1 How the experiments are run (`src/precisionpilot/experiments/`)
Accuracy does not depend on the GPU, but latency does. The work is therefore split into an
accuracy stage that runs once and a latency stage that runs on every GPU:

| Stage | Module | Runs on | Outputs |
|---|---|---|---|
| `run-accuracy` | `experiments/accuracy.py` | Kaggle T4 | `results/accuracy/reference.json` (FP32 / all-FP16 / all-INT8 on val_subset, val and test, plus seconds per evaluation), `accuracy_table.csv` (all 1,024 configs on val_subset), `sensitivity.csv` (Experiment 1, full val) |
| `run-gpu` | `experiments/pipeline.py`, `search_suite.py`, `verify.py` | RTX 3050 Ti and T4 | `results/<gpu>/lut.csv`, `ctx_train.csv` + `ctx_model.csv` (context-aware latency model), `search_results.csv`, `random_curve.csv`, `verify.csv`, `engine_accuracy.csv`; `results/accuracy/final_test.csv` |
| `report` | `analysis/report.py` | anywhere | `results/figures/*.png`, `results/tables/*.md / .csv` |

**Exhaustive accuracy table (prefix-tree evaluation).**
- Configurations that share their first *k* unit precisions also share the activations after
  unit *k*. A depth-first walk over the binary prefix tree therefore costs
  2 + 4 + … + 2^10 = 2,046 unit forwards per batch, instead of 10 × 1,024 = 10,240.
- The walk and the normal forward pass use the same `run_unit` code path, so the table gives
  **exactly** the accuracy a direct evaluation would; `tests/test_experiments.py` checks this.

**How search cost is charged.**
- Every search replays accuracies from this table, but is still charged one evaluation per
  distinct configuration it queries.
- Reported search time is *evaluations × measured seconds per evaluation* (`sec_per_eval` in
  `reference.json`).
- The searches make exactly the same decisions they would make running the model live.

**Two latency models.** Every method is run once under the additive lookup table and once under
the context-aware model; rows carry a `latency_model` column (`additive` | `context`) and the main
figures use `context` when present. Search accuracy decisions are identical in both; only the latency
model differs, which isolates its effect. The context model's fitting needs about 56 extra full-engine
builds per batch size (`latency.context_random` controls the random part, `latency.context_model`
switches the whole step off).

**Search suite (Experiments 2–5).**
- For each GPU × batch size {1, 8, 32} × budget {0.5, 1, 2 pp pre-registered; 0.1, 0.2, 0.3 pp
  supplementary}, the suite runs: all-FP16,
  all-INT8, greedy-skip (static), greedy-stop, greedy-skip + local search, greedy-dyn (dynamic),
  random search (20 seeds,
  each with the same number of evaluations as greedy-skip), and exhaustive search.
- It records the configuration, validation accuracy, drop vs FP32, estimated latency, speedup
  over all-FP16, gap to the optimum, number of evaluations and estimated search time.
- `random_curve.csv` records random search's mean, std and median gap to the optimum at 5 …
  1,024 evaluations; this feeds the main chart.

**Verification (Experiment 6).**
- For each batch size, the configurations worth reporting are built as full TensorRT engines
  and timed. These are all-FP16, all-INT8, and greedy, greedy + local search, exhaustive and
  the median random result at each budget.
- Measured median and p95 latency, throughput, engine size and activation memory are compared
  with the lookup-table estimate.
- The same engines are also run on the test split, giving **real** INT8 accuracy
  (`engine_accuracy.csv`) alongside the simulated test accuracy (`final_test.csv`).

### 7.2 Figures (`src/precisionpilot/analysis/`)
| Figure | File | Shows |
|---|---|---|
| 1 | `fig1_sensitivity.png` | Accuracy drop per unit when only that unit is INT8 |
| 2 | `fig2_cost_vs_quality_<gpu>_bs<b>_b<budget>.png` | **Main result:** latency gap to optimum vs number of evaluations; random curve (mean ± std) vs greedy variants vs exhaustive |
| 3 | `fig3_acc_vs_latency_<gpu>_bs<b>_b<budget>.png` | All 1,024 configurations, the budget line, baselines and the chosen configurations |
| 4 | `fig4_heatmap_<gpu>.png`, `fig4_heatmap_optimum_<gpu>.png` | INT8 vs FP16 per unit, for every batch size × budget (greedy and optimum) |
| 5 | `fig5_heatmap_gpus.png` | Greedy selections side by side across GPUs |
| 6 | `fig6_speedups.png`, `fig6_speedups_b<budget>.png` | Estimated speedup over all-FP16 by batch size and GPU (main budget, and one per supplementary budget) |
| 7 | `fig7_lut_error.png` | Latency estimate vs measured: additive table and (held-out) context-aware model |
| 8 | `fig8_memory_<gpu>.png` | Memory: model size, and measured GPU memory per batch size, for all-FP16 vs the selected mixed configuration vs all-INT8 |
| 9 | `fig9_latency_models_<gpu>.png` | **Measured** latency of what each latency model selected (additive greedy vs context-aware greedy) next to all-FP16 and all-INT8 |

Figures 2, 3 and 6 are produced for the main budget and for every supplementary budget.

Tables: `summary.md/csv` (every method × GPU × batch × budget × latency model, with random search
averaged over seeds), `sensitivity.md`, `verify.md`, `final_test.md`, `memory.md/csv`,
`latency_model_comparison.md/csv` (measured latency of every selection under each model) and
`latency_model_error.md/csv` (estimate error of both models).

Figure style:
- A fixed colour per method: the proposed greedy method is blue, random search orange and
  exhaustive search aqua. Baselines are neutral gray with direct labels.
- The heatmap uses a light and a dark step of a single blue ramp, with the text "FP16" or
  "INT8" in every cell, so it doesn't rely on colour alone.

### 7.3 Status
| # | Experiment | What it shows | Status |
|---|---|---|---|
| 1 | Per-unit sensitivity scan | Which units are fragile | **done** (§7.5) |
| 2 | Search cost vs result quality | Evaluations vs final latency (main chart) | **done on T4** (§7.6) |
| 3 | Accuracy budget sweep (0.5/1/2 pp, plus supplementary 0.1/0.2/0.3) | Config change with budget | **done on T4** (§7.6) |
| 4 | Batch-size variation (1/8/32) | Config change with batch size | **done on T4** (§7.6) |
| 5 | Second GPU | Hardware dependence | T4 done; RTX 3050 Ti pending |
| 6 | Latency estimation accuracy | LUT vs real TensorRT; additive vs context-aware model | additive **done on T4** (§7.6); context-aware pending re-run |
| 7 | Memory: FP16 vs mixed vs INT8 | Model size, activation and GPU memory | first numbers from T4 (§7.7); measured footprint pending re-run |

### 7.4 Result: fine-tuning and FP32/FP16 baselines (Kaggle Tesla T4)
Training ran for 8 epochs (3,000 steps) with AMP and took **426 s** (~52 s per epoch). The log
is in `checkpoints/resnet18_cifar10.json`.

| Epoch | Train loss | Train acc (%) | Spare acc (%) |
|---|---|---|---|
| 1 | 0.962 | 81.66 | 95.9 |
| 2 | 0.679 | 94.44 | 95.6 |
| 3 | 0.633 | 96.27 | 95.7 |
| 4 | 0.606 | 97.44 | 96.6 |
| 5 | 0.585 | 98.35 | 96.7 |
| 6 | 0.574 | 98.73 | 97.3 |
| 7 | 0.566 | 99.05 | 97.2 |
| 8 | 0.563 | 99.18 | 97.1 |

(Training loss includes label smoothing of 0.1, which is why it stays well above zero.)

| Split | FP32 top-1 (%) | All-FP16 top-1 (%) | n |
|---|---|---|---|
| Validation | **95.86** | 95.88 | 5,000 |
| Test | **96.48** | 96.46 | 5,000 |

The target of at least 95% is met. All-FP16 matches FP32 to within ±0.02 pp, so FP16 is a
safe reference precision and the accuracy budget is effectively spent on INT8 alone.
Source: `results/baselines/baselines_t4.json`.

### 7.5 Result: INT8 accuracy on the Tesla T4 (Experiment 1 and the exhaustive table)
Source: `results/accuracy/` from the first Kaggle benchmark run (`run-accuracy`, 177 s).

| Precision | val_subset (n=1,000) | val (n=5,000) | test (n=5,000) |
|---|---|---|---|
| FP32 | 96.50 | 95.86 | 96.48 |
| all-FP16 | 96.50 | 95.88 | 96.48 |
| all-INT8 (simulated) | 96.20 | 95.68 | 96.64 |

**Per-unit sensitivity** (full validation split, one unit INT8, rest FP16, drop vs all-FP16):

| Unit | stem | layer1.0 | layer1.1 | layer2.0 | layer2.1 | layer3.0 | layer3.1 | layer4.0 | layer4.1 | head |
|---|---|---|---|---|---|---|---|---|---|---|
| Drop (pp) | +0.06 | +0.12 | −0.02 | +0.04 | +0.02 | 0.00 | +0.04 | −0.04 | +0.04 | 0.00 |

**Findings so far (accuracy side only, latency pending):**
- INT8 is almost free for this model. Every single-unit drop is at most 0.12 pp, and all-INT8 loses
  0.30 pp on `val_subset` (0.18 pp on `val`) and *gains* 0.16 pp on `test`.
- The usual expectation that the first and last units are the fragile ones is **not** observed
  here: the stem and head drops are 0.06 pp and 0.00 pp. With percentile-calibrated
  per-channel PTQ, ResNet-18 on CIFAR-10 is robust at every unit.
- The accuracy budget barely binds. Configurations within budget out of 1,024:

  | Budget (pp) | 0 | 0.1 | 0.2 | 0.3 | **0.5** | **1.0** | **2.0** |
  |---|---|---|---|---|---|---|---|
  | Feasible configs | 351 | 527 | 635 | 837 | 1,009 | 1,024 | 1,024 |

  At the pre-registered budgets of 1 pp and 2 pp **every** configuration is feasible, so the
  optimum there is simply the lowest-latency configuration and accuracy plays no role. Only 0.5 pp
  excludes anything (15 configurations). This is why the supplementary budgets were added (§6.1).
- The noise floor is high. On `val_subset` one image is 0.1 pp, and the standard error of a
  1,000-image accuracy near 96.5% is about 0.6 pp. Differences between configurations of a few
  tenths of a point are therefore within sampling noise, which is also why some configurations
  score *above* FP32.

Implication for the study: the sensitivity-guided method should still reach the same latency as
the exhaustive optimum cheaply, but the accuracy-constrained part of the claim can only be
tested at budgets of 0.3 pp and below. This is reported as a finding; the pre-registered budgets
are unchanged and the tighter ones are added separately and labelled (§6.1).

### 7.6 Result: Tesla T4 TensorRT study (Experiments 2–6)
Source: `results/t4/`, `results/figures/`, `results/tables/`. TensorRT 10.16 on a Kaggle Tesla T4,
FP16 engine I/O (`notes` column of `lut.csv`: `trt=10.16.1.11 io=fp16`). The `run-gpu` stage took
1,007 s. Accuracy-side numbers are from §7.5 and are hardware independent; the exhaustive table and
sensitivity files were **byte-identical** across two separate Kaggle runs, so the pipeline is
deterministic.

**Per-unit latency (median ms, T4, one unit alone):**

| Unit | bs 1 FP16 | bs 1 INT8 | bs 8 FP16 | bs 8 INT8 | bs 32 FP16 | bs 32 INT8 |
|---|---|---|---|---|---|---|
| stem | 0.058 | 0.070 | 0.253 | 0.343 | 1.330 | 1.824 |
| layer1.0 | 0.041 | 0.051 | 0.209 | 0.176 | 1.087 | 0.735 |
| layer1.1 | 0.041 | 0.055 | 0.213 | 0.175 | 1.126 | 0.909 |
| layer2.0 | 0.047 | 0.056 | 0.154 | 0.128 | 0.689 | 0.444 |
| layer2.1 | 0.043 | 0.053 | 0.156 | 0.125 | 0.544 | 0.438 |
| layer3.0 | 0.048 | 0.066 | 0.133 | 0.111 | 0.467 | 0.317 |
| layer3.1 | 0.049 | 0.062 | 0.145 | 0.106 | 0.599 | 0.395 |
| layer4.0 | 0.090 | 0.098 | 0.158 | 0.137 | 0.413 | 0.266 |
| layer4.1 | 0.104 | 0.118 | 0.189 | 0.143 | 0.642 | 0.293 |
| head | 0.034 | 0.037 | 0.034 | 0.038 | 0.038 | 0.043 |

- **Batch size 1:** INT8 is slower than FP16 for *every* unit (+8% to +38%). Per-kernel launch and
  Q/DQ overhead dominate; there is no compute to save.
- **Batch sizes 8 and 32:** INT8 is faster for all eight residual blocks (down to 0.46× the FP16
  time for `layer4.1` at batch 32), but still **slower for the stem and the head**, at every batch size.

**What the search selected (using the lookup table):**

| Batch | Budget | Selected configuration | Note |
|---|---|---|---|
| 1 | all budgets | `0000000000` (all-FP16) | INT8 never faster, so the selector correctly keeps FP16 and spends only the 11 sensitivity evaluations (outline §14) |
| 8, 32 | 0.1 pp | greedy `0011111110`; optimum by the table `0111101111` | greedy is 0.6% (bs 8) / 3.7% (bs 32) slower than the table's optimum |
| 8, 32 | ≥ 0.2 pp | `0111111110` (stem and head stay FP16), identical to the table's optimum | 18 evaluations, vs 1,024 for exhaustive search |

So **the selected configuration depends on batch size** (none vs eight INT8 units), and on the budget
at the tightest setting. Greedy needs 18 evaluations (11 for the sensitivity scan plus 7), or only 11 at
batch size 1 where nothing is flipped. Random search given the same number of evaluations ends up
4.5% to 11.7% slower than the optimum on average, depending on batch size and budget (the `random`
rows of `results/t4/search_results.csv`). It needs about 200 evaluations to get within ~3% and still
has a 65% chance of missing the optimum at 200 (`random_curve.csv`, bs 32, 0.2 pp). Greedy at 0.2 pp
and above matches the table's optimum at every batch size. At 0.1 pp the pairwise-swap local search
does **not** help much: it leaves bs 8 unchanged (still 0.58% off, 22 evaluations) and moves bs 32
from 3.70% to 3.66% off (25 evaluations).

**Experiment 6: how good is the additive latency estimate?** (18 configurations × batch sizes,
`results/t4/verify.csv`, `fig7_lut_error.png`)

| Batch | Mean error | Range |
|---|---|---|
| 1 | −17.4% | −28.9% … +0.6% |
| 8 | +4.2% | −15.3% … +47.5% |
| 32 | +28.3% | +10.6% … **+107.0%** |

Median |error| is 15.6%, mean 20.0%, max 107.0%. The error is largest for INT8-heavy
configurations. A unit measured alone pays an INT8↔FP16 conversion at both ends; in a full engine,
consecutive INT8 units keep the tensor in INT8 and fuse, so the sum **overestimates INT8 cost**
(all-INT8 at batch 32: estimated 5.66 ms, measured 2.74 ms). At batch 1 the estimate is
*below* the measurement (−17% on average); this has not been explained yet.

**Consequence for the selection (measured, full TensorRT engines, T4):**

| Batch | all-FP16 | all-INT8 | greedy @ 0.2 pp (`0111111110`) |
|---|---|---|---|
| 1 | 0.657 ms (1,523 img/s) | 0.661 ms (1,512 img/s) | = all-FP16 |
| 8 | 1.938 ms (4,128 img/s) | **1.005 ms (7,963 img/s)** | 1.229 ms (6,509 img/s) |
| 32 | 6.173 ms (5,184 img/s) | **2.736 ms (11,695 img/s)** | 3.926 ms (8,151 img/s) |

Greedy gives a real speedup over FP16 (1.58× at batch 8, 1.57× at batch 32), but the lookup table
wrongly predicts that INT8 stem/head are costly, so the selected configuration is **22% (bs 8) and
43% (bs 32) slower than all-INT8**, which is within the 0.3 pp budget (its validation drop is exactly
0.30 pp, on the boundary) and had a *higher* held-out test accuracy than FP32 (96.64% vs 96.48%).
At 0.1 pp the table's "optimum" (`0111101111`, 4.34 ms at bs 32) is also measurably slower than
greedy's pick (4.24 ms), so the table's ranking of near-ties is unreliable too. **The search is sound
with respect to its latency model (it matches the table's exhaustive optimum, cheaply); the
limiting factor is the accuracy of the additive latency model, not the search algorithm.**

**Real INT8 accuracy vs the simulation** (`engine_accuracy.csv` vs `final_test.csv`, 5,000 test
images, 18 configurations): the TensorRT engines and the fake-quant simulation agree to within
**±0.08 pp** on every configuration (mean difference −0.002 pp, none above 0.1 pp). The
simulation is therefore a faithful accuracy proxy for the real kernels here. On the test split
no verified configuration is more than 0.02 pp below FP32 (simulated), and the greedy picks are
0.06–0.14 pp *above* it, which is within noise.

### 7.7 Memory: does mixing INT8 and FP16 save memory?
Source: `engine_mb` and `activation_memory_mb` in `results/t4/verify.csv` (first T4 run, TensorRT
10.16). The measured GPU footprint (`gpu_footprint_mb`) and the selections of the context-aware
model are added by the re-run.

| Batch | Configuration | Engine size (MB) | Activation scratch (MB) |
|---|---|---|---|
| 1 | all-FP16 | 22.63 | 2.81 |
| 1 | all-INT8 | 11.55 (−49%) | 0.60 (−79%) |
| 8 | all-FP16 | 22.65 | 19.27 |
| 8 | all-INT8 | 11.57 (−49%) | 4.82 (−75%) |
| 8 | greedy `0111111110` (additive-table pick) | 11.53 (−49%) | 20.87 (**+8%**) |
| 32 | all-FP16 | 22.55 | 77.07 |
| 32 | all-INT8 | 11.54 (−49%) | 19.27 (−75%) |
| 32 | greedy `0111111110` (additive-table pick) | 11.51 (−49%) | 83.49 (**+8%**) |

**Answer.** Yes for the model itself, but not automatically for runtime activation memory:
- **Weights / model size: about 49–50% smaller.** FP16 stores 2 bytes per weight and INT8 1 byte,
  so the engine halves (22.6 MB → 11.5 MB). Any configuration that quantizes the weight-heavy
  residual blocks gets nearly all of this, because the stem (9.4 K parameters) and the head
  (5.1 K) hold under 0.2% of the parameters: the greedy pick that leaves both in FP16 is still 49%
  smaller than all-FP16.
- **Activation memory: all-INT8 is about 75% smaller** (INT8 tensors are one byte instead of two,
  and fusion removes some intermediates), but the mixed pick with an **FP16 stem is 8% *larger* than
  all-FP16**, not smaller. The largest activation tensors are the early ones, and a mixed
  configuration both keeps them at 2 bytes (the stem stays FP16) and adds reformat buffers at every
  FP16↔INT8 boundary. This explanation is an inference from the numbers, not separately verified.
  So if peak activation memory matters, a mixed configuration must also quantize the stem; that is one
  more reason the additive table's reluctance to quantize it was costly.

The re-run adds `gpu_footprint_mb` (what the engine actually occupies on the device) so the report
can show the end-to-end saving rather than the two components.

The whole chain is checked end to end by `python -m precisionpilot pipeline --smoke`. That run
uses synthetic data, random weights and two mock GPU profiles; its numbers only show that the
code works and are not results.

### 7.8 Interactive demo (`src/precisionpilot/demo/`)
`python -m precisionpilot demo` starts a small FastAPI server with a single-page frontend
written in plain HTML, CSS and JavaScript, with no build step.

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | GPUs and batch sizes with results, unit names, reference accuracies, sensitivity |
| `GET /api/select?gpu&batch_size&budget` | Greedy selection for **any** budget (0–10 pp), computed live from the exhaustive accuracy table and that GPU's latency table, with the exhaustive optimum, all-FP16 and all-INT8 for comparison. Measured TensorRT latency and test accuracy are attached where they exist |
| `GET /api/samples` | CIFAR-10 test images |
| `GET /api/predict?sample_id&config` | Top-3 class probabilities from FP32 and from the chosen mixed-precision configuration |
| `POST /api/upload?config` | The same, for an uploaded image (centre-cropped and resized to 32×32) |

The page has three sections:
- **Explore:** GPU, batch-size and budget controls; a strip showing each of the 10 units as
  FP16 or INT8 with its sensitivity; four headline numbers (speedup, latency, accuracy change,
  evaluations used); and a comparison table against the baselines.
- **Predict:** live inference on sample or uploaded images.
- **Method:** a three-step explanation.

Design:
- One token system: a 4 px spacing scale, two type families (Inter and JetBrains Mono), a
  neutral palette with one blue ramp that matches the report figures, one radius and one
  shadow.
- Selected light and dark themes, a responsive grid checked at 390 px and 1280 px, and
  reduced-motion support.
- Precision is always shown as text ("FP16"/"INT8") as well as colour.

On Apple GPUs (MPS), which have no fused fake-quantization kernels, an equivalent hand-written
quantize step is used. A test checks that it matches the native kernel bit for bit.

## 8. Limitations

*(Collected throughout the project.)*
- Accuracy is measured with **simulated** INT8. The real TensorRT kernels can differ slightly,
  for example in rounding and fused-epilogue behaviour. TensorRT test accuracy of the chosen
  configurations is reported alongside the simulated numbers to quantify this.
- The latency of a configuration is **estimated** additively from per-unit measurements. Real
  engines fuse across unit boundaries and pay reformat costs at precision changes, so the
  estimate's error is measured and reported (Experiment 6). On the T4 the error reaches +107%
  (median |error| 15.6%) and measurably misleads the selection: the chosen configuration is
  22–43% slower than all-INT8 at batch sizes 8 and 32 (§7.6).
- The context-aware model is itself an approximation (a linear model with boundary terms); it cannot
  represent effects that depend on more than a unit's immediate neighbours. Its error is measured on
  configurations held out from fitting. Both models are compared on real hardware (§7.6, Experiment 6).
- Memory figures from TensorRT are for one engine at a static batch size; they exclude the CUDA
  context and PyTorch's own allocator state.
- Accuracies for all searches come from a 1,000-image subset (one image = 0.1 pp, standard error
  about 0.6 pp). Differences of a few tenths of a point between configurations are within noise.
- All timing results so far are from a single GPU model (Kaggle T4, shared infrastructure; the
  exact GPU instance can vary between sessions).
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
| `tests/test_experiments.py` | prefix-tree exhaustive table equals direct per-config evaluation; search suite invariants (exhaustive gap 0, nothing beats the optimum, search-time accounting, random curve); chosen configs; summary table; end-to-end smoke pipeline (all result files + figures); CLI |
| `tests/test_quant.py` (export) | also: the exporter never moves the live GPU-resident modules to the CPU (regression test for a bug that would have broken the second batch size of the GPU run); the TensorRT INT8 flag is set exactly when the graph has Q/DQ nodes |
| `tests/test_context.py` | feature construction with virtual FP16 ends; fitting set is unique, structured and deterministic; the model recovers a synthetic boundary-cost world exactly; ridge prior keeps a data-starved fit sane; save/load round trip; on the mock backend the context model has far lower held-out error than the additive table; weight size is 2 / 1 bytes per parameter; memory and latency-model comparison tables; error summary reports only held-out configurations |
| `tests/test_demo.py` | page and static files; `/api/meta`; selection at arbitrary budgets (optimum costs 1,024 evaluations, greedy feasible and never better than the optimum, budget 0 gives no accuracy loss); input validation (unknown GPU or batch size, out-of-range budget, malformed config); sample and uploaded-image prediction; unreadable upload; not-ready state |
| `tests/test_search.py` (dynamic greedy) | also: dynamic greedy finds runs of INT8 units that the static greedy cannot (pair look-ahead), extends a run to reach the optimum, and equals static greedy on additive problems |
| `tests/test_search.py` | 25 random synthetic problems with interacting units: greedy is always feasible, exhaustive equals brute force, local search never makes things worse; greedy optimal on additive problems; skip vs stop; fallbacks; random-search reproducibility and evaluation budget; neighbourhood; gap and Pareto front |

## 11. Project log

| Phase | Content | Status |
|---|---|---|
| 0 | Scaffold, packaging, config, CLI, env-check, CI, literature check | done |
| 1 | Data splits, model, FP32/FP16 baselines, Kaggle fine-tuning | done (training run pending on Kaggle) |
| 2 | Simulated INT8, calibration, sensitivity scan, ONNX Q/DQ export | done |
| 3 | Latency harness, backends, lookup table, TensorRT verification | done (TensorRT runs pending on GPU machines) |
| 4 | Greedy / random / exhaustive / local search | done |
| 5 | Experiments 1–6, Kaggle T4 benchmark | done (GPU runs pending) |
| 6 | Plots and summary tables | done |
| 7 | Web demo | done |
| 9 | Context-aware latency model, dynamic greedy, memory metrics (added after the first T4 run exposed the additive model's error) | implemented; T4 re-run and RTX 3050 Ti run pending |
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
