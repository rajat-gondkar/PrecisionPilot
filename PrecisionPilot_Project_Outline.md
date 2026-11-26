# PrecisionPilot: Sensitivity-Guided Mixed-Precision Selection for GPU Inference

*Project outline and reference document*

---

## 1. Overview

Neural networks can run their math in different number formats: FP32 (most precise, slowest), FP16 (half the bits, faster) and INT8 (8-bit integers, smallest and often fastest). Lower precision saves memory and time but can reduce accuracy, and different layers react differently to it.

**PrecisionPilot** is a lightweight framework that automatically decides which layers of a pretrained model should run in INT8 and which should stay in FP16, so that inference is as fast as possible while staying within a user-defined accuracy-loss budget. It does this with a **sensitivity-guided search** that needs far fewer evaluations than brute-force search.

---

## 2. Problem Statement

- Running everything in high precision wastes memory and time.
- Running everything in INT8 can hurt accuracy badly, because some layers (often the first and last) are fragile.
- Testing every possible per-layer combination is infeasible as the number of layers grows (2^N configurations).
- The best configuration also depends on **batch size** and **GPU**, so one fixed choice does not fit all.

**Central question:** Can a cheap, sensitivity-guided search find mixed-precision configurations that stay within an accuracy budget, with far fewer evaluations than brute-force or random search, while adapting to batch size and hardware?

---

## 3. Claim Being Tested

> A sensitivity-guided search finds mixed-precision configurations within the accuracy budget using far fewer evaluations than brute-force or random search, and the best configuration depends on batch size and GPU.

---

## 4. Scope

**Included**
- One pretrained model: ResNet-18
- One dataset: CIFAR-10 (or a 2,000-5,000 image ImageNet subset)
- Two precision choices per unit: FP16 or INT8 (FP32 used as the accuracy reference)
- Post-training quantization only (no retraining)
- Offline selection (decide once per model + GPU + batch size)
- Batch sizes 1, 8, 32
- Two GPUs: RTX 3050 Ti (main) and Kaggle T4 (hardware comparison)

**Excluded (keep the project small)**
- Quantization-aware training
- Runtime or dynamic precision switching
- Energy measurement
- Large models (ResNet-50+, transformers)
- Many-tool comparisons

---

## 5. Background: Precision Formats

| Format | Bits | Notes |
|---|---|---|
| FP32 | 32 | Reference for accuracy. Largest and slowest. |
| FP16 | 16 | Smaller and faster on Tensor Core GPUs. Different numeric range, so validate. |
| INT8 | 8 | Values mapped to integers using scale (and zero point). Smallest, often fastest, but not automatically faster; depends on hardware, operator support and backend. |
| Mixed | varies | Different layers use different formats. Goal is quality close to FP32 with speed close to INT8. |

---

## 6. Method

### 6.1 Model grouping
Split ResNet-18 into about **10 units**: stem, 8 residual blocks, final FC. Each unit gets one precision choice (FP16 or INT8). This gives 2^10 = 1,024 possible configurations, small enough to run an exhaustive search as ground truth.

### 6.2 Accuracy measurement
Use PyTorch **simulated (fake) INT8 quantization** per unit. It is flexible and avoids backend fiddliness when testing many configurations.

### 6.3 Latency measurement
1. Measure each unit's latency in FP16 and INT8 once, per batch size and per GPU.
2. Build a **lookup table** to estimate the latency of any configuration.
3. Verify the final winning configurations with real TensorRT runs.
4. Report the estimation error (per-unit times will not sum exactly because of layer fusion).

### 6.4 Search algorithm (the contribution)
1. **Sensitivity scan:** quantize one unit at a time to INT8 while the rest stay FP16, and measure the accuracy drop on a small validation subset (about 10 evaluations).
2. **Score each unit:** rank by latency saved per accuracy lost, so units that are cheap to quantize and give a big speedup come first.
3. **Greedy search:** start with everything in FP16, flip units to INT8 in rank order, and stop (or back off one step) when the accuracy budget would be exceeded.
4. **Final check:** evaluate the chosen configuration on the held-out test set.
5. **Optional refinement:** a small local search that tries swapping pairs of units if greedy lands far from optimal.

### 6.5 Data splits
- **Calibration set:** used for INT8 calibration
- **Validation set:** used for the sensitivity scan and the search
- **Test set:** used only for the final reported results. Never select configurations using the test set.

### 6.6 Fixed rules
- Set the accuracy budget (for example 1 percentage point drop versus FP32) **before** seeing results, and do not change it afterward.
- Keep batch size, preprocessing and warm-up identical across experiments.
- Use warm-up runs, then 50-100 repeated measurements, and report the median and p95.

---

## 7. Baselines

| Baseline | Purpose |
|---|---|
| FP32 | Accuracy reference |
| All-FP16 | Safe, fast reference |
| All-INT8 | Fastest but possibly too inaccurate |
| Random search (same evaluation budget as the proposed method) | Shows the sensitivity guidance actually matters |
| Exhaustive search | Ground-truth optimum; lets you report "within X% of optimal" |

---

## 8. Experiments

| # | Experiment | What it shows |
|---|---|---|
| 1 | Per-unit sensitivity scan | Which layers are fragile (usually first and last) |
| 2 | Search cost vs. result quality | Evaluations needed by each method vs. final latency (**main chart**) |
| 3 | Accuracy budget sweep (0.5%, 1%, 2%) | How the chosen configuration changes with the budget |
| 4 | Batch-size variation (1, 8, 32) | Whether the best configuration changes |
| 5 | Second GPU (Kaggle T4) | Hardware dependence of the best configuration |
| 6 | Latency estimation accuracy | Lookup-table estimate vs. real TensorRT measurement |

---

## 9. Metrics

- Top-1 accuracy and accuracy drop versus FP32
- Latency: estimated and measured (median and p95)
- Throughput (images per second)
- Model size and peak memory
- Number of evaluations and total search time
- Gap to the exhaustive-search optimum

---

## 10. Visualizations and Presentation

1. **Layer sensitivity bar chart:** accuracy drop per unit (most intuitive visual)
2. **Search cost vs. quality chart:** main result
3. **Accuracy vs. latency scatter:** each configuration is a point, the accuracy budget is a line, the proposed method and baselines are highlighted
4. **Heatmap:** which units are INT8 vs. FP16, per budget and batch size
5. **Summary table:** all metrics in one place
6. **Optional Streamlit/Gradio demo:** sliders for accuracy budget and batch size that show the chosen configuration and live predictions on sample images

---

## 11. Tech Stack

- **Language:** Python (C++ only if needed)
- **Deep learning:** PyTorch, torchvision
- **Inference/optimization:** TensorRT (NVIDIA)
- **Profiling:** PyTorch Profiler, `nvidia-smi`
- **Analysis and plots:** NumPy, pandas, Matplotlib
- **Demo (optional):** Streamlit or Gradio

---

## 12. Hardware Plan

- **Main machine: RTX 3050 Ti laptop (4 GB VRAM).** Ampere GPU with Tensor Cores, supports FP16 and INT8. Keep batch sizes modest and models small.
  - Stay plugged in, use performance mode, close other GPU apps, add cooldown gaps between runs to limit thermal throttling.
- **Secondary: Kaggle T4.** Used for a second hardware data point. Kaggle GPUs can vary across sessions, so do not use it as the primary benchmark machine.
- Check that PyTorch, CUDA and TensorRT versions match before anything else; this is the most common time sink.

---

## 13. Timeline (about 50-65 hours)

| Phase | Hours |
|---|---|
| Setup, ResNet-18, CIFAR-10, FP32/FP16 baselines | 8-10 |
| Simulated INT8 per unit and sensitivity scan | 8-10 |
| Per-unit latency table and TensorRT verification | 8-12 |
| Greedy search, random and exhaustive baselines | 6-8 |
| Experiments (budgets, batch sizes, Kaggle T4) | 8-10 |
| Plots, demo, writing | 8-12 |
| Literature check | 4-6 |

Suggested order: do the literature check early, get the simple FP32/FP16/INT8 pipeline working end-to-end first, then add the sensitivity scan and search.

---

## 14. Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Latency table is not additive because of layer fusion | Verify final configurations in real runs and report the estimation error |
| INT8 gives no speedup at small batch sizes | Report as a finding; the selector correctly chooses FP16 |
| TensorRT mixed precision is difficult | Use PyTorch simulated quantization for the sensitivity study, report estimated latency, and state this clearly |
| Greedy search is far from optimal | Report the gap, or add pairwise-swap local search |
| Exhaustive search is slow | Use a 1,000-image validation subset, confirm the winner on the full set |
| Thermal throttling on laptop | Cooldown gaps, repeated runs, median reporting |
| INT8 accuracy drops too much | Try different calibration set sizes or methods |

---

## 15. Novelty Positioning

FP32/FP16/INT8 comparison with a threshold selector is already done by existing tools and is not novel by itself. Layer sensitivity and mixed-precision search are also well studied (for example HAWQ and HAQ, plus other sensitivity-based methods).

The defensible contribution is a **lightweight, practical, accuracy-constrained method** with:
- low search cost compared with brute-force and random search,
- measured closeness to the exhaustive optimum,
- batch-size and GPU awareness.

**Safe wording:**
> "A lightweight, accuracy-constrained mixed-precision selection framework using sensitivity-guided search, evaluated against random and exhaustive search across batch sizes and GPUs."

Avoid claiming a new field or a new idea. Do a literature check first and phrase the contribution around what is genuinely different from the closest existing work. Report limitations honestly, including cases where the method does not help.

---

## 16. Deliverables

- Code repository with reproducible scripts and fixed random seeds
- Per-unit sensitivity and latency tables for each GPU and batch size
- Result plots and summary tables
- Optional interactive demo
- Project report or paper covering: introduction, related work, method, experiments, results, limitations, conclusion

---

## 17. Expected Outcome

A working framework that, given a pretrained ResNet-18, a GPU, a batch size and an accuracy budget, produces a per-layer FP16/INT8 configuration that meets the budget with near-optimal latency, found with a small fraction of the evaluations that brute-force search needs. The results should also show how the best configuration shifts with batch size and hardware.

---

## 18. Quick Checklist

- [ ] Environment working (PyTorch, CUDA, TensorRT versions matched)
- [ ] Accuracy budget fixed before experiments
- [ ] Data split into calibration, validation and test
- [ ] FP32, FP16 and INT8 baselines done
- [ ] Sensitivity scan done
- [ ] Latency lookup table built
- [ ] Greedy, random and exhaustive search implemented
- [ ] TensorRT verification of final configurations
- [ ] Budget sweep, batch-size and second-GPU experiments
- [ ] Plots and summary table
- [ ] Literature check and related work written
- [ ] Report or demo finished
