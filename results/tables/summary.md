| gpu | batch_size | budget_pp | latency_model | budget_kind | method | val_acc | drop_pp | est_latency_ms | speedup | gap_pct | gap_std | n_evals | feasible_rate | config | model_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t4 | 1 | 0.10 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.10 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 0.67 | 0.83 | 20.11 |  | 1.00 | 0.00 | 1111111111 |  |
| t4 | 1 | 0.10 | additive | supplementary | Greedy, stop at first violation | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.10 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.10 | additive | supplementary | Greedy + local search | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.10 | additive | supplementary | Random search (same evals) | 96.49 | 0.01 | 0.59 | 0.95 | 5.85 | 1.94 | 11.00 | 1.00 | (varies) |  |
| t4 | 1 | 0.10 | additive | supplementary | Exhaustive (optimum) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1024.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.20 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.20 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 0.67 | 0.83 | 20.11 |  | 1.00 | 0.00 | 1111111111 |  |
| t4 | 1 | 0.20 | additive | supplementary | Greedy, stop at first violation | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.20 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.20 | additive | supplementary | Greedy + local search | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.20 | additive | supplementary | Random search (same evals) | 96.47 | 0.03 | 0.59 | 0.95 | 5.69 | 1.83 | 11.00 | 1.00 | (varies) |  |
| t4 | 1 | 0.20 | additive | supplementary | Exhaustive (optimum) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1024.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.30 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.30 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 0.67 | 0.83 | 20.11 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 1 | 0.30 | additive | supplementary | Greedy, stop at first violation | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.30 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.30 | additive | supplementary | Greedy + local search | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.30 | additive | supplementary | Random search (same evals) | 96.41 | 0.09 | 0.58 | 0.95 | 5.27 | 1.58 | 11.00 | 1.00 | (varies) |  |
| t4 | 1 | 0.30 | additive | supplementary | Exhaustive (optimum) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1024.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.50 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.50 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 0.67 | 0.83 | 20.11 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 1 | 0.50 | additive | pre-registered | Greedy, stop at first violation | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.50 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.50 | additive | pre-registered | Greedy + local search | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 0.50 | additive | pre-registered | Random search (same evals) | 96.33 | 0.17 | 0.58 | 0.95 | 5.16 | 1.45 | 11.00 | 1.00 | (varies) |  |
| t4 | 1 | 0.50 | additive | pre-registered | Exhaustive (optimum) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1024.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 1.00 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 1.00 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 0.67 | 0.83 | 20.11 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 1 | 1.00 | additive | pre-registered | Greedy, stop at first violation | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 1.00 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 1.00 | additive | pre-registered | Greedy + local search | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 1.00 | additive | pre-registered | Random search (same evals) | 96.33 | 0.17 | 0.58 | 0.95 | 5.16 | 1.45 | 11.00 | 1.00 | (varies) |  |
| t4 | 1 | 1.00 | additive | pre-registered | Exhaustive (optimum) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1024.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 2.00 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 2.00 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 0.67 | 0.83 | 20.11 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 1 | 2.00 | additive | pre-registered | Greedy, stop at first violation | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 2.00 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 2.00 | additive | pre-registered | Greedy + local search | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 11.00 | 1.00 | 0000000000 |  |
| t4 | 1 | 2.00 | additive | pre-registered | Random search (same evals) | 96.33 | 0.17 | 0.58 | 0.95 | 5.16 | 1.45 | 11.00 | 1.00 | (varies) |  |
| t4 | 1 | 2.00 | additive | pre-registered | Exhaustive (optimum) | 96.50 | 0.00 | 0.55 | 1.00 | 0.00 |  | 1024.00 | 1.00 | 0000000000 |  |
| t4 | 8 | 0.10 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 1.64 | 1.00 | 16.27 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 8 | 0.10 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 1.48 | 1.11 | 4.88 |  | 1.00 | 0.00 | 1111111111 |  |
| t4 | 8 | 0.10 | additive | supplementary | Greedy, stop at first violation | 96.50 | 0.00 | 1.42 | 1.16 | 0.58 |  | 18.00 | 1.00 | 0011111110 |  |
| t4 | 8 | 0.10 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.50 | 0.00 | 1.42 | 1.16 | 0.58 |  | 18.00 | 1.00 | 0011111110 |  |
| t4 | 8 | 0.10 | additive | supplementary | Greedy + local search | 96.50 | 0.00 | 1.42 | 1.16 | 0.58 |  | 22.00 | 1.00 | 0011111110 |  |
| t4 | 8 | 0.10 | additive | supplementary | Random search (same evals) | 96.53 | -0.03 | 1.49 | 1.10 | 5.60 | 2.08 | 18.00 | 1.00 | (varies) |  |
| t4 | 8 | 0.10 | additive | supplementary | Exhaustive (optimum) | 96.40 | 0.10 | 1.41 | 1.16 | 0.00 |  | 1024.00 | 1.00 | 0111101111 |  |
| t4 | 8 | 0.20 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 1.64 | 1.00 | 18.42 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 8 | 0.20 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 1.48 | 1.11 | 6.81 |  | 1.00 | 0.00 | 1111111111 |  |
| t4 | 8 | 0.20 | additive | supplementary | Greedy, stop at first violation | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.20 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.20 | additive | supplementary | Greedy + local search | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.20 | additive | supplementary | Random search (same evals) | 96.41 | 0.09 | 1.47 | 1.12 | 6.17 | 2.52 | 18.00 | 1.00 | (varies) |  |
| t4 | 8 | 0.20 | additive | supplementary | Exhaustive (optimum) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.30 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 1.64 | 1.00 | 18.42 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 8 | 0.30 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 1.48 | 1.11 | 6.81 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 8 | 0.30 | additive | supplementary | Greedy, stop at first violation | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.30 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.30 | additive | supplementary | Greedy + local search | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.30 | additive | supplementary | Random search (same evals) | 96.34 | 0.16 | 1.46 | 1.13 | 5.31 | 2.61 | 18.00 | 1.00 | (varies) |  |
| t4 | 8 | 0.30 | additive | supplementary | Exhaustive (optimum) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.50 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 1.64 | 1.00 | 18.42 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 8 | 0.50 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 1.48 | 1.11 | 6.81 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 8 | 0.50 | additive | pre-registered | Greedy, stop at first violation | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.50 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.50 | additive | pre-registered | Greedy + local search | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 0.50 | additive | pre-registered | Random search (same evals) | 96.28 | 0.22 | 1.45 | 1.13 | 4.61 | 2.26 | 18.00 | 1.00 | (varies) |  |
| t4 | 8 | 0.50 | additive | pre-registered | Exhaustive (optimum) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 1.00 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 1.64 | 1.00 | 18.42 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 8 | 1.00 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 1.48 | 1.11 | 6.81 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 8 | 1.00 | additive | pre-registered | Greedy, stop at first violation | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 1.00 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 1.00 | additive | pre-registered | Greedy + local search | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 1.00 | additive | pre-registered | Random search (same evals) | 96.26 | 0.24 | 1.45 | 1.13 | 4.51 | 2.22 | 18.00 | 1.00 | (varies) |  |
| t4 | 8 | 1.00 | additive | pre-registered | Exhaustive (optimum) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 2.00 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 1.64 | 1.00 | 18.42 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 8 | 2.00 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 1.48 | 1.11 | 6.81 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 8 | 2.00 | additive | pre-registered | Greedy, stop at first violation | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 2.00 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 2.00 | additive | pre-registered | Greedy + local search | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 8 | 2.00 | additive | pre-registered | Random search (same evals) | 96.26 | 0.24 | 1.45 | 1.13 | 4.51 | 2.22 | 18.00 | 1.00 | (varies) |  |
| t4 | 8 | 2.00 | additive | pre-registered | Exhaustive (optimum) | 96.30 | 0.20 | 1.39 | 1.18 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.10 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 6.94 | 1.00 | 30.37 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 32 | 0.10 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 5.66 | 1.22 | 6.46 |  | 1.00 | 0.00 | 1111111111 |  |
| t4 | 32 | 0.10 | additive | supplementary | Greedy, stop at first violation | 96.50 | 0.00 | 5.52 | 1.26 | 3.70 |  | 18.00 | 1.00 | 0011111110 |  |
| t4 | 32 | 0.10 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.50 | 0.00 | 5.52 | 1.26 | 3.70 |  | 18.00 | 1.00 | 0011111110 |  |
| t4 | 32 | 0.10 | additive | supplementary | Greedy + local search | 96.40 | 0.10 | 5.51 | 1.26 | 3.66 |  | 25.00 | 1.00 | 0111111100 |  |
| t4 | 32 | 0.10 | additive | supplementary | Random search (same evals) | 96.50 | -0.00 | 5.93 | 1.17 | 11.53 | 3.56 | 18.00 | 1.00 | (varies) |  |
| t4 | 32 | 0.10 | additive | supplementary | Exhaustive (optimum) | 96.40 | 0.10 | 5.32 | 1.30 | 0.00 |  | 1024.00 | 1.00 | 0111101111 |  |
| t4 | 32 | 0.20 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 6.94 | 1.00 | 34.28 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 32 | 0.20 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 5.66 | 1.22 | 9.66 |  | 1.00 | 0.00 | 1111111111 |  |
| t4 | 32 | 0.20 | additive | supplementary | Greedy, stop at first violation | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.20 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.20 | additive | supplementary | Greedy + local search | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.20 | additive | supplementary | Random search (same evals) | 96.40 | 0.10 | 5.77 | 1.20 | 11.69 | 4.95 | 18.00 | 1.00 | (varies) |  |
| t4 | 32 | 0.20 | additive | supplementary | Exhaustive (optimum) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.30 | additive | supplementary | All-FP16 | 96.50 | 0.00 | 6.94 | 1.00 | 34.28 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 32 | 0.30 | additive | supplementary | All-INT8 | 96.20 | 0.30 | 5.66 | 1.22 | 9.66 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 32 | 0.30 | additive | supplementary | Greedy, stop at first violation | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.30 | additive | supplementary | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.30 | additive | supplementary | Greedy + local search | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.30 | additive | supplementary | Random search (same evals) | 96.31 | 0.19 | 5.65 | 1.23 | 9.30 | 4.89 | 18.00 | 1.00 | (varies) |  |
| t4 | 32 | 0.30 | additive | supplementary | Exhaustive (optimum) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.50 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 6.94 | 1.00 | 34.28 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 32 | 0.50 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 5.66 | 1.22 | 9.66 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 32 | 0.50 | additive | pre-registered | Greedy, stop at first violation | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.50 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.50 | additive | pre-registered | Greedy + local search | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 0.50 | additive | pre-registered | Random search (same evals) | 96.24 | 0.26 | 5.58 | 1.25 | 7.96 | 4.09 | 18.00 | 1.00 | (varies) |  |
| t4 | 32 | 0.50 | additive | pre-registered | Exhaustive (optimum) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 1.00 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 6.94 | 1.00 | 34.28 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 32 | 1.00 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 5.66 | 1.22 | 9.66 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 32 | 1.00 | additive | pre-registered | Greedy, stop at first violation | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 1.00 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 1.00 | additive | pre-registered | Greedy + local search | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 1.00 | additive | pre-registered | Random search (same evals) | 96.22 | 0.28 | 5.57 | 1.25 | 7.80 | 4.01 | 18.00 | 1.00 | (varies) |  |
| t4 | 32 | 1.00 | additive | pre-registered | Exhaustive (optimum) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 2.00 | additive | pre-registered | All-FP16 | 96.50 | 0.00 | 6.94 | 1.00 | 34.28 |  | 1.00 | 1.00 | 0000000000 |  |
| t4 | 32 | 2.00 | additive | pre-registered | All-INT8 | 96.20 | 0.30 | 5.66 | 1.22 | 9.66 |  | 1.00 | 1.00 | 1111111111 |  |
| t4 | 32 | 2.00 | additive | pre-registered | Greedy, stop at first violation | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 2.00 | additive | pre-registered | Sensitivity-guided greedy (ours) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 2.00 | additive | pre-registered | Greedy + local search | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 18.00 | 1.00 | 0111111110 |  |
| t4 | 32 | 2.00 | additive | pre-registered | Random search (same evals) | 96.22 | 0.28 | 5.57 | 1.25 | 7.80 | 4.01 | 18.00 | 1.00 | (varies) |  |
| t4 | 32 | 2.00 | additive | pre-registered | Exhaustive (optimum) | 96.30 | 0.20 | 5.16 | 1.34 | 0.00 |  | 1024.00 | 1.00 | 0111111110 |  |
