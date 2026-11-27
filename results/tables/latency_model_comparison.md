| gpu | batch_size | budget_pp | latency_model | method | config | n_int8 | drop_pp | feasible | n_evals | est_latency_ms | measured_ms | real_speedup_vs_fp16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t4 | 1 | 0.10 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.10 | additive | All-INT8 | 1111111111 | 10 | 0.30 | False | 1 | 0.67 | 0.66 | 0.99 |
| t4 | 1 | 0.10 | additive | Sensitivity-guided greedy (ours) | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.10 | additive | Greedy + local search | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.10 | additive | Exhaustive (optimum) | 0000000000 | 0 | 0.00 | True | 1024 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.20 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.20 | additive | All-INT8 | 1111111111 | 10 | 0.30 | False | 1 | 0.67 | 0.66 | 0.99 |
| t4 | 1 | 0.20 | additive | Sensitivity-guided greedy (ours) | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.20 | additive | Greedy + local search | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.20 | additive | Exhaustive (optimum) | 0000000000 | 0 | 0.00 | True | 1024 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.30 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.30 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 0.67 | 0.66 | 0.99 |
| t4 | 1 | 0.30 | additive | Sensitivity-guided greedy (ours) | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.30 | additive | Greedy + local search | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.30 | additive | Exhaustive (optimum) | 0000000000 | 0 | 0.00 | True | 1024 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.50 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.50 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 0.67 | 0.66 | 0.99 |
| t4 | 1 | 0.50 | additive | Sensitivity-guided greedy (ours) | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.50 | additive | Greedy + local search | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 0.50 | additive | Exhaustive (optimum) | 0000000000 | 0 | 0.00 | True | 1024 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 1.00 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 1.00 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 0.67 | 0.66 | 0.99 |
| t4 | 1 | 1.00 | additive | Sensitivity-guided greedy (ours) | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 1.00 | additive | Greedy + local search | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 1.00 | additive | Exhaustive (optimum) | 0000000000 | 0 | 0.00 | True | 1024 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 2.00 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 2.00 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 0.67 | 0.66 | 0.99 |
| t4 | 1 | 2.00 | additive | Sensitivity-guided greedy (ours) | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 2.00 | additive | Greedy + local search | 0000000000 | 0 | 0.00 | True | 11 | 0.55 | 0.66 | 1.00 |
| t4 | 1 | 2.00 | additive | Exhaustive (optimum) | 0000000000 | 0 | 0.00 | True | 1024 | 0.55 | 0.66 | 1.00 |
| t4 | 8 | 0.10 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 1.64 | 1.94 | 1.00 |
| t4 | 8 | 0.10 | additive | All-INT8 | 1111111111 | 10 | 0.30 | False | 1 | 1.48 | 1.00 | 1.93 |
| t4 | 8 | 0.10 | additive | Sensitivity-guided greedy (ours) | 0011111110 | 7 | 0.00 | True | 18 | 1.42 | 1.35 | 1.44 |
| t4 | 8 | 0.10 | additive | Greedy + local search | 0011111110 | 7 | 0.00 | True | 22 | 1.42 | 1.35 | 1.44 |
| t4 | 8 | 0.10 | additive | Exhaustive (optimum) | 0111101111 | 8 | 0.10 | True | 1024 | 1.41 | 1.37 | 1.41 |
| t4 | 8 | 0.20 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 1.64 | 1.94 | 1.00 |
| t4 | 8 | 0.20 | additive | All-INT8 | 1111111111 | 10 | 0.30 | False | 1 | 1.48 | 1.00 | 1.93 |
| t4 | 8 | 0.20 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 0.20 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 0.20 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 0.30 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 1.64 | 1.94 | 1.00 |
| t4 | 8 | 0.30 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 1.48 | 1.00 | 1.93 |
| t4 | 8 | 0.30 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 0.30 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 0.30 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 0.50 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 1.64 | 1.94 | 1.00 |
| t4 | 8 | 0.50 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 1.48 | 1.00 | 1.93 |
| t4 | 8 | 0.50 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 0.50 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 0.50 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 1.00 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 1.64 | 1.94 | 1.00 |
| t4 | 8 | 1.00 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 1.48 | 1.00 | 1.93 |
| t4 | 8 | 1.00 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 1.00 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 1.00 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 2.00 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 1.64 | 1.94 | 1.00 |
| t4 | 8 | 2.00 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 1.48 | 1.00 | 1.93 |
| t4 | 8 | 2.00 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 2.00 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 1.39 | 1.23 | 1.58 |
| t4 | 8 | 2.00 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 1.39 | 1.23 | 1.58 |
| t4 | 32 | 0.10 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 6.94 | 6.17 | 1.00 |
| t4 | 32 | 0.10 | additive | All-INT8 | 1111111111 | 10 | 0.30 | False | 1 | 5.66 | 2.74 | 2.26 |
| t4 | 32 | 0.10 | additive | Sensitivity-guided greedy (ours) | 0011111110 | 7 | 0.00 | True | 18 | 5.52 | 4.24 | 1.45 |
| t4 | 32 | 0.10 | additive | Greedy + local search | 0111111100 | 7 | 0.10 | True | 25 | 5.51 | 4.39 | 1.41 |
| t4 | 32 | 0.10 | additive | Exhaustive (optimum) | 0111101111 | 8 | 0.10 | True | 1024 | 5.32 | 4.33 | 1.42 |
| t4 | 32 | 0.20 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 6.94 | 6.17 | 1.00 |
| t4 | 32 | 0.20 | additive | All-INT8 | 1111111111 | 10 | 0.30 | False | 1 | 5.66 | 2.74 | 2.26 |
| t4 | 32 | 0.20 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 0.20 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 0.20 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 0.30 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 6.94 | 6.17 | 1.00 |
| t4 | 32 | 0.30 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 5.66 | 2.74 | 2.26 |
| t4 | 32 | 0.30 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 0.30 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 0.30 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 0.50 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 6.94 | 6.17 | 1.00 |
| t4 | 32 | 0.50 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 5.66 | 2.74 | 2.26 |
| t4 | 32 | 0.50 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 0.50 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 0.50 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 1.00 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 6.94 | 6.17 | 1.00 |
| t4 | 32 | 1.00 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 5.66 | 2.74 | 2.26 |
| t4 | 32 | 1.00 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 1.00 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 1.00 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 2.00 | additive | All-FP16 | 0000000000 | 0 | 0.00 | True | 1 | 6.94 | 6.17 | 1.00 |
| t4 | 32 | 2.00 | additive | All-INT8 | 1111111111 | 10 | 0.30 | True | 1 | 5.66 | 2.74 | 2.26 |
| t4 | 32 | 2.00 | additive | Sensitivity-guided greedy (ours) | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 2.00 | additive | Greedy + local search | 0111111110 | 8 | 0.20 | True | 18 | 5.16 | 3.93 | 1.57 |
| t4 | 32 | 2.00 | additive | Exhaustive (optimum) | 0111111110 | 8 | 0.20 | True | 1024 | 5.16 | 3.93 | 1.57 |
