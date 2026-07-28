# Inference timing comparison

Energy-plus-forces inference on an NVIDIA A100-PCIE-40GB.

- Dataset: ANI-1x CCX test split, 48,957 configurations and 676,395 atoms
- Batch size: 1024
- Warmups: 2
- Measured repetitions: 5
- Values: median inference time per atom, in microseconds
- `new 2D polynomial` uses the merged autotuned results from the `poly_2d_real_data` runs
- Percentages are calculated from the displayed one-decimal timing values
- “Speedup over Triton” is reported as (t_{Triton}/t_{2D}), using the displayed one-decimal timings

| `(l_max, n_max)` | Default upstream (µs/atom) | Triton (µs/atom) | New 2D polynomial (µs/atom) | Triton → 2D decrease | Speedup over Triton (t_{Triton}/t_{2D}) |
|---:|---:|---:|---:|---:|---:|
| (1, 2) | 1.4 | 1.8 | 1.8 | 0.0% | 1.00× |
| (2, 2) | 2.8 | 3.1 | 3.2 | −3.2% | 0.97× |
| (2, 3) | 3.2 | 3.3 | 3.4 | −3.0% | 0.97× |
| (2, 4) | 3.3 | 3.4 | 3.5 | −2.9% | 0.97× |
| (3, 2) | 5.4 | 4.9 | 5.1 | −4.1% | 0.96× |
| (3, 3) | 6.2 | 5.2 | 5.4 | −3.8% | 0.96× |
| (3, 4) | **7.7** | **7.2** | **6.9** | **+4.2%** | **1.04×** |
| (3, 5) | — | 10.1 | 10.5 | −4.0% | 0.96× |
| (4, 3) | — | 11.9 | 11.2 | +5.9% | 1.06× |
| (4, 4) | — | 28.9 | 16.5 | +42.9% | 1.75× |
| (4, 7) | — | 239.1 | 113.2 | +52.7% | 2.11× |

For the `(3,4)` reference configuration, the measured values are approximately
7.7 µs/atom upstream, 7.2 µs/atom with Triton, and 6.9 µs/atom with the new
2D polynomial path.

Sources:

- `results/nick/nick_exact_upstream_energy_forces_b1024_2warmups_speed_eval.pt`
- `results/nick/nick_exact_triton_all_energy_forces_b1024_2warmups_speed_eval.pt`
- `results/nick/nick_exact_triton_l3_n5_energy_forces_b1024_2warmups_speed_eval.pt`
- `results/nick/nick_exact_triton_l4_n7_energy_forces_b1024_2warmups_speed_eval.pt`
- `results/poly_2d_real_data/triton_energy_forces_b1024_autotune_full_l4n7_refresh.json`
