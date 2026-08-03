Conservative catastrophe-filtered analysis
==========================================

Input: /vast/home/logan_bolton/Github/hippynn-optimizations-expanded/examples/methane_sweep/results/per_point/random_common_80k
Runs: 24; common structures per run: 80000

A shared hard-source mask is applied to every architecture and metric. Seeds are averaged within architecture; a source is removed when at least 2 architectures have force RMSE >= 200.
Shared hard source structures removed: 1.

Only the shared hard-source mask was applied (--shared-only).

No input rows were deleted or changed. A run-structure prediction is excluded from the diagnostic filtered metric only when:
  robust z of log1p(error) > 10, and
  error / median(other runs' error) >= 100.

Energy MAE = mean(abs(energy_error)).
Energy MSE = mean(energy_error squared).
Force MAE = mean(per-structure force_mae).
Force RMSE = sqrt(mean(per-structure force_rmse squared)).

Energy MAE and MSE share the energy diagnostic mask. Force MAE and Force RMSE have their own masks. Raw metrics remain the official unmodified test-set results.
