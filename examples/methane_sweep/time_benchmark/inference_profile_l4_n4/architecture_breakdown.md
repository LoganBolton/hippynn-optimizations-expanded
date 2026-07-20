# HIP-HOP-NN (4, 4) inference architecture profile

- Energy-only GPU kernels: **5.18 us/atom**
- Energy plus force GPU kernels: **28.69 us/atom**
- Additional coordinate-gradient cost: **23.51 us/atom**

## Primary optimization target

Differentiating the invariant polynomials costs **17.05 us/atom**, or **72.5% of the additional force cost**. This is the backward/derivative invocation of `evaluate_polynomials_kernel`.

Differentiating tensor message passing costs **3.79 us/atom**, or **16.1% of the additional force cost**. Together, invariant and message derivatives account for **88.6%** of force overhead.

The force pass calculates `-dE/dR`: the gradient of molecular energy with respect to atomic coordinates. It does not calculate a loss, parameter gradients, or an optimizer update.

Workload: batch 1024, 48,957 fixed test structures, 676,395 atoms, NVIDIA A100-PCIE-40GB.
