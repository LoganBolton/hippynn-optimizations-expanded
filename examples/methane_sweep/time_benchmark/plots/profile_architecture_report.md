# HIP-HOP GPU profile by model architecture

These are **forward-pass GPU kernel times extracted from batch-64 training traces**, summed without overlapping architecture ranges. Only `model::` ranges are included; loss, parameter backpropagation, optimizer updates, and force calculation are excluded. These runs used shuffled training batches and are not an absolute-speed comparison equivalent to the batch-1024 inference benchmark.

| Model | Total ms | Message passing ms | Invariants ms | Feature/atom layers ms | Largest component |
|---|---:|---:|---:|---:|---|
| l=2, n=4 | 6.48 | 0.61 | 0.28 | 3.40 | Feature mixing & atom layers (52.5%) |
| l=4, n=3 | 12.80 | 1.32 | 1.52 | 7.17 | Feature mixing & atom layers (56.0%) |
| l=3, n=5 | 10.82 | 0.61 | 3.43 | 4.62 | Feature mixing & atom layers (42.7%) |
| l=4, n=4 | 12.02 | 0.75 | 4.88 | 4.40 | Invariant polynomials (40.6%) |

## Architectural meaning

- **Neighbor list & geometry:** Indexes the real atoms, builds the atom-pair list inside the cutoff, and computes pair distances/directions.
- **Tensor message passing:** The fused neighbor aggregation: each atom receives tensor-valued messages from nearby atoms.
- **Invariant polynomials:** Turns equivariant tensor features into rotation-invariant many-body features by evaluating the HIP-HOP invariant polynomials.
- **Feature mixing & atom layers:** Dense matrix multiplications that mix channels after message passing and the atom-local neural-network layers, used during the forward pass.
- **Invariant normalization:** Group-normalizes the invariant channels.
- **Energy readout:** Maps per-atom features to atomic energies and sums them into each molecule's energy.
- **Other model tensor ops:** Activations, reductions, indexing, and tensor reshaping/copies that cannot be assigned more narrowly from these traces.
- **Memory transfer/init:** GPU memory copies, allocation initialization, and memset operations.

## Optimization target

Across these four inference passes, the first target is **Feature mixing & atom layers**. The second target is **Invariant polynomials**. The absolute-time plot shows whether an optimization matters for the expensive models; the percentage plot shows how each model changes its computational balance.
Concretely, optimize the forward `evaluate_polynomials_kernel` first: it consumes 4.88 of 12.02 GPU ms/batch in `(l_max=4, n_max=4)`.
For `(2,4)` and `(4,3)`, the largest bucket is instead the dense feature mixing and atom-local layers. That time is spread across several cuBLAS/CUTLASS matrix multiplications, rather than one identifiable kernel.

The gray `Other model tensor ops` segment is intentionally conservative. The existing traces name the fused message-passing and invariant kernels precisely, but do not put architecture labels around every internal HIP-HOP operation. It should not be treated as one kernel or one optimization target.

There is no `Calculating forces` segment because force calculation is not part of this forward energy-inference view.
