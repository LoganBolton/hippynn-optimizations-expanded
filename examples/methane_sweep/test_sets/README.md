# Reserved methane test sets

Files in this directory define shared, zero-based frame indices into
`datasets/methane.extxyz`. Once reserved, these indices must not be used for
training, validation, hyperparameter selection, or early stopping.

The current common test set is sampled from `[3,080,000, 7,732,488)`. It is
therefore disjoint from sequential training pools through 3,000,000 frames and
from the corresponding legacy 80,000-frame test slice
`[3,000,000, 3,080,000)`.

Load it with:

```python
indices = np.load("examples/methane_sweep/test_sets/methane_common_test_n80000_seed20260720.npy")
```
