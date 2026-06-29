import torch
import numpy as np

meta_path = "../TEST_METHANE_MODEL_l4_n4_seed2025/FinalTraining_invariant_metadata.pt"
values_path = "../TEST_METHANE_MODEL_l4_n4_seed2025/FinalTraining_invariant_values.npz"

meta = torch.load(meta_path, map_location="cpu")
values = np.load(values_path)

print("Metadata file:", meta_path)
print("Values file:", values_path)
print()
print("Top-level metadata keys:", list(meta.keys()))
print("Value key map:", meta.get("value_key_map"))
print("Saved value arrays:", values.files)
print()

for layer_name, layer_meta in meta["metadata"].items():
    print("=" * 88)
    print("Layer:", layer_name)
    print("n_max:", layer_meta["n_max"])
    print("l_max:", layer_meta["l_max"])
    print("n_invariants:", layer_meta["n_invariants"])
    print("input_dimension:", layer_meta["input_dimension"])
    print("output_shape_seen:", meta["output_shapes"].get(layer_name))
    print("invariant_term_counts:", layer_meta["invariant_term_counts"])
    print("invariant_max_orders:", layer_meta["invariant_max_orders"])
    print("polynomial_sizes:", layer_meta["polynomial_sizes"])
    print()
    print("Invariant codes:")
    for idx, code in enumerate(layer_meta["invariant_codes"]):
        print(f"  {idx:02d}: {code}")
    print()

for value_key, layer_name in meta.get("value_key_map", {}).items():
    arr = values[value_key]
    print("=" * 88)
    print(f"Values array {value_key} -> {layer_name}")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("first 5 rows:")
    print(arr[:5])
    print("column means:")
    print(arr.mean(axis=0))
    print("column stds:")
    print(arr.std(axis=0))