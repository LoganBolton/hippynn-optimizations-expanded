import argparse
import statistics
import string
import time

import torch

import hippynn.layers.hiplayers.invariants as invariants_mod
from hippynn.layers.hiplayers.invariants import (
    compute_invariant_polynomial_collection,
    default_invariants_list,
    split_invariant,
)

"""
Compare polynomial invariant construction/evaluation with and without duplicate
monomial merging.

Example:
    PYTHONPATH=. python benchmarks/benchmark_polynomial_dedup.py --points 10
    PYTHONPATH=. python benchmarks/benchmark_polynomial_dedup.py --points 10 --eval-repeats 1
"""


DEDUP_STATS = {}


def _coefficient_tensor(tensor_bases, invariant_code):
    invariant_code = invariant_code.replace("->", "")
    invar_indices, invar_tensors = split_invariant(invariant_code)

    if len(invar_indices) == 1:
        return invar_indices, invar_tensors, None

    num_terms = len(invar_indices)
    letters_used = set()
    for term_idx in range(num_terms):
        letters_used.update(invar_indices[term_idx])

    unused_letters = list(set(string.ascii_letters) - letters_used)
    assert len(unused_letters) >= num_terms

    einsum_front = ""
    einsum_back = ""
    tensors_to_contract = []
    for term_idx in range(num_terms):
        code = invar_indices[term_idx]
        einsum_front += "," + code + unused_letters[term_idx]
        einsum_back += unused_letters[term_idx]
        tensors_to_contract.append(tensor_bases[len(code)])

    coef_tensor = torch.einsum(einsum_front[1:] + "->" + einsum_back, *tensors_to_contract)
    return invar_indices, invar_tensors, coef_tensor


def compute_invariant_polynomial_with_stats(tensor_bases, invariant_input_offsets, invariant_code):
    invar_indices, invar_tensors, coef_tensor = _coefficient_tensor(tensor_bases, invariant_code)

    if coef_tensor is None:
        tensor_name = invar_tensors[0]
        offset = invariant_input_offsets[tensor_name]
        DEDUP_STATS[invariant_code] = {
            "dense_terms": 1,
            "raw_zero_terms_removed": 0,
            "raw_nonzero_terms": 1,
            "unique_terms": 1,
            "zero_cancelled_terms": 0,
            "final_terms": 1,
        }
        return torch.FloatTensor((1,)), torch.IntTensor(((offset,),))

    num_terms = len(invar_indices)
    nonzero_basis_choices = torch.nonzero(coef_tensor, as_tuple=False)
    nonzero_coefs = coef_tensor[tuple(nonzero_basis_choices.T)]
    feature_offsets = torch.tensor(
        [invariant_input_offsets[invar_tensors[dim]] for dim in range(num_terms)],
        dtype=nonzero_basis_choices.dtype,
        device=nonzero_basis_choices.device,
    )
    monomial_terms = nonzero_basis_choices + feature_offsets
    monomial_terms = torch.sort(monomial_terms, dim=1).values
    unique_terms, duplicate_map = torch.unique(monomial_terms, dim=0, return_inverse=True)
    unique_coefs = nonzero_coefs.new_zeros(unique_terms.shape[0])
    unique_coefs.scatter_add_(0, duplicate_map, nonzero_coefs)
    nonzero_terms = unique_coefs != 0

    DEDUP_STATS[invariant_code] = {
        "dense_terms": coef_tensor.numel(),
        "raw_zero_terms_removed": coef_tensor.numel() - nonzero_basis_choices.shape[0],
        "raw_nonzero_terms": monomial_terms.shape[0],
        "unique_terms": unique_terms.shape[0],
        "zero_cancelled_terms": int((~nonzero_terms).sum().item()),
        "final_terms": int(nonzero_terms.sum().item()),
    }

    coefs_tensor = unique_coefs[nonzero_terms].to(dtype=torch.float32, device="cpu")
    terms_tensor = unique_terms[nonzero_terms].to(dtype=torch.int32, device="cpu")
    return coefs_tensor, terms_tensor


def compute_invariant_polynomial_no_dedup(tensor_bases, invariant_input_offsets, invariant_code):
    invar_indices, invar_tensors, coef_tensor = _coefficient_tensor(tensor_bases, invariant_code)

    if coef_tensor is None:
        tensor_name = invar_tensors[0]
        offset = invariant_input_offsets[tensor_name]
        return torch.FloatTensor((1,)), torch.IntTensor(((offset,),))

    num_terms = len(invar_indices)
    nonzero_basis_choices = torch.nonzero(coef_tensor, as_tuple=False)
    nonzero_coefs = coef_tensor[tuple(nonzero_basis_choices.T)]
    feature_offsets = torch.tensor(
        [invariant_input_offsets[invar_tensors[dim]] for dim in range(num_terms)],
        dtype=nonzero_basis_choices.dtype,
        device=nonzero_basis_choices.device,
    )
    monomial_terms = nonzero_basis_choices + feature_offsets
    return (
        nonzero_coefs.to(dtype=torch.float32, device="cpu"),
        monomial_terms.to(dtype=torch.int32, device="cpu"),
    )


def evaluate_polynomial_collection_torch(x, poly_collection):
    coefs, terms, polynomial_sizes, _ = poly_collection.get_polynomials()
    outputs = []
    monomial_start = 0
    for polynomial_size in polynomial_sizes.tolist():
        values = x.new_zeros(x.shape[0])
        for monomial_idx in range(monomial_start, monomial_start + polynomial_size):
            term_indices = terms[monomial_idx]
            term_indices = term_indices[term_indices >= 0]
            values = values + coefs[monomial_idx].to(x) * x[:, term_indices].prod(dim=1)
        outputs.append(values)
        monomial_start += polynomial_size
    return torch.stack(outputs, dim=1)


def compare_outputs(ref, test):
    diff = ref - test
    max_abs_diff = diff.abs().max().item()
    max_ref = ref.abs().max().item()
    max_test = test.abs().max().item()
    relative_to_scale = max_abs_diff / max(max_ref, max_test, 1.0)
    return torch.allclose(ref, test, rtol=1e-4, atol=1e-4), max_abs_diff, relative_to_scale, max_ref, max_test


def time_call(fn, repeats):
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return result, timings


def build_collection(n_max, l_max, dedup):
    original = invariants_mod.computeInvariantPolynomial
    invariants_mod.computeInvariantPolynomial = (
        compute_invariant_polynomial_with_stats if dedup else compute_invariant_polynomial_no_dedup
    )
    try:
        return compute_invariant_polynomial_collection(
            n_max,
            l_max,
            invariants=default_invariants_list,
            input_tensor_ordering=["zero", "one", "two", "three"][: l_max + 1],
        )
    finally:
        invariants_mod.computeInvariantPolynomial = original


def summarize(times):
    return statistics.median(times), min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--l-max", type=int, default=3)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--build-repeats", type=int, default=3)
    parser.add_argument("--eval-repeats", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.points, (args.l_max + 1) ** 2)

    collections = {}
    for name, dedup in [("no_dedup", False), ("dedup", True)]:
        if dedup:
            DEDUP_STATS.clear()
        collection, build_times = time_call(
            lambda dedup=dedup: build_collection(args.n_max, args.l_max, dedup),
            args.build_repeats,
        )
        collections[name] = collection
        coefs, terms, sizes, _ = collection.get_polynomials()
        build_median, build_best = summarize(build_times)
        print(
            f"{name}: build median={build_median:.6f}s best={build_best:.6f}s "
            f"monomials={terms.shape[0]} invariants={sizes.numel()} coef_sum={coefs.abs().sum().item():.6g}",
            flush=True,
        )
        if dedup:
            dense_terms = sum(stat["dense_terms"] for stat in DEDUP_STATS.values())
            raw_zero_terms_removed = sum(stat["raw_zero_terms_removed"] for stat in DEDUP_STATS.values())
            raw_nonzero_terms = sum(stat["raw_nonzero_terms"] for stat in DEDUP_STATS.values())
            unique_terms = sum(stat["unique_terms"] for stat in DEDUP_STATS.values())
            zero_cancelled_terms = sum(stat["zero_cancelled_terms"] for stat in DEDUP_STATS.values())
            final_terms = sum(stat["final_terms"] for stat in DEDUP_STATS.values())
            print(
                f"dedup_breakdown: dense_terms={dense_terms} "
                f"raw_zero_terms_removed={raw_zero_terms_removed} "
                f"raw_nonzero_terms={raw_nonzero_terms} "
                f"duplicate_terms_removed={raw_nonzero_terms - unique_terms} "
                f"unique_terms_before_zero_filter={unique_terms} "
                f"post_merge_zero_terms_removed={zero_cancelled_terms} "
                f"final_terms={final_terms}",
                flush=True,
            )

    ref = evaluate_polynomial_collection_torch(x, collections["no_dedup"])
    test = evaluate_polynomial_collection_torch(x, collections["dedup"])
    close, max_abs_diff, rel_diff, max_ref, max_test = compare_outputs(ref, test)
    print(
        f"outputs_close={close} max_abs_diff={max_abs_diff:.6g} "
        f"rel_to_max={rel_diff:.6g} max_no_dedup={max_ref:.6g} max_dedup={max_test:.6g}",
        flush=True,
    )

    x64 = x.to(torch.float64)
    ref64 = evaluate_polynomial_collection_torch(x64, collections["no_dedup"])
    test64 = evaluate_polynomial_collection_torch(x64, collections["dedup"])
    close64, max_abs_diff64, rel_diff64, max_ref64, max_test64 = compare_outputs(ref64, test64)
    print(
        f"outputs_close_float64={close64} max_abs_diff={max_abs_diff64:.6g} "
        f"rel_to_max={rel_diff64:.6g} max_no_dedup={max_ref64:.6g} max_dedup={max_test64:.6g}",
        flush=True,
    )

    if args.eval_repeats <= 0:
        print("eval timing skipped; pass --eval-repeats N to time evaluation", flush=True)
        return

    print(f"beginning eval timing with repeats={args.eval_repeats}", flush=True)
    for name, collection in collections.items():
        _, eval_times = time_call(
            lambda collection=collection: evaluate_polynomial_collection_torch(x, collection),
            args.eval_repeats,
        )
        eval_median, eval_best = summarize(eval_times)
        print(f"{name}: eval median={eval_median:.6f}s best={eval_best:.6f}s", flush=True)


if __name__ == "__main__":
    main()
