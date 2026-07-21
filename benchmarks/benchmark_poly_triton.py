#!/usr/bin/env python3
"""Benchmark legacy and point-by-polynomial Triton polynomial evaluation."""

import argparse
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from hippynn.custom_kernels.poly_triton import (
    EvaluatePolynomials,
    _EvaluatePolynomialsLegacy,
    evaluate_polynomials_kernel,
    evaluate_polynomials_kernel_legacy,
)
from hippynn.layers.hiplayers.invariants import compute_invariant_polynomial_collection


BATCH_SIZE = 512
ATOMS_PER_STRUCTURE = (5, 10, 16, 24, 32)
POINTS = tuple(BATCH_SIZE * atoms for atoms in ATOMS_PER_STRUCTURE)
CONFIGS = (
    (16, 8, 2, 2),
    (16, 16, 2, 2),
    (16, 32, 4, 2),
    (32, 8, 4, 2),
    (32, 16, 4, 2),
    (32, 32, 4, 2),
    (64, 8, 4, 2),
    (64, 16, 4, 2),
    (64, 32, 8, 2),
    (128, 8, 8, 2),
    (128, 16, 8, 2),
)


def prepare_level(polynomials, derivative_level, num_points, dtype):
    coefs, terms, polynomial_sizes, input_dimension = polynomials.get_polynomials(
        derivative_level
    )
    offsets = polynomials.get_polynomial_offsets(derivative_level)
    x = torch.randn(num_points, input_dimension, device="cuda", dtype=dtype)
    input_width = triton.next_power_of_2(input_dimension)
    degree = terms.shape[1]
    degree_width = triton.next_power_of_2(degree)
    if input_dimension != input_width:
        x = F.pad(x, (0, input_width - input_dimension)).contiguous()
    if degree != degree_width:
        terms = F.pad(terms, (0, degree_width - degree), value=-1).contiguous()
    output = torch.empty(
        (num_points, len(polynomial_sizes)), device="cuda", dtype=dtype
    )
    return x, coefs, terms, polynomial_sizes, offsets, output, input_width, degree_width


def launch_raw(prepared, implementation, config=None):
    x, coefs, terms, sizes, offsets, output, input_width, degree_width = prepared
    num_points = x.shape[0]
    num_polynomials = len(sizes)
    num_monomials = len(coefs)
    kernel_dtype = tl.float64 if x.dtype == torch.float64 else tl.float32
    if implementation == "legacy":
        evaluate_polynomials_kernel_legacy[(triton.cdiv(num_points, 128),)](
            x, coefs, terms, sizes, output, num_polynomials, num_monomials,
            input_width, num_points, degree_width, NUM_POINTS_TO_LOAD=128,
            NUM_MONOMIALS_TO_LOAD=8, dtype=kernel_dtype, num_warps=2,
            num_stages=2,
        )
    else:
        point_tile, monomial_tile, num_warps, num_stages = config
        grid = (triton.cdiv(num_points, point_tile), num_polynomials)
        # Bypass the autotuner so each candidate can be measured independently.
        evaluate_polynomials_kernel.fn[grid](
            x, coefs, terms, sizes, offsets, output, num_polynomials,
            num_monomials, input_width, num_points, degree_width,
            point_bucket=triton.next_power_of_2(num_points),
            NUM_POINTS_TO_LOAD=point_tile,
            NUM_MONOMIALS_TO_LOAD=monomial_tile, dtype=kernel_dtype,
            num_warps=num_warps, num_stages=num_stages,
        )
    return output


def bench_ms(fn, warmup, rep):
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def check_correctness(polynomials):
    torch.manual_seed(1234)
    x_legacy = torch.randn(24, 25, device="cuda", requires_grad=True)
    x_new = x_legacy.detach().clone().requires_grad_(True)
    y_legacy = _EvaluatePolynomialsLegacy.apply(x_legacy, polynomials)
    y_new = EvaluatePolynomials.apply(x_new, polynomials)
    grad_output = torch.randn_like(y_new)
    grad_legacy = torch.autograd.grad(y_legacy, x_legacy, grad_output)[0]
    grad_new = torch.autograd.grad(y_new, x_new, grad_output)[0]
    torch.testing.assert_close(y_new, y_legacy, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_new, grad_legacy, rtol=2e-4, atol=2e-4)
    return {
        "forward_max_abs_error": float((y_new - y_legacy).abs().max()),
        "backward_max_abs_error": float((grad_new - grad_legacy).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--rep-ms", type=int, default=250)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    device = torch.cuda.get_device_properties(0)
    polynomials = compute_invariant_polynomial_collection(n_max=4, l_max=4)
    polynomials.set_device("cuda")
    # Materialize derivative metadata outside all timed regions.
    polynomials.get_polynomials(1)
    forward_sizes = polynomials.get_polynomials(0)[2]
    backward_sizes = polynomials.get_polynomials(1)[2]

    results = {
        "system": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": f"{device.major}.{device.minor}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__,
        },
        "workload": {
            "l_max": 4,
            "n_max": 4,
            "batch_size": BATCH_SIZE,
            "atoms_per_structure": list(ATOMS_PER_STRUCTURE),
            "points": list(POINTS),
            "polynomial_sizes": {
                "forward": forward_sizes.tolist(),
                "backward": backward_sizes.tolist(),
            },
        },
        "correctness": check_correctness(polynomials),
        "raw_kernel_us": {},
        "autograd_us": {},
    }

    for level in (0, 1):
        level_rows = {}
        for num_points in POINTS:
            prepared = prepare_level(polynomials, level, num_points, torch.float32)
            row = {
                "legacy": 1000 * bench_ms(
                    lambda p=prepared: launch_raw(p, "legacy"),
                    args.warmup_ms, args.rep_ms,
                )
            }
            for config in CONFIGS:
                point_tile, monomial_tile, num_warps, num_stages = config
                label = (
                    f"p{point_tile}_m{monomial_tile}"
                    f"_w{num_warps}_s{num_stages}"
                )
                row[label] = 1000 * bench_ms(
                    lambda p=prepared, c=config: launch_raw(p, "2d", c),
                    args.warmup_ms, args.rep_ms,
                )
            level_rows[str(num_points)] = row
        results["raw_kernel_us"][f"derivative_level_{level}"] = level_rows

    for num_points in POINTS:
        row = {}
        for label, function in (
            ("legacy", _EvaluatePolynomialsLegacy),
            ("2d_autotuned", EvaluatePolynomials),
        ):
            x = torch.randn(num_points, 25, device="cuda", requires_grad=True)

            def forward(fn=function, input_x=x):
                return fn.apply(input_x, polynomials)

            def forward_backward(fn=function, input_x=x):
                y = fn.apply(input_x, polynomials)
                torch.autograd.grad(y.sum(), input_x)

            row[f"{label}_forward"] = 1000 * bench_ms(
                forward, args.warmup_ms, args.rep_ms
            )
            row[f"{label}_forward_backward"] = 1000 * bench_ms(
                forward_backward, args.warmup_ms, args.rep_ms
            )
        results["autograd_us"][str(num_points)] = row

    for section in ("raw_kernel_us", "autograd_us"):
        print(f"\n{section}")
        print(json.dumps(results[section], indent=2))
    print("\ncorrectness", json.dumps(results["correctness"], indent=2))
    print("\npolynomial_sizes", json.dumps(results["workload"]["polynomial_sizes"], indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
