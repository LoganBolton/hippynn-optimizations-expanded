import pytest
import torch

from hippynn.layers.hiplayers.tensors import HopInvariantLayerTorch, TensorExtractor
from hippynn.layers.hiplayers.invariants import HopInvariantLayer, compute_invariant_polynomial_collection, default_invariants_list, split_invariant
from hippynn.layers.hiplayers.interactions import _invariant_counts


def test_default_invariant_definitions_parse():
    for invariant in default_invariants_list:
        indices, tensors = split_invariant(invariant)
        assert len(indices) == len(tensors)

        for index, tensor in zip(indices, tensors):
            assert len(index) == {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4}[tensor]


def test_polynomial_invariant_counts_match_interaction_lmax4():
    for n_max in range(1, 5):
        polyCollection = compute_invariant_polynomial_collection(n_max, 4)
        _, _, polynomial_sizes, _ = polyCollection.get_polynomials()

        assert len(polynomial_sizes) == _invariant_counts[n_max, 4]


def test_polynomial_invariant_count_lmax3_nmax5():
    polyCollection = compute_invariant_polynomial_collection(5, 3)
    _, _, polynomial_sizes, _ = polyCollection.get_polynomials()

    assert len(polynomial_sizes) == _invariant_counts[5, 3]


def test_hop_invariant_layer_metadata_tracks_counts_and_sizes():
    invariant_layer = HopInvariantLayer(n_max=4, l_max=4)
    metadata = invariant_layer.invariant_metadata()

    assert metadata["n_invariants"] == _invariant_counts[4, 4]
    assert len(metadata["invariant_codes"]) == metadata["n_invariants"]
    assert len(metadata["polynomial_sizes"]) == metadata["n_invariants"]
    assert metadata["polynomial_sizes"].min().item() > 0


def evaluate_polynomial_collection_torch(x, polyCollection):
    coefs, terms, polynomial_sizes, _ = polyCollection.get_polynomials()

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


def test_polynomial_invariants_are_rotation_invariant_lmax4():
    n_point = 11
    torch.manual_seed(0)

    rhats = torch.randn(n_point, 3)
    rhats = rhats / rhats.norm(dim=1, keepdim=True)

    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    tensor_extractor = TensorExtractor(l_max=4)
    tensor_features = torch.cat(tensor_extractor(rhats), dim=1)
    rotated_tensor_features = torch.cat(tensor_extractor(rhats @ rotation), dim=1)

    polyCollection = compute_invariant_polynomial_collection(n_max=4, l_max=4)
    invariants = evaluate_polynomial_collection_torch(tensor_features, polyCollection)
    rotated_invariants = evaluate_polynomial_collection_torch(rotated_tensor_features, polyCollection)

    assert torch.allclose(invariants, rotated_invariants, rtol=1e-4, atol=1e-4)

def test_polynomial_invariants():

    n_point = 3
    torch.manual_seed(0)

    try:
        import triton
        triton_available = True
    except:
        triton_available = False

    if triton_available and torch.cuda.is_available():

        from hippynn.custom_kernels.poly_triton import EvaluatePolynomials

        configurations = [(l_max, n_max) for l_max in range(5) for n_max in range(1, 5)]
        configurations.append((3, 5))
        for l_max, n_max in configurations:
            n_tensor_comp = (l_max+1)**2
            tensor_features = torch.randn((n_point, n_tensor_comp), requires_grad=True, device='cuda')

            polyCollection = compute_invariant_polynomial_collection(n_max, l_max)
            polyCollection.set_device('cuda')
            invars_poly = EvaluatePolynomials.apply(tensor_features, polyCollection)

            invars_torch = evaluate_polynomial_collection_torch(tensor_features, polyCollection)

            assert torch.allclose(invars_poly, invars_torch, rtol=1e-4, atol=1e-4)

            tensor_features = tensor_features.to(torch.float64)

            assert torch.autograd.gradcheck(EvaluatePolynomials.apply, (tensor_features, polyCollection))
            assert torch.autograd.gradgradcheck(EvaluatePolynomials.apply, (tensor_features, polyCollection))

def test_invariants_wrapper():

    n_point = 3
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    try:
        import triton
        triton_available = True
    except:
        triton_available = False

    configurations = [(l_max, n_max) for l_max in range(5) for n_max in range(1, 5)]
    if triton_available and torch.cuda.is_available():
        configurations.append((3, 5))

    for l_max, n_max in configurations:
        n_tensor_comp = (l_max+1)**2
        tensor_features = torch.randn((n_point, n_tensor_comp), requires_grad=True, device=device)

        invariantLayer = HopInvariantLayer(n_max, l_max)
        invariantLayer = invariantLayer.to(device)
        invars_poly = invariantLayer(tensor_features)

        if triton_available and torch.cuda.is_available():
            polyCollection = compute_invariant_polynomial_collection(n_max, l_max)
            polyCollection.set_device(device)
            invars_torch = evaluate_polynomial_collection_torch(tensor_features, polyCollection)
        else:
            torchInvariantLayer = HopInvariantLayerTorch(n_max, l_max)
            torchInvariantLayer = torchInvariantLayer.to(device)
            invars_torch = torchInvariantLayer(tensor_features)

        assert torch.allclose(invars_poly, invars_torch, rtol=1e-4, atol=1e-4)

        # During this check tensor features need to be float32 because the old
        # HopInvariantLayerTorch only supports float32. Thus, gradcheck is only
        # available when Triton is active.
        if triton_available and torch.cuda.is_available():
            tensor_features = tensor_features.to(torch.float64)

            assert torch.autograd.gradcheck(invariantLayer, (tensor_features,))
            assert torch.autograd.gradgradcheck(invariantLayer, (tensor_features,))
