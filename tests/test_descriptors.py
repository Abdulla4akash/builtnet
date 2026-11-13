import numpy as np
import pytest

from builtnet import descriptors as D
from builtnet import rules
from builtnet.network import MultilayerNetwork, edges_to_csr


def random_multilayer(n=120, layers=3, p=0.06, seed=0, overlap=True):
    rng = np.random.default_rng(seed)
    out = {}
    for layer in range(layers):
        mask = np.triu(rng.random((n, n)) < p, 1)
        out[f"L{layer}"] = edges_to_csr(np.argwhere(mask), n)
    if overlap:  # copy part of the first layer into the second
        e = np.argwhere(np.triu(out["L0"].toarray(), 1))
        extra = e[rng.random(len(e)) < 0.5]
        out["L1"] = edges_to_csr(np.concatenate([np.argwhere(np.triu(out["L1"].toarray(), 1)), extra]), n)
    return MultilayerNetwork(n, out)


def test_triangle_counts_agree_between_the_two_sparse_methods():
    a = random_multilayer(seed=1).flattened()
    assert np.array_equal(D.triangles_per_node(a), D.triangles_per_node_sparse(a))


def test_clique_layer_has_clustering_one_and_degree_size_minus_one():
    groups = np.repeat(np.arange(4), [3, 5, 2, 7])
    net = MultilayerNetwork(17, {"g": edges_to_csr(rules.clique_edges(groups), 17)})
    d = D.position_descriptors(net)
    sizes = np.array([3, 5, 2, 7])[groups]
    assert np.array_equal(d["degree_total"], sizes - 1)
    assert np.allclose(d["clustering"][sizes >= 3], 1.0)
    assert np.all(np.isnan(d["clustering"][sizes == 2]))  # undefined, not zero


def test_excess_closure_hand_example():
    # Ego 0 has contacts 1,2 in layer A (a triangle) and 3,4 in layer B (a triangle).
    # Layer C links 1 and 3, closing one cross-layer wedge.
    a = edges_to_csr(np.array([[0, 1], [0, 2], [1, 2]]), 5)
    b = edges_to_csr(np.array([[0, 3], [0, 4], [3, 4]]), 5)
    c = edges_to_csr(np.array([[1, 3]]), 5)
    out = D.closure(MultilayerNetwork(5, {"A": a, "B": b, "C": c}))
    # P = C(4,2) = 6; T_pure = 2; T_unique = 3.
    assert out["c_pure"][0] == pytest.approx(2 / 6)
    assert out["c_actual"][0] == pytest.approx(3 / 6)
    assert out["excess_closure"][0] == pytest.approx(0.25)
    # Without layer C nothing closes across layers.
    out0 = D.closure(MultilayerNetwork(5, {"A": a, "B": b}))
    assert out0["excess_closure"][0] == pytest.approx(0.0)


def test_closure_variants_agree_without_multiplex_ties():
    rng = np.random.default_rng(2)
    n = 120
    pairs = np.argwhere(np.triu(rng.random((n, n)) < 0.15, 1))
    which = rng.integers(0, 3, size=len(pairs))  # each tie goes to exactly one layer
    net = MultilayerNetwork(n, {f"L{i}": edges_to_csr(pairs[which == i], n) for i in range(3)})
    assert net.multiplexity().data.max() == 1
    ref = D.closure(net, "paper")
    for variant in ("mlnlib", "tiepair"):
        other = D.closure(net, variant)
        for key in ref:
            assert np.allclose(ref[key], other[key], equal_nan=True), (variant, key)


def test_tiepair_variant_is_bounded_with_multiplex_ties():
    net = random_multilayer(n=150, seed=6, overlap=True)
    assert net.multiplexity().data.max() > 1
    c = D.closure(net, "tiepair")
    ok = np.isfinite(c["excess_closure"])
    assert ok.any()
    assert np.all(c["c_pure"][ok] <= c["c_actual"][ok] + 1e-12)
    assert np.all((c["c_actual"][ok] >= 0) & (c["c_actual"][ok] <= 1))
    assert np.all((c["excess_closure"][ok] >= -1e-12) & (c["excess_closure"][ok] <= 1 + 1e-12))
    # A triangle present in two layers: the paper's text gives a negative value,
    # mlnlib a value above one, and the tie-pair count gives one.
    tri = np.array([[0, 1], [0, 2], [1, 2]])
    twice = MultilayerNetwork(3, {"A": edges_to_csr(tri, 3), "B": edges_to_csr(tri, 3)})
    assert D.closure(twice, "paper")["excess_closure"][0] == pytest.approx(-0.5)
    assert D.closure(twice, "mlnlib")["excess_closure"][0] == pytest.approx(3.0)
    assert D.closure(twice, "tiepair")["excess_closure"][0] == pytest.approx(1.0)


def test_mlnlib_variant_matches_mlnlib():
    mln = pytest.importorskip("mlnlib")
    pd = pytest.importorskip("pandas")
    net = random_multilayer(n=150, seed=3, overlap=True)
    names = net.layer_names
    bits = sum((net.layers[name] * (2**i)) for i, name in enumerate(names)).tocsr().astype(np.int64)
    nodes = pd.DataFrame({"id": np.arange(net.n), "label": np.arange(net.n)})
    layers = pd.DataFrame(
        {"layer": range(len(names)), "label": names, "binary": [2**i for i in range(len(names))]}
    )
    ref = mln.MultiLayerNetwork(nodes=nodes, edges=bits, layers=layers).get_excess_closure()
    ref_c = np.array([ref["clustering_coefficient"][i] for i in range(net.n)])
    ref_x = np.array([ref["excess_closure"][i] for i in range(net.n)])
    mine = D.closure(net, "mlnlib")
    # mlnlib reports 0 where the ratio is undefined; this code reports NaN.
    assert np.allclose(np.nan_to_num(mine["c_actual"]), ref_c)
    assert np.allclose(np.nan_to_num(mine["excess_closure"]), ref_x)


def test_descriptors_are_invariant_to_relabelling_and_nan_for_inactive():
    net = random_multilayer(seed=4)
    perm = np.random.default_rng(5).permutation(net.n)
    relabelled = MultilayerNetwork(net.n, {k: v[perm][:, perm].tocsr() for k, v in net.layers.items()})
    a, b = D.position_descriptors(net), D.position_descriptors(relabelled)
    for key in a:
        assert np.allclose(a[key][perm], b[key], equal_nan=True), key
    net.active[:10] = False
    c = D.position_descriptors(net)
    assert all(np.all(np.isnan(v[:10])) for v in c.values())


def test_participation_is_zero_for_one_layer_egos_and_one_for_even_spread():
    a = edges_to_csr(np.array([[0, 1], [0, 2], [3, 4]]), 6)
    b = edges_to_csr(np.array([[0, 4], [0, 5]]), 6)
    d = D.position_descriptors(MultilayerNetwork(6, {"A": a, "B": b}))
    assert d["participation"][0] == pytest.approx(1.0)  # two ties in each layer
    assert d["participation"][1] == pytest.approx(0.0)  # all ties in layer A


def test_position_descriptors_can_add_the_bounded_closure_variant():
    net = random_multilayer(seed=5)
    d = D.position_descriptors(net, also_closure=("tiepair",))
    assert np.array_equal(d["excess_closure"], D.closure(net, "paper")["excess_closure"], equal_nan=True)
    assert np.array_equal(d["excess_closure_tiepair"], D.closure(net, "tiepair")["excess_closure"], equal_nan=True)
    ok = np.isfinite(d["excess_closure_tiepair"])
    assert np.all(d["excess_closure_tiepair"][ok] >= -1e-12) and np.all(d["excess_closure_tiepair"][ok] <= 1 + 1e-12)
    with pytest.raises(ValueError):
        D.closures(net, ("nonsense",))
