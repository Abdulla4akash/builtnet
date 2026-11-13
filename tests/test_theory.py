"""The exact formulas of builtnet.theory against the package's own rule code."""

import math

import numpy as np
import pytest

from builtnet import rules
from builtnet import theory as T
from builtnet.network import edges_to_csr


def simulate_group(n: int, k: int, select: str, reps: int, seed: int = 1, coords: bool = False):
    rng = np.random.default_rng(seed)
    deg, clus = [], []
    for _ in range(reps):
        xy = rng.random((n, 2)) * 1000 if coords else None
        e = rules.capped_group_edges(np.zeros(n, dtype=np.int64), k, select, xy, None, rng)
        a = edges_to_csr(e, n).astype(float)
        d = np.asarray(a.sum(axis=1)).ravel()
        tri = np.asarray((a @ a).multiply(a).sum(axis=1)).ravel() / 2
        ok = d >= 2
        deg.append(d.mean())
        clus.append(np.mean(tri[ok] / (d[ok] * (d[ok] - 1) / 2)))
    return float(np.mean(deg)), float(np.mean(clus))


@pytest.mark.parametrize("n,k", [(15, 11), (40, 11), (120, 17)])
def test_random_out_links_degree_is_exact_and_clustering_close_to_tie_density(n, k):
    d, c = simulate_group(n, k, "random", reps=300)
    assert d == pytest.approx(T.group_degree(n, k, "random"), rel=0.01)
    assert d == pytest.approx(2 * k - k * k / (n - 1), rel=0.01)
    assert c == pytest.approx(T.group_clustering(n, k, "random"), abs=0.03)


@pytest.mark.parametrize("n", [30, 80])
def test_random_pairs_degree_is_the_cap_and_clustering_the_density(n):
    d, c = simulate_group(n, 10, "random_pairs", reps=300)
    assert d == pytest.approx(10.0)
    assert c == pytest.approx(10 / (n - 1), abs=0.01)


def test_groups_within_the_cap_are_fully_linked():
    assert T.group_degree(5, 100, "nearest") == 4
    assert T.group_clustering(5, 100, "random") == 1.0
    assert T.group_degree(101, 100, "random") == 100  # n <= cap + 1
    assert T.group_degree(7, None) == 6
    assert T.group_degree(1, 100) == 0.0


def test_capped_symmetric_draw_stays_at_the_cap_and_below_the_uniform_clustering():
    d, c = simulate_group(60, 10, "random_capped", reps=100)
    assert 9.5 < d <= 10.0
    # the sequential draw closes fewer triangles than a uniform 10-regular graph (about 0.14 against 0.17)
    assert 0.75 * 10 / 59 < c <= 10 / 59 + 0.005
    assert T.group_clustering(60, 10, "random_capped") == pytest.approx(10 / 59)  # an upper bound for k != 100


def test_layer_mean_degree_weights_each_member():
    sizes = np.array([3, 3, 1, 50])
    # members of the two triples have 2 ties, the single person none, the 50 have 2k - k^2/49 with k = 11
    want = (6 * 2 + 50 * (22 - 121 / 49)) / 100
    assert T.layer_mean_degree(sizes, 100, 11, "random") == pytest.approx(want)
    assert T.layer_mean_degree(sizes, 100, 11, "random", drop_over=10) == pytest.approx(12 / 100)


def test_no_neighbour_share_is_exp_minus_mu_on_a_poisson_pattern():
    rng = np.random.default_rng(3)
    mu, r = 2.0, 50.0
    lam = mu / (math.pi * r * r)
    side = math.sqrt(20000 / lam)
    m = rng.poisson(20000)
    xy = rng.random((m, 2)) * side
    e = rules.household_neighbour_edges(np.arange(m), xy, 10, r, rng)
    deg = np.asarray(edges_to_csr(e, m).sum(axis=1)).ravel()
    inner = np.all((xy > r) & (xy < side - r), axis=1)
    assert np.mean(deg[inner] == 0) == pytest.approx(float(T.neighbour_none(mu)), abs=0.01)
    assert float(T.neighbour_none(mu, radius=False)) == 0.0


def test_nearest_needs_the_tables():
    if not T.TABLES or not __import__("pathlib").Path(T.TABLES).exists():
        pytest.skip("theory tables missing")
    for n in (150, 500, 2000):
        d = T.group_degree(n, 100, "nearest")
        assert 100 <= d <= 200
        assert 0.5 < T.group_clustering(n, 100, "nearest") <= 1.0
