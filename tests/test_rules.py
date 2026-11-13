import numpy as np
import pytest

from builtnet.network import edges_to_csr
from builtnet import rules


def degrees(edges, n):
    return np.asarray(edges_to_csr(edges, n).sum(axis=1)).ravel()


def test_edges_to_csr_is_symmetric_simple_and_deduplicated():
    a = edges_to_csr(np.array([[0, 1], [1, 0], [0, 1], [2, 2], [1, 2]]), 4)
    assert (a != a.T).nnz == 0
    assert a.diagonal().sum() == 0
    assert a.nnz == 4  # ties 0-1 and 1-2, stored both ways
    assert set(a.data.tolist()) == {1}


def test_clique_edges_links_exactly_the_pairs_sharing_a_group():
    groups = np.array([0, 0, 0, 1, 1, -1, 2, 5, 5, 5, 5])
    e = rules.clique_edges(groups)
    assert len(e) == 3 + 1 + 0 + 6
    assert np.all(groups[e[:, 0]] == groups[e[:, 1]])
    assert np.all(groups[e] >= 0)
    assert len({tuple(sorted(p)) for p in e.tolist()}) == len(e)


def test_cap_none_and_small_groups_are_fully_linked():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(5), 8)
    xy = rng.random((40, 2))
    full = degrees(rules.clique_edges(groups), 40)
    for select in ("nearest", "random"):
        e = rules.capped_group_edges(groups, cap=10, select=select, coords=xy, rng=rng)
        assert np.array_equal(degrees(e, 40), full)


def test_threshold_is_cap_plus_one_by_default():
    rng = np.random.default_rng(1)
    xy = rng.random((300, 2))
    # 101 members with cap 100: everyone is linked to all 100 others.
    g101 = np.zeros(101, dtype=int)
    assert np.all(degrees(rules.capped_group_edges(g101, 100, "random", xy[:101], rng=rng), 101) == 100)
    # 102 members: each names 100 of 101 others, so someone is left out.
    g102 = np.zeros(102, dtype=int)
    d = degrees(rules.capped_group_edges(g102, 100, "random", xy[:102], rng=rng), 102)
    assert d.min() >= 100 and d.max() <= 101


def test_nearest_cap_on_a_line_picks_the_adjacent_points():
    rng = np.random.default_rng(2)
    n, cap = 50, 4
    xy = np.stack([np.arange(n) * 10.0, np.zeros(n)], axis=1)
    a = edges_to_csr(rules.capped_group_edges(np.zeros(n, int), cap, "nearest", xy, rng=rng), n)
    # An interior point's four nearest are the two on each side.
    for i in range(5, 45):
        assert set(a[i].indices.tolist()) >= {i - 2, i - 1, i + 1, i + 2}
    assert a[0, 1] == 1 and a[0, 4] == 1  # the end point names 1..4


def test_nearest_cap_is_invariant_to_relabelling():
    n, cap = 400, 20
    xy = np.random.default_rng(3).random((n, 2)) * 1000
    perm = np.random.default_rng(4).permutation(n)
    a = edges_to_csr(
        rules.capped_group_edges(np.zeros(n, int), cap, "nearest", xy, rng=np.random.default_rng(5)), n
    )
    b = edges_to_csr(
        rules.capped_group_edges(np.zeros(n, int), cap, "nearest", xy[perm], rng=np.random.default_rng(6)), n
    )
    # Node j of the relabelled world is node perm[j] of the original.
    assert (a[perm][:, perm] != b).nnz == 0


def test_random_cap_degree_matches_its_expectation_and_is_seeded():
    n, cap = 2000, 100
    g = np.zeros(n, int)
    e1 = rules.capped_group_edges(g, cap, "random", rng=np.random.default_rng(7))
    e2 = rules.capped_group_edges(g, cap, "random", rng=np.random.default_rng(7))
    assert np.array_equal(e1, e2)
    d = degrees(e1, n)
    p = cap / (n - 1)
    expected = (n - 1) * (1 - (1 - p) ** 2)  # tie if either names the other
    assert d.min() >= cap
    assert abs(d.mean() - expected) < 1.0
    assert np.all(e1[:, 0] != e1[:, 1])


def test_full_if_leq_below_cap_plus_one_is_rejected():
    with pytest.raises(ValueError):
        rules.capped_group_edges(np.zeros(10, int), 5, "random", full_if_leq=3, rng=np.random.default_rng(0))


def test_household_neighbours_respect_radius_k_and_membership():
    # Five households on a line, 30 m apart; households 0 and 1 have two members.
    hh_xy = np.stack([np.arange(5) * 30.0, np.zeros(5)], axis=1)
    household = np.array([0, 0, 1, 1, 2, 3, 4])
    rng = np.random.default_rng(8)
    a = edges_to_csr(rules.household_neighbour_edges(household, hh_xy, k=10, radius=50.0, rng=rng), 7)
    # Within 50 m only the adjacent household qualifies.
    assert a[0, 2] == 1 and a[0, 3] == 1 and a[1, 2] == 1 and a[1, 3] == 1
    assert a[0, 1] == 0  # household members are not neighbours of each other
    assert a[0, 4] == 0  # 60 m away
    assert a[4, 5] == 1 and a[5, 6] == 1 and a[4, 6] == 0
    # Without a radius and with k=1 every household names its single nearest.
    b = edges_to_csr(rules.household_neighbour_edges(household, hh_xy, k=1, radius=None, rng=rng), 7)
    assert np.all(np.asarray(b.sum(axis=1)).ravel() >= 1)


def test_colocated_households_are_neighbours_and_ties_break_at_random():
    # Thirty households at one point (one building), k=10.
    hh_xy = np.zeros((30, 2))
    household = np.arange(30)
    picks = []
    for seed in range(2):
        e = rules.household_neighbour_edges(household, hh_xy, k=10, radius=50.0, rng=np.random.default_rng(seed))
        assert len(e) == 300  # each names exactly ten
        picks.append({tuple(p) for p in e.tolist()})
    assert picks[0] != picks[1]


def test_compose_groups():
    g = rules.compose_groups(np.array([0, 0, 1, 1, -1]), np.array([3, 3, 3, 4, 3]))
    assert g[0] == g[1] and g[2] != g[3] and g[2] != g[0] and g[4] == -1


def test_capped_symmetric_draw_never_exceeds_the_cap():
    n, cap = 600, 100
    e = rules.capped_group_edges(np.zeros(n, int), cap, "random_capped", rng=np.random.default_rng(9))
    d = degrees(e, n)
    assert d.max() == cap
    assert (d == cap).mean() > 0.95  # nearly everyone reaches the cap
    assert len({tuple(sorted(p)) for p in e.tolist()}) == len(e)  # no pair drawn twice


def test_random_pairs_gives_the_requested_number_of_distinct_ties():
    n, cap = 700, 100
    e = rules.capped_group_edges(np.zeros(n, int), cap, "random_pairs", rng=np.random.default_rng(10))
    assert len(e) == cap * n // 2
    assert np.all(e[:, 0] < e[:, 1]) and e.min() == 0 and e.max() == n - 1
    assert len({tuple(p) for p in e.tolist()}) == len(e)
    assert degrees(e, n).mean() == cap
    # Every pair is reachable: with all pairs requested the group is complete.
    full = rules._random_pairs(40, 40 * 39 // 2, np.random.default_rng(0))
    assert len({tuple(p) for p in full.tolist()}) == 40 * 39 // 2


def test_cap_can_depend_on_group_size():
    groups = np.concatenate([np.zeros(8, int), np.ones(300, int)])
    e = rules.capped_group_edges(groups, lambda n: 11, "random", rng=np.random.default_rng(11))
    d = degrees(e, 308)
    assert np.all(d[:8] == 7)  # eleven links would cover everyone: fully linked
    assert d[8:].min() >= 11 and 20 < d[8:].mean() < 22


def test_swedish_link_function_matches_the_published_values():
    from builtnet.rulesets import se_k
    assert [se_k(n) for n in (12, 100, 999)] == [11, 11, 11]
    assert [se_k(n) for n in (1000, 2500, 10000)] == [17, 38, 143]
