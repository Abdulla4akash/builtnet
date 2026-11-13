import numpy as np
import pytest

from builtnet import rules
from builtnet import world as W
from builtnet.rulesets import NL_2018, NL_2018_K, NL_SERIES_K, SE_LIKE_K, build

# A hand-made family.
#   0, 1  grandparents, partners
#   2, 3  their children (siblings); 4 is the partner of 2
#   5, 6  children of 2 and 4;  7 child of 3 alone;  8 unrelated
PARENT = np.array([[-1, -1], [-1, -1], [0, 1], [0, 1], [-1, -1], [2, 4], [2, 4], [3, -1], [-1, -1]])
PARTNER = np.array([1, 0, 4, -1, 2, -1, -1, -1, -1])


def pairs(types):
    e = rules.kin_edges(PARENT, PARTNER, types)
    return {tuple(sorted(p)) for p in e.tolist()}


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("parent", {(0, 2), (1, 2), (0, 3), (1, 3), (2, 5), (4, 5), (2, 6), (4, 6), (3, 7)}),
        ("sibling", {(2, 3), (5, 6)}),
        ("grandparent", {(0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)}),
        ("aunt_uncle", {(3, 5), (3, 6), (2, 7)}),
        ("cousin", {(5, 7), (6, 7)}),
        ("co_parent", {(0, 1), (2, 4)}),
        ("partner", {(0, 1), (2, 4)}),
        ("in_law", {(0, 4), (1, 4), (3, 4)}),
    ],
)
def test_each_relation_on_a_hand_made_family(kind, expected):
    assert pairs((kind,)) == expected


def test_unrelated_person_has_no_kin_and_unknown_types_are_rejected():
    assert all(8 not in p for p in pairs(rules.KIN_TYPES))
    with pytest.raises(ValueError):
        rules.kin_edges(PARENT, PARTNER, ("godparent",))


def test_late_parents_make_siblings_and_cousins_but_have_no_ties():
    # 0 and 1 share a parent who is on record but no longer in the population;
    # 2 is the child of 0, 3 the child of 1, 4 has a different late parent.
    parent = np.array([[-1, -1], [-1, -1], [0, -1], [1, -1], [-1, -1]])
    partner = np.full(5, -1)
    late = np.array([0, 0, -1, -1, 1])

    def got(types):
        return {tuple(sorted(p)) for p in rules.kin_edges(parent, partner, types, late).tolist()}

    assert got(("sibling",)) == {(0, 1)}
    assert got(("aunt_uncle",)) == {(1, 2), (0, 3)}
    assert got(("cousin",)) == {(2, 3)}
    assert got(("grandparent",)) == set()  # the grandparent is not a person in the network
    assert got(rules.KIN_TYPES) == {(0, 2), (1, 3), (0, 1), (1, 2), (0, 3), (2, 3)}
    assert rules.kin_edges(parent, partner, ("sibling",)).shape[0] == 0  # without the record, no tie


CFG = W.WorldConfig(n_households=1500)


def test_generated_kin_links_are_consistent():
    w = W.generate(CFG, 5)
    for c in (0, 1):
        p = w.parent[:, c]
        has = p >= 0
        assert np.all(p[has] != np.flatnonzero(has))
        assert np.all(w.age[p[has]] - w.age[has] >= 16)
    assert np.all(w.parent[w.parent[:, 0] < 0, 1] < 0)  # a second parent only with a first
    has = w.partner >= 0
    assert np.array_equal(w.partner[w.partner[has]], np.flatnonzero(has))
    assert np.all(w.household[w.partner[has]] == w.household[has])
    kids = w.role == W.CHILD
    assert np.all(w.parent[kids, 0] >= 0)
    assert np.all(w.household[w.parent[kids, 0]] == w.household[kids])
    assert np.all((w.age[w.role == W.ADULT] >= 18) & (w.age[w.role == W.ADULT] <= 66))
    assert np.all(w.age[w.role == W.SENIOR] >= 67)
    away = (w.parent[:, 0] >= 0) & (w.household[np.maximum(w.parent[:, 0], 0)] != w.household)
    assert away.sum() > 100  # some parents live elsewhere
    assert np.bincount(w.parent[w.parent >= 0]).max() <= 2 * CFG.kin_max_children
    has = np.flatnonzero(w.partner >= 0)  # partners never share a parent
    mate = w.partner[has]
    for a in (0, 1):
        for b in (0, 1):
            assert not np.any((w.parent[has, a] >= 0) & (w.parent[has, a] == w.parent[mate, b]))
    assert not np.any((w.late_parent[has] >= 0) & (w.late_parent[has] == w.late_parent[mate]))
    late = w.late_parent >= 0
    assert late.sum() > 100 and not np.any(late & (w.parent[:, 0] >= 0))  # only for people with no parent here
    assert np.bincount(w.late_parent[late]).max() <= CFG.kin_max_children
    assert np.all(w.role[late] != W.CHILD)


def test_kin_settings_leave_every_other_table_unchanged():
    a = W.generate(CFG, 6)
    b = W.generate(W.WorldConfig(n_households=1500, kin_parent_share=0.0, partner_share=0.2), 6)
    for name in ("xy", "household", "building", "income", "school", "school_class", "employer", "site"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    kids = a.role == W.CHILD
    assert np.array_equal(a.age[kids], b.age[kids])
    assert not np.array_equal(a.parent, b.parent)


def test_kin_layer_is_added_without_changing_the_other_layers():
    w = W.generate(CFG, 7)
    plain, with_kin = build(w, NL_2018, 0), build(w, NL_2018_K, 0)
    assert "kin" not in plain.layers and "kin" in with_kin.layers
    for name in plain.layers:
        assert (plain.layers[name] != with_kin.layers[name]).nnz == 0
    assert with_kin.layers["kin"].nnz > 0


def test_rule_sets_see_different_kin():
    w = W.generate(CFG, 8)
    nl, series, se = (build(w, rs, 0).layers["kin"] for rs in (NL_2018_K, NL_SERIES_K, SE_LIKE_K))
    assert (nl > series).nnz == 0 and series.nnz > nl.nnz  # the yearly series only adds relations
    old = np.flatnonzero((w.age > 37) & (w.parent[:, 0] >= 0) & (w.household[np.maximum(w.parent[:, 0], 0)] != w.household))
    assert old.size > 0
    assert all(se[i, w.parent[i, 0]] == 0 for i in old[:50])  # not seen from co-residence
    assert all(nl[i, w.parent[i, 0]] == 1 for i in old[:50])
    assert se.nnz < nl.nnz
    # Siblings through a late parent: on record in the Dutch file, unseen from co-residence above the cut-off.
    sib = np.flatnonzero((w.late_parent >= 0) & (w.age > 37))
    pairs = [(i, j) for i in sib[:200] for j in np.flatnonzero(w.late_parent == w.late_parent[i]) if j != i]
    assert len(pairs) > 0
    assert all(nl[i, j] == 1 and se[i, j] == 0 for i, j in pairs)
