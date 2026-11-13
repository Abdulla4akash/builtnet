"""Cards, the real-tables interface and the igraph-free triangle count."""

import dataclasses
import itertools

import numpy as np
import pytest
import scipy.sparse as sp

from builtnet import cards
from builtnet import rulesets as R
from builtnet import scenarios as S
from builtnet import world as W
from builtnet.descriptors import position_descriptors, triangles_per_node
from builtnet.tables import world_from_tables

PRESETS = {"nl-2018": R.NL_2018_K, "nl-series": R.NL_SERIES_K, "dk": R.DK_K, "dk-pairs": R.DK_PAIRS_K, "se": R.SE_LIKE_K}


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_each_card_reproduces_its_preset(name):
    rs = cards.load(name)
    preset = PRESETS[name]
    for f in dataclasses.fields(R.RuleSet):
        if f.name == "name":
            continue
        assert getattr(rs, f.name) == getattr(preset, f.name), (name, f.name)
    assert rs.name == preset.name


def test_variant_cards_change_only_what_they_say():
    a, b = cards.load("nl-2018"), cards.load("nl-2018-classcap")
    assert [f.name for f in dataclasses.fields(R.RuleSet) if getattr(a, f.name) != getattr(b, f.name)] == ["name", "school_cap"]
    c = cards.load("se-code-nowindow")
    assert c.school_max_years_since is None and c.work_cap is R.se_k_code
    assert cards.provenance("se-code")["school_drop_over"]["status"] == "documented"  # inherited from se
    assert "SE25" in cards.describe("se-code-nowindow")["all_sources"]
    assert set(PRESETS) <= set(cards.available())
    d, t = cards.load("dk"), cards.load("dk-twoway")
    assert [f.name for f in dataclasses.fields(R.RuleSet) if getattr(d, f.name) != getattr(t, f.name)] == ["name", "work_select"]
    assert t.work_select == "random" and cards.provenance("dk-twoway")["work_select"]["status"] == "inferred"


def test_triangle_count_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 40
    a = (rng.random((n, n)) < 0.2).astype(int)
    a = np.triu(a, 1)
    a = a + a.T
    want = np.zeros(n, dtype=int)
    for i, j, k in itertools.combinations(range(n), 3):
        if a[i, j] and a[j, k] and a[i, k]:
            want[[i, j, k]] += 1
    assert np.array_equal(triangles_per_node(sp.csr_matrix(a)), want)


def test_world_from_tables_reproduces_the_synthetic_build(monkeypatch):
    tiny = dict(n_households=300, region_km=3.0, n_cores=2, employer_tail=0.8, employer_max_size=160,
                multi_site_min_size=40, multi_site_prob=1.0, site_mean_size=20.0)
    monkeypatch.setitem(S.SCALES, "tiny", tiny)
    w = W.generate(S.base("tiny"), 5)
    t = world_from_tables(household=w.household, home_xy=w.xy, household_xy=w.household_xy, employer=w.employer,
                          site=w.site, school=w.school, school_level=w.school_level, school_year=w.school_year,
                          school_class=w.school_class, plot=w.plot, municipality=w.municipality, age=w.age,
                          parent=w.parent, partner=w.partner, late_parent=w.late_parent, edu_level=w.edu_level,
                          grad_years_ago=w.grad_years_ago, income=w.income)
    for rs in (R.NL_2018_K, R.DK_K):
        d1 = position_descriptors(R.build(w, rs, 3))
        d2 = position_descriptors(R.build(t, rs, 3))
        for k in ("degree_total", "degree_work", "degree_school", "degree_neighbour", "degree_kin", "clustering"):
            assert np.allclose(np.nan_to_num(d1[k]), np.nan_to_num(d2[k])), (rs.name, k)
