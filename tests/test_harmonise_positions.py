import numpy as np
import pytest

from builtnet import harmonise as H
from builtnet import measures as M
from builtnet.positions import assign_position_classes, fit_position_classes

pytest.importorskip("sklearn")  # position classes need scikit-learn (optional extra "analysis")


def test_matched_degree_removes_a_difference_that_is_only_degree():
    rng = np.random.default_rng(0)
    deg1, deg2 = rng.integers(1, 40, 20_000).astype(float), rng.integers(20, 60, 20_000).astype(float)
    x1, x2 = 1 / deg1, 1 / deg2  # x is a function of degree alone
    assert abs(x2.mean() - x1.mean()) > 0.05
    r = H.matched_degree_contrast(x1, deg1, x2, deg2, n_bins=40)
    assert abs(r["contrast"]) < 0.005
    assert 0 < r["coverage"] < 1  # only the overlapping degrees are compared


def test_matched_degree_keeps_a_real_difference_and_handles_no_overlap():
    rng = np.random.default_rng(1)
    deg = rng.integers(1, 40, 5000).astype(float)
    r = H.matched_degree_contrast(np.zeros(5000), deg, np.full(5000, 0.3), deg)
    assert r["contrast"] == pytest.approx(0.3) and r["coverage"] == pytest.approx(1.0)
    r = H.matched_degree_contrast(np.zeros(10), np.full(10, 1.0), np.ones(10), np.full(10, 9.0), n_bins=2)
    assert np.isnan(r["contrast"]) and r["coverage"] == 0.0


def test_simulator_correction_and_error_removed():
    assert H.simulator_corrected(naive=-50.0, sim_other_rules=20.0, sim_common_rules=60.0) == -10.0
    assert H.error_removed(estimate=-12.0, naive=-50.0, truth=-10.0) == pytest.approx(0.95)
    assert H.error_removed(estimate=-90.0, naive=-50.0, truth=-10.0) == pytest.approx(-1.0)
    assert np.isnan(H.error_removed(1.0, 2.0, 2.0))


def test_adjusted_gap_removes_a_pure_composition_difference():
    rng = np.random.default_rng(2)
    strata = np.r_[rng.choice(2, 6000, p=[0.8, 0.2]), rng.choice(2, 6000, p=[0.2, 0.8])]
    a = np.r_[np.ones(6000, bool), np.zeros(6000, bool)]
    x = strata * 10.0 + rng.normal(size=12000)  # depends on the stratum only
    raw = x[a].mean() - x[~a].mean()
    assert abs(raw) > 5
    assert abs(M.adjusted_gap(x, a, ~a, strata)) < 0.2


def test_label_shuffle_detects_a_gap_and_respects_units():
    rng = np.random.default_rng(3)
    n_h = 400
    unit = np.repeat(np.arange(n_h), 3)
    label = rng.integers(1, 6, n_h)
    noise = rng.normal(size=unit.size)
    r = M.label_shuffle_pvalue(noise + 0.8 * (label[unit] == 5), unit, label, 5, 1, 199, rng)
    assert r["p"] <= 0.01 and r["gap"] > 0.5
    r = M.label_shuffle_pvalue(noise, unit, label, 5, 1, 199, rng, strata=np.arange(n_h) % 4)
    assert r["p"] > 0.01
    assert M.group_means(np.array([1.0, np.nan, 3.0, 5.0]), np.array([1, 1, 1, 2]), (1, 2, 3)).tolist()[:2] == [2.0, 5.0]


def test_position_classes_are_fixed_by_the_reference():
    rng = np.random.default_rng(4)
    n = 3000
    desc = {
        "degree_total": np.r_[rng.poisson(5, n // 2), rng.poisson(80, n // 2)].astype(float),
        "clustering": np.r_[rng.uniform(0.7, 1, n // 2), rng.uniform(0, 0.3, n // 2)],
    }
    desc["degree_total"][:10] = np.nan  # not in the network
    desc["clustering"][10:30] = np.nan  # undefined, but present
    model = fit_position_classes(desc, k=2, seed=0)
    cls = assign_position_classes(model, desc)
    assert np.all(cls[:10] == -1) and np.all(cls[10:] >= 0)
    low, high = cls[30 : n // 2], cls[n // 2 :]
    assert (low == low[0]).mean() > 0.98 and (high == high[0]).mean() > 0.98 and low[0] != high[0]
    shifted = {k: v.copy() for k, v in desc.items()}
    shifted["degree_total"][n // 2 :] = 5.0  # high-degree people lose their ties
    shifted["clustering"][n // 2 :] = 0.9
    moved = assign_position_classes(model, shifted)
    assert (moved[n // 2 :] == low[0]).mean() > 0.9  # they now fall in the other, unchanged class
