import numpy as np
import pytest

from builtnet import measures as M


def test_gap_profile_and_wasserstein_on_a_pure_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    a = np.arange(4000) < 2000
    x[a] += 2.0
    x[:5] = np.nan  # undefined values are ignored, not treated as zero
    g = M.gap_profile(x, a, ~a)
    assert g["mean"] == pytest.approx(2.0, abs=0.1)
    assert g["median"] == pytest.approx(2.0, abs=0.1)
    assert np.allclose(g["deciles"], 2.0, atol=0.25)
    assert g["n_a"] == 1995
    iqr = np.subtract(*np.nanquantile(x, [0.75, 0.25]))
    assert M.scaled_wasserstein(x, a, ~a) == pytest.approx(2.0 / iqr, rel=0.1)


def test_construction_share_identity_and_extremes():
    r = M.construction_share(10.0, 14.0, 13.0, 19.0)
    assert r["naive"] == pytest.approx(r["population"] + r["construction"])
    assert r["population"] == pytest.approx(4.0)
    assert r["construction"] == pytest.approx(5.0)
    assert r["interaction"] == pytest.approx(2.0)
    assert r["share"] == pytest.approx(5 / 9)
    assert M.construction_share(1, 1, 5, 5)["share"] == 0.0  # rules change nothing
    assert M.construction_share(1, 5, 1, 5)["share"] == 1.0  # populations identical


def test_distinguishability_is_near_zero_for_exchangeable_groups_and_high_for_separated():
    rng = np.random.default_rng(1)
    n = 3000
    f = {"u": rng.normal(size=n), "v": rng.normal(size=n)}
    a = rng.random(n) < 0.5
    blocks = np.arange(n) // 3
    assert abs(M.distinguishability(f, a, ~a, blocks)) < 0.1
    f["u"] = f["u"] + 3.0 * a
    assert M.distinguishability(f, a, ~a, blocks) > 0.9
