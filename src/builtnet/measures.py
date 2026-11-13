"""Comparing the structural positions of two groups, or of two populations."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

DECILES = np.arange(0.1, 1.0, 0.1)


def _clean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=float)[mask]
    return v[np.isfinite(v)]


def gap_profile(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float | np.ndarray]:
    """Group A minus group B, in the descriptor's own units.

    Returns the difference in means, in medians and at each decile, plus the
    number of people with a defined value in each group.
    """
    xa, xb = _clean(x, a), _clean(x, b)
    if xa.size == 0 or xb.size == 0:
        nan = float("nan")
        return {"mean": nan, "median": nan, "deciles": np.full(DECILES.size, nan), "n_a": xa.size, "n_b": xb.size}
    return {
        "mean": float(xa.mean() - xb.mean()),
        "median": float(np.median(xa) - np.median(xb)),
        "deciles": np.quantile(xa, DECILES) - np.quantile(xb, DECILES),
        "n_a": int(xa.size),
        "n_b": int(xb.size),
    }


def scaled_wasserstein(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 distance between the groups, in pooled interquartile ranges."""
    xa, xb = _clean(x, a), _clean(x, b)
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    q1, q3 = np.quantile(np.concatenate([xa, xb]), [0.25, 0.75])
    if q3 == q1:
        return float("nan")
    return float(wasserstein_distance(xa, xb) / (q3 - q1))


def distinguishability(
    features: dict[str, np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    blocks: np.ndarray,
    seed: int = 0,
    n_splits: int = 5,
) -> float:
    """How well position profiles alone tell group A from group B.

    Returns 2 * AUC - 1 from a cross-validated gradient-boosted classifier:
    0 means the groups are structurally indistinguishable, 1 perfectly
    separable. `blocks` holds an id per person (for example the household);
    people in one block are never split between training and test folds.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    use = a | b
    x = np.column_stack([np.asarray(features[k], dtype=float)[use] for k in sorted(features)])
    y = a[use].astype(int)
    g = np.asarray(blocks)[use]
    scores = np.full(y.size, np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(x, y, g):
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=seed)
        model.fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    return float(2 * roc_auc_score(y, scores) - 1)


def construction_share(m11: float, m12: float, m21: float, m22: float) -> dict[str, float]:
    """Split a naive between-population difference into two parts.

    m[p][r] is a summary (for example a mean descriptor) for population p
    wired with rule-set r. The naive comparison sets population 2 under its
    own rules against population 1 under its own:

        naive = m22 - m11 = population effect + construction effect   (exact)

    The population effect averages the like-for-like differences over both
    rule-sets; the construction effect averages the rule differences over
    both populations.
    """
    population = 0.5 * ((m21 - m11) + (m22 - m12))
    construction = 0.5 * ((m12 - m11) + (m22 - m21))
    interaction = (m22 - m12) - (m21 - m11)
    denom = abs(population) + abs(construction)
    return {
        "naive": m22 - m11,
        "population": population,
        "construction": construction,
        "interaction": interaction,
        "share": abs(construction) / denom if denom > 0 else float("nan"),
    }


def group_means(x: np.ndarray, labels: np.ndarray, levels) -> np.ndarray:
    """Mean of x over people with a defined value, for each label in `levels`."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    out = np.full(len(levels), np.nan)
    for i, lev in enumerate(levels):
        m = ok & (labels == lev)
        if m.any():
            out[i] = x[m].mean()
    return out


def adjusted_gap(x: np.ndarray, a: np.ndarray, b: np.ndarray, strata: np.ndarray) -> float:
    """Mean of group A minus group B after giving B the composition of A.

    `strata` holds a stratum id per person (for example age band crossed with
    urban density). Each person in B is weighted by the ratio of A's share to
    B's share in their stratum. Strata with nobody from B are left out of A.
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    a, b = a & ok, b & ok
    if not a.any() or not b.any():
        return float("nan")
    n_s = int(strata.max()) + 1
    ca = np.bincount(strata[a], minlength=n_s).astype(float)
    cb = np.bincount(strata[b], minlength=n_s).astype(float)
    use = (ca > 0) & (cb > 0)
    if not use.any():
        return float("nan")
    sa = np.bincount(strata[a], weights=x[a], minlength=n_s)
    sb = np.bincount(strata[b], weights=x[b], minlength=n_s)
    w = ca[use] / ca[use].sum()
    return float((w * (sa[use] / ca[use] - sb[use] / cb[use])).sum())


def label_shuffle_pvalue(
    x: np.ndarray,
    unit: np.ndarray,
    unit_label: np.ndarray,
    high,
    low,
    n_perm: int,
    rng: np.random.Generator,
    strata: np.ndarray | None = None,
) -> dict[str, float]:
    """Permutation test of a gap in means between two labels inside one network.

    Labels belong to units (a person, or a household): `unit` gives each
    person's unit and `unit_label` each unit's label. Labels are shuffled
    across units, within `strata` if given, and the gap `mean(high) -
    mean(low)` is recomputed. Returns the observed gap and a two-sided
    p-value. The test assumes units are exchangeable, which a network built
    from shared contexts does not guarantee; that is what is being checked.
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    n_u = unit_label.shape[0]
    total = np.bincount(unit[ok], weights=x[ok], minlength=n_u)
    count = np.bincount(unit[ok], minlength=n_u).astype(float)

    def gap(lab: np.ndarray) -> float:
        hi, lo = lab == high, lab == low
        ch, cl = count[hi].sum(), count[lo].sum()
        if ch == 0 or cl == 0:
            return float("nan")
        return float(total[hi].sum() / ch - total[lo].sum() / cl)

    observed = gap(unit_label)
    if strata is None:
        groups = [np.arange(n_u)]
    else:
        order = np.argsort(strata, kind="stable")
        cuts = np.flatnonzero(np.diff(strata[order])) + 1
        groups = np.split(order, cuts)
    hits = 0
    lab = unit_label.copy()
    for _ in range(n_perm):
        for g in groups:
            lab[g] = unit_label[rng.permutation(g)]
        if abs(gap(lab)) >= abs(observed):
            hits += 1
    return {"gap": observed, "p": (1 + hits) / (1 + n_perm)}
