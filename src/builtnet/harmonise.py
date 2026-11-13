"""Harmonisation: estimating a like-for-like contrast from unlike networks.

Two populations are wired with different rule-sets. The target is the
difference between them under one common rule-set. The methods here are the
pieces; `experiments/grid.py` assembles them into M0 to M5 of the plan.
"""

from __future__ import annotations

import numpy as np


def _weighted_bin_means(x: np.ndarray, bins: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    count = np.bincount(bins, minlength=n_bins).astype(float)
    total = np.bincount(bins, weights=x, minlength=n_bins)
    mean = np.full(n_bins, np.nan)
    ok = count > 0
    mean[ok] = total[ok] / count[ok]
    return mean, count


def matched_degree_contrast(
    x1: np.ndarray, deg1: np.ndarray, x2: np.ndarray, deg2: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Mean of x in group 2 minus group 1, compared at the same degree (M2).

    Degree bins are quantiles of the pooled degrees. Within each bin the two
    means are differenced, and the differences are averaged with the pooled
    bin sizes as weights. Bins that are empty on either side are left out;
    `coverage` is the share of people in the bins that were used.
    """
    ok1 = np.isfinite(x1) & np.isfinite(deg1)
    ok2 = np.isfinite(x2) & np.isfinite(deg2)
    x1, deg1, x2, deg2 = x1[ok1], deg1[ok1], x2[ok2], deg2[ok2]
    if x1.size == 0 or x2.size == 0:
        return {"contrast": float("nan"), "coverage": 0.0}
    pooled = np.concatenate([deg1, deg2])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)[1:-1]))
    b1 = np.searchsorted(edges, deg1, side="right")
    b2 = np.searchsorted(edges, deg2, side="right")
    nb = edges.size + 1
    m1, c1 = _weighted_bin_means(x1, b1, nb)
    m2, c2 = _weighted_bin_means(x2, b2, nb)
    use = (c1 > 0) & (c2 > 0)
    if not use.any():
        return {"contrast": float("nan"), "coverage": 0.0}
    w = (c1 + c2)[use]
    return {
        "contrast": float(((m2 - m1)[use] * w).sum() / w.sum()),
        "coverage": float(w.sum() / (c1 + c2).sum()),
    }


def simulator_corrected(naive: float, sim_other_rules: float, sim_common_rules: float) -> float:
    """Naive contrast minus a simulated construction effect (M4).

    `sim_other_rules` and `sim_common_rules` are the same summary on one
    simulated population wired with the other network's rule-set and with the
    common rule-set. Their difference estimates what the rules alone add.
    """
    return naive - (sim_other_rules - sim_common_rules)


def error_removed(estimate: float, naive: float, truth: float) -> float:
    """Share of the naive error that a method removes: 1 is perfect, 0 no better, below 0 worse."""
    base = abs(naive - truth)
    if base == 0:
        return float("nan")
    return 1.0 - abs(estimate - truth) / base
