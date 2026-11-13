"""Expected layer degree and clustering under construction rules, from group sizes and local density.

A rule turns memberships into ties; for the group rules of `rules.capped_group_edges`
and the household-neighbour rule this module gives the expected degree of a member and the expected local
clustering inside the layer, as functions of group size n, cap k and, for the neighbour rule, the expected
number mu of other households within the radius.

Exact results (checked against simulation in tests/test_theory.py):

- full linking of a group of n: degree n - 1, clustering 1;
- k random out-links each, union ("random"): a pair is tied with p = 1 - (1 - k/(n-1))^2, so the expected
  degree is (n - 1) p = 2k - k^2/(n - 1); clustering is close to p (within a few per cent, slightly below);
- k n / 2 random pairs ("random_pairs"): degree k, clustering k / (n - 1);
- the household-neighbour rule with radius r on a Poisson pattern: the share of households with no neighbour
  is exp(-mu).

Tabulated results (Monte Carlo with the package's own rule code, scripts/tabulate_theory.py):

- k nearest by home, union ("nearest"): degree / k and clustering as functions of k and n, for two home
  patterns; there is no density parameter because nearest-neighbour choices do not depend on scale;
- the capped symmetric draw ("random_capped"): degree just below k; clustering below k / (n - 1), by 1 to 2 per
  cent at k = 100 and by about 16 per cent at k = 10 with n = 60 (the sequential draw closes fewer triangles);
- the household-neighbour rule: linked households L(mu) and clustering C(mu), with and without the radius.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from functools import lru_cache

import numpy as np

TABLES = os.environ.get("BUILTNET_THEORY_TABLES") or pathlib.Path(__file__).resolve().parent / "data" / "theory_tables.json"


@lru_cache(maxsize=4)
def tables(path: str | None = None) -> dict:
    with open(path or TABLES) as fh:
        return json.load(fh)


def _interp_n(rows: list[dict], n: float, field: str) -> float:
    """Interpolate a tabulated quantity in log(n - k - 1); hold the last value beyond the table."""
    rows = sorted(rows, key=lambda r: r["n"])
    k = rows[0]["k"]
    x = np.log([r["n"] - k - 1 for r in rows])
    y = np.array([r[field] for r in rows])
    if n <= k + 1:
        return float("nan")
    return float(np.interp(math.log(n - k - 1), x, y))


def _cap_of(cap, n: int) -> int | None:
    if cap is None:
        return None
    return int(cap(n) if callable(cap) else cap)


def group_degree(n: int, cap, select: str = "nearest", full_if_leq: int | None = None,
                 shape: str = "square", table_path: str | None = None) -> float:
    """Expected number of ties of a member of a group of n under a group rule."""
    if n < 2:
        return 0.0
    k = _cap_of(cap, n)
    if k is None or n <= max(k + 1, full_if_leq or 0) or k >= n - 1:
        return float(n - 1)
    if k <= 0:
        return 0.0
    if select == "random":
        return 2.0 * k - k * k / (n - 1)
    if select == "random_pairs":
        return float(min(k, n - 1))
    if select == "random_capped":
        rows = [r for r in tables(table_path)["group"] if r["select"] == "random_capped" and r["k"] == 100]
        return k * _interp_n(rows, n * 100 / k, "degree") / 100 if rows else float(k)
    if select == "nearest":
        return k * _nearest(n, k, shape, "degree", table_path)
    raise ValueError(select)


def group_clustering(n: int, cap, select: str = "nearest", full_if_leq: int | None = None,
                     shape: str = "square", table_path: str | None = None) -> float:
    """Expected local clustering inside the layer of a member of a group of n (members with two or more ties)."""
    if n < 3:
        return float("nan")
    k = _cap_of(cap, n)
    if k is None or n <= max(k + 1, full_if_leq or 0) or k >= n - 1:
        return 1.0
    if select == "random":
        q = k / (n - 1)
        return 1.0 - (1.0 - q) ** 2
    if select == "random_pairs":
        return k / (n - 1)
    if select == "random_capped":
        # The sequential draw closes fewer triangles than a uniform k-regular graph: tabulated at k = 100
        # (about 1 to 2 per cent below k / (n - 1)); for other caps k / (n - 1) is an upper bound.
        if k == 100:
            rows = [r for r in tables(table_path)["group"] if r["select"] == "random_capped" and r["k"] == 100]
            if rows:
                return _interp_n(rows, n, "clustering")
        return k / (n - 1)
    if select == "nearest":
        return _nearest(n, k, shape, "clustering", table_path)
    raise ValueError(select)


def _nearest(n: int, k: int, shape: str, field: str, table_path: str | None) -> float:
    """Tabulated degree / k or clustering for k nearest; tables exist for k = 10 and k = 100."""
    rows_by_k = {}
    for r in tables(table_path)["group"]:
        if r["select"] == "nearest" and r["shape"] == shape:
            rows_by_k.setdefault(r["k"], []).append(r)
    ks = sorted(rows_by_k)
    kk = min(ks, key=lambda x: abs(math.log(x / k)))
    rows = rows_by_k[kk]
    if field == "degree":
        rows = [dict(r, degree=r["degree"] / r["k"]) for r in rows]
    # Rescale n so that the excess over the cap keeps its ratio to the cap.
    return _interp_n(rows, kk + 1 + (n - k - 1) * kk / k, field)


# --- Population level ---------------------------------------------------------------------------------------

def layer_mean_degree(group_sizes: np.ndarray, population: int, cap, select: str = "nearest",
                      full_if_leq: int | None = None, drop_over: int | None = None, shape: str = "square",
                      table_path: str | None = None) -> float:
    """Mean layer degree over a population of `population` people, from the sizes of its groups.

    Each member of a group of n contributes group_degree(n). People in no group contribute 0.
    """
    sizes, counts = np.unique(np.asarray(group_sizes, dtype=np.int64), return_counts=True)
    total = 0.0
    for n, c in zip(sizes, counts):
        if drop_over is not None and n > drop_over:
            continue
        total += c * n * group_degree(int(n), cap, select, full_if_leq, shape, table_path)
    return total / population


def layer_mean_clustering(group_sizes: np.ndarray, cap, select: str = "nearest", full_if_leq: int | None = None,
                          drop_over: int | None = None, shape: str = "square", table_path: str | None = None) -> float:
    """Mean within-layer clustering over members of groups of three or more (each member weighted equally)."""
    sizes, counts = np.unique(np.asarray(group_sizes, dtype=np.int64), return_counts=True)
    num = den = 0.0
    for n, c in zip(sizes, counts):
        if n < 3 or (drop_over is not None and n > drop_over):
            continue
        num += c * n * group_clustering(int(n), cap, select, full_if_leq, shape, table_path)
        den += c * n
    return num / den if den else float("nan")


def neighbour_linked(mu: np.ndarray | float, radius: bool = True, table_path: str | None = None) -> np.ndarray:
    """Expected number of linked households under the 10-nearest-household rule, by local mu."""
    rows = sorted((r for r in tables(table_path)["neighbour"] if (r["radius_m"] is not None) == radius),
                  key=lambda r: r["mu"])
    if not radius:
        return np.full(np.shape(mu), float(np.mean([r["linked"] for r in rows])))
    x = np.array([0.0] + [r["mu"] for r in rows])
    y = np.array([0.0] + [r["linked"] for r in rows])
    return np.interp(mu, x, y)


def neighbour_none(mu: np.ndarray | float, radius: bool = True) -> np.ndarray:
    """Share of households with no neighbour tie: exp(-mu) with the radius, 0 without."""
    mu = np.asarray(mu, dtype=float)
    return np.exp(-mu) if radius else np.zeros_like(mu)
