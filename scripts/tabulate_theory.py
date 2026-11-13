"""Reference curves for builtnet.theory, by Monte Carlo with the package's own rule code.

    python scripts/tabulate_theory.py OUT.json [--quick]

What is tabulated (all within one layer):

1. Group rules. For a single group of n members and a cap k: the expected degree of a member and the mean local
   clustering (members with at least two ties), for each selection mode of `rules.capped_group_edges`:
   "random" (k out-links, union), "random_capped", "random_pairs", and "nearest" (k nearest by home, union).
   For "nearest" two home patterns: uniform in a square and a round Gaussian cloud, to show how much the shape
   of the pattern matters. Scale does not matter for nearest-neighbour choices, so no density parameter exists.
2. The household-neighbour rule. Households placed as a Poisson pattern with mu expected households within the
   radius r of a point; each names its 10 nearest within r (2018 rule) or with no radius (yearly series).
   With one person per household: the expected number of linked households L(mu), the share with none P0(mu),
   and the mean local clustering C(mu).

Every curve is computed with `builtnet.rules`, so it describes the implementation the experiments use.
The bundled tables (src/builtnet/data/theory_tables.json) were made with seed 20261113 and the default target.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp

from builtnet import rules
from builtnet.network import edges_to_csr


def layer_stats(edges: np.ndarray, n: int) -> tuple[float, float, float]:
    """Mean degree, mean local clustering over nodes with degree >= 2, share with degree 0."""
    a = edges_to_csr(edges, n)
    a = ((a + a.T) > 0).astype(np.float64).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    deg = np.asarray(a.sum(axis=1)).ravel()
    tri = np.asarray((a @ a).multiply(a).sum(axis=1)).ravel() / 2.0
    ok = deg >= 2
    clus = float(np.mean(tri[ok] / (deg[ok] * (deg[ok] - 1) / 2))) if ok.any() else float("nan")
    return float(deg.mean()), clus, float(np.mean(deg == 0))


def group_curve(select: str, k: int, n: int, shape: str, rng: np.random.Generator, target: int) -> dict:
    reps = max(3, math.ceil(target / n))
    d, c, sizes = [], [], []
    for _ in range(reps):
        groups = np.zeros(n, dtype=np.int64)
        coords = None
        if select == "nearest":
            coords = rng.random((n, 2)) * 1000.0 if shape == "square" else rng.normal(scale=1000.0, size=(n, 2))
        e = rules.capped_group_edges(groups, k, select, coords, None, rng)
        md, mc, _ = layer_stats(e, n)
        d.append(md)
        c.append(mc)
    return {"select": select, "shape": shape, "k": k, "n": n, "reps": reps,
            "degree": float(np.mean(d)), "degree_se": float(np.std(d) / math.sqrt(reps)),
            "clustering": float(np.nanmean(c)), "clustering_se": float(np.nanstd(c) / math.sqrt(reps))}


def neighbour_curve(mu: float, radius: bool, rng: np.random.Generator, households: int) -> dict:
    r = 50.0
    lam = mu / (math.pi * r * r)  # households per square metre
    side = math.sqrt(households / lam)
    m = rng.poisson(households)
    xy = rng.random((m, 2)) * side
    e = rules.household_neighbour_edges(np.arange(m), xy, 10, r if radius else None, rng)
    a = edges_to_csr(e, m)
    a = ((a + a.T) > 0).astype(np.float64).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    deg = np.asarray(a.sum(axis=1)).ravel()
    tri = np.asarray((a @ a).multiply(a).sum(axis=1)).ravel() / 2.0
    reach = r if radius else 3.0 * math.sqrt(10.0 / (lam * math.pi))  # stay clear of the edges of the square
    inner = np.all((xy > reach) & (xy < side - reach), axis=1)
    ok = inner & (deg >= 2)
    return {"mu": mu, "radius_m": r if radius else None, "households": int(m), "inner": int(inner.sum()),
            "linked": float(deg[inner].mean()), "none": float(np.mean(deg[inner] == 0)),
            "clustering": float(np.mean(tri[ok] / (deg[ok] * (deg[ok] - 1) / 2))) if ok.any() else float("nan")}


def main() -> None:
    out_path = sys.argv[1]
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(20261113)
    target = 20_000 if quick else 200_000
    t0 = time.time()
    tables = {"group": [], "neighbour": [], "seed": 20261113, "target_member_samples": target}

    def ngrid(k: int) -> list[int]:
        g = [k + 2, k + 5, k + 10, k + 25, int(1.5 * k), 2 * k, 3 * k, 5 * k, 10 * k, 20 * k]
        return sorted({x for x in g if x <= (600 if quick else 5000)})

    for k in (11, 17, 31, 73, 100):
        for n in ngrid(k):
            tables["group"].append(group_curve("random", k, n, "-", rng, target))
    for select in ("random_capped", "random_pairs"):
        for n in ngrid(100):
            tables["group"].append(group_curve(select, 100, n, "-", rng, target))
    for shape in ("square", "gauss"):
        for k in (10, 100):
            for n in ngrid(k):
                tables["group"].append(group_curve("nearest", k, n, shape, rng, target))
        print(f"group curves done ({shape}) after {time.time() - t0:.0f} s", flush=True)

    for mu in (0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50):
        for radius in (True, False):
            tables["neighbour"].append(neighbour_curve(mu, radius, rng, 4_000 if quick else 40_000))
    tables["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as fh:
        json.dump(tables, fh, indent=1)
    print(f"wrote {out_path} in {tables['seconds']} s")


if __name__ == "__main__":
    main()
