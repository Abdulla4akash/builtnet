"""Position descriptors computed on a built multilayer network.

Every descriptor is one interpretable number per person. Values that are
undefined for a person (for example clustering with fewer than two contacts)
are NaN, never silently zero.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .network import MultilayerNetwork


def triangles_per_node(a: sp.csr_matrix, batch: int = 2000) -> np.ndarray:
    """Number of triangles each node belongs to, in a simple undirected graph (sparse matrix products)."""
    n = a.shape[0]
    a = a.tocsr()
    out = np.zeros(n, dtype=np.int64)
    for start in range(0, n, batch):
        rows = a[start : start + batch]
        paths = (rows @ a).multiply(rows)
        out[start : start + batch] = np.rint(np.asarray(paths.sum(axis=1)).ravel() / 2).astype(np.int64)
    return out


def triangles_per_node_sparse(a: sp.csr_matrix, batch: int = 2000) -> np.ndarray:
    """Same count by sparse matrix products. Slower; kept as an independent check."""
    n = a.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for start in range(0, n, batch):
        rows = a[start : start + batch]
        paths = (rows @ a).multiply(rows)
        out[start : start + batch] = np.asarray(paths.sum(axis=1)).ravel() // 2
    return out


def weighted_triangles_per_node(m: sp.csr_matrix, batch: int = 2000) -> np.ndarray:
    """Sum over alter pairs {v, w} of m[u,v] * m[v,w] * m[w,u]."""
    n = m.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for start in range(0, n, batch):
        rows = m[start : start + batch]
        paths = (rows @ m).multiply(rows)
        out[start : start + batch] = np.asarray(paths.sum(axis=1)).ravel() // 2
    return out


def closed_tie_pairs_per_node(m: sp.csr_matrix, flat: sp.csr_matrix, batch: int = 2000) -> np.ndarray:
    """Sum over alter pairs {v, w} of m[u,v] * m[u,w], counting pairs with v-w tied."""
    n = m.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for start in range(0, n, batch):
        rows = m[start : start + batch]
        paths = (rows @ flat).multiply(rows)
        out[start : start + batch] = np.asarray(paths.sum(axis=1)).ravel() // 2
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


CLOSURE_VARIANTS = ("paper", "mlnlib", "tiepair")


def closures(net: MultilayerNetwork, variants=("paper",)) -> dict[str, dict[str, np.ndarray]]:
    """Pure, actual and excess closure of Bokanyi, Heemskerk and Takes (2023), in one or more variants.

    With m[u,v] the number of layers linking u and v, and k[u] the sum of u's
    layer degrees, the number of alter tie pairs around u is

        P[u] = C(k[u], 2) - sum over v of C(m[u,v], 2).

    c_pure = T_pure / P, where T_pure sums, over layers, the triangles around u
    whose three edges lie in that layer. c_actual = T / P. Excess closure is
    (c_actual - c_pure) / (1 - c_pure).

    A variant sets T:
      "paper"   unique node triplets closed in the flattened network, as the
                paper's text defines it;
      "mlnlib"  sum over alter pairs of the product of the three
                multiplexities, as PLANET-NL's mlnlib 0.3.0 computes it;
      "tiepair" number of alter tie pairs whose two alters are linked in at
                least one layer. This is the only variant guaranteed to give
                0 <= c_pure <= c_actual <= 1.
    All three agree whenever no pair of nodes is linked in more than one layer.
    When many pairs are linked in several layers, "paper" can fall below zero,
    because a triangle that lies in three layers counts three times as pure
    and once as actual. See docs/excess-closure.md.
    """
    unknown = set(variants) - set(CLOSURE_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variant: {sorted(unknown)}")
    m = net.multiplexity()
    k = np.asarray(m.sum(axis=1)).ravel().astype(np.int64)
    same_alter = m.copy()
    same_alter.data = same_alter.data * (same_alter.data - 1) // 2
    p = k * (k - 1) // 2 - np.asarray(same_alter.sum(axis=1)).ravel()

    t_pure = np.zeros(net.n, dtype=np.int64)
    for a in net.layers.values():
        t_pure += triangles_per_node(a)
    flat = m.copy()
    flat.data[:] = 1
    c_pure = _ratio(t_pure, p)

    out = {}
    for variant in variants:
        if variant == "paper":
            t = triangles_per_node(flat)
        elif variant == "mlnlib":
            t = weighted_triangles_per_node(m)
        else:
            t = closed_tie_pairs_per_node(m, flat)
        c_actual = _ratio(t, p)
        excess = np.full(net.n, np.nan)
        ok = np.isfinite(c_pure) & (c_pure < 1)
        excess[ok] = (c_actual[ok] - c_pure[ok]) / (1 - c_pure[ok])
        out[variant] = {"c_pure": c_pure, "c_actual": c_actual, "excess_closure": excess}
    return out


def closure(net: MultilayerNetwork, variant: str = "paper") -> dict[str, np.ndarray]:
    """One variant of `closures`."""
    return closures(net, (variant,))[variant]


def position_descriptors(net: MultilayerNetwork, also_closure: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """All network-side descriptors for every node.

    `also_closure` names further excess-closure variants to add as
    `excess_closure_<variant>`; the default `excess_closure` is "paper".

    degree_total       number of distinct contacts
    degree_<layer>     contacts in one layer
    clustering         local clustering coefficient on the flattened network
    excess_closure     see `closure`
    multiplex_share    share of contacts tied in two or more layers
    participation      1 when ties spread evenly over layers, 0 when all are in
                       one (Battiston, Nicosia and Latora 2014)
    alter_degree_ratio mean number of contacts of one's contacts, divided by
                       one's own
    Inactive nodes get NaN throughout.
    """
    m = net.multiplexity()
    flat = m.copy()
    flat.data[:] = 1
    deg = np.asarray(flat.sum(axis=1)).ravel().astype(np.float64)
    out: dict[str, np.ndarray] = {"degree_total": deg.copy()}

    layer_deg = {}
    for name, a in net.layers.items():
        layer_deg[name] = np.asarray(a.sum(axis=1)).ravel().astype(np.float64)
        out[f"degree_{name}"] = layer_deg[name].copy()

    tri = triangles_per_node(flat).astype(np.float64)
    out["clustering"] = _ratio(tri, deg * (deg - 1) / 2)
    both = closures(net, ("paper",) + tuple(also_closure))
    out.update(both["paper"])
    for v in also_closure:
        out[f"excess_closure_{v}"] = both[v]["excess_closure"]

    multi = m.copy()
    multi.data = (multi.data >= 2).astype(np.int32)
    out["multiplex_share"] = _ratio(np.asarray(multi.sum(axis=1)).ravel().astype(float), deg)

    n_layers = len(net.layers)
    overlap = sum(layer_deg.values()) if layer_deg else np.zeros(net.n)
    part = np.full(net.n, np.nan)
    if n_layers > 1:
        ok = overlap > 0
        sq = sum((d[ok] / overlap[ok]) ** 2 for d in layer_deg.values())
        part[ok] = n_layers / (n_layers - 1) * (1 - sq)
    out["participation"] = part

    alter_mean = _ratio(np.asarray(flat @ deg).ravel(), deg)
    out["alter_degree_ratio"] = _ratio(alter_mean, deg)

    inactive = ~net.active
    for v in out.values():
        v[inactive] = np.nan
    return out
