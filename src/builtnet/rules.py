"""Edge-construction rules: from affiliation tables to network layers.

Each function returns an array of node pairs. A pair (a, b) means "a is linked
to b" as the rule states it; `edges_to_csr` then symmetrises by union, so a tie
exists if either person names the other.

The rules are generic operators. Named rule-sets that imitate published
register networks are assembled from them in `rulesets.py`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

# Random displacement (metres) added before a nearest-neighbour search so that
# exact distance ties, such as households in one building, are broken at random.
_TIE_JITTER_M = 1e-6


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Members of each group, in CSR form.

    `groups` holds a group id per node, with negative ids meaning "no group".
    Returns (indptr, members, sizes) over the groups that occur.
    """
    groups = np.asarray(groups)
    nodes = np.flatnonzero(groups >= 0)
    if nodes.size == 0:
        return np.zeros(1, dtype=np.int64), nodes, np.zeros(0, dtype=np.int64)
    _, inv, sizes = np.unique(groups[nodes], return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    members = nodes[order]
    indptr = np.concatenate([[0], np.cumsum(sizes)])
    return indptr, members, sizes


def clique_edges(groups: np.ndarray) -> np.ndarray:
    """Link every pair of nodes that share a group id."""
    indptr, members, sizes = _group_index(groups)
    out = []
    for s in np.unique(sizes):
        if s < 2:
            continue
        starts = indptr[:-1][sizes == s]
        block = members[starts[:, None] + np.arange(s)[None, :]]  # (groups, s)
        i, j = np.triu_indices(int(s), 1)
        out.append(np.stack([block[:, i].ravel(), block[:, j].ravel()], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(out)


def _random_others(s: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """For each of s members, k distinct other members chosen uniformly.

    Returns local indices with shape (s, k).
    """
    out = np.empty((s, k), dtype=np.int64)
    chunk = max(1, int(2_000_000 // s))
    for start in range(0, s, chunk):
        stop = min(s, start + chunk)
        keys = rng.random((stop - start, s))
        keys[np.arange(stop - start), np.arange(start, stop)] = np.inf  # never self
        out[start:stop] = np.argpartition(keys, k - 1, axis=1)[:, :k]
    return out


def _capped_symmetric_draw(s: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Random ties in which nobody ends with more than `cap`.

    People are taken in random order; each draws partners uniformly among the
    others who still have fewer than `cap` ties and are not yet tied to them,
    until they reach `cap` or nobody is left. Most people end with exactly
    `cap` ties and a few with fewer. Returns local index pairs.
    """
    deg = np.zeros(s, dtype=np.int64)
    tied: list[list[int]] = [[] for _ in range(s)]
    out = []
    for i in rng.permutation(s):
        need = cap - deg[i]
        if need <= 0:
            continue
        ok = deg < cap
        ok[i] = False
        ok[tied[i]] = False
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            continue
        pick = cand if cand.size <= need else rng.choice(cand, size=int(need), replace=False)
        deg[pick] += 1
        deg[i] += pick.size
        tied[i].extend(pick.tolist())
        for j in pick.tolist():
            tied[j].append(int(i))
        out.append(np.stack([np.full(pick.size, i), pick], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


def _random_pairs(s: int, n_ties: int, rng: np.random.Generator) -> np.ndarray:
    """`n_ties` distinct unordered pairs among s members, uniformly at random."""
    total = s * (s - 1) // 2
    n_ties = min(n_ties, total)
    codes = rng.choice(total, size=n_ties, replace=False)
    # Decode a pair index into (i, j) with i < j, rows of the upper triangle.
    i = (s - 2 - np.floor(np.sqrt(-8.0 * codes + 4.0 * s * (s - 1) - 7) / 2.0 - 0.5)).astype(np.int64)
    j = codes + i + 1 - s * (s - 1) // 2 + (s - i) * ((s - i) - 1) // 2
    return np.stack([i, j], axis=1)


SELECT_MODES = ("nearest", "random", "random_capped", "random_pairs")


def capped_group_edges(
    groups: np.ndarray,
    cap,
    select: str = "nearest",
    coords: np.ndarray | None = None,
    full_if_leq: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Link group members, limiting how many each person is linked to.

    `cap` is a number, a function of group size returning a number, or None
    (every group fully linked). A group whose members could all be linked
    within the cap (size <= cap + 1, or size <= `full_if_leq` if given) is
    fully linked. In larger groups `select` says how ties are chosen:

      "nearest"        each person names the `cap` others nearest by `coords`,
                       ties broken at random (Dutch colleague rule);
      "random"         each person names `cap` others uniformly at random
                       (Swedish out-link draw);
      "random_capped"  random ties in which nobody exceeds `cap`
                       (Danish colleague rule, reading a in docs/rule-sets.md);
      "random_pairs"   cap * size / 2 pairs drawn uniformly at random, so the
                       mean number of ties is `cap` (Danish rule, reading b).

    For "nearest" and "random" a tie exists if either person names the other,
    so a person can end with more than `cap` ties.
    """
    if cap is None:
        return clique_edges(groups)
    if select not in SELECT_MODES:
        raise ValueError(f"unknown select: {select}")
    if rng is None:
        raise ValueError("a random generator is required when a cap is set")
    if select == "nearest" and coords is None:
        raise ValueError("coords are required for select='nearest'")
    fixed = not callable(cap)
    if fixed and full_if_leq is not None and full_if_leq < cap + 1:
        raise ValueError("full_if_leq must be at least cap + 1")

    indptr, members, sizes = _group_index(groups)
    small = np.full(len(groups), -1, dtype=np.int64)
    out = []
    for g, s in enumerate(sizes):
        s = int(s)
        mem = members[indptr[g] : indptr[g + 1]]
        k = int(cap if fixed else cap(s))
        limit = max(k + 1, full_if_leq or 0)
        if s <= limit or k >= s - 1:
            small[mem] = g
            continue
        if k <= 0:
            continue
        if select == "nearest":
            xy = coords[mem] + rng.normal(scale=_TIE_JITTER_M, size=(s, 2))
            _, nn = cKDTree(xy).query(xy, k=k + 1)
            local = np.stack([np.repeat(np.arange(s), k), nn[:, 1:].ravel()], axis=1)
        elif select == "random":
            picks = _random_others(s, k, rng)
            local = np.stack([np.repeat(np.arange(s), k), picks.ravel()], axis=1)
        elif select == "random_capped":
            local = _capped_symmetric_draw(s, k, rng)
        else:
            local = _random_pairs(s, k * s // 2, rng)
        out.append(mem[local])
    out.append(clique_edges(small))
    return np.concatenate(out)


def _cross_expand(
    ga: np.ndarray, gb: np.ndarray, indptr: np.ndarray, members: np.ndarray
) -> np.ndarray:
    """All (member of ga[i], member of gb[i]) node pairs for each group pair."""
    sa = indptr[ga + 1] - indptr[ga]
    sb = indptr[gb + 1] - indptr[gb]
    cnt = sa * sb
    total = int(cnt.sum())
    if total == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pair = np.repeat(np.arange(ga.size), cnt)
    offset = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ia = offset // sb[pair]
    ib = offset % sb[pair]
    a = members[indptr[ga[pair]] + ia]
    b = members[indptr[gb[pair]] + ib]
    return np.stack([a, b], axis=1)


def household_neighbour_edges(
    household: np.ndarray,
    household_xy: np.ndarray,
    k: int = 10,
    radius: float | None = 50.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Link the members of each household to the members of nearby households.

    Each household names its `k` closest other households, within `radius`
    metres if a radius is given. Distance ties are broken at random.
    `household` holds a household id per node, numbered 0..H-1, and
    `household_xy` the coordinates of each household.
    """
    if rng is None:
        raise ValueError("a random generator is required")
    n_h = household_xy.shape[0]
    if n_h < 2:
        return np.zeros((0, 2), dtype=np.int64)
    kk = min(k, n_h - 1)
    xy = household_xy + rng.normal(scale=_TIE_JITTER_M, size=household_xy.shape)
    bound = np.inf if radius is None else float(radius)
    dist, nn = cKDTree(xy).query(xy, k=kk + 1, distance_upper_bound=bound)
    src = np.repeat(np.arange(n_h), kk)
    dst = nn[:, 1:].ravel()
    ok = np.isfinite(dist[:, 1:].ravel())
    src, dst = src[ok], dst[ok]

    order = np.argsort(household, kind="stable")
    sizes = np.bincount(household, minlength=n_h)
    indptr = np.concatenate([[0], np.cumsum(sizes)])
    return _cross_expand(src, dst, indptr, order)


def compose_groups(*keys: np.ndarray) -> np.ndarray:
    """One group id per node from several keys; -1 wherever any key is negative."""
    keys = [np.asarray(k) for k in keys]
    valid = np.ones(keys[0].shape, dtype=bool)
    for k in keys:
        valid &= k >= 0
    out = np.full(keys[0].shape, -1, dtype=np.int64)
    if valid.any():
        stacked = np.stack([k[valid] for k in keys], axis=1)
        _, inv = np.unique(stacked, axis=0, return_inverse=True)
        out[valid] = inv.ravel()
    return out


KIN_TYPES = ("parent", "sibling", "grandparent", "aunt_uncle", "cousin", "co_parent", "partner", "in_law")


def _pairs(m: sp.spmatrix) -> np.ndarray:
    """Off-diagonal non-zero positions of a sparse matrix, as node pairs."""
    c = m.tocoo()
    keep = (c.row != c.col) & (c.data != 0)
    return np.stack([c.row[keep], c.col[keep]], axis=1).astype(np.int64)


def kin_edges(
    parent: np.ndarray, partner: np.ndarray, types=KIN_TYPES, late_parent: np.ndarray | None = None
) -> np.ndarray:
    """Kin ties derived from legal parent links and partners.

    `parent` has shape (n, 2) and `partner` shape (n,), with -1 for none.
    `late_parent` (n,) optionally names a parent who is on record but no
    longer in the population: such a parent has no ties, but still makes
    their children siblings and their grandchildren cousins.
    `types` selects relations; each name covers both directions:

      "parent"       parent and child
      "sibling"      people who share at least one parent
      "grandparent"  parent of a parent, and grandchild
      "aunt_uncle"   sibling of a parent, and niece or nephew
      "cousin"       child of a parent's sibling
      "co_parent"    the two parents of a child
      "partner"      partners
      "in_law"       a partner's parents and siblings, and a sibling's partner

    This is how the Dutch and Danish offices build the family layer: every
    relation is extrapolated from parent-child pairs (docs/rule-sets.md).
    """
    unknown = set(types) - set(KIN_TYPES)
    if unknown:
        raise ValueError(f"unknown kin types: {sorted(unknown)}")
    parent = np.asarray(parent, dtype=np.int64).reshape(-1, 2)
    partner = np.asarray(partner, dtype=np.int64)
    n = parent.shape[0]
    child = np.repeat(np.arange(n), 2)
    par = parent.ravel()
    ok = par >= 0
    child, par = child[ok], par[ok]
    size = n
    if late_parent is not None:  # late parents are extra nodes after the n people
        late_parent = np.asarray(late_parent, dtype=np.int64)
        has_late = np.flatnonzero(late_parent >= 0)
        if has_late.size:
            child = np.concatenate([child, has_late])
            par = np.concatenate([par, n + late_parent[has_late]])
            size = n + int(late_parent.max()) + 1
    P = sp.csr_matrix((np.ones(child.size, dtype=np.int32), (child, par)), shape=(size, size))  # child -> parent
    P.data[:] = 1
    has = np.flatnonzero(partner >= 0)
    W = sp.csr_matrix((np.ones(has.size, dtype=np.int32), (has, partner[has])), shape=(size, size))

    S = (P @ P.T).tocsr()  # share a parent
    S.setdiag(0)
    S.eliminate_zeros()
    S.data[:] = 1

    out = []
    if "parent" in types:
        out.append(_pairs(P))
    if "sibling" in types:
        out.append(_pairs(S))
    if "grandparent" in types:
        out.append(_pairs(P @ P))
    if "aunt_uncle" in types or "cousin" in types:
        A = (P @ S).tocsr()  # child -> parent's sibling
        A.data[:] = 1
        if "aunt_uncle" in types:
            out.append(_pairs(A))
        if "cousin" in types:
            out.append(_pairs(A @ P.T))
    if "co_parent" in types:
        both = (parent[:, 0] >= 0) & (parent[:, 1] >= 0)
        out.append(parent[both])
    if "partner" in types:
        out.append(np.stack([has, partner[has]], axis=1))
    if "in_law" in types:
        out.append(_pairs(W @ P))
        out.append(_pairs(W @ S))
        out.append(_pairs(S @ W))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    e = np.concatenate(out)
    return e[(e[:, 0] != e[:, 1]) & (e[:, 0] < n) & (e[:, 1] < n)]  # people only
