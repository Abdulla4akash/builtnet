"""Multilayer network container.

Node indices are stable across rule-sets: a node that a rule-set excludes keeps
its index and simply has no edges, so the same person can be compared across
networks built from one population.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


def edges_to_csr(edges: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency matrix from an array of node pairs.

    Pairs may be given in either order, may repeat, and are treated as
    undirected: a directed nomination (A names B) becomes the tie A-B. Self
    loops are dropped.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(rows.size, dtype=np.int32)
    a = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    a.sum_duplicates()
    a.data[:] = 1
    a.sort_indices()
    return a


@dataclass
class MultilayerNetwork:
    """Undirected multilayer network on a fixed node set."""

    n: int
    layers: dict[str, sp.csr_matrix] = field(default_factory=dict)
    active: np.ndarray | None = None  # nodes that exist under the rule-set

    def __post_init__(self) -> None:
        if self.active is None:
            self.active = np.ones(self.n, dtype=bool)

    @property
    def layer_names(self) -> list[str]:
        return list(self.layers)

    def multiplexity(self) -> sp.csr_matrix:
        """Matrix whose entry (u, v) is the number of layers linking u and v."""
        m = sp.csr_matrix((self.n, self.n), dtype=np.int32)
        for a in self.layers.values():
            m = m + a
        m = m.tocsr()
        m.sort_indices()
        return m

    def flattened(self) -> sp.csr_matrix:
        """0/1 matrix with a tie wherever at least one layer has one."""
        m = self.multiplexity()
        m.data[:] = 1
        return m

    def n_edges(self, name: str | None = None) -> int:
        a = self.flattened() if name is None else self.layers[name]
        return int(a.nnz // 2)

    def without(self, *names: str) -> "MultilayerNetwork":
        keep = {k: v for k, v in self.layers.items() if k not in names}
        return MultilayerNetwork(self.n, keep, self.active.copy())
