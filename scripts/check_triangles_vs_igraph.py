"""Check that the igraph-free triangle count equals igraph's on full-size synthetic networks.

    python scripts/check_triangles_vs_igraph.py SEED   (needs python-igraph; run in an environment that has it)
"""
import sys

import igraph as ig
import numpy as np
import scipy.sparse as sp

from builtnet import scenarios as S
from builtnet import world as W
from builtnet.descriptors import triangles_per_node
from builtnet.rulesets import DK_K, NL_2018_K, SE_LIKE_K, build

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for pop in (S.compact, S.dispersed):
    w = W.generate(pop("main"), seed)
    for rs in (NL_2018_K, DK_K, SE_LIKE_K):
        flat = build(w, rs, seed).multiplexity()
        flat.data[:] = 1
        upper = sp.triu(flat, 1).tocoo()
        g = ig.Graph(n=flat.shape[0], edges=np.stack([upper.row, upper.col], axis=1).tolist())
        loc = np.asarray(g.transitivity_local_undirected(mode="zero"))
        deg = np.asarray(flat.sum(axis=1)).ravel()
        want = np.rint(loc * deg * (deg - 1) / 2).astype(np.int64)
        got = triangles_per_node(flat)
        print(pop.__name__, rs.name, "identical" if np.array_equal(got, want) else f"DIFFER in {(got != want).sum()} nodes", flush=True)
