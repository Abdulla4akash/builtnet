# Excess closure: three computations that disagree on multiplex ties

Excess closure (Bokányi, Heemskerk and Takes 2023, *Scientific Reports* 13, 9209; Methods) compares how closed a person's contacts are across layers with how closed they are within layers. With m(u,v) the number of layers linking u and v and k(u) the sum of u's layer degrees:

- P(u) = C(k(u), 2) − Σ_v C(m(u,v), 2), the number of pairs of ties around u;
- c_pure = (triangles around u whose three edges lie in one layer, summed over layers) / P(u);
- c_actual = (closed triplets around u) / P(u);
- excess closure = (c_actual − c_pure) / (1 − c_pure).

The paper describes the measure as lying between 0 and 1. `builtnet.descriptors.closure` offers three readings of c_actual:

| Variant | Numerator of c_actual |
|---|---|
| `paper` | unique closed node triplets, as the text says |
| `mlnlib` | Σ over pairs of alters of m(u,v)·m(v,w)·m(w,u), as the `mlnlib` library computes it (independent implementation; agrees with the library's output on random multilayer networks) |
| `tiepair` | Σ over pairs of alters of m(u,v)·m(u,w) where v and w are tied in at least one layer: the number of pairs of ties that are closed |

All three agree when no pair is tied in more than one layer. When pairs are tied in several layers they differ. One triangle present in two layers (three people who are both household members and close family) gives, for each of the three, P = 4 and c_pure = 0.5, and then excess closure −0.5 (`paper`), 3.0 (`mlnlib`) or 1.0 (`tiepair`). Only `tiepair` stays within [0, 1] by construction; it belongs to the weighted-clustering family of Barrat et al. (2004).

Recommendation: report `tiepair` beside any other reading in networks where many ties lie in several layers, such as household and family layers.
