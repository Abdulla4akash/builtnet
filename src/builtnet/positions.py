"""Position classes: a small set of profile types, fixed under one rule-set.

The classes are found by clustering standardised position profiles under a
reference rule-set. The same standardisation and the same class centres are
then applied under other rule-sets, so "this person changed class" has one
meaning throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_FEATURES = (
    "degree_total",
    "degree_household",
    "degree_neighbour",
    "degree_school",
    "degree_work",
    "degree_kin",
    "clustering",
    "excess_closure",
    "multiplex_share",
    "participation",
)


@dataclass
class PositionClasses:
    features: tuple[str, ...]
    fill: np.ndarray  # reference median, used where a value is undefined
    mean: np.ndarray
    sd: np.ndarray
    centres: np.ndarray  # (k, features), in standardised units


def _matrix(desc: dict[str, np.ndarray], features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix (degrees on a log scale) and the mask of people in the network."""
    present = np.isfinite(desc["degree_total"])
    cols = []
    for f in features:
        v = np.asarray(desc[f], dtype=float) if f in desc else np.where(present, 0.0, np.nan)
        cols.append(np.log1p(v) if f.startswith("degree") else v)
    return np.column_stack(cols), present


def fit_position_classes(
    desc: dict[str, np.ndarray], k: int = 6, seed: int = 0, features: tuple[str, ...] = CLASS_FEATURES
) -> PositionClasses:
    """Find `k` classes among the people present under the reference rule-set."""
    from sklearn.cluster import KMeans

    x, present = _matrix(desc, features)
    x = x[present]
    fill = np.nanmedian(x, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    x = np.where(np.isfinite(x), x, fill[None, :])
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit((x - mean) / sd)
    return PositionClasses(tuple(features), fill, mean, sd, km.cluster_centers_)


def assign_position_classes(model: PositionClasses, desc: dict[str, np.ndarray]) -> np.ndarray:
    """Class of each person under any rule-set; -1 for people not in the network."""
    x, present = _matrix(desc, model.features)
    x = np.where(np.isfinite(x), x, model.fill[None, :])
    z = (x - model.mean) / model.sd
    out = np.full(z.shape[0], -1, dtype=np.int64)
    step = 20_000
    for start in range(0, z.shape[0], step):
        block = z[start : start + step]
        d = ((block[:, None, :] - model.centres[None, :, :]) ** 2).sum(axis=2)
        out[start : start + step] = d.argmin(axis=1)
    out[~present] = -1
    return out
