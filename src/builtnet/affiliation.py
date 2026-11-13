"""Position descriptors computed on the affiliation tables, before any edge rule.

These cannot be affected by caps or sampling, because no edges exist yet. They
can still depend on how a group is defined (employer or site, year group or
class), so the unit is an argument.
"""

from __future__ import annotations

import numpy as np

from . import rules
from .world import World


def _median_distance(groups: np.ndarray, xy: np.ndarray, max_sample: int, rng: np.random.Generator) -> np.ndarray:
    """Median home distance from each person to the other members of their group.

    Groups larger than `max_sample` are measured against a random sample of
    that many members. NaN for people with no group or no co-member.
    """
    out = np.full(len(groups), np.nan)
    indptr, members, sizes = rules._group_index(groups)
    for g, s in enumerate(sizes):
        if s < 2:
            continue
        mem = members[indptr[g] : indptr[g + 1]]
        ref = mem if s <= max_sample else rng.choice(mem, size=max_sample, replace=False)
        diff = xy[mem][:, None, :] - xy[ref][None, :, :]
        d = np.sqrt((diff**2).sum(axis=2))
        d[mem[:, None] == ref[None, :]] = np.nan  # not one's own distance to oneself
        out[mem] = np.nanmedian(d, axis=1)
    return out


def _joint_others(world: World, *groups: np.ndarray) -> np.ndarray:
    """Number of other people who share every one of the given groups with each person."""
    joint = rules.compose_groups(*groups)
    return np.maximum(world.group_sizes(joint) - 1, 0)


def affiliation_descriptors(
    world: World,
    work_unit: str = "employer",
    school_unit: str = "year",
    max_sample: int = 2000,
) -> dict[str, np.ndarray]:
    """Descriptors of each person's memberships.

    n_contexts        how many of household (with someone else), school and
                      work the person belongs to
    size_household    people in the household, the person included
    size_school       people in the school group (NaN if none)
    size_work         people in the work group (NaN if none)
    comember_overlap  of the distinct people who share any context, the share
                      who share two or more (NaN if nobody shares any)
    dist_school       median home distance to school co-members, metres
    dist_work         median home distance to work co-members, metres
    """
    if work_unit not in ("employer", "site"):
        raise ValueError(f"unknown work_unit: {work_unit}")
    if school_unit not in ("year", "class"):
        raise ValueError(f"unknown school_unit: {school_unit}")
    work = world.employer if work_unit == "employer" else world.site
    school = world.year_group if school_unit == "year" else world.school_class
    hh = world.household

    size_h = world.group_sizes(hh).astype(float)
    size_s = world.group_sizes(school).astype(float)
    size_w = world.group_sizes(work).astype(float)
    out: dict[str, np.ndarray] = {
        "n_contexts": (size_h > 1).astype(float) + (school >= 0) + (work >= 0),
        "size_household": size_h,
        "size_school": np.where(school >= 0, size_s, np.nan),
        "size_work": np.where(work >= 0, size_w, np.nan),
    }

    # Distinct co-members and co-members in two or more contexts, by inclusion-exclusion.
    singles = np.maximum(size_h - 1, 0) + np.maximum(size_s - 1, 0) + np.maximum(size_w - 1, 0)
    pairs = _joint_others(world, hh, school) + _joint_others(world, hh, work) + _joint_others(world, school, work)
    triple = _joint_others(world, hh, school, work)
    distinct = singles - pairs + triple
    multi = pairs - 2 * triple
    overlap = np.full(world.n, np.nan)
    ok = distinct > 0
    overlap[ok] = multi[ok] / distinct[ok]
    out["comember_overlap"] = overlap

    rng = np.random.default_rng(np.random.SeedSequence([world.seed, 7001]))
    out["dist_school"] = _median_distance(school, world.xy, max_sample, rng)
    out["dist_work"] = _median_distance(work, world.xy, max_sample, rng)
    return out
