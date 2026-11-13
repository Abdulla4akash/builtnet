"""Build a `World` from plain membership tables, so that any rule-set or card can be applied to real data.

    from builtnet import cards
    from builtnet.rulesets import build
    from builtnet.tables import world_from_tables

    w = world_from_tables(household=hh, home_xy=xy, employer=emp, school=sch, school_year=yr, age=age)
    net = build(w, cards.load("nl-2018"), seed=1)

Every argument is an array with one entry per person (or per household for `household_xy`); -1 means "none".
Only what a rule-set uses must be supplied: for example the Dutch colleague rule needs `employer` and
`home_xy`, the Danish one `site`, the neighbour rule `household` and `household_xy` (or `home_xy`), the
Swedish neighbour rule `plot`, the Swedish school rule `edu_level`, `grad_years_ago` and `municipality`,
and the family layer `parent` (and optionally `partner` and `late_parent`). Everything else is filled with
"none". Nothing is read from or written to disk.
"""

from __future__ import annotations

import numpy as np

from .world import ADULT, CHILD, PRIMARY, SECONDARY, SENIOR, World, WorldConfig


def _ids(x, n: int) -> np.ndarray:
    if x is None:
        return np.full(n, -1, dtype=np.int64)
    a = np.asarray(x, dtype=np.int64)
    if a.shape != (n,):
        raise ValueError(f"expected an array of length {n}, got shape {a.shape}")
    return a


def _dense(ids: np.ndarray) -> np.ndarray:
    """Renumber non-negative ids to 0..k-1, keeping -1."""
    out = np.full(ids.shape, -1, dtype=np.int64)
    ok = ids >= 0
    if ok.any():
        _, out[ok] = np.unique(ids[ok], return_inverse=True)
    return out


def world_from_tables(*, household, home_xy=None, household_xy=None, employer=None, site=None, school=None,
                      school_level=None, school_year=None, school_class=None, plot=None, municipality=None,
                      age=None, parent=None, partner=None, late_parent=None, edu_level=None, grad_years_ago=None,
                      income=None) -> World:
    household = _dense(np.asarray(household, dtype=np.int64))
    n = household.size
    if (household < 0).any():
        raise ValueError("every person needs a household id (one-person households are their own)")
    n_h = int(household.max()) + 1
    if home_xy is not None:
        home_xy = np.asarray(home_xy, dtype=float).reshape(n, 2)
    if household_xy is None:
        if home_xy is None:
            household_xy = np.zeros((n_h, 2))
        else:
            household_xy = np.zeros((n_h, 2))
            np.add.at(household_xy, household, home_xy)
            household_xy /= np.bincount(household, minlength=n_h)[:, None]
    household_xy = np.asarray(household_xy, dtype=float).reshape(n_h, 2)
    if home_xy is None:
        home_xy = household_xy[household]
    employer = _dense(_ids(employer, n))
    site = _ids(site, n)
    if (site >= 0).any():
        site = _dense(site)
        site_employer = np.full(int(site.max()) + 1, -1, dtype=np.int64)
        site_employer[site[site >= 0]] = employer[site >= 0]
    else:
        site = employer.copy()  # one site per employer when sites are unknown
        site_employer = np.arange(int(employer.max()) + 1 if (employer >= 0).any() else 0)
    age = np.asarray(age, dtype=np.int64) if age is not None else np.full(n, 40, dtype=np.int64)
    role = np.where(age < 18, CHILD, np.where(age >= 67, SENIOR, ADULT))
    school = _dense(_ids(school, n))
    level = _ids(school_level, n)
    level = np.where((school >= 0) & (level < 0), PRIMARY, level)
    pl = _ids(plot, n)
    w = World(
        config=WorldConfig(), seed=0, n=n, xy=home_xy, household=household,
        building=pl.copy() if (pl >= 0).any() else household.copy(),
        municipality=_ids(municipality, n), role=role, age=age,
        income=np.asarray(income, dtype=np.int64) if income is not None else np.zeros(n, dtype=np.int64),
        school=school, school_level=level, school_year=_ids(school_year, n), school_class=_dense(_ids(school_class, n)),
        employer=employer, site=site, household_xy=household_xy,
        site_xy=np.zeros((site_employer.size, 2)), site_employer=site_employer,
        parent=None if parent is None else np.asarray(parent, dtype=np.int64).reshape(n, 2),
        partner=None if partner is None else _ids(partner, n),
        late_parent=None if late_parent is None else _ids(late_parent, n),
        plot=pl if (pl >= 0).any() else None,
        edu_level=None if edu_level is None else _ids(edu_level, n),
        grad_years_ago=None if grad_years_ago is None else _ids(grad_years_ago, n),
    )
    return w


__all__ = ["world_from_tables", "PRIMARY", "SECONDARY"]
