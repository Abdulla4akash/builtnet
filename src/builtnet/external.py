"""Affiliation tables from a population built by others.

`world_from_tables` turns plain person-level arrays (age, household, home
coordinates, residence, county, household income, work place, school) into a
`World`, so the same rule-sets and descriptors can be applied to it. It was
written for the University of Virginia synthetic populations (CC BY 4.0), but
takes arrays, not files.

What such a population does not have, and what is done about it:
  - no employer above the work place: employer and site are the same unit;
  - no school year or class: the year group is school and age, and classes
    are a random split of each year group;
  - no kin records and no completed education: those tables stay empty, so
    only four-layer rule-sets apply.
"""

from __future__ import annotations

import numpy as np

from .world import ADULT, CHILD, PRIMARY, SECONDARY, SENIOR, World, WorldConfig, split_into_classes

EARTH_RADIUS_M = 6_371_000.0


def _reindex(ids: np.ndarray) -> np.ndarray:
    """Consecutive ids from 0 for the non-negative entries; negatives stay -1."""
    out = np.full(ids.shape, -1, dtype=np.int64)
    ok = ids >= 0
    if ok.any():
        _, out[ok] = np.unique(ids[ok], return_inverse=True)
    return out


def project(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Longitude and latitude in degrees to metres on a local plane (equirectangular)."""
    lon0, lat0 = np.mean(lon), np.mean(lat)
    x = EARTH_RADIUS_M * np.cos(np.radians(lat0)) * np.radians(lon - lon0)
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return np.stack([x, y], axis=1)


def income_fifths(household: np.ndarray, hh_income: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Income group 1 to 5 per person: fifths of households by income, ties broken at random.

    Households with a missing income get group 0, which no comparison uses.
    """
    n_h = int(household.max()) + 1
    h_income = np.full(n_h, np.nan)
    h_income[household] = hh_income
    known = np.flatnonzero(np.isfinite(h_income))
    group = np.zeros(n_h, dtype=np.int8)
    order = known[np.lexsort((rng.random(known.size), h_income[known]))]
    group[order] = (np.arange(order.size) * 5 // max(order.size, 1)).astype(np.int8) + 1
    return group[household]


def world_from_tables(
    age: np.ndarray,
    household: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    residence: np.ndarray,
    county: np.ndarray,
    hh_income: np.ndarray,
    work: np.ndarray,
    school: np.ndarray,
    seed: int = 0,
    class_max: int = 28,
) -> World:
    """Build a `World` from person-level arrays. Ids may be any integers; -1 means none.

    Every member of a household must share one home coordinate, residence,
    county and household income.
    """
    n = age.size
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8101]))
    age = np.asarray(age, dtype=np.int64)
    household = _reindex(np.asarray(household))
    xy = project(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    first = np.unique(household, return_index=True)[1]
    role = np.where(age < 18, CHILD, np.where(age < 67, ADULT, SENIOR)).astype(np.int8)

    school = _reindex(np.asarray(school))
    in_school = school >= 0
    level = np.where(in_school, np.where(age <= 11, PRIMARY, SECONDARY), -1).astype(np.int8)
    year = np.where(in_school, np.clip(age, 0, 99), -1)  # the year group is school and age
    klass = split_into_classes(school, year, class_max, rng)

    work = _reindex(np.asarray(work))
    n_sites = int(work.max()) + 1 if n else 0
    return World(
        config=WorldConfig(),  # placeholder: nothing here was generated from these settings
        seed=seed,
        n=n,
        xy=xy,
        household=household,
        building=_reindex(np.asarray(residence)),
        municipality=_reindex(np.asarray(county)),
        role=role,
        age=age,
        income=income_fifths(household, np.asarray(hh_income, dtype=float), rng),
        school=school,
        school_level=level,
        school_year=year,
        school_class=klass,
        employer=work,
        site=work.copy(),
        household_xy=xy[first],
        site_xy=np.zeros((n_sites, 2)),  # work-place coordinates are not needed by any rule
        site_employer=np.arange(n_sites, dtype=np.int64),
    )
