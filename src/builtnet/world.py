"""Synthetic populations as affiliation tables.

A world is a set of tables, not a network: who lives where, who shares a
household and a building, who attends which school, year and class, and who
works for which employer at which site. Rule-sets turn these tables into
edges (see `rulesets.py`).

The default parameter values are illustrative. They are not calibrated to any
country and nothing generated here describes real people.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

CHILD, ADULT, SENIOR = 0, 1, 2
PRIMARY, SECONDARY = 0, 1


@dataclass(frozen=True)
class WorldConfig:
    n_households: int = 4500
    region_km: float = 12.0

    # Settlement
    n_cores: int = 4  # built-up areas; household shares fall as 1/rank
    core_share: float = 0.8  # share of households living in a core
    core_peak_density: float = 3000.0  # households per km2 at a core's centre
    apartment_share_core: float = 0.4  # core households in multi-household buildings
    apartment_share_rural: float = 0.02
    building_mean_households: float = 12.0
    property_mean_buildings: float = 1.0  # above 1: blocks of flats are grouped into properties of about this many
    flat_small_household_bias: float = 0.0  # above 0: smaller households are more often in shared buildings
    n_municipalities: int = 6

    # Households
    household_size_p: tuple[float, ...] = (0.38, 0.33, 0.12, 0.12, 0.05)  # sizes 1..5

    # Income groups (five levels, by household)
    residential_sorting: float = 0.5  # 0: unrelated to place; 1: set by place

    # Schools
    primary_school_size: int = 220
    secondary_school_size: int = 900
    class_max: int = 28
    school_decay_m: float = 1500.0

    # Completed education (used only by rule-sets that link graduation cohorts)
    upper_secondary_share: float = 0.45  # highest completed level; the rest finish compulsory school only
    tertiary_share: float = 0.35

    # Work
    employment_rate: float = 0.75
    employment_income_slope: float = 0.0  # top minus bottom income group
    employer_tail: float = 1.0  # P(size >= s) ~ s**-employer_tail
    employer_max_share: float = 0.25  # largest employer, as a share of workers
    employer_max_size: int | None = None  # largest employer in people; overrides the share if set
    multi_site_min_size: int = 150
    multi_site_prob: float = 0.6
    site_mean_size: float = 60.0
    work_decay_m: float = 4000.0
    workplace_sorting: float = 0.5  # 0: employers hire all income groups alike
    size_income_slope: float = 0.0  # >0: larger employers hire higher income groups

    # Further planted income gradients (0 means none)
    density_income_slope: float = 0.0  # >0: higher income groups live in denser places
    overlap_income_slope: float = 0.0  # >0: higher income groups study and work nearer home

    # Kin (legal parent-child links and partners); used only by rule-sets with a kin layer
    partner_share: float = 0.9  # childless two-adult households whose adults are partners
    adult_family_share: float = 0.6  # households of 3+ adults that are parents with grown children
    kin_parent_share: float = 0.9  # adults aged 18 with a living parent in the region; falls with age
    kin_max_children: int = 6  # most children one person can have
    kin_parent_pool: float = 0.5  # share of older adults who can be picked as a parent
    kin_decay_m: float = 3000.0  # mean home distance to a parent living elsewhere
    kin_late_parent_share: float = 0.85  # grown-ups without a living parent here whose late parent is on record
    kin_sibship_mean: float = 2.4  # children per late parent, on average


@dataclass
class World:
    """Affiliation tables. Person-level arrays have length `n`; -1 means none."""

    config: WorldConfig
    seed: int
    n: int
    xy: np.ndarray  # home coordinates, metres
    household: np.ndarray
    building: np.ndarray
    municipality: np.ndarray
    role: np.ndarray  # CHILD, ADULT, SENIOR
    age: np.ndarray
    income: np.ndarray  # 1..5
    school: np.ndarray
    school_level: np.ndarray  # PRIMARY, SECONDARY
    school_year: np.ndarray  # year group within the level, from 1
    school_class: np.ndarray
    employer: np.ndarray
    site: np.ndarray
    household_xy: np.ndarray
    site_xy: np.ndarray
    site_employer: np.ndarray
    parent: np.ndarray | None = None  # (n, 2) legal parents, -1 where none
    partner: np.ndarray | None = None  # (n,) partner, -1 where none
    late_parent: np.ndarray | None = None  # (n,) id of a parent no longer in the population, -1 where none
    plot: np.ndarray | None = None  # (n,) the property (plot of land) a home stands on; the building unless buildings are grouped
    edu_level: np.ndarray | None = None  # (n,) latest completed level: 0 compulsory, 1 upper secondary, 2 tertiary; -1 none yet
    grad_years_ago: np.ndarray | None = None  # (n,) years since that level was completed; -1 where none

    def __post_init__(self) -> None:
        if self.plot is None:
            self.plot = self.building.copy()
        if self.edu_level is None:
            self.edu_level = np.full(self.n, -1, dtype=np.int64)
        if self.grad_years_ago is None:
            self.grad_years_ago = np.full(self.n, -1, dtype=np.int64)
        if self.parent is None:
            self.parent = np.full((self.n, 2), -1, dtype=np.int64)
        if self.partner is None:
            self.partner = np.full(self.n, -1, dtype=np.int64)
        if self.late_parent is None:
            self.late_parent = np.full(self.n, -1, dtype=np.int64)

    @property
    def year_group(self) -> np.ndarray:
        """School and year: the unit registers see when classes are not recorded."""
        out = np.full(self.n, -1, dtype=np.int64)
        ok = self.school >= 0
        out[ok] = self.school[ok] * 100 + self.school_year[ok]
        return out

    def group_sizes(self, groups: np.ndarray) -> np.ndarray:
        """Size of each person's group; 0 where the person has none."""
        out = np.zeros(self.n, dtype=np.int64)
        ok = groups >= 0
        if ok.any():
            _, inv, cnt = np.unique(groups[ok], return_inverse=True, return_counts=True)
            out[ok] = cnt[inv]
        return out


def _spawn(seed: int, k: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(k)]


def _settlement(cfg: WorldConfig, rng: np.random.Generator, rng_plot: np.random.Generator | None = None):
    """Household coordinates, building ids, property ids and municipality."""
    side = cfg.region_km * 1000.0
    n_h = cfg.n_households
    centres = side * (0.15 + 0.7 * rng.random((cfg.n_cores, 2)))
    shares = 1.0 / np.arange(1, cfg.n_cores + 1)
    shares = cfg.core_share * shares / shares.sum()
    core = rng.choice(cfg.n_cores + 1, size=n_h, p=np.append(shares, 1 - cfg.core_share))
    core[core == cfg.n_cores] = -1  # rural

    xy = side * rng.random((n_h, 2))
    sigma = np.zeros(cfg.n_cores)
    for c in range(cfg.n_cores):
        idx = np.flatnonzero(core == c)
        # Peak density of a 2-D Gaussian with n points is n / (2 pi sigma^2).
        sigma[c] = 1000.0 * np.sqrt(max(idx.size, 1) / (2 * np.pi * cfg.core_peak_density))
        xy[idx] = centres[c] + rng.normal(scale=sigma[c], size=(idx.size, 2))

    building = np.arange(n_h)
    block_property = {}  # building id -> property id, only where buildings are grouped
    next_id = n_h
    for c in list(range(cfg.n_cores)) + [-1]:
        idx = np.flatnonzero(core == c)
        share = cfg.apartment_share_rural if c == -1 else cfg.apartment_share_core
        apt = idx[rng.random(idx.size) < share]
        if apt.size < 2:
            continue
        n_b = max(1, int(round(apt.size / cfg.building_mean_households)))
        if c == -1:
            b_xy = side * rng.random((n_b, 2))
        else:
            b_xy = centres[c] + rng.normal(scale=0.7 * sigma[c], size=(n_b, 2))
        which = rng.integers(0, n_b, size=apt.size)
        xy[apt] = b_xy[which]
        building[apt] = next_id + which
        if cfg.property_mean_buildings > 1 and rng_plot is not None:  # own stream: homes stay where they are
            # Neighbouring blocks share a property: walk the blocks along a space-filling curve and cut it into runs.
            cell = np.clip((b_xy / side * 1024).astype(np.int64), 0, 1023)
            code = np.zeros(n_b, dtype=np.int64)
            for bit in range(10):
                code |= ((cell[:, 0] >> bit) & 1) << (2 * bit)
                code |= ((cell[:, 1] >> bit) & 1) << (2 * bit + 1)
            along = np.argsort(code, kind="stable")
            run = 1 + rng_plot.poisson(cfg.property_mean_buildings - 1, size=n_b)
            owner = np.repeat(np.arange(n_b), run)[:n_b]
            for b, o in zip(along.tolist(), owner.tolist()):
                block_property[next_id + b] = next_id + int(along[np.flatnonzero(owner == o)[0]])
        next_id += n_b
    xy = np.clip(xy, 0.0, side)

    extra = side * rng.random((max(0, cfg.n_municipalities - cfg.n_cores), 2))
    muni_centres = np.concatenate([centres, extra])[: cfg.n_municipalities]
    d = ((xy[:, None, :] - muni_centres[None, :, :]) ** 2).sum(axis=2)
    prop = building.copy()
    if block_property:
        prop = np.array([block_property.get(int(b), int(b)) for b in building], dtype=np.int64)
    return xy, building, prop, d.argmin(axis=1)


def _people(cfg: WorldConfig, rng: np.random.Generator, shared: np.ndarray | None = None):
    """Household sizes, and each person's household, role and age.

    `shared` marks households in a building with other households. With
    `flat_small_household_bias` above 0 the same sizes are dealt out so that
    such households tend to be smaller.
    """
    n_h = cfg.n_households
    p = np.asarray(cfg.household_size_p, dtype=float)
    size = rng.choice(np.arange(1, p.size + 1), size=n_h, p=p / p.sum())
    u = rng.random(n_h)
    if cfg.flat_small_household_bias > 0 and shared is not None:
        key = cfg.flat_small_household_bias * shared + rng.random(n_h)
        size[np.argsort(-key, kind="stable")] = np.sort(size)
    adults = np.zeros(n_h, dtype=np.int64)
    seniors = np.zeros(n_h, dtype=np.int64)
    one = size == 1
    adults[one & (u < 0.62)] = 1
    seniors[one & (u >= 0.62)] = 1
    two = size == 2
    adults[two & (u < 0.52)] = 2
    seniors[two & (u >= 0.52) & (u < 0.90)] = 2
    adults[two & (u >= 0.90)] = 1  # one adult and one child
    big = size >= 3
    adults[big & (u < 0.7)] = 2  # two adults and children
    adults[big & (u >= 0.7) & (u < 0.8)] = 1  # one adult and children
    adults[big & (u >= 0.8)] = size[big & (u >= 0.8)]  # adults only

    household = np.repeat(np.arange(n_h), size)
    start = np.cumsum(size) - size
    pos = np.arange(household.size) - start[household]
    role = np.full(household.size, CHILD, dtype=np.int8)
    role[pos < adults[household] + seniors[household]] = SENIOR
    role[pos < adults[household]] = ADULT
    age = rng.integers(0, 18, size=household.size)
    is_adult, is_senior = role == ADULT, role == SENIOR
    age[is_adult] = rng.integers(18, 67, size=is_adult.sum())
    age[is_senior] = rng.integers(67, 91, size=is_senior.sum())
    return household, role, age


def _density_score(xy: np.ndarray, k: int = 10) -> np.ndarray:
    """Standardised local density of each household: high where the k-th nearest is close."""
    kk = min(k, xy.shape[0] - 1)
    dist, _ = cKDTree(xy).query(xy, k=kk + 1)
    score = -np.log(np.maximum(dist[:, -1], 1.0))
    return (score - score.mean()) / (score.std() + 1e-12)


def _income(cfg: WorldConfig, xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Income group 1..5 per household, with a smooth spatial component."""
    side = cfg.region_km * 1000.0
    n_bumps = 8
    c = side * rng.random((n_bumps, 2))
    w = side * rng.uniform(0.08, 0.3, size=n_bumps)
    a = rng.normal(size=n_bumps)
    d2 = ((xy[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    field = (a[None, :] * np.exp(-d2 / (2 * w[None, :] ** 2))).sum(axis=1)
    field = (field - field.mean()) / (field.std() + 1e-12)
    rho = cfg.residential_sorting
    delta = cfg.density_income_slope
    z = rho * field + np.sqrt(max(0.0, 1 - rho**2 - delta**2)) * rng.normal(size=xy.shape[0])
    if delta:
        z = z + delta * _density_score(xy)
    rank = np.argsort(np.argsort(z))
    return (rank * 5 // xy.shape[0]).astype(np.int8) + 1


def _reach(cfg: WorldConfig, income: np.ndarray) -> np.ndarray:
    """Multiplier on travel distances; below 1 for high incomes if overlap_income_slope > 0."""
    return np.exp(-cfg.overlap_income_slope * (income - 3) / 2.0)


def _schools(cfg: WorldConfig, xy: np.ndarray, age: np.ndarray, income: np.ndarray, rng: np.random.Generator):
    n = age.size
    school = np.full(n, -1, dtype=np.int64)
    level = np.full(n, -1, dtype=np.int8)
    year = np.full(n, -1, dtype=np.int64)
    klass = np.full(n, -1, dtype=np.int64)
    next_school = 0
    specs = [(PRIMARY, 4, 11, 3, cfg.primary_school_size), (SECONDARY, 12, 17, 11, cfg.secondary_school_size)]
    for lev, lo, hi, offset, target in specs:
        pupils = np.flatnonzero((age >= lo) & (age <= hi))
        if pupils.size == 0:
            continue
        n_s = max(1, int(round(pupils.size / target)))
        s_xy = xy[rng.choice(pupils, size=n_s, replace=False)]
        d = np.sqrt(((xy[pupils][:, None, :] - s_xy[None, :, :]) ** 2).sum(axis=2))
        w = np.exp(-d / (cfg.school_decay_m * _reach(cfg, income[pupils]))[:, None])
        cdf = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
        pick = np.minimum((rng.random(pupils.size)[:, None] > cdf).sum(axis=1), n_s - 1)
        school[pupils] = next_school + pick
        level[pupils] = lev
        year[pupils] = age[pupils] - offset
        next_school += n_s

    klass = split_into_classes(school, year, cfg.class_max, rng)
    return school, level, year, klass


def split_into_classes(school: np.ndarray, year: np.ndarray, class_max: int, rng: np.random.Generator) -> np.ndarray:
    """Split each school year into classes of at most `class_max`, as evenly as possible, at random."""
    klass = np.full(school.size, -1, dtype=np.int64)
    ok = np.flatnonzero(school >= 0)
    if ok.size == 0:
        return klass
    key = school[ok] * 100 + year[ok]
    order = ok[np.lexsort((rng.random(ok.size), key))]
    sorted_key = school[order] * 100 + year[order]
    _, first, count = np.unique(sorted_key, return_index=True, return_counts=True)
    group_of = np.repeat(np.arange(count.size), count)
    pos = np.arange(order.size) - first[group_of]
    n_classes = np.ceil(count / class_max).astype(np.int64)
    class_in_year = pos * n_classes[group_of] // count[group_of]
    class_start = np.cumsum(n_classes) - n_classes
    klass[order] = class_start[group_of] + class_in_year
    return klass


def _employer_sizes(cfg: WorldConfig, n_workers: int, rng: np.random.Generator) -> np.ndarray:
    if cfg.employer_max_size is not None:
        max_size = max(2, min(int(cfg.employer_max_size), n_workers))
    else:
        max_size = max(2, int(cfg.employer_max_share * n_workers))
    sizes: list[int] = []
    total = 0
    while total < n_workers:
        draw = np.floor(rng.random(1024) ** (-1.0 / cfg.employer_tail)).astype(np.int64)
        for s in np.minimum(draw, max_size):
            s = int(min(s, n_workers - total))
            sizes.append(s)
            total += s
            if total >= n_workers:
                break
    return np.asarray(sizes, dtype=np.int64)


def _work(cfg: WorldConfig, xy, role, income, rng: np.random.Generator):
    n = role.size
    employer = np.full(n, -1, dtype=np.int64)
    site = np.full(n, -1, dtype=np.int64)
    rate = cfg.employment_rate + cfg.employment_income_slope * (income - 3) / 4.0
    workers = np.flatnonzero((role == ADULT) & (rng.random(n) < np.clip(rate, 0.0, 1.0)))
    if workers.size == 0:
        return employer, site, np.zeros((0, 2)), np.zeros(0, dtype=np.int64)

    emp_size = _employer_sizes(cfg, workers.size, rng)
    site_emp: list[int] = []
    site_size: list[int] = []
    for e, s in enumerate(emp_size):
        if s >= cfg.multi_site_min_size and rng.random() < cfg.multi_site_prob:
            k = min(max(2, int(round(s / cfg.site_mean_size))), int(s))
            parts = 1 + rng.multinomial(s - k, rng.dirichlet(2.0 * np.ones(k)))
        else:
            parts = np.array([s])
        site_emp.extend([e] * len(parts))
        site_size.extend(int(v) for v in parts)
    site_emp_a = np.asarray(site_emp, dtype=np.int64)
    site_size_a = np.asarray(site_size, dtype=np.int64)
    site_xy = xy[rng.integers(0, n, size=site_emp_a.size)] + rng.normal(scale=300.0, size=(site_emp_a.size, 2))

    # Each employer has a wage level; larger employers lean higher if size_income_slope > 0.
    zlog = np.log(emp_size)
    zlog = (zlog - zlog.mean()) / (zlog.std() + 1e-12)
    slope = float(np.clip(cfg.size_income_slope, -1.0, 1.0))
    wage = slope * zlog + np.sqrt(1 - slope**2) * rng.normal(size=emp_size.size)
    z_income = (income[workers] - 3) / np.sqrt(2.0)
    kappa = 2.0 * cfg.workplace_sorting
    decay = cfg.work_decay_m * _reach(cfg, income[workers])

    free = np.ones(workers.size, dtype=bool)
    for s in rng.permutation(site_emp_a.size):
        cand = np.flatnonzero(free)
        d = np.sqrt(((xy[workers[cand]] - site_xy[s]) ** 2).sum(axis=1))
        w = np.exp(-d / decay[cand]) * np.exp(-0.5 * kappa * (z_income[cand] - wage[site_emp_a[s]]) ** 2)
        w = w + 1e-300
        take = rng.choice(cand, size=int(site_size_a[s]), replace=False, p=w / w.sum())
        free[take] = False
        site[workers[take]] = s
        employer[workers[take]] = site_emp_a[s]
    return employer, site, site_xy, site_emp_a


def _kin(cfg: WorldConfig, xy, household, role, age, rng: np.random.Generator):
    """Legal parent links, partners, and ages made consistent with them.

    Inside a household:
      - two grown-ups with children are partners and the children's parents;
        a lone grown-up is the only parent;
      - two grown-ups without children are partners with probability
        `partner_share`;
      - three or more grown-ups are, with probability `adult_family_share`,
        a couple with their grown-up children; otherwise they are unrelated.
    Grown-up ages are redrawn so that partners are close in age and parents
    are about 20 to 40 years older than their children. Nothing else in the
    world depends on the age of a grown-up.

    Across households: a grown-up with no parent at home has, with a
    probability that falls with age, a parent living elsewhere. The parent is
    drawn from a pool of older adults, is about 20 to 40 years older, and
    lives at a distance that is exponential with mean `kin_decay_m`. The
    parent's partner is the second parent if at least 16 years older than the
    child. Most grown-ups who still have no parent get a late parent: one who
    is on record but no longer in the population, shared with siblings.
    People who share a parent are siblings; rule-sets derive every other
    relation from these links (see `rules.kin_edges`).

    Returns (parent, partner, late_parent, age).
    """
    n = household.size
    age = age.copy()
    parent = np.full((n, 2), -1, dtype=np.int64)
    partner = np.full(n, -1, dtype=np.int64)
    grown = role != CHILD

    # Household members in a fixed order, grown-ups first. Every household has one.
    who = np.lexsort((np.arange(n), ~grown, household))
    hh = household[who]
    start = np.flatnonzero(np.r_[True, hh[1:] != hh[:-1]])
    size = np.diff(np.r_[start, n])
    n_h = start.size
    n_grown = np.add.reduceat(grown[who].astype(np.int64), start)
    first = who[start]
    second = who[np.minimum(start + 1, n - 1)]  # meaningful only where n_grown >= 2
    pos_h = np.repeat(np.arange(n_h), size)
    rank = np.arange(n) - start[pos_h]

    has_kids = n_grown < size
    two = n_grown == 2
    many = n_grown >= 3
    u = rng.random(n_h)
    couple = (two & (has_kids | (u < cfg.partner_share))) | (many & (u < cfg.adult_family_share))

    # Ages. A parent of children under 18 is 20 to 40 years older than the oldest.
    oldest_kid = np.maximum.reduceat(np.where(grown, -1, age)[who], start)  # -1 if none
    hk = np.flatnonzero(has_kids)
    age[first[hk]] = np.minimum(oldest_kid[hk] + rng.integers(20, 41, size=hk.size), 66)
    fam = np.flatnonzero(many & couple)
    age[first[fam]] = rng.integers(48, 67, size=fam.size)
    cp = np.flatnonzero(couple)
    senior = role[first[cp]] == SENIOR
    drift = np.rint(rng.normal(0.0, 3.5, size=cp.size)).astype(np.int64)
    low = np.where(senior, 67, np.maximum(18, oldest_kid[cp] + 18))
    age[second[cp]] = np.clip(age[first[cp]] + drift, low, np.where(senior, 90, 66))
    partner[first[cp]] = second[cp]
    partner[second[cp]] = first[cp]

    # Grown-up children at home.
    at_home = np.flatnonzero(grown[who] & (many & couple)[pos_h] & (rank >= 2))
    h = pos_h[at_home]
    top = np.maximum(np.minimum(age[first[h]], age[second[h]]) - 20, 18)
    age[who[at_home]] = rng.integers(18, top + 1)
    parent[who[at_home], 0] = first[h]
    parent[who[at_home], 1] = second[h]

    # Children under 18.
    kid = np.flatnonzero(~grown[who])
    parent[who[kid], 0] = first[pos_h[kid]]
    both = kid[n_grown[pos_h[kid]] == 2]
    parent[who[both], 1] = second[pos_h[both]]

    # Parents living elsewhere.
    seeker = np.flatnonzero(grown & (parent[:, 0] < 0))
    p_has = cfg.kin_parent_share * np.clip((75.0 - age[seeker]) / 57.0, 0.0, 1.0)
    seeker = seeker[rng.random(seeker.size) < p_has]
    pool = np.flatnonzero(grown & (age >= 36) & (rng.random(n) < cfg.kin_parent_pool))
    room = cfg.kin_max_children - np.bincount(parent[parent >= 0], minlength=n)  # children still possible
    chosen = _match_parents(cfg, xy, age, household, seeker, xy[pool], age[pool], household[pool], room[pool], rng)
    got = chosen >= 0
    child, chosen = seeker[got], pool[chosen[got]]
    parent[child, 0] = chosen
    other = partner[chosen]
    ok2 = (other >= 0) & (age[np.maximum(other, 0)] - age[child] >= 16)
    parent[child[ok2], 1] = other[ok2]

    # Parents no longer in the population. They are on record, so their children are still
    # siblings, and their grandchildren cousins, although the parent has no ties.
    late_parent = np.full(n, -1, dtype=np.int64)
    orphan = np.flatnonzero(grown & (parent[:, 0] < 0))
    orphan = orphan[rng.random(orphan.size) < cfg.kin_late_parent_share]
    if orphan.size >= 2:
        n_late = max(1, int(round(orphan.size / cfg.kin_sibship_mean)))
        anchor = rng.choice(orphan, size=n_late, replace=False)  # a late parent lived where one child lives
        late_age = age[anchor] + rng.integers(20, 41, size=n_late)
        chosen = _match_parents(
            cfg, xy, age, household, orphan, xy[anchor], late_age,
            np.full(n_late, -1), np.full(n_late, cfg.kin_max_children), rng,
        )
        late_parent[orphan] = chosen

    # Partners are never each other's siblings: if both ended with the same parent, one loses it.
    second_of = np.flatnonzero((partner >= 0) & (partner < np.arange(n)))
    mate = partner[second_of]
    same = np.zeros(second_of.size, dtype=bool)
    for a in (0, 1):
        for b in (0, 1):
            same |= (parent[second_of, a] >= 0) & (parent[second_of, a] == parent[mate, b])
    parent[second_of[same]] = -1
    same_late = (late_parent[second_of] >= 0) & (late_parent[second_of] == late_parent[mate])
    late_parent[second_of[same_late]] = -1
    return parent, partner, late_parent, age


def _match_parents(cfg, xy, age, household, child, cand_xy, cand_age, cand_household, cand_room, rng):
    """Give each child a parent among the candidates; -1 where none fits.

    A child aims at a parent 20 to 40 years older who lives at a distance that
    is exponential with mean `kin_decay_m`, and takes the nearest candidate of
    about that age who is at least 16 years older, lives in another
    household and still has room for a child.
    """
    out = np.full(child.size, -1, dtype=np.int64)
    if child.size == 0 or cand_age.size == 0:
        return out
    room = np.asarray(cand_room, dtype=np.int64).copy()
    target_age = age[child] + rng.integers(20, 41, size=child.size)
    dist = rng.exponential(cfg.kin_decay_m, size=child.size)
    angle = rng.uniform(0.0, 2 * np.pi, size=child.size)
    target_xy = xy[child] + np.stack([dist * np.cos(angle), dist * np.sin(angle)], axis=1)
    band_of_cand = (cand_age - 35) // 5  # five-year bands
    band_of_target = (target_age - 35) // 5
    for b in np.unique(band_of_target):
        cand = np.flatnonzero(band_of_cand == b)
        idx = np.flatnonzero(band_of_target == b)
        if cand.size == 0:
            continue
        k = min(6, cand.size)
        _, nn = cKDTree(cand_xy[cand]).query(target_xy[idx], k=k)
        nn = np.asarray(nn).reshape(idx.size, k)
        me = child[idx]
        chosen = np.full(idx.size, -1, dtype=np.int64)
        for j in range(k):  # nearest candidate first; move on if unsuitable or full
            c = cand[nn[:, j]]
            want = np.flatnonzero(
                (chosen < 0) & (cand_household[c] != household[me]) & (cand_age[c] - age[me] >= 16)
            )
            if want.size == 0:
                continue
            by_parent = want[np.argsort(c[want], kind="stable")]
            cs = c[by_parent]
            new_group = np.r_[True, cs[1:] != cs[:-1]]
            queue = np.arange(cs.size) - np.maximum.accumulate(np.where(new_group, np.arange(cs.size), 0))
            take = by_parent[queue < room[cs]]
            chosen[take] = c[take]
            np.subtract.at(room, c[take], 1)
        out[idx] = chosen
    return out


def _education(cfg: WorldConfig, age: np.ndarray, rng: np.random.Generator):
    """Latest completed level of education and how many years ago it was completed.

    Compulsory school ends at 16, upper secondary at 19 and tertiary between
    22 and 27. A person who is still too young for their final level counts
    with the level they have completed so far. Unrelated to income by design.
    """
    n = age.size
    u = rng.random(n)
    final = np.where(u < cfg.tertiary_share, 2, np.where(u < cfg.tertiary_share + cfg.upper_secondary_share, 1, 0))
    tertiary_age = rng.integers(22, 28, size=n)
    level = np.full(n, -1, dtype=np.int64)
    done_at = np.full(n, -1, dtype=np.int64)
    for lev, at in ((0, np.full(n, 16)), (1, np.full(n, 19)), (2, tertiary_age)):
        ok = (final >= lev) & (age >= at)
        level[ok] = lev
        done_at[ok] = at[ok]
    years_ago = np.where(level >= 0, age - done_at, -1)
    return level, years_ago


def generate(config: WorldConfig, seed: int) -> World:
    """Generate one population. The same config and seed give the same world."""
    # Spawned children are stable: adding a stream leaves the earlier ones unchanged.
    r_place, r_people, r_income, r_school, r_work, r_kin, r_edu, r_plot = _spawn(seed, 8)
    h_xy, h_building, h_property, h_muni = _settlement(config, r_place, r_plot)
    shared = np.bincount(h_building, minlength=int(h_building.max()) + 1)[h_building] > 1
    household, role, age = _people(config, r_people, shared)
    h_income = _income(config, h_xy, r_income)
    xy = h_xy[household]
    income = h_income[household]
    school, level, year, klass = _schools(config, xy, age, income, r_school)
    employer, site, site_xy, site_emp = _work(config, xy, role, income, r_work)
    parent, partner, late_parent, age = _kin(config, xy, household, role, age, r_kin)
    edu_level, grad_years_ago = _education(config, age, r_edu)
    return World(
        config=config,
        seed=seed,
        n=household.size,
        xy=xy,
        household=household,
        building=h_building[household],
        plot=h_property[household],
        municipality=h_muni[household],
        role=role,
        age=age,
        income=income,
        school=school,
        school_level=level,
        school_year=year,
        school_class=klass,
        employer=employer,
        site=site,
        household_xy=h_xy,
        site_xy=site_xy,
        site_employer=site_emp,
        parent=parent,
        partner=partner,
        late_parent=late_parent,
        edu_level=edu_level,
        grad_years_ago=grad_years_ago,
    )
