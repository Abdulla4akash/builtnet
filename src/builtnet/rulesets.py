"""Named rule-sets: how a statistics office would wire one population.

`build` applies a rule-set to a world and returns a multilayer network on the
same node indices. Each layer draws its randomness from its own stream, so
changing one layer's rule never changes the random choices in another.

The presets imitate the published descriptions collected in docs/rule-sets.md. They
are approximations, labelled "-like" where rules are missing or unclear, and
are not reproductions of any real network.

Two families of preset exist. The four-layer presets (NL_2018, DK, ...) have
no family layer and are the ones used in the first reports. The five-layer
presets (NL_2018_K, DK_K, ...) add the family layer of each rule-set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import rules
from .network import MultilayerNetwork, edges_to_csr
from .world import World

CORE_LAYERS = ("household", "neighbour", "school", "work")
ALL_LAYERS = CORE_LAYERS + ("kin",)
LAYER_ORDER = ALL_LAYERS  # fixes each layer's random stream; "kin" was appended last

# The relations both the Dutch 2018 file and the Danish network derive from
# legal parent-child records (docs/rule-sets.md).
KIN_CORE = ("parent", "sibling", "grandparent", "aunt_uncle", "cousin", "co_parent")


@dataclass(frozen=True)
class RuleSet:
    name: str

    # Work layer
    work_unit: str = "employer"  # "employer", "site", or "site_if_large"
    work_split_min: int = 200  # for "site_if_large": employers at least this big are split into sites
    work_cap: object = 100  # a number, a function of group size, or None
    work_select: str = "nearest"  # see rules.SELECT_MODES
    work_full_if_leq: int | None = None  # default cap + 1

    # Neighbour layer
    neighbour_mode: str = "knn"  # "knn" over households, or "building"
    neighbour_k: int = 10
    neighbour_radius: float | None = 50.0  # metres; None for no limit
    building_cap: object = None  # "building" mode: random links each person names
    building_with_household: bool = False  # "building" mode: do co-residents of a household count as neighbours

    # School layer
    school_unit: str = "year"  # "class", "year", "level_municipality_year", or "graduation_cohort"
    school_max_years_since: int | None = None  # "graduation_cohort": only cohorts that finished this recently
    school_cap: object = None  # random links each person names in larger groups
    school_drop_over: int | None = None  # groups larger than this are dropped

    # Family layer (used only if "kin" is among the layers)
    kin_types: tuple[str, ...] = KIN_CORE
    kin_parent_gap: tuple[int, int] | None = None  # keep parent links with an age gap in this range
    kin_away_max_child_age: int | None = None  # a parent living elsewhere is known only up to this child age
    kin_partner_max_gap: int | None = None  # keep partners at most this many years apart

    # Who is in the network
    min_age: int | None = None
    layers: tuple[str, ...] = CORE_LAYERS

    def but(self, name: str | None = None, **changes) -> "RuleSet":
        """A copy with some rules changed."""
        return replace(self, name=name or self.name, **changes)


def _work_groups(world: World, rs: RuleSet) -> np.ndarray:
    if rs.work_unit == "employer":
        return world.employer
    if rs.work_unit == "site":
        return world.site
    if rs.work_unit == "site_if_large":
        big = world.group_sizes(world.employer) >= rs.work_split_min
        n_emp = int(world.employer.max()) + 1 if world.n else 0
        return np.where(big, world.site + n_emp, world.employer)
    raise ValueError(f"unknown work_unit: {rs.work_unit}")


def _school_groups(world: World, rs: RuleSet) -> np.ndarray:
    if rs.school_unit == "class":
        return world.school_class
    if rs.school_unit == "year":
        return world.year_group
    if rs.school_unit == "level_municipality_year":
        return rules.compose_groups(
            np.where(world.school >= 0, world.school_level, -1),
            np.where(world.school >= 0, world.municipality, -1),
            world.school_year,
        )
    if rs.school_unit == "graduation_cohort":
        # Everyone who completed the same level, in the same municipality, in the same year.
        years = world.grad_years_ago
        if rs.school_max_years_since is not None:
            years = np.where(years <= rs.school_max_years_since, years, -1)
        return rules.compose_groups(world.edu_level, np.where(years >= 0, world.municipality, -1), years)
    raise ValueError(f"unknown school_unit: {rs.school_unit}")


def _kin_links(world: World, rs: RuleSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parent, partner and late-parent links as the rule-set can see them."""
    parent = world.parent.copy()
    partner = world.partner.copy()
    late = world.late_parent.copy()
    if rs.kin_away_max_child_age is not None:  # a late parent was never seen sharing a home
        late[world.age > rs.kin_away_max_child_age] = -1
    child = np.arange(world.n)
    for c in (0, 1):
        p = parent[:, c]
        has = p >= 0
        gap = world.age[np.maximum(p, 0)] - world.age
        drop = np.zeros(world.n, dtype=bool)
        if rs.kin_parent_gap is not None:
            lo, hi = rs.kin_parent_gap
            drop |= (gap < lo) | (gap > hi)
        if rs.kin_away_max_child_age is not None:
            away = world.household[np.maximum(p, 0)] != world.household[child]
            drop |= away & (world.age > rs.kin_away_max_child_age)
        parent[has & drop, c] = -1
    if rs.kin_partner_max_gap is not None:
        has = partner >= 0
        gap = np.abs(world.age[np.maximum(partner, 0)] - world.age)
        partner[has & (gap > rs.kin_partner_max_gap)] = -1
    return parent, partner, late


def build(world: World, rs: RuleSet, seed: int = 0) -> MultilayerNetwork:
    """Wire `world` with rule-set `rs`."""
    active = np.ones(world.n, dtype=bool)
    if rs.min_age is not None:
        active = world.age >= rs.min_age

    def only_active(groups: np.ndarray) -> np.ndarray:
        return np.where(active, groups, -1)

    streams = {
        name: np.random.default_rng(np.random.SeedSequence([seed, i]))
        for i, name in enumerate(LAYER_ORDER)
    }
    layers = {}
    for name in LAYER_ORDER:
        if name not in rs.layers:
            continue
        rng = streams[name]
        if name == "household":
            e = rules.clique_edges(only_active(world.household))
        elif name == "neighbour":
            if rs.neighbour_mode == "knn":
                e = rules.household_neighbour_edges(
                    world.household, world.household_xy, rs.neighbour_k, rs.neighbour_radius, rng
                )
                e = e[active[e[:, 0]] & active[e[:, 1]]]
            elif rs.neighbour_mode == "building":
                # The unit is the property, which is the building unless the population groups buildings.
                e = rules.capped_group_edges(
                    only_active(world.plot), rs.building_cap, "random", rng=rng
                )
                if not rs.building_with_household:
                    e = e[world.household[e[:, 0]] != world.household[e[:, 1]]]
            else:
                raise ValueError(f"unknown neighbour_mode: {rs.neighbour_mode}")
        elif name == "school":
            g = only_active(_school_groups(world, rs))
            if rs.school_drop_over is not None:
                g = np.where(world.group_sizes(g) > rs.school_drop_over, -1, g)
            e = rules.capped_group_edges(g, rs.school_cap, "random", rng=rng)
        elif name == "kin":
            # Relations are derived from every link, then people outside the network are dropped.
            parent, partner, late = _kin_links(world, rs)
            e = rules.kin_edges(parent, partner, rs.kin_types, late)
            e = e[active[e[:, 0]] & active[e[:, 1]]]
        else:  # work
            e = rules.capped_group_edges(
                only_active(_work_groups(world, rs)),
                rs.work_cap,
                rs.work_select,
                world.xy,
                rs.work_full_if_leq,
                rng,
            )
        layers[name] = edges_to_csr(e, world.n)
    return MultilayerNetwork(world.n, layers, active)


# --- Four-layer presets ---------------------------------------------------------
# Checked against the published sources (docs/rule-sets.md) on 10 November 2026. Each is an approximation; the
# comments say what is left out. These have no family layer.


def se_k(n: int) -> int:
    """Swedish link function: random out-links each person draws in a group of n.

    From the fitted values in Wohlert (2024), as read in docs/rule-sets.md:
    about 10.37 * n**0.014 below 1,000 and 3.112 + 0.014 * n from 1,000. The
    sign of the exponent and the rounding are inferences, not printed values.
    """
    if n < 1000:
        return int(round(10.37189 * n**0.014))
    return int(round(3.112 + 0.014 * n))


def se_k_code(n: int) -> int:
    """The Swedish link function with the constants of the builders' released scripts (28 October 2026).

    Below 1,000 members it gives the same numbers as `se_k`; above, 3.1121 + 0.014456 n instead of
    3.112 + 0.014 n. An independent implementation of the published formula; no code was reused.
    """
    if n < 1000:
        return int(round(1037.18861929423 * n ** (-0.9857059767416111) * n * 0.01))
    return int(round(3.1121031746058954 + 0.014456349204199803 * n))


# Dutch 2018 file. Company, nearest 100 by home, 10 closest households within
# 50 m, school year groups. Left out: institutional households; the directed
# form (the union is used, as the analysis papers do).
NL_2018 = RuleSet("NL-2018")

# Dutch yearly series. As NL-2018 without the distance limit. Left out: the
# monthly pooling over jobs; the second neighbour relation (up to 20 random
# people within 200 m), which Bokanyi et al. 2026 also exclude.
NL_SERIES = NL_2018.but("NL-series", neighbour_radius=None)

# Denmark. Workplace (site), at most 100 fully linked, otherwise random ties
# with nobody above 100 (reading a). DK_PAIRS is reading b. Left out: people
# with a fictional workplace; flatmates counted as neighbours.
DK = NL_2018.but("DK-like", work_unit="site", work_select="random_capped")
DK_PAIRS = NL_2018.but("DK-like (pairs)", work_unit="site", work_select="random_pairs")

# Sweden, first version (10 November). No fixed cap: everyone draws se_k(n)
# random out-links in each group. Kept so that the first reports can be
# reproduced. It leaves most people without school or neighbour ties, which
# the published Swedish network does not; SE_LIKE_K below corrects that.
# Left out: the probability rule for workplaces "too big or remote"; the
# split by field of study above 1,000; graduation cohorts of adults (only
# current students aged 15 and over have a school group here).
SE_LIKE = RuleSet(
    "SE-like",
    work_unit="site_if_large",
    work_split_min=200,
    work_cap=se_k,
    work_select="random",
    neighbour_mode="building",
    building_cap=se_k,
    school_unit="level_municipality_year",
    school_cap=se_k,
    school_drop_over=10_000,
    min_age=15,
)

PRESETS = {rs.name: rs for rs in (NL_2018, NL_SERIES, DK, DK_PAIRS, SE_LIKE)}

# --- Five-layer presets: the same rules plus each rule-set's family layer ------
# Checked against the published sources (docs/rule-sets.md) on 11 November 2026.

# Dutch 2018 file and Denmark: eleven relation types extrapolated from legal
# parent-child records, no relations by marriage. Partners are household ties.
# Left out: the half/full/unknown split of siblings (one tie either way).
NL_2018_K = NL_2018.but("NL-2018+kin", layers=ALL_LAYERS)
DK_K = DK.but("DK-like+kin", layers=ALL_LAYERS)
DK_PAIRS_K = DK_PAIRS.but("DK-like (pairs)+kin", layers=ALL_LAYERS)

# Dutch yearly series: adds partners and in-laws. Left out: step-family and
# the partners of cousins, of nieces and nephews and of aunts and uncles.
NL_SERIES_K = NL_SERIES.but(
    "NL-series+kin", layers=ALL_LAYERS, kin_types=KIN_CORE + ("partner", "in_law")
)

# Sweden: kin is inferred from co-residence and age gaps, pooled over 2000 to
# 2017. Partners are at most 13 years apart; parent and child are 15 to 60
# years apart. No co-parent type. A parent living elsewhere is known only if
# parent and child shared a home in those years; the cut-off of 37 (aged 20
# or under in 2000) is my approximation of that, not a published value.
#
# School: everyone who completed the same level of education in the same
# municipality in the same year (docs/rule-sets.md), for cohorts of the last 19
# years, which is my reading of the 2000 to 2017 window plus the two earlier
# years the thesis mentions. Neighbours: everyone on the same property,
# household members included; the published degree distribution has 1.5
# million people with exactly one neighbour tie and almost nobody with none,
# which fits only if co-residents count. A building stands in for a property.
SE_LIKE_K = SE_LIKE.but(
    "SE-like+kin",
    layers=ALL_LAYERS,
    school_unit="graduation_cohort",
    school_max_years_since=19,
    building_with_household=True,
    kin_types=("parent", "sibling", "partner", "grandparent", "aunt_uncle", "cousin"),
    kin_parent_gap=(15, 60),
    kin_away_max_child_age=37,
    kin_partner_max_gap=13,
)

PRESETS_KIN = {rs.name: rs for rs in (NL_2018_K, NL_SERIES_K, DK_K, DK_PAIRS_K, SE_LIKE_K)}
