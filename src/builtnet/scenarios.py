"""Named populations, planted scenarios and seed lists.

The populations are stylised, not fitted to any country. Three settings are
chosen so that the reference population matches a published figure for the
Dutch 2018 file when it is wired with the Dutch 2018 rules:

  - employer-size tail and ceiling: about two thirds of workers are in
    employers above 101 employees (CBS 2022, Table 3.1: 2.59 million at or
    below, 4.90 million above; docs/rule-sets.md);
  - share of households in built-up areas and their peak density: about 2.6%
    of people have no neighbour tie (same table: 16.81 million of 17.25
    million people have one);
  - the kin settings in `world.WorldConfig`: about 15 kin ties per person
    (CBS 2022 Table 3.2 and van der Laan et al. 2023 Figure 1: 241.6 to 270.2
    million directed family edges; docs/rule-sets.md).

Everything else is illustrative.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .world import World, WorldConfig

# --- Seeds ---------------------------------------------------------------------
# Development seeds are for building, debugging and choosing settings.
# Evaluation seeds are kept apart from development seeds, so that a pre-registered run never reuses a population seen during development.
DEV_SEEDS = range(0, 1000)
EVALUATION_SEEDS = range(1000, 2000)


def is_evaluation_seed(seed: int) -> bool:
    return seed >= EVALUATION_SEEDS.start


# --- Scales --------------------------------------------------------------------
# "main" is about 96,000 people. "small" is about 9,600, for quick checks only:
# its largest employer is a tenth of the main one, so cap effects are weaker.
SCALES = {
    "main": dict(n_households=45_000, region_km=38.0, n_cores=12, employer_max_size=5000),
    "small": dict(n_households=4_500, region_km=12.0, n_cores=4, employer_max_size=500),
}
_REFERENCE = dict(employer_tail=0.9, core_share=0.995, core_peak_density=5000.0)


def base(scale: str = "main", **changes) -> WorldConfig:
    """The reference population for one-rule-at-a-time and planted-gap experiments."""
    return WorldConfig(**{**_REFERENCE, **SCALES[scale], **changes})


# --- Stylised pair (plan section 5.1) ---------------------------------------------
# They differ in settlement density, household sizes and employer-size mix.
# "compact" is the reference population with more flats. "dispersed" has an
# eighth of its households outside built-up areas, lower densities, smaller
# households and fewer very large employers. Its values are illustrative.
def compact(scale: str = "main") -> WorldConfig:
    s = SCALES[scale]
    return WorldConfig(
        n_households=s["n_households"],
        region_km=s["region_km"] * 10 / 12,
        n_cores=s["n_cores"],
        core_share=0.995,
        core_peak_density=6000.0,
        apartment_share_core=0.55,
        employer_tail=0.9,
        employer_max_size=s["employer_max_size"],
    )


def dispersed(scale: str = "main") -> WorldConfig:
    s = SCALES[scale]
    return WorldConfig(
        n_households=int(s["n_households"] * 1.05),  # smaller households, similar head count
        region_km=s["region_km"] * 30 / 12,
        n_cores=s["n_cores"],
        core_share=0.87,
        core_peak_density=1500.0,
        apartment_share_core=0.20,
        household_size_p=(0.43, 0.31, 0.11, 0.10, 0.05),
        employer_tail=1.0,
        employer_max_size=int(s["employer_max_size"] * 0.6),
    )


def sweden_like(scale: str = "main") -> WorldConfig:
    """The dispersed population with three features the Swedish rules act on.

    Seven in ten built-up households live in flats; blocks of flats are
    grouped into properties of about four blocks; small households are more
    often in flats; and each built-up area is its own municipality, so
    municipality sizes are unequal. The share of flats is chosen so that about
    half of all people have more than ten neighbour ties under the Swedish-like
    rules, as the published degree distribution suggests (Panayiotou et al.
    2025, Figure 1 and page 4; read from a figure, so approximate). See
    experiments/diagnostics/swedish_like_check.py for how the Swedish-like
    rule-set then compares with the published network.
    """
    return replace(
        dispersed(scale),
        apartment_share_core=0.7,
        property_mean_buildings=4.0,
        flat_small_household_bias=0.6,
        n_municipalities=SCALES[scale]["n_cores"],
    )


# --- Planted income gradients (plan section 5.5, E3) --------------------------------
# Each mechanism is one generator parameter at four strengths. "Top" is income
# group 5 and "bottom" is group 1; a positive value favours the top group.
STRENGTHS = ("none", "small", "medium", "large")
MECHANISMS = {
    "large_employers": ("size_income_slope", (0.0, 0.2, 0.4, 0.6)),
    "density": ("density_income_slope", (0.0, 0.2, 0.4, 0.6)),
    "participation": ("employment_income_slope", (0.0, 0.1, 0.2, 0.3)),
    "overlap": ("overlap_income_slope", (0.0, 0.25, 0.5, 0.75)),
}


def planted(scale: str = "main") -> dict[str, WorldConfig]:
    """All planted scenarios. The 'none' arm is shared by the four mechanisms."""
    out = {"none": base(scale)}
    for mech, (field, values) in MECHANISMS.items():
        for name, v in zip(STRENGTHS[1:], values[1:]):
            out[f"{mech}:{name}"] = base(scale, **{field: v})
    return out


# --- Simulators for harmonisation method M4 (plan section 5.5, E4) -------------------
def simulators(scale: str = "main", target: WorldConfig | None = None) -> dict[str, WorldConfig]:
    """Stand-ins for the other country's population, right and deliberately wrong.

    "correct" has the settings of the target population, the dispersed one by
    default (a fresh seed is used when it is generated). The others are wrong
    in one stated way.
    """
    d = dispersed(scale) if target is None else target
    return {
        "correct": d,
        "wrong_travel": replace(d, work_decay_m=2 * d.work_decay_m, school_decay_m=2 * d.school_decay_m),
        "wrong_firms": replace(d, employer_tail=1.2, employer_max_size=max(2, d.employer_max_size // 3)),
        "wrong_everything": base(scale),  # a generic population: settlement, households and firms all differ
    }


# --- Relocation (plan section 5.5, E5) -----------------------------------------------
def relocated(world: World, share: float, mean_move_m: float, seed: int) -> World:
    """A second snapshot: the same memberships, with some households moved away.

    A random `share` of households move by a distance that is exponential
    with mean `mean_move_m`, in a random direction, staying inside the
    region. A moved household gets a building of its own.
    """
    rng = np.random.default_rng(np.random.SeedSequence([world.seed, seed, 5001]))
    n_h = world.household_xy.shape[0]
    side = world.config.region_km * 1000.0
    move = rng.random(n_h) < share
    dist = rng.exponential(mean_move_m, size=n_h)
    angle = rng.uniform(0.0, 2 * np.pi, size=n_h)
    h_xy = world.household_xy.copy()
    step = np.stack([dist * np.cos(angle), dist * np.sin(angle)], axis=1)
    h_xy[move] = np.clip(h_xy[move] + step[move], 0.0, side)
    h_building = np.full(n_h, -1, dtype=np.int64)
    h_building[world.household] = world.building
    new_ids = int(world.building.max()) + 1 + np.arange(int(move.sum()))
    h_building[move] = new_ids
    return replace(world, xy=h_xy[world.household], household_xy=h_xy, building=h_building[world.household])
