# builtnet

**Construction rules of register-based population networks, as code.**

Statistics offices and research groups in the Netherlands, Denmark and Sweden build social networks of whole populations from administrative registers. A tie in these networks is not observed: a rule produces it from memberships (link an employee to the 100 colleagues who live nearest, or to a random 100, or to a number that depends on workplace size; link a household to the ten nearest households within 50 m, or with no limit). The rules differ between countries and between releases, and the differences change what the networks say.

`builtnet` provides:

- **Construction cards** (`builtnet.cards`): machine-readable descriptions of the published rule-sets (Dutch 2018 file, Dutch yearly series, Danish, Swedish) and documented alternative readings, each field with its source and a status (*documented*, *inferred*, *assumed*). See [docs/rule-sets.md](docs/rule-sets.md) and the reporting standard in [docs/construction-card.md](docs/construction-card.md).
- **The rules as operators** (`builtnet.rules`, `builtnet.rulesets.build`): apply any card to a population and get a multilayer network.
- **A real-data interface** (`builtnet.tables.world_from_tables`): apply a card to your own membership tables (households, employers, workplaces, schools, properties, parent links). Only numpy and scipy are needed, so it runs inside secure environments with few packages.
- **Synthetic populations** (`builtnet.world`, `builtnet.scenarios`) with known truth, for experiments on what rules do.
- **Measures** (`builtnet.descriptors`, `builtnet.measures`): degree by layer, local clustering, three readings of excess closure ([docs/excess-closure.md](docs/excess-closure.md)), multiplex share, and the split of a naive difference between two networks into a population part and a construction part.
- **Theory** (`builtnet.theory`): the expected degree and clustering of each layer under each rule, from group sizes and local density, without building a network.

## Install

```
pip install git+https://github.com/Abdulla4akash/builtnet
```

Python 3.11 or later; numpy and scipy. `pip install "builtnet[analysis]"` adds scikit-learn for position classes.

## Quick start

```python
from builtnet import cards, scenarios, world
from builtnet.rulesets import build
from builtnet.descriptors import position_descriptors

w = world.generate(scenarios.base("small"), seed=1)          # a synthetic population of about 9,600 people
for name in ("nl-2018", "nl-series", "dk", "se"):
    net = build(w, cards.load(name), seed=1)
    d = position_descriptors(net)
    print(name, round(float(d["degree_total"][net.active].mean()), 1), round(float(d["clustering"][net.active].mean()), 3))
```

On your own tables:

```python
from builtnet.tables import world_from_tables

w = world_from_tables(household=hh_id, home_xy=xy_metres, employer=company_id, school=school_id, school_year=year, age=age)
net = build(w, cards.load("nl-2018"), seed=1)
```

Without building anything:

```python
from builtnet import theory
theory.group_degree(500, 100, "nearest")          # Dutch colleague rule in a company of 500: about 115 ties
theory.group_clustering(500, 100, "random_capped") # Danish reading: about 0.20
theory.neighbour_none(2.0)                        # share of households with no neighbour tie when 2 are expected within 50 m
```

## What the theory says, in one table

| Rule | Degree in a group of n, cap k | Clustering inside the layer |
|---|---|---|
| Full linking | n − 1 | 1 |
| k random out-links (Swedish) | 2k − k²/(n − 1) | about the tie density; falls as about 2k/n |
| Capped symmetric draw (Danish) | just below k | just below k/(n − 1) |
| k nearest by home (Dutch) | 1.0 to 1.26 k, independent of density | stays at about 0.6 to 0.7 whatever n |
| 10 nearest households within 50 m | about μ households up to μ ≈ 5, then 11.37; none with probability e^(−μ) | about 0.56 to 0.59 at every density |

The rule decides which feature of the population a measure reports: under random selection, work-layer clustering tracks employer size; under nearest-home selection, it hardly does.

## Status and citation

Version 0.1.0, 13 November 2026. Research software accompanying the paper *Built, not observed: construction rules and the comparability of register-based population networks* (preprint, 2026). Please cite both; see [CITATION.cff](CITATION.cff).

The construction cards approximate their sources, and each lists what it leaves out. Corrections from the builders of the networks are welcome as issues. No register microdata are included or needed.

Licence: MIT.
