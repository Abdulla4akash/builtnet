---
title: 'builtnet: construction rules of register-based population networks, as code'
tags:
  - Python
  - social networks
  - population registers
  - multilayer networks
  - network science
authors:
  - name: S M Abdulla Al Mamun
    affiliation: 1
affiliations:
  - name: Independent researcher, Manchester, United Kingdom
    index: 1
date: 13 November 2026
bibliography: paper.bib
---

# Summary

Statistics Netherlands, Statistics Denmark and research groups in Sweden have built social networks of whole national populations from administrative registers [@vanderlaan2022; @vanderlaan2023; @cremers2025; @panayiotou2025]. A tie in these networks is not observed. A rule produces it from memberships. The Dutch 2018 file links an employee of a large company to the 100 colleagues who live nearest. The Danish network links a random sample. The Swedish network gives each person a number of random links that depends on group size. Neighbour ties come from the ten nearest households, within 50 metres in one release and with no limit in the next. The networks are analysed for social structure, segregation and cohesion [@bokanyi2023; @kazmina2024; @bokanyi2026], and comparisons between countries and releases are being prepared. Those comparisons mix differences between populations with differences between rules [@almamun2026].

`builtnet` states these rules as code. It has four parts:

- machine-readable construction cards for the published rule-sets and their documented alternative readings;
- an implementation of each rule that applies a card to a population;
- a real-data interface that takes membership tables;
- theory that predicts what a rule does to the network from group sizes and local density alone, without building edges.

# Statement of need

The groups that hold register data can rebuild their own network under another rule, but none of the builders publishes code, and the rules are spread over discussion papers, catalogue notes and theses in three languages. A reader who wants to know whether two networks are comparable has to reconstruct the rules by hand. `builtnet` does that once, with a source and a status (*documented*, *inferred* or *assumed*) for every field of every card, and makes three things possible:

1. **Rebuild and compare.** Inside a secure environment, `builtnet.tables.world_from_tables` turns membership tables into a population and `builtnet.rulesets.build` applies any card. The core needs only numpy and scipy, so it runs where few packages are installed.
2. **Experiment with known truth.** Synthetic populations (`builtnet.world`, `builtnet.scenarios`) let researchers cross populations with rule-sets. `builtnet.measures.construction_share` splits a naive difference between two networks into a population part and a construction part (a Kitagawa–Das Gupta decomposition [@kitagawa1955; @dasgupta1993]).
3. **Predict without building.** `builtnet.theory` gives the expected degree and within-layer clustering of a member of a group of n under each rule. It covers full linking, k random out-links ($2k - k^2/(n-1)$), the capped draw, random pairs and k nearest by home, and gives the household-neighbour rule as a function of local household density. The full-linking results and the share of mutual nearest neighbours are classical [@newman2001; @schilling1986]; the package maps them to the register rules. Checked against simulation, the group rules agree within 1%. Formulas fed with open statistics reproduce published figures of the real networks, for example the out-degree distribution of Dutch primary-school classmates [@almamun2026].

The package also contains a reporting standard (`docs/construction-card.md`), the one-page statement that a release of a register network would need for comparison, and three readings of excess closure, a measure whose definitions disagree on networks with many multi-layer ties (`docs/excess-closure.md`).

# Design

Rules are operators on memberships. A `RuleSet` is a frozen dataclass whose fields are the choices a builder makes: the unit of each layer, the size threshold, the cap, the selection (nearest by home, random, capped draw, random pairs), the distance limit and the eligibility conditions. Cards in TOML map onto these fields, so a new reading is a new card rather than new code. Layers are stored as sparse matrices; triangle counts use sparse products, with no dependence on igraph. Every random draw takes an explicit generator, so a network is a deterministic function of population, card and seed.

# Limitations

The cards follow public documents. Where these are silent or conflict (for example the sampling algorithm for large Danish workplaces), the card states the reading and marks it *inferred* or *assumed*, and alternative readings are provided. The cards are not the builders' code. Kin layers follow parent links and do not reproduce every relation type of every release.

# Acknowledgements

The code, tests and text were written with substantial help from AI assistants (Claude, Anthropic). The author reviewed the work and takes responsibility for it.

# References
