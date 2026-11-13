# Construction card: a reporting standard for register-based networks

A one-page statement that anyone who releases or analyses a register-based network can fill in. It records the choices that produce the ties, so that two networks can be compared, or rebuilt under one rule-set. The bundled TOML cards in `src/builtnet/cards/` are machine-readable versions; `builtnet.cards.load` turns them into rule-sets.

For each field give the value, its **status** (*documented* in the builder's own documentation, *inferred* from applications or examples, *assumed* by the analyst) and the **source** (document and page).

## Whole network

- **Population and reference date.** Who is a node (all registered residents? an age limit?) and on which date.
- **Direction.** Are edges directed (named by one side) or symmetric? How are directed edges turned into ties in the analysis (union, intersection)?
- **Pooling.** Is the network a snapshot or pooled over a period (months, years)?
- **Layers released.** Which layers, and whether the membership (person–group) tables are released beside the edges.

## For each layer

| Field | What to state |
|---|---|
| Unit | The group a tie is drawn within: household, address, building or property, company or workplace, school class or year group or cohort, family relation |
| Eligibility | Who can be in the layer (age, status, record available) |
| Size threshold | Up to which group size every pair is tied |
| Cap | The number of ties each person names in larger groups, or the function of group size |
| Selection | How the ties are chosen above the threshold: nearest by home, at random, at random with nobody above the cap, random pairs |
| Symmetrisation | Whether a tie needs one or both sides to name it |
| Distance | Any distance limit (for example 50 m), and on what coordinates |
| Drops | Groups dropped or split (for example groups above 10,000; a split by field above 1,000) |
| Missing records | Who lacks the membership record, and how many |
| Changes | Any change of rule between releases, with the date |

## Why

The same planted difference between two populations gives a gap in work-layer degree of 13.9 ties under the Dutch rules, 8.5 under the Danish-like rules and 3.3 under the Swedish-like rules, and between 39% and 98% of a naive difference in degree or clustering between two populations is produced by the rules (*Built, not observed*, 2026). A card is what a reader needs to tell the two parts apart.
