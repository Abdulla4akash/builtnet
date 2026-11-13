# The four rule-sets, as published

How the four register-based population networks turn memberships into ties, read from their primary documents. Short paraphrases with page references; read the sources for the exact wording. The presets in `builtnet.rulesets` and the cards in `builtnet/cards/` follow this table; each preset approximates its source and lists what it leaves out.

## Sources

| Key | Source |
|---|---|
| CBS22 | van der Laan, D. J. (2022). *A person network of the Netherlands.* Statistics Netherlands discussion paper. https://www.cbs.nl/-/media/_pdf/2022/20/person_network_netherlands.pdf |
| ESR23 | van der Laan, J., de Jonge, E., Das, M., te Riele, S. and Emery, T. (2023). A whole population network and its application for the social sciences. *European Sociological Review* 39(1), 145–160. https://doi.org/10.1093/esr/jcac026 |
| CBS23 | Statistics Netherlands (2023). *Tijdreeks persoonsnetwerkbestanden: overzicht van de verschillen met de eerste versie van het persoonsnetwerk.* |
| CAT-COL, CAT-BUR, CAT-KLA, CAT-HUI, CAT-FAM | Statistics Netherlands Microdata Services (2026). File documentation of the yearly person-network files: colleagues, neighbours, classmates, household members, family |
| DK25 | Cremers et al. (2025). *Scientific Reports* 15, 18383. https://doi.org/10.1038/s41598-025-98072-2; DK25S its supplementary information |
| SE25 | Panayiotou, Wohlert, Bask, Bask, Magnani and Mäkinen (2025). *Scientific Reports* 15, 30300. https://doi.org/10.1038/s41598-025-15966-x |
| SETH | Wohlert, I. K. (2024). *Creation and analysis of a population-scale social network based on Swedish registry data.* Master's thesis, Uppsala University. DiVA diva2:1887403 |

## Side by side

| | NL-2018 (2018 file) | NL-series (yearly files) | DK | SE |
|---|---|---|---|---|
| Nodes, date | Everyone in the population register on 1 October 2018 (CBS22 p.3) | Everyone in the register on 1 January of each year, 2009 onwards (CAT-FAM p.6, CAT-COL p.3) | Registered residents, yearly 2008 to 2021 (DK25 p.3) | People aged 15 and over with register data in 2017 (SE25 p.2, SETH p.26) |
| Direction | Directed; neighbour and large-company colleague edges one-way (CBS22 pp.3, 11) | Colleague and neighbour files not symmetric (CAT-COL p.7, CAT-BUR p.7) | Neighbour sampling one-way; sampled colleague edges made two-way (DK25S p.8) | Built from out-links, analysed as undirected (SE25 p.10, SETH p.29) |
| Work | Company. Up to 101 employees: all pairs. Larger: each person's 100 colleagues living nearest, random tie-break (CBS22 p.8) | Per month and company up to 100 nearest; pooled over months and jobs; then the 100 nearest (CAT-COL pp.6–7) | Workplace. Up to 100: all pairs. Larger: random sample of 100, made two-way (DK25 p.3, DK25S p.8) | Company under 200 employees: k(n) random out-links per person; larger companies split into workplaces first (SE25 p.10, SETH p.41) |
| Neighbours | Members of up to 10 closest households within 50 m; random tie-break at position 10 (CBS22 p.7) | Residents of the 10 nearest addresses, no distance limit; plus up to 20 random people within 200 m as a separate relation (CAT-BUR pp.6–7) | Members of the 10 closest households within 50 m (DK25 p.3) | Same property; k(n) random out-links per person (SE25 p.10, SETH pp.40–41) |
| School | Class groups from school, location, year and type (CBS22 p.8, Table 2.2) | Six levels; location dropped for higher education; special education by age (CAT-KLA p.7, CBS23 p.8) | Same school, programme and grade, or start-year cohort in tertiary education (DK25 p.3) | Same level, municipality and graduation year; split by field from 1,000; groups over 10,000 dropped; k(n) random out-links (SE25 pp.9–10) |
| Household | Statistics Netherlands household (CBS22 p.6) | Same address; partners moved to the family layer (CAT-HUI p.6, CBS23 p.3) | Statistics Denmark household on 1 January (DK25 p.3, DK25S p.8) | Co-resident relatives only (SE25 p.9) |
| Family | From legal parent–child records; no relations by marriage (CBS22 p.3, ESR23 p.3) | Adds partners, in-laws, step-family and partners of kin (CAT-FAM p.6) | Eleven relation types from legal parent–child records (DK25 p.3) | Inferred from co-residence and age gaps, pooled over 2000 to 2017 (SE25 p.9) |

**k(n), the Swedish link function** (SETH): about 10.37 n^0.014 random out-links in a group of n below 1,000 members (about eleven), and about 3.112 + 0.014 n from 1,000. The builders' released scripts use 3.1121 + 0.014456 n above 1,000; below 1,000 both give the same numbers after rounding.

## Known open points

- **Classmate cap in the 2018 file.** Applications of the first Dutch network report a random selection of 100 classmates for people with more; the discussion paper describes no cap. `NL_2018_K` links the whole group; the alternative is `NL_2018_K.but(school_cap=100)`.
- **Colleague cap in the yearly series.** File documentation: at most 100 colleagues per person over all employers; a later Statistics Netherlands publication: up to 100 per company.
- **Danish sampling of large workplaces.** The algorithm is not published. Three readings are implemented: (a) random ties with nobody above 100 (`dk`, `random_capped`), (b) random pairs with a mean of 100 (`dk-pairs`), and (c) each person draws 100 colleagues and every tie is made two-way (`dk-twoway`, `random`). Readings (a) and (b) give the same results in our experiments. The published colleague edge count for 2021 (125.07 million, DK25S Table 1) is reproduced within 2.4% by (c) from Statistics Denmark's workplace sizes, and is 30.6% higher than (a) gives. Under (c), however, nobody in a large workplace has fewer than 100 colleagues, which the Danish supplement contradicts. Under (c) the Danish work layer gives people in large workplaces about 150 to 160 colleagues, close to the Dutch nearest-100 union.
- **Swedish school layer.** The released scripts keep every graduation cohort; the municipality or year of education is missing for a third or more of people, and the documents do not say who.
