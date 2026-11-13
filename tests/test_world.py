import numpy as np

from builtnet import world as W
from builtnet.rulesets import DK, NL_2018, NL_SERIES, SE_LIKE, build

CFG = W.WorldConfig(n_households=1500)


def test_generation_is_deterministic_and_seed_dependent():
    a, b, c = W.generate(CFG, 1), W.generate(CFG, 1), W.generate(CFG, 2)
    for name in ("xy", "household", "income", "school_class", "employer", "site"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.n != c.n or not np.array_equal(a.xy, c.xy)


def test_tables_are_internally_consistent():
    w = W.generate(CFG, 3)
    assert w.household.max() + 1 == CFG.n_households
    # One household, one home, one building, one income group.
    for arr in (w.xy[:, 0], w.xy[:, 1], w.building, w.income, w.municipality):
        assert np.all(arr == arr[np.unique(w.household, return_index=True)[1]][w.household])
    # A building has one location.
    first = np.unique(w.building, return_index=True)[1]
    lookup = dict(zip(w.building[first].tolist(), map(tuple, w.xy[first])))
    assert all(lookup[b] == tuple(p) for b, p in zip(w.building.tolist(), w.xy))
    # Income groups are fifths of households.
    h_income = w.income[np.unique(w.household, return_index=True)[1]]
    assert np.all(np.abs(np.bincount(h_income, minlength=6)[1:] / CFG.n_households - 0.2) < 0.01)
    assert set(np.unique(w.role)) <= {W.CHILD, W.ADULT, W.SENIOR}
    assert np.all((w.age[w.role == W.CHILD] < 18) & (w.age[w.role == W.CHILD] >= 0))
    assert np.all(w.age[w.role == W.ADULT] >= 18)


def test_schools():
    w = W.generate(CFG, 4)
    in_school = w.school >= 0
    assert np.array_equal(in_school, (w.age >= 4) & (w.age <= 17))
    assert np.all(w.school_level[in_school & (w.age <= 11)] == W.PRIMARY)
    assert np.all(w.school_level[in_school & (w.age >= 12)] == W.SECONDARY)
    sizes = np.bincount(w.school_class[in_school])
    assert sizes.max() <= CFG.class_max and sizes.min() >= 1
    # A class sits inside one school year.
    yg = w.year_group[in_school]
    cl = w.school_class[in_school]
    assert len(set(zip(cl.tolist(), yg.tolist()))) == len(set(cl.tolist()))
    assert np.all(w.school_class[~in_school] == -1)


def test_work():
    w = W.generate(CFG, 5)
    employed = w.employer >= 0
    assert np.all(w.role[employed] == W.ADULT)
    assert np.array_equal(employed, w.site >= 0)
    assert np.all(w.site_employer[w.site[employed]] == w.employer[employed])
    assert 0.65 < employed[w.role == W.ADULT].mean() < 0.85
    sizes = np.bincount(w.employer[employed])
    assert sizes.min() >= 1 and sizes.max() <= max(2, int(CFG.employer_max_share * employed.sum()))


def test_planted_size_income_slope_moves_high_income_into_large_employers():
    base = W.generate(CFG, 6)
    planted = W.generate(W.WorldConfig(n_households=1500, size_income_slope=0.9), 6)
    def gap(w):
        emp = w.employer >= 0
        size = np.log(w.group_sizes(w.employer)[emp])
        return size[w.income[emp] == 5].mean() - size[w.income[emp] == 1].mean()
    assert gap(planted) > gap(base) + 0.5
    # Homes and households are untouched by a work-side parameter.
    assert np.array_equal(base.xy, planted.xy) and np.array_equal(base.income, planted.income)


def test_built_layers_follow_the_rules():
    w = W.generate(CFG, 7)
    net = build(w, NL_2018, seed=0)
    deg = {k: np.asarray(a.sum(axis=1)).ravel() for k, a in net.layers.items()}
    assert np.array_equal(deg["household"], w.group_sizes(w.household) - 1)
    emp_size = w.group_sizes(w.employer)
    small = (emp_size > 0) & (emp_size <= 101)
    assert np.array_equal(deg["work"][small], emp_size[small] - 1)
    assert np.all(deg["work"][emp_size > 101] >= 100)
    assert np.all(deg["work"][emp_size == 0] == 0)
    yg = w.group_sizes(w.year_group)
    assert np.array_equal(deg["school"], np.maximum(yg - 1, 0))
    nb = net.layers["neighbour"].tocoo()
    assert np.all(w.household[nb.row] != w.household[nb.col])
    d = np.sqrt(((w.xy[nb.row] - w.xy[nb.col]) ** 2).sum(axis=1))
    assert d.max() < 50.0


def test_changing_one_rule_leaves_other_layers_identical():
    w = W.generate(CFG, 8)
    a, b, c = build(w, NL_2018, 0), build(w, DK, 0), build(w, NL_SERIES, 0)
    assert (a.layers["neighbour"] != b.layers["neighbour"]).nnz == 0
    assert (a.layers["school"] != b.layers["school"]).nnz == 0
    assert (a.layers["work"] != c.layers["work"]).nnz == 0
    # Without a radius everyone has neighbours.
    assert np.all(np.asarray(c.layers["neighbour"].sum(axis=1)).ravel() >= 1)


def test_age_limit_removes_children_and_keeps_indices():
    w = W.generate(CFG, 9)
    net = build(w, SE_LIKE, 0)
    assert net.n == w.n
    assert np.array_equal(net.active, w.age >= 15)
    flat_deg = np.asarray(net.flattened().sum(axis=1)).ravel()
    assert np.all(flat_deg[~net.active] == 0)


def test_education_is_consistent_with_age_and_leaves_other_tables_alone():
    w = W.generate(CFG, 9)
    assert np.all(w.edu_level[w.age < 16] == -1) and np.all(w.grad_years_ago[w.age < 16] == -1)
    done = w.edu_level >= 0
    assert np.all(w.age[done] >= 16) and np.all(w.grad_years_ago[done] >= 0)
    assert np.all(w.age[w.edu_level == 1] - w.grad_years_ago[w.edu_level == 1] == 19)
    finished_at = w.age[w.edu_level == 2] - w.grad_years_ago[w.edu_level == 2]
    assert finished_at.min() >= 22 and finished_at.max() <= 27
    other = W.generate(W.WorldConfig(n_households=1500, tertiary_share=0.6, upper_secondary_share=0.3), 9)
    for name in ("xy", "household", "income", "school_class", "employer", "site", "parent", "partner", "age"):
        assert np.array_equal(getattr(w, name), getattr(other, name)), name
    assert (other.edu_level == 2).mean() > (w.edu_level == 2).mean() + 0.1


def test_swedish_like_school_and_neighbour_groups():
    from builtnet.rulesets import SE_LIKE_K

    w = W.generate(CFG, 10)
    net = build(w, SE_LIKE_K, 0)
    school = np.asarray(net.layers["school"].sum(axis=1)).ravel()
    assert np.all(school[(w.grad_years_ago > 19) | (w.grad_years_ago < 0)] == 0)  # older cohorts are not covered
    recent = (w.grad_years_ago >= 0) & (w.grad_years_ago <= 19) & (w.age >= 15)
    assert (school[recent] > 0).mean() > 0.9
    a = net.layers["school"].tocoo()
    assert np.all(w.edu_level[a.row] == w.edu_level[a.col]) and np.all(w.grad_years_ago[a.row] == w.grad_years_ago[a.col])
    assert np.all(w.municipality[a.row] == w.municipality[a.col])
    # Everyone on a property is a neighbour, household members included.
    neighbour = net.layers["neighbour"]
    b = neighbour.tocoo()
    assert np.all(w.building[b.row] == w.building[b.col])
    residents = np.bincount(w.building)[w.building]
    in_small = np.flatnonzero((w.partner >= 0) & (residents <= 8))  # small enough to be fully linked
    assert in_small.size > 50 and all(neighbour[i, w.partner[i]] == 1 for i in in_small[:100])


def test_properties_group_blocks_and_small_households_move_into_flats():
    plain = W.generate(W.WorldConfig(n_households=3000), 11)
    assert np.array_equal(plain.plot, plain.building)  # by default a property is a building
    cfg = W.WorldConfig(n_households=3000, property_mean_buildings=4.0, flat_small_household_bias=0.6)
    w = W.generate(cfg, 11)
    for name in ("xy", "building", "municipality"):  # where homes are does not change
        first_a = np.unique(plain.household, return_index=True)[1]
        first_b = np.unique(w.household, return_index=True)[1]
        assert np.array_equal(getattr(plain, name)[first_a], getattr(w, name)[first_b]), name
    # A property is a set of whole buildings, and a house is its own property.
    b_first = np.unique(w.building, return_index=True)[1]
    lookup = dict(zip(w.building[b_first].tolist(), w.plot[b_first].tolist()))
    assert all(lookup[b] == p for b, p in zip(w.building.tolist(), w.plot.tolist()))
    h_first = np.unique(w.household, return_index=True)[1]
    households_in_building = np.bincount(w.building[h_first])
    blocks = np.flatnonzero(households_in_building > 1)
    houses = np.flatnonzero(households_in_building == 1)
    assert all(lookup[b] == b for b in houses[:200])
    n_properties = len({lookup[b] for b in blocks})
    assert 2.5 < blocks.size / n_properties < 6
    # Same sizes overall, smaller households in shared buildings.
    size_a = np.bincount(plain.household)
    size_b = np.bincount(w.household)
    assert np.array_equal(np.sort(size_a), np.sort(size_b))
    shared = households_in_building[w.building[h_first]] > 1
    assert size_b[shared].mean() < size_b[~shared].mean() - 0.3
