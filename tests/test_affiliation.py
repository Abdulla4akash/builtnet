import numpy as np

from builtnet import world as W
from builtnet.affiliation import affiliation_descriptors


def tiny_world() -> W.World:
    """Four people: 0 and 1 share a home; 0, 1 and 2 share an employer; 3 is alone."""
    n = 4
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [300.0, 400.0], [9000.0, 0.0]])
    household = np.array([0, 0, 1, 2])
    none = np.full(n, -1)
    return W.World(
        config=W.WorldConfig(), seed=0, n=n, xy=xy, household=household, building=household.copy(),
        municipality=np.zeros(n, dtype=int), role=np.full(n, W.ADULT), age=np.full(n, 40), income=np.full(n, 3),
        school=none.copy(), school_level=none.copy(), school_year=none.copy(), school_class=none.copy(),
        employer=np.array([0, 0, 0, -1]), site=np.array([0, 0, 1, -1]),
        household_xy=xy[[0, 2, 3]], site_xy=np.zeros((2, 2)), site_employer=np.array([0, 0]),
    )


def test_hand_example():
    d = affiliation_descriptors(tiny_world())
    assert d["n_contexts"].tolist() == [2, 2, 1, 0]
    assert d["size_household"].tolist() == [2, 2, 1, 1]
    assert d["size_work"][:3].tolist() == [3, 3, 3] and np.isnan(d["size_work"][3])
    assert np.all(np.isnan(d["size_school"]))
    # Person 0 shares something with 1 and 2; with 1 it is both home and work.
    assert d["comember_overlap"][:3].tolist() == [0.5, 0.5, 0.0] and np.isnan(d["comember_overlap"][3])
    assert d["dist_work"][:3].tolist() == [250.0, 250.0, 500.0] and np.isnan(d["dist_work"][3])


def test_unit_definition_changes_the_table_descriptors():
    d = affiliation_descriptors(tiny_world(), work_unit="site")
    assert d["size_work"][:3].tolist() == [2, 2, 1]
    assert d["dist_work"][:2].tolist() == [0.0, 0.0] and np.isnan(d["dist_work"][2])


def test_generated_world_and_sampling_of_large_groups():
    w = W.generate(W.WorldConfig(n_households=1500), 2)
    full = affiliation_descriptors(w)
    sampled = affiliation_descriptors(w, max_sample=50)
    assert np.array_equal(np.isfinite(full["dist_work"]), np.isfinite(sampled["dist_work"]))
    ok = np.isfinite(full["dist_work"])
    assert np.corrcoef(full["dist_work"][ok], sampled["dist_work"][ok])[0, 1] > 0.9
    assert np.nanmax(full["comember_overlap"]) <= 1 and np.nanmin(full["comember_overlap"]) >= 0
