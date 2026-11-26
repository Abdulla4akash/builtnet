"""Verify stored aggregates and write the numeric LaTeX macros used in the paper."""
import argparse
from math import exp, isclose
from statistics import median

from evidence import CARDS, LAYERS, PAPER, REPO, card_files, json_read

REGISTERED = {'N4b': True, 'N4c': True, 'N5b': False, 'N5c': False, 'D1b': False}
CAP100 = 'a_capped_draw_registered'
PAIRS100 = 'b_random_pairs_mean_100'
TWO_WAY = 'c_100_random_out_links_made_two_way'


def require(condition, description):
    if not condition:
        raise ValueError(f'Evidence mismatch: {description}')


def fmt(value, places):
    return f'{value:,.{places}f}'.replace(',', '{,}')


def derive():
    s = json_read(PAPER / 'src' / 'software-summary.json')
    t = json_read(REPO / 'src' / 'builtnet' / 'data' / 'theory_tables.json')
    p = json_read(PAPER / 'src' / 'source-summary.json')
    env = s['environment']
    require(env['source_commit'] == 'e90714d250969b9eaa05646eaffbdc6cc4c846b0', 'software source commit')
    require(env['cpus'] == 2 and env['thread_limit'] == 1 and env['seed'] == 1 and env['repeats'] == 3,
            'benchmark design')
    passed = sum(row['passed'] for row in s['tests']['modules'].values())
    skipped = sum(row['skipped'] for row in s['tests']['modules'].values())
    require(passed == 79 and skipped == 0, 'recorded test verdict')
    names = card_files()
    require(len(names) == 9 and set(CARDS).issubset(names), 'catalogue inventory')
    require(len(t['group']) == 105 and len(t['neighbour']) == 32, 'theory grid inventory')
    numbers = {
        'CardCount': len(names), 'BaseCardCount': len(CARDS), 'LayerCount': len(LAYERS),
        'TestCount': passed, 'SkippedTests': skipped, 'TestModules': len(s['tests']['modules']),
        'RepeatCount': env['repeats'], 'CpuCount': env['cpus'], 'ThreadCount': env['thread_limit'],
        'ExampleSeed': env['seed'], 'TestTime': fmt(s['tests']['wall_seconds'], 2),
        'PythonVersion': env['python'], 'NumpyVersion': env['numpy'], 'ScipyVersion': env['scipy'],
        'GroupRows': len(t['group']), 'NeighbourRows': len(t['neighbour']),
        'TheorySeed': str(t['seed']), 'TheoryMembers': fmt(t['target_member_samples'], 0),
    }
    for label, key, households, expected_people, swedish_active in [
        ('Small', 'small', 4500, 9729, 7940), ('Main', 'main', 45000, 96712, 78831),
    ]:
        sample = s['scales'][key]
        n = sample['people']
        require(sample['households'] == households and n == expected_people, f'{key} population size')
        require(set(sample['cards']) == set(CARDS) == set(sample['seconds']), f'{key} card inventory')
        generate = sample['generation_seconds']
        require(len(generate) == env['repeats'] and min(generate) > 0, f'{key} generation times')
        numbers[label + 'People'] = fmt(n, 0)
        numbers[label + 'Households'] = fmt(households, 0)
        numbers[label + 'Generation'] = fmt(median(generate), 2)
        for name, suffix in zip(CARDS, ('Dutch', 'Series', 'Danish', 'Swedish')):
            c = sample['cards'][name]
            expected_active = swedish_active if name == 'se' else n
            require(0 < c['active'] <= n and c['active'] == expected_active, f'{key} {name} active mask')
            require(set(c['layers']) == set(LAYERS) and all(v >= 0 for v in c['layers'].values()),
                    f'{key} {name} layer counts')
            require(max(c['layers'].values()) <= c['unique_edges'] <= sum(c['layers'].values()),
                    f'{key} {name} union')
            require(isclose(c['mean_degree'], 2 * c['unique_edges'] / c['active'], abs_tol=1e-12)
                    and 0 <= c['mean_clustering'] <= 1, f'{key} {name} descriptors')
            defined = c['clustering_defined']
            require(0 < defined <= c['active'], f'{key} {name} defined clustering denominator')
            readme = c['readme_clustering']
            require((readme is not None) == (defined == c['active']), f'{key} {name} README undefined mean')
            if readme is not None:
                require(isclose(readme, c['mean_clustering']), f'{key} {name} README clustering')
            elapsed = sample['seconds'][name]
            require(len(elapsed) == env['repeats'] and all(len(row) == 2 and min(row) > 0 for row in elapsed),
                    f'{key} {name} stage timing')
            head = label + suffix
            numbers[head + 'Active'] = fmt(c['active'], 0)
            numbers[head + 'Defined'] = fmt(defined, 0)
            numbers[head + 'Degree'] = fmt(c['mean_degree'], 1)
            numbers[head + 'Closure'] = fmt(c['mean_clustering'], 3)
            numbers[head + 'Build'] = fmt(median(row[0] for row in elapsed), 2)
            numbers[head + 'Describe'] = fmt(median(row[1] for row in elapsed), 2)

    random_rows = [row for row in t['group'] if row['select'] == 'random']
    degree_error, closure_error = [], []
    for row in random_rows:
        k, n = row['k'], row['n']
        q = k / (n - 1)
        degree_error.append(abs(row['degree'] / (k * (2 - q)) - 1))
        closure_error.append(abs(row['clustering'] - (1 - (1 - q) ** 2)))
    require(0.00033 < max(degree_error) < 0.00034 and 0.0111 < max(closure_error) < 0.0112,
            'random-nomination check against archived grid')
    numbers['RandomDegreeError'] = fmt(100 * max(degree_error), 3)
    numbers['RandomClosureError'] = fmt(max(closure_error), 4)

    matched = [r for r in t['neighbour'] if r['mu'] == 2 and r['radius_m'] is not None]
    require(len(matched) == 1, 'neighbour row at mu=2')
    numbers['NeighbourTwoDegree'] = fmt(matched[0]['linked'], 2)
    numbers['NeighbourTwoNone'] = fmt(matched[0]['none'], 3)
    numbers['NeighbourTwoPoisson'] = fmt(exp(-2), 3)

    require(p['summary'] == {'tests':19, 'passed':16, 'failed':['N5b', 'N5c', 'D1b']},
            'companion verification verdict')
    for key, passed in REGISTERED.items():
        row = p['registered'][key]
        err = row['prediction'] / row['value'] - 1
        require(isclose(err, row['rel_error'], abs_tol=1e-12), f'{key} percentage error')
        require((abs(err) <= row['tol']) == row['passed'] == passed, f'{key} preregistered verdict')
    central = p['registered']['D1b']['value']
    require(isclose(p['registered']['D1b']['prediction'], p['danish_readings'][CAP100]['edges_million']),
            'Danish baseline between archived files')
    numbers['DanishPublished'] = fmt(central, 2)
    for key, label in [(CAP100, 'Capped'), (PAIRS100, 'Pairs'), (TWO_WAY, 'TwoWay')]:
        row = p['danish_readings'][key]
        err = row['edges_million'] / central - 1
        require(isclose(err, row['rel_error_vs_published'], abs_tol=1e-12), f'{label} Danish error')
        numbers['Danish'+label] = fmt(row['edges_million'], 2)
        numbers['Danish'+label+'Error'] = fmt(100 * err, 1)
    for school, label in [('N4', 'Primary'), ('N5', 'Secondary')]:
        row = p['registered'][school+'b']
        numbers[label+'MeanPublished'] = fmt(row['value'], 2)
        numbers[label+'MeanPredicted'] = fmt(row['prediction'], 2)
    for key, label in [('primary_quantiles', 'Primary'), ('secondary_quantiles', 'Secondary')]:
        q = p[key]
        require(len(q['predicted']) == len(q['published']) == 8, f'{label} school quantiles')
        numbers[label+'MaxQuantileError'] = fmt(max(abs(a-b) for a, b in zip(q['predicted'], q['published'])), 0)
    numbers['SecondaryMedianPublished'] = fmt(p['registered']['N5c']['value'], 0)
    numbers['SecondaryMedianPredicted'] = fmt(p['registered']['N5c']['prediction'], 0)

    # Hand calculation from docs/excess-closure.md. Ego has two alters in two layers each.
    denominator, pure = 2 * 2, 2
    c_pure = pure / denominator
    variants = tuple((triangles / denominator - c_pure) / (1 - c_pure) for triangles in (1, 2**3, denominator))
    require(variants == (-0.5, 3.0, 1.0), 'documented multiplex counterexample')
    for label, value in zip(('Paper', 'Library', 'TiePair'), variants):
        numbers['Closure'+label] = fmt(value, 1)
    return numbers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if the tracked macros differ')
    args = parser.parse_args()
    output = PAPER / 'numbers.tex'
    generated = '% Generated by src/check_numbers.py.\n' + ''.join(
        f'\\newcommand{{\\{key}}}{{{value}}}\n' for key, value in derive().items())
    if args.check:
        require(output.read_text() == generated, 'numbers.tex differs from stored evidence')
    else:
        output.write_text(generated)
