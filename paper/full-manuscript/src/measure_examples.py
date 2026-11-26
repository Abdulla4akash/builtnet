"""Time existing README examples and run the release tests on the allocated cores.

Writes aggregate evidence only. This is separate from the manuscript build.
"""
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import scipy

from builtnet import cards, scenarios, world
from builtnet.descriptors import position_descriptors
from builtnet.rulesets import build

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
LOCAL = PAPER / '.local'
CARDS = ('nl-2018', 'nl-series', 'dk', 'se')
SCALES = ('small', 'main')
SEED = 1
REPEATS = 3


def snapshot(net, desc):
    active = net.active
    clustering = desc['clustering'][active]
    finite = np.isfinite(clustering)
    return {
        'active': int(active.sum()),
        'mean_degree': float(desc['degree_total'][active].mean()),
        'readme_clustering': None if not finite.all() else float(clustering.mean()),
        'clustering_defined': int(finite.sum()),
        'mean_clustering': float(clustering[finite].mean()),
        'unique_edges': int(sum(desc['degree_total'][active]) // 2),
        'layers': {name: int(layer.nnz // 2) for name, layer in net.layers.items()},
    }


def time_examples():
    scales = {}
    for scale in SCALES:
        generation = []
        samples = {name: [] for name in CARDS}
        values = {}
        for _ in range(REPEATS):
            start = time.perf_counter()
            w = world.generate(scenarios.base(scale), SEED)
            generation.append(time.perf_counter() - start)
            for name in CARDS:
                card = cards.load(name)
                start = time.perf_counter()
                net = build(w, card, seed=SEED)
                after_build = time.perf_counter()
                desc = position_descriptors(net)
                after_metrics = time.perf_counter()
                samples[name].append([after_build - start, after_metrics - after_build])
                summary = snapshot(net, desc)
                if name in values and summary != values[name]:
                    raise ValueError(f'Example changed across timing repeats: {scale}, {name}')
                values[name] = summary
                del net, desc
        scales[scale] = {'people': w.n, 'households': len(w.household_xy),
                         'generation_seconds': generation, 'cards': values, 'seconds': samples}
        print(f'{scale}: {w.n} people, generation median {statistics.median(generation):.2f}s', flush=True)
    return scales


def run_tests():
    LOCAL.mkdir(exist_ok=True)
    xml = LOCAL / 'tests.xml'
    command = [sys.executable, '-B', '-m', 'pytest', '-o', 'addopts=', '-q', '-rs',
               '-p', 'no:cacheprovider', '--junitxml', str(xml), 'tests']
    start = time.perf_counter()
    result = subprocess.run(command, cwd=REPO, check=False, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    (LOCAL / 'pytest.txt').write_text(result.stdout + result.stderr)
    print(result.stdout, flush=True)
    result.check_returncode()
    suite = ET.parse(xml).find('testsuite')
    summary = {}
    for test in suite.findall('testcase'):
        module = test.get('classname').rsplit('.', 1)[-1]
        row = summary.setdefault(module, {'passed': 0, 'skipped': 0})
        row['skipped' if test.find('skipped') is not None else 'passed'] += 1
    skips = [node.get('message', '') for node in suite.findall('testcase/skipped')]
    return {'modules': summary, 'skip_reasons': skips, 'wall_seconds': elapsed}


def main():
    if len(os.sched_getaffinity(0)) > 2:
        raise SystemExit('Run on the allocated two cores or restrict CPU affinity first.')
    cpu = re.search(r'^model name\s*:\s*(.+)$', Path('/proc/cpuinfo').read_text(), re.M)
    environment = {
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'cpu': cpu[1] if cpu else platform.processor(), 'cpus': len(os.sched_getaffinity(0)),
        'thread_limit': 1,
        'seed': SEED, 'repeats': REPEATS,
        'source_commit': subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', 'README.md', 'src', 'tests'], cwd=REPO, text=True).strip(),
    }
    tests = run_tests()
    scales = time_examples()
    out = {'environment': environment, 'tests': tests, 'scales': scales}
    (PAPER / 'src' / 'software-summary.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
