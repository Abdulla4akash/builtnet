"""Render all manuscript figures from saved aggregates and bundled theory tables."""
from math import exp
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from check_numbers import CAP100, PAIRS100, TWO_WAY, derive
from evidence import CARDS, LAYERS, PAPER, REPO, json_read

# Okabe--Ito colours, with marker shapes or line styles for parallel series.
BLUE, VERMILION, GREEN, PURPLE, ORANGE, SKY, BLACK = (
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#20252B')
CARD_COLOURS = (BLUE, SKY, VERMILION, GREEN)
LAY_COLOURS = (BLUE, GREEN, ORANGE, VERMILION, PURPLE)
MARKERS = ('o', 's', '^', 'D', 'v')
LABELS = ('Dutch 2018', 'Dutch yearly', 'Danish hard cap', 'Swedish')
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': '#616873', 'text.color': BLACK,
    'xtick.color': BLACK, 'ytick.color': BLACK, 'grid.color': '#e1e5e9',
    'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42,
})


def figure(y, x=None, **kwargs):
    fig, ax = plt.subplots(figsize=(6.15, 3.05), layout='constrained', **kwargs)
    ax.set_ylabel(y)
    if x:
        ax.set_xlabel(x)
    ax.grid(axis='y')
    return fig, ax


def save(fig, name):
    for ext in ('pdf', 'png'):
        meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None
        fig.savefig(PAPER / 'figures' / f'{name}.{ext}', dpi=180, metadata=meta)
    plt.close(fig)


def example(s):
    small, main = (s['scales'][key] for key in ('small', 'main'))
    for field, axis, name in [
        ('mean_degree', 'Mean union degree', 'example-degree'),
        ('mean_clustering', 'Mean local clustering (defined egos)', 'example-clustering'),
    ]:
        fig, ax = figure(axis)
        x = np.arange(len(CARDS))
        ax.bar(x - .19, [small['cards'][c][field] for c in CARDS], .38, color=BLUE, label='Small')
        ax.bar(x + .19, [main['cards'][c][field] for c in CARDS], .38, color=GREEN, label='Main', hatch='//')
        ax.set_xticks(x, LABELS)
        ax.legend(ncol=2, loc='upper right')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.17)
        save(fig, name)

    fig, ax = figure('Undirected edges in the main example')
    x = np.arange(len(CARDS))
    for i, (layer, colour) in enumerate(zip(LAYERS, LAY_COLOURS)):
        heights = [main['cards'][c]['layers'][layer] for c in CARDS]
        ax.bar(x + (i - 2) * .15, heights, .145, color=colour, label=layer.title(),
               edgecolor='white', linewidth=.3)
    ax.set_xticks(x, LABELS)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(6, 6))
    ax.legend(ncol=3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    save(fig, 'example-layers')

    fig, ax = figure('Extra layer incidences per 100 union edges')
    for i, (card, colour) in enumerate(zip(CARDS, CARD_COLOURS)):
        values = []
        for row in (small, main):
            c = row['cards'][card]
            values.append(100 * (sum(c['layers'].values()) / c['unique_edges'] - 1))
        ax.plot((0, 1), values, color=colour, marker=MARKERS[i], label=LABELS[i], lw=1.6)
    ax.set_xlim(-.1, 1.1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.set_xticks((0, 1), ('Small', 'Main'))
    ax.legend(ncol=2)
    save(fig, 'example-overlap')


def timing(s):
    scales = [s['scales'][key] for key in ('small', 'main')]
    x = np.arange(2)
    labels = [f"{r['people']:,} people" for r in scales]
    for stage, name, ylabel in [(0, 'timing-build', 'Build wall time (s)'),
                                 (1, 'timing-descriptors', 'Descriptor wall time (s)')]:
        fig, ax = figure(ylabel)
        for i, (key, label, colour) in enumerate(zip(CARDS, LABELS, CARD_COLOURS)):
            samples = [[pair[stage] for pair in row['seconds'][key]] for row in scales]
            xx = x + (i - 1.5) * .075
            med = [median(y) for y in samples]
            ax.errorbar(xx, med, yerr=[[m-min(v) for m, v in zip(med,samples)],
                                      [max(v)-m for m, v in zip(med,samples)]],
                        color=colour, marker=MARKERS[i], lw=1.5, capsize=3, label=label)
        ax.set_xlim(-.29, 1.29)
        ax.set_yscale('log')
        ax.set_xticks(x, labels)
        ax.legend(ncol=2, loc='upper left')
        save(fig, name)
    fig, ax = figure('Generation wall time (s)')
    for i, row in enumerate(scales):
        y = row['generation_seconds']
        ax.scatter(i + np.linspace(-.07, .07, len(y)), y, color=BLUE, s=30, zorder=3)
        ax.hlines(median(y), i-.17, i+.17, color=VERMILION, lw=2, zorder=2)
    ax.set_xlim(-.4, 1.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.set_xticks(x, labels)
    save(fig, 'timing-generation')


def groups(t):
    r = t['group']
    random = [row for row in r if row['select'] == 'random']
    fig, ax = figure(r'Archived mean degree / $k$', r'Group size relative to nominations, $(n-1)/k$')
    for i, cap in enumerate(sorted({row['k'] for row in random})):
        z = [row for row in random if row['k'] == cap]
        ax.scatter([(row['n']-1)/cap for row in z], [row['degree']/cap for row in z],
                   color=LAY_COLOURS[i], marker=MARKERS[i], s=26, label=f'$k={cap}$')
    gx = np.geomspace(1, 150, 200)
    ax.plot(gx, 2 - 1/gx, color=BLACK, lw=1.2, zorder=0, label='Exact expectation')
    ax.set_xscale('log')
    ax.legend(ncol=3, loc='lower right')
    save(fig, 'group-random-degree')

    fig, ax = figure('Archived minus approximate clustering', r'Group size relative to nominations, $(n-1)/k$')
    for i, cap in enumerate(sorted({row['k'] for row in random})):
        z = [row for row in random if row['k'] == cap]
        x = np.array([(row['n']-1)/cap for row in z])
        ax.errorbar(x, [row['clustering'] for row in z] - (1 - (1-1/x)**2),
                    yerr=[2*row['clustering_se'] for row in z],
                    color=LAY_COLOURS[i], marker=MARKERS[i], ms=4, lw=1, capsize=2, label=f'$k={cap}$')
    ax.axhline(0, color=BLACK, lw=1, ls=':')
    ax.set_xscale('log')
    ax.legend(ncol=2, loc='lower right')
    save(fig, 'group-random-clustering')

    groupings = (('nearest','square','Nearest, uniform square',BLUE),
                 ('nearest','gauss','Nearest, Gaussian',GREEN),
                 ('random','-','Random nominations',VERMILION),
                 ('random_capped','-','Sequential hard cap',PURPLE))
    for field, ylabel, name in [('degree','Mean undirected degree','group-spatial-degree'),
                               ('clustering','Mean within-layer clustering','group-spatial-clustering')]:
        fig, ax = figure(ylabel, 'Members in one group')
        for i, (rule, shape, label, colour) in enumerate(groupings):
            rr = sorted((row for row in r if (row['select'], row['shape'], row['k']) == (rule, shape, 100)),
                        key=lambda row: row['n'])
            ax.plot([row['n'] for row in rr], [row[field] for row in rr], color=colour,
                    marker=MARKERS[i], ms=4, ls=('--' if i == 1 else '-'), lw=1.5, label=label)
        ax.set_xscale('log')
        ax.legend(loc='best', ncol=2, fontsize=9)
        save(fig, name)


def neighbours(t):
    neighbours = [sorted((r for r in t['neighbour'] if (r['radius_m'] is not None) == radius),
                         key=lambda r: r['mu']) for radius in (True, False)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.30, 2.70), layout='constrained')
    for ax, field, y in [(left, 'linked', 'Mean linked households'), (right, 'none', 'Share without a neighbour')]:
        for rows, colour, marker, label in zip(neighbours, (BLUE, GREEN), MARKERS, ('50 m', 'No radius')):
            ax.plot([r['mu'] for r in rows], [r[field] for r in rows], marker=marker, color=colour, lw=1.3,
                    ms=4, label=label)
        if field == 'none':
            x = np.geomspace(.25, 50, 250)
            ax.plot(x, [exp(-mu) for mu in x], color=BLACK, ls='--', lw=1, label=r'Poisson $e^{-\mu}$')
        ax.set_xscale('log')
        ax.set_xlabel(r'Expected count within 50 m, $\mu$')
        ax.set_ylabel(y)
        ax.grid(axis='y')
    right.legend(fontsize=9)
    save(fig, 'neighbour-density')

    fig, ax = figure('Mean local clustering', r'Expected count within 50 m, $\mu$')
    for rows, colour, marker, label in zip(neighbours, (BLUE, GREEN), MARKERS, ('50 m', 'No radius')):
        ax.plot([r['mu'] for r in rows], [r['clustering'] for r in rows], marker=marker, color=colour,
                lw=1.3, ms=4, label=label)
    ax.set_xscale('log')
    ax.set_ylim(.50, .63)
    ax.legend()
    save(fig, 'neighbour-clustering')


def ambiguity(p, numbers):
    fig, ax = figure('Undirected colleague edges (millions)')
    rows = [p['danish_readings'][k]['edges_million'] for k in (CAP100, PAIRS100, TWO_WAY)]
    ax.bar(range(3), rows, color=(VERMILION, ORANGE, BLUE), width=.55)
    ax.axhline(p['registered']['D1b']['value'], color=BLACK, ls='--', label='Published aggregate')
    ax.set_xticks(range(3), ('Hard cap', 'Uniform pairs', 'Two-way nominations'))
    ax.set_ylim(0, 145)
    ax.legend()
    save(fig, 'danish-aggregate')

    fig, ax = figure('Predicted out-degree quantile', 'Published out-degree quantile')
    for key, label, colour, marker in [
        ('primary_quantiles','Primary',BLUE,'o'), ('secondary_quantiles','Secondary',VERMILION,'s'),
    ]:
        q = p[key]
        ax.scatter(q['published'], q['predicted'], label=label, marker=marker, color=colour, s=35)
    ax.axline((0, 0), slope=1, color=BLACK, ls=':', label='Equality')
    ax.set_xscale('linear')
    ax.set_ylim(0, 480)
    ax.set_xlim(0, 300)
    ax.legend()
    save(fig, 'school-quantiles')

    fig, ax = figure('Excess closure in the two-layer triangle')
    ax.axhspan(0, 1, color=GREEN, alpha=.12, label='Unit interval')
    ax.axhline(0, color=BLACK, lw=.8)
    ax.bar(range(3), [float(numbers['Closure'+lab]) for lab in ('Paper','Library','TiePair')],
           color=(BLUE, VERMILION, GREEN), width=.5)
    ax.set_xticks(range(3), ('Distinct triplets', 'Library weighting', 'Tie pairs'))
    ax.legend()
    save(fig, 'closure-counterexample')


def main():
    values = derive()
    model = json_read(REPO / 'src' / 'builtnet' / 'data' / 'theory_tables.json')
    observations = json_read(PAPER / 'src' / 'software-summary.json')
    predictions = json_read(PAPER / 'src' / 'source-summary.json')
    example(observations)
    timing(observations)
    groups(model)
    neighbours(model)
    ambiguity(predictions, values)


if __name__ == '__main__':
    main()
