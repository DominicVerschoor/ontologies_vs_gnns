"""
Figures for the Results chapter, drawn from the saved result CSVs and LP training histories.

Run from the repo root in the `sulo` env:  python results_figures.py
Writes PNGs (300 dpi) to outputs/f1/figures/. Nothing is trained; every value is read from
outputs/f1/. Colors: the first three slots of the dataviz reference palette (validated
all-pairs for three series), one per KG, always with a marker shape as a second encoding.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

OUT = Path('outputs/f1')
FIG = OUT / 'figures'
NP, LP = OUT / 'node_prediction', OUT / 'link_prediction'

KGS = ['adhoc', 'hybrid_sulo', 'pure_sulo']
KG_NAME = {'adhoc': 'Ad-hoc', 'hybrid_sulo': 'Hybrid', 'pure_sulo': 'Pure SULO'}
KG_COLOR = {'adhoc': '#2a78d6', 'hybrid_sulo': '#eb6834', 'pure_sulo': '#1baf7a'}
KG_MARKER = {'adhoc': 'o', 'hybrid_sulo': 's', 'pure_sulo': '^'}
ENCODERS = ['rgcn', 'graphsage']
ENC_NAME = {'rgcn': 'R-GCN', 'graphsage': 'GraphSAGE'}
TARGETS = ['team', 'position', 'circuit']
TARGET_NAME = {'team': 'Team', 'position': 'Position', 'circuit': 'Circuit'}
UNITS = ['sulo_typing', 'precedesRace', 'precedesSeason', 'hasTeamMember', 'rank_reification']
UNIT_NAME = {'sulo_typing': 'SULO typing', 'precedesRace': 'precedesRace',
             'precedesSeason': 'precedesSeason', 'hasTeamMember': 'hasTeamMember',
             'rank_reification': 'rank reification'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
WIDTH = 6.3  # inches, about \textwidth

plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'lines.linewidth': 2,
})


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', facecolor='white')
    plt.close(fig)
    print('wrote', FIG / f'{name}.png')


def kg_legend(fig, ncol=3, y=0.99, extra=()):
    handles = [plt.Line2D([], [], color=KG_COLOR[k], marker=KG_MARKER[k], linestyle='', markersize=6,
                          label=KG_NAME[k]) for k in KGS] + list(extra)
    fig.tight_layout()
    fig.legend(handles=handles, loc='lower center', ncol=ncol + len(extra), bbox_to_anchor=(0.5, y))


def final_seeds(task, enc, kg):
    """Phase-4 test rows (one per final seed) of a main tuning run."""
    d = pd.read_csv(OUT / task / enc / f'{kg}_sequential_tuning_results.csv')
    return d[d['phase'].str.startswith('4')]


def seed_dots(ax, x, values, kg, width=0.22):
    """Seed values as dots with a short horizontal bar at their mean."""
    values = np.asarray(values, dtype=float)
    jitter = np.linspace(-0.06, 0.06, len(values))
    ax.plot(x + jitter, values, linestyle='', marker=KG_MARKER[kg], markersize=4.5,
            color=KG_COLOR[kg], alpha=0.75, markeredgewidth=0)
    ax.plot([x - width / 2, x + width / 2], [values.mean()] * 2, color=KG_COLOR[kg], linewidth=2,
            solid_capstyle='round')


def frequency_line(ax, value, label=True):
    ax.axhline(value, color=INK2, linestyle=(0, (3, 2)), linewidth=1)
    if label:
        ax.annotate('frequency', xy=(1, value), xycoords=('axes fraction', 'data'), xytext=(2, 0),
                    textcoords='offset points', va='center', fontsize=7, color=INK2)


def frequency_mrr():
    b = pd.read_csv(LP / 'baselines.csv', index_col=0)
    return b['frequency MRR'].to_dict()


# --------------------------------------------------------------------------- E1
def e1_nc():
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.6))
    for i, enc in enumerate(ENCODERS):
        for j, kg in enumerate(KGS):
            seed_dots(ax, i + (j - 1) * 0.28, final_seeds('node_prediction', enc, kg)['test_roc_auc'], kg)
    ax.set_xticks(range(len(ENCODERS)), [ENC_NAME[e] for e in ENCODERS])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel('Test ROC-AUC')
    ax.grid(axis='x', visible=False)
    kg_legend(fig)
    save(fig, 'e1_nc_test_roc_auc')


def e1_lp():
    freq = frequency_mrr()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), sharey=True)
    for ax, target in zip(axes, TARGETS):
        for i, enc in enumerate(ENCODERS):
            for j, kg in enumerate(KGS):
                seed_dots(ax, i + (j - 1) * 0.28, final_seeds('link_prediction', enc, kg)[f'test_{target}_mrr'], kg)
        frequency_line(ax, freq[target], label=target == 'circuit')
        ax.set_title(TARGET_NAME[target])
        ax.set_xticks(range(len(ENCODERS)), [ENC_NAME[e] for e in ENCODERS])
        ax.set_xlim(-0.6, 1.6)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Test MRR')
    axes[0].set_ylim(0, 1)
    kg_legend(fig)
    save(fig, 'e1_lp_test_mrr')


# --------------------------------------------------------------------------- E2
def delta_panel(ax, summary, delta_col, std_col):
    """Delta of each unit in both directions, error bar = the variant's seed std."""
    s = summary.set_index('variant')
    y = np.arange(len(UNITS))[::-1]
    for offset, prefix, kg, label in [(0.14, 'hybrid_minus_', 'hybrid_sulo', r'Hybrid $-$ unit ($\Delta^{-}$)'),
                                      (-0.14, 'adhoc_plus_', 'adhoc', r'Ad-hoc $+$ unit ($\Delta^{+}$)')]:
        rows = s.loc[[prefix + u for u in UNITS]]
        ax.errorbar(rows[delta_col], y + offset, xerr=rows[std_col], linestyle='', marker=KG_MARKER[kg],
                    markersize=5, color=KG_COLOR[kg], elinewidth=1, capsize=0, label=label)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [UNIT_NAME[u] for u in UNITS])
    ax.grid(axis='y', visible=False)


def direction_legend(fig, y=0.99):
    handles = [plt.Line2D([], [], color=KG_COLOR['hybrid_sulo'], marker='s', linestyle='', markersize=6,
                          label=r'Hybrid $-$ unit ($\Delta^{-}$)'),
               plt.Line2D([], [], color=KG_COLOR['adhoc'], marker='o', linestyle='', markersize=6,
                          label=r'Ad-hoc $+$ unit ($\Delta^{+}$)')]
    fig.tight_layout()
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, y))


def e2_nc():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True, sharex=True)
    for ax, enc in zip(axes, ENCODERS):
        summary = pd.read_csv(NP / 'ablation' / enc / 'ablation_summary.csv')
        delta_panel(ax, summary, 'delta_test_roc_auc', 'test_roc_auc_std')
        ax.set_title(ENC_NAME[enc])
        ax.set_xlabel(r'$\Delta$ test ROC-AUC')
    direction_legend(fig)
    save(fig, 'e2_nc_delta_test_roc_auc')


def e2_lp():
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.2), sharey=True)
    for r, enc in enumerate(ENCODERS):
        summary = pd.read_csv(LP / 'ablation' / enc / 'ablation_summary.csv')
        for c, target in enumerate(TARGETS):
            ax = axes[r, c]
            delta_panel(ax, summary, f'delta_test_{target}_mrr', f'test_{target}_mrr_std')
            if r == 0:
                ax.set_title(TARGET_NAME[target])
            if r == 1:
                ax.set_xlabel(r'$\Delta$ test MRR')
        axes[r, 0].set_ylabel(ENC_NAME[enc], fontweight='bold')
    for c in range(3):
        lim = max(abs(v) for ax in axes[:, c] for v in ax.get_xlim())
        for ax in axes[:, c]:
            ax.set_xlim(-lim, lim)
    direction_legend(fig)
    save(fig, 'e2_lp_delta_test_mrr')


DIRECT_RANK = ['adhoc', 'hybrid_minus_rank_reification'] + [f'adhoc_plus_{u}' for u in UNITS[:-1]]
REIFIED_RANK = ['hybrid_sulo', 'adhoc_plus_rank_reification'] + [f'hybrid_minus_{u}' for u in UNITS[:-1]]
RANK_COLOR = {'direct': KG_COLOR['adhoc'], 'reified': KG_COLOR['hybrid_sulo']}
RANK_MARKER = {'direct': 'o', 'reified': 's'}


def variant_label(v):
    if v in ('adhoc', 'hybrid_sulo'):
        return KG_NAME[v]
    base, unit = ('Hybrid $-$ ', v.removeprefix('hybrid_minus_')) if v.startswith('hybrid_minus_') \
        else ('Ad-hoc $+$ ', v.removeprefix('adhoc_plus_'))
    return base + UNIT_NAME[unit]


def rank_legend(fig, y=0.99):
    handles = [plt.Line2D([], [], color=RANK_COLOR[k], marker=RANK_MARKER[k], linestyle='', markersize=6,
                          label=f'{k} rank') for k in ('direct', 'reified')]
    fig.tight_layout()
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, y))


def e2_lp_position_seeds():
    """Every R-GCN seed's test Position MRR, by variant, grouped by how the rank is stored."""
    d = pd.read_csv(LP / 'ablation' / 'rgcn' / 'ablation_results.csv')
    order = DIRECT_RANK + REIFIED_RANK
    fig, ax = plt.subplots(figsize=(WIDTH * 0.75, 3.4))
    for i, v in enumerate(order):
        kind = 'direct' if v in DIRECT_RANK else 'reified'
        vals = d.loc[d['variant'] == v, 'test_position_mrr'].to_numpy()
        ax.plot(vals, np.full(len(vals), len(order) - 1 - i), linestyle='', marker=RANK_MARKER[kind],
                markersize=5, color=RANK_COLOR[kind], alpha=0.75, markeredgewidth=0)
    ax.axhline(len(REIFIED_RANK) - 0.5, color=GRID, linewidth=1)
    frequency_x = frequency_mrr()['position']
    ax.axvline(frequency_x, color=INK2, linestyle=(0, (3, 2)), linewidth=1)
    ax.annotate('frequency', xy=(frequency_x, 1), xycoords=('data', 'axes fraction'), xytext=(2, 2),
                textcoords='offset points', fontsize=7, color=INK2)
    ax.set_yticks(range(len(order))[::-1], [variant_label(v) for v in order])
    ax.set_xlabel('Test MRR, Position target (one dot per seed)')
    ax.grid(axis='y', visible=False)
    rank_legend(fig)
    save(fig, 'e2_lp_rgcn_position_seeds')


# --------------------------------------------------------------------------- training curves
def history(path):
    return torch.load(path, map_location='cpu', weights_only=False)['history']


def curve(ax, values, color, linestyle='-', alpha=0.6, linewidth=1):
    ax.plot(np.arange(len(values)), values, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)


def warm_up(ax):
    ax.axvspan(0, 50, color=GRID, alpha=0.5, linewidth=0)


def lp_curves():
    abl = LP / 'ablation' / 'rgcn' / 'checkpoints'
    dep = LP / 'depth' / 'rgcn' / 'checkpoints'
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharex=True)

    ax = axes[0]
    for v in DIRECT_RANK + REIFIED_RANK:
        kind = 'direct' if v in DIRECT_RANK else 'reified'
        for seed in range(5):
            curve(ax, history(abl / f'{v}_seed{seed}.pt')['val_mrr_position'], RANK_COLOR[kind], alpha=0.45)
    ax.set_title('(a) Position, $K = 3$, E2 variants')
    ax.set_ylabel('Validation MRR')

    ax = axes[1]
    for kg in KGS:
        for seed in range(5):
            curve(ax, history(dep / 'K4' / f'{kg}_seed{seed}.pt')['val_mrr_position'], KG_COLOR[kg], alpha=0.7)
    ax.set_title('(b) Position, $K = 4$')

    ax = axes[2]
    for k, style in [('K3', (0, (3, 2))), ('K4', '-')]:
        for seed in range(5):
            curve(ax, history(dep / k / f'pure_sulo_seed{seed}.pt')['val_mrr_team'], KG_COLOR['pure_sulo'],
                  linestyle=style, alpha=0.8)
    for seed in range(5):
        curve(ax, history(abl / f'adhoc_seed{seed}.pt')['val_mrr_team'], KG_COLOR['adhoc'], alpha=0.5)
    ax.set_title('(c) Team')

    for ax in axes:
        warm_up(ax)
        ax.set_xlabel('Epoch')
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 0.8)
    handles = [plt.Line2D([], [], color=RANK_COLOR['direct'], label='(a) direct rank'),
               plt.Line2D([], [], color=RANK_COLOR['reified'], label='(a) reified rank'),
               plt.Line2D([], [], color=KG_COLOR['adhoc'], label='Ad-hoc'),
               plt.Line2D([], [], color=KG_COLOR['hybrid_sulo'], label='Hybrid'),
               plt.Line2D([], [], color=KG_COLOR['pure_sulo'], label='Pure SULO'),
               plt.Line2D([], [], color=KG_COLOR['pure_sulo'], linestyle=(0, (3, 2)), label='(c) Pure SULO, $K = 3$')]
    fig.tight_layout()
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'lp_rgcn_val_curves')


# --------------------------------------------------------------------------- E3
NC_EVIDENCE_K = {'adhoc': 2, 'hybrid_sulo': 3, 'pure_sulo': 3}


def e3_nc_k_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True)
    for ax, enc in zip(axes, ENCODERS):
        d = pd.read_csv(NP / 'k_sweep' / enc / 'k_sweep_summary.csv')
        for j, kg in enumerate(KGS):
            r = d[d['kg'] == kg].sort_values('K')
            x = r['K'] + (j - 1) * 0.06
            ax.errorbar(x, r['test_roc_auc_mean'], yerr=r['test_roc_auc_std'], color=KG_COLOR[kg],
                        marker=KG_MARKER[kg], markersize=5, elinewidth=1, capsize=0, linewidth=1.5)
            e = r[r['K'] == NC_EVIDENCE_K[kg]]
            ax.plot(e['K'] + (j - 1) * 0.06, e['test_roc_auc_mean'], marker='o', markersize=12,
                    markerfacecolor='none', markeredgecolor=KG_COLOR[kg], linestyle='', markeredgewidth=1.2)
        ax.set_title(ENC_NAME[enc])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel('Layers $K$')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Test ROC-AUC')
    ring = plt.Line2D([], [], marker='o', markersize=10, markerfacecolor='none', markeredgecolor=INK2,
                      linestyle='', label='grid rank first in reach')
    kg_legend(fig, extra=[ring])
    save(fig, 'e3_nc_k_sweep')


def lp_depth_rows(enc):
    """Per seed rows at K = 3 and K = 4 for every KG (Ad-hoc/Hybrid K = 3 are the E2 controls)."""
    abl = pd.read_csv(LP / 'ablation' / enc / 'ablation_results.csv')
    k3 = pd.concat([abl[abl['variant'].isin(['adhoc', 'hybrid_sulo'])],
                    pd.read_csv(LP / 'depth' / enc / 'K3.csv')]).assign(K=3)
    k4 = pd.read_csv(LP / 'depth' / enc / 'K4.csv').assign(K=4)
    return pd.concat([k3, k4])


def lp_evidence_hops():
    """target -> {kg: hops from the query node to the answer-identifying literals}, as computed from the
    schemas by link_prediction_analysis.ipynb (`evidence_hops` over `EVIDENCE_CHAINS`)."""
    t = pd.read_csv(LP / 'analysis' / 'hop_table.csv', index_col=0)
    cols = {'adhoc': 'Ad-hoc', 'hybrid_sulo': 'Hybrid', 'pure_sulo': 'Pure'}
    rows = {'team': "LP team: the driver's literals, from the entry",
            'position': "LP position: a session's rank, from the entry",
            'circuit': "LP circuit: the race's literals, from the race"}
    return {target: {kg: int(t.loc[row, cols[kg]]) for kg in KGS} for target, row in rows.items()}


def e3_lp_depth():
    """Same encoding as e3_nc_k_sweep: mean ± std over the 5 seeds per K, ring where the evidence is first
    in reach (K = hops). Only K = 3 and K = 4 exist for LP (shared Hybrid-winner configuration)."""
    freq = frequency_mrr()
    hops = lp_evidence_hops()
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.2), sharey=True)
    for r, enc in enumerate(ENCODERS):
        d = lp_depth_rows(enc)
        for c, target in enumerate(TARGETS):
            ax = axes[r, c]
            for j, kg in enumerate(KGS):
                s = d[d['variant'] == kg].groupby('K')[f'test_{target}_mrr'].agg(['mean', 'std'])
                x = s.index + (j - 1) * 0.06
                ax.errorbar(x, s['mean'], yerr=s['std'], color=KG_COLOR[kg], marker=KG_MARKER[kg],
                            markersize=5, elinewidth=1, capsize=0, linewidth=1.5)
                if hops[target][kg] in s.index:
                    k = hops[target][kg]
                    ax.plot(k + (j - 1) * 0.06, s.loc[k, 'mean'], marker='o', markersize=12,
                            markerfacecolor='none', markeredgecolor=KG_COLOR[kg], linestyle='',
                            markeredgewidth=1.2)
            frequency_line(ax, freq[target], label=c == 2)
            ax.set_xticks([3, 4], ['$K = 3$', '$K = 4$'])
            ax.set_xlim(2.7, 4.3)
            ax.grid(axis='x', visible=False)
            if r == 0:
                ax.set_title(TARGET_NAME[target])
            else:
                ax.set_xlabel('Layers $K$')
        axes[r, 0].set_ylabel(f'{ENC_NAME[enc]}\nTest MRR')
    axes[0, 0].set_ylim(0, 1)
    ring = plt.Line2D([], [], marker='o', markersize=10, markerfacecolor='none', markeredgecolor=INK2,
                      linestyle='', label='evidence first in reach')
    kg_legend(fig, extra=[ring])
    save(fig, 'e3_lp_depth')


# With vs without a session position: hue AND shape differ, so the two groups stay apart in grayscale and
# for color-blind readers. Violet (categorical slot 7) is not one of the three KG hues; the "without" group
# is a neutral gray because it is the no-evidence baseline.
SESSION_STYLE = {'with': dict(color='#6250d6', marker='o'), 'without': dict(color='#8a8984', marker='^')}


def e3_position_by_session():
    d = pd.read_csv(LP / 'analysis' / 'position_by_session_per_seed.csv')
    runs = list(dict.fromkeys(d['run']))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.85, 2.9))
    for i, run in enumerate(runs):
        y = len(runs) - 1 - i
        rows = d[d['run'] == run]
        ax.plot(rows['with a session position'], np.full(len(rows), y + 0.12), linestyle='', markersize=5,
                alpha=0.85, markeredgewidth=0, **SESSION_STYLE['with'])
        ax.plot(rows['without'], np.full(len(rows), y - 0.12), linestyle='', markersize=5,
                alpha=0.85, markeredgewidth=0, **SESSION_STYLE['without'])
    labels = [r.replace('−', '$-$').replace('Pure,', 'Pure SULO,').replace('K = ', '$K = $')
              .replace('$K = $3', '$K = 3$').replace('$K = $4', '$K = 4$') for r in runs]
    ax.set_yticks(range(len(runs))[::-1], labels)
    ax.set_xlabel('R-GCN test MRR, Position target (one dot per seed)')
    ax.grid(axis='y', visible=False)
    handles = [plt.Line2D([], [], linestyle='', markersize=6, label='entries with a session position (1,114)',
                          **SESSION_STYLE['with']),
               plt.Line2D([], [], linestyle='', markersize=6, label='entries without (1,461)',
                          **SESSION_STYLE['without'])]
    fig.tight_layout()
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'e3_position_by_session')


if __name__ == '__main__':
    FIG.mkdir(parents=True, exist_ok=True)
    e1_nc()
    e1_lp()
    e2_nc()
    e2_lp()
    e2_lp_position_seeds()
    lp_curves()
    e3_nc_k_sweep()
    e3_lp_depth()
    e3_position_by_session()
