"""Plot development-set scores and unique-GT diagnostics, without image data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
result = Path('/workspace/artifacts/validation_results_20261027T112905Z_947c4b1f')
summary = json.loads((result/'summary.json').read_text())
diagnostics = json.loads((root/'docs/evidence/validation_diagnostics.json').read_text())
baseline, preserve = diagnostics['runs']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = {'union_cc': '#1961a9', 'preserve': '#dc7a1d'}
for mode in ['union_cc', 'preserve']:
    rows = [r for r in summary['comparisons'] if r['postprocess'] == mode]
    axes[0, 0].plot([r['confidence'] for r in rows], [r['PQ'] for r in rows],
                    '-o', color=colors[mode], label=mode)
axes[0, 0].set(title='Global PQ by confidence (preserve may overlap)',
               xlabel='Confidence threshold', ylabel='Development PQ')
axes[0, 0].legend(frameon=False)
bins = baseline['unmatched_gt_best_iou_bins']
labels = ['No overlap', '(0, 0.1)', '[0.1, 0.3)', '[0.3, 0.4)', '[0.4, 0.5]']
counts = list(bins.values())[:5]
axes[0, 1].bar(labels, counts, color=colors['union_cc'])
axes[0, 1].set(title='Baseline missed GT: best IoU to any prediction', ylabel='GT instances')
axes[0, 1].tick_params(axis='x', rotation=20)
for i, value in enumerate(counts):
    axes[0, 1].text(i, value+2, str(value), ha='center')
x = np.arange(4)
for offset, run, label, color in [(-.18, baseline, 'union_cc 0.25', colors['union_cc']),
                                 (.18, preserve, 'preserve 0.15', colors['preserve'])]:
    recall = [run['recall_by_area_quartile'][str(i)]['unique_gt_recall'] for i in range(1, 5)]
    axes[1, 0].bar(x+offset, recall, width=.36, label=label, color=color)
axes[1, 0].set(xticks=x, xticklabels=['Smallest 25%', 'Q2', 'Q3', 'Largest 25%'],
               ylim=(0, 1), title='Unique-GT recall by area (177 GT per bin)', ylabel='Recall')
axes[1, 0].legend(frameon=False)
rows = [r for r in summary['comparisons'] if r['postprocess'] == 'preserve']
extras = [r['TP'] + r['FN'] - 708 for r in rows]
axes[1, 1].bar([str(r['confidence']) for r in rows], extras, color=colors['preserve'])
axes[1, 1].set(title='Preserve: extra TP pairs beyond unique matched GT',
               xlabel='Confidence threshold', ylabel='Additional qualifying pairs')
for i, value in enumerate(extras):
    axes[1, 1].text(i, value+.7, str(value), ha='center')
for ax in axes.ravel():
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('91 development images / 101 annotation groups / 708 GT instances', fontsize=14)
fig.text(.5, .015, 'Same-set tuning diagnostics. Not a leaderboard gain or independent final evaluation.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, .045, 1, .95))
out = root/'docs/evidence/validation_summary.png'
fig.savefig(out, dpi=150)
print(out)
