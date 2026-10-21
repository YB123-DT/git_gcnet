"""Plot recorded MOSI histories only; no checkpoint loading or model execution."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
WINDOWS = ((1, 8), (17, 24), (41, 48), (73, 80), (89, 96), (105, 112), (137, 144))
STYLE = {
    'Flat_1e3': ('Flat: 1e-3 -> 1e-4', '#0072B2', '-'),
    'Flat_3e4': ('Flat: 3e-4 -> 3e-5', '#D55E00', '--'),
    'V3_3e4': ('V3: 3e-4 -> 3e-5', '#009E73', '-.'),
}


def trailing(values, weights=None, width=8):
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    result = np.full_like(values, np.nan)
    for end in range(width, len(values) + 1):
        selection = slice(end - width, end)
        result[end - 1] = np.average(values[selection], weights=weights[selection])
    return result


def main():
    data = json.loads((ROOT / 'curves.json').read_text())
    assert data['training_mask_N_and_update_traces_exact_across_150_epochs']
    columns = data['columns']; summary = {}; flat_rows = []
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), layout='constrained')
    for name, series in data['series'].items():
        rows = [dict(zip(columns, row)) for row in series['rows']]
        assert [row['epoch'] for row in rows] == list(range(1, 151))
        values = {key: np.array([row[key] for row in rows]) for key in columns}
        assert all(np.isfinite(v).all() for v in values.values())
        assert np.all(np.diff(values['cumulative_per_rate_best_wf1']) >= -1e-8)
        assert np.all(values['optimizer_steps'] == 2)
        expected_lr = 1e-3 if name == 'Flat_1e3' else 3e-4
        np.testing.assert_allclose(values['lr'][:100], expected_lr, rtol=1e-6)
        np.testing.assert_allclose(values['lr'][100:], expected_lr / 10, rtol=1e-6)
        label, color, linestyle = STYLE[name]
        for ax, key in zip(axes.flat, ('train_mse', 'test_mean_mse',
                                      'test_mean_wf1', 'cumulative_per_rate_best_wf1')):
            if key != 'cumulative_per_rate_best_wf1':
                ax.plot(values['epoch'], values[key], color=color, alpha=.17, lw=.7)
                smooth = trailing(values[key], values['train_N'] if key == 'train_mse' else None)
            else:
                smooth = values[key]
            ax.plot(values['epoch'], smooth, color=color, ls=linestyle, lw=1.8, label=label)
        windows = []
        for start, stop in WINDOWS:
            selected = rows[start - 1:stop]
            windows.append(dict(start=start, stop=stop,
                train_mse=float(np.average([r['train_mse'] for r in selected],
                                           weights=[r['train_N'] for r in selected])),
                test_mean_mse=float(np.mean([r['test_mean_mse'] for r in selected])),
                test_mean_wf1=float(np.mean([r['test_mean_wf1'] for r in selected]))))
        best = max(rows, key=lambda row: row['test_mean_wf1'])
        summary[name] = dict(windows=windows, best_same_epoch=best,
                             final_per_rate_oracle_wf1=rows[-1]['cumulative_per_rate_best_wf1'])
        flat_rows.extend(dict(model=name, **row) for row in rows)
    titles = ('A  Training MSE (log scale)', 'B  Test MSE: same-checkpoint 8-rate mean (log scale)',
              'C  Test W-F1: same-checkpoint 8-rate mean',
              'D  Cumulative per-rate TEST-best mean (oracle)')
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlabel('Epoch'); ax.set_xlim(1, 150)
        ax.axvline(101, color='.4', ls=':', lw=1.2)
        ax.grid(alpha=.17)
    for ax in axes[0]:
        ax.set_yscale('log'); ax.set_ylabel('MSE')
    for ax in axes[1]:
        ax.set_ylabel('W-F1 (%)'); ax.set_ylim(20, 85)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[1, 1].text(.98, .04, 'Different rates may select different epochs',
                   transform=axes[1, 1].transAxes, ha='right', fontsize=8, color='.3')
    fig.suptitle('MOSI seed66 | original GCNet random missing | INTERNAL TEST-ORACLE DIAGNOSTIC\n'
                 'Faint: raw epochs; bold A-C: trailing 8-epoch mean; dotted: LR reduction at epoch101',
                 fontsize=11)
    for extension in ('svg', 'pdf'):
        fig.savefig(ROOT / ('curves.' + extension))
    plt.close(fig)
    with (ROOT / 'curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model'] + columns)
        writer.writeheader(); writer.writerows(flat_rows)
    (ROOT / 'window_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(rows=len(flat_rows), models=len(summary), epochs_per_model=150,
                         source_traces_equal=True, artifacts=['curves.svg', 'curves.pdf',
                         'curves.csv', 'window_summary.json'])))


if __name__ == '__main__':
    main()
