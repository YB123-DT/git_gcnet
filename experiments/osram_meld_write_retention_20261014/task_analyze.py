"""Paired task/state summary; diagnostic cohorts are NOT full-test scores."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _write


class Accumulator:
    def __init__(self):
        self.confusion = np.zeros((7, 7), dtype=np.int64)
        self.normal = np.zeros((7, 7), dtype=np.int64)
        self.count = self.corrections = self.harms = 0
        self.ce = self.normal_ce = self.delta_ce = self.association_delta = 0.
        self.drift = self.error = self.after_error = 0.
        self.sum_x2 = self.sum_y2 = self.sum_xy = 0.
        self.sign_table = np.zeros((3, 3), dtype=np.int64)

    def add(self, row):
        y, p, p0 = row['label'], row['prediction'], row['normal_prediction']
        self.confusion[y, p] += 1
        self.normal[y, p0] += 1
        self.count += 1
        self.corrections += row['correction']
        self.harms += row['harm']
        self.ce += row['ce']
        self.normal_ce += row['normal_ce']
        self.delta_ce += row['delta_ce']
        x, v = row['association_error_change'], row['delta_ce']
        self.association_delta += x
        self.sum_x2 += x * x
        self.sum_y2 += v * v
        self.sum_xy += x * v
        self.drift += row['normal_association_drift']
        self.error += row['source_relative_error']
        self.after_error += row['write_after_relative_error']
        # Fixed numerical tolerance, not a selected outcome threshold.
        def category(value):
            return 0 if value < -1e-7 else (2 if value > 1e-7 else 1)
        self.sign_table[category(x), category(v)] += 1

    @staticmethod
    def metrics(c):
        true_count = c.sum(1)
        predicted_count = c.sum(0)
        correct = np.diag(c).astype(float)
        recall = np.divide(correct, true_count, out=np.zeros(7), where=true_count > 0)
        f1 = np.divide(2 * correct, true_count + predicted_count, out=np.zeros(7),
                       where=true_count + predicted_count > 0)
        return dict(acc=100 * correct.sum() / max(c.sum(), 1),
                    ua=100 * recall.mean(), wf1=100 * (f1 * true_count).sum() / max(c.sum(), 1),
                    class_recall_percent=(100 * recall).tolist(), class_support=true_count.tolist())

    def summary(self):
        n = self.count
        xmean, ymean = self.association_delta / n, self.delta_ce / n
        xvar = max(self.sum_x2 / n - xmean ** 2, 0.)
        yvar = max(self.sum_y2 / n - ymean ** 2, 0.)
        corr = ((self.sum_xy / n - xmean * ymean) / np.sqrt(xvar * yvar)
                if xvar > 1e-16 and yvar > 1e-16 else None)
        metrics = self.metrics(self.confusion)
        normal = self.metrics(self.normal)
        return dict(N=n, **metrics, normal_metrics=normal,
                    delta_acc=metrics['acc'] - normal['acc'], delta_ua=metrics['ua'] - normal['ua'],
                    delta_wf1=metrics['wf1'] - normal['wf1'], corrections=self.corrections, harms=self.harms,
                    mean_ce=self.ce / n, mean_normal_ce=self.normal_ce / n, mean_delta_ce=ymean,
                    mean_association_error_change=xmean, association_vs_task_pearson=corr,
                    mean_normal_association_drift=self.drift / n,
                    mean_source_relative_error=self.error / n, mean_write_after_relative_error=self.after_error / n,
                    paired_direction_counts=self.sign_table.tolist(),
                    confusion=self.confusion.tolist())


def aggregate(files):
    groups = {}
    seen_targets = {}
    rows = 0
    for filename in files:
        with gzip.open(filename, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                rows += 1
                rate, distance, branch = row['rate'], row['distance'], row['branch']
                pattern = ''.join(m for m, active in zip('ATV', row['availability']) if active)
                degraded = 'degraded' if row['normal_association_degraded'] else 'not_degraded'
                for availability in ('ALL', pattern):
                    for degradation in ('ALL', degraded):
                        key = ('source_modality_paired', rate, distance, row['modality'], availability, branch, degradation)
                        groups.setdefault(key, Accumulator()).add(row)
                # At fixed distance each target has a single source time. Its
                # branch state is shared by all observed source modalities.
                target_key = (rate, distance, str(row['conversation_id']), row['target_t'], branch)
                signature = (row['source_t'], row['label'], row['prediction'], tuple(row['logits']))
                if target_key in seen_targets:
                    if signature != seen_targets[target_key]:
                        raise ValueError('Modality duplicates disagree on paired target logits')
                    continue
                seen_targets[target_key] = signature
                for availability in ('ALL', pattern):
                    key = ('unique_target', rate, distance, 'ALL', availability, branch, 'ALL')
                    groups.setdefault(key, Accumulator()).add(row)
    result = []
    names = ('population', 'rate', 'distance', 'modality', 'availability', 'branch', 'normal_association_group')
    for key, accumulator in sorted(groups.items()):
        summary = accumulator.summary()
        # A unique target can have several source-modal association probes.
        # First-modality selection must not be reported as association evidence.
        if key[0] == 'unique_target':
            for field in ('mean_association_error_change', 'association_vs_task_pearson',
                          'mean_normal_association_drift', 'mean_source_relative_error',
                          'mean_write_after_relative_error', 'paired_direction_counts'):
                summary.pop(field)
        result.append({**dict(zip(names, key)), **summary})
    return dict(raw_rows=rows, direction_categories=['improved', 'numerically_unchanged', 'worsened'],
                direction_axes=['association relative-error change', 'target CE change'], groups=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_summary = json.loads((args.input / 'SUMMARY.json').read_text())
    if source_summary['status'] not in ('complete', 'partial_smoke', 'partial_rates'):
        raise ValueError('Cannot summarize unfinished/failed replay')
    files = [args.input / rate['rows_file'] for rate in source_summary['rates'].values()]
    result = aggregate(files)
    result.update(protocol='INTERNAL DIAGNOSTIC ONLY', training=False, source_summary=source_summary,
                  warning='Cohort/source-target-conditioned scores, not full-test or deployment scores. Distance1 is sanity-only.',
                  class_order='labels 0..6 in unchanged MELD pickle label order',
                  ua_definition='fixed seven-class macro recall; absent classes contribute zero')
    args.output.mkdir(parents=True, exist_ok=False)
    _write(args.output / 'TASK_AGGREGATE.json', result)
    fields = ['population', 'rate', 'distance', 'modality', 'availability', 'branch',
              'normal_association_group', 'N', 'acc', 'ua', 'wf1', 'delta_acc', 'delta_ua',
              'delta_wf1', 'corrections', 'harms', 'mean_ce', 'mean_normal_ce', 'mean_delta_ce',
              'mean_association_error_change', 'association_vs_task_pearson', 'mean_normal_association_drift']
    with (args.output / 'TASK_AGGREGATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['groups'])
    lines = ['# Fixed-query four-state task diagnostic', '', 'INTERNAL DIAGNOSTIC ONLY', '',
             'All scores below refer to paired eligible target cohorts, not the complete test set.',
             'Primary task table deduplicates source modality at fixed distance.', '',
             '| Rate | Distance | Branch | N | ACC | UA | W-F1 | ΔCE | Corrections | Harms |',
             '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['groups']:
        if row['population'] == 'unique_target' and row['availability'] == 'ALL' and row['distance'] != 1:
            lines.append(f"| {row['rate']:.1f} | {row['distance']} | {row['branch']} | {row['N']} | "
                         f"{row['acc']:.3f} | {row['ua']:.3f} | {row['wf1']:.3f} | "
                         f"{row['mean_delta_ce']:.6f} | {row['corrections']} | {row['harms']} |")
    lines += ['', '## Source Text association vs task', '',
              'CSV/JSON provide per-rate, distance, current availability and normal-association degradation strata.',
              'Association/task direction matrices compare each alternative with Normal on the same source-target pair.',
              'Rows are improvement / numerical equality / worsening in source-key error; columns use target CE.',
              'Positive ΔCE is task harm. Positive association error change is poorer source association.',
              'Normal degradation is measured against that same source immediately after its real joint write.',
              'A correlation or decoder association is not proof of causal sentiment semantics or Query failure.',
              'No cross-distance overall score is shown: the same target can occur at several distances.', '']
    (args.output / 'RESULT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
