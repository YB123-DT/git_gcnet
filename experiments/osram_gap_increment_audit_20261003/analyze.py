"""Offline, descriptive Base-to-Full Gap increment audit; never fits a gate."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
CATEGORIES = ('rescue', 'harm', 'both_correct', 'both_wrong')
SCORES = ('local_wf1', 'base_wf1', 'full_wf1', 'gap_wf1_delta', 'base_wf1_delta', 'delta_gap_mean')
OBS_SCORES = ('rescue_mean', 'harm_mean', 'difference_rescue_minus_harm', 'auc_rescue_positive')
OBS_COUNTS = ('rescue_n', 'harm_n', 'rescue_missing', 'harm_missing')


def number(value):
    if value in ('', None):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def prepare_rows(rows):
    result, keys = [], set()
    for original in rows:
        r = dict(original)
        r['seed'], r['rate'] = int(r['seed']), float(r['rate'])
        key = r['seed'], r['rate'], r['utterance_id']
        if key in keys:
            raise ValueError(f'Duplicate sample key {key}')
        keys.add(key)
        if r['availability'] not in PATTERNS:
            raise ValueError('Invalid availability')
        for name in ('label', 'pred_local', 'pred_base', 'pred_full'):
            r[name] = number(r[name])
            if r[name] is None:
                raise ValueError(f'Nonfinite {name}')
        y, pb, pf = r['label'], r['pred_base'], r['pred_full']
        delta = (y - pb) ** 2 - (y - pf) ** 2
        if 'delta_gap' in r and not math.isclose(float(r['delta_gap']), delta, abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError('Saved delta_gap differs from predictions')
        r['delta_gap'] = delta
        bc, fc = (pb > 0) == (y > 0), (pf > 0) == (y > 0)
        category = ('both_correct' if bc else 'rescue') if fc else ('harm' if bc else 'both_wrong')
        if y == 0:
            category = 'neutral'
        if 'polarity_category' in r and r['polarity_category'].replace('neutral_excluded', 'neutral') != category:
            raise ValueError('Saved polarity category differs from predictions')
        r['polarity_category'] = category
        for name in r:
            if name.startswith('obs_'):
                r[name] = number(r[name])
        result.append(r)
    return result


def wf1(rows, prediction):
    if not rows:
        return None
    weighted = 0.
    for label in (False, True):
        support = sum((r['label'] > 0) == label for r in rows)
        tp = sum((r['label'] > 0) == label and (r[prediction] > 0) == label for r in rows)
        fp = sum((r['label'] > 0) != label and (r[prediction] > 0) == label for r in rows)
        fn = sum((r['label'] > 0) == label and (r[prediction] > 0) != label for r in rows)
        weighted += support * (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return 100 * weighted / len(rows)


def score_rows(rows):
    valid = [r for r in rows if r['label'] != 0]
    out = dict(n=len(valid), all_utterances=len(rows))
    out.update({cat: sum(r['polarity_category'] == cat for r in valid) for cat in CATEGORIES})
    for name in ('local', 'base', 'full'):
        out[name + '_wf1'] = wf1(valid, 'pred_' + name)
    out['gap_wf1_delta'] = out['full_wf1'] - out['base_wf1'] if valid else None
    out['base_wf1_delta'] = out['base_wf1'] - out['local_wf1'] if valid else None
    out['delta_gap_mean'] = statistics.mean(r['delta_gap'] for r in valid) if valid else None
    return out


def observable_rows(rows, names):
    result = []
    for name in names:
        groups = {cat: [r.get(name) for r in rows if r['polarity_category'] == cat] for cat in ('rescue', 'harm')}
        valid = {cat: [v for v in values if v is not None] for cat, values in groups.items()}
        out = dict(observable=name)
        for cat in groups:
            out[cat + '_n'] = len(valid[cat])
            out[cat + '_missing'] = len(groups[cat]) - len(valid[cat])
            out[cat + '_mean'] = statistics.mean(valid[cat]) if valid[cat] else None
        rescue, harm = valid['rescue'], valid['harm']
        out['difference_rescue_minus_harm'] = out['rescue_mean'] - out['harm_mean'] if rescue and harm else None
        # Exact rank AUROC, ties receive half credit. Direction is not optimized.
        out['auc_rescue_positive'] = (sum((r > h) + .5 * (r == h) for r in rescue for h in harm)
                                     / (len(rescue) * len(harm))) if rescue and harm else None
        result.append(out)
    return result


def macro(rows, scores, counts):
    out = {name: sum(r.get(name, 0) for r in rows) for name in counts}
    for name in scores:
        valid = [r for r in rows if r.get(name) is not None]
        # Equal rates within each seed, then equal seeds. No pooled-rate AUROC.
        seeds = sorted({r['seed'] for r in valid})
        out[name] = statistics.mean(statistics.mean(r[name] for r in valid if r['seed'] == seed) for seed in seeds) if seeds else None
        out[name + '_nonempty_cells'] = len(valid)
    return out


def analyze(rows):
    names = sorted({name for r in rows for name in r if name.startswith('obs_')})
    per_rate, patterns, observables = [], [], []
    for seed, rate in sorted({(r['seed'], r['rate']) for r in rows}):
        current = [r for r in rows if r['seed'] == seed and r['rate'] == rate]
        for pattern in ('ALL',) + PATTERNS:
            chosen = current if pattern == 'ALL' else [r for r in current if r['availability'] == pattern]
            meta = dict(seed=seed, rate=rate, availability=pattern)
            scored = dict(meta, **score_rows(chosen))
            (per_rate if pattern == 'ALL' else patterns).append(scored)
            observables.extend(dict(meta, **r) for r in observable_rows(chosen, names))
    counts = ('n', 'all_utterances') + CATEGORIES
    summary = dict(overall=macro(per_rate, SCORES, counts),
                   high_missing=macro([r for r in per_rate if r['rate'] >= .5], SCORES, counts))
    pattern_macro = [dict(availability=p, **macro([r for r in patterns if r['availability'] == p], SCORES, counts)) for p in PATTERNS]
    obs_macro = [dict(availability=p, observable=name, **macro(
        [r for r in observables if r['availability'] == p and r['observable'] == name], OBS_SCORES, OBS_COUNTS))
        for p in ('ALL',) + PATTERNS for name in names]
    return dict(summary=summary, per_rate=per_rate, availability_per_rate=patterns,
                availability_macro=pattern_macro, observables_per_rate=observables, observables_macro=obs_macro)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return 'NA' if value is None else f'{value:.3f}'


def report(data):
    text = ['# Same-checkpoint Base / Gap intervention', '', 'INTERNAL DIAGNOSTIC ONLY', '',
            'Original cfg84 Full checkpoints; no training. P_L masks Base and Gap only at the classification input; '
            'P_B retains Base and masks Gap; P_F retains both. Memory/query/write/Local/head remain fixed.',
            'W-F1 (%) uses label != 0 and prediction > 0. Counts are rate exposures, not independent utterances. '
            'Each seed/rate is evaluated first; macro means weight nonempty rates equally within each seed, then seeds equally.',
            'Per-rate BEST checkpoints are Test-oracle internal checkpoints, not formal validation-selected paper results.', '',
            '|Rate|Local|Base|Full|Full−Base|Gap rescue|Gap harm|N|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    entries = [(f"{r['seed']} / {r['rate']:.1f}", r) for r in data['per_rate']]
    entries += [('8-rate mean', data['summary']['overall']), ('High missing', data['summary']['high_missing'])]
    for name, r in entries:
        text.append(f"|{name}|" + '|'.join(fmt(r[k]) for k in ('local_wf1', 'base_wf1', 'full_wf1', 'gap_wf1_delta')) + f"|{r['rescue']}|{r['harm']}|{r['n']}|")
    text += ['', '## Observable comparison (equal-rate macro)', '',
             '|Observable|Rescue mean|Harm mean|Paired-rate mean difference|Raw AUROC|AUC cells|Valid rescue / harm|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in data['observables_macro']:
        if r['availability'] == 'ALL':
            text.append(f"|{r['observable']}|" + '|'.join(fmt(r[k]) for k in OBS_SCORES) + f"|{r['auc_rescue_positive_nonempty_cells']}|{r['rescue_n']} / {r['harm_n']}|")
    text += ['', 'AUROC treats rescue as positive and larger raw observable as a higher score. Below 0.5 indicates inverse '
             'direction, not absence of signal. No sign flipping, threshold selection, learned gate, significance test, or routing is performed.',
             'AUROC is undefined when either group is absent. Undefined zero-vector cosines and ratios are excluded with explicit missing counts. '
             'Rescue/harm mean differences are averaged only over rate cells containing both groups; separately averaged means may use different cells.',
             'Checkpoint weights differ across rates: raw norms are not pooled as primary evidence. See observables_per_rate.csv and '
             'availability-stratified rows in observables_macro.csv to inspect consistency and confounding by current modality support.',
             'Gold labels define retrospective rescue/harm groups only. These descriptive associations do not establish deployable discrimination, '
             'emotion shift, reliability, or causal mechanism. No head/query intervention was run in this first-stage report.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.input.open(newline='') as stream:
        rows = prepare_rows(list(csv.DictReader(stream)))
    for seed in {r['seed'] for r in rows}:
        if sorted({r['rate'] for r in rows if r['seed'] == seed}) != [i / 10 for i in range(8)]:
            raise ValueError('Expected all eight rates for each seed')
        expected = None
        for rate in [i / 10 for i in range(8)]:
            current = {r['utterance_id']: r['label'] for r in rows if r['seed'] == seed and r['rate'] == rate}
            if expected is not None and current != expected:
                raise ValueError('Sample IDs or labels differ across rates')
            expected = current
    data = analyze(rows)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Input changed during analysis')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, values in data.items():
        if isinstance(values, list):
            write_csv(args.output / f'{name}.csv', values)
    write_csv(args.output / 'utterances.csv', rows)
    (args.output / 'summary.json').write_text(json.dumps(data['summary'], indent=2, allow_nan=False) + '\n')
    (args.output / 'provenance.json').write_text(json.dumps(dict(input=str(args.input.resolve()), sha256=digest), indent=2) + '\n')
    (args.output / 'RESULT.md').write_text(report(data))
    print(json.dumps(data['summary'], indent=2))


if __name__ == '__main__':
    main()
