"""Descriptive classification-input head ablation; no head-routing fitting."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.osram_gap_increment_audit_20261003 import analyze as common

SCORES = ('full_wf1', 'masked_wf1', 'contribution')
COUNTS = ('n', 'all_utterances', 'mask_corrections', 'mask_harms')


def prepare_rows(rows):
    result, keys, reference = [], set(), {}
    for original in rows:
        r = dict(original)
        r['seed'], r['rate'], r['head'] = int(r['seed']), float(r['rate']), int(r['head'])
        if r['intervention'] not in ('base', 'gap') or r['head'] not in range(8):
            raise ValueError('Invalid head intervention')
        if r['availability'] not in common.PATTERNS:
            raise ValueError('Invalid availability')
        for name in ('label', 'pred_full', 'pred_masked'):
            r[name] = common.number(r[name])
            if r[name] is None:
                raise ValueError(f'Nonfinite {name}')
        sample = r['seed'], r['rate'], r['utterance_id']
        key = sample + (r['intervention'], r['head'])
        if key in keys:
            raise ValueError('Duplicate head intervention row')
        keys.add(key)
        ref = (r['label'], r['availability'], r['pred_full'])
        if sample in reference and reference[sample] != ref:
            raise ValueError('Full reference differs between head interventions')
        reference[sample] = ref
        if r['intervention'] == 'gap' and r['availability'] == 'ATV':
            if not math.isclose(r['pred_full'], r['pred_masked'], abs_tol=1e-6, rel_tol=0) or (r['pred_full'] > 0) != (r['pred_masked'] > 0):
                raise ValueError('ATV Gap-head intervention changed prediction')
        result.append(r)
    return result


def validate_complete(rows):
    grouped = defaultdict(set)
    samples = defaultdict(dict)
    for r in rows:
        key = r['seed'], r['rate'], r['utterance_id']
        grouped[key].add((r['intervention'], r['head']))
        samples[(r['seed'], r['rate'])][r['utterance_id']] = r['label']
    expected = {(kind, head) for kind in ('base', 'gap') for head in range(8)}
    if not grouped or any(values != expected for values in grouped.values()):
        raise ValueError('Expected all 16 interventions per sample')
    for seed in {r['seed'] for r in rows}:
        rates = sorted(rate for s, rate in samples if s == seed)
        if rates != [i / 10 for i in range(8)]:
            raise ValueError('Expected eight rates')
        if any(samples[(seed, rate)] != samples[(seed, 0.)] for rate in rates):
            raise ValueError('Sample ID / label set differs across rates')


def score_rows(rows):
    valid = [r for r in rows if r['label'] != 0]
    full, masked = common.wf1(valid, 'pred_full'), common.wf1(valid, 'pred_masked')
    return dict(n=len(valid), all_utterances=len(rows), full_wf1=full, masked_wf1=masked,
                contribution=(full - masked if valid else None),
                mask_corrections=sum((r['pred_full'] > 0) != (r['label'] > 0) and (r['pred_masked'] > 0) == (r['label'] > 0) for r in valid),
                mask_harms=sum((r['pred_full'] > 0) == (r['label'] > 0) and (r['pred_masked'] > 0) != (r['label'] > 0) for r in valid))


def analyze(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['seed'], r['rate'], r['intervention'], r['head'])].append(r)
    per_rate = []
    for (seed, rate, kind, head), current in sorted(grouped.items()):
        for pattern in ('ALL',) + common.PATTERNS:
            chosen = current if pattern == 'ALL' else [r for r in current if r['availability'] == pattern]
            per_rate.append(dict(seed=seed, rate=rate, intervention=kind, head=head,
                                 availability=pattern, **score_rows(chosen)))
    macro = []
    for kind, head in sorted({(r['intervention'], r['head']) for r in rows}):
        for scope in ('overall', 'high_missing'):
            for pattern in ('ALL',) + common.PATTERNS:
                selected = [r for r in per_rate if r['intervention'] == kind and r['head'] == head and r['availability'] == pattern
                            and (scope == 'overall' or r['rate'] >= .5)]
                macro.append(dict(intervention=kind, head=head, scope=scope, availability=pattern,
                                  **common.macro(selected, SCORES, COUNTS)))
    matrix = []
    for kind, head in sorted({(r['intervention'], r['head']) for r in rows}):
        for scope in ('overall', 'high_missing'):
            selected = [r for r in macro if r['intervention'] == kind and r['head'] == head and r['scope'] == scope]
            matrix.append(dict(intervention=kind, head=head, scope=scope,
                               **{r['availability']: r['contribution'] for r in selected}))
    return dict(per_rate=per_rate, macro=macro, matrix=matrix)


def report(data):
    text = ['# Forward-read head intervention', '', 'INTERNAL DIAGNOSTIC ONLY', '',
            'Same original Full checkpoint per rate. Only a selected 64-d forward read slice is zeroed before the classification input. '
            'Base-head intervention masks that Base slice; Gap-head intervention masks that slice in all three Gap slots (inactive slots remain zero). '
            'This is not a per-modality Gap-head intervention, and does not modify query, write, scan, Local, or task head.',
            'Contribution = W-F1(Full) − W-F1(masked), in percentage points. Positive means masking reduces performance. '
            'Head indices are zero-based, 0–7. Current availability conditions label columns; they are not missing-pattern-specific retrained models.',
            'Nonzero MOSI labels only; polarity uses prediction > 0. Per-rate metrics precede equal-rate macro averaging. '
            'High missing means rates 0.5/0.6/0.7. Counts sum repeated rate exposures, not independent observations.', '']
    for kind in ('base', 'gap'):
        text += [f'## {kind.title()} head contribution matrix (8-rate macro)', '',
                 '|Head|Overall|A|T|V|AT|AV|TV|ATV|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in data['matrix']:
            if r['intervention'] == kind and r['scope'] == 'overall':
                text.append(f"|{r['head']}|" + '|'.join(common.fmt(r[k]) for k in ('ALL',) + common.PATTERNS) + '|')
    text += ['', '## Aggregate paired intervention statistics', '',
             '|Intervention|Head|Scope|Full W-F1|Masked W-F1|Contribution|Mask corrections|Mask harms|N|',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in data['macro']:
        if r['availability'] == 'ALL':
            text.append(f"|{r['intervention']}|{r['head']}|{r['scope']}|" + '|'.join(common.fmt(r[k]) for k in SCORES) +
                        f"|{r['mask_corrections']}|{r['mask_harms']}|{r['n']}|")
    text += ['', 'Mask corrections / harms compare Full → masked, not the reverse. Raw per-rate and per-availability metrics and '
             'sample counts are in per_rate.csv; high-missing and all stratified macro results are in macro.csv.',
             'Checkpoint weights differ across rates. A one-seed leave-one-head-out response is not proof of head specialization '
             'or conditional routing; marginal effects can overlap and need not add. No significance claim, gate fitting, '
             'test-label-based head selection, or Query/Addressing third-stage intervention was performed.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.input.open(newline='') as stream:
        rows = prepare_rows(csv.DictReader(stream))
    validate_complete(rows)
    data = analyze(rows)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Input changed during analysis')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, values in data.items():
        common.write_csv(args.output / f'{name}.csv', values)
    (args.output / 'RESULT.md').write_text(report(data))
    (args.output / 'provenance.json').write_text(json.dumps(dict(input=str(args.input.resolve()), sha256=digest), indent=2) + '\n')
    print(json.dumps(data['matrix'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
