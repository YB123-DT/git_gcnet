"""Paired no-Text comparison from saved predictions only; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np


def wf1(y, pred):
    truth, decision = y > 0, pred > 0
    score = 0.0
    for c in (False, True):
        tp = np.sum((truth == c) & (decision == c))
        fp = np.sum((truth != c) & (decision == c))
        fn = np.sum((truth == c) & (decision != c))
        denom = 2 * tp + fp + fn
        score += (2 * tp / denom if denom else 0) * np.sum(truth == c)
    return 100 * float(score / len(y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-gap-root', type=Path, required=True)
    parser.add_argument('--t-only-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, sources = [], []
    for seed in (66, 67, 68):
        roots = [p / f'seed_{seed}' for p in (args.all_gap_root, args.t_only_root)]
        metrics = [json.loads((p / 'metrics.json').read_text()) for p in roots]
        assert all(m['selection_protocol'] == 'per-rate-test-oracle' for m in metrics)
        assert metrics[0]['mask_sha256'] == metrics[1]['mask_sha256']
        for i in range(8):
            rate = i / 10
            files = [p / f'predictions_miss_{rate:.1f}'.replace('.', 'p') for p in roots]
            files = [Path(str(p) + '.npz') for p in files]
            data = [np.load(p, allow_pickle=False) for p in files]
            y, a = data[0]['labels'], data[0]['availability']
            assert np.array_equal(y, data[1]['labels'])
            assert np.array_equal(a, data[1]['availability'])
            assert np.isin(a, (0, 1)).all() and (a.sum(-1) >= 1).all()
            assert all(np.isfinite(d['predictions']).all() for d in data)
            sources.extend(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files)
            patterns = {'all_no_Text': a[:, 1] == 0,
                        'A': (a == [1, 0, 0]).all(-1),
                        'V': (a == [0, 0, 1]).all(-1),
                        'AV': (a == [1, 0, 1]).all(-1)}
            for pattern, mask in patterns.items():
                mask = mask & (y != 0)
                if not mask.any():
                    continue
                yy = y[mask]
                p0, p1 = [d['predictions'][mask] for d in data]
                c0, c1 = (p0 > 0) == (yy > 0), (p1 > 0) == (yy > 0)
                f0, f1 = wf1(yy, p0), wf1(yy, p1)
                records.append(dict(seed=seed, rate=rate, pattern=pattern, n=int(mask.sum()),
                                    all_gap_wf1=f0, t_only_wf1=f1, delta_pp=f1-f0,
                                    corrections=int((~c0 & c1).sum()), harms=int((c0 & ~c1).sum())))
    macro = []
    for seed in ('all', 66, 67, 68):
        for pattern in ('all_no_Text', 'A', 'V', 'AV'):
            for scope in ('all_nonempty_rates', 'high_missing'):
                group = [r for r in records if r['pattern'] == pattern
                         and (seed == 'all' or r['seed'] == seed)
                         and (scope != 'high_missing' or r['rate'] >= .5)]
                if group:
                    macro.append(dict(seed=seed, pattern=pattern, scope=scope, groups=len(group),
                                      **{k: mean(r[k] for r in group) for k in ('all_gap_wf1', 't_only_wf1', 'delta_pp')},
                                      **{k: sum(r[k] for r in group) for k in ('n', 'corrections', 'harms')}))
    summary = dict(label='INTERNAL DIAGNOSTIC ONLY', inference_runs=0, training_runs=0,
                   setting='Current no-Text utterances under random missing, NOT persistent missing',
                   alignment='Identical stored row order, labels, availability and evaluation mask hashes; NPZ has no explicit sample IDs',
                   aggregation='Equal mean of nonempty seed/rate groups; nonzero labels, positive iff prediction > 0',
                   sources=sources, per_rate=records, macro=macro)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps([r for r in macro if r['seed'] == 'all'], indent=2))


if __name__ == '__main__':
    main()
