"""Read-only regrouping of exact selected Flat predictions; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(labels, prediction, selected):
    binary = selected & (labels != 0)
    n = int(binary.sum())
    return dict(N_all=int(selected.sum()), N_binary=n,
        ACC=float(100 * accuracy_score(labels[binary] > 0, prediction[binary] > 0)) if n else None,
        WF1=float(100 * f1_score(labels[binary] > 0, prediction[binary] > 0,
                                average='weighted', zero_division=0)) if n else None,
        errors=int(((labels > 0) != (prediction > 0))[binary].sum()),
        MSE=float(np.mean((labels[selected] - prediction[selected]) ** 2)) if selected.any() else None)


def analyze(root):
    summary_path = root / 'SUMMARY.json'
    summary = json.loads(summary_path.read_text())
    result = dict(protocol='INTERNAL TEST-ORACLE SCREEN; saved selected predictions only',
                  source=str(root), summary_sha256=sha(summary_path),
                  polarity='labels != 0; positive iff prediction > 0; MSE includes neutral', rates={})
    for rate, best in sorted(summary['best_by_rate'].items()):
        path = root / best['predictions']
        assert sha(path) == best['prediction_sha256'], path
        with np.load(path, allow_pickle=False) as data:
            y = data['labels'].reshape(-1)
            p = data['prediction'].reshape(-1)
            av = data['availability']
            ids, times = data['conversation_ids'], data['utterance_indices']
        assert av.shape == (len(y), 3) and len(p) == len(y)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        assert np.isin(av, (0, 1)).all() and (av.sum(-1) >= 1).all()
        assert hashlib.sha256(av.tobytes()).hexdigest() == best['metrics']['mask_sha256']
        seen_t, state, last, identities = np.zeros(len(y), dtype=bool), {}, {}, set()
        for i, (cid, t) in enumerate(zip(ids.tolist(), times.tolist())):
            assert (cid, t) not in identities
            identities.add((cid, t))
            assert t == last.get(cid, -1) + 1, (cid, t)
            seen_t[i] = state.get(cid, False)
            state[cid] = seen_t[i] or bool(av[i, 1])
            last[cid] = t
        all_rows = np.ones(len(y), dtype=bool)
        total = metrics(y, p, all_rows)
        for key in ('ACC', 'WF1', 'MSE'):
            assert abs(total[key] - best['metrics']['router'][key]) < 1e-5, (rate, key)
        patterns = {name: tuple(int(m in name) for m in 'ATV')
                    for name in ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')}
        by_pattern = {name: metrics(y, p, (av == pattern).all(-1))
                      for name, pattern in patterns.items()}
        assert sum(row['N_all'] for row in by_pattern.values()) == len(y)
        assert sum(row['errors'] for row in by_pattern.values()) == total['errors']
        assert sum(row['N_binary'] for row in by_pattern.values()) == total['N_binary']
        text_groups = {'T_present': av[:, 1] == 1,
                       'T_missing_history_seen_T': (av[:, 1] == 0) & seen_t,
                       'T_missing_history_never_T': (av[:, 1] == 0) & ~seen_t}
        assert sum(int(x.sum()) for x in text_groups.values()) == len(y)
        if rate == '0.7':
            assert (av.sum(-1) == 1).all()
        result['rates'][rate] = dict(epoch=best['epoch'], prediction_sha256=sha(path),
            mask_sha256=best['metrics']['mask_sha256'], overall=total, by_pattern=by_pattern,
            by_text_history={name: metrics(y, p, select) for name, select in text_groups.items()})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    for rate in ('0.5', '0.6', '0.7'):
        print(json.dumps(dict(rate=rate, **result['rates'][rate])))
