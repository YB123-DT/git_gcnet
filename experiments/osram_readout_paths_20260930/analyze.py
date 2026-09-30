"""Paired readout responses; descriptive diagnostics, never coefficient selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def losses(labels, predictions, dataset):
    if dataset == 'CMUMOSI':
        return (predictions.reshape(-1).astype(float) - labels) ** 2
    z = predictions.astype(float)
    z = z - z.max(axis=1, keepdims=True)
    return np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(labels)), labels.astype(int)]


def paired_metrics(labels, predictions, baseline, dataset):
    labels = np.asarray(labels)
    loss = losses(labels, predictions, dataset)
    ref_loss = losses(labels, baseline, dataset)
    selected = labels != 0 if dataset == 'CMUMOSI' else np.ones(len(labels), dtype=bool)
    if dataset == 'CMUMOSI':
        target = labels[selected] > 0
        p = predictions.reshape(-1)[selected] > 0
        b = baseline.reshape(-1)[selected] > 0
    else:
        target = labels[selected].astype(int)
        p, b = predictions.argmax(-1)[selected], baseline.argmax(-1)[selected]
    correct, original = p == target, b == target
    metric = lambda x: float(f1_score(target, x, average='weighted', zero_division=0)) if target.size else None
    wf1, ref_wf1 = metric(p), metric(b)
    result = dict(sample_count=len(labels), metric_count=int(selected.sum()),
        loss=float(loss.mean()), delta_loss=float((loss-ref_loss).mean()),
        loss_improved=int((loss < ref_loss).sum()), loss_worsened=int((loss > ref_loss).sum()),
        wrong_to_right=int((~original & correct).sum()), right_to_wrong=int((original & ~correct).sum()),
        weighted_f1=wf1, delta_weighted_f1=wf1-ref_wf1 if wf1 is not None else None,
        accuracy=float(accuracy_score(target, p)) if target.size else None,
        delta_accuracy=float(correct.mean()-original.mean()) if target.size else None)
    if dataset == 'CMUMOSI':
        values = predictions.reshape(-1)
        result.update(mae=float(np.abs(values-labels).mean()),
            correlation=float(np.corrcoef(labels, values)[0, 1]) if len(labels)>1 and np.std(labels)>0 and np.std(values)>0 else 0.)
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty analysis')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    rows, conversation_rows = [], []
    files = sorted(root.rglob('*.npz'))
    if not files:
        raise ValueError('no prediction files')
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            y, p = data['labels'], data['predictions']
            ids, ut = data['conversation_ids'].astype(str), data['utterance_indices']
            av, settings = data['availability'], data['settings']
            expected_settings = np.asarray([(a,a,m) for a in (.8,1.,1.2) for m in (.8,1.,1.2)] + [(.8,1.,1.),(1.2,1.,1.),(1.,.8,1.),(1.,1.2,1.)])
            if not np.array_equal(settings, expected_settings):
                raise ValueError('13 settings differ from the predeclared design')
            seed, rate, split = int(data['seed']), float(data['rate']), str(data['split'])
            if split not in ('train', 'validation'):
                raise ValueError('test predictions prohibited')
            if len(set(zip(ids, ut.tolist()))) != len(y) or p.shape[:2] != (13, len(y)):
                raise ValueError('duplicate IDs or prediction alignment failure')
            if not np.isfinite(p).all() or not np.isfinite(y).all():
                raise ValueError('nonfinite predictions')
            baseline_idx = np.flatnonzero(np.all(settings == 1., axis=1))
            if len(baseline_idx) != 1:
                raise ValueError('exactly one identity required')
            baseline = p[int(baseline_idx[0])]
            patterns = np.array([''.join(m for m, bit in zip(('A','T','V'), a) if bit) or 'NONE' for a in av])
            for k, (alpha, beta, mu) in enumerate(settings):
                meta = dict(seed=seed, rate=rate, split=split, alpha=float(alpha), beta=float(beta), mu=float(mu))
                for pattern in ('ALL','A','T','V','AT','AV','TV','ATV'):
                    selected = np.ones(len(y), dtype=bool) if pattern == 'ALL' else patterns == pattern
                    if selected.any():
                        rows.append(dict(**meta, pattern=pattern, **paired_metrics(y[selected], p[k,selected], baseline[selected], 'CMUMOSI')))
                for cid in np.unique(ids):
                    selected = ids == cid
                    conversation_rows.append(dict(**meta, conversation_id=cid,
                        **paired_metrics(y[selected], p[k,selected], baseline[selected], 'CMUMOSI')))
    write_csv(output/'by_seed_rate_pattern.csv', rows)
    write_csv(output/'by_conversation.csv', conversation_rows)
    summary = []
    keys = sorted({(r['split'],r['alpha'],r['beta'],r['mu'],r['pattern']) for r in rows})
    for split, a, b, m, pattern in keys:
        selected = [r for r in rows if (r['split'],r['alpha'],r['beta'],r['mu'],r['pattern']) == (split,a,b,m,pattern)]
        by_seed = []
        for seed in sorted({r['seed'] for r in selected}):
            sr = [r for r in selected if r['seed']==seed]
            valid_f1 = [r['delta_weighted_f1'] for r in sr if r['delta_weighted_f1'] is not None]
            by_seed.append(dict(seed=seed, rate_cells=len(sr),
                delta_loss=float(np.mean([r['delta_loss'] for r in sr])),
                delta_wf1_pp=float(np.mean(valid_f1)*100) if valid_f1 else None))
        dl = [r['delta_loss'] for r in by_seed]
        df = [r['delta_wf1_pp'] for r in by_seed if r['delta_wf1_pp'] is not None]
        summary.append(dict(split=split,alpha=a,beta=b,mu=m,pattern=pattern,
            delta_loss_mean=float(np.mean(dl)),delta_loss_seed_sd=float(np.std(dl,ddof=1)) if len(dl)>1 else None,
            delta_wf1_pp_mean=float(np.mean(df)) if df else None,delta_wf1_pp_seed_sd=float(np.std(df,ddof=1)) if len(df)>1 else None,
            seeds_loss_improved=sum(x<0 for x in dl),seeds_wf1_improved=sum(x>0 for x in df),
            wrong_to_right_sum=sum(r['wrong_to_right'] for r in selected),
            right_to_wrong_sum=sum(r['right_to_wrong'] for r in selected),by_seed=by_seed))
    # Conversation-balanced response: average repeated seed/rate measurements
    # within each conversation before resampling conversations, not utterances.
    conversation_summary = []
    for split,a,b,m in sorted({(r['split'],r['alpha'],r['beta'],r['mu']) for r in conversation_rows}):
        selected=[r for r in conversation_rows if (r['split'],r['alpha'],r['beta'],r['mu'])==(split,a,b,m)]
        values=np.array([np.mean([r['delta_loss'] for r in selected if r['conversation_id']==cid]) for cid in sorted({r['conversation_id'] for r in selected})])
        rng=np.random.default_rng(20260930)
        boot=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
        conversation_summary.append(dict(split=split,alpha=a,beta=b,mu=m,conversations=len(values),
            delta_loss_conversation_mean=float(values.mean()), improved_conversations=int((values<0).sum()),
            descriptive_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist()))
    report=dict(label='FROZEN PAIRED READOUT DIAGNOSIS; historical Test-oracle checkpoints; no deployment coefficient selection',
        files=len(files),aggregation='rate macro within seed, then seed macro; patterns conditional, absent cells omitted',
        count_warning='transition sums count seed/rate evaluations, NOT unique utterances',
        uncertainty='Conversation bootstrap is descriptive, unadjusted for 13 comparisons; validation only10 conversations. Not a significance claim.',
        rows=summary,conversation_response=conversation_summary)
    (output/'SUMMARY.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for row in summary:
        if row['pattern']=='ALL':
            print(row['split'],(row['alpha'],row['beta'],row['mu']), 'deltaMSE',round(row['delta_loss_mean'],6),'deltaWF1pp',round(row['delta_wf1_pp_mean'],4),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    analyze(args.input,args.output)
