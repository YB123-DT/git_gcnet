"""Verified paired NeuMiss three-seed results; no new inference or selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from experiments.osram_three_block_joint_mosi_20261025.scheduler import now, save
from experiments.osram_neumiss_fixed_rate_stage2_20261025.scheduler import digest

RATES = tuple(f'{i/10:.1f}' for i in range(8))
METRICS = ('ACC','WF1','MSE')


def check_epoch(summary,rate,history):
    if summary.get('status') != 'complete':
        raise ValueError('Only completed runs can participate')
    if rate == '0.0':
        if (summary.get('epoch') != 0 or history or not summary.get('no_op')
                or not summary['best'].get('no_op') or summary['best']['epoch'] != 0
                or any(summary['baseline'][k] != summary['best']['metrics'][k] for k in METRICS)):
            raise ValueError('ATV epoch-zero no-op must exactly reproduce its frozen parent')
    elif (summary.get('epoch') != 150 or [r['epoch'] for r in history] != list(range(1,151))
            or any(r['training_rate'] != rate or r['optimizer_steps'] != 2 for r in history)):
        raise ValueError('Expected contiguous 150-epoch fixed-rate training with 300 updates')


def validated_row(path,seed,rate):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    summary = json.loads(path.read_text())
    history = json.loads((path.parent/'history.json').read_text())
    check_epoch(summary,rate,history)
    identity = summary['identity']
    parent = identity['parent']
    if (identity['seed'] != seed or identity.get('mode') != 'neumiss'
            or identity['training_rate'] != rate or identity['evaluation_rate'] != rate
            or identity.get('cyclic_training') is not False
            or identity['mask_protocol'] != 'gcnet-random-mask-v1'):
        raise ValueError(f'{path}: NeuMiss seed/fixed-rate/mask identity mismatch')
    if parent['config']['seed'] != seed:
        raise ValueError(f'{path}: frozen parent is from a different seed')
    parent_path = Path(parent['checkpoint'])
    expected = parent['checkpoint_sha256']
    if summary['parent_id'] != expected or digest(parent_path) != expected:
        raise ValueError(f'{path}: frozen parent checkpoint digest mismatch')
    accepted = json.loads((parent_path.parent/'SUMMARY.json').read_text())['best_by_rate'][rate]
    if accepted['checkpoint_sha256'] != expected or (parent_path.parent/accepted['checkpoint']).resolve() != parent_path.resolve():
        raise ValueError(f'{path}: parent is not the accepted same-seed/rate BEST')
    for key in ('parent_unchanged','parent_grad_none','baseline_replay_exact',
                'only_requested_train_rate','only_requested_test_rate'):
        if summary['checks'].get(key) is not True:
            raise ValueError(f'{path}: missing required check {key}')
    best = summary['best']
    if rate != '0.0':
        if not summary['checks'].get('checkpoint_replay_exact'):
            raise ValueError('Missing selected checkpoint replay proof')
        if digest(path.parent/best['checkpoint']) != best['checkpoint_sha256']:
            raise ValueError('Selected block checkpoint digest mismatch')
        chosen = max(history,key=lambda row:row['test']['WF1'])
        if chosen['epoch'] != best['epoch'] or chosen['test'] != best['metrics']:
            raise ValueError('Selected metrics do not match earliest TEST W-F1 best epoch')
    predictions = path.parent/best['predictions']
    if digest(predictions) != best['prediction_sha256']:
        raise ValueError('Selected predictions digest mismatch')
    with np.load(predictions,allow_pickle=False) as arrays:
        y,p,b = arrays['labels'].reshape(-1),arrays['prediction'].reshape(-1),arrays['base_prediction'].reshape(-1)
        if hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != identity['test_mask_sha256']:
            raise ValueError('Selected prediction mask digest differs from parent replay mask')
        valid = y != 0
        def scores(pred):
            return dict(ACC=100*accuracy_score(y[valid]>0,pred[valid]>0),
                WF1=100*f1_score(y[valid]>0,pred[valid]>0,average='weighted'),
                MSE=float(np.mean((y-pred)**2)))
        actual,baseline = scores(p),scores(b)
        with np.load(parent_path.parent/accepted['predictions'],allow_pickle=False) as reference:
            for key in ('labels','availability','conversation_ids','utterance_indices','speakers'):
                if key in reference:
                    np.testing.assert_array_equal(arrays[key].reshape(-1),reference[key].reshape(-1))
    row = dict(seed=seed,rate=rate,selected_epoch=best['epoch'])
    for key in METRICS:
        if abs(actual[key]-best['metrics'][key])>1e-5 or abs(baseline[key]-summary['baseline'][key])>1e-5:
            raise ValueError(f'{path}: independently recomputed {key} differs')
        row.update({key:actual[key],'parent_'+key:baseline[key],'delta_'+key:actual[key]-baseline[key]})
    return row,dict(summary=str(path),parent_id=expected,test_mask_sha256=identity['test_mask_sha256'],
        epoch=summary['epoch'],selected_epoch=best['epoch'],independent_prediction_metrics=True,
        reused=seed==66,source_commit=identity['source_commit'],
        block_source_sha256=identity['source_hashes']['osram_four_block_wiring/blocks.py'])


def aggregate_rows(rows):
    if len(rows)!=24 or {(r['seed'],r['rate']) for r in rows}!={(s,r) for s in (66,67,68) for r in RATES}:
        raise ValueError('Expected exactly 24 unique seed/rate results')
    fields = [key for key in rows[0] if key not in ('seed','rate','selected_epoch')]
    seeds = [dict(seed=s,rate=scope,selected_epoch=None,
        **{k:statistics.mean(r[k] for r in rows if r['seed']==s and r['rate'] in rates) for k in fields})
        for s in (66,67,68) for scope,rates in (('mean',RATES),('high',RATES[5:]))]
    groups = []
    for scope in (*RATES,'mean','high'):
        selected = [r for r in rows+seeds if r['rate']==scope]
        groups.append(dict(rate=scope,n_seeds=3,metrics={k:dict(mean=statistics.mean(r[k] for r in selected),
            sample_sd=statistics.stdev(r[k] for r in selected)) for k in fields}))
    return seeds,groups


def collect(args,state):
    rows,proofs = [],{}
    paths = [(args.mosi66_root/'train'/f'neumiss_{r}'/'SUMMARY.json',66,r) for r in RATES]
    paths += [(Path(j['summary']),j['seed'],j['rate']) for j in state['train']]
    for path,seed,rate in paths:
        row,proof = validated_row(path,seed,rate)
        rows.append(row)
        proofs[f'{seed}_{rate}'] = proof
    if len({p['block_source_sha256'] for p in proofs.values()}) != 1:
        raise ValueError('NeuMiss block source differs across the three-seed comparison')
    seeds,groups = aggregate_rows(rows)
    save(args.root/'RESULT.json',dict(status='complete',source_commit=args.source_commit,completed_at=now(),
        protocol='NeuMiss fixed-rate frozen-parent INTERNAL TEST-ORACLE; TEST W-F1 earliest strict tie',
        per_rate=rows,seed_aggregates=seeds,across_seed=groups,evidence=proofs))
    with (args.root/'RESULT.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows+seeds)
    with (args.root/'GROUPS.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rate','n_seeds','metric','mean','sample_sd'])
        for group in groups:
            for key,value in group['metrics'].items():
                writer.writerow([group['rate'],3,key,value['mean'],value['sample_sd']])
    lines = ['# NeuMiss MOSI three-seed fixed-rate stage two','',
        f'Source `{args.source_commit}`. INTERNAL TEST-ORACLE; not an independent test generalization claim.','',
        'Seed66 is reused unchanged. Seeds67/68 add fourteen 150-epoch branches and two epoch-zero ATV replays.',
        'Each rate uses its own frozen accepted Flat. Gold-zero labels remain in training/metric MSE;',
        'binary ACC/W-F1 exclude gold zero. SD is sample SD across seed means, not across rates.','',
        '| Seed | Rate/scope | ACC | W-F1 | MSE | Parent W-F1 | Δ W-F1 | Best epoch |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in seeds+rows:
        lines.append(f'| {row["seed"]} | {row["rate"]} | {row["ACC"]:.4f} | {row["WF1"]:.4f} | '
            f'{row["MSE"]:.4f} | {row["parent_WF1"]:.4f} | {row["delta_WF1"]:+.4f} | {row["selected_epoch"]} |')
    lines += ['','| Scope | ACC mean ± SD | W-F1 mean ± SD | Δ W-F1 mean ± SD |',
        '| --- | ---: | ---: | ---: |']
    for group in groups:
        values = [f'{group["metrics"][k]["mean"]:.4f} ± {group["metrics"][k]["sample_sd"]:.4f}'
                  for k in ('ACC','WF1','delta_WF1')]
        lines.append('| '+group['rate']+' | '+' | '.join(values)+' |')
    (args.root/'RESULT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args(argv)
    state = json.loads((args.root/'QUEUE.json').read_text())
    args.mosi66_root = Path(state['mosi66_root'])
    args.source_commit = state['source_commit']
    collect(args,state)


if __name__ == '__main__':
    main()
