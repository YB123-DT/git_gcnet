"""Fivefold seed66 ACC-selected ablations, never hide missing folds."""
import argparse
import csv
import json
from pathlib import Path

RATES = tuple(f'{i/10:.1f}' for i in range(8))
METRICS = ('ACC','UA','WF1','CE')


def collect(root):
    rows, runs, means = [], {}, {}
    for variant in ('A0','A1','A2','A3'):
        complete = []
        for fold in range(1,6):
            path = root/'runs'/f'fold_{fold}'/variant/'SUMMARY.json'
            run = json.loads(path.read_text()) if path.exists() else dict(status='not_started',epoch=0)
            runs[f'fold{fold}_{variant}'] = run
            if run['status'] == 'complete':
                complete.append(run)
            for rate, best in run.get('best_by_rate',{}).items():
                rows.append(dict(variant=variant,fold=fold,status=run['status'],rate=rate,
                                 selected_epoch=best['epoch'],**best['metrics']))
        if len(complete) == 5:
            per_rate = {r:{k:sum(s['best_by_rate'][r]['metrics'][k] for s in complete)/5
                           for k in METRICS} for r in RATES}
            means[variant] = dict(per_rate=per_rate,aggregation='arithmetic mean of five fold metrics',
                mean={k:sum(per_rate[r][k] for r in RATES)/8 for k in METRICS},
                high={k:sum(per_rate[r][k] for r in RATES[5:])/3 for k in METRICS})
    for fold in range(1,6):
        siblings = [runs[f'fold{fold}_{v}'] for v in ('A0','A1','A2','A3')]
        identities = [r['identity'] for r in siblings if 'identity' in r]
        for identity in identities[1:]:
            for key in ('initial_backbone_sha256','mask_reference','split','environment','source_hashes'):
                if identity[key] != identities[0][key]:
                    raise ValueError(f'Nonpaired fold{fold} identity: {key}')
        history = []
        for v in ('A0','A1','A2','A3'):
            path = root/'runs'/f'fold_{fold}'/v/'history.json'
            if path.exists():
                history.append(json.loads(path.read_text()))
        for e in range(min((len(h) for h in history),default=0)):
            if len({h[e]['mask_trace_sha256'] for h in history}) != 1:
                raise ValueError(f'Nonpaired fold{fold}, epoch{e+1} trainingmask trace')
    return dict(status='complete' if len(means)==4 else 'incomplete',
                label='INTERNAL TEST-ORACLE SCREEN; per-rate TEST ACC; same-call UA/WF1',
                runs=runs,fold_means=means,per_rate=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = collect(args.output)
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields = ('variant','fold','status','rate','selected_epoch','ACC','UA','WF1','CE','N','mask_sha256')
    with (args.output/'per_rate.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['per_rate'])
    text = ['# IEMOCAP-6 A0–A3 seed66 五折','',result['label'],'',
            '|组|完成fold数|五折八率ACC|五折八率UA|五折八率W-F1|',
            '|---|---:|---:|---:|---:|']
    for v in ('A0','A1','A2','A3'):
        count = sum(result['runs'][f'fold{f}_{v}']['status']=='complete' for f in range(1,6))
        if v in result['fold_means']:
            m = result['fold_means'][v]['mean']
            text.append(f"|{v}|{count}/5|{m['ACC']:.3f}|{m['UA']:.3f}|{m['WF1']:.3f}|")
        else:
            text.append(f'|{v}|{count}/5|—|—|—|')
    text.extend(['','未满五折不填最终均值。五折数值为等权折均值，非跨折预测池化。',
                 '每折每率按TEST ACC选epoch，UA/W-F1来自同次预测；原GCNet Missing不变。','',
                 '|组|Fold|Miss|选点Epoch|状态|ACC|UA|W-F1|',
                 '|---|---:|---:|---:|---|---:|---:|---:|'])
    for row in result['per_rate']:
        text.append(f"|{row['variant']}|{row['fold']}|{row['rate']}|{row['selected_epoch']}|{row['status']}|"
                    f"{row['ACC']:.3f}|{row['UA']:.3f}|{row['WF1']:.3f}|")
    (args.output/'RESULT.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text[:12]))


if __name__ == '__main__':
    main()
