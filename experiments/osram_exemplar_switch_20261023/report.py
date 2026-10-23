"""Small summary of completed, matched frozen exemplar switch predictions."""
import csv
import json


def aggregate(rates, selected):
    rows = [rates[r] for r in selected]
    result = {kind: {metric: sum(row[kind][metric] for row in rows)/len(rows)
                     for metric in ('ACC', 'WF1', 'MSE')}
              for kind in ('Flat', 'OFF', 'ON')}
    result.update(direct_WF1=result['ON']['WF1']-result['OFF']['WF1'],
                  main_path_WF1=result['OFF']['WF1']-result['Flat']['WF1'],
                  total_WF1=result['ON']['WF1']-result['Flat']['WF1'],
                  corrections_rate_sum=sum(r['corrections'] for r in rows),
                  harms_rate_sum=sum(r['harms'] for r in rows))
    return result


def publish(summary, output):
    if summary['status'] != 'complete' or len(summary['variants']) != 2:
        raise ValueError('Require both complete frozen audits')
    result = dict(label='INTERNAL TEST-ORACLE DIAGNOSTIC', aggregate={}, per_rate=[])
    lines = ['# Frozen MOSI66 TRAIN-exemplar ON/OFF', '',
        'INTERNAL TEST-ORACLE DIAGNOSTIC. No retraining. Exact original test masks.', '',
        'OFF bypasses only retrieval from the SAME ON-selected checkpoint; Base/Gap/Local/Flat/head stay trained and unchanged.',
        'Primary ON/OFF share one Memory scan; an additional disabled-wrapper forward independently verifies OFF.', '',
        '| Variant | Scope | Original Flat WF1 | OFF WF1 | ON WF1 | OFF−Flat | ON−OFF | ON−Flat |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, record in summary['variants'].items():
        if record['status'] != 'complete' or set(record['rates']) != {f'{i/10:.1f}' for i in range(8)}:
            raise ValueError('Incomplete checkpoint coverage')
        rates = record['rates']
        result['aggregate'][name] = {}
        for scope, selected in [('mean', tuple(rates)), ('high', ('0.5', '0.6', '0.7'))]:
            a = aggregate(rates, selected)
            result['aggregate'][name][scope] = a
            lines.append(f"| {name} | {scope} | {a['Flat']['WF1']:.3f} | {a['OFF']['WF1']:.3f} | {a['ON']['WF1']:.3f} | {a['main_path_WF1']:+.3f} | {a['direct_WF1']:+.3f} | {a['total_WF1']:+.3f} |")
        for rate, r in rates.items():
            result['per_rate'].append(dict(variant=name, rate=rate, epoch=r['selected_epoch'],
                **{kind+'_'+m: r[kind][m] for kind in ('Flat', 'OFF', 'ON') for m in ('ACC', 'WF1', 'MSE')},
                corrections_OFF_to_ON=r['corrections'], harms_OFF_to_ON=r['harms'],
                prediction_abs_shift=r['mean_abs_prediction_change'], residual_norm=r['residual_norm'], anchor_norm=r['anchor_norm']))
    lines += ['', '## Per-rate switch contribution', '',
        '| Variant | Missing | Epoch | OFF WF1 | ON WF1 | ON−OFF | Corrections | Harms | Mean abs prediction change |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['per_rate']:
        lines.append(f"| {r['variant']} | {r['rate']} | {r['epoch']} | {r['OFF_WF1']:.3f} | {r['ON_WF1']:.3f} | {r['ON_WF1']-r['OFF_WF1']:+.3f} | {r['corrections_OFF_to_ON']} | {r['harms_OFF_to_ON']} | {r['prediction_abs_shift']:.6f} |")
    lines += ['', '## Limits and stopping', '',
        '- Single seed; eight per-rate Test-best checkpoints, not a single deployable checkpoint.',
        '- Same checkpoint and exact mask per switch; OFF receives no new epoch selection.',
        '- Binary ACC/WF1 exclude y=0 and use pred>0; MSE includes all686 valid targets.',
        '- Corrections/harms are OFF→ON; summed counts across rates repeat targets under different masks.',
        '- OFF−Flat reflects differences between separately trained main paths; it is not a causal training decomposition.',
        '- ON−OFF is frozen branch sensitivity, not a separately trained ablation or proof of a training bottleneck.',
        '- All outputs/weights remain frozen. No HPO, further seed, training or module modification automatically follows.', '']
    (output/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'RESULT.md').write_text('\n'.join(lines))
    with (output/'METRICS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['per_rate'][0]))
        writer.writeheader(); writer.writerows(result['per_rate'])
    return result
