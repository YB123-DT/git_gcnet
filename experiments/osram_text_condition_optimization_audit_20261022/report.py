"""Describe local finite-step sensitivity, not a new trained model score."""
import csv
import statistics

from experiments.osram_v3_source_attention_audit_20261022.run import write


def report_dataset(output, strata, records, state):
    groups = ('overall', 'ObservedEncoder', 'WriteKey', 'WriteValue', 'Query', 'Memory', 'LocalFlatHead', 'Other')
    rows = []
    for rate in ['pooled']+[f'{i/10:.1f}' for i in range(8)]:
        selected = records if rate == 'pooled' else [r for r in records if r['rate'] == rate]
        for group in groups:
            geometries = [r['geometry']['overall'] if group == 'overall' else r['geometry']['groups'][group]
                          for r in selected]
            cos = [r['cosine'] for r in geometries if r['cosine'] is not None]
            rows.append(dict(rate=rate, group=group, pairs=len(selected), nonzero_pairs=len(cos),
                mean_cosine=statistics.mean(cos) if cos else None,
                median_cosine=statistics.median(cos) if cos else None,
                negative_fraction=sum(c < 0 for c in cos)/len(cos) if cos else None))
    transfer = {}
    for condition, key in (('visible_to_missing','delta_missing'),('missing_to_visible','delta_visible')):
        updater = 'visible' if condition.startswith('visible') else 'missing'
        changes = [r['virtual_steps'][updater][key] for r in records]
        base_key = 'missing_loss' if updater == 'visible' else 'visible_loss'
        tolerance = [1e-6*max(1.,abs(r[base_key])) for r in records]
        transfer[condition] = dict(N=len(changes),mean_delta_MSE=statistics.mean(changes) if changes else None,
            median_delta_MSE=statistics.median(changes) if changes else None,
            harm_count=sum(x > t for x,t in zip(changes,tolerance)),
            improve_count=sum(x < -t for x,t in zip(changes,tolerance)),
            neutral_count=sum(abs(x) <= t for x,t in zip(changes,tolerance)),
            numerical_tolerance='1e-6*max(1,abs(original opposite-condition MSE))')
    jacobians = [r['prediction_jacobian_geometry']['overall']['cosine'] for r in records
                 if r.get('prediction_jacobian_geometry') is not None]
    summary = dict(dataset=state['dataset'],checkpoint=state['checkpoint'],strata=strata,
        gradient_rows=rows,transfer=transfer,N_pairs=len(records),weights_restored_exact=state['weights_restored_exact'],
        residual_sign_opposed_fraction=statistics.mean(r['residual_sign_opposed'] for r in records),
        prediction_jacobian_mean_cosine=statistics.mean(jacobians) if jacobians else None,
        max_relative_update=max((r['virtual_steps'][s]['relative_update'] for r in records
                                 for s in ('visible','missing')), default=0),
        limitations=['One seed66 fixed miss0.3 Test-oracle checkpoint,not per-rate deployed performance.',
            'Eval-mode gradients remove dropout noise; they are not stochastic training trajectories.',
            'Paired prefixes encode past/current separately with the original utterance-local encoder to fix historical GEMM shapes; test replay remains unmodified.',
            'Temporary SGD is not Adam or persistent optimization; same-pair finite effects are local sensitivity.',
            'MSE residual signs can reverse loss gradients even with aligned prediction Jacobians; inspect both.',
            'Test readout zeroing may be out of distribution; history benefit is model response,not information quality.',
            'Pairs are label-free current-Text interventions on legal GCNet background masks,not fresh performance samples.',
            'Repeated conversations across rates are not independent samples; no significance/long-term transfer claim.'])
    write(output/'SUMMARY.json',summary)
    with (output/'GRADIENT_SUMMARY.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# '+state['dataset']+' Text-condition optimization audit','','INTERNAL DIAGNOSTIC ONLY','',
        'One fixed original Flat66 miss0.3 checkpoint. No new training. Temporary SGD is fully restored.',
        'Test stratification uses exact eight-rate GCNet masks; gradients/updates use TRAIN labels only.','',
        '| Test group (pooled slots) | N | Local WF1 | Full WF1 | Corrections | Harms | LocalMSE-FullMSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in strata['pooled_test_slots'].items():
        fmt=lambda x:'N/A' if x is None else f'{x:.6f}'
        lines.append(f"| {name} | {row['N']} | {fmt(row['local']['WF1'])} | {fmt(row['full']['WF1'])} | {row['corrections']} | {row['harms']} | {fmt(row['local_minus_full_MSE'])} |")
    lines+=['','| Gradient group | Nonzero pairs | Mean cosine | Negative fraction |',
            '|---|---:|---:|---:|']
    for row in rows:
        if row['rate']=='pooled':
            fmt=lambda x:'N/A' if x is None else f'{x:.6f}'
            lines.append(f"| {row['group']} | {row['nonzero_pairs']} | {fmt(row['mean_cosine'])} | {fmt(row['negative_fraction'])} |")
    lines+=['','| Temporary update | Mean cross-condition deltaMSE | Harms | Improvements | Neutral |',
            '|---|---:|---:|---:|---:|']
    for name,row in transfer.items():
        lines.append(f"| {name} | {row['mean_delta_MSE']:.8f} | {row['harm_count']} | {row['improve_count']} | {row['neutral_count']} |")
    lines+=['','## Limits','']+['- '+x for x in summary['limitations']]
    lines+=['',f"Opposite MSE residual signs: {summary['residual_sign_opposed_fraction']:.6f}; prediction-Jacobian mean cosine: {summary['prediction_jacobian_mean_cosine']}."]
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return summary
