"""Complete-only paired scalar metrics; the legacy 'router' key is an alias."""
import json
from pathlib import Path
import statistics

import numpy as np

from .worker import RATES,REFERENCES
from experiments.osram_mosi_gcnet_router_3seed_20261019.report import report_outputs


def write_report(output,queue, *, source_role_mode='residual',method_label='Coupled Source Role'):
    if len(queue['jobs'])!=1 or queue['jobs'][0]['status']!='complete':raise ValueError('Completed one-job screen required')
    job=queue['jobs'][0];path=Path(job['output']);s=json.loads((path/'SUMMARY.json').read_text())
    dataset=job['dataset'];epochs=150 if dataset=='CMUMOSI' else 60
    if (s.get('status')!='complete' or s.get('epoch')!=epochs
            or not s['config']['osram_source_role'] or s['router_parameters']!=0
            or s['config'].get('osram_source_role_mode','residual')!=source_role_mode):raise ValueError('Incomplete or wrong model')
    prior=json.loads((REFERENCES[dataset]/'SUMMARY.json').read_text())
    from .worker import validate_reference
    validate_reference(s['identity'],prior,s['initial_backbone_sha256'],epochs,source_role_mode=source_role_mode)
    from experiments.osram_text_path_router_joint_mosi_20261019.run import summarize_arrays
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    from gcnet_missing_m3 import train_gcnet as trainer
    validate_selected(path,s['best_by_rate'])
    history=json.loads((path/'history.json').read_text());rows=[]
    if len(history)!=epochs:raise ValueError('Incomplete history')
    for rate in RATES:
        with np.load(path/'evaluation'/('random_'+rate.replace('.','p')+'.npz'),allow_pickle=False) as f:
            arrays={k:f[k] for k in f.files}
        with np.load(REFERENCES[dataset]/'evaluation'/('random_'+rate.replace('.','p')+'.npz'),allow_pickle=False) as f:
            for k in ('labels','availability','conversation_ids','utterance_indices','speakers'):
                np.testing.assert_array_equal(arrays[k],f[k])
        actual=summarize_arrays(arrays,trainer,dataset)['router']
        for k,v in actual.items():
            if not np.isclose(v,s['evaluation'][rate]['router'][k],rtol=0,atol=1e-8):raise ValueError('Metric mismatch')
        scores=[h['test'][rate]['router']['WF1'] for h in history]
        if s['selected_epoch_by_rate'][rate]!=scores.index(max(scores))+1:raise ValueError('Incorrect selection')
        rows.append(dict(dataset=dataset,seed=66,rate=rate,selected_epoch=s['selected_epoch_by_rate'][rate],
            flat_ACC=prior['evaluation'][rate]['router']['ACC'],flat_WF1=prior['evaluation'][rate]['router']['WF1'],**actual))
    lines=['# '+dataset+' '+method_label+' seed66','','INTERNAL TEST-ORACLE SCREEN','',
        'One-stage original task MSE; no Router, second head, extra query, write change or auxiliary loss.',
        'Existing matched Flat reference reused; original initialization, evaluation masks and training-mask traces checked.',
        'Legacy scalar metric key router means the primary prediction, not a Router module.',
        'Per-rate TEST W-F1 best; different rates may use different weights, not one deployment checkpoint.','',
        '| Rate | Flat ACC | New ACC | Flat W-F1 | New W-F1 | Delta W-F1 pp |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['rate']} | {row['flat_ACC']:.4f} | {row['ACC']:.4f} | {row['flat_WF1']:.4f} | {row['WF1']:.4f} | {row['WF1']-row['flat_WF1']:+.4f} |")
    for scope,subset in (('mean',rows),('high',rows[5:])):
        v={k:statistics.mean(r[k] for r in subset) for k in ('flat_ACC','ACC','flat_WF1','WF1')}
        lines.append(f"| {scope} | {v['flat_ACC']:.4f} | {v['ACC']:.4f} | {v['flat_WF1']:.4f} | {v['WF1']:.4f} | {v['WF1']-v['flat_WF1']:+.4f} |")
    lines+=['',f'Source-role mode: {source_role_mode}.',
        ('Direct weighted source sum: gamma=2*sigmoid(score), NO original read add-back; initialization is tolerance-equal only.'
         if source_role_mode=='direct' else 'Original read plus bounded correction: gamma=.25*tanh(score).'),
        f"Added parameters: {s['source_role_parameters']}; total: {s['total_parameters']}.",
        f"Training wall seconds: {s['seconds']:.1f}; peak allocated MiB: {s['peak_memory_mib']:.1f}.",
        'No independent-test generalization claim; source labels denote additive joint Value-slot contributions, not pure modality semantics.',
        'Diagnostics in per-epoch history are explicitly the LAST TRAINING BATCH, not epoch-global means.']
    report_outputs(output,dict(run=s,flat_reference=prior,checks=dict(selected_predictions_recomputed=True,
        exact_cohort_and_masks=True)),rows,lines)
