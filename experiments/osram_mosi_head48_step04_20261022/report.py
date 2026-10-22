"""Four same-seed completed configurations; no new inference or test masks."""
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.osram_mosi_memory_grid_20261022.config import RATES
from experiments.osram_mosi_memory_grid_20261022.report import validate_completed,METRICS
from .worker import MODE,candidate_config

ROOT=Path('/data2/yb/remote_experiments/osram_mosi_memory_grid_20261022/execution_v1/runs')
CONTROLS={mode:ROOT/mode for mode in ('baseline','step_0p4','head_48')}
CORE=('gcnet_missing_m3/osram.py','gcnet_missing_m3/model.py','gcnet_missing_m3/train_gcnet.py',
      'gcnet_missing_m3/mixed_rate.py','gcnet_modality_jepa/gcnet_legacy_random_mask.py',
      'experiments/osram_text_path_router_joint_mosi_20261019/run.py')


def summarize(output):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    output=Path(output)
    runs={name:json.loads((path/'SUMMARY.json').read_text()) for name,path in CONTROLS.items()}
    current=json.loads((output/'SUMMARY.json').read_text())
    baseline=runs['baseline']['identity']
    for name,run in runs.items():validate_completed(name,run)
    if current['status']=='complete':
        validate_completed(MODE,current)
        expected=asdict(candidate_config(TrainConfig(**baseline['config']),MODE))
        if asdict(TrainConfig(**current['identity']['config']))!=expected:
            raise ValueError('Unexpected changes beyond head48/write_step0.4')
        if current['parameters']['total']!=11245857:
            raise ValueError('Unexpected head48 parameter count')
        if current['identity']['initial_model_sha256']!=runs['head_48']['identity']['initial_model_sha256']:
            raise ValueError('Write step changed head48 fresh initialization')
        runs[MODE]=current
    for name,run in runs.items():
        identity=run['identity']
        for key in ('mask_hashes','split','seed_protocol','data_files','environment'):
            if identity[key]!=baseline[key]:raise ValueError('Paired '+key+' differs: '+name)
        if any(identity['source_hashes'][p]!=baseline['source_hashes'][p] for p in CORE):
            raise ValueError('Numerical core differs: '+name)
    aggregate={name:{scope:{key:sum(run['evaluation'][r]['router'][key] for r in rates)/len(rates)
            for key in METRICS} for scope,rates in (('mean',RATES),('high',RATES[5:]))}
            for name,run in runs.items()}
    rows=[dict(model=name,rate=rate,epoch=run['best_by_rate'][rate]['epoch'],
              **{key:run['evaluation'][rate]['router'][key] for key in METRICS})
          for name,run in runs.items() for rate in RATES]
    result=dict(label='INTERNAL TEST-ORACLE SCREEN',candidate_complete=current['status']=='complete',
        candidate_status=current['status'],candidate_epoch=current.get('epoch'),aggregate=aggregate,per_rate=rows,
        controls={name:dict(path=str(path),summary_sha256=hashlib.sha256((path/'SUMMARY.json').read_bytes()).hexdigest())
                  for name,path in CONTROLS.items()})
    (output/'REPORT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'PER_RATE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','rate','epoch',*METRICS]);writer.writeheader();writer.writerows(rows)
    lines=['# MOSI66 head48 plus write_step0.4','','INTERNAL TEST-ORACLE SCREEN','',
        'One new from-scratch150-epoch job; existing three controls reused read-only.',
        'GCNet random missing,cyclic0.0–0.7,LR3e-4→3e-5,per-rate Test-best W-F1.',
        'Per-rate best checkpoints may differ; not single-checkpoint deployment performance.',
        'Only completed candidates enter scores; this combination was chosen after individual test screens.','',
        '| Configuration | Mean ACC | Mean W-F1 | High ACC | High W-F1 |',
        '|---|---:|---:|---:|---:|']
    for name,v in aggregate.items():
        lines.append(f"| {name} | {v['mean']['ACC']:.3f} | {v['mean']['WF1']:.3f} | {v['high']['ACC']:.3f} | {v['high']['WF1']:.3f} |")
    lines+=['',f"New candidate status: {current['status']},epoch={current.get('epoch')}",
        '','Head48 also changes Base/Gap/Flat input width; it is not a parameter-matched pure storage ablation.',
        'No claims of stable multi-seed improvement from this single additional seed66 run.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result
