"""Explicit stage entrypoints; no automatic chained or five-seed launch."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.osram_causal_nojepa_20260910.run import configuration, runner

ROOT=Path('/data2/yb/remote_experiments/osram_supervised_teacher_20260914')


def teacher_config(base):
    return replace(base,training_objective='emotion-only',train_rate_mode='fixed',
                   fixed_missing_rate=0.,checkpoint_selection='validation',evaluate_test=False,
                   teacher_mode='ema',teacher_checkpoint=None,initial_backbone_checkpoint=None)


def student_config(base,path,training_objective='joint'):
    if training_objective not in {'joint','joint-reg-only','emotion-only'}:
        raise ValueError('student objective must be joint, joint-reg-only, or emotion-only')
    return replace(base,training_objective=training_objective,train_rate_mode='cyclic',
                   fixed_missing_rate=None,
                   checkpoint_selection='test-oracle-per-rate',evaluate_test=True,
                   teacher_mode='pretrained-frozen',teacher_checkpoint=str(path),
                   initial_backbone_checkpoint=None)


def run(stage,seed,student_objective='joint'):
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment
    from gcnet_missing_m3.pretrained_teacher import export_teacher_projectors, read_source
    base,source,_=configuration(seed)
    teacher_path=ROOT/'teacher'/f'seed_{seed}'/'teacher_projectors.pt'
    cfg=(teacher_config(base) if stage=='teacher'
         else student_config(base,teacher_path,student_objective))
    if stage=='student':
        checkpoint,_,_=read_source(teacher_path)
        if checkpoint.get('diagnostic_only',False):
            raise ValueError('Do not use the one-batch smoke Teacher for a full experiment')
        cost_path=teacher_path.parent/'COST.json'
        if not cost_path.exists():
            raise ValueError('Stage1 cost/provenance is required before Stage2')
    if stage=='teacher':
        output=ROOT/'teacher'/f'seed_{seed}'
    else:
        student_dir={'joint':'student','joint-reg-only':'student-reg-only',
                     'emotion-only':'student-emotion-only'}[student_objective]
        output=ROOT/student_dir/f'seed_{seed}'
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    provenance=dict(stage=stage,status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                    from_scratch=True,student_backbone_transfer=False,reference=str(source),
                    config=asdict(cfg),source_sha256={name:runner.sha(REPO/name) for name in (
                        'gcnet_missing_m3/loss.py','gcnet_missing_m3/model.py',
                        'gcnet_missing_m3/pretrained_teacher.py',
                        'gcnet_missing_m3/train_gcnet.py',
                        'experiments/osram_supervised_teacher_20260914/run.py')})
    runner.write_json(output/'PROVENANCE.json',provenance)
    try:
        torch.set_num_threads(6)
        roots=[str(runner.FEATURES/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        run_experiment(cfg,*roots,output_dir=str(output))
        history=json.loads((output/'history.json').read_text())
        cost=dict(epochs_completed=len(history),optimizer_steps=sum(h['train']['optimizer_steps'] for h in history),
                  elapsed_seconds=time.monotonic()-start)
        if stage=='teacher':
            export_teacher_projectors(output/'best.pt',teacher_path)
            metrics=json.loads((output/'metrics.json').read_text())
            cost['selected_epoch']=metrics['best_epoch']
        else:
            metrics=json.loads((output/'metrics.json').read_text())
            assert metrics['teacher_integrity']['unchanged']
            assert runner.mask_hashes(metrics)==runner.mask_hashes(json.loads((source/'metrics.json').read_text()))
            cost['additional_teacher_pretraining']=json.loads(cost_path.read_text())
        runner.write_json(output/'COST.json',cost)
    except BaseException as error:
        provenance.update(status='failed',error=f'{type(error).__name__}: {error}')
        runner.write_json(output/'PROVENANCE.json',provenance)
        raise
    provenance.update(status='complete',elapsed_seconds=time.monotonic()-start)
    runner.write_json(output/'PROVENANCE.json',provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('teacher','student'))
    p.add_argument('--seed',required=True,type=int,choices=runner.SEEDS)
    p.add_argument('--student-objective',
                   choices=('joint','joint-reg-only','emotion-only'),
                   default='joint',help='Stage-2 auxiliary objective')
    p.add_argument('--check',action='store_true',help='Print config only, never train')
    args=p.parse_args()
    if args.check:
        base,_,_=configuration(args.seed)
        cfg=(teacher_config(base) if args.stage=='teacher' else
             student_config(base,ROOT/'teacher'/f'seed_{args.seed}'/'teacher_projectors.pt',
                            args.student_objective))
        print(json.dumps(asdict(cfg),indent=2))
    else:
        run(args.stage,args.seed,args.student_objective)
