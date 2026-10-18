"""Frozen paired readout diagnosis for the fixed small-LR MELD three seeds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.osram_meld_optimizer_screen_20261018 import run as screen
from experiments.osram_meld_small_lr_3seed_20261018.run import SEED66
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler,write_json

SOURCES={66:SEED66,**{s:Path('/home/qinjinghui/remote_experiments/osram_meld_small_lr_3seed_20261018/runs_v1')/f'seed_{s}' for s in (67,68)}}
GPU=1
UUID='GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'
CONDITIONS=('L','BG','LBG')


def expected_config(seed):
    if type(seed) is not int or seed not in SOURCES:raise ValueError('Only the three fixed seeds authorized')
    return dict(screen.make_config(screen.source_config(),'O6'),seed=seed)


def check_sources():
    screen.check_snapshot()
    manifest=json.loads(Path(__file__).with_name('SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        if screen.original.sha(ROOT/name)!=digest:raise ValueError('Diagnostic source changed: '+name)
    return manifest


def render(rows):
    lines=['# MELD small-LR: frozen Local / Memory / Full','',
        'INTERNAL DIAGNOSTIC ONLY. Readout interventions, not separately trained classifiers.',
        'Same per-rate ACC-best checkpoint and exact selected masks. One CHM scan per batch.',
        'L: Local in adapter + original skip; BG: Base/active Gap only, both Local paths off;',
        'LBG: Full. Norm, adapter biases and task head remain unchanged.','',
        '| Seed | Readout | ACC | UA | W-F1 | CE |','|---|---|---:|---:|---:|---:|']
    for seed,row in rows.items():
        for condition in CONDITIONS:
            values=row['mean'][condition]
            lines.append('| '+str(seed)+' | '+condition+' | '+' | '.join(f'{values[k]:.3f}' for k in ('ACC','UA','WF1','CE'))+' |')
    lines+=['','## Per-rate results','',
        '| Seed | Rate | Readout | ACC | UA | W-F1 |','|---|---:|---|---:|---:|---:|']
    for seed,row in rows.items():
        for rate,result in row['rates'].items():
            for c in CONDITIONS:
                m=result['metrics'][c]
                lines.append(f'| {seed} | {rate} | {c} | '+ ' | '.join(f'{m[k]:.3f}' for k in ('ACC','UA','WF1'))+' |')
    lines+=['','Removing inputs can be out of distribution. These scores measure this frozen',
        'readout response, not the intrinsic information ceiling or a learned Local/Memory probe.']
    return '\n'.join(lines)+'\n'


def worker(args):
    manifest=check_sources()
    if SOURCES[args.seed]!=args.source:raise ValueError('Wrong selected source')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,GCNET_MELD_PICKLE=str(screen.original.DATASET),
        GCNET_DATASET_ROOT=str(screen.MODEL_SOURCE/'dataset'),GCNET_CACHE_ROOT=str(args.output/'feature_cache'))
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment,repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu,state_digest
    import torch,numpy as np
    gpu=guard_gpu(args.gpu_uuid)
    if gpu['host_index']!=GPU or gpu['uuid']!=UUID:raise ValueError('Diagnostic bound to healthy GPU1 only')
    environment=require_replication_environment(torch);repair_uuid_rng(torch)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if screen.original.sha(screen.original.DATASET)!=screen.original.DATA_SHA:raise ValueError('Feature version changed')
    source=json.loads((args.source/'SUMMARY.json').read_text())
    if source['status']!='complete' or source['config']!=expected_config(args.seed):
        raise ValueError('Need completed fixed small-LR Flat source')
    if args.output.exists():raise FileExistsError('Frozen diagnostic output already exists')
    args.output.mkdir(parents=True)
    trainer=screen.original.load_historical(screen.MODEL_SOURCE)
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_meld_fixed_rates_20261018.run import selected_result
    from experiments.osram_meld_small_lr_readout_20261018.evaluate import evaluate_rate
    config=trainer.TrainConfig(**source['config'])
    _,_,test,ad,td,vd=get_loaders(audio_root='MELD-pickle',text_root='MELD-pickle',
        video_root='MELD-pickle',dataset='MELD',num_folder=1,batch_size=config.batch_size,num_workers=0,
        seed=config.seed,validation_fraction=config.validation_fraction,evaluation_protocol=config.evaluation_protocol)
    dimensions=(ad,td,vd);model=_build_model(config,dimensions,torch.device('cuda:0'))
    summary=dict(status='running',training=False,seed=args.seed,config=source['config'],source=str(args.source),
        source_summary_sha256=screen.original.sha(args.source/'SUMMARY.json'),source_manifest=manifest,
        numerical_commit=screen.original.HISTORICAL,data_sha256=screen.original.DATA_SHA,
        environment=environment,gpu=gpu,pid=os.getpid(),started_unix=time.time(),rates={})
    write_json(args.output/'SUMMARY.json',summary)
    try:
        for i in range(8):
            rate=str(i/10);checkpoint=args.source/f'best_miss_0p{i}.pt'
            original=selected_result(args.source,i/10)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True)
            frozen=state_digest(model)
            reference=state['selection_artifact']
            # Crucial: do not use independently resampled final predictions_miss!
            with np.load(args.source/reference['path'],allow_pickle=False) as saved:
                result,artifacts=evaluate_rate(model,test[config.fold-1],config,dimensions,saved,trainer)
            for metric in ('ACC','UA','WF1'):
                if abs(result['metrics']['LBG'][metric]-original[metric])>1e-9:
                    raise ValueError('Full replay differs from exact selected Test-best score')
            if frozen!=state_digest(model):raise ValueError('Diagnostic modified model weights')
            np.savez_compressed(args.output/f'predictions_miss_0p{i}.npz',**artifacts)
            summary['rates'][rate]=dict(result,epoch=state['epoch'],checkpoint_sha256=original['checkpoint_sha256'],
                selected_artifact_sha256=reference['sha256'],mask_sha256=reference['availability_sha256'])
            write_json(args.output/'SUMMARY.json',summary)
            print('seed',args.seed,'rate',rate,result['metrics'],flush=True)
        for name,rates in [('mean',list(summary['rates'])),('high',['0.5','0.6','0.7'])]:
            summary[name]={c:{k:sum(summary['rates'][r]['metrics'][c][k] for r in rates)/len(rates)
                for k in ('ACC','UA','WF1','CE')} for c in CONDITIONS}
        summary.update(status='complete',finished_unix=time.time())
        (args.output/'RESULT.md').write_text(render({str(args.seed):summary}))
    except BaseException as error:
        summary.update(status='failed',error=repr(error));raise
    finally:write_json(args.output/'SUMMARY.json',summary)


class ReadoutScheduler(Scheduler):
    def __init__(self,output):
        super().__init__(output/'queue',gpus=(GPU,),per_gpu=1,code_root=ROOT,
            smoke_costs={'Flat':dict(peak_memory_mib=4096,disk_bytes=100*2**20)})
    def admit(self,job):
        source=SOURCES[job['config']['seed']];path=source/'SUMMARY.json'
        if not path.exists():return None
        state=json.loads(path.read_text())
        if state['status']=='failed':
            self.failure(job,'Training source failed; no frozen diagnostic performed');return None
        if state['status']!='complete':return None
        return super().admit(job)
    def command(self,job,config_path,gpu_uuid):
        seed=job['config']['seed']
        return [self.python,'-u',str(Path(__file__)),'--worker','--seed',str(seed),
            '--source',str(SOURCES[seed]),'--output',job['output'],'--gpu-uuid',gpu_uuid]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--seed',type=int,choices=(66,67,68));p.add_argument('--source',type=Path)
    p.add_argument('--gpu-uuid');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.worker:
        if any(getattr(args,k) is None for k in ('seed','source','gpu_uuid')):p.error('Missing worker arguments')
        worker(args);return
    check_sources()
    if args.output.exists() and not args.resume:raise FileExistsError('Existing queue; inspect before --resume')
    scheduler=ReadoutScheduler(args.output)
    if scheduler.gpu_snapshot()[GPU]['uuid']!=UUID:raise ValueError('GPU1 identity changed')
    write_json(args.output/'RUNNER.json',dict(status='running',pid=os.getpid(),training=False,
        seeds=[66,67,68],source_paths={str(s):str(v) for s,v in SOURCES.items()}))
    jobs=[dict(id=f'D{s}',method='Flat',config=expected_config(s),output=str(args.output/f'seed_{s}')) for s in SOURCES]
    completed=scheduler.run_jobs(jobs)
    rows={str(row['config']['seed']):row['summary'] for row in completed if row['status']=='complete'}
    result=dict(status='complete' if len(rows)==3 else 'failed',training=False,seeds=rows,
        failures={row['id']:row.get('failures') for row in completed if row['status']!='complete'})
    write_json(args.output/'RESULTS.json',result)
    (args.output/'RESULT.md').write_text(render(rows))
    write_json(args.output/'RUNNER.json',dict(status=result['status'],pid=os.getpid(),training=False))


if __name__=='__main__':main()
