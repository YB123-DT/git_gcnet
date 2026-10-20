"""Exactly21 approved jobs on one healthy GPU, with measured concurrent admission."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import (
    BAD_UUID, devices, make_job, launch, poll, validate_summary, recover_jobs)
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib, write

MODULE = 'experiments.osram_source_role_direct_cross_20261020.worker'
PYTHONS = {'CMUMOSEI':'/data2/yb/reproduction_workspace/envs/s0/bin/python',
    'MELD':'/data2/yb/conda_envs/cmerc-repro/bin/python',
    'IEMOCAPSix':'/data2/yb/conda_envs/cmerc-repro/bin/python'}


def job(output,sha,uuid,dataset,seed,fold,phase='train'):
    value=make_job(MODULE,output,sha,uuid,'joint',seed,dataset,fold=fold,phase=phase)
    value['command'][0]=PYTHONS[dataset]
    value['command'] += ['--dataset',dataset]
    value['checkpoint_selection']='test-oracle-per-rate'
    return value


def plan_jobs(output,source_commit,uuid):
    if not uuid.startswith('GPU-') or uuid==BAD_UUID: raise ValueError('Healthy GPU UUID required')
    return [job(Path(output)/'runs'/dataset/f'fold_{fold}'/f'seed_{seed}'/'direct',source_commit,uuid,dataset,seed,fold)
        for seed in (66,67,68) for dataset,folds in (('CMUMOSEI',(1,)),('MELD',(1,)),('IEMOCAPSix',range(1,6))) for fold in folds]


def admission_limit(total,used,peak,maximum):
    if not math.isfinite(peak) or peak<=0 or maximum<1: raise ValueError('Positive measured peak and concurrency required')
    return max(0,min(maximum,int((total-used-2048)//(1.5*peak))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--max-parallel',type=int,default=6)
    parser.add_argument('--poll-seconds',type=float,default=10.)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.max_parallel<=8 or args.poll_seconds<=0: raise ValueError('Concurrency1..8 and positive poll interval required')
    jobs=plan_jobs(args.output,args.source_commit,args.gpu_uuid)
    device=devices([args.gpu_uuid])[args.gpu_uuid]
    if args.output.exists() and not args.resume: raise FileExistsError('Fresh independent queue required')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'QUEUE.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'QUEUE.json'
    if args.resume:
        state=json.loads(path.read_text())
        if state['source_commit']!=args.source_commit or state['gpu_uuid']!=args.gpu_uuid:
            raise ValueError('Resume source/GPU identity differs')
        if state['status']=='complete': raise ValueError('Completed queue cannot repeat')
        expected=[(j['dataset'],j['seed'],j['fold'],j['output']) for j in jobs]
        if expected!=[(j['dataset'],j['seed'],j['fold'],j['output']) for j in state['jobs']]: raise ValueError('Resume plan differs')
        recover_jobs(dict(jobs=state['jobs']+state['smokes']))
    else:
        state=dict(status='preflight',source_commit=args.source_commit,pid=os.getpid(),
            gpu_uuid=args.gpu_uuid,physical_gpu=device['index'],max_parallel=args.max_parallel,
            jobs=jobs,smokes=[job(args.output/'smoke'/d,args.source_commit,args.gpu_uuid,d,66,5 if d=='IEMOCAPSix' else 1,'smoke')
                for d in ('CMUMOSEI','MELD','IEMOCAPSix')],started_at=time.time(),
            experiment_label='INTERNAL TEST-ORACLE SCREEN',no_other_jobs_killed=True,
            resource_policy='Measured peak x1.5 plus2GiB card margin;RAM>=64GiB;no batch/dimension changes')
    children={}
    def record():
        state['updated_at']=time.time();write(path,state)
    try:
        if shutil.disk_usage(args.output).free<48*2**30: raise RuntimeError('Require48GiB free; preserve all old checkpoints')
        for python in set(PYTHONS.values()):
            if not Path(python).is_file(): raise FileNotFoundError(python)
        record()
        # Each dataset has different dtype/runtime/sequence lengths. Gate each
        # separately, using the very same immutable implementation as training.
        for smoke in state['smokes']:
            if smoke['status']=='complete': continue
            if smoke['status']!='pending': raise RuntimeError('Smoke not recoverable automatically')
            while available_gib()<64 or device['total_mib']-devices([args.gpu_uuid])[args.gpu_uuid]['used_mib']<12288:
                state['status']='waiting_for_smoke_resources';record();time.sleep(args.poll_seconds)
            state['status']='smoke';process=launch(smoke);record()
            while not poll(process,smoke): time.sleep(min(5,args.poll_seconds))
            record()
            if smoke['status']!='complete': raise RuntimeError('Dataset smoke failed: '+smoke['dataset'])
        peaks={s['dataset']:float(validate_summary(s)['peak_memory_mib']) for s in state['smokes']}
        if any(not math.isfinite(p) or p<=0 for p in peaks.values()): raise ValueError('Invalid measured memory peak')
        state['smoke_peaks_mib']=peaks
        while True:
            for index,process in list(children.items()):
                if poll(process,state['jobs'][index]): children.pop(index)
            failed=[j for j in state['jobs'] if j['status']=='failed']
            pending=[j for j in state['jobs'] if j['status']=='pending']
            if not children and (failed or not pending): break
            state.update(status='draining_after_failure' if failed else 'training',available_ram_gib=available_gib())
            if not failed and available_gib()>=64:
                for index,j in enumerate(state['jobs']):
                    if j['status']!='pending' or len(children)>=args.max_parallel: continue
                    current=devices([args.gpu_uuid])[args.gpu_uuid]
                    peak=peaks[j['dataset']]
                    if admission_limit(current['total_mib'],current['used_mib'],peak,args.max_parallel)<1: continue
                    # Include not-yet-allocated child peaks to avoid a stale
                    # nvidia-smi sample oversubscribing during dataset loading.
                    usage=process_usage(args.gpu_uuid)
                    reserved=sum(max(0,1.5*peaks[c['dataset']]-usage.get(c['pid'],0.))
                        for c in state['jobs'] if c['status']=='running')
                    if current['total_mib']-current['used_mib']-reserved<1.5*peak+2048: continue
                    if available_gib()<64: break
                    children[index]=launch(j);record();time.sleep(min(3,args.poll_seconds))
            if not children and pending: state['status']='waiting_for_resources'
            record();time.sleep(args.poll_seconds)
        if any(j['status']!='complete' for j in state['jobs']): raise RuntimeError('One or more approved jobs failed; see preserved logs')
        from .report import write_report
        write_report(args.output,state)
        state.update(status='complete',completed_at=time.time());record()
    except BaseException as error:
        state.update(status='failed',error=repr(error),surviving_children=[p.pid for p in children.values()]);record();raise


def process_usage(uuid):
    """Reserve unallocated peaks of loading children, not just visible CUDA use."""
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    result={}
    for line in rows.strip().splitlines():
        gpu,pid,memory=(part.strip() for part in line.split(','))
        if gpu==uuid: result[int(pid)]=float(memory)
    return result


if __name__=='__main__': main()
