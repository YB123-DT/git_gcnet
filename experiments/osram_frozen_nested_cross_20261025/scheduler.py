"""Finite56-job, seed66 frozen Nested extension; no baseline retraining."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_frozen_nested_gcnet_20261025.scheduler import parse_gpu_map
from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits
from experiments.osram_signature_stage2_extension_20261025.scheduler import finish
from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save
from .protocol import source_root

MODULE = 'experiments.osram_frozen_nested_cross_20261025.run'
DATASETS = ('CMUMOSEI','MELD','IEMOCAPSix')
RATES = tuple(str(i/10) for i in range(8))


def jobs_for(phase):
    if phase not in ('smoke','train'):raise ValueError('Only smoke/train')
    specs = [(d,5 if d == 'IEMOCAPSix' else 1,'0.7') for d in DATASETS] if phase == 'smoke' else [
        (d,f,r) for r in RATES for d in DATASETS for f in (range(1,6) if d == 'IEMOCAPSix' else (1,))]
    return [dict(id=f'{d}_66_fold{f}_{r}',dataset=d,seed=66,fold=f,rate=r,status='pending') for d,f,r in specs]


def launch(args,job,uuid,phase):
    output = args.root/phase/job['id']
    recovering = output.exists()
    if recovering and (not args.resume or not (output/'last.pt').exists()):
        raise ValueError('Existing output needs explicit compatible full-state recovery')
    output.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,'-u','-m',MODULE,'--dataset',job['dataset'],'--seed','66',
        '--fold',str(job['fold']),'--rate',job['rate'],'--output',str(output),
        '--gpu-uuid',uuid,'--source-commit',args.source_commit]
    if phase == 'smoke':command.append('--smoke')
    if recovering:command.append('--resume')
    cwd = str(source_root(job['dataset']))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (output/'worker.log').open('a',buffering=1) as stream:
        child = subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    job.update(status='running',pid=child.pid,command=command,cwd=cwd,source_commit=args.source_commit,
        output=str(output),gpu_uuid=uuid,log=str(output/'worker.log'),started_at=now(),resume=recovering)
    return child


def process_matches(job):
    try:
        proc = Path('/proc')/str(job['pid'])
        values = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        return (proc.stat().st_uid == os.getuid() and values == job['command'] and MODULE in values
            and job['source_commit'] in values and (proc/'cwd').resolve() == Path(job['cwd']).resolve())
    except (KeyError,OSError):return False


def reconcile(jobs,phase,source):
    import torch
    from .protocol import budget
    for job in jobs:
        if job['status'] == 'pending' or (job['status'] == 'running' and process_matches(job)):continue
        if 'output' not in job:
            job.update(status='pending');continue
        if (Path(job['output'])/'SUMMARY.json').exists():
            finish(job,0,phase)
            if job['status'] != 'failed':continue
        try:
            saved = torch.load(Path(job['output'])/'last.pt',map_location='cpu',weights_only=False)
            identity = saved['identity']
            assert identity['source_commit'] == source and identity['method'] == 'nested_gnn_rooted_evidence'
            assert identity['dataset'] == job['dataset'] and identity['seed'] == 66
            assert identity['fold'] == job['fold'] and identity['evaluation_rate'] == job['rate']
            assert identity['smoke'] == (phase == 'smoke')
            assert 0 < saved['epoch'] <= (2 if phase == 'smoke' else budget(job['dataset']))
            assert all(k in saved for k in ('block','optimizer','rng','identity','history','epoch','best'))
            job.update(status='pending',recovery='compatible block/Adam/RNG/history')
        except (OSError,KeyError,ValueError,RuntimeError,EOFError,AssertionError):
            job.update(status='failed',error='No validated completion/full recovery state')


def refresh_measurements(jobs,peaks):
    """Observe a real full missing-rate cycle before multiplying concurrency."""
    coverage = {d:set() for d in DATASETS}
    for job in jobs:
        if job['status'] not in ('running','complete') or 'output' not in job:continue
        root = Path(job['output']);dataset = job['dataset']
        summary_path = root/'SUMMARY.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            for field,name in [('peak_memory_mib','gpu_mib'),('peak_rss_gib','rss_gib')]:
                measured = float(summary.get(field,0.))
                if not math.isfinite(measured):raise ValueError('Nonfinite actual resource measurement')
                peaks[dataset][name] = max(peaks[dataset][name],measured)
        history_path = root/'history.json'
        if history_path.exists() and len(coverage[dataset]) < 8:
            for row in json.loads(history_path.read_text()):coverage[dataset].update(row['training_rates'])
    return {d:values == {i/10 for i in range(8)} for d,values in coverage.items()}


def run_phase(args,state,phase,peaks=None):
    jobs = state.setdefault(phase,jobs_for(phase))
    if args.resume:reconcile(jobs,phase,args.source_commit)
    if any(j['status'] == 'running' and j.get('gpu_uuid') not in args.gpu_map.values() for j in jobs):
        raise ValueError('Live worker outside requested GPU map')
    children = {}
    while any(j['status'] in ('running','pending') for j in jobs):
        for job in jobs:
            if job['status'] != 'running':continue
            process = children.get(job['id'])
            code = process.poll() if process else (None if process_matches(job) else 0)
            if code is not None:
                finish(job,code,phase);children.pop(job['id'],None)
        if any(j['status'] == 'failed' for j in jobs):
            for job in jobs:
                if job['status'] == 'pending':job.update(status='blocked',error='Worker failure; live jobs preserved')
        else:
            snapshot = resources(args.root,list(args.gpu_map.values()))
            state['latest_resources'] = snapshot
            warmed = refresh_measurements(jobs,peaks) if phase == 'train' else {}
            state['real_cycle_capacity_verified'] = warmed
            pending = next((j for j in jobs if j['status'] == 'pending' and (phase == 'smoke'
                or warmed[j['dataset']] or not any(p['dataset'] == j['dataset']
                    and p['status'] in ('running','complete') for p in jobs))),None)
            if pending:
                for uuid in sorted(args.gpu_map.values(),key=lambda u:sum(
                        j['status'] == 'running' and j.get('gpu_uuid') == u for j in jobs)):
                    peers = [j for j in jobs if j['status'] == 'running' and j.get('gpu_uuid') == uuid]
                    allowed = (not peers and snapshot['gpus'][uuid]['free_mib'] >= 8192
                        and snapshot['ram_available_gib'] >= 32 and snapshot['disk_free_gib'] >= 24
                        if phase == 'smoke' else fits(snapshot,uuid,pending,peers,peaks,args.per_gpu))
                    if allowed:
                        children[pending['id']] = launch(args,pending,uuid,phase);break
        state['updated_at'] = now();save(args.root/'QUEUE.json',state)
        if any(j['status'] in ('pending','running') for j in jobs):time.sleep(5)
    if any(j['status'] != ('smoke_passed' if phase == 'smoke' else 'complete') for j in jobs):
        raise RuntimeError(phase+' contains failed/blocked jobs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-map',type=parse_gpu_map,required=True)
    parser.add_argument('--per-gpu',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.per_gpu <= 4:parser.error('Maximum4taskworkers/card')
    args.root = args.root.resolve();args.root.mkdir(parents=True,exist_ok=args.resume)
    with (args.root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = args.root/'QUEUE.json'
        state = json.loads(path.read_text()) if args.resume and path.exists() else dict(
            source_commit=args.source_commit,status='preflight',started_at=now(),
            expected_train_jobs=56,seed=66,method='nested_gnn_rooted_evidence')
        if state['source_commit'] != args.source_commit or state['method'] != 'nested_gnn_rooted_evidence':
            raise ValueError('Controller source/method recovery mismatch')
        state.update(pid=os.getpid(),command=[sys.executable]+sys.argv,cwd=os.getcwd(),
            python=sys.executable,gpu_map=args.gpu_map,max_per_gpu=args.per_gpu)
        try:
            old = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                for index,uuid in args.gpu_map.items():
                    os.environ['CUDA_VISIBLE_DEVICES'] = uuid;guard_gpu(uuid,index)
            finally:
                if old is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                else:os.environ['CUDA_VISIBLE_DEVICES'] = old
            state['status'] = 'smoke';run_phase(args,state,'smoke')
            peaks = {}
            for job in state['smoke']:
                summary = json.loads(Path(job['summary']).read_text())
                measured = dict(gpu_mib=float(summary['peak_memory_mib']),rss_gib=float(summary['peak_rss_gib']))
                if not all(math.isfinite(v) and v>0 for v in measured.values()):
                    raise ValueError('Positive measured VRAM/RAM required')
                peaks[job['dataset']] = measured
            state.update(status='train',admission_measurements=peaks)
            run_phase(args,state,'train',peaks)
            from .summarize import summarize
            result = summarize(args.root,args.source_commit,state['train'])
            if result['status'] != 'complete':raise RuntimeError('Final saved artifact validation failed')
            state.update(status='complete',completed_at=now());save(path,state)
        except BaseException as error:
            state.update(status='failed',error=repr(error),finished_at=now());save(path,state);raise


if __name__ == '__main__':main()
