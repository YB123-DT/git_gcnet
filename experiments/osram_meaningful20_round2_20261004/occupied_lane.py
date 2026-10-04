"""One explicitly busy-allowed lane; reuses an already-passed immutable CUDA smoke."""
from __future__ import annotations
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def busy_admission(gpu, profile, *, disk_free_gib):
    """Only saturation is waived. Never fabricate or zero the observed utilization."""
    values = [gpu.get('free_mib'), gpu.get('utilization'), gpu.get('temperature'),
              profile.get('peak_mib'), profile.get('artifact_gib'), disk_free_gib]
    if any(not isinstance(value,(int,float)) or not math.isfinite(value) for value in values):
        return False, 'invalid resource measurement'
    peak = profile['peak_mib']
    if peak <= 0 or profile['artifact_gib'] <= 0: return False, 'missing full profile'
    if gpu['temperature'] >= 90: return False, 'GPU temperature'
    if gpu.get('throughput_ratio',1.) < .9: return False, 'measured throughput regression'
    if gpu['free_mib'] < peak + max(2048,.2*peak): return False, 'GPU memory margin'
    if disk_free_gib < profile['artifact_gib'] + 16: return False, 'disk budget including immutable versions'
    return True, 'explicit busy-GPU authorization; memory/temperature/disk checks retained'


def allocated_memory():
    output = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                     '--format=csv,noheader,nounits'],text=True,timeout=8)
    allocations = {}
    for line in output.splitlines():
        if not line.strip(): continue
        gpu_uuid,pid,used = (part.strip() for part in line.split(','))
        used = float(used)
        if not math.isfinite(used) or used < 0: raise ValueError('Invalid live process memory measurement')
        allocations[(gpu_uuid,int(pid))] = used
    return allocations


def group_admission(gpu, profile, active, allocations, *, max_per_gpu, disk_free_gib, global_reserve_mib=2048, live_allocation_floor=False):
    """Free memory already excludes live allocations; reserve only their missing peak."""
    if max_per_gpu not in (1,2,4,11,12): return False,'Only explicit per-GPU caps 1, 2, 4, 11 or 12 are supported'
    if global_reserve_mib != 2048 and not (max_per_gpu == 12 and global_reserve_mib == 1792):
        return False, 'Nondefault reserve is restricted to explicit twelve-job launch'
    same_gpu = [job for job in active if job.get('gpu_uuid')==gpu['uuid']]
    if len(same_gpu) >= max_per_gpu: return False,f'Per-GPU concurrency cap {max_per_gpu}'
    admitted,reason = busy_admission(gpu,profile,disk_free_gib=disk_free_gib)
    if not admitted: return admitted,reason
    pending = 0.
    for job in same_gpu:
        peak = job.get('profile',{}).get('peak_mib')
        if not isinstance(peak,(int,float)) or not math.isfinite(peak) or peak<=0:
            return False,'An active job lacks a finite measured peak; cannot reserve safely'
        used = allocations.get((gpu['uuid'],job.get('pid')),0.)
        if not isinstance(used,(int,float)) or not math.isfinite(used) or used<0:
            return False,'Invalid active process memory measurement'
        if live_allocation_floor:
            peak = max(peak, used)
        pending += max(0.,1.2*peak+512-used)
    required = pending + 1.2*profile['peak_mib'] + 512 + global_reserve_mib
    if gpu['free_mib'] < required:
        return False,f'Pending allocation reserve: need {required:.1f} MiB free, have {gpu["free_mib"]:.1f}'
    return True,f'Admitted {len(same_gpu)+1}/{max_per_gpu}; pending reserve {pending:.1f} MiB'


def immutable_modules(snapshot):
    snapshot = Path(snapshot).resolve()
    sys.path.insert(0,str(snapshot))
    from experiments.osram_meaningful20_20261003 import manifest, preflight, queue
    from experiments.osram_meaningful20_round2_20261004 import run, queue as round_queue, manifest as round_manifest
    for module in (manifest,preflight,queue,run,run.shared,round_queue,round_manifest):
        if not Path(module.__file__).resolve().is_relative_to(snapshot):
            raise ValueError('Controller imported mutable source instead of the specified snapshot')
    return manifest,preflight,queue,run,round_queue,round_manifest


def validate_deferred(record, *, candidate, design_sha256, source_sha256):
    """Explicit user waiver, never represented as a passed CUDA check."""
    import hashlib
    if (record.get('status') != 'cuda_smoke_deferred_by_user'
            or record.get('candidate') != candidate
            or record.get('design_sha256') != design_sha256
            or record.get('source_sha256') != source_sha256):
        raise ValueError('Deferred readiness identity mismatch')
    cpu = record['cpu']
    contents = Path(cpu['log']).read_bytes()
    if cpu.get('returncode') != 0 or hashlib.sha256(contents).hexdigest() != cpu['sha256'] or b'\nOK\n' not in contents:
        raise ValueError('Existing CPU evidence missing')
    profile = record['profile']
    if profile.get('measurement_status') != 'estimated_not_measured' or profile['peak_mib'] <= 0:
        raise ValueError('Deferred run must disclose estimated resources')
    return profile


def train_child(record_path):
    import json
    record = json.loads(Path(record_path).read_text())
    m,p,q,run,_,_ = immutable_modules(record['snapshot'])
    if m.sha(__file__) != record['controller_sha256']: raise ValueError('External controller changed')
    if m.digest_json(m.verify_snapshot(record['snapshot'])['source_sha256']) != record['snapshot_sha256']:
        raise ValueError('Immutable model snapshot changed')
    os.environ['CUDA_VISIBLE_DEVICES'] = record['gpu_uuid']
    os.chdir(record['snapshot'])
    observed = []
    def audited_admission(gpu,profile,*,disk_free_gib):
        result = busy_admission(gpu,profile,disk_free_gib=disk_free_gib)
        observed.append(dict(gpu=dict(gpu),profile=dict(profile),disk_free_gib=disk_free_gib,
                             admitted=result[0],reason=result[1]))
        return result
    original = run.shared.admission
    run.shared.admission = audited_admission
    if record.get('cuda_smoke_deferred_by_user'):
        run.shared.validate_readiness = validate_deferred
    sys.argv = ['experiments.osram_meaningful20_round2_20261004.run',*record['training_argv']]
    try:
        run.main()
    finally:
        run.shared.admission = original
        path = Path(record['output'])/'PROVENANCE.json'
        if path.exists():
            provenance = m.read(path)
            provenance['external_controller'] = dict(path=record['controller'],sha256=record['controller_sha256'],
                record=str(Path(record_path).resolve()),record_sha256=m.sha(record_path),
                override='User explicitly authorized occupied GPUs; only utilization veto waived',
                admission_observations=observed)
            m.write(path,provenance)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key in ('root','snapshot','manifest','reference-root','baseline-audit','data-manifest','cpu-log'):
        result.add_argument('--'+key,type=Path,required=True)
    for key in ('candidate','gpu','gpu-uuid','cpu-command'):
        result.add_argument('--'+key,required=True)
    result.add_argument('--max-per-gpu',type=int,choices=(1,2,4,11,12),default=1,
                        help='Explicit real-job cap on this GPU; default remains one')
    result.add_argument('--defer-cuda-smoke-by-user', action='store_true')
    result.add_argument('--estimated-peak-mib', type=float, default=1700.)
    result.add_argument('--global-reserve-mib', type=int, choices=(1792,2048), default=2048)
    result.add_argument('--live-allocation-floor', action='store_true',
                        help='Reserve observed allocation plus growth margin when old estimates understate usage')
    return result


def launch(args):
    if args.max_per_gpu not in (1,2,4,11,12): raise ValueError('Per-GPU cap must be 1, 2, 4, 11 or 12')
    for key in ('root','snapshot','manifest','reference_root','baseline_audit','data_manifest','cpu_log'):
        setattr(args,key,getattr(args,key).resolve())
    m,p,q,run,round_queue,round_manifest = immutable_modules(args.snapshot)
    source = m.verify_snapshot(args.snapshot)
    accepted = m.read(args.manifest)
    if args.candidate not in round_manifest.validate_round(accepted): raise ValueError('Unaccepted candidate')
    card = next(card for card in accepted['cards'] if card['id']==args.candidate)
    resources = p.query_gpus()
    p.validate_gpu(args.gpu,args.gpu_uuid,{index:row['uuid'] for index,row in resources.items()})
    if not args.cpu_log.is_file() or '\nOK\n' not in args.cpu_log.read_text():
        raise ValueError('The existing shared CPU smoke must have passed')
    check = args.root/'checks'/args.candidate
    if args.defer_cuda_smoke_by_user:
        evidence = dict(status='deferred_by_user', profile=dict(peak_mib=args.estimated_peak_mib,
            artifact_gib=10., measurement_status='estimated_not_measured',
            estimate_basis='Existing NEW40 batch32 CUDA peaks 1454-1594 MiB for six lightweight methods; unmeasured candidate, extra allocation reserve retained'))
    else:
        evidence = m.read(check/'profile.json')
    if not args.defer_cuda_smoke_by_user and (evidence.get('status')!='passed' or evidence.get('candidate')!=args.candidate
            or str(evidence.get('gpu_index'))!=args.gpu or evidence.get('gpu_uuid')!=args.gpu_uuid
            or evidence.get('data_manifest_sha256')!=m.sha(args.data_manifest)
            or evidence.get('reference_config_sha256')!=m.sha(args.reference_root/'seed_66/config.json')):
        raise ValueError('A matching passed CUDA profile is required; this lane never reruns smoke')
    ready_path = args.root/'readiness'/args.candidate/'READY.json'
    ready = dict(status='ready',candidate=args.candidate,snapshot_root=str(args.snapshot),
        design_sha256=card['design_sha256'],source_sha256=source['source_sha256'],
        cpu=dict(command=args.cpu_command,returncode=0,log=str(args.cpu_log),sha256=m.sha(args.cpu_log)),
        cuda=(dict(status='deferred_by_user', reason='User explicitly requested no smoke, launch first')
              if args.defer_cuda_smoke_by_user else
              dict(command=evidence['command'],returncode=0,log=evidence['log'],sha256=evidence['log_sha256'],
                   gpu_index=args.gpu,gpu_uuid=args.gpu_uuid)),profile=evidence['profile'])
    if args.defer_cuda_smoke_by_user:
        ready['status'] = 'cuda_smoke_deferred_by_user'
    validator = validate_deferred if args.defer_cuda_smoke_by_user else p.validate_readiness
    validator(ready,candidate=args.candidate,design_sha256=card['design_sha256'],source_sha256=source['source_sha256'])
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/f'occupied_candidate_{args.candidate}.lock').open('a') as lane:
        fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output = args.root/'runs'/args.candidate/'seed_66'
        key = args.candidate+':66'
        run_id = uuid.uuid4().hex
        directory = args.root/'occupied_lanes'/args.candidate
        queue_path = args.root/'QUEUE.json'
        with (args.root/f'occupied_gpu{args.gpu}.admission.lock').open('a') as gpu_lock, \
             (args.root/'queue.lock').open('a') as lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            fcntl.flock(lock,fcntl.LOCK_EX)
            round_queue.bind_manifest_version(args)
            state = m.read(queue_path)
            if key in state['jobs'] or output.exists() or directory.exists():
                raise ValueError('Existing job/output/attempt: refuse duplicate occupied-lane launch')
            active = []
            for name,previous in state['jobs'].items():
                if previous['status']=='complete': continue
                current = q.reconcile_job(previous)
                state['jobs'][name] = current
                if current['status'] in ('running','launch_intent','inspection_pending'):
                    active.append(current)
            resources = p.query_gpus()
            p.validate_gpu(args.gpu,args.gpu_uuid,{index:row['uuid'] for index,row in resources.items()})
            if any(str(job.get('gpu_index'))==args.gpu and job.get('gpu_uuid')!=args.gpu_uuid for job in active):
                raise ValueError('Active job GPU index/UUID differs from live mapping')
            allocations = allocated_memory()
            disk_free = shutil.disk_usage(args.root).free/1024**3 - q.reserved_disk_gib(active)
            admitted,reason = group_admission(resources[args.gpu],evidence['profile'],active,allocations,
                max_per_gpu=args.max_per_gpu,disk_free_gib=disk_free,global_reserve_mib=args.global_reserve_mib,
                live_allocation_floor=args.live_allocation_floor)
            if not admitted: raise RuntimeError('Occupied-lane admission rejected before job creation: '+reason)
            if ready_path.exists():
                existing = m.read(ready_path)
                if existing != ready: raise ValueError('Existing readiness differs; preserve and inspect it')
            else: m.write(ready_path,ready)
            training = ['--candidate',args.candidate,'--seed','66','--output',str(output),
                '--manifest',str(args.manifest),'--manifest-sha256',m.sha(args.manifest),
                '--snapshot',str(args.snapshot),'--readiness',str(ready_path),
                '--reference-root',str(args.reference_root),'--baseline-audit',str(args.baseline_audit),
                '--data-manifest',str(args.data_manifest),'--baseline-audit-sha256',m.sha(args.baseline_audit),
                '--data-manifest-sha256',m.sha(args.data_manifest),'--readiness-sha256',m.sha(ready_path),
                '--gpu',args.gpu,'--gpu-uuid',args.gpu_uuid,'--run-id',run_id]
            controller = Path(__file__).resolve()
            record_path = directory/'CONTROLLER.json'
            command = [sys.executable,'-u',str(controller),'--train-only',str(record_path)]
            record = dict(controller=str(controller),controller_sha256=m.sha(controller),snapshot=str(args.snapshot),
                snapshot_sha256=m.digest_json(source['source_sha256']),code_commit=source['code_commit'],
                candidate=args.candidate,seed=66,gpu_index=args.gpu,gpu_uuid=args.gpu_uuid,output=str(output),
                run_id=run_id,training_argv=training,command=command,invocation=sys.argv,
                cuda_profile_sha256=None if args.defer_cuda_smoke_by_user else m.sha(check/'profile.json'),
                cuda_smoke_deferred_by_user=args.defer_cuda_smoke_by_user,
                resources_before_launch=resources[args.gpu],
                max_per_gpu=args.max_per_gpu,launch_admission=dict(reason=reason,disk_free_after_reservations_gib=disk_free,
                    global_reserve_mib=args.global_reserve_mib,
                    live_allocation_floor=args.live_allocation_floor,
                    active_run_ids=[job.get('run_id') for job in active],
                    process_allocations_mib={f'{gpu_uuid}:{pid}':used for (gpu_uuid,pid),used in allocations.items()}),
                input_sha256={str(path):m.sha(path) for path in (args.manifest,args.baseline_audit,args.data_manifest,args.cpu_log,ready_path)},
                admission_override='Explicit user request: busy GPUs allowed; no utilization falsification',created_at=m.now())
            m.write(record_path,record)
            log = directory/'train.log'
            job = dict(status='launch_intent',candidate=args.candidate,seed=66,output=str(output),run_id=run_id,
                gpu_index=args.gpu,gpu_uuid=args.gpu_uuid,command=command,attempt=1,launched_at=m.now(),
                snapshot_sha256=record['snapshot_sha256'],profile=evidence['profile'],log=str(log),
                max_per_gpu=args.max_per_gpu,
                external_controller=str(record_path),external_controller_sha256=m.sha(record_path))
            state['jobs'][key]=job
            m.write(queue_path,state)
            environment = dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,PYTHONPATH=str(args.snapshot),
                               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            try:
                with log.open('x') as stream:
                    process = subprocess.Popen(command,cwd=args.snapshot,env=environment,
                                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(q.process_identity(process.pid) or {'pid':process.pid})
                job['status']='running'
                m.write(queue_path,state)
            except BaseException as error:
                job.update(status='inspection_pending',detail=repr(error)); m.write(queue_path,state)
                raise
        print(f'{args.candidate}: training PID {process.pid} on physical GPU{args.gpu}; log {log}',flush=True)
        code = process.wait()
        status,detail = run.completion_status(output)
        with (args.root/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state = m.read(queue_path)
            if state['jobs'].get(key,{}).get('run_id')!=run_id: raise ValueError('Queue identity changed')
            state['jobs'][key].update(process_exit_code=code,status='complete' if code==0 and status=='complete' else 'failed',
                                      detail=detail,finished_at=m.now())
            m.write(queue_path,state)
        if code or status!='complete': raise RuntimeError(f'Occupied lane failed: exit={code}; artifacts={status}; {detail}')


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--train-only': train_child(Path(sys.argv[2]))
    else: launch(parser().parse_args())
