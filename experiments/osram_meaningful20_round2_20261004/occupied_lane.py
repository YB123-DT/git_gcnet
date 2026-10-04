"""One explicitly busy-allowed lane; reuses an already-passed immutable CUDA smoke."""
from __future__ import annotations
import argparse
import fcntl
import math
import os
from pathlib import Path
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


def immutable_modules(snapshot):
    snapshot = Path(snapshot).resolve()
    sys.path.insert(0,str(snapshot))
    from experiments.osram_meaningful20_20261003 import manifest, preflight, queue
    from experiments.osram_meaningful20_round2_20261004 import run, queue as round_queue, manifest as round_manifest
    for module in (manifest,preflight,queue,run,run.shared,round_queue,round_manifest):
        if not Path(module.__file__).resolve().is_relative_to(snapshot):
            raise ValueError('Controller imported mutable source instead of the specified snapshot')
    return manifest,preflight,queue,run,round_queue,round_manifest


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
    return result


def launch(args):
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
    evidence = m.read(check/'profile.json')
    if (evidence.get('status')!='passed' or evidence.get('candidate')!=args.candidate
            or str(evidence.get('gpu_index'))!=args.gpu or evidence.get('gpu_uuid')!=args.gpu_uuid
            or evidence.get('data_manifest_sha256')!=m.sha(args.data_manifest)
            or evidence.get('reference_config_sha256')!=m.sha(args.reference_root/'seed_66/config.json')):
        raise ValueError('A matching passed CUDA profile is required; this lane never reruns smoke')
    ready_path = args.root/'readiness'/args.candidate/'READY.json'
    ready = dict(status='ready',candidate=args.candidate,snapshot_root=str(args.snapshot),
        design_sha256=card['design_sha256'],source_sha256=source['source_sha256'],
        cpu=dict(command=args.cpu_command,returncode=0,log=str(args.cpu_log),sha256=m.sha(args.cpu_log)),
        cuda=dict(command=evidence['command'],returncode=0,log=evidence['log'],sha256=evidence['log_sha256'],
                  gpu_index=args.gpu,gpu_uuid=args.gpu_uuid),profile=evidence['profile'])
    p.validate_readiness(ready,candidate=args.candidate,design_sha256=card['design_sha256'],source_sha256=source['source_sha256'])
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/f'occupied_gpu{args.gpu}.lock').open('a') as lane:
        fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output = args.root/'runs'/args.candidate/'seed_66'
        key = args.candidate+':66'
        run_id = uuid.uuid4().hex
        directory = args.root/'occupied_lanes'/args.candidate
        queue_path = args.root/'QUEUE.json'
        with (args.root/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            round_queue.bind_manifest_version(args)
            state = m.read(queue_path)
            if key in state['jobs'] or output.exists() or directory.exists():
                raise ValueError('Existing job/output/attempt: refuse duplicate occupied-lane launch')
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
                cuda_profile_sha256=m.sha(check/'profile.json'),resources_before_launch=resources[args.gpu],
                input_sha256={str(path):m.sha(path) for path in (args.manifest,args.baseline_audit,args.data_manifest,args.cpu_log,ready_path)},
                admission_override='Explicit user request: busy GPUs allowed; no utilization falsification',created_at=m.now())
            m.write(record_path,record)
            log = directory/'train.log'
            job = dict(status='launch_intent',candidate=args.candidate,seed=66,output=str(output),run_id=run_id,
                gpu_index=args.gpu,gpu_uuid=args.gpu_uuid,command=command,attempt=1,launched_at=m.now(),
                snapshot_sha256=record['snapshot_sha256'],profile=evidence['profile'],log=str(log),
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
