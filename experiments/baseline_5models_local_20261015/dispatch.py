"""Persistent, restartable local two-GPU queue; no shell/network execution."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import hashlib

from runtime import ROOT,RUN_ROOT,GPUS,PYTHON,enforce_local,atomic_json,code_signature,gpu_free,now,valid_result


def alive(pid,run_dir):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        if stat.split(') ')[1].split()[0]=='Z':return False
        return str(run_dir).encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,ProcessLookupError):return False


def snapshot():
    files=code_signature();key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    target=RUN_ROOT/'code_snapshots'/key
    if not (target/'snapshot.json').exists():
        target.mkdir(parents=True,exist_ok=True)
        for rel in files:
            p=target/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
        # Config metadata consumed by MMIN wrapper; source/checksum metadata for every model.
        for p in (ROOT/'adapters').rglob('*.json'):
            if 'upstream' in p.parts:continue
            dst=target/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        if code_signature(target) != files:
            raise RuntimeError('Source changed during snapshot; retry after edits finish')
        provenance=[]
        for repo in [ROOT.parents[1],Path('/data2/yb/paper/git_gcnet'),Path('/data2/yb/paper/GCNet_TPAMI/.worktrees/comp-meld'),Path('/data2/yb/paper/GCNet_TPAMI/.worktrees/ecerc-features'),ROOT/'sources/MoMKE',Path('/data2/yb/paper/SDR-GNN_official_2024')]:
            def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True)
            try:provenance.append(dict(path=str(repo),commit=git('rev-parse','HEAD').strip(),branch=git('branch','--show-current').strip(),remotes=git('remote','-v'),status=git('status','--short')))
            except subprocess.CalledProcessError:continue
        for suffix in ('*.patch','*.diff'):
            for p in (ROOT/'adapters').rglob(suffix):
                if 'upstream' in p.parts:continue
                dst=target/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        atomic_json(target/'snapshot.json',dict(sha256=key,files=files,created_at=now(),repositories=provenance))
    return target,key


def launch(task,uuid,code):
    run=Path(task['run_dir']);run.mkdir(parents=True,exist_ok=True)
    if read_json(task['config_path']) != task['config']:raise RuntimeError('Queued config file differs from registry')
    previous=read_json(run/'launch.json',{})
    manifest=read_json(run/'manifest.json',{})
    if any((run/name).exists() for name in ('launch.json','manifest.json','last.pt')):
        if (run/'config.json').exists() and read_json(run/'config.json') != task['config']:
            raise RuntimeError('Resume config mismatch')
        if (run/'launch.json').exists() and previous.get('config') != task['config']:
            raise RuntimeError('Resume launch config mismatch')
        if (run/'manifest.json').exists():
            recorded=manifest.get('config',{})
            if any(recorded.get(k) != v for k,v in task['config'].items()) or set(recorded)-set(task['config'])-{'device','seq_len'}:
                raise RuntimeError('Resume manifest config mismatch')
        snapshots=[value.get('code_snapshot') for value in (previous,manifest) if value]
        if not snapshots or not all(s and s == snapshots[0] for s in snapshots):
            raise RuntimeError('Original resume snapshot identity missing or inconsistent')
        code=Path(snapshots[0])
        if not (code/'snapshot.json').exists():raise RuntimeError('Original resume snapshot unavailable')
    args=[PYTHON,str(code/'train_one.py'),'--config',task['config_path'],'--run-dir',str(run)]
    env=os.environ.copy();env.update(BASELINE_RUN_ROOT=str(RUN_ROOT),CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(RUN_ROOT/'tmp'))
    (RUN_ROOT/'tmp').mkdir(exist_ok=True,parents=True)
    atomic_json(run/'launch.json',dict(command=args,gpu_uuid=uuid,hostname=socket.gethostname(),code_snapshot=str(code),config=task['config'],started_at=now()))
    # Wrapper records the actual exit code even if dispatcher itself is restarted.
    args=[PYTHON,str(code/'supervise.py'),'--run-dir',str(run),'--',*args]
    log=(run/'run.log').open('a')
    process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=run)
    log.close()
    atomic_json(run/'supervisor.json',dict(pid=process.pid,gpu_uuid=uuid,command=args,started_at=now()))
    return process


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def budget_decision(tasks,gates):
    pilots=[t for t in tasks if t['phase']=='budget_pilot']
    if gates.get('MoMKE_causal_MELD',{}).get('ready'):return
    if not all(valid_result(t['run_dir']) for t in pilots):return
    # Precommitted train-only rule. No test field is read by this function.
    decisions=[]
    for mr in (.3,.6):
        means={}
        for t in pilots:
            if t['config']['missing_rate']!=mr:continue
            c=t['config'];hist=[json.loads(line) for line in (Path(t['run_dir'])/'history.jsonl').read_text().splitlines()]
            losses=[r['train_loss'] for r in hist if r['epoch']>c['stage_epoch']][-10:]
            means[c['epochs']]=sum(losses)/len(losses)
        decisions.append(means[200]<=means[100]*.98)
    epochs,stage=(200,100) if any(decisions) else (100,50)
    note=dict(selected_epochs=epochs,stage_epoch=stage,rule='Choose 200/100 if either MR shows >=2% lower mean final-10 Stage2 TRAIN loss, else 100/50; no test values read',improvement_flags=decisions,at=now())
    atomic_json(ROOT/'audit/momke_budget_decision.json',note)
    registry=read_json(ROOT/'experiment_registry.json')
    for task in registry['tasks']:
        if task['phase']=='formal' and task['config']['model']=='MoMKE' and task['config']['dataset']=='MELD':
            if (Path(task['run_dir'])/'launch.json').exists():raise RuntimeError('MoMKE budget already launched before decision')
            task['config'].update(epochs=epochs,stage_epoch=stage);atomic_json(task['config_path'],task['config'])
            if task['config']['model_version']=='causal' and task['config']['seed']==66 and task['config']['missing_rate'] in (.3,.6):
                pilot=next(p for p in pilots if p['config']==task['config'])
                task['run_dir']=pilot['run_dir'];task['reused_from_budget_pilot']=pilot['id']
    atomic_json(ROOT/'experiment_registry.json',registry)
    for version in ('causal','original'):
        gate=f'MoMKE_{version}_MELD'
        if gates.get(gate,{}).get('smoke_passed') or (version=='causal' and gates.get('MoMKE_causal_MELD_pilot',{}).get('ready')):
            gates[gate]=dict(ready=True,smoke_passed=True,budget=note)
    atomic_json(ROOT/'gates.json',gates)
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args();enforce_local()
    RUN_ROOT.mkdir(exist_ok=True,parents=True)
    (RUN_ROOT/'logs').mkdir(exist_ok=True)
    lock=(RUN_ROOT/'dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        atomic_json(RUN_ROOT/'dispatcher.json',dict(state='RUNNING',pid=os.getpid(),hostname=socket.gethostname(),gpu_uuids=GPUS,started_at=now()))
        children={};verified=set();last_aggregate=0.;previous_completed=-1
        while True:
            registry=read_json(ROOT/'experiment_registry.json');tasks=sorted(registry['tasks'],key=lambda t:t['priority']);gates=read_json(ROOT/'gates.json',{})
            active=[];failed=[];waiting=[]
            for t in tasks:
                d=Path(t['run_dir']);sup=read_json(d/'supervisor.json',{});state=read_json(d/'status.json',{})
                if t['id'] in verified:continue
                child=read_json(d/'child.json',{})
                if alive(sup.get('pid',0),d) or alive(child.get('pid',0),d):active.append(t);continue
                if (d/'launch.json').exists():
                    held=False
                    for name in ('run.lock','supervisor.lock'):
                        with (d/name).open('a') as probe:
                            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            except BlockingIOError:held=True
                    if held:active.append(t);continue
                if sup.get('pid') in children:
                    children.pop(sup['pid']).wait()
                if valid_result(d):
                    if read_json(d/'config.json') != t['config'] or read_json(t['config_path']) != t['config']:raise RuntimeError('Refusing incompatible completed result: '+t['id'])
                    verified.add(t['id']);continue
                exit_state=read_json(d/'exit.json',{})
                if exit_state or state.get('state')=='FAILED':
                    failed.append(t);continue
                waiting.append(t)
            if budget_decision(tasks,gates):continue
            occupied={read_json(Path(t['run_dir'])/'launch.json',{}).get('gpu_uuid') for t in active}
            for uuid in GPUS:
                if len(active)>=2 or uuid in occupied or not gpu_free(uuid):continue
                candidate=None
                for t in waiting:
                    if not gates.get(t['gate'],{}).get('ready'):continue
                    # Avoid occupying both GPUs while the identical MMIN teacher is trained once.
                    c=t['config']
                    if c['model']=='MMIN' and any(a['config']['model']=='MMIN' and all(a['config'][k]==c[k] for k in ('dataset','fold','seed')) for a in active):continue
                    candidate=t;break
                if candidate:
                    code,key=snapshot();proc=launch(candidate,uuid,code);children[proc.pid]=proc
                    active.append(candidate);waiting.remove(candidate);occupied.add(uuid)
                    print(f'LAUNCH pid={proc.pid} gpu={uuid} task={candidate["id"]} code={key}',flush=True)
            for task in tasks:
                task['status']='COMPLETED' if task['id'] in verified else 'RUNNING' if task in active else 'FAILED' if task in failed else 'PENDING' if gates.get(task['gate'],{}).get('ready') else 'WAITING_GATE'
            atomic_json(ROOT/'experiment_registry.json',registry)
            status=dict(state='RUNNING' if active else 'WAITING' if waiting else 'FINISHED',pid=os.getpid(),at=now(),verified_completed=len(verified),failed=len(failed),remaining=len(waiting),running=[dict(id=t['id'],run_dir=t['run_dir'],training_pid=read_json(Path(t['run_dir'])/'child.json',{}).get('pid'),**read_json(Path(t['run_dir'])/'supervisor.json',{})) for t in active],gated=sum(not gates.get(t['gate'],{}).get('ready') for t in waiting))
            atomic_json(RUN_ROOT/'dispatcher.json',status)
            if time.monotonic()-last_aggregate>300 or len(verified)!=previous_completed:
                with (RUN_ROOT/'logs/aggregate.log').open('a') as log:
                    result=subprocess.run([PYTHON,str(ROOT/'aggregate_results.py')],stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:print('AGGREGATE_FAILED see logs/aggregate.log',flush=True)
                last_aggregate=time.monotonic();previous_completed=len(verified)
            if not active and not waiting:
                print(json.dumps(status),flush=True);break
            if args.once:break
            time.sleep(15)
    finally:
        lock.close()

if __name__=='__main__':main()
