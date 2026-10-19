"""Low-RSS, stdlib-only admission for the two approved remote experiment queues.

This process may wait under tmux; waiting is NOT model training or a GPU claim.
It never clears caches/swap, kills processes, changes batches, or retries jobs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def available_gib(path=Path('/proc/meminfo')):
    for line in Path(path).read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024**2
    raise ValueError('No MemAvailable field; refuse memory admission')


def validate_plan(plan):
    if plan['minimum_available_gib'] < 32:
        raise ValueError('Memory reserve must not be lowered to force admission')
    for job in plan['jobs']:
        if Path(job['output']).exists():
            raise FileExistsError('Existing run directory: '+job['output'])
    if len({job['output'] for job in plan['jobs']}) != len(plan['jobs']):
        raise ValueError('Duplicate output paths')


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def prepare_source(plan):
    digest=hashlib.sha256()
    with Path(plan['archive']).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    if digest.hexdigest()!=plan['archive_sha256']:
        raise ValueError('Source archive hash differs')
    source=Path(plan['source_directory'])
    source.mkdir(exist_ok=False)
    subprocess.run(['tar','-xzf',plan['archive'],'-C',str(source)],check=True)
    write(source/'DEPLOYED_SOURCE.json',dict(commit=plan['source_commit'],archive_sha256=digest.hexdigest()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--interval',type=int,default=30)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text());validate_plan(plan)
    if args.state.exists():raise FileExistsError('Existing resource-gate state; inspect rather than duplicate')
    state=dict(status='waiting_for_ram',source_commit=plan['source_commit'],
               minimum_available_gib=plan['minimum_available_gib'],jobs=[],started_at=time.time())
    children=[]

    def poll():
        for process,job in children:
            code=process.poll()
            if code is not None and job['status']=='running':
                queue=Path(job['output'])/'QUEUE.json'
                result=json.loads(queue.read_text()) if queue.exists() else {}
                job.update(exit_code=code,status='complete' if code==0 and result.get('status')=='complete' else 'failed')
                if job['status']=='failed':raise RuntimeError('Queue failed; inspect '+job['log'])

    try:
        prepared=False
        for planned in plan['jobs']:
            # Require two positive checks to avoid admitting on a transient spike.
            stable=0
            while stable<2:
                poll();available=available_gib()
                stable=stable+1 if available>=plan['minimum_available_gib'] else 0
                state.update(status='waiting_for_ram',waiting_queue=planned['name'],
                             available_gib=available,updated_at=time.time())
                write(args.state,state)
                print(json.dumps(dict(event='memory_admission',queue=planned['name'],
                    available_gib=available,required_gib=plan['minimum_available_gib'],stable_checks=stable)),flush=True)
                if stable<2:time.sleep(max(10,args.interval))
            if not prepared:prepare_source(plan);prepared=True
            # GPU and legacy-source preflight remain mandatory in each child.
            log=Path(planned['log']);log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as stream:
                process=subprocess.Popen(planned['command'],cwd=plan['source_directory'],
                                         stdout=stream,stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL,start_new_session=True)
            job=dict(planned,pid=process.pid,status='running');children.append((process,job));state['jobs'].append(job)
            state.update(status='running',updated_at=time.time());write(args.state,state)
            # Allow imports/data loading to show up before admitting the next group.
            time.sleep(max(30,args.interval))
        while any(process.poll() is None for process,_ in children):
            poll();state.update(status='running',available_gib=available_gib(),updated_at=time.time())
            write(args.state,state);time.sleep(max(10,args.interval))
        poll();state.update(status='complete',completed_at=time.time())
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:write(args.state,state)


if __name__=='__main__':main()
