"""Local bounded SSH recovery: install the RAM gate once, never bypass it."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
REMOTE='/home/qinjinghui/remote_experiments/osram_text_path_router_3seed_20261019'
STATE=REMOTE+'/RESOURCE_GATE_STATE.json'
SESSION='router_ram_00b7552'
PLAN=HERE/'RESOURCE_PLAN_00b7552.json'


def ssh_args():
    return ['ssh','-S','none','-o','ControlMaster=no','-o','BatchMode=yes',
            '-o','ConnectTimeout=8','-o','ServerAliveInterval=4','-o','ServerAliveCountMax=2','biggpu']


def remote_state():
    code=("import json,pathlib; p=pathlib.Path("+repr(STATE)+
          "); print(p.read_text() if p.exists() else json.dumps({'status':'not_installed'}))")
    result=subprocess.run(ssh_args()+['/usr/bin/python3 -c '+shlex.quote(code)],
                          capture_output=True,text=True,timeout=22,check=True)
    return json.loads(result.stdout)


def start_command(source=None):
    if source is None:source=json.loads(PLAN.read_text())
    gate=REMOTE+'/resource_gate_00b7552.py'
    plan=REMOTE+'/RESOURCE_PLAN_00b7552.json'
    command=('/usr/bin/python3 -u '+shlex.quote(gate)+' --plan '+shlex.quote(plan)+
             ' --state '+shlex.quote(STATE)+' --interval 30 > '+
             shlex.quote(REMOTE+'/resource_gate_00b7552.log')+' 2>&1')
    return ('set -eu; test "$(sha256sum '+shlex.quote(source['archive'])+
            ' | cut -d " " -f 1)" = '+shlex.quote(source['archive_sha256'])+'; '
            'if [ ! -f '+shlex.quote(STATE)+' ] && ! tmux has-session -t '+SESSION+' 2>/dev/null; then '
            'test ! -s '+shlex.quote(REMOTE+'/resource_gate_00b7552.log')+'; '
            'tmux new-session -d -s '+SESSION+' '+shlex.quote(command)+'; fi')


def save(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--gate-script',type=Path,default=HERE/'resource_gate.py')
    parser.add_argument('--attempts',type=int,default=60)
    parser.add_argument('--interval',type=int,default=60)
    args=parser.parse_args()
    if args.state.exists():raise FileExistsError('Existing deployment state; inspect rather than duplicate')
    plan=json.loads(args.plan.read_text())
    digest=hashlib.sha256()
    with args.archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    if digest.hexdigest()!=plan['archive_sha256']:raise ValueError('Local source archive differs')
    state=dict(status='waiting_for_ssh',source_commit=plan['source_commit'],
               tmux_session=SESSION,server='biggpu',attempts=[],model_training_started=False)
    save(args.state,state)
    for attempt in range(1,args.attempts+1):
        row=dict(attempt=attempt,time=time.time())
        try:
            current=remote_state()
            if current['status']=='not_installed':
                files=((args.archive,plan['archive']),
                       (args.gate_script,REMOTE+'/resource_gate_00b7552.py'),
                       (args.plan,REMOTE+'/RESOURCE_PLAN_00b7552.json'))
                for local,remote in files:
                    subprocess.run(['scp','-o','ControlPath=none','-o','BatchMode=yes','-o','ConnectTimeout=8',
                        str(local),'biggpu:'+remote],capture_output=True,text=True,timeout=300,check=True)
                subprocess.run(ssh_args()+[start_command(plan)],capture_output=True,text=True,timeout=35,check=True)
                time.sleep(2)
                current=remote_state()
            if current.get('source_commit')!=plan['source_commit']:
                raise ValueError('Remote gate missing or belongs to a different source')
            state.update(status='remote_gate_failed' if current['status']=='failed' else 'deployed',
                         remote_resource_state=current,completed_at=time.time())
            row['status']='deployed';state['attempts'].append(row);save(args.state,state)
            print(json.dumps(dict(event='resource_gate_deployed',remote_status=current['status'])),flush=True)
            return
        except (OSError,subprocess.SubprocessError,json.JSONDecodeError) as error:
            row.update(status='connection_or_transfer_failed',error=str(error))
            state['attempts'].append(row);save(args.state,state)
            print(json.dumps(row),flush=True)
            time.sleep(max(30,args.interval))
        except ValueError as error:
            state.update(status='deployment_failed',error=str(error),completed_at=time.time())
            save(args.state,state);raise
    state.update(status='ssh_retry_limit_reached',completed_at=time.time());save(args.state,state)


if __name__=='__main__':main()
