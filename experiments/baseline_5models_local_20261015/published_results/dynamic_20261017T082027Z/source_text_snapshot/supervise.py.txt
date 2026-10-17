"""Persist child exit status even after dispatcher restart, with a per-run launch lock."""
import argparse
import fcntl
from pathlib import Path
import os
import subprocess
from runtime import enforce_local,atomic_json,now

if __name__=='__main__':
    enforce_local()
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    d=Path(a.run_dir)
    lock=(d/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=a.command[1:] if a.command[0]=='--' else a.command
    try:
        child=subprocess.Popen(cmd)
        atomic_json(d/'child.json',dict(pid=child.pid,supervisor_pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],started_at=now()))
        rc=child.wait()
        atomic_json(d/'exit.json',dict(exit_code=rc,at=now(),pid=child.pid))
        if rc:
            atomic_json(d/'status.json',dict(state='FAILED',exit_code=rc,pid=child.pid,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],at=now(),error='See failure.json and run.log; no automatic retry'))
        raise SystemExit(rc)
    except Exception as exc:
        atomic_json(d/'exit.json',dict(exit_code=1,at=now(),error=str(exc)))
        atomic_json(d/'status.json',dict(state='FAILED',exit_code=1,at=now(),error=str(exc)))
        raise
