"""Opportunistic dispatch of real jobs; never controls another user's launcher."""
import json
from pathlib import Path
import subprocess


def load_policy(path, manifest_ids):
    policy=json.loads(Path(path).read_text())
    bundle,other=policy['bundle_ids'],policy['other_ids']
    if (len(bundle)!=12 or len(other)!=8 or len(set(bundle+other))!=20
            or set(bundle+other)!=set(manifest_ids)
            or policy['bundle_gpu_index']!='3'
            or policy['bundle_gpu_uuid']!='GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'
            or policy['other_gpu_indices']!=['0','1','2','5','6','7']):
        raise ValueError('Expected disjoint 12-on-GPU3 / 8-on-other-healthy-GPUs policy')
    return policy


def compute_processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'],text=True,timeout=8)
    rows=[]
    for line in output.splitlines():
        if not line.strip(): continue
        uuid,pid=(part.strip() for part in line.split(','))
        rows.append((uuid,int(pid)))
    return rows


def own_experiment(pid,root):
    """Only this campaign's actual trainers/profilers, not same-account jobs."""
    try:
        args=Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode().split('\0')
        if '-m' not in args or '--output' not in args: return False
        module=args[args.index('-m')+1]
        if module not in (
            'experiments.osram_meaningful20_round2_20261004.run',
            'experiments.osram_meaningful20_round2_20261004.cuda_check'):
            return False
        return Path(args[args.index('--output')+1]).resolve().is_relative_to(Path(root).resolve())
    except (OSError,UnicodeError,ValueError,IndexError):
        return False


def allowed_gpu(policy,candidate,index,root):
    index=str(index)
    if candidate in policy['other_ids']:
        return index in policy['other_gpu_indices']
    if candidate not in policy['bundle_ids'] or index!=policy['bundle_gpu_index']:
        return False
    try:
        return all(own_experiment(pid,root) for uuid,pid in compute_processes()
                   if uuid==policy['bundle_gpu_uuid'])
    except (OSError,ValueError,subprocess.SubprocessError):
        # An unreadable process table is not evidence that a GPU became free.
        return False
