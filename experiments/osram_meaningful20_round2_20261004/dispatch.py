"""Opportunistic dispatch of real jobs; never controls another user's launcher."""
import json
import math
from pathlib import Path
import subprocess

from experiments.osram_meaningful20_20261003.manifest import read, verify_snapshot
from experiments.osram_meaningful20_20261003.preflight import admission, validate_readiness


def load_policy(path, manifest_ids):
    policy=json.loads(Path(path).read_text())
    bundle,other=policy['bundle_ids'],policy['other_ids']
    if (len(bundle)!=12 or len(other)!=8 or len(set(bundle+other))!=20
            or set(bundle+other)!=set(manifest_ids)
            or policy['bundle_gpu_index']!='3'
            or policy['bundle_gpu_uuid']!='GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'
            or policy['other_gpu_indices']!=['0','1','2','5','6','7']
            or policy.get('bundle_concurrency_limit') != 12
            or policy.get('other_concurrency_limit') != 3
            or policy.get('formal_concurrency_limit') != 15):
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


def allocated_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                     '--format=csv,noheader,nounits'], text=True, timeout=8)
    rows = {}
    for line in output.splitlines():
        if line.strip():
            uuid, pid, used = (part.strip() for part in line.split(','))
            used = float(used)
            if not math.isfinite(used) or used < 0: raise ValueError('Invalid process memory measurement')
            rows[int(pid)] = (uuid, used)
    return rows


class Dispatch:
    """An admission reservation, not a GPU allocation or a memory-holding job."""
    def __init__(self, policy):
        self.policy = policy
        self.limits = dict(bundle=policy['bundle_concurrency_limit'],
                           other=policy['other_concurrency_limit'], total=policy['formal_concurrency_limit'])

    def group(self, candidate):
        return 'bundle' if candidate in self.policy['bundle_ids'] else 'other'

    def target_gpu(self, candidate, index):
        if self.group(candidate) == 'bundle': return str(index) == self.policy['bundle_gpu_index']
        return str(index) in self.policy['other_gpu_indices']

    @staticmethod
    def memory_budget(profile):
        # Torch peaks exclude CUDA contexts and other non-PyTorch allocations.
        return 1.2 * profile['peak_mib'] + 512

    def plan(self, args, state, cards, active):
        waiting = state.setdefault('dispatch_waiting', {})
        count = sum(self.group(job['candidate']) == 'bundle' for job in active)
        state['bundle_concurrent_high_water'] = max(state.get('bundle_concurrent_high_water', 0), count)
        if state['phase'] != 'screening': return dict(ready=True, profiles={}, pending=[])
        profiles, missing, sources = {}, [], {}
        for candidate in self.policy['bundle_ids']:
            path = args.readiness_root / candidate / 'READY.json'
            if not path.is_file():
                missing.append(candidate); continue
            record = read(path)
            source = str(Path(record['snapshot_root']).resolve())
            if source not in sources: sources[source] = verify_snapshot(source)['source_sha256']
            profiles[candidate] = validate_readiness(record, candidate=candidate,
                design_sha256=cards[candidate]['design_sha256'], source_sha256=sources[source])
        if missing:
            waiting['bundle'] = f'twelve-profile readiness: {len(profiles)}/12 ready; missing {", ".join(missing)}'
            return dict(ready=False, profiles=profiles, pending=[], reason=waiting['bundle'])
        completed = [name for name in profiles if state['jobs'].get(f'{name}:66', {}).get('status') == 'complete']
        if completed and state['bundle_concurrent_high_water'] < 12:
            waiting['bundle'] = 'Cohort overlap lost before twelve live trainers; inspect instead of silently serializing'
            return dict(ready=False, profiles=profiles, pending=[], reason=waiting['bundle'])
        running = {job['candidate'] for job in active if self.group(job['candidate']) == 'bundle'}
        pending = [name for name in profiles if name not in running and name not in completed]
        peak = sum(self.memory_budget(profile) for profile in profiles.values())
        state['bundle_memory_budget_mib'] = peak + 2048
        waiting['bundle'] = f'12/12 profiles ready; aggregate budget {peak + 2048:.1f} MiB; awaiting live admission'
        return dict(ready=True, profiles=profiles, pending=pending)

    def admit(self, candidate, index, args, state, active, profile, plan, gpu, disk):
        if self.group(candidate) == 'bundle' and gpu['uuid'] != self.policy['bundle_gpu_uuid']:
            return False, 'Bundle GPU index/UUID differs from pinned GPU3 identity'
        try:
            allocated = allocated_memory()
        except (OSError, ValueError, subprocess.SubprocessError):
            return False, 'Cannot measure pending process allocations; waiting for resources'
        pending = 0.
        for job in active:
            if job['gpu_uuid'] != gpu['uuid']: continue
            uuid, used = allocated.get(job.get('pid'), (gpu['uuid'], 0.))
            if uuid != gpu['uuid']: return False, 'Process/GPU allocation identity changed'
            pending += max(0., self.memory_budget(job['profile']) - used)
        if self.group(candidate) == 'bundle' and state['phase'] == 'screening':
            if not plan['ready']: return False, plan['reason']
            required = pending + sum(self.memory_budget(plan['profiles'][name]) for name in plan['pending']) + 2048
            required_disk = sum(plan['profiles'][name]['artifact_gib'] for name in plan['pending']) + 16
            if gpu['free_mib'] < required:
                return False, f'Twelve-job aggregate memory: need {required:.1f} MiB free, have {gpu["free_mib"]:.1f}; no serial fallback'
            if disk < required_disk:
                return False, f'Twelve-job aggregate artifact disk: need {required_disk:.1f} GiB, have {disk:.1f}'
        else:
            gpu = dict(gpu, free_mib=gpu['free_mib'] - pending - 512)
            if self.group(candidate) == 'other' and any(self.group(job['candidate']) == 'bundle' for job in active):
                disk -= sum(plan['profiles'][name]['artifact_gib'] for name in plan['pending'])
        # Keep utilization/temperature safety, also enforced by the immutable child.
        return admission(gpu, profile, disk_free_gib=disk)


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
