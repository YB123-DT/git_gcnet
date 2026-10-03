"""Fail-closed GPU admission and evidence-backed per-method readiness."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import subprocess

from .manifest import read, sha, verify_snapshot

HEALTHY_INDICES = {'0', '1', '2', '3', '5', '6', '7'}
BANNED_UUID = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'


def query_gpus():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in output.strip().splitlines():
        index, uuid, free, utilization, temperature = [v.strip() for v in line.split(',')]
        rows[index] = {'uuid': uuid, 'free_mib': float(free),
                       'utilization': float(utilization), 'temperature': float(temperature)}
    if '4' not in rows: raise ValueError('Cannot identify the banned physical GPU4 UUID')
    return rows


def validate_gpu(index, uuid, mapping):
    index = str(index)
    banned = mapping.get('4')
    if not banned or index not in HEALTHY_INDICES or uuid in (banned, BANNED_UUID) or mapping.get(index) != uuid:
        raise ValueError('GPU4, unknown GPU, or host index/UUID mismatch')


def admission(gpu, profile, *, disk_free_gib):
    values = [gpu.get('free_mib'), gpu.get('utilization'), gpu.get('temperature'),
              profile.get('peak_mib'), profile.get('artifact_gib'), disk_free_gib]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return False, 'invalid resource measurement'
    peak = profile['peak_mib']
    if peak <= 0 or profile['artifact_gib'] <= 0: return False, 'missing full profile'
    if gpu['temperature'] >= 90: return False, 'GPU temperature'
    if gpu.get('throughput_ratio', 1.) < .9: return False, 'measured throughput regression'
    if gpu['free_mib'] < peak + max(2048, .2 * peak): return False, 'GPU memory margin'
    if gpu['utilization'] >= 90: return False, 'GPU compute saturation'
    if disk_free_gib < profile['artifact_gib'] + 16: return False, 'disk budget including immutable versions'
    return True, 'admitted'


def validate_readiness(record, *, candidate, design_sha256, source_sha256):
    if (record.get('status') != 'ready' or record.get('candidate') != candidate
            or record.get('design_sha256') != design_sha256 or record.get('source_sha256') != source_sha256):
        raise ValueError('Candidate/source/design readiness mismatch')
    for device in ('cpu', 'cuda'):
        evidence = record.get(device, {})
        path = Path(evidence.get('log', '/__missing_readiness_log__'))
        if (evidence.get('returncode') != 0 or not evidence.get('command')
                or not path.is_file() or evidence.get('sha256') != sha(path)):
            raise ValueError(f'Missing or failed actual {device} test evidence')
    cuda = record['cuda']
    if (str(cuda.get('gpu_index')) not in HEALTHY_INDICES or not cuda.get('gpu_uuid')
            or cuda['gpu_uuid'] == BANNED_UUID):
        raise ValueError('CUDA evidence must identify a healthy host GPU and UUID')
    profile = record.get('profile', {})
    if (profile.get('batch_size') != 32 or profile.get('train_and_eval') is not True
            or any(not isinstance(profile.get(k), (int, float)) or not math.isfinite(profile[k])
                   or profile[k] <= 0 for k in ('peak_mib', 'artifact_gib'))):
        raise ValueError('Real batch32 train/eval memory and retained-version disk profile required')
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--design-sha256', required=True)
    args = parser.parse_args()
    source = verify_snapshot(args.snapshot)
    validate_readiness(read(args.readiness), candidate=args.candidate,
                       design_sha256=args.design_sha256, source_sha256=source['source_sha256'])
    print('Readiness evidence and immutable source verified; no training launched.')


if __name__ == '__main__': main()
