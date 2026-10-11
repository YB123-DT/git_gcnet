"""Two approved concat/control runs; reuse audited 100+50 full-state trainer."""
import argparse
from dataclasses import asdict
import fcntl
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_complete150_20261011 import run as continuation
from experiments.osram_complete150_20261011.run import read, write, sha, now

METHODS = ('nested_rooted_concat', 'nested_concat_capacity_control')
GPU = 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66')
DATA = Path('/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json')


def prepare(root):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_core20_20261005.run import candidate_config
    assert not (root / 'MANIFEST.json').exists(), 'Do not overwrite an existing experiment'
    source = Path(__file__).resolve().parents[2]
    snapshot = read(source / 'SNAPSHOT.json')
    for name, digest in snapshot['source_sha256'].items():
        assert sha(source / name) == digest, name
    reference = read(REFERENCE / 'config.json')
    # Reuse cfg84 guards, but discard the small-flat change.
    candidate_config(reference, 'small_flat', seed=66)
    base = asdict(TrainConfig(**reference))
    assert base['epochs'] == 100 and base['learning_rate'] == .001
    assert base['lr_schedule'] == 'constant'
    data = read(DATA)
    verified = {}
    for name, digest in data['files'].items():
        assert sha(name) == digest, name
        st = Path(name).stat()
        verified[name] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=digest)
    tasks = []
    for method in METHODS:
        config = asdict(TrainConfig(**dict(base, osram_meaningful_block=method)))
        delta = {k: {'baseline': base[k], 'candidate': v} for k, v in config.items() if v != base[k]}
        assert set(delta) == {'osram_meaningful_block'}
        tasks.append(dict(id=method, dataset='CMUMOSI', model=method, seed=66, fold=1,
            reference=str(REFERENCE), original=None, source=str(source),
            config_path=str(REFERENCE/'config.json'), config_sha256=sha(REFERENCE/'config.json'),
            config=config, configuration_delta=delta, mode='fresh100-then-full-state50'))
    write(root/'MANIFEST.json', dict(label=continuation.LABEL, runner_commit=snapshot['code_commit'],
        runner_sha256=sha(continuation.__file__), tasks=tasks, data={'CMUMOSI': data},
        data_verified=verified, sources={str(source): dict(commit=snapshot['code_commit'],
        hashes=snapshot['source_sha256'])}, prepared_at=now(),
        environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                         host=platform.node(), platform=platform.platform())))


def dispatch(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'QUEUE.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root/'QUEUE.json').exists(), 'Inspect existing state before any restart'
    manifest = read(root/'MANIFEST.json')
    assert len(manifest['tasks']) == 2
    rows = [dict(id=t['id'], status='pending') for t in manifest['tasks']]
    state = dict(status='running', pid=os.getpid(), tasks=rows, gpu_whitelist=[1],
                 max_per_gpu=2, started_at=now())
    children = {}
    failed = False
    while True:
        for index, child in list(children.items()):
            code = child.poll()
            if code is not None:
                path = root/'runs'/rows[index]['id']/'PROVENANCE.json'
                ok = code == 0 and path.exists() and read(path).get('outputs_verified', False)
                rows[index].update(status='complete' if ok else 'failed', exit_code=code)
                failed |= not ok
                del children[index]
        pending = next((i for i,r in enumerate(rows) if r['status']=='pending'), None)
        if pending is not None and not failed:
            raw = subprocess.check_output(['nvidia-smi','--id=1',
                '--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'], text=True)
            index, uuid, free = [x.strip() for x in raw.split(',')]
            assert index == '1' and uuid == GPU
            if float(free)-len(children)*9000 >= 9000 and shutil.disk_usage(root).free > 40*2**30:
                logs = root/'logs'; logs.mkdir(exist_ok=True)
                log = logs/(rows[pending]['id']+'.log')
                command = [sys.executable,'-u','-m','experiments.osram_nested_concat_20261011.run',
                           '--root',str(root),'--train',str(pending)]
                with log.open('x') as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                rows[pending].update(status='running', pid=child.pid, gpu=1, uuid=GPU,
                                     command=command, log=str(log))
                children[pending] = child
        state['status'] = 'failed_draining' if failed else 'running'
        write(root/'QUEUE.json',state)
        if not children and (failed or all(r['status']=='complete' for r in rows)):
            state.update(status='failed' if failed else 'complete', finished_at=now())
            write(root/'QUEUE.json',state)
            return
        time.sleep(10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--dispatch', action='store_true')
    p.add_argument('--train', type=int, choices=(0,1))
    a = p.parse_args()
    if a.prepare:
        prepare(a.root)
    elif a.dispatch:
        dispatch(a.root)
    elif a.train is not None:
        continuation.train(a.root, a.train, GPU)
    else:
        p.error('choose prepare, dispatch or train')


if __name__ == '__main__':
    main()
