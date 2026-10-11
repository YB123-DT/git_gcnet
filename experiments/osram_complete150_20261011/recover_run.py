"""Finite62-run completion: missing MOSI Flat and paired IEMOCAP150."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REMOTE = Path('/data2/yb/remote_experiments')
CROSS = REMOTE / 'osram_nested_cross_dataset_20261006'
GPUS = (0, 1, 2, 3, 7)
RESERVE = 9000
LABEL = 'INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(p)


def now():
    return datetime.now(timezone.utc).isoformat()


def specs():
    rows = [dict(dataset='CMUMOSI', model='flat', seed=s, fold=1) for s in (67, 68)]
    # Alternating methods/datasets gives early paired progress, not30 serial folds.
    rows += [dict(dataset=d, model=m, seed=s, fold=f) for f in range(1, 6)
             for s in (66, 67, 68) for d in ('IEMOCAPFour', 'IEMOCAPSix')
             for m in ('flat', 'nested')]
    for row in rows:
        row['id'] = '{dataset}_{model}_s{seed}_f{fold}'.format(**row)
    return rows


def verify_source(source):
    snap = read(source / 'SNAPSHOT.json')
    hashes = {n: h for n, h in snap['source_sha256'].items()
              if n.endswith('.py') and n.startswith(('gcnet_missing_m3/', 'gcnet_modality_jepa/', 'gcnet/'))}
    assert hashes
    for name, digest in hashes.items():
        assert sha(source / name) == digest, name
    return dict(path=str(source), commit=snap['code_commit'], hashes=hashes)


def prepare(root, commit):
    assert not (root / 'MANIFEST.json').exists()
    data = read(CROSS / 'INPUTS_3SEED.json')['data']
    data = {k: v for k, v in data.items() if k.startswith('IEMOCAP')}
    data['CMUMOSI'] = read(REMOTE / 'osram_meaningful20_round2_20261004/DATA.json')
    seen = {}
    for desc in data.values():
        for name, digest in desc['files'].items():
            if name not in seen:
                assert sha(name) == digest, name
                st = Path(name).stat()
                seen[name] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=digest)
    sources, tasks = {}, []
    for task in specs():
        ds, seed, fold = task['dataset'], task['seed'], task['fold']
        if ds == 'CMUMOSI':
            ref = REMOTE / f'osram_mosi_memory_gap_ablation_20260920/full/seed_{seed}'
            source = REMOTE / 'osram_new40_gpu0123_20261004/source_ad211c0'
        else:
            cls = 4 if ds == 'IEMOCAPFour' else 6
            ref = REMOTE / f'osram_iemocap_cfg84_nojepa_5session_20260919/iemocap{cls}/seed_{seed}/fold_{fold}'
            source = CROSS / 'source_644ee14'
        original = CROSS / ds / f'seed_{seed}/fold_{fold}' if task['model'] == 'nested' else None
        cfg_path = (original or ref) / 'config.json'
        cfg = read(cfg_path)
        assert cfg['seed'] == seed and cfg['fold'] == fold and cfg['dataset'] == ds
        assert cfg['epochs'] == 100 and cfg['learning_rate'] == .001
        assert cfg.get('lr_schedule', 'constant') == 'constant'
        assert cfg['training_objective'] == 'emotion-only' and cfg['osram_readout_fusion'] == 'flat'
        assert cfg.get('osram_meaningful_block', 'none') == ('nested_gnn_rooted_evidence' if original else 'none')
        assert not cfg.get('paired_history_views', False)
        assert read(ref / 'PROVENANCE.json')['status'] == 'complete'
        if original:
            pr = read(original / 'PROVENANCE.json')
            assert pr['status'] == 'complete' and pr['outputs_verified']
            assert sha(original / 'last_training.pt') == pr['artifact_sha256']['last_training.pt']
            source = Path(pr['source'])
            task['original_state_sha256'] = pr['artifact_sha256']['last_training.pt']
        else:
            assert not (ref / 'last_training.pt').exists(), 'Prefer full recovery when available'
        if str(source) not in sources:
            sources[str(source)] = verify_source(source)
        task.update(reference=str(ref), original=str(original) if original else None,
                    source=str(source), config_path=str(cfg_path), config_sha256=sha(cfg_path),
                    config=cfg, mode='full-state-continuation' if original else 'fresh100-then-full-state50')
        tasks.append(task)
    assert len(tasks) == 62 and len({t['id'] for t in tasks}) == 62
    write(root / 'MANIFEST.json', dict(label=LABEL, runner_commit=commit,
          runner_sha256=sha(Path(__file__)), tasks=tasks, data=data,
          data_verified=seen, sources=sources, prepared_at=now()))


def configure(cfg, continuation=False):
    assert cfg.epochs == 100 and cfg.learning_rate == .001 and cfg.lr_schedule == 'constant'
    return replace(cfg, epochs=150, learning_rate=.0001) if continuation else cfg


def override_lr(state):
    assert state['next_epoch'] == 100 and len(state['history']) == 100
    assert state['optimizer']['state'] and state['scheduler'] is None and state['scaler'] is None
    assert state['rng'] and len(state['best_references']) == 8
    assert all(g['lr'] == .001 for g in state['optimizer']['param_groups'])
    for group in state['optimizer']['param_groups']:
        group['lr'] = .0001


def train(root, index, uuid):
    manifest = read(root / 'MANIFEST.json')
    assert sha(root / 'source_4eed138/run.py') == manifest['runner_sha256']
    task = manifest['tasks'][index]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    assert uuid in [v[0] for v in gpu_info().values()]
    source = Path(task['source'])
    for n, h in manifest['sources'][str(source)]['hashes'].items():
        assert sha(source / n) == h
    for n in manifest['data'][task['dataset']]['files']:
        st = Path(n).stat(); saved = manifest['data_verified'][n]
        assert (st.st_size, st.st_mtime_ns) == (saved['size'], saved['mtime_ns']), n
    assert sha(task['config_path']) == task['config_sha256']
    os.environ['GCNET_DATASET_ROOT'] = manifest['data'][task['dataset']]['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(root / ('cache_' + task['dataset']))
    sys.path.insert(0, str(source))
    import torch
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_missing_m3.training_resume import TrainingState, _atomic
    torch.set_num_threads(1)
    cfg100 = configure(trainer.TrainConfig(**task['config']))
    output = root / 'runs' / task['id']
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'RUN.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = dict(task=task, label=LABEL, status='running', pid=os.getpid(),
                  gpu_uuid=uuid, started_at=now(), runner_commit=manifest['runner_commit'],
                  manifest_sha256=sha(root / 'MANIFEST.json'), stages=[])
    prior = read(output / 'PROVENANCE.json')
    assert prior['status'] in ('running', 'failed'), prior['status']
    assert list(output.glob('stage*/last_training.pt')), 'Full state required'
    record.update(recovered_after_scheduler_handoff=True, recovery_runner_sha256=sha(Path(__file__)))
    write(output / 'PROVENANCE.json', record)
    identity = dict(source=manifest['sources'][str(source)]['commit'],
                    manifest=sha(root / 'MANIFEST.json'), task=task['id'])
    features = manifest['data'][task['dataset']]['feature_roots']
    original_train = trainer.train_epoch
    sig = inspect.signature(original_train)
    trace_dir = None
    expected_lr = None
    def traced(*pos, **kw):
        args = sig.bind(*pos, **kw).arguments
        rates = [float(g['lr']) for g in args['optimizer'].param_groups]
        assert rates and all(x == expected_lr for x in rates), rates
        write(trace_dir / f"epoch_{args['epoch']+1:03d}.json", dict(epoch=args['epoch']+1, rates=rates))
        return original_train(*pos, **kw)
    trainer.train_epoch = traced
    try:
        if task['original']:
            original = Path(task['original'])
            assert sha(original / 'last_training.pt') == task['original_state_sha256']
        else:
            original = output / 'stage100'
            trace_dir, expected_lr = original / 'learning_rates', .001
            id100 = dict(identity, config=hashlib.sha256(json.dumps(asdict(cfg100), sort_keys=True).encode()).hexdigest())
            if not (output / 'stage150/last_training.pt').exists():
                trainer.run_experiment(cfg100, *features, str(original),
                                      training_state=TrainingState(original, identity=id100))
            assert len(read(original / 'history.json')) == 100
            record['stages'].append(dict(stage=100, status='complete', from_scratch=True))
            write(output / 'PROVENANCE.json', record)
        original_sha = sha(original / 'last_training.pt')
        state = torch.load(original / 'last_training.pt', map_location='cpu', weights_only=False)
        override_lr(state)
        history100 = state['history']
        cfg150 = configure(cfg100, continuation=True)
        id150 = dict(identity, config=hashlib.sha256(json.dumps(asdict(cfg150), sort_keys=True).encode()).hexdigest(),
                     original_state_sha256=original_sha, lr_override=.0001)
        state['identity'] = id150
        dest = output / 'stage150'
        if (dest / 'last_training.pt').exists():
            resumed = torch.load(dest / 'last_training.pt', map_location='cpu', weights_only=False)
            assert resumed['identity'] == id150 and 100 <= resumed['next_epoch'] <= 150
            schedule_identity = resumed['schedule_identity']
            del resumed, state
        else:
            dest.mkdir(exist_ok=False)
            for refs in state['best_references'].values():
                for ref in refs.values():
                    path = Path(ref['path'])
                    assert not path.is_absolute() and '..' not in path.parts
                    assert sha(original / path) == ref['sha256']
                    target = dest / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(original / path, target)
            _atomic(dest / 'last_training.pt', lambda f: torch.save(state, f))
            schedule_identity = state['schedule_identity']
            del state
        trace_dir, expected_lr = dest / 'learning_rates', .0001
        metrics = trainer.run_experiment(cfg150, *features, str(dest), training_state=TrainingState(
            dest, identity=id150, schedule_identity=schedule_identity))
        history = read(dest / 'history.json')
        assert len(history) == 150 and history[:100] == history100
        assert len(list(trace_dir.glob('epoch_*.json'))) == 50
        before = read(original / 'metrics.json')
        assert metrics['mask_sha256'] == before['mask_sha256'] == read(Path(task['reference']) / 'metrics.json')['mask_sha256']
        key = 'accuracy' if task['dataset'].startswith('IEMOCAP') else 'weighted_f1'
        assert metrics['selection_metric'] == key
        assert all(metrics['test'][str(i/10)][key] >= before['test'][str(i/10)][key] for i in range(8))
        assert sha(original / 'last_training.pt') == original_sha
        files = ['config.json','metrics.json','history.json','last_training.pt']
        files += [f'{kind}_miss_0p{i}.{ext}' for i in range(8) for kind, ext in [('best','pt'),('predictions','npz')]]
        record.update(status='complete', outputs_verified=True, finished_at=now(),
                      original_state_sha256=original_sha, effective_config150=asdict(cfg150),
                      artifact_sha256={f: sha(dest / f) for f in files})
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), finished_at=now())
        raise
    finally:
        write(output / 'PROVENANCE.json', record)


def gpu_info():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                                  '--format=csv,noheader,nounits'], text=True)
    return {int(i): (u.strip(), int(f)) for line in raw.splitlines()
            for i,u,f in [line.split(',')] if int(i) in GPUS}


def dispatch(root):
    lock = (root / 'QUEUE.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'QUEUE.json').exists(), 'Reconcile existing queue before restarting'
    tasks = read(root / 'MANIFEST.json')['tasks']
    rows = [dict(id=t['id'], status='pending') for t in tasks]
    state = dict(status='running', pid=os.getpid(), gpu_whitelist=GPUS, max_per_gpu=2, tasks=rows)
    children = {}; halted = False
    while True:
        for i, child in list(children.items()):
            code = child.poll()
            if code is not None:
                p = root / 'runs' / rows[i]['id'] / 'PROVENANCE.json'
                ok = code == 0 and p.exists() and read(p).get('outputs_verified', False)
                rows[i].update(status='complete' if ok else 'failed', exit_code=code)
                halted |= not ok
                del children[i]
        for gpu,(uuid,free) in gpu_info().items():
            if halted: break
            active = [r for r in rows if r['status'] == 'running' and r.get('gpu') == gpu]
            # Conservative full reservation even after CUDA allocation: never
            # over-admit CPU-loading jobs. At most two concurrent jobs per GPU.
            if len(active) >= 2 or free - len(active)*RESERVE < RESERVE: continue
            if shutil.disk_usage(root).free < (30 + len(children)*5)*2**30: continue
            i = next((i for i,r in enumerate(rows) if r['status']=='pending'), None)
            if i is None: break
            log = root / 'logs' / (rows[i]['id']+'.log'); log.parent.mkdir(exist_ok=True)
            command = [sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--train',str(i),'--gpu-uuid',uuid]
            with log.open('x') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
                             MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            rows[i].update(status='running', pid=child.pid, gpu=gpu, uuid=uuid, command=command, log=str(log))
            children[i] = child
            write(root/'QUEUE.json',state)
            time.sleep(4)
        state['status'] = 'failed_draining' if halted else 'running'
        write(root/'QUEUE.json',state)
        if not children and (halted or all(r['status']=='complete' for r in rows)):
            state.update(status='failed' if halted else 'complete', finished_at=now())
            write(root/'QUEUE.json',state)
            return
        time.sleep(15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--dispatch', action='store_true')
    p.add_argument('--train', type=int)
    p.add_argument('--gpu-uuid')
    p.add_argument('--commit')
    a = p.parse_args()
    if a.prepare: prepare(a.root, a.commit)
    elif a.dispatch: dispatch(a.root)
    elif a.train is not None: train(a.root, a.train, a.gpu_uuid)
    else: p.error('Choose --prepare, --dispatch or --train')


if __name__ == '__main__':
    main()

