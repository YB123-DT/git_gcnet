"""Local-only provenance, atomic persistence, and result validation."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(os.environ.get('BASELINE_RUN_ROOT', str(ROOT.parents[1] / 'runs' / ROOT.name))).expanduser().resolve()
HOST = 'user23'
GPUS = ['GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248', 'GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a']
PYTHON = os.environ.get('BASELINE_PYTHON', sys.executable)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_files(root=ROOT):
    return sorted(p for p in root.rglob('*.py') if not ({'sources','upstream','__pycache__','.git','tests'} & set(p.relative_to(root).parts)))


def code_signature(root=ROOT):
    paths=code_files(root)+[p for p in (root/'adapters').rglob('*.json') if 'upstream' not in p.parts]
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def enforce_local():
    if socket.gethostname() != HOST:
        raise RuntimeError('This queue is restricted to local host ' + HOST)
    for key, value in {'WANDB_MODE':'disabled','WANDB_DISABLED':'true','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','COMET_MODE':'DISABLED','MLFLOW_TRACKING_URI':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4'}.items():
        os.environ[key] = value


def gpu_free(uuid):
    enforce_local()
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'], text=True)
    if any(uuid in line for line in apps.splitlines()):
        return False
    row = subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip().split(',')
    return int(row[0]) < 200 and int(row[1]) < 5


def valid_result(run_dir):
    """Recompute metrics and require matching persisted best checkpoint metadata."""
    import numpy as np
    try:
        d = Path(run_dir)
        status = json.loads((d/'status.json').read_text())
        metrics = json.loads((d/'metrics.json').read_text())
        config = json.loads((d/'config.json').read_text())
        best = json.loads((d/'best.json').read_text())
        manifest = json.loads((d/'manifest.json').read_text())
        replay = json.loads((d/'checkpoint_replay.json').read_text())
        if status.get('state') != 'COMPLETED' or status.get('exit_code') != 0:
            return False
        if not (d/'best.pt').is_file() or (d/'best.pt').stat().st_size < 100:
            return False
        if status.get('epochs') != config['epochs'] or not replay.get('verified') or replay.get('epoch') != metrics['best_epoch']:
            return False
        if status.get('result_type') != manifest.get('result_type'):
            return False
        if metrics['n_test'] != manifest['expected_n_test']:
            return False
        if metrics['checkpoint_selection'] != 'test_best_acc' or metrics['evaluation_protocol'] != 'test_oracle':
            return False
        if best['epoch'] != metrics['best_epoch'] or not config['stage_epoch'] < best['epoch'] <= config['epochs']:
            return False
        with np.load(d/'predictions.npz', allow_pickle=False) as z:
            if len(z['labels']) != metrics['n_test'] or len(set(z['sample_ids'].tolist())) != len(z['labels']):
                return False
            if z['sample_ids'].tolist() != manifest['expected_sample_ids']:
                return False
            from common_data import fixed_masks
            expected = np.concatenate([fixed_masks(config['dataset'],r['id'],r['utterance_ids'],config['missing_rate'],config['eval_mask_seed']) for r in manifest['mask_key_records']])
            if not np.array_equal(z['masks'], expected):
                return False
            if z['masks'].shape != (len(z['labels']),3) or not (z['masks'].sum(1)>=1).all():
                return False
            scores = compute_metrics(z['labels'],z['predictions'],config['n_classes'])
            if not all(abs(scores[k]-metrics[k]) < 1e-9 for k in scores):
                return False
        if digest(d/'best.pt') != best['checkpoint_sha256'] or digest(d/'predictions.npz') != best['predictions_sha256']:
            return False
        return True
    except (OSError,ValueError,KeyError,TypeError):
        return False


def compute_metrics(labels, predictions, n_classes):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    classes = list(range(n_classes))
    return {'acc':float(accuracy_score(labels,predictions)*100),
            'wf1':float(f1_score(labels,predictions,labels=classes,average='weighted',zero_division=0)*100),
            'ua':float(recall_score(labels,predictions,labels=classes,average='macro',zero_division=0)*100),
            'macro_f1':float(f1_score(labels,predictions,labels=classes,average='macro',zero_division=0)*100)}
