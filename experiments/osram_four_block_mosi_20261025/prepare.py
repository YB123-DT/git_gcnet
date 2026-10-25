"""Immutable frozen-parent inputs; fitted artifacts use TRAIN exclusively."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from osram_four_block_wiring import FrozenFlatReader, SupportBank, FixedLatentProjection
from experiments.osram_mosi_cmerc_20261023 import worker as control
from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
from experiments.osram_tasr_20261018 import runner_helpers as h

PARENT = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')
RATES = tuple(str(i / 10) for i in range(8))


def setup(uuid, output):
    original = json.loads((control.REFERENCE / 'SUMMARY.json').read_text())
    accepted = json.loads((PARENT / 'SUMMARY.json').read_text())
    if accepted['status'] != 'complete' or accepted['epoch'] != 150:
        raise ValueError('Require the completed accepted cmerc Flat parent')
    selected = accepted['best_by_rate']['0.3']
    path = PARENT / selected['checkpoint']
    digest = h.sha(path)
    if digest != selected['checkpoint_sha256']:
        raise ValueError('Accepted rate0.3 checkpoint SHA256 differs')
    bundle = control.setup(uuid, Path(output), original)
    state = torch.load(path, map_location='cpu', weights_only=False)
    canonical = lambda value: json.loads(json.dumps(value))
    if state['epoch'] != selected['epoch'] or canonical(state['identity']) != accepted['identity']:
        raise ValueError('Accepted checkpoint epoch/identity differs')
    current_config = canonical(asdict(bundle['cfg']))
    if any(current_config.get(k) != v for k,v in accepted['identity']['config'].items()):
        raise ValueError('Frozen parent configuration differs')
    for key in ('torch', 'numpy', 'sklearn'):
        if bundle['environment'][key] != accepted['environment'][key]:
            raise ValueError('Frozen parent environment differs: ' + key)
    bundle['model'].load_state_dict(state['model'], strict=True)
    reader = FrozenFlatReader(bundle['model'])
    identity = dict(parent_id=digest, checkpoint_sha256=digest, checkpoint=str(path),
                    epoch=state['epoch'], rate='0.3', environment=bundle['environment'],
                    config=asdict(bundle['cfg']), summary_sha256=h.sha(PARENT / 'SUMMARY.json'),
                    added_default_config_fields={k:v for k,v in current_config.items()
                                                 if k not in accepted['identity']['config']})
    bundle['parent_identity'] = identity
    return bundle, reader, identity


def _read(reader, view):
    return reader(view['incomplete'], view['availability'], view['qmask'],
                  view['umask'], view['lengths'])


def build_test_cache(bundle, reader):
    masks = torch.load(PARENT / 'fixed_masks.pt', map_location='cpu', weights_only=False)['test']
    accepted = json.loads((PARENT / 'SUMMARY.json').read_text())
    if set(masks) != set(RATES):
        raise ValueError('Require all eight accepted test mask banks')
    for rate, records in masks.items():
        digest = hashlib.sha256(b''.join(r['availability'].numpy().tobytes() for r in records)).hexdigest()
        if digest != accepted['identity']['mask_hashes']['test'][rate]:
            raise ValueError('Accepted test mask SHA256 differs at rate ' + rate)
    selected = accepted['best_by_rate']['0.3']
    prediction_path = PARENT / selected['predictions']
    if h.sha(prediction_path) != selected['prediction_sha256']:
        raise ValueError('Accepted prediction SHA256 differs')
    trainer, loader = bundle['trainer'], bundle['test_loader']
    device = next(reader.parent.parameters()).device
    cache = {}
    with h.isolated_rng(66 + 99193), torch.no_grad():
        for rate in RATES:
            loader.sampler.set_epoch(0)
            rows = []
            for index, raw in enumerate(loader):
                data = trainer._move_batch(raw, device)
                meta, record = scalar.metadata(data), masks[rate][index]
                for key, value in meta.items():
                    np.testing.assert_array_equal(value, record[key])
                availability = record['availability'].to(device)
                view = trainer._prepare_view_from_primary_masks(data, availability, availability, bundle['dimensions'])
                frame = _read(reader, view)
                torch.testing.assert_close(reader.decode_delta(frame, torch.zeros_like(frame.anchor)),
                                           frame.prediction, rtol=0, atol=0)
                rows.append(dict(frame=frame.to('cpu'), labels=view['labels'].cpu(),
                    umask=view['umask'].cpu(), conversation_ids=list(map(str, view['conversation_ids'])),
                    metadata=meta, availability=availability.transpose(0, 1)[data[7].bool()].cpu().numpy()))
            if len(rows) != len(masks[rate]):
                raise ValueError('Exact test mask batch count differs')
            cache[rate] = rows
    with np.load(prediction_path, allow_pickle=False) as saved:
        rows = cache['0.3']
        for key in rows[0]['metadata']:
            np.testing.assert_array_equal(np.concatenate([r['metadata'][key] for r in rows]), saved[key])
        np.testing.assert_array_equal(np.concatenate([r['availability'] for r in rows]), saved['availability'])
        predictions = np.concatenate([r['frame'].prediction.transpose(0, 1)[r['umask'].bool()].numpy() for r in rows])
        np.testing.assert_allclose(predictions, saved['prediction'], rtol=1e-5, atol=1e-6)
    return cache


def build_support_bank(bundle, reader, parent_id):
    trainer, loader = bundle['trainer'], bundle['train_loader']
    schedules = trainer._schedules(bundle['cfg'], 'train')
    device = next(reader.parent.parameters()).device
    hidden, labels, predictions, ids, rate_ids, patterns = [], [], [], [], [], []
    mask_hashes, cohorts, counts = {}, {}, {}
    with h.isolated_rng(66 + 76194), torch.no_grad():
        for rate_index, rate in enumerate(RATES):
            loader.sampler.set_epoch(0)
            digest, cohort, count = hashlib.sha256(), [], 0
            for raw in loader:
                data = trainer._move_batch(raw, device)
                view = trainer._prepare_view(data, schedules[float(rate)], 0, bundle['dimensions'])
                frame = _read(reader, view)
                valid = view['umask'].bool()
                a = frame.availability.transpose(0, 1)[valid].cpu()
                batch_ids = [str(view['conversation_ids'][b]) for b, t in valid.nonzero().tolist()]
                cohort.extend(map(str, view['conversation_ids']))
                ids.extend(batch_ids)
                hidden.append(frame.hidden.transpose(0, 1)[valid].cpu())
                labels.append(view['labels'][valid].reshape(-1, 1).cpu())
                predictions.append(frame.prediction.transpose(0, 1)[valid].cpu())
                patterns.append((a.long() * torch.tensor([4, 2, 1])).sum(-1))
                rate_ids.append(torch.full((len(a),), rate_index, dtype=torch.long))
                digest.update(view['availability'].cpu().numpy().tobytes())
                count += len(a)
            cohorts[rate] = hashlib.sha256(json.dumps(cohort, separators=(',', ':')).encode()).hexdigest()
            mask_hashes[rate], counts[rate] = digest.hexdigest(), count
    if len(set(cohorts.values())) != 1 or len(set(counts.values())) != 1:
        raise ValueError('TRAIN support cohorts differ across rates')
    parent_key = parent_id['parent_id'] if isinstance(parent_id, dict) else parent_id
    bank = SupportBank(torch.cat(hidden), torch.cat(labels), torch.cat(predictions), ids,
                       torch.cat(rate_ids), torch.cat(patterns), parent_key, split='train')
    manifest = dict(split='train', parent_id=parent_key, seed=66 + 76194,
                    cohort_sha256=cohorts, counts=counts, mask_sha256=mask_hashes,
                    labels_source='TRAIN only', query_labels_used=False)
    return bank, manifest


def fit_projection(bundle, reader):
    trainer, loader = bundle['trainer'], bundle['train_loader']
    device = next(reader.parent.parameters()).device
    rows = [[], [], []]
    cohort = []
    reader.parent.eval()
    with h.isolated_rng(66 + 76195), torch.no_grad():
        loader.sampler.set_epoch(0)
        for raw in loader:
            data = trainer._move_batch(raw, device)
            visible = data[7].T.unsqueeze(-1).expand(-1, -1, 3).to(data[0].dtype)
            view = trainer._prepare_view_from_primary_masks(data, visible, visible, bundle['dimensions'])
            valid = view['umask'].T.bool()
            parts = torch.split(view['complete'], bundle['dimensions'], -1)
            for m, (name, part) in enumerate(zip(('audio', 'text', 'visual'), parts)):
                projected = reader.parent.observed_set.projectors[name](part[valid])
                rows[m].append(projected.cpu().double())
            cohort.extend(map(str, view['conversation_ids']))
    means, bases = [], []
    for pieces in rows:
        x = torch.cat(pieces)
        if x.shape[0] < 64 or not torch.isfinite(x).all():
            raise ValueError('Insufficient finite TRAIN rows for rank64 PCA')
        mean = x.mean(0)
        _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
        means.append(mean.float())
        bases.append(vh[:64].float())
    projection = FixedLatentProjection(torch.stack(means), torch.stack(bases))
    manifest = dict(split='train', rows=len(x), rank=64, centered=True, fit_dtype='float64',
                    cohort_sha256=hashlib.sha256(json.dumps(cohort, separators=(',', ':')).encode()).hexdigest(),
                    mean_sha256=hashlib.sha256(projection.mean.numpy().tobytes()).hexdigest(),
                    basis_sha256=hashlib.sha256(projection.basis.numpy().tobytes()).hexdigest(),
                    parent_id=bundle.get('parent_identity', {}).get('parent_id'), memory_scans=0)
    return projection, manifest
