"""Match each frozen MOSI parent to its own fixed-rate stage-two experiment."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from osram_four_block_wiring import FrozenFlatReader
from experiments.osram_mosi_cmerc_20261023 import worker as control
from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
from experiments.osram_tasr_20261018 import runner_helpers as h

PARENT = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')
RATES = tuple(str(i / 10) for i in range(8))


def validate_rate(rate):
    if isinstance(rate, bool):
        raise ValueError('rate must be one of 0.0 through 0.7')
    try:
        normalized = str(float(rate))
    except (ValueError, TypeError):
        raise ValueError('rate must be one of 0.0 through 0.7') from None
    if normalized not in RATES:
        raise ValueError('rate must be one of 0.0 through 0.7')
    return normalized


def should_train(rate, mode='neumiss'):
    rate = validate_rate(rate)
    if mode not in ('neumiss', 'signature'):
        raise ValueError('stage two supports NeuMiss and Signature only')
    return mode == 'signature' or rate != '0.0'


def train_schedule(bundle, rate):
    return bundle['trainer']._schedules(bundle['cfg'], 'train')[float(validate_rate(rate))]


def setup(uuid, output, rate):
    rate = validate_rate(rate)
    original = json.loads((control.REFERENCE / 'SUMMARY.json').read_text())
    accepted = json.loads((PARENT / 'SUMMARY.json').read_text())
    if accepted['status'] != 'complete' or accepted['epoch'] != 150:
        raise ValueError('Require the completed accepted cmerc Flat150 parent')
    selected = accepted['best_by_rate'][rate]
    path = PARENT / selected['checkpoint']
    digest = h.sha(path)
    if digest != selected['checkpoint_sha256']:
        raise ValueError('Accepted checkpoint SHA256 differs at rate ' + rate)
    bundle = control.setup(uuid, Path(output), original)
    state = torch.load(path, map_location='cpu', weights_only=False)
    canonical = lambda value: json.loads(json.dumps(value))
    if state['epoch'] != selected['epoch'] or canonical(state['identity']) != accepted['identity']:
        raise ValueError('Accepted checkpoint epoch/identity differs at rate ' + rate)
    if 'rate' in state and validate_rate(state['rate']) != rate:
        raise ValueError('Selected checkpoint contains a different rate')
    if 'selection' in state and validate_rate(state['selection']['rate']) != rate:
        raise ValueError('Selected checkpoint ledger contains a different rate')
    current_config = canonical(asdict(bundle['cfg']))
    if any(current_config.get(k) != v for k, v in accepted['identity']['config'].items()):
        raise ValueError('Frozen parent configuration differs')
    for key in ('torch', 'numpy', 'sklearn'):
        if bundle['environment'][key] != accepted['environment'][key]:
            raise ValueError('Frozen parent environment differs: ' + key)
    bundle['model'].load_state_dict(state['model'], strict=True)
    reader = FrozenFlatReader(bundle['model'])
    identity = dict(parent_id=digest, checkpoint_sha256=digest, checkpoint=str(path),
        epoch=state['epoch'], rate=rate, environment=bundle['environment'],
        config=asdict(bundle['cfg']), summary_sha256=h.sha(PARENT / 'SUMMARY.json'),
        source='accepted cmerc Flat per-rate BEST', parent_frozen=True,
        train_rate=rate, test_rates=[rate], neumiss_zero_rate_bypass=not should_train(rate),
        added_default_config_fields={k: v for k, v in current_config.items()
                                     if k not in accepted['identity']['config']})
    bundle['parent_identity'] = identity
    return bundle, reader, identity


def test_cache(bundle, reader, rate):
    rate = validate_rate(rate)
    if bundle['parent_identity']['rate'] != rate:
        raise ValueError('Test rate must match the selected frozen parent')
    accepted = json.loads((PARENT / 'SUMMARY.json').read_text())
    records = torch.load(PARENT / 'fixed_masks.pt', map_location='cpu', weights_only=False)['test'][rate]
    digest = hashlib.sha256(b''.join(r['availability'].numpy().tobytes() for r in records)).hexdigest()
    if digest != accepted['identity']['mask_hashes']['test'][rate]:
        raise ValueError('Accepted test mask SHA256 differs at rate ' + rate)
    selected = accepted['best_by_rate'][rate]
    prediction_path = PARENT / selected['predictions']
    if h.sha(prediction_path) != selected['prediction_sha256']:
        raise ValueError('Accepted prediction SHA256 differs at rate ' + rate)
    trainer, loader = bundle['trainer'], bundle['test_loader']
    device = next(reader.parent.parameters()).device
    rows = []
    with h.isolated_rng(66 + 99193), torch.no_grad():
        loader.sampler.set_epoch(0)
        for index, raw in enumerate(loader):
            if index >= len(records):
                raise ValueError('Exact test mask batch count differs')
            data = trainer._move_batch(raw, device)
            meta, record = scalar.metadata(data), records[index]
            for key, value in meta.items():
                np.testing.assert_array_equal(value, record[key])
            availability = record['availability'].to(device)
            view = trainer._prepare_view_from_primary_masks(data, availability, availability, bundle['dimensions'])
            frame = reader(view['incomplete'], view['availability'], view['qmask'], view['umask'], view['lengths'])
            torch.testing.assert_close(reader.decode_delta(frame, torch.zeros_like(frame.anchor)),
                                       frame.prediction, rtol=0, atol=0)
            rows.append(dict(frame=frame.to('cpu'), labels=view['labels'].cpu(), umask=view['umask'].cpu(),
                conversation_ids=list(map(str, view['conversation_ids'])), metadata=meta,
                availability=availability.transpose(0, 1)[data[7].bool()].cpu().numpy()))
    if not rows or len(rows) != len(records):
        raise ValueError('Exact test mask batch count differs')
    with np.load(prediction_path, allow_pickle=False) as saved:
        for key in rows[0]['metadata']:
            np.testing.assert_array_equal(np.concatenate([r['metadata'][key] for r in rows]), saved[key])
        np.testing.assert_array_equal(np.concatenate([r['availability'] for r in rows]), saved['availability'])
        predictions = np.concatenate([r['frame'].prediction.transpose(0, 1)[r['umask'].bool()].numpy() for r in rows])
        np.testing.assert_allclose(predictions, saved['prediction'], rtol=1e-5, atol=1e-6)
    return rows


# Public preparation helper, rather than a pytest test when imported elsewhere.
test_cache.__test__ = False
