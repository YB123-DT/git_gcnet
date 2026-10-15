"""Shared, local-only features and reproducible A/T/V missingness.

The mask law follows GCNet/ComP's intended CPM-Net rule: uniformly preserve
one modality, then independently retain each remaining modality with the
probability needed for the requested expected rate (capped at 2/3).
Unlike the legacy batch-level rejection sampler, it is independent of batches.
"""
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from runtime import RUN_ROOT

WORKSPACE = Path(os.environ.get('PAPER_WORKSPACE', '/data2/yb/paper')).expanduser().resolve()
MELD_PATH = WORKSPACE / 'baseline_sources_20261006/ECERC/data/meld/meld_emotion_semantic_features_roberta.pkl'
IEMOCAP_ROOT = WORKSPACE / 'GCNet_TPAMI/dataset/IEMOCAP'
CACHE_ROOT = RUN_ROOT / 'data_cache'
MASK_PROTOCOL = 'preserve_one_plus_independent_v1'
EVAL_MASK_SEED = 20261015


def canonical_dataset(name):
    key = str(name).lower().replace('-', '').replace('_', '')
    aliases = {'meld': 'MELD', 'iemocap4': 'IEMOCAPFour',
               'iemocapfour': 'IEMOCAPFour', 'iemocap6': 'IEMOCAPSix',
               'iemocapsix': 'IEMOCAPSix'}
    if key not in aliases:
        raise ValueError('Unsupported dataset: {}'.format(name))
    return aliases[key]


def _retention_probability(mr):
    mr = float(mr)
    if not 0 <= mr <= 1:
        raise ValueError('missing rate must be in [0, 1]')
    return max(0.0, min(1.0, (3 * (1 - mr) - 1) / 2))


def random_masks(n, mr, rng):
    """Return n x 3 float32 masks; rng is NumPy Generator or RandomState."""
    p = _retention_probability(mr)
    masks = (rng.random((int(n), 3)) < p).astype(np.float32)
    preserved = (rng.integers(0, 3, size=int(n)) if hasattr(rng, 'integers')
                 else rng.randint(0, 3, size=int(n)))
    masks[np.arange(int(n)), preserved] = 1
    return masks


def fixed_masks(dataset, dialogue_id, utterance_ids, mr, eval_seed=EVAL_MASK_SEED):
    """A stable mask per utterance, shared across models and training seeds."""
    p = _retention_probability(mr)
    dataset = canonical_dataset(dataset)
    masks = np.empty((len(utterance_ids), 3), dtype=np.float32)
    for i, uid in enumerate(utterance_ids):
        key = json.dumps([MASK_PROTOCOL, dataset, str(dialogue_id), str(uid),
                          format(float(mr), '.12g'), int(eval_seed)],
                         ensure_ascii=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(key).digest()
        # 64-bit uniform values from a cryptographic digest avoid Python hash
        # randomization and dependence on NumPy PRNG implementation/version.
        words = [int.from_bytes(digest[j:j + 8], 'little') for j in range(0, 32, 8)]
        masks[i] = [float(value / 2**64 < p) for value in words[1:]]
        masks[i, words[0] % 3] = 1
    return masks


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validate(records, classes, dimensions, speakers):
    for record in records:
        n = len(record['labels'])
        if n != len(record['utterance_ids']) or n != len(record['speakers']):
            raise ValueError('unaligned utterances in {}'.format(record['id']))
        if not np.all((record['labels'] >= 0) & (record['labels'] < classes)):
            raise ValueError('invalid emotion label')
        if not np.all((record['speakers'] >= 0) & (record['speakers'] < speakers)):
            raise ValueError('invalid speaker id')
        for modality, dim in zip(('a', 't', 'v'), dimensions):
            value = record[modality]
            if value.shape != (n, dim) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid {} features in {}'.format(modality, record['id']))


def _load_meld(path):
    with path.open('rb') as stream:
        values = pickle.load(stream, encoding='latin1')
    if len(values) != 17:
        raise ValueError('ECERC MELD requires 17 fields')
    spks, labels = values[:2]
    splits = {'train': values[14], 'valid': values[16], 'test': values[15]}
    sets = list(map(set, splits.values()))
    if set.union(*sets) != set(labels) or any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('MELD splits are not complete and disjoint')
    records = []
    for split, ids in splits.items():
        for did in ids:
            speaker = np.asarray(spks[did])
            if speaker.shape != (len(labels[did]), 9) or not np.all(speaker.sum(1) == 1):
                raise ValueError('MELD requires nine-speaker one-hot vectors')
            records.append({'id': str(did), 'utterance_ids': [str(i) for i in range(len(labels[did]))],
                            'a': np.asarray(values[11][did], dtype=np.float32),
                            't': np.asarray(values[3][did], dtype=np.float32),
                            'v': np.asarray(values[12][did], dtype=np.float32),
                            'labels': np.asarray(labels[did], dtype=np.int64),
                            'speakers': speaker.argmax(1).astype(np.int64), 'split': split})
    return records, {'feature_fields': {'a': 11, 't': 3, 'v': 12},
                     'text_feature': 'ECERC eroberta1', 'num_classes': 7,
                     'num_speakers': 9, 'dimensions': [300, 1024, 342]}


def _load_iemocap(path, root, classes):
    with path.open('rb') as stream:
        utterances, labels, speakers, _, train_ids, test_ids = pickle.load(stream, encoding='latin1')
    ids = sorted(train_ids | test_ids)
    if set(ids) != set(utterances):
        raise ValueError('IEMOCAP source dialogue coverage differs')
    records = []
    feature_names = ['wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT']
    for did in ids:
        modality_values = [[], [], []]
        for uid, speaker in zip(utterances[did], speakers[did]):
            paths = [root / feature_names[0] / (uid + '.npy'),
                     root / feature_names[1] / (uid + '.npy'),
                     root / feature_names[2] / uid / ('compress_' + speaker + '.npy')]
            for values, feature_path in zip(modality_values, paths):
                value = np.load(feature_path).squeeze()
                if value.ndim == 2:
                    value = value.mean(0)
                values.append(value)
        records.append({'id': did, 'utterance_ids': list(utterances[did]),
                        **{mod: np.asarray(v, dtype=np.float32) for mod, v in zip(('a', 't', 'v'), modality_values)},
                        'labels': np.asarray(labels[did], dtype=np.int64),
                        'speakers': np.asarray([{'F': 0, 'M': 1}[s] for s in speakers[did]], dtype=np.int64),
                        'session': int(did[4])})
    return records, {'feature_names': dict(zip(('a', 't', 'v'), feature_names)),
                     'feature_root': str(root), 'visual_selection': 'actual utterance speaker compress_F/M.npy',
                     'num_classes': classes, 'num_speakers': 2, 'dimensions': [512, 1024, 1024]}


def load_dataset(name, cache_root=CACHE_ROOT):
    """Return {'records': [...], 'metadata': {...}}; no GPU or remote IO."""
    name = canonical_dataset(name)
    classes = 7 if name == 'MELD' else (4 if name == 'IEMOCAPFour' else 6)
    path = MELD_PATH if name == 'MELD' else IEMOCAP_ROOT / ('IEMOCAP_features_raw_{}way.pkl'.format(classes))
    fingerprint = {'schema': 2, 'path': str(path), 'size': path.stat().st_size,
                   'mtime_ns': path.stat().st_mtime_ns}
    if name != 'MELD':
        fingerprint['feature_directories'] = {p.name: p.stat().st_mtime_ns for p in (IEMOCAP_ROOT / 'features').iterdir() if p.is_dir()}
    # NumPy 2 pickles reference numpy._core, unavailable in legacy NumPy 1
    # training environments. Keep their caches separate without new packages.
    cache_name = '{}_numpy{}.pkl'.format(name, np.__version__.split('.')[0])
    cache = Path(cache_root) / cache_name if cache_root is not None else None
    if cache is not None and cache.exists():
        with cache.open('rb') as stream:
            cached = pickle.load(stream)
        if cached['metadata']['source_fingerprint'] == fingerprint:
            return cached
    records, metadata = _load_meld(path) if name == 'MELD' else _load_iemocap(path, IEMOCAP_ROOT / 'features', classes)
    _validate(records, classes, metadata['dimensions'], metadata['num_speakers'])
    feature_digest = hashlib.sha256()
    for record in records:
        feature_digest.update(json.dumps([record['id'], record['utterance_ids']], separators=(',', ':')).encode())
        for key in ('a', 't', 'v', 'labels', 'speakers'):
            feature_digest.update(np.ascontiguousarray(record[key]).tobytes())
    metadata.update({'dataset': name, 'label_source': str(path), 'label_source_sha256': _sha256(path),
                     'selected_features_labels_sha256': feature_digest.hexdigest(),
                     'source_fingerprint': fingerprint, 'mask_protocol': MASK_PROTOCOL,
                     'evaluation_mask_seed': EVAL_MASK_SEED,
                     'dialogue_count': len(records), 'utterance_count': sum(len(r['labels']) for r in records)})
    result = {'records': records, 'metadata': metadata}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix('.{}.tmp'.format(os.getpid()))
        with temp.open('wb') as stream:
            pickle.dump(result, stream, protocol=4)
        os.replace(str(temp), str(cache))
    return result
