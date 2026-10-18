"""Audit the pinned MMGCN MELD release and replace only canonical features."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from experiments.osram_meld_alternate_features_20261017.prepare import NumpyOnlyUnpickler, sha


SOURCE_COMMIT = '85732d984f70c1c84dd47c81aa97f1271397b899'
SOURCE_URL = f'https://raw.githubusercontent.com/hujingwen6666/MMGCN/{SOURCE_COMMIT}/MELD_features/MELD_features_raw1.pkl'
SOURCE_SHA256 = 'd043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0'
SOURCE_BLOB_SHA1 = '55873a76e17fa3e054bcc5afc534cd1dde2eb2d8'
CANONICAL_SHA256 = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'
SPLITS = {'train': 14, 'dev': 16, 'test': 15}
STREAMS = {'audio': (4, 11), 'text': (3, 3), 'visual': (5, 12)}


class MMGCNUnpickler(NumpyOnlyUnpickler):
    def find_class(self, module, name):
        if module in ('builtins', '__builtin__') and name == 'set':
            return set
        return super().find_class(module, name)


def compare_and_convert(canonical, source):
    if len(canonical) != 17 or len(source) != 10:
        raise ValueError('Require canonical 17-field and MMGCN 10-field schema')
    ids = set(canonical[0])
    if any(set(source[i]) != ids for i in range(7)):
        raise ValueError('MMGCN must cover the complete canonical dialogue set')
    splits = {name: set(canonical[index]) for name, index in SPLITS.items()}
    if set.union(*splits.values()) != ids or sum(map(len, splits.values())) != len(ids):
        raise ValueError('Canonical split overlap or missing dialogues')
    if set(source[7]) != splits['train'] | splits['dev'] or set(source[8]) != splits['test']:
        raise ValueError('Unexpected upstream train+dev/test coverage')
    for cid in canonical[0]:
        if len(source[0][cid]) != len(canonical[1][cid]):
            raise ValueError('Source utterance ID length mismatch')
        if len(set(source[0][cid])) != len(source[0][cid]):
            raise ValueError('Duplicate source utterance ID')
        for source_index, canonical_index in ((1, 0), (2, 1), (6, 13)):
            if not np.array_equal(np.asarray(source[source_index][cid]), np.asarray(canonical[canonical_index][cid])):
                raise ValueError(f'Metadata mismatch: dialogue {cid}, field {source_index}')
    features = {name: {} for name in STREAMS}
    audit = dict(split_counts={}, dimensions={}, by_split={}, metadata_equal=True,
                 all_training_float32_features_equal=True,
                 preserved_fields=[0, 1, 2, 13, 14, 15, 16],
                 replaced_fields=list(range(3, 13)),
                 upstream_train_includes_canonical_dev=True,
                 canonical_train_dev_test_preserved=True)
    for split, index in SPLITS.items():
        audit['split_counts'][split] = sum(len(canonical[1][cid]) for cid in canonical[index])
        audit['by_split'][split] = {}
        for name, (si, ci) in STREAMS.items():
            summary = dict(N=0, equal_raw_N=0, equal_float32_N=0,
                           source_zero_N=0, ecerc_zero_N=0, zero_flags_equal=True,
                           all_finite=True)
            for cid in canonical[index]:
                a, b = np.asarray(source[si][cid]), np.asarray(canonical[ci][cid])
                n = len(canonical[1][cid])
                if a.ndim != 2 or b.ndim != 2 or len(a) != n or len(b) != n or not a.shape[1]:
                    raise ValueError(f'Invalid feature shape at {split}/{cid}/{name}')
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Nonfinite source or canonical feature')
                dims = dict(source=int(a.shape[1]), ecerc=int(b.shape[1]))
                if audit['dimensions'].setdefault(name, dims) != dims:
                    raise ValueError('Feature dimension changes across utterances')
                converted = a.astype(np.float32)
                if not np.isfinite(converted).all():
                    raise ValueError('Nonfinite feature after float32 conversion')
                features[name][cid] = converted
                same_shape = a.shape == b.shape
                summary['N'] += n
                summary['equal_raw_N'] += int(np.all(a == b, axis=1).sum()) if same_shape else 0
                summary['equal_float32_N'] += int(np.all(converted == b.astype(np.float32), axis=1).sum()) if same_shape else 0
                za, zb = np.all(a == 0, axis=1), np.all(b == 0, axis=1)
                summary['source_zero_N'] += int(za.sum())
                summary['ecerc_zero_N'] += int(zb.sum())
                summary['zero_flags_equal'] = summary['zero_flags_equal'] and np.array_equal(za, zb)
            if summary['equal_float32_N'] != summary['N']:
                audit['all_training_float32_features_equal'] = False
            audit['by_split'][split][name] = summary
    output = list(canonical)
    output[3:11] = [features['text']] * 8
    output[11], output[12] = features['audio'], features['visual']
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix('.json')
    if args.output.exists() or report_path.exists():
        raise FileExistsError('Do not overwrite prepared data or its audit')
    if sha(args.source) != SOURCE_SHA256 or sha(args.canonical) != CANONICAL_SHA256:
        raise ValueError('Pinned source or accepted ECERC checksum mismatch')
    content = args.source.read_bytes()
    if hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() != SOURCE_BLOB_SHA1:
        raise ValueError('Download does not match the official Git blob')
    del content
    with args.source.open('rb') as stream:
        source = MMGCNUnpickler(stream).load()
    with args.canonical.open('rb') as stream:
        canonical = pickle.load(stream, encoding='latin1')
    output, audit = compare_and_convert(canonical, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not audit['all_training_float32_features_equal']:
        with args.output.open('xb') as stream:
            pickle.dump(output, stream, protocol=4)
        audit['output_sha256'] = sha(args.output)
    else:
        audit['output_sha256'] = None
    audit.update(source_url=SOURCE_URL, source_commit=SOURCE_COMMIT,
                 source_git_blob_sha1=SOURCE_BLOB_SHA1, source_sha256=SOURCE_SHA256,
                 source_bytes=args.source.stat().st_size,
                 canonical_path=str(args.canonical), canonical_sha256=CANONICAL_SHA256,
                 source_path=str(args.source), output_path=str(args.output),
                 numpy_version=np.__version__,
                 training_required=not audit['all_training_float32_features_equal'])
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
