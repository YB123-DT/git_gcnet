"""Replace only MELD feature arrays; never alter canonical utterances or masks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import unicodedata

import numpy as np

CANONICAL_SHA = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'
CANONICAL_CLASSES = ('neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger')
MAGTKD_CLASSES = ('anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise')
SPLITS = {'train': 14, 'dev': 16, 'test': 15}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == '_reconstruct':
            return np.core.multiarray._reconstruct
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')


def released_pickle(path):
    with Path(path).open('rb') as stream:
        return NumpyOnlyUnpickler(stream).load()


def normalize_text(value):
    # The ECERC metadata preserves Windows-1252 quotation bytes as controls.
    for code in range(128, 160):
        try:
            value = value.replace(chr(code), bytes([code]).decode('cp1252'))
        except UnicodeDecodeError:
            pass
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def magtkd_records(bundle, speaker_map):
    names = {int(v): k for k, v in speaker_map.items()}
    result = {}
    for did in bundle['vids']:
        times = bundle['dia2utt'][did]
        keys = ('text', 'audio_kd', 'video_kd', 'labels', 'speakers')
        if any(len(bundle[k][did]) != len(times) for k in keys):
            raise ValueError('MAGTKD utterance stream lengths differ')
        for t, uid in enumerate(times):
            key = (int(did), int(uid))
            if key in result:
                raise ValueError('Duplicate MAGTKD utterance')
            result[key] = dict(text=np.asarray(bundle['text'][did][t]),
                audio=np.asarray(bundle['audio_kd'][did][t]),
                visual=np.asarray(bundle['video_kd'][did][t]),
                label=MAGTKD_CLASSES[int(bundle['labels'][did][t])],
                source_speaker=names[int(bundle['speakers'][did][t])])
    return result


def mmalign_records(bundle, embedding):
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError('Invalid MM-Align embedding table')
    result = {}
    for identifier, item in bundle.items():
        parts = str(identifier).split('_')
        if len(parts) != 2:
            raise ValueError('MM-Align requires original dialogue_utterance IDs')
        key = tuple(map(int, parts))
        ids = np.asarray(item['token_ids'])
        if (ids.ndim != 1 or not len(ids) or not np.isfinite(ids).all()
                or not np.equal(ids, np.floor(ids)).all()
                or ids.min() < 0 or ids.max() >= len(embedding)):
            raise ValueError('Invalid token embedding lookup')
        audio, visual = (np.asarray(item[k]) for k in ('audio_features', 'video_features'))
        if audio.ndim != 2 or visual.ndim != 2 or len(audio) != len(ids) or len(visual) != len(ids):
            raise ValueError('MM-Align aligned sequence dimensions differ')
        if key in result:
            raise ValueError('Duplicate MM-Align utterance')
        result[key] = dict(text=embedding[ids.astype(np.int64)].mean(0),
            audio=audio.mean(0), visual=visual.mean(0),
            label=CANONICAL_CLASSES[int(item['label'])])
    return result


def convert(reference, metadata, records):
    if len(reference) != 17:
        raise ValueError('Require the unchanged 17-field ECERC canonical metadata')
    features = {k: {} for k in ('audio', 'text', 'visual')}
    counts, dimensions, speaker_disagreements = {}, {}, []
    for split, index in SPLITS.items():
        grouped = {}
        expected_ids = set()
        for row in metadata[split]:
            did, uid = int(row['Dialogue_ID']), int(row['Utterance_ID'])
            key = (did, uid)
            if key in expected_ids:
                raise ValueError('Duplicate official CSV utterance')
            expected_ids.add(key)
            grouped.setdefault(did, []).append(row)
        missing = expected_ids - set(records[split])
        extra = set(records[split]) - expected_ids
        if missing or extra:
            raise ValueError(f'{split} coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}; '
                             'refuse to drop or fabricate canonical utterances')
        vids = reference[index]
        offset = min(vids)
        if {int(v) - offset for v in vids} != set(grouped):
            raise ValueError('Official and canonical dialogue IDs do not align')
        count = 0
        for vid in vids:
            rows = grouped[int(vid) - offset]
            if len(rows) != len(reference[1][vid]):
                raise ValueError('Canonical conversation length changed')
            arrays = {name: [] for name in features}
            # Only verify speaker identities; NEVER replace canonical selectors
            # with MAGTKD's different global speaker numbering.
            speaker_map, inverse_speakers = {}, {}
            for t, row in enumerate(rows):
                key = (int(row['Dialogue_ID']), int(row['Utterance_ID']))
                item = records[split][key]
                label = CANONICAL_CLASSES[int(reference[1][vid][t])]
                if row['Emotion'] != label or item['label'] != label:
                    raise ValueError(f'{split}/{key}: emotion-label disagreement')
                if normalize_text(row['Utterance']) != normalize_text(reference[13][vid][t]):
                    raise ValueError(f'{split}/{key}: utterance text differs')
                if 'speaker' in item and item['speaker'] != row['Speaker']:
                    raise ValueError(f'{split}/{key}: source speaker differs')
                if ('source_speaker' in item and normalize_text(item['source_speaker']) !=
                        normalize_text(row['Speaker'])):
                    speaker_disagreements.append(dict(split=split, dialogue=key[0], utterance=key[1],
                        upstream=item['source_speaker'], canonical=row['Speaker']))
                speaker = int(np.argmax(reference[0][vid][t]))
                name = row['Speaker']
                if speaker_map.setdefault(name, speaker) != speaker or inverse_speakers.setdefault(speaker, name) != name:
                    raise ValueError(f'{split}/{key}: canonical speaker identity inconsistent')
                for name in features:
                    value = np.asarray(item[name], dtype=np.float32)
                    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                        raise ValueError(f'{split}/{key}: invalid {name} vector')
                    if dimensions.setdefault(name, len(value)) != len(value):
                        raise ValueError('Feature dimension changes across samples')
                    arrays[name].append(value)
                count += 1
            for name in features:
                features[name][vid] = np.stack(arrays[name])
        counts[split] = count
    output = list(reference)
    output[3:11] = [features['text']] * 8
    output[11], output[12] = features['audio'], features['visual']
    return output, dict(utterances=counts, dimensions=dimensions,
        upstream_speaker_disagreements=speaker_disagreements,
        preserved_fields=[0, 1, 2, 13, 14, 15, 16],
        only_replaced_feature_fields=list(range(3, 13)),
        canonical_ids_order_labels_speakers_splits_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', choices=('magtkd', 'mmalign'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--metadata', type=Path,
                        help='Official MELD CSV directory (train/dev/test_sent_emo.csv)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.canonical) != CANONICAL_SHA:
        raise ValueError('Canonical reference is not the accepted MELD feature version')
    if args.output.exists():
        raise FileExistsError('Do not overwrite an existing prepared feature bundle')
    # Trusted project baseline is hash-pinned; externally downloaded files use
    # the restricted NumPy-only unpickler above.
    with args.canonical.open('rb') as stream:
        reference = pickle.load(stream, encoding='latin1')
    records, metadata, inputs = {}, {}, {}
    if args.release == 'magtkd':
        meta_root = args.source / 'upstream_metadata/datasets/MELD'
        mapping = json.loads((args.source / 'upstream_metadata/MELD/feature/speaker2id.json').read_text())
    else:
        meta_root = args.source
        embedding = released_pickle(args.source / 'embedding.p')
        inputs['embedding.p'] = sha(args.source / 'embedding.p')
    for split in SPLITS:
        feature = args.source / (f'first_stage_{split}_features.pkl' if args.release == 'magtkd' else f'{split}.pkl')
        meta_path = (args.metadata / f'{split}_sent_emo.csv' if args.metadata else
                     meta_root / (f'{split}_meld_emo.csv' if args.release == 'magtkd' else f'{split}_sent_emo.csv'))
        with meta_path.open(newline='', encoding='utf-8-sig') as stream:
            metadata[split] = list(csv.DictReader(stream))
        bundle = released_pickle(feature)
        records[split] = magtkd_records(bundle, mapping) if args.release == 'magtkd' else mmalign_records(bundle, embedding)
        inputs[str(feature.name)] = sha(feature)
        inputs[str(meta_path.name)] = sha(meta_path)
    converted, audit = convert(reference, metadata, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        pickle.dump(converted, stream, protocol=4)
    audit.update(release=args.release, source=str(args.source), source_sha256=inputs,
                 canonical_sha256=CANONICAL_SHA, output_sha256=sha(args.output),
                 streams=['text', 'audio_kd', 'video_kd'] if args.release == 'magtkd' else
                         ['mean_glove_lookup', 'mean_aligned_audio', 'mean_aligned_visual'])
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
