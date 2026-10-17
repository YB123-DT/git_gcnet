"""Read-only paired correctness analysis of already-fitted legacy MELD probes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def align_predictions(labels, local, other, included):
    labels, included = np.asarray(labels), np.asarray(included)
    if included.dtype != np.bool_ or included.shape != labels.shape or labels.ndim != 1:
        raise ValueError('Require a boolean inclusion vector aligned to full test labels')
    if not included.any() or not np.array_equal(local['labels'], labels):
        raise ValueError('Local label order differs from full cached test labels')
    if not np.array_equal(other['labels'], labels[included]):
        raise ValueError('Other probe label order differs from donor-included cached labels')
    left, right = np.asarray(local['predictions']), np.asarray(other['predictions'])
    if left.shape != labels.shape or right.shape != labels[included].shape:
        raise ValueError('Prediction lengths do not match their original cohorts')
    return labels[included], left[included], right


def cross_table(labels, local, other):
    labels, local, other = map(np.asarray, (labels, local, other))
    if labels.ndim != 1 or not len(labels) or labels.shape != local.shape or local.shape != other.shape:
        raise ValueError('Need nonempty aligned label/prediction vectors')
    lc, oc = local == labels, other == labels
    row = dict(N=len(labels), both_correct=int((lc & oc).sum()),
        local_correct_other_wrong=int((lc & ~oc).sum()),
        local_wrong_other_correct=int((~lc & oc).sum()), both_wrong=int((~lc & ~oc).sum()))
    for name, pred in [('local', local), ('other', other)]:
        row[name+'_ACC'] = 100*float(accuracy_score(labels, pred))
        row[name+'_UA'] = 100*float(recall_score(labels, pred, labels=np.arange(7), average='macro', zero_division=0))
        row[name+'_WF1'] = 100*float(f1_score(labels, pred, average='weighted', zero_division=0))
    row['delta_ACC_pp'] = 100*(row['local_wrong_other_correct']-row['local_correct_other_wrong'])/row['N']
    row['delta_UA_pp'] = row['other_UA']-row['local_UA']
    row['delta_WF1_pp'] = row['other_WF1']-row['local_WF1']
    errors = int((~lc).sum())
    row['local_error_N'] = errors
    row['other_correct_given_local_wrong_pct'] = 100*row['local_wrong_other_correct']/errors if errors else None
    assert sum(row[k] for k in ['both_correct','local_correct_other_wrong','local_wrong_other_correct','both_wrong']) == row['N']
    assert abs(row['delta_ACC_pp']-(row['other_ACC']-row['local_ACC'])) < 1e-10
    return row


def load_predictions(path):
    with np.load(path, allow_pickle=False) as saved:
        data = {k:saved[k] for k in ('labels','logits','predictions')}
    if data['logits'].shape != (len(data['labels']),7) or not np.isfinite(data['logits']).all():
        raise ValueError(f'Invalid seven-class logits: {path}')
    if not np.array_equal(data['predictions'],data['logits'].argmax(1)):
        raise ValueError(f'Saved prediction/argmax mismatch: {path}')
    if not np.issubdtype(data['labels'].dtype,np.integer) or not np.isin(data['labels'],np.arange(7)).all():
        raise ValueError(f'Invalid class labels: {path}')
    metrics = json.loads(path.with_name('RESULT.json').read_text())['metrics']
    assert metrics['n'] == len(data['labels'])
    assert abs(metrics['accuracy']-accuracy_score(data['labels'],data['predictions'])) < 1e-12
    assert abs(metrics['weighted_f1']-f1_score(data['labels'],data['predictions'],average='weighted',zero_division=0)) < 1e-12
    return data


def analyze(root):
    rows, manifest = [], {}
    for index in range(8):
        name = f'rate_0p{index}'
        outer = root/name
        folder = outer/name if (outer/name/'probes').is_dir() else outer
        status = json.loads((outer/'STATUS.json').read_text())
        if status.get('status') != 'complete':
            raise ValueError(f'Incomplete legacy probe extraction: {outer}')
        feature = folder/'test_features.npz'
        with np.load(feature,allow_pickle=False) as cached:
            labels, ids, positions = cached['labels'], cached['conversation_ids'].astype(str), cached['utterance_indices']
        if len(set(zip(ids,positions))) != len(labels):
            raise ValueError('Duplicate cached utterance identity')
        provenance = json.loads((outer/'PROVENANCE.json').read_text())
        if 'gcnet_legacy_fixed' not in provenance['source'] or provenance['mask_epoch'] != 0:
            raise ValueError('Expected recorded fixed-mask legacy source, never current random-mask weights')
        manifest[str(feature)] = sha(feature)
        manifest[str(outer/'PROVENANCE.json')] = sha(outer/'PROVENANCE.json')
        for seed in (66,67,68):
            donor_path = folder/'probes'/f'donors_seed{seed}.json'
            donor = json.loads(donor_path.read_text())['test']
            included = np.asarray(donor['included'])
            if included.dtype != np.bool_ or not np.array_equal(donor['target_conversation'],ids):
                raise ValueError('Donor target identities differ from cached feature order')
            donor_indices = np.asarray(donor['indices'])
            if not np.array_equal(included,donor_indices>=0) or np.any(donor_indices[included]>=len(labels)):
                raise ValueError('Invalid donor inclusion/index mapping')
            if np.any(ids[donor_indices[included]] == ids[included]):
                raise ValueError('Donor must come from another conversation')
            manifest[str(donor_path)] = sha(donor_path)
            base = folder/'probes'/f'seed_{seed}'
            local_path = base/'Local/predictions.npz'
            local = load_predictions(local_path)
            manifest[str(local_path)] = sha(local_path)
            for variant in ('Base','Memory','LocalMemory'):
                path = base/variant/'predictions.npz'
                other = load_predictions(path)
                y,left,right = align_predictions(labels,local,other,included)
                row = cross_table(y,left,right)
                row.update(rate=f'{index/10:.1f}',probe_seed=seed,variant=variant,
                    excluded_N=int((~included).sum()))
                rows.append(row)
                manifest[str(path)] = sha(path)
    return rows, manifest


def render(rows):
    text = ['# Legacy MELD Probe correctness cross tables', '', 'INTERNAL DIAGNOSTIC ONLY.', '',
        'No training or inference. Read saved predictions only. Backbone seed66; probe seeds66/67/68.',
        'Missing protocol: gcnet-legacy-fixed-v1, NOT current gcnet-random-mask-v1.',
        'Original probe environment was Torch2.2.2+cu121; this is not the current cmerc-repro baseline.',
        '', '## Seed66, missing0.7: exact paired counts', '',
        '| Other probe | Both correct | Local correct / other wrong | Local wrong / other correct | Both wrong | N |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['rate']=='0.7' and r['probe_seed']==66:
            text.append('| '+r['variant']+' | '+' | '.join(str(r[k]) for k in ['both_correct','local_correct_other_wrong','local_wrong_other_correct','both_wrong','N'])+' |')
    text += ['', '## Eight-rate averages over the three probe initializations', '',
        'Counts below are average counts per rate/initialization, not distinct utterances or three backbone seeds.', '',
        '| Other | Local wrong / other correct | Local correct / other wrong | ΔACC pp | ΔUA pp | ΔWF1 pp |',
        '|---|---:|---:|---:|---:|---:|']
    for variant in ('Base','Memory','LocalMemory'):
        subset = [r for r in rows if r['variant']==variant]
        keys = ['local_wrong_other_correct','local_correct_other_wrong','delta_ACC_pp','delta_UA_pp','delta_WF1_pp']
        text.append('| '+variant+' | '+' | '.join(f'{np.mean([r[k] for r in subset]):.3f}' for k in keys)+' |')
    text += ['', '## All rates: seed66 counts', '',
        '| Rate | Other | Both correct | Local correct / other wrong | Local wrong / other correct | Both wrong | N | ΔACC pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['probe_seed']==66:
            text.append(f"| {r['rate']} | {r['variant']} | "+' | '.join(str(r[k]) for k in ['both_correct','local_correct_other_wrong','local_wrong_other_correct','both_wrong','N'])+f" | {r['delta_ACC_pp']:+.3f} |")
    text += ['', '## Alignment and interpretation limits', '',
        '- Local is evaluated on the full cache; all three other probes use donor-included rows. Local is filtered with that exact boolean mask before comparison.',
        '- Check cached conversation/utterance identities, donor target order and labels; check saved predictions equal argmax(logits) and reproduce original RESULT.json metrics.',
        '- Predictions.npz does not itself contain IDs. Its row identity relies on the checked original extraction/fit construction and cache/donor order, not an independent ID field in that file.',
        '- Base and Memory probes include availability and history-presence/count metadata, whereas Local includes availability. These are separately fitted probe heads, not original Flat readout interventions.',
        '- Local-wrong/other-correct demonstrates complementary predictions, not a deployable routing rule. Gold correctness must not be used as a switch.',
        '- Base versus Memory differences are not an isolated Gap contribution: both the inputs and fitted heads differ.',
        '- Only Local versus LocalMemory compares the learned combined probe with Local; compare its corrections AND harms. Test-oracle probe selection remains optimistic.',
        '- Three probe initializations share the same frozen backbone seed66. These tables do not establish three-seed current-model gains.',
        '- No statistical significance or storage/query/decision bottleneck conclusion is established by these counts alone.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use an independent output directory, never overwrite historical probe artifacts')
    rows, manifest = analyze(args.input)
    args.output.mkdir(parents=True)
    payload = dict(label='INTERNAL DIAGNOSTIC ONLY',missing_protocol='gcnet-legacy-fixed-v1',
        backbone_seed=66,probe_seeds=[66,67,68],training=False,inference=False,
        input_root=str(args.input),script_sha256=sha(Path(__file__)),input_sha256=manifest,rows=rows)
    (args.output/'CROSS_TABLES.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    with (args.output/'CROSS_TABLES.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/'RESULT.md').write_text(render(rows))
    for row in rows:
        if row['rate']=='0.7' and row['probe_seed']==66:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
