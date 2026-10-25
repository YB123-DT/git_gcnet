"""Read-only IEMOCAP-6 seed68 Signature path audit, with ACC and six-class UA."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

RATES = tuple(f'{i/10:.1f}' for i in range(8))
NUMERICAL_SOURCE_SHA = '21e46f2ecd092345818659694db8ff372672c35e'
MODES = ('flat', 'normal', 'degree2_no_area', 'interior_shuffle_6601',
         'interior_shuffle_6602', 'interior_shuffle_6603')


def validate_scope(folds, rates):
    if (not folds or len(set(folds)) != len(folds)
            or any(type(f) is not int or f not in range(1, 6) for f in folds)
            or not rates or len(set(rates)) != len(rates) or any(r not in RATES for r in rates)):
        raise ValueError('Only distinct IEMOCAP-6 seed68 folds1..5 and rates0.0..0.7 are allowed')
    return list(folds), list(rates)


def paired_metrics(labels, candidate, reference):
    """All six classes, including class zero; empty strata are N/A, never zero."""
    from experiments.osram_signature_iemocap_cyclic_20261025.summarize import prediction_metrics
    y, p, r = np.asarray(labels).reshape(-1), np.asarray(candidate), np.asarray(reference)
    if p.shape != (len(y), 6) or r.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Aligned finite six-class logits required')
    result = dict(N=len(y), ACC=None, UA=None, WF1=None, CE=None, recall=None, support=None,
                  corrections_vs_reference=0, harms_vs_reference=0, prediction_changes=0)
    if not len(y):
        return result
    result.update(prediction_metrics(y, p)[0])
    new, old = p.argmax(-1), r.argmax(-1)
    result.update(corrections_vs_reference=int(((new == y) & (old != y)).sum()),
        harms_vs_reference=int(((new != y) & (old == y)).sum()),
        prediction_changes=int((new != old).sum()))
    return result


def analyze(arrays):
    y = arrays['labels']
    text = arrays['availability'][:, 1].astype(bool)
    groups = dict(all=np.ones(len(y), dtype=bool), shuffle_eligible=arrays['eligible'],
                  no_history=~arrays['has_history'], T_present=text,
                  T_missing_seen=~text & arrays['seen_T'],
                  T_missing_never=~text & ~arrays['seen_T'])
    return {group: {mode: paired_metrics(y[keep], arrays[mode][keep], arrays['normal'][keep])
                    for mode in MODES} for group, keep in groups.items()}


def aggregate_records(records):
    """Main means require every requested original fold/rate cell, never pooling."""
    cells = {(row['fold'], row['rate']): row for row in records}
    if len(cells) != len(records):
        raise ValueError('Duplicate fold/rate audit records')
    scopes = []
    for fold in range(1, 6):
        scopes.append(('fold_eight_rate_mean', fold, 'mean', [(fold, r) for r in RATES]))
    for rate in RATES:
        scopes.append(('fivefold_rate_mean', None, rate, [(f, rate) for f in range(1, 6)]))
    scopes += [('fivefold_eight_rate_mean', None, 'mean', [(f, r) for f in range(1, 6) for r in RATES]),
               ('fivefold_high_rate_mean', None, 'high', [(f, r) for f in range(1, 6) for r in RATES[5:]])]
    result = []
    for scope, fold, rate, needed in scopes:
        if not all(cell in cells for cell in needed):
            continue
        rows = [cells[cell] for cell in needed]
        for group in rows[0]['groups']:
            for mode in rows[0]['groups'][group]:
                scores = [r['groups'][group][mode] for r in rows]
                present = [s for s in scores if s['N']]
                record = dict(scope=scope, seed=68, fold=fold, rate=rate, group=group,
                    mode=mode, cells=len(scores), nonempty_cells=len(present),
                    aggregation='equal fold/rate cell arithmetic mean; not pooled predictions')
                for key in ('ACC', 'UA', 'WF1'):
                    record[key] = float(np.mean([s[key] for s in present])) if present else None
                # Strata absent in some folds are explicitly labelled; not an all-fold cohort score.
                record['cohort_complete'] = len(present) == len(scores)
                result.append(record)
    return result


def write_report(output, records, identity):
    output = Path(output)
    full = {(f, r) for f in range(1, 6) for r in RATES}
    result = dict(status='complete' if {(r['fold'], r['rate']) for r in records} == full else 'subset_complete',
        identity=identity, per_fold_rate=records, aggregate=aggregate_records(records))
    (output/'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# IEMOCAP-6 seed68 frozen Signature path audit', '',
        'INTERNAL Test-oracle mechanism sensitivity only. No training, new selection or resampled test masks.',
        'ACC and UA are both reported; UA is macro recall over fixed classes0..5, including class0.',
        'Primary means weight each fold/rate cell equally, not each utterance. Empty cohorts are N/A.', '',
        '| Fold | Rate | Cohort | Mode | N | ACC | UA | W-F1 | Corrections / harms vs normal |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    fmt = lambda v: 'N/A' if v is None else f'{v:.4f}'
    csv_rows = []
    for item in records:
        for group, modes in item['groups'].items():
            for mode, score in modes.items():
                csv_rows.append(dict(scope='fold_rate', fold=item['fold'], rate=item['rate'],
                    group=group, mode=mode, **score))
                if group == 'all':
                    lines.append(f'| {item["fold"]} | {item["rate"]} | {group} | {mode} | {score["N"]} | '
                        + ' | '.join(fmt(score[k]) for k in ('ACC', 'UA', 'WF1'))
                        + f' | {score["corrections_vs_reference"]} / {score["harms_vs_reference"]} |')
    lines += ['', '## Equal-cell arithmetic summaries', '',
        '| Scope | Fold | Rate | Cohort | Mode | Nonempty / cells | ACC | UA | W-F1 |',
        '| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for row in result['aggregate']:
        lines.append('| ' + ' | '.join(str(row.get(k)) for k in ('scope', 'fold', 'rate', 'group', 'mode'))
            + f' | {row["nonempty_cells"]}/{row["cells"]} | '
            + ' | '.join(fmt(row[k]) for k in ('ACC', 'UA', 'WF1')) + ' |')
    lines += ['', '## Limits', '',
        '- Endpoint-preserving shuffles test only interior content/time assignment, not all history order.',
        '- Removing antisymmetric level-two area preserves the first level and symmetric terms; this is a frozen, potentially out-of-distribution intervention.',
        '- CHM itself remains order-sensitive. This audit cannot prove Flat is order-blind.',
        '- Six-class UA includes zero recall for classes absent in a cohort, consistent with the original fixed-six-class metric.',
        '- Seed68 was selected for inspection after observing results; this is exploratory, not independent generalization evidence.',
        '- Per-class recall/support and cohort rows are in JSON/CSV; no incomplete fivefold/eight-rate main summary is emitted.']
    (output/'RESULT.md').write_text('\n'.join(lines) + '\n')
    csv_rows += result['aggregate']
    fields = list(dict.fromkeys(k for row in csv_rows for k in row))
    with (output/'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1, 6)))
    parser.add_argument('--rates', nargs='+', default=list(RATES))
    args = parser.parse_args(argv)
    folds, rates = validate_scope(args.folds, args.rates)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare
    from experiments.osram_signature_iemocap_cyclic_20261025.summarize import selected_row
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from osram_four_block_wiring import SignatureBlock
    from .controls import descriptor_control
    torch.set_num_threads(1)
    repair_uuid_rng(torch)
    identity = dict(dataset='IEMOCAPSix', seed=68, audit_source_commit=args.source_commit,
        numerical_source_commit=NUMERICAL_SOURCE_SHA, numerical_source=str(prepare.ROOT),
        gpu_uuid=args.gpu_uuid, python=sys.executable, torch=torch.__version__, numpy=np.__version__,
        mode='frozen inference only', mask_protocol='gcnet-random-mask-v1',
        shuffle_seeds=[6601, 6602, 6603], source_hashes={p.name:h.sha(p) for p in
            (Path(__file__), Path(__file__).with_name('controls.py'))})
    records, tick = [], time.monotonic()
    h.write_json(args.output/'STATUS.json', dict(status='running', identity=identity, pid=os.getpid()))
    try:
        for fold in folds:
            for rate in rates:
                source = args.selected_root/f'IEMOCAPSix_68_fold{fold}_{rate}'
                checked = selected_row(dict(seed=68, fold=fold, rate=rate, output=str(source)), NUMERICAL_SOURCE_SHA)
                summary = json.loads((source/'SUMMARY.json').read_text())
                if summary['identity']['stage2_epochs'] != 100:
                    raise ValueError('Only completed original100epoch Signature runs are accepted')
                saved = torch.load(source/summary['best']['checkpoint'], map_location='cpu', weights_only=False)
                canonical = lambda value: json.loads(json.dumps(value))
                if (canonical(saved['identity']) != summary['identity'] or saved['epoch'] != summary['best']['epoch']
                        or saved['parent_id'] != summary['parent_id']):
                    raise ValueError('Selected Signature checkpoint identity/epoch/parent mismatch')
                for name, digest in summary['identity']['source_hashes'].items():
                    if h.sha(prepare.ROOT/name) != digest:
                        raise ValueError('Original immutable numerical source changed: ' + name)
                bundle, reader, parent = prepare.setup(68, fold, args.gpu_uuid, args.output/f'fold{fold}_{rate}', rate)
                if parent['parent_id'] != summary['parent_id']:
                    raise ValueError('Different frozen Flat checkpoint')
                cached = prepare.test_cache(bundle, reader, rate)
                block = SignatureBlock().to('cuda:0').eval().requires_grad_(False)
                block.load_state_dict(saved['block'], strict=True)
                parent_digest, block_digest = h.backbone_digest(reader.parent), h.backbone_digest(block)
                pieces = []
                with torch.no_grad():
                    for batch_index, row in enumerate(cached):
                        frame = row['frame'].to('cuda:0')
                        valid = frame.valid.T
                        observed = frame.availability & frame.valid[..., None]
                        seen = observed.long().cumsum(0)-observed.long() > 0
                        history = frame.valid.long().cumsum(0)-frame.valid.long() > 0
                        normal, eligible, _ = descriptor_control(block, frame, 'normal')
                        torch.testing.assert_close(normal, block.descriptor(frame), rtol=0, atol=0)
                        piece = dict(row['metadata'], availability=row['availability'],
                            eligible=eligible.T[valid].cpu().numpy(),
                            has_history=history.T[valid].cpu().numpy(), seen_T=seen[..., 1].T[valid].cpu().numpy(),
                            flat=frame.prediction.transpose(0, 1)[valid].cpu().numpy())
                        for mode in MODES[1:]:
                            descriptor = normal
                            if mode != 'normal':
                                kind = 'interior_shuffle' if mode.startswith('interior_shuffle') else mode
                                seed = int(mode.rsplit('_', 1)[-1]) if kind == 'interior_shuffle' else 68
                                descriptor, _, _ = descriptor_control(block, frame, kind,
                                    shuffle_seed=seed+batch_index*100003)
                            delta = block.out(descriptor)*(frame.valid & history)[..., None]
                            prediction = reader.decode_delta(frame, delta)
                            piece[mode] = prediction.transpose(0, 1)[valid].cpu().numpy()
                        pieces.append(piece)
                arrays = {key:np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
                with np.load(source/summary['best']['predictions'], allow_pickle=False) as old:
                    for key in ('labels', 'conversation_ids', 'utterance_indices', 'availability'):
                        np.testing.assert_array_equal(arrays[key], old[key])
                    np.testing.assert_array_equal(arrays['normal'], old['prediction'])
                    np.testing.assert_array_equal(arrays['flat'], old['base_prediction'])
                if parent_digest != h.backbone_digest(reader.parent) or block_digest != h.backbone_digest(block):
                    raise RuntimeError('Read-only audit changed weights')
                if any(p.grad is not None for p in list(reader.parent.parameters())+list(block.parameters())):
                    raise RuntimeError('Read-only audit created gradients')
                mask_sha = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
                if mask_sha != summary['identity']['test_mask_sha256']:
                    raise ValueError('Selected test mask changed')
                record = dict(seed=68, fold=fold, rate=rate, selected_epoch=checked['selected_epoch'],
                    selected_summary_sha256=h.sha(source/'SUMMARY.json'),
                    selected_checkpoint_sha256=summary['best']['checkpoint_sha256'],
                    selected_prediction_sha256=summary['best']['prediction_sha256'], parent_id=parent['parent_id'],
                    test_mask_sha256=mask_sha, replay_max_absolute_error=0., logits_bitexact=True,
                    parent_unchanged=True, block_unchanged=True, all_grad_none=True,
                    eligible_count=int(arrays['eligible'].sum()), groups=analyze(arrays))
                np.savez_compressed(args.output/f'fold{fold}_rate_{rate}_predictions.npz', **arrays)
                h.write_json(args.output/f'fold{fold}_rate_{rate}.json', record)
                records.append(record)
                print(json.dumps(dict(fold=fold, rate=rate, completed=len(records),
                    scores=record['groups']['all'], seconds=time.monotonic()-tick)), flush=True)
                del bundle, reader, block, cached, saved, pieces
                gc.collect()
                torch.cuda.empty_cache()
        identity.update(seconds=time.monotonic()-tick,
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        for name, digest in identity['source_hashes'].items():
            if h.sha(Path(__file__).with_name(name)) != digest:
                raise RuntimeError('Audit source changed during inference')
        result = write_report(args.output, records, identity)
        h.write_json(args.output/'STATUS.json', dict(status=result['status'], identity=identity,
            completed_cells=len(records), original_predictions_bitexact=True, no_training=True))
    except BaseException as error:
        h.write_json(args.output/'STATUS.json', dict(status='failed', identity=identity,
            error=repr(error), completed_cells=[(r['fold'], r['rate']) for r in records]))
        raise


if __name__ == '__main__':
    main()
