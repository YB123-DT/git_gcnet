"""Frozen four-condition historical Text contribution diagnostic; no fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np

from experiments.osram_history_text_context_20261010.diagnostic import (
    select_plan, four_masks, contributions, summarize,
)

GPU7 = 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'
LABEL = 'INTERNAL DIAGNOSTIC ONLY; inherited per-rate Test-oracle checkpoints'


def evaluate(model, view, baseline, dimensions, limit=None):
    import torch
    from experiments.osram_history_drift_20261002.diagnostic import capture
    original = view['availability'].cpu().numpy()
    plans = []
    coverage = dict(targets=0, eligible=0, skipped=0, scans=0,
                    max_current_local_error=0., max_focal_local_error=0.,
                    max_baseline_prediction_error=0.)
    for b, length in enumerate(view['lengths']):
        for t in range(int(length)):
            coverage['targets'] += 1
            plan = select_plan(original[:int(length), b], t)
            if plan is None:
                coverage['skipped'] += 1
            else:
                plans.append(dict(plan, batch=b, target=t))
                coverage['eligible'] += 1
    if limit is not None:
        plans = plans[:limit]
    offsets = np.cumsum([0, *dimensions])
    rows = []
    # At most 16 separate trajectories/call; each four-arm case shares shape.
    for start in range(0, len(plans), 4):
        chunk = plans[start:start+4]
        total = max(p['target'] for p in chunk)+1
        features, availability, speaker, valid, lengths = [], [], [], [], []
        for plan in chunk:
            b, t = plan['batch'], plan['target']
            masks = four_masks(original[:int(view['lengths'][b]), b], t, plan)
            for arm in ('A+', 'A-', 'B+', 'B-'):
                x = view['incomplete'][:total, b].clone()
                padded_mask = np.zeros((total, 3), dtype=original.dtype)
                extent = min(total, len(masks[arm]))
                padded_mask[:extent] = masks[arm][:extent]
                a = torch.as_tensor(padded_mask, device=x.device, dtype=view['availability'].dtype)
                removed = original[:total, b] > padded_mask
                for m in range(3):
                    idx = torch.as_tensor(removed[:, m], device=x.device)
                    x[idx, offsets[m]:offsets[m+1]] = 0
                assert torch.equal(x[t], view['incomplete'][t, b])
                x[t+1:] = 0
                a[t+1:] = 0
                features.append(x)
                availability.append(a)
                speaker.append(view['qmask'][b, :total])
                v = torch.arange(total, device=x.device) <= t
                valid.append(v.to(view['umask'].dtype))
                lengths.append(t+1)
        x, a = torch.stack(features, 1), torch.stack(availability, 1)
        q, u = torch.stack(speaker), torch.stack(valid)
        with torch.no_grad():
            snap = capture(model, lambda: model([x], a, q, u, lengths, predict_missing=False))
        coverage['scans'] += 1
        for idx, plan in enumerate(chunk):
            b, t, i = plan['batch'], plan['target'], plan['text_position']
            indices = slice(4*idx, 4*idx+4)
            local = snap['local'][t, indices]
            current_error = float((local-local[:1]).abs().max())
            focal = snap['local'][i, indices]
            focal_error = max(float((focal[0]-focal[2]).abs().max()),
                              float((focal[1]-focal[3]).abs().max()))
            predictions = {arm: float(snap['prediction'][t, 4*idx+j].reshape(-1)[0])
                           for j, arm in enumerate(('A+', 'A-', 'B+', 'B-'))}
            base_error = abs(predictions['A+']-float(baseline['prediction'][t, b].reshape(-1)[0]))
            if max(current_error, focal_error, base_error) > 3e-6:
                raise AssertionError(f'paired isolation/parity failed: {current_error}, {focal_error}, {base_error}')
            for key, value in [('max_current_local_error', current_error),
                               ('max_focal_local_error', focal_error),
                               ('max_baseline_prediction_error', base_error)]:
                coverage[key] = max(coverage[key], value)
            label = float(view['labels'][b, t])
            row = dict(conversation=str(view['conversation_ids'][b]), target=t,
                       text_position=i, other_position=plan['other_position'],
                       other_modality=('A', 'T', 'V')[plan['other_modality']],
                       text_lag=t-i, other_lag=t-plan['other_position'],
                       text_other_distance=abs(i-plan['other_position']),
                       availability=original[t, b].tolist(),
                       pattern=''.join(m for m, bit in zip('ATV', original[t, b]) if bit),
                       label=label, predictions=predictions,
                       masks_sha256={arm: hashlib.sha256(mask[:t+1].tobytes()).hexdigest()
                                     for arm, mask in four_masks(original[:int(view['lengths'][b]), b], t, plan).items()},
                       **contributions(label, predictions))
            rows.append(row)
        print(json.dumps(dict(phase='paired_progress', completed=len(rows), planned=len(plans))), flush=True)
    coverage['evaluated'] = len(rows)
    return rows, coverage


def write_report(root, records):
    groups = {}
    for field in ('pattern', 'other_modality', 'text_lag_bin', 'other_lag_bin'):
        groups[field] = {}
        for record in records:
            groups[field][str(record['rate'])] = {}
            for row in record['rows']:
                if field.endswith('_bin'):
                    lag = row[field.replace('_bin', '')]
                    value = '1' if lag == 1 else ('2-4' if lag < 5 else '5+')
                else:
                    value = row[field]
                groups[field][str(record['rate'])].setdefault(value, []).append(row)
            groups[field][str(record['rate'])] = {k: summarize(v) for k, v in groups[field][str(record['rate'])].items()}
    summaries = {str(r['rate']): summarize(r['rows']) for r in records}
    from experiments.osram_frozen_memory_audit_20261009.run import write
    aggregate = {key: float(np.mean([s[key] for s in summaries.values() if s['n']]))
                 for key in ('delta_A_mean', 'delta_B_mean', 'interaction_mean', 'interaction_abs_mean')}
    aggregate['reversal_fraction'] = float(np.mean([(s['helpful_to_harmful']+s['harmful_to_helpful'])/s['n'] for s in summaries.values() if s['n']]))
    write(root/'SUMMARY.json', dict(label=LABEL, per_rate=summaries, equal_rate=aggregate, strata=groups))
    lines = ['# Fixed Text contribution under changed history', '', LABEL, '',
             'Original Flat seed66. Background deletion exactly one A/V bit outside the focal Text utterance. Nearest eligible Text; closest other deletable bit, deterministic position/modality tie-breaks. No labels used for planning, no training. Four independent causal trajectories; target Local and focal input pairing checked. Delta=MSE(Text absent)-MSE(Text present). Positive means helpful.', '',
             '| Rate | N | Delta A mean | Delta B mean | Mean abs J | Help→harm | Harm→help |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for rate, s in summaries.items():
        lines.append(f"| {rate} | {s['n']} | {s['delta_A_mean']:.6f} | {s['delta_B_mean']:.6f} | {s['interaction_abs_mean']:.6f} | {s['helpful_to_harmful']} | {s['harmful_to_helpful']} |")
    lines += ['', 'SUMMARY.json includes all four-condition eligible-subset W-F1, quantiles, fixed1e-6 sign deadband, rescue/harm counts and per-rate availability/deletion-modality/lag strata. The eligible subset is not the full test set. Repeated rates are not independent samples. Descriptive contribution reversals do not by themselves establish malfunction, a natural-language causal effect or an actionable new module. Single-seed, one background deletion strength; no threshold search.', '']
    (root/'RESULT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    uuid = subprocess.check_output(['nvidia-smi', '--id=7', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if uuid != GPU7 or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU7:
        raise ValueError('Healthy GPU7 UUID pin required; GPU4 forbidden')
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    from experiments.osram_frozen_memory_audit_20261009.run import write, now, extract
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(label=LABEL, status='running', pid=os.getpid(), started_utc=now(),
                  gpu=7, gpu_uuid=uuid, check_only=args.check_only, records=[], command=sys.argv)
    write(args.output/'STATUS.json', status)
    try:
        import torch
        from gcnet_modality_jepa.train_gcnet import get_loaders
        from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        from experiments.osram_readout_paths_20260930.run import state_hash, sha, split_ids
        from experiments.osram_frozen_memory_audit_20261009.probe import metrics
        torch.set_num_threads(2)
        torch.manual_seed(66)
        raw = json.loads((args.reference/'config.json').read_text())
        cfg = TrainConfig(**raw)
        assert cfg.dataset == 'CMUMOSI' and cfg.mosi_task_mode == 'regression' and cfg.task_regression_loss == 'mse'
        roots = [str(args.dataset_root/'CMUMOSI/features'/name) for name in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        train, val, test, *dims = get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
            num_folder=1, dataset='CMUMOSI', batch_size=cfg.batch_size, num_workers=0, seed=66,
            validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
        ids = split_ids(dict(train=train[0], validation=val[0], test=test[0]))
        device = torch.device('cuda:0')
        model = _build_model(cfg, tuple(dims)).to(device).requires_grad_(False).eval()
        reference = json.loads((args.reference/'metrics.json').read_text())
        code = {str(p): sha(p) for p in Path(__file__).parent.glob('*.py')}
        write(args.output/'PROVENANCE.json', dict(label=LABEL, host=platform.node(), torch=torch.__version__,
            source_config=raw, source_config_sha256=sha(args.reference/'config.json'), code_sha256=code,
            source_snapshot=json.loads((Path(_build_model.__code__.co_filename).parents[2]/'SNAPSHOT.json').read_text()),
            test_conversations=ids['test'], data_label_sha256=sha(args.dataset_root/'CMUMOSI/CMUMOSI_features_raw_2way.pkl')))
        records = []
        for rate in ([.7] if args.check_only else [i/10 for i in range(8)]):
            cp = args.reference/f'best_miss_{rate:.1f}.pt'.replace('.', 'p', 1)
            checkpoint_hash = sha(cp)
            saved = torch.load(cp, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'], strict=True)
            initial_hash = state_hash(model)
            data, views, mask_hash = extract(model, test[0], _schedules(cfg, 'test')[rate], tuple(dims), device, ids['test'])
            original = metrics(data['labels'], data['prediction'])
            assert abs(original['weighted_f1_nonzero']-reference['test'][str(rate)]['weighted_f1']) < 1e-10
            rows, coverage = [], []
            for cpu_view, cpu_base in views:
                view = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_view.items()}
                base = {k: v.to(device) for k, v in cpu_base.items()}
                r, c = evaluate(model, view, base, tuple(dims), limit=4 if args.check_only else None)
                rows.extend(r)
                coverage.append(c)
                if args.check_only:
                    break
            assert len({(r['conversation'], r['target']) for r in rows}) == len(rows)
            assert state_hash(model) == initial_hash and sha(cp) == checkpoint_hash
            record = dict(rate=rate, checkpoint=str(cp), checkpoint_sha256=checkpoint_hash,
                          checkpoint_epoch=int(saved['epoch']), original_full_test=original, mask_sha256=mask_hash,
                          frozen_unchanged=True, coverage=coverage, rows=rows)
            write(args.output/f'rate_{rate:.1f}.json', record)
            records.append(record)
            status['records'].append(dict(rate=rate, pairs=len(rows), frozen_unchanged=True))
            write(args.output/'STATUS.json', status)
            print(json.dumps(dict(phase='rate_complete', rate=rate, pairs=len(rows))), flush=True)
        write_report(args.output, records)
        status.update(status='complete', finished_utc=now())
    except BaseException:
        status.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        write(args.output/'STATUS.json', status)


if __name__ == '__main__':
    main()
