"""Same-weight original/common-root MOSEI N10 readout replay, no training."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/data2/yb/remote_experiments/osram_mosei_flat128_nested_20261024/execution_v1/runs/n10')
GPU = 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'
RATES = tuple(f'{i / 10:.1f}' for i in range(8))
PATTERNS = {'A': (1, 0, 0), 'T': (0, 1, 0), 'V': (0, 0, 1),
            'AT': (1, 1, 0), 'AV': (1, 0, 1), 'TV': (0, 1, 1), 'ATV': (1, 1, 1)}


def group_metrics(trainer, y, full, common, availability, extra=None):
    from experiments.osram_mosei_n10_scale_20261024.run import paired_counts
    groups = {'ALL': np.ones(len(y), dtype=bool), **{
        name: (availability == pattern).all(-1) for name, pattern in PATTERNS.items()}}
    groups.update(extra or {})
    output = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        row = dict(N_all_labels=n, N_non_neutral=int((mask & (y != 0)).sum()))
        for method, values in (('original', full), ('common_root', common)):
            metrics = trainer._metrics('CMUMOSEI', y[mask], values[mask], 'regression') if row['N_non_neutral'] else None
            row[method] = dict(ACC=100 * metrics['accuracy'] if metrics else None,
                              WF1=100 * metrics['weighted_f1'] if metrics else None,
                              MSE=float(np.mean((values[mask] - y[mask]) ** 2)) if n else None)
        row['delta_WF1_common_minus_original_pp'] = (
            row['common_root']['WF1'] - row['original']['WF1'] if row['N_non_neutral'] else None)
        row['original_vs_common'] = paired_counts(y, common, full, mask)
        row['common_vs_original'] = paired_counts(y, full, common, mask)
        row['mean_abs_prediction_difference'] = float(np.mean(np.abs(common[mask] - full[mask]))) if n else None
        row['max_abs_prediction_difference'] = float(np.max(np.abs(common[mask] - full[mask]))) if n else None
        output[name] = row
    return output


def report(output, summary):
    lines = ['# Frozen root-specificity removal', '', 'INTERNAL DIAGNOSTIC ONLY',
             'Same N10 checkpoints/masks; no training. Only [x_i,r_i,r_bar] -> [x_i,r_bar,r_bar].',
             'Original-vs-common corrections mean original correct / common wrong; harms mean the reverse.',
             'Pooled groups repeat utterances across rates; their W-F1 is not the eight-rate mean.', '']
    if 'aggregate' in summary:
        lines += ['| Readout | Mean8 ACC | Mean8 W-F1 | High W-F1 |', '|---|---:|---:|---:|']
        for name, row in summary['aggregate'].items():
            lines.append(f"|{name}|{row['mean']['ACC']:.6f}|{row['mean']['WF1']:.6f}|{row['high']['WF1']:.6f}|")
    if 'pooled' in summary:
        lines += ['', '| Current pattern | N(nonzero) | Original W-F1 | Common-root W-F1 | Delta | Original corrections | Original harms |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name, row in summary['pooled'].items():
            f, c, pair = row['original']['WF1'], row['common_root']['WF1'], row['original_vs_common']
            def fmt(v):
                return 'N/A' if v is None else f'{v:.6f}'
            lines.append(f"|{name}|{row['N_non_neutral']}|{fmt(f)}|{fmt(c)}|{fmt(row['delta_WF1_common_minus_original_pp'])}|{pair['corrections']}|{pair['harms']}|")
    lines += ['', 'Status: ' + summary['status'],
              'This frozen, possibly out-of-distribution intervention does not prove a mechanism or generalization benefit.']
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    with (output / 'PER_RATE.csv').open('w', newline='') as stream:
        fields = ['rate', 'group', 'N_all_labels', 'N_non_neutral', 'original_WF1', 'common_root_WF1',
                  'delta_pp', 'original_corrections', 'original_harms']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rate, item in summary['rates'].items():
            for name, row in item['groups'].items():
                writer.writerow(dict(rate=rate, group=name, N_all_labels=row['N_all_labels'],
                    N_non_neutral=row['N_non_neutral'], original_WF1=row['original']['WF1'],
                    common_root_WF1=row['common_root']['WF1'], delta_pp=row['delta_WF1_common_minus_original_pp'],
                    original_corrections=row['original_vs_common']['corrections'],
                    original_harms=row['original_vs_common']['harms']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    from .intervention import suppress_root_specificity
    from experiments.osram_nested_cmerc_tuning_20261023.run import verify_source_snapshot
    from experiments.osram_mosei_flat128_nested_20261024.worker import setup
    from experiments.osram_mosei_n10_scale_20261024.run import aggregate, validate_checkpoint_identity
    from experiments.osram_nested_delta_off_missing_text_20261010.readout import flat_prediction
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite prior evidence')
    manifest = verify_source_snapshot(ROOT, args.source_commit)
    free = int(subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    if free < 6144:
        raise RuntimeError('GPU1 needs at least6144MiB free; no other job is modified')
    source = json.loads((SOURCE / 'SUMMARY.json').read_text())
    if (source['status'], source['method'], source['epoch']) != ('complete', 'mosei_f128_n10', 60):
        raise ValueError('Require completed MOSEI N10 seed66')
    for name, digest in source['identity']['source_hashes'].items():
        if name.startswith(('gcnet_missing_m3/', 'gcnet_modality_jepa/')) and h.sha(ROOT / name) != digest:
            raise ValueError('Numerical model source changed: ' + name)
    args.output.mkdir(parents=True)
    tick = time.monotonic()
    summary = dict(status='preflight', label='INTERNAL DIAGNOSTIC ONLY', source_commit=args.source_commit,
        source_manifest_sha256=manifest['manifest_sha256'], source=str(SOURCE),
        source_summary_sha256=h.sha(SOURCE / 'SUMMARY.json'), gpu_uuid=GPU, pid=os.getpid(),
        intervention='readout([x_i,r_i,r_bar]) -> readout([x_i,r_bar,r_bar])',
        training_updates=0, rates={})
    scans = [0]
    all_arrays = []
    try:
        torch.set_num_threads(1)
        bundle, plain = setup('CMUMOSEI', 'mosei_f128_n10', GPU, args.output / 'data', seed=66, fold=1, variant='N10')
        del plain
        cfg = json.loads(json.dumps(bundle['cfg'].__dict__))
        if cfg != source['config'] or bundle['environment'] != source['environment']:
            raise ValueError('Effective config/environment differs from N10')
        model = bundle['model'].eval().requires_grad_(False)
        trainer, loader = bundle['trainer'], bundle['test_loader']
        core = model.osram.meaningful_block.core.core
        if core.plain_gin:
            raise ValueError('Require rooted, not plain GIN')
        summary.update(status='running', config=cfg, environment=bundle['environment'], gpu=bundle['gpu'],
                       data=bundle['data'], mask_reference=bundle['mask_reference'])
        handle = model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0] + 1))
        try:
            for rate in RATES:
                selected = source['best_by_rate'][rate]
                checkpoint, prediction = SOURCE / selected['checkpoint'], SOURCE / selected['predictions']
                if h.sha(checkpoint) != selected['checkpoint_sha256'] or h.sha(prediction) != selected['prediction_sha256']:
                    raise ValueError('Protected checkpoint/prediction changed')
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                validate_checkpoint_identity(state, selected, source['config'])
                model.load_state_dict(state['model'], strict=True)
                del state
                before = h.backbone_digest(model)
                with np.load(prediction, allow_pickle=False) as saved:
                    expected = {k: saved[k].copy() for k in saved.files}
                loader.sampler.set_epoch(0)
                fulls, commons = [], []
                cursor, start_scans, calls, removed_max = 0, scans[0], 0, 0.
                with h.isolated_rng(99259), torch.no_grad():
                    for index, raw in enumerate(loader):
                        data = trainer._move_batch(raw, torch.device('cuda:0'))
                        meta = metadata(data)
                        n = len(meta['labels'])
                        for key, value in meta.items():
                            np.testing.assert_array_equal(value, expected[key][cursor:cursor + n])
                        record = bundle['masks'][rate][index]
                        for key, value in meta.items():
                            np.testing.assert_array_equal(value, record[key])
                        av = record['availability'].to(data[7].device)
                        view = trainer._prepare_view_from_primary_masks(data, av, av, bundle['dimensions'])
                        valid = data[7].bool()
                        np.testing.assert_array_equal(av.transpose(0, 1)[valid].cpu().numpy(),
                                                      expected['availability'][cursor:cursor + n])
                        capture = {}
                        def pre(_module, inputs):
                            if capture:
                                raise RuntimeError('Unexpected multiple input-adapter calls')
                            capture['raw'] = tuple(x.detach().clone() for x in inputs)
                        def post(_module, _inputs, outputs):
                            capture['corrected'] = tuple(x.detach().clone() for x in outputs)
                        hooks = (model.osram.meaningful_block.register_forward_pre_hook(pre),
                                 model.osram.meaningful_block.register_forward_hook(post))
                        try:
                            direct = h.forward(model, view).squeeze(-1)
                        finally:
                            for hook in hooks:
                                hook.remove()
                        original = flat_prediction(model, *capture['raw'], readout=capture['corrected'])
                        np.testing.assert_allclose(original.T[valid].cpu().numpy(), direct.T[valid].cpu().numpy(),
                                                   atol=1e-6, rtol=1e-6)
                        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                        with suppress_root_specificity(core) as stats:
                            changed = model.osram.meaningful_block(*capture['raw'])
                        common = flat_prediction(model, *capture['raw'], readout=changed)
                        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state()):
                            raise RuntimeError('Frozen GNN/readout replay advanced RNG')
                        calls += stats['calls']
                        removed_max = max(removed_max, stats['max_removed_root_difference'])
                        for old, new in zip(capture['corrected'][1:], changed[1:]):
                            if not torch.equal(old[..., 512:], new[..., 512:]):
                                raise RuntimeError('Backward half changed')
                        if bool(core.readout._forward_pre_hooks):
                            raise RuntimeError('Intervention hook leaked')
                        for p, target in ((original, fulls), (common, commons)):
                            values = p.T[valid].cpu().numpy()
                            if not np.isfinite(values).all():
                                raise RuntimeError('Nonfinite prediction')
                            target.append(values)
                        cursor += n
                if cursor != len(expected['labels']) or scans[0] - start_scans != len(loader):
                    raise RuntimeError('Incomplete replay or extra Memory scan')
                full, common = np.concatenate(fulls), np.concatenate(commons)
                stored = expected['prediction'].reshape(-1)
                np.testing.assert_allclose(full, stored, atol=1e-6, rtol=1e-6)
                np.testing.assert_array_equal(full > 0, stored > 0)
                first = expected['utterance_indices'] == 0
                np.testing.assert_allclose(common[first], full[first], atol=1e-6, rtol=0)
                np.testing.assert_array_equal(common[first] > 0, full[first] > 0)
                y, availability = expected['labels'], expected['availability']
                early_a_positive = ((availability == [1, 0, 0]).all(-1) & (y > 0)
                    & np.isin(expected['utterance_indices'], [1, 2]))
                groups = group_metrics(trainer, y, full, common, availability,
                                       {'early_A_positive': early_a_positive})
                file = args.output / f'predictions_miss_{rate.replace(".", "p")}.npz'
                np.savez_compressed(file, labels=y, original=full, common_root=common, availability=availability,
                    conversation_ids=expected['conversation_ids'], utterance_indices=expected['utterance_indices'])
                if h.backbone_digest(model) != before or h.sha(checkpoint) != selected['checkpoint_sha256']:
                    raise RuntimeError('Frozen model/checkpoint changed')
                summary['rates'][rate] = dict(selected_epoch=selected['epoch'], checkpoint_sha256=selected['checkpoint_sha256'],
                    original_prediction_sha256=selected['prediction_sha256'], predictions=str(file),
                    predictions_sha256=h.sha(file), groups=groups, root_readout_calls=calls,
                    max_removed_root_difference=removed_max, memory_scans=scans[0] - start_scans,
                    max_original_prediction_error=float(np.max(np.abs(full - stored))))
                all_arrays.append(dict(labels=y, full=full, common=common, availability=availability,
                                       early=early_a_positive))
                summary.update(seconds=time.monotonic() - tick, total_memory_scans=scans[0])
                h.write_json(args.output / 'SUMMARY.json', summary)
                report(args.output, summary)
                print(json.dumps(dict(rate=rate, original=groups['ALL']['original'], common=groups['ALL']['common_root'])), flush=True)
        finally:
            handle.remove()
        summary['aggregate'] = {name: aggregate({r: item['groups']['ALL'][name] for r, item in summary['rates'].items()})
                                for name in ('original', 'common_root')}
        joined = {key: np.concatenate([item[key] for item in all_arrays]) for key in all_arrays[0]}
        summary['pooled'] = group_metrics(trainer, joined['labels'], joined['full'], joined['common'],
            joined['availability'], {'early_A_positive': joined['early']})
        if verify_source_snapshot(ROOT, args.source_commit) != manifest:
            raise RuntimeError('Immutable source changed')
        summary.update(status='complete', seconds=time.monotonic() - tick, original_predictions_reproduced=True,
                       all_parameters_unchanged=True, one_memory_scan_per_batch=True,
                       protected_artifacts_preserved=True, rng_preserved=True)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        h.write_json(args.output / 'SUMMARY.json', summary)
        report(args.output, summary)


if __name__ == '__main__':
    main()
