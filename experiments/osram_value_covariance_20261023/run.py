"""Frozen MOSI/MOSEI history-Value covariance and actual task-response replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

RATES = ('0.3', '0.7')
SIGMAS = (.001, .01)
DRAWS = 16


def validate_scope(dataset, rates=RATES, draws=DRAWS, sigmas=SIGMAS):
    if (dataset not in ('CMUMOSI', 'CMUMOSEI') or
            tuple(map(float, rates)) != (.3, .7) or draws != DRAWS or tuple(sigmas) != SIGMAS):
        raise ValueError('Locked scope: MOSI/MOSEI66, rates .3/.7, 16 draws, sigma .001/.01')


def validate_original_bundle(bundle):
    """Fail closed for both datasets, including post-read wrappers and objectives."""
    cfg, model = bundle['cfg'], bundle['model']
    expected = dict(seed=66, backbone_type='osram', latent_dim=256,
        osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_write_step=.6, beta_mode='embedded', osram_readout_fusion='flat',
        osram_ablation='full', osram_emotion_ablation='full', osram_gap_read='residual',
        gap_residual_strength=1., osram_meaningful_block='none', osram_readout_candidate='none',
        osram_counterfactual_readout='none', completion_path='none', core20_method='none',
        readout_type='shared', fusion_type='mean', training_objective='emotion-only',
        task_regression_loss='mse', mosi_task_mode='regression', missing_mask_protocol='gcnet-random-mask-v1')
    for name, value in expected.items():
        if getattr(cfg, name, None) != value:
            raise ValueError('Original Flat config required: ' + name)
    forbidden = ('osram_bidirectional', 'osram_forward_slot_reuse', 'osram_disable_local_skip',
        'osram_base_first_gap', 'osram_history_recency', 'osram_causal_gru_history',
        'osram_source_tracking', 'osram_tasr', 'osram_source_role', 'osram_source_relation',
        'osram_source_role_v3', 'osram_conditional_reread', 'osram_relation_block',
        'osram_relation_dual_readout', 'osram_gap_increment_filter', 'osram_decision_correction',
        'osram_history_query_adapter', 'osram_post_grn', 'osram_local_skip_gate',
        'osram_memory_only_adapter', 'osram_history_input_gate', 'osram_local_evidence_gate',
        'osram_hierarchical_evidence_gate', 'osram_ced_block', 'paired_history_views',
        'classification_completion', 'local_context_residual', 'node_interaction_residual', 'text_core')
    if any(getattr(cfg, name, False) for name in forbidden):
        raise ValueError('Extra module/ablation is not allowed in original Flat diagnostic')
    if getattr(cfg, 'osram_adapter_hidden_dim', 0) not in (0, 1600):
        raise ValueError('Do not substitute reduced-width Flat')
    if sum(p.numel() for p in model.parameters()) != 13509793:
        raise ValueError('Original LargeFlat parameter count changed')
    if model.osram.beta_mode != 'embedded' or model.osram.forward_slot_reuse:
        raise ValueError('Original embedded-beta causal forward-half interface required')


def setup(dataset, reference, gpu_uuid, output):
    if dataset == 'CMUMOSI':
        from experiments.osram_kv_task_utility_20261022.run import setup as original
        bundle = original(reference, gpu_uuid, output)
        validate_original_bundle(bundle)
        return bundle
    import torch
    import sklearn
    from experiments.osram_mosei_gcnet_router_20261020 import worker
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_kv_task_utility_20261022.run import validate_mask_bank
    if (torch.__version__, np.__version__, sklearn.__version__) != ('2.2.2+cu121', '1.24.3', '1.3.2'):
        raise ValueError('Keep accepted s0 Torch/NumPy/sklearn environment')
    gpu = guard_gpu(gpu_uuid)
    manifest, proof = worker.load_manifest()
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(output.parent / 'feature_cache' / dataset)
    from gcnet_missing_m3 import train_gcnet as trainer
    summary = json.loads((reference / 'SUMMARY.json').read_text())
    cfg = trainer.TrainConfig(**summary['identity']['config'])
    if (cfg.dataset != dataset or cfg.seed != 66 or cfg.missing_mask_protocol != 'gcnet-random-mask-v1'
            or cfg.osram_meaningful_block != 'none' or cfg.osram_write_step != .6
            or cfg.osram_num_heads != 8 or cfg.osram_key_dim != 64 or cfg.osram_value_dim != 64
            or cfg.osram_output_dim != 1600 or cfg.osram_bidirectional
            or cfg.osram_readout_fusion != 'flat' or cfg.osram_disable_local_skip
            or summary.get('identity', {}).get('mode', 'flat') != 'flat'):
        raise ValueError('Require original seed66 causal LargeFlat head64 with Local Skip')
    train, dev, test, *dims = worker.build_loaders(manifest, cfg, trainer)
    splits = {name: loader.protocol_metadata for name, loader in
              [('train', train[0]), ('validation', dev[0]), ('test', test[0])]}
    if splits != summary['identity']['split']:
        raise ValueError('Protected MOSEI split metadata changed')
    selected = summary['best_by_rate']['0.3']
    checkpoint = reference / selected['checkpoint']
    if h.sha(checkpoint) != selected['checkpoint_sha256'] or h.sha(reference / selected['predictions']) != selected['prediction_sha256']:
        raise ValueError('Protected checkpoint/prediction SHA256 changed')
    model = _build_model(cfg, tuple(dims), torch.device('cuda:0'))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=True)
    model.eval().requires_grad_(False)
    masks = torch.load(reference / 'fixed_masks.pt', map_location='cpu')['test']
    mask_hashes = validate_mask_bank(masks, summary['identity']['mask_hashes']['test'])
    bundle = dict(model=model, trainer=trainer, loader=test[0], dimensions=tuple(dims), cfg=cfg,
        masks=masks, summary=summary, reference=reference, identity=dict(
            checkpoint=str(checkpoint), checkpoint_sha256=h.sha(checkpoint), selected_rate='0.3',
            selected_epoch=selected['epoch'], fixed_masks_sha256=h.sha(reference / 'fixed_masks.pt'),
            test_mask_hashes=mask_hashes, prediction_sha256=selected['prediction_sha256'],
            data_files=manifest['files'], data_manifest=proof, split=splits, config=asdict(cfg), gpu=gpu,
            environment=dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__)))
    validate_original_bundle(bundle)
    return bundle


class Capture:
    """Read-only capture of the real single scan inputs and original Flat Local."""
    def __init__(self, osram):
        from experiments.osram_meld_flat_source_combinations_20261013.evaluate import FlatReadoutCapture
        self.osram, self.flat, self.calls = osram, FlatReadoutCapture(osram), 0

    def __enter__(self):
        self.original = self.osram._scan
        self.had_instance = '_scan' in self.osram.__dict__
        self.instance = self.osram.__dict__.get('_scan')
        def capture(keys, values, queries, availability, valid, reverse, **kwargs):
            if reverse or self.calls:
                raise RuntimeError('Expected exactly one causal original Memory scan')
            self.calls += 1
            self.keys = {k: v.detach() for k, v in keys.items()}
            self.values = {k: v.detach() for k, v in values.items()}
            self.queries = queries.detach()
            self.availability, self.valid = availability.detach(), valid.detach()
            return self.original(keys, values, queries, availability, valid, reverse, **kwargs)
        self.osram._scan = capture
        self.flat.__enter__()
        return self

    def __exit__(self, *args):
        self.flat.__exit__(*args)
        if self.had_instance:
            self.osram._scan = self.instance
        else:
            del self.osram._scan


def run_rate(bundle, rate, stream, output, *, max_batches=None):
    import torch
    from .core import replay_value_noise, evaluate_sequence_draws, task_direction_variance
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018.runner_helpers import forward, isolated_rng
    model, trainer, loader = (bundle[k] for k in ('model', 'trainer', 'loader'))
    loader.sampler.set_epoch(0)
    selected = bundle['summary']['best_by_rate']['0.3']
    with np.load(bundle['reference'] / selected['predictions'], allow_pickle=False) as archive:
        saved = {k: archive[k].copy() for k in archive.files}
    cursor = batches = total_rows = 0
    max_parity = max_linear = max_linear_relative = 0.
    empirical_sqerror = analytic_sqnorm = 0.
    dataset = bundle['cfg'].dataset
    artifact_dir = output / 'covariance'
    artifact_dir.mkdir(exist_ok=True)
    with isolated_rng(66 + 99193):
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            data = trainer._move_batch(raw, next(model.parameters()).device)
            meta = metadata(data)
            bank = bundle['masks'][rate][index]
            for k, value in meta.items():
                np.testing.assert_array_equal(value, bank[k])
            a = bank['availability'].to(data[7].device)
            view = trainer._prepare_view_from_primary_masks(data, a, a, bundle['dimensions'])
            with torch.no_grad(), Capture(model.osram) as capture:
                prediction = forward(model, view)
            if capture.calls != 1 or capture.flat.local_calls != 1 or capture.flat.adapter_calls != 1:
                raise RuntimeError('Expected one original scan/Flat/Skip call')
            if index == 0:
                with torch.no_grad():
                    torch.testing.assert_close(forward(model, view), prediction, rtol=0, atol=0)
            valid = view['umask'].T.bool()
            if bool(a[~valid].any()) or bool((a.sum(-1)[valid] < 1).any()):
                raise ValueError('Saved masks must preserve padding and one observed modality')
            take = prediction.transpose(0, 1)[view['umask'].bool()].detach().cpu().numpy().reshape(-1)
            n = len(take)
            if rate == '0.3':
                for k, value in meta.items():
                    np.testing.assert_array_equal(value, saved[k][cursor:cursor+n])
                np.testing.assert_array_equal(a.transpose(0, 1)[view['umask'].bool()].cpu().numpy(), saved['availability'][cursor:cursor+n])
                np.testing.assert_allclose(take, saved['prediction'][cursor:cursor+n].reshape(-1), rtol=1e-6, atol=1e-6)
                np.testing.assert_array_equal(take > 0, saved['prediction'][cursor:cursor+n].reshape(-1) > 0)
            cursor += n
            arrays = {'G': [], 'kappa': [], 'cov_empirical': [], 't': [], 'batch_index': []}
            history = torch.zeros(valid.shape[1], dtype=torch.long, device=valid.device)
            rows = list(replay_value_noise(model.osram, capture.keys, capture.values,
                capture.queries, a, valid, draws=DRAWS, sigmas=SIGMAS, seed=66 + index))
            if [t for t, _ in rows] != list(range(valid.shape[0])):
                raise RuntimeError('Replay must preserve every original sequence time position')
            clean_sequence = torch.stack([row['clean'] for _, row in rows], dim=0)
            delta_sequence = torch.stack([row['deltas'] for _, row in rows], dim=2)
            covariance_sequence = torch.stack([row['G'] for _, row in rows], dim=0)
            # Preserve the production [T,B] Flat GEMM shape for clean, every
            # noisy draw and every head intervention. Per-time or flattened-draw
            # evaluation changes CUDA accumulation rounding at tiny noise scales.
            with torch.no_grad():
                sequence_predictions = evaluate_sequence_draws(model, capture.flat.local,
                    clean_sequence, delta_sequence, valid)
            clean_predictions = sequence_predictions['clean_prediction'].squeeze(-1)
            torch.testing.assert_close(clean_predictions[valid], prediction[..., 0][valid], rtol=1e-6, atol=1e-6)
            max_parity = max(max_parity, float((clean_predictions[valid] - prediction[..., 0][valid]).abs().max()))
            noisy_predictions = sequence_predictions['noisy_predictions'].squeeze(-1)
            head_predictions = sequence_predictions['head_predictions'].squeeze(-1)
            task_variances = task_direction_variance(model, capture.flat.local,
                clean_sequence, covariance_sequence, valid).detach()
            for t, row in rows:
                for key in ('clean', 'deltas', 'G', 'kappa', 'cov_empirical'):
                    if not torch.isfinite(row[key]).all():
                        raise RuntimeError('Nonfinite replay tensor: ' + key)
                if float(row['kappa'].min()) < -1e-9:
                    raise RuntimeError('Negative covariance diagonal')
                active = valid[t]
                if not bool(active.any()):
                    continue
                replay = clean_predictions[t]
                noisy = noisy_predictions[:, :, t]
                head_pred = head_predictions[:, t]
                taskvar = task_variances[t]
                G, empirical = row['G'][active], row['cov_empirical'][active]
                empirical_sqerror += float((empirical.double() - G.double()).square().sum())
                analytic_sqnorm += float(G.double().square().sum())
                max_linear = max(max_linear, float(row['max_linear_abs_error']))
                max_linear_relative = max(max_linear_relative, float(row['linear_relative_error']))
                arrays['G'].append(G.detach().cpu().numpy())
                arrays['kappa'].append(row['kappa'][active].detach().cpu().numpy())
                arrays['cov_empirical'].append(empirical.detach().cpu().numpy())
                arrays['t'].append(np.full(int(active.sum()), t, dtype=np.int32))
                arrays['batch_index'].append(active.nonzero().flatten().cpu().numpy())
                for b in active.nonzero().flatten().tolist():
                    y = float(view['labels'][b, t]); p = float(replay[b])
                    role_active = torch.cat((active[b:b+1], ~a[t, b].bool()))
                    head_kappa_tensor = row['kappa'][b, :, role_active].mean(-1)
                    head_kappa = head_kappa_tensor.detach().cpu().tolist()
                    for si, sigma in enumerate(SIGMAS):
                        delta = row['deltas'][si, :, b]
                        result = dict(dataset=dataset, rate=rate, sigma=sigma,
                            conversation_id=str(view['conversation_ids'][b]), utterance_index=t,
                            pattern=''.join(m for m, bit in zip('ATV', a[t, b].tolist()) if bit),
                            history_count=int(history[b]), label=y, clean_prediction=p,
                            clean_correct='neutral' if y == 0 else bool((p > 0) == (y > 0)),
                            role_active=role_active.cpu().tolist(),
                            clean_margin=abs(p), noisy_predictions=noisy[si, :, b].detach().cpu().tolist(),
                            kappa_mean=float(head_kappa_tensor.mean()),
                            read_noise_rms=float(delta.square().mean().sqrt()), value_rms=float(row['value_rms']),
                            task_variance=float(taskvar[b].sum()), head_kappa=head_kappa,
                            first_order_prediction_shift_rms=float(sigma * taskvar[b].sum().sqrt()),
                            head_intervention_sigma=.01, head_intervention_draw=0,
                            head_task_variance=taskvar[b].cpu().tolist(),
                            head_prediction_shift=(head_pred[:, b] - p).abs().detach().cpu().tolist())
                        stream.write(json.dumps(result, allow_nan=False) + '\n')
                        total_rows += 1
                history += active.long()
            if arrays['G']:
                np.savez_compressed(artifact_dir / f'rate_{rate}_batch_{index:04d}.npz',
                    **{k: np.concatenate(v) for k, v in arrays.items()},
                    conversation_ids=np.asarray([str(x) for x in view['conversation_ids']]))
            stream.flush(); batches += 1
            print(f'{dataset} rate={rate} batch={index+1} N={cursor} parity={max_parity:.3g} linear={max_linear:.3g}', flush=True)
    if max_batches is None and (cursor != len(saved['labels']) or batches != len(bundle['masks'][rate])):
        raise RuntimeError('Incomplete all-test-utterance coverage')
    return dict(utterances=cursor, rows=total_rows, batches=batches, full_test_coverage=max_batches is None,
        max_clean_prediction_parity_error=max_parity, max_linear_abs_error=max_linear,
        max_linear_relative_error=max_linear_relative,
        covariance_relative_frobenius_error=(empirical_sqerror / analytic_sqnorm) ** .5 if analytic_sqnorm else 0.,
        covariance_comparison='MC pooled 16 draws times value coordinates; finite-sampling error expected',
        saved_miss03_prediction_parity=rate == '0.3', fixed_miss03_checkpoint=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('CMUMOSI', 'CMUMOSEI'), required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    validate_scope(args.dataset)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_tasr_20261018.runner_helpers import backbone_digest, write_json, sha
    if args.output.exists():
        raise FileExistsError('Use a fresh independent diagnostic output')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(__file__).with_name('core.py'), Path(__file__).with_name('report.py')]
    sources += [root / name for name in ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py',
        'gcnet_missing_m3/train_gcnet.py', 'gcnet_modality_jepa/gcnet_legacy_random_mask.py')]
    state = dict(status='preflight', label='INTERNAL DIAGNOSTIC ONLY', dataset=args.dataset,
        source_commit=args.source_commit, source_hashes={str(p.relative_to(root)): sha(p) for p in sources},
        pid=os.getpid(), server='biggpu', smoke=args.smoke, sigmas=SIGMAS, draws=DRAWS, new_training_steps=0)
    write_json(args.output / 'STATUS.json', state); start = time.monotonic()
    try:
        torch.set_num_threads(1)
        bundle = setup(args.dataset, args.reference, args.gpu_uuid, args.output)
        model = bundle['model']; before = backbone_digest(model)
        state.update(status='running', identity=bundle['identity'], model_digest=before, rates={})
        write_json(args.output / 'STATUS.json', state)
        torch.cuda.reset_peak_memory_stats()
        with gzip.open(args.output / 'ROWS.jsonl.gz', 'wt') as stream:
            for rate in (RATES[:1] if args.smoke else RATES):
                state['rates'][rate] = run_rate(bundle, rate, stream, args.output, max_batches=1 if args.smoke else None)
                write_json(args.output / 'STATUS.json', state)
        if backbone_digest(model) != before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen diagnostic changed original weights or parameter gradients')
        from .report import main as report_main
        report_main(['--rows', str(args.output / 'ROWS.jsonl.gz'), '--output', str(args.output)])
        state.update(status='complete', weights_unchanged_exact=True, rows_sha256=sha(args.output / 'ROWS.jsonl.gz'),
                     seconds=time.monotonic() - start, peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20)
    except BaseException as error:
        state.update(status='failed', error=repr(error), seconds=time.monotonic() - start)
        raise
    finally:
        write_json(args.output / 'STATUS.json', state)


if __name__ == '__main__':
    main()
