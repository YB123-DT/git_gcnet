"""Frozen Nested diagnostics: one causal scan, multiple masked readouts.

INTERNAL DIAGNOSTIC ONLY. No optimizer, training, checkpoint reselection, or
Flat baseline. Deleted sources are removed before graph interaction AND after
decoding, so no delta-Local or cross-evidence bypass can preserve their content.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VARIANTS = ('full', 'local_only', 'local_base', 'gap_off', 'base_off', 'no_local')
PATTERNS = {'A': (1, 0, 0), 'T': (0, 1, 0), 'V': (0, 0, 1),
            'AT': (1, 1, 0), 'AV': (1, 0, 1), 'TV': (0, 1, 1), 'ATV': (1, 1, 1)}


def config_for_device(config, device):
    """TrainConfig is frozen: device overrides must create a new instance."""
    return replace(config, device=str(device))


def safe_cosine(x, y):
    import torch
    norm = x.norm() * y.norm()
    return torch.where(norm > 1e-8, (x * y).sum() / norm.clamp_min(1e-8), norm.new_tensor(float('nan')))


def replay_hidden(osram, local, base, gap, availability, umask, variant):
    """Only recompute Nested + Flat readout; never encoder/query/write/scan.

    Availability stays truthful. Explicit active override independently removes
    ablated evidence from tokenizer topology. Local-only bypasses Nested.
    """
    import torch
    if variant not in VARIANTS:
        raise ValueError(variant)
    valid = umask.T.bool()
    mask = valid[..., None]
    gap_active = valid[..., None] & ~availability.bool()
    l = torch.where(mask, local, torch.zeros_like(local)).contiguous()
    b = torch.where(mask, base, torch.zeros_like(base))
    g = torch.where(gap_active[..., None], gap, torch.zeros_like(gap))
    keep_base = variant not in ('local_only', 'base_off')
    keep_gap = variant not in ('local_only', 'local_base', 'gap_off')
    keep_local = variant != 'no_local'
    if not keep_base:
        b = torch.zeros_like(b)
    if not keep_gap:
        g = torch.zeros_like(g)
    if not keep_local:
        l = torch.zeros_like(l)
    if variant != 'local_only':
        core = osram.meaningful_block.core
        old = core.forward
        def masked_core(this, current, evidence, active, av):
            active = active.clone()
            if not keep_base:
                active[..., 0] = False
            if not keep_gap:
                active[..., 1:] = False
            evidence = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
            return old(current, evidence, active, av)
        # Avoid changing the object at all for normal replay.
        if variant == 'full':
            read_l, b, g = osram.meaningful_block(l, b, g, availability, umask)
        else:
            had_instance_forward = 'forward' in core.__dict__
            core.forward = types.MethodType(masked_core, core)
            try:
                read_l, b, g = osram.meaningful_block(l, b, g, availability, umask)
            finally:
                if had_instance_forward:
                    core.forward = old
                else:
                    del core.forward
    else:
        read_l = l
    if not keep_base:
        b = torch.zeros_like(b)
    if not keep_gap:
        g = torch.zeros_like(g)
    if not keep_local:
        read_l = torch.zeros_like(read_l)
    read_l = torch.where(mask, read_l, torch.zeros_like(read_l))
    b = torch.where(mask, b, torch.zeros_like(b))
    g = torch.where(gap_active[..., None], g, torch.zeros_like(g))
    assert torch.count_nonzero(g[~gap_active]) == 0
    x = torch.cat((read_l, b, g.flatten(2)), -1)
    skip = osram.local_skip(l) if keep_local else torch.zeros_like(osram.local_skip(l))
    hidden = osram.emotion_norm(skip + osram.emotion_adapter(x))
    return torch.where(mask, hidden, torch.zeros_like(hidden))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_model(config, dimensions, device):
    """Same constructor arguments as train_gcnet, with dataset-specific shape."""
    from gcnet_missing_m3.model import MissingM3GraphModel
    from gcnet_missing_m3.train_gcnet import _dataset_shape
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import set_random_seed
    shape = _dataset_shape(config.dataset)
    set_random_seed(SeedBundle(config.seed).derive('missing_m3_model_init:fold:5'))
    keywords = {name: getattr(config, name) for name in inspect.signature(MissingM3GraphModel).parameters
                if hasattr(config, name)}
    for name in ('base_model', 'adim', 'tdim', 'vdim', 'D_e', 'graph_hidden_size'):
        keywords.pop(name, None)
    keywords.update(n_speakers=int(shape['num_speakers']), n_classes=int(shape['num_classes']),
                    time_attn=config.time_attention, no_cuda=device.type != 'cuda',
                    complete_state_jepa=config.training_objective == 'complete-state',
                    write_state_completion=config.training_objective == 'write-state',
                    future_state_jepa=config.training_objective == 'future-state')
    return MissingM3GraphModel(config.base_model, *dimensions, config.hidden, config.hidden // 2,
                              **keywords).to(device).eval().requires_grad_(False)


class Capture:
    """Capture pre-Nested input and original _scan tensors, never modified reads."""
    def __init__(self, model):
        self.model = model
        self.scans = 0
        self.query_rows = []

    def pre_block(self, module, args):
        self.inputs = tuple(x.detach().clone() for x in args)

    def profile(self, frame, event, arg):
        if event == 'return' and frame.f_code is self.model.osram._scan.__func__.__code__:
            from experiments.osram_gap_increment_audit_20261003.run import query_observables
            s = frame.f_locals
            if s['reverse']:
                raise RuntimeError('Expected one forward causal scan')
            self.query_rows = query_observables(s['queries'], s['base'], s['gap'], s['availability'],
                                               s['valid'], s['diagnostics'],
                                               self.model.osram.num_heads, self.model.osram.value_dim)
            self.scans += 1

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError('An existing Python profiler cannot be overwritten')
        self.handle = self.model.osram.meaningful_block.register_forward_pre_hook(self.pre_block)
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *args):
        sys.setprofile(None)
        self.handle.remove()


def metrics(dataset, labels, predictions, task_mode):
    import numpy as np
    from gcnet_missing_m3.train_gcnet import _metrics
    result = _metrics(dataset, labels, predictions, task_mode)
    if dataset in ('IEMOCAPFour', 'IEMOCAPSix'):
        from sklearn.metrics import recall_score
        count = 4 if dataset == 'IEMOCAPFour' else 6
        result['unweighted_accuracy'] = float(recall_score(labels, predictions, labels=list(range(count)),
                                                          average='macro', zero_division=0))
    return {k: (float(v) if np.isfinite(v) else None) for k, v in result.items()}


def evaluate(model, loader, config, dimensions, device, schedule=None, pattern=None,
             query_writer=None, query_metadata=None):
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import (_move_batch, _prepare_view,
        _prepare_view_from_primary_masks, _collect_predictions)
    collected = {v: [] for v in VARIANTS}
    ys, avs, ids = [], [], []
    query_totals = {}
    max_error = 0.0
    total_scans = 0
    with torch.no_grad():
        for raw in loader:
            data = _move_batch(raw, device)
            if pattern is None:
                view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
            else:
                valid = data[7].T.bool()
                availability = torch.zeros((*valid.shape, 3), dtype=torch.uint8, device=device)
                availability[valid] = torch.tensor(pattern, dtype=torch.uint8, device=device)
                view = _prepare_view_from_primary_masks(data, availability, availability, dimensions)
            with Capture(model) as capture:
                logits, _, _, predicted_missing = model([view['incomplete']], view['availability'],
                    view['qmask'], view['umask'], view['lengths'], predict_missing=False)
            if predicted_missing is not None or capture.scans != 1:
                raise RuntimeError('Unexpected auxiliary prediction or multiple memory scans')
            total_scans += capture.scans
            original, labels, _ = _collect_predictions(config.dataset, logits, view['labels'],
                                                       view['umask'], config.mosi_task_mode)
            valid = view['umask'].bool()
            batch_ids = [f"{view['conversation_ids'][b]}:{t}" for b, t in valid.nonzero().tolist()]
            assert len(batch_ids) == len(labels)
            ids.extend(batch_ids)
            ys.append(labels)
            avs.append(view['availability'].transpose(0, 1)[valid].cpu().numpy())
            for row in capture.query_rows:
                full_row = dict(row, sample_id=batch_ids[row['artifact_row']],
                                artifact_row=row['artifact_row'] + len(ids) - len(batch_ids),
                                **(query_metadata or {}))
                if query_writer is not None:
                    query_writer.writerow(full_row)
                cell = query_totals.setdefault(f"{row['modality']}/head_{row['head']}", {})
                for key, value in row.items():
                    if key in ('artifact_row', 'modality', 'head'):
                        continue
                    total = cell.setdefault(key, dict(count=0, defined_count=0, sum=0.))
                    total['count'] += 1
                    if np.isfinite(value):
                        total['defined_count'] += 1
                        total['sum'] += float(value)
            predictions = {}
            for variant in VARIANTS:
                if variant == 'local_base':
                    continue
                hidden = replay_hidden(model.osram, *capture.inputs, variant)
                replay_logits = model.smax_fc(hidden)
                predicted, expected, _ = _collect_predictions(config.dataset, replay_logits,
                    view['labels'], view['umask'], config.mosi_task_mode)
                np.testing.assert_array_equal(labels, expected)
                predictions[variant] = predicted
                if variant == 'full':
                    # Compare logits, not merely argmax; exact classes required.
                    delta = (replay_logits[view['umask'].T.bool()] - logits[view['umask'].T.bool()]).abs()
                    max_error = max(max_error, float(delta.max()) if delta.numel() else 0.)
                    torch.testing.assert_close(replay_logits[view['umask'].T.bool()],
                                               logits[view['umask'].T.bool()], rtol=2e-5, atol=2e-5)
                    np.testing.assert_allclose(predicted, original, rtol=2e-5, atol=2e-5)
                    if config.dataset.startswith('IEMOCAP'):
                        np.testing.assert_array_equal(predicted, original)
                    else:
                        np.testing.assert_array_equal(predicted > 0, original > 0)
            predictions['local_base'] = predictions['gap_off']
            for variant in VARIANTS:
                collected[variant].append(predictions[variant])
    labels = np.concatenate(ys)
    artifacts = dict(labels=labels, availability=np.concatenate(avs), sample_ids=np.asarray(ids))
    for variant in VARIANTS:
        artifacts['pred_' + variant] = np.concatenate(collected[variant])
    query_summary = {cell: {key: dict(count=t['count'], defined_count=t['defined_count'],
                          mean=t['sum']/t['defined_count'] if t['defined_count'] else None)
                          for key, t in values.items()} for cell, values in query_totals.items()}
    return artifacts, {variant: metrics(config.dataset, labels, artifacts['pred_' + variant],
                                       config.mosi_task_mode) for variant in VARIANTS}, query_summary, total_scans, max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    # Must precede importing trainer/dataloader/config; label files use this env.
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape, _schedules
    from gcnet_modality_jepa.train_gcnet import get_loaders
    if args.device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') == '4':
        raise RuntimeError('biggpu physical GPU4 is forbidden')
    args.output.mkdir(parents=True, exist_ok=True)
    status = dict(status='running', training=False, pid=os.getpid(), source=str(args.source),
                  gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), protocol='INTERNAL DIAGNOSTIC ONLY; source Test-oracle BEST')
    status['environment'] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda)
    status['code_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in
        [Path(__file__), ROOT/'gcnet_missing_m3/model.py', ROOT/'gcnet_missing_m3/osram.py',
         ROOT/'gcnet_missing_m3/train_gcnet.py', ROOT/'gcnet_missing_m3/meaningful_input.py',
         ROOT/'gcnet_missing_m3/meaningful_input_new40.py', ROOT/'gcnet_missing_m3/meaningful_new40_structure.py',
         ROOT/'gcnet_missing_m3/meaningful_blocks_common.py']}
    write(args.output / 'STATUS.json', status)
    try:
        config = TrainConfig(**json.loads((args.source / 'config.json').read_text()))
        from config import PATH_TO_LABEL
        label_path = Path(PATH_TO_LABEL[config.dataset])
        status['label_file'] = str(label_path)
        status['label_sha256'] = sha(label_path)
        status['config_sha256'] = sha(args.source / 'config.json')
        write(args.output / 'PROVENANCE.json', status)
        if (config.osram_meaningful_block != 'nested_gnn_rooted_evidence' or config.osram_bidirectional
                or config.training_objective != 'emotion-only' or config.osram_readout_fusion != 'flat'):
            raise ValueError('Source is not the finalized causal old Nested no-JEPA model')
        torch.set_num_threads(2)
        device = torch.device(args.device)
        config = config_for_device(config, device)
        folder = 'IEMOCAP' if config.dataset.startswith('IEMOCAP') else config.dataset
        feature_root = args.dataset_root / folder / 'features'
        roots = [str(feature_root / name) for name in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        _, _, test, ad, td, vd = get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
            num_folder=int(_dataset_shape(config.dataset)['num_folds']), dataset=config.dataset,
            batch_size=config.batch_size, num_workers=0, seed=config.seed,
            validation_fraction=config.validation_fraction, evaluation_protocol=config.evaluation_protocol)
        loader = test[config.fold - 1]
        dimensions = (ad, td, vd)
        model = build_model(config, dimensions, device)
        schedules = _schedules(config, 'test')
        source_metrics = json.loads((args.source / 'metrics.json').read_text())
        summary = dict(dataset=config.dataset, seed=config.seed, fold=config.fold,
            protocol=status['protocol'], training=False, random_rates={}, fixed_patterns={},
            config_sha256=sha(args.source / 'config.json'), source=str(args.source),
            intervention='pre-Nested content+topology mask and post-Nested hard mask; original memory scan unchanged',
            sample_id_semantics='conversation_id:zero_based_sequence_position; not original utterance filename',
            query_semantics='raw query and residual-addressed original pre-write OSRAM read; zero norm -> CSV NaN')
        query_stream = (args.output / 'query_observables.csv').open('w', newline='')
        query_fields = ['artifact_row', 'modality', 'head', 'cos_base_gap_query',
            'cos_gap_residual_query', 'cos_base_gap_read', 'rho', 'eta', 'norm_q_base',
            'norm_q_gap', 'norm_base_read', 'norm_gap_read', 'sample_id', 'mode', 'rate', 'pattern']
        query_writer = csv.DictWriter(query_stream, fieldnames=query_fields)
        query_writer.writeheader()
        checkpoints = {}
        jobs = [('random', str(i / 10), i / 10, None) for i in range(8)]
        jobs += [('fixed', p, .7 if sum(a) == 1 else .3 if sum(a) == 2 else 0., a) for p, a in PATTERNS.items()]
        for mode, name, rate, pattern in jobs:
            ckpt = args.source / f'best_miss_{rate:.1f}'.replace('.', 'p')
            ckpt = ckpt.with_name(ckpt.name + '.pt')
            checksum = checkpoints.setdefault(str(ckpt), sha(ckpt))
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            before = state_hash(model)
            artifacts, scores, queries, scans, replay_error = evaluate(model, loader, config,
                dimensions, device, schedule=schedules[rate] if mode == 'random' else None, pattern=pattern,
                query_writer=query_writer,
                query_metadata=dict(mode=mode, rate=rate, pattern=name if mode == 'fixed' else ''))
            query_stream.flush()
            checks = dict(full_replay_logit_max_abs_error=replay_error,
                          state_unchanged=state_hash(model) == before,
                          checkpoint_unchanged=sha(ckpt) == checksum)
            assert checks['state_unchanged'] and checks['checkpoint_unchanged']
            if mode == 'random':
                reference_path = args.source / f'predictions_miss_{rate:.1f}'.replace('.', 'p')
                reference_path = reference_path.with_name(reference_path.name + '.npz')
                with np.load(reference_path, allow_pickle=False) as ref:
                    for key in ('labels', 'availability'):
                        np.testing.assert_array_equal(artifacts[key], ref[key])
                    np.testing.assert_allclose(artifacts['pred_full'], ref['predictions'], rtol=2e-5, atol=2e-5)
                    if config.dataset.startswith('IEMOCAP'):
                        np.testing.assert_array_equal(artifacts['pred_full'], ref['predictions'])
                    else:
                        np.testing.assert_array_equal(artifacts['pred_full'] > 0, ref['predictions'] > 0)
                    checks['reference_max_abs_error'] = float(np.max(np.abs(artifacts['pred_full'] - ref['predictions'])))
                    checks['mask_labels_exact'] = True
                    checks['strict_class_match'] = True
                    checks['sample_ids_present_in_reference'] = 'sample_ids' in ref
                    if 'sample_ids' in ref:
                        np.testing.assert_array_equal(artifacts['sample_ids'], ref['sample_ids'])
                checks['reference_sha256'] = sha(reference_path)
                # Source trainer exposes test dictionaries under per_rate; fail
                # if a present metric cannot be reproduced, never silently pick.
                original_score = source_metrics.get('test', {}).get(name, {})
                if not original_score:
                    original_score = source_metrics.get('per_rate', {}).get(name, {})
                if not original_score or 'weighted_f1' not in original_score:
                    raise ValueError('Source per-rate test metric is missing')
                assert abs(scores['full']['weighted_f1'] - original_score['weighted_f1']) < 1e-7
                checks['source_metric_found'] = bool(original_score)
            filename = f'random_rate_{name}.npz' if mode == 'random' else f'fixed_{name}.npz'
            np.savez_compressed(args.output / filename, **artifacts)
            section = summary['random_rates' if mode == 'random' else 'fixed_patterns']
            section[name] = dict(checkpoint=str(ckpt), checkpoint_sha256=checksum, epoch=state.get('epoch'),
                checkpoint_rate=rate, scan_count=scans, sample_count=len(artifacts['labels']),
                metrics=scores, checks=checks, prediction_file=filename, query_summary=queries)
            write(args.output / 'SUMMARY.json', summary)
            print(f'{mode}={name} n={len(artifacts["labels"])} full_WF1={100*scores["full"]["weighted_f1"]:.6f}', flush=True)
        query_stream.close()
        summary['status'] = 'complete'
        write(args.output / 'SUMMARY.json', summary)
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        if 'query_stream' in locals() and not query_stream.closed:
            query_stream.close()
        write(args.output / 'STATUS.json', status)
        write(args.output / 'PROVENANCE.json', status)


if __name__ == '__main__':
    main()
