"""Fixed-current-query task replay on four historical state trajectories.

INTERNAL DIAGNOSTIC ONLY. No training, no new masks, no replacement task head.
The original scan computes current projections once; only pre-write state varies.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
    FlatReadoutCapture, _build_model, _sha256, _write,
)
from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
    CANONICAL_MASK_PROTOCOL, replay_view_from_availability, scatter_saved_availability,
)
from experiments.osram_meld_write_retention_20261014.audit import (
    BRANCHES, MODALITIES, _metric_arrays, candidate_pool, replay_forks,
)
from experiments.osram_meld_write_retention_20261014.run import (
    WriteCapture, guard_gpu, state_digest,
)


def validate_source(source):
    required = dict(dataset='MELD', seed=66, missing_mask_protocol=CANONICAL_MASK_PROTOCOL,
                    backbone_type='osram', osram_bidirectional=False,
                    osram_readout_fusion='flat', osram_meaningful_block='none',
                    training_objective='emotion-only', osram_disable_local_skip=False,
                    osram_output_dim=1600)
    if any(source.get(k) != value for k, value in required.items()):
        raise ValueError('Require original canonical seed66 no-JEPA causal Flat1600 with Skip')
    if source.get('osram_adapter_hidden_dim', 0) not in (0, 1600):
        raise ValueError('Task diagnostic cannot substitute reduced-width Flat')
    # No post-read module, readout ablation or forward-half duplication is legal.
    forbidden = ('osram_relation_block', 'osram_post_grn', 'osram_history_input_gate',
                 'osram_local_evidence_gate', 'osram_hierarchical_evidence_gate',
                 'osram_local_skip_gate', 'osram_memory_only_adapter',
                 'osram_history_recency', 'osram_decision_correction',
                 'osram_gap_increment_filter', 'local_context_residual',
                 'classification_completion', 'osram_forward_slot_reuse',
                 'osram_history_query_adapter')
    if any(source.get(k, False) for k in forbidden):
        raise ValueError('Require unmodified original Flat path')
    for field, default in [('osram_ablation', 'full'), ('osram_emotion_ablation', 'full'),
                           ('readout_type', 'shared'), ('osram_readout_candidate', 'none'),
                           ('fusion_type', 'mean')]:
        if source.get(field, default) != default:
            raise ValueError(f'Unexpected {field}: {source.get(field)}')


class TaskCapture:
    """Capture tensors actually consumed by _scan and the original Flat.

    Wrapping _scan (rather than re-projecting) includes any projection details
    used by the source model. Capture is removed before branch reads/Flat calls.
    """

    def __init__(self, osram):
        self.osram = osram
        self.flat = FlatReadoutCapture(osram)
        self.calls = 0

    def __enter__(self):
        self.original = self.osram._scan
        self.had_instance = '_scan' in self.osram.__dict__
        self.instance = self.osram.__dict__.get('_scan')
        def capture(keys, values, queries, availability, valid, reverse, **kwargs):
            if reverse or self.calls:
                raise RuntimeError('Task replay requires one causal scan')
            self.calls += 1
            self.queries = queries.detach().clone()
            self.availability = availability.detach().clone()
            self.valid = valid.detach().clone()
            result = self.original(keys, values, queries, availability, valid, reverse, **kwargs)
            self.base, self.gap = (v.detach().clone() for v in result[:2])
            return result
        self.osram._scan = capture
        self.flat.__enter__()
        return self

    @property
    def local(self):
        return self.flat.local

    def __exit__(self, *args):
        self.flat.__exit__(*args)
        if self.had_instance:
            self.osram._scan = self.instance
        else:
            del self.osram._scan


@torch.no_grad()
def original_read(osram, memory, queries, observed_keys, availability, valid):
    """Exact original _scan read algebra, before the target's joint write.

    All queries and observed keys are captured originals, NOT reconstructed
    from counterfactual state. No change to normalization/ridge/residual strength.
    """
    dtype = queries.dtype
    batch = memory.shape[0]
    keys = observed_keys * availability.to(dtype)[:, None, None, :]
    base = (memory @ queries[:, 0].unsqueeze(-1)).squeeze(-1).reshape(batch, -1)
    base = base * valid.to(dtype)[:, None]
    gaps = []
    strength = 0.0 if osram.osram_gap_read == 'raw' else osram.gap_residual_strength
    for index in range(3):
        query = queries[:, index + 1]
        projected = osram._address_residual(keys, query)
        residual_query = query - strength * (query - projected)
        read = (memory @ residual_query.unsqueeze(-1)).squeeze(-1).reshape(batch, -1)
        read = read * (1.0 - availability[:, index].to(dtype))[:, None] * valid.to(dtype)[:, None]
        gaps.append(read)
    return base, torch.stack(gaps, dim=1)


@torch.no_grad()
def replay_task_logits(model, local, memory, queries, observed_keys, availability, valid):
    osram = model.osram
    base, gap = original_read(osram, memory, queries, observed_keys, availability, valid)
    # Causal cfg84's backward slots are identically zero, kept in the old interface.
    base_context = torch.cat((base, torch.zeros_like(base)), -1)
    gap_context = torch.cat((gap, torch.zeros_like(gap)), -1)
    missing = 1.0 - availability.to(local.dtype)
    flat_input = torch.cat((local, base_context, (gap_context * missing[..., None]).flatten(1)), -1)
    hidden = osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(flat_input))
    hidden = torch.where(valid[:, None], hidden, torch.zeros_like(hidden))
    return model.smax_fc(hidden)


@torch.no_grad()
def task_rows(model, snapshots, target, normal_logits, labels, ids, rate, *, distances=(1, 2, 4, 8)):
    """One row per source modality / paired target / branch, not per head.

    Different modalities at the same source share a single joint-write fork:
    their task predictions are identical, their source association probes differ.
    """
    time_to_index = {snapshot[0]: i for i, snapshot in enumerate(snapshots)}
    availability_cpu = target.availability.bool().cpu().numpy()
    valid_cpu = target.valid.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    for source_index, (source_t, _, keys, values, _, post) in enumerate(snapshots):
        sources = [(b, m) for b in range(len(ids)) for m in range(3)
                   if availability_cpu[source_t, b, m]]
        if not sources:
            continue
        batches = torch.tensor([b for b, _ in sources], device=keys.device)
        source_keys = torch.stack([keys[b, ..., m] for b, m in sources])
        source_values = torch.stack([values[b, ..., m] for b, m in sources])
        pools = [candidate_pool(snapshots, source_index, b, m) for b, m in sources]
        after = _metric_arrays(post[batches], source_keys, source_values, pools)['relative_error'].mean(1)
        for distance, states, parity in replay_forks(model.osram, snapshots, source_index, distances=distances):
            t = source_t + distance
            current_keys = snapshots[time_to_index[t]][2]
            normal = normal_logits[t]
            valid = target.valid[t]
            labels_t = labels[:, t]
            normal_ce = normal.new_zeros(len(labels_t))
            normal_ce[valid] = F.cross_entropy(normal[valid], labels_t[valid], reduction='none')
            original_pred = normal.argmax(-1)
            normal_assoc = _metric_arrays(states['normal'][batches], source_keys, source_values, pools)
            normal_error = normal_assoc['relative_error'].mean(1)
            boundary_logits = {}
            for branch in BRANCHES:
                logits = replay_task_logits(model, target.local[t], states[branch], target.queries[t],
                                            current_keys, target.availability[t], valid)
                if branch == 'normal':
                    torch.testing.assert_close(logits[valid], normal[valid], rtol=2e-5, atol=2e-5)
                    if not torch.equal(logits[valid].argmax(-1), normal[valid].argmax(-1)):
                        raise RuntimeError('Normal replay changed a predicted class')
                if distance == 1:
                    boundary_logits[branch] = logits[valid]
                    if branch == 'decay_only':
                        torch.testing.assert_close(boundary_logits['normal'], logits[valid], rtol=2e-5, atol=2e-5)
                    if branch == 'hold':
                        torch.testing.assert_close(boundary_logits['write_only'], logits[valid], rtol=2e-5, atol=2e-5)
                ce = logits.new_zeros(len(labels_t))
                ce[valid] = F.cross_entropy(logits[valid], labels_t[valid], reduction='none')
                scored = (normal_assoc if branch == 'normal' else
                          _metric_arrays(states[branch][batches], source_keys, source_values, pools))
                def mean_finite(name, index):
                    array = scored[name][index]
                    finite = array[np.isfinite(array)]
                    return float(finite.mean()) if finite.size else None
                logits_cpu = logits.cpu().numpy()
                ce_cpu = ce.cpu().numpy()
                normal_ce_cpu = normal_ce.cpu().numpy()
                original_pred_cpu = original_pred.cpu().numpy()
                for i, (b, m) in enumerate(sources):
                    if not valid_cpu[t, b]:
                        continue
                    prediction = int(logits_cpu[b].argmax())
                    label = int(labels_cpu[b, t])
                    normal_prediction = int(original_pred_cpu[b])
                    error = mean_finite('relative_error', i)
                    yield dict(conversation_id=str(ids[b]), rate=float(rate), source_t=int(source_t),
                               target_t=int(t), distance=int(distance), modality=MODALITIES[m], branch=branch,
                               sanity_only=distance == 1,
                               availability=availability_cpu[t, b].astype(int).tolist(),
                               label=label, prediction=prediction, normal_prediction=normal_prediction,
                               logits=logits_cpu[b].astype(float).tolist(),
                               ce=float(ce_cpu[b]), normal_ce=float(normal_ce_cpu[b]),
                               delta_ce=float(ce_cpu[b] - normal_ce_cpu[b]),
                               correction=bool(prediction == label and normal_prediction != label),
                               harm=bool(prediction != label and normal_prediction == label),
                               source_relative_error=error, normal_source_relative_error=float(normal_error[i]),
                               write_after_relative_error=float(after[i]),
                               normal_association_drift=float(normal_error[i] - after[i]),
                               association_error_change=float(error - normal_error[i]),
                               normal_association_degraded=bool(normal_error[i] > after[i] + 1e-7),
                               source_centered_cosine=mean_finite('centered_cosine', i),
                               source_retrieval_top1=mean_finite('retrieval_top1', i),
                               state_replay_max_error=float(parity),
                               normal_logit_max_error=float((logits[b] - normal[b]).abs().max()) if branch == 'normal' else None)


def evaluate_rate(model, loader, config, dimensions, saved, rate, stream, max_batches):
    from gcnet_missing_m3.train_gcnet import _collect_predictions, _move_batch
    cursor = batches = count = 0
    max_normal_logit_error = 0.
    for raw in loader:
        if max_batches is not None and batches >= max_batches:
            break
        data = _move_batch(raw, next(model.parameters()).device)
        start = cursor
        availability, cursor = scatter_saved_availability(saved['availability'], cursor, data[7])
        view = replay_view_from_availability(data, availability, dimensions)
        with torch.no_grad(), TaskCapture(model.osram) as capture, WriteCapture(model.osram, view['umask']) as writer:
            logits, _, _, completion = model([view['incomplete']], view['availability'], view['qmask'],
                                            view['umask'], view['lengths'], predict_missing=False)
        pred, labels, _ = _collect_predictions(config.dataset, logits, view['labels'], view['umask'], config.mosi_task_mode)
        np.testing.assert_array_equal(pred, saved['predictions'][start:cursor])
        np.testing.assert_array_equal(labels, saved['labels'][start:cursor])
        if completion is not None or capture.calls != 1 or capture.flat.local_calls != 1 or capture.flat.adapter_calls != 1:
            raise RuntimeError('Require one unchanged original Flat forward')
        for row in task_rows(model, writer.snapshots, capture, logits, view['labels'], data[-1], rate):
            if row['branch'] == 'normal':
                max_normal_logit_error = max(max_normal_logit_error, row['normal_logit_max_error'])
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            count += 1
        batches += 1
        stream.flush()
        print(f'rate={rate:.1f} batches={batches} utterances={cursor} rows={count}', flush=True)
    if max_batches is None and cursor != len(saved['predictions']):
        raise ValueError('Incomplete saved-mask replay')
    return dict(batches=batches, utterances=cursor, rows=count, saved_predictions_exact=True,
                saved_labels_exact=True, max_normal_logit_error=max_normal_logit_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset-root', 'meld-pickle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--rates', type=float, nargs='+', default=[i / 10 for i in range(8)])
    parser.add_argument('--max-batches', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive')
    if len(set(args.rates)) != len(args.rates) or any(rate not in [i / 10 for i in range(8)] for rate in args.rates):
        parser.error('Require distinct rates from 0.0 through 0.7')
    source_config = json.loads((args.source / 'config.json').read_text())
    validate_source(source_config)
    required = [args.meld_pickle]
    for rate in args.rates:
        suffix = f'{rate:.1f}'.replace('.', 'p')
        required.extend([args.source / f'best_miss_{suffix}.pt', args.source / f'predictions_miss_{suffix}.npz'])
    if any(not path.is_file() for path in required):
        raise FileNotFoundError('Missing original source artifacts; no fallback')
    gpu = guard_gpu(args.gpu_uuid)
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    os.environ['GCNET_MELD_PICKLE'] = str(args.meld_pickle)
    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
    from gcnet_modality_jepa.train_gcnet import get_loaders
    config = replace(TrainConfig(**source_config), device='cuda:0')
    torch.set_num_threads(2)
    roots = [args.dataset_root / 'MELD/features' / name for name in
             ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    _, _, test, ad, td, vd = get_loaders(audio_root=str(roots[0]), text_root=str(roots[1]), video_root=str(roots[2]),
        num_folder=int(_dataset_shape('MELD')['num_folds']), dataset='MELD', batch_size=config.batch_size,
        num_workers=0, seed=config.seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    model = _build_model(config, (ad, td, vd), torch.device('cuda:0'))
    args.output.mkdir(parents=True)
    summary = dict(status='running', training=False, protocol='INTERNAL DIAGNOSTIC ONLY',
                   source=str(args.source), source_config=source_config, gpu=gpu, command=sys.argv,
                   masks='existing saved final-evaluation masks, unchanged; not reconstructed selection-epoch masks',
                   distances=[2, 4, 8], sanity_distance=1, fixed_current_queries=True,
                   rates={}, torch_version=torch.__version__, git_commit=subprocess.check_output(
                       ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   git_diff=subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True),
                   source_config_sha256=_sha256(args.source / 'config.json'),
                   dataset_sha256=_sha256(args.meld_pickle),
                   code_sha256={str(path.relative_to(ROOT)): _sha256(path) for path in
                       (Path(__file__), Path(__file__).with_name('audit.py'), Path(__file__).with_name('run.py'),
                        ROOT / 'gcnet_missing_m3/osram.py', ROOT / 'gcnet_missing_m3/model.py')})
    started = time.monotonic()
    _write(args.output / 'SUMMARY.json', summary)
    try:
        for rate in args.rates:
            suffix = f'{rate:.1f}'.replace('.', 'p')
            checkpoint = args.source / f'best_miss_{suffix}.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            epoch = state.get('epoch')
            del state
            before = state_digest(model)
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            filename = f'task_miss_{suffix}.jsonl.gz'
            with np.load(args.source / f'predictions_miss_{suffix}.npz') as saved, gzip.open(args.output / filename, 'wt') as stream:
                result = evaluate_rate(model, test[config.fold - 1], config, (ad, td, vd), saved, rate, stream, args.max_batches)
            after = state_digest(model)
            # Iterating the loader can consume CPU Torch RNG (base_seed), even
            # in eval mode; do not mistake that for stochastic branch reads.
            if before != after or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Frozen parameters changed or accumulated gradients')
            if not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                raise RuntimeError('Frozen branch replay consumed CUDA randomness')
            summary['rates'][str(rate)] = dict(**result, checkpoint_epoch=epoch,
                checkpoint_sha256=_sha256(checkpoint), rows_file=filename,
                seconds=time.monotonic() - tick, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                state_sha256_before=before, state_sha256_after=after,
                cpu_rng_unchanged=torch.equal(rng, torch.get_rng_state()), cuda_rng_unchanged=True)
            _write(args.output / 'SUMMARY.json', summary)
        summary['status'] = 'partial_smoke' if args.max_batches else ('complete' if len(args.rates) == 8 else 'partial_rates')
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        summary['seconds'] = time.monotonic() - started
        _write(args.output / 'SUMMARY.json', summary)


if __name__ == '__main__':
    main()
